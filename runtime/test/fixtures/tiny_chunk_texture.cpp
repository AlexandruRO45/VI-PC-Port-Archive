// A mapped guest with two adjacent indexed quads. Each chunk owns its index buffer, descriptor heap and one
// reference to a common texture. A leaves while B still draws; after both leave, reload A with different texels.
// vs_m2/ps_m2 are the existing synthetic RDNA2 pair used in d3d12_m2_textured_test.
#include "../d3d12_test_shaders.h"

#include <array>
#include <cstdio>
#include <cstring>
#include <iterator>
#include <memory>
#include <stdexcept>

using namespace d3d12test;

extern "C" __declspec(dllimport) HRESULT WINAPI D3D12XboxCreateDevice(IUnknown *, const void *, const IID *,
                                                                      void **);

namespace {
constexpr UINT width = 64, height = 32, row_pitch = 256;
using Color = std::array<std::uint8_t, 4>;
constexpr Color clear{0, 0, 0, 255};
enum ChunkMask : unsigned { empty = 0, west = 1, east = 2 };

void require(bool ok, const char *message) {
    if (!ok)
        throw std::runtime_error(message);
}
void checked(HRESULT hr, const char *call) {
    if (FAILED(hr)) {
        std::fprintf(stderr, "%s: HRESULT %08lx\n", call, static_cast<unsigned long>(hr));
        throw std::runtime_error(call);
    }
}
#define CHECK_HR(call) checked((call), #call)

Color texel(unsigned generation, unsigned x, unsigned y) {
    return {std::uint8_t(32 + 48 * x), std::uint8_t(10 + 42 * y), std::uint8_t(24 + 90 * generation), 255};
}

std::vector<std::uint8_t> shader(const std::filesystem::path &path) {
    std::ifstream in(path, std::ios::binary);
    require(in.is_open(), "cannot open synthetic shader");
    std::vector<std::uint8_t> bytes{std::istreambuf_iterator<char>(in), {}};
    require(bytes.size() >= 4 && std::memcmp(bytes.data(), "FSSM", 4) == 0, "invalid synthetic shader");
    return shader_container(bytes);
}

using HeapHandle = std::uint64_t(WINAPI *)(void *);
std::uint64_t handle(ID3D12DescriptorHeap *heap, unsigned index) {
    return reinterpret_cast<HeapHandle>(slot(heap, index))(heap);
}
struct Chunk {
    Ref<ID3D12Resource> texture, indices;
    Ref<ID3D12DescriptorHeap> heap;
};

struct Scene {
    Ref<ID3D12Device> device;
    Ref<ID3D12CommandQueue> queue;
    Ref<ID3D12CommandAllocator> allocator;
    Ref<ID3D12GraphicsCommandList> list;
    Ref<ID3D12Fence> fence;
    Ref<ID3D12RootSignature> root;
    Ref<ID3D12PipelineState> pipeline;
    Ref<ID3D12DescriptorHeap> sampler, rt_heap;
    Ref<ID3D12Resource> target, readback, upload;
    std::array<std::unique_ptr<Chunk>, 2> chunks;
    D3D12_CPU_DESCRIPTOR_HANDLE rtv{};
    unsigned allocations = 0, final_releases = 0;

    void heap(D3D12_DESCRIPTOR_HEAP_TYPE kind, bool visible, Ref<ID3D12DescriptorHeap> &out) {
        D3D12_DESCRIPTOR_HEAP_DESC desc{
            kind, 1, visible ? D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE : D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0};
        CHECK_HR(device->CreateDescriptorHeap(&desc, IID_PPV_ARGS(&out.p)));
    }
    void create(D3D12_HEAP_TYPE memory, const D3D12_RESOURCE_DESC &desc, D3D12_RESOURCE_STATES state,
                Ref<ID3D12Resource> &out) {
        D3D12_HEAP_PROPERTIES props{};
        props.Type = memory;
        CHECK_HR(device->CreateCommittedResource(&props, D3D12_HEAP_FLAG_NONE, &desc, state, nullptr,
                                                 IID_PPV_ARGS(&out.p)));
    }
    static D3D12_RESOURCE_DESC image(UINT w, UINT h, D3D12_RESOURCE_FLAGS flags = D3D12_RESOURCE_FLAG_NONE) {
        D3D12_RESOURCE_DESC d{};
        d.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        d.Width = w;
        d.Height = h;
        d.DepthOrArraySize = 1;
        d.MipLevels = 1;
        d.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        d.SampleDesc.Count = 1;
        d.Flags = flags;
        return d;
    }
    void init(const std::filesystem::path &dir) {
        std::uint32_t params[24]{};
        const IID iid = __uuidof(ID3D12Device);
        CHECK_HR(D3D12XboxCreateDevice(nullptr, params, &iid, reinterpret_cast<void **>(&device.p)));
        D3D12_COMMAND_QUEUE_DESC q{};
        CHECK_HR(device->CreateCommandQueue(&q, IID_PPV_ARGS(&queue.p)));
        CHECK_HR(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator.p)));
        CHECK_HR(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.p, nullptr,
                                           IID_PPV_ARGS(&list.p)));
        CHECK_HR(list->Close());
        CHECK_HR(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence.p)));

        D3D12_DESCRIPTOR_RANGE ranges[2]{{D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0, 0, 0},
                                         {D3D12_DESCRIPTOR_RANGE_TYPE_SAMPLER, 1, 0, 0, 0}};
        D3D12_ROOT_PARAMETER parameters[2]{};
        for (unsigned i = 0; i != 2; ++i) {
            parameters[i].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
            parameters[i].DescriptorTable = {1, &ranges[i]};
            parameters[i].ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
        }
        D3D12_VERSIONED_ROOT_SIGNATURE_DESC desc{};
        desc.Version = D3D_ROOT_SIGNATURE_VERSION_1_0;
        desc.Desc_1_0 = {2, parameters, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_NONE};
        auto blob = serialize_root_signature(desc);
        require(!blob.empty(), "root signature serialization");
        CHECK_HR(device->CreateRootSignature(0, blob.data(), blob.size(), IID_PPV_ARGS(&root.p)));
        auto vs = shader(dir / "fssm" / "vs_m2.fssm"), ps = shader(dir / "fssm" / "ps_m2.fssm");
        D3D12_GRAPHICS_PIPELINE_STATE_DESC p{};
        p.pRootSignature = root.p;
        p.VS = {vs.data(), vs.size()};
        p.PS = {ps.data(), ps.size()};
        p.BlendState.RenderTarget[0].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
        p.SampleMask = UINT_MAX;
        p.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
        p.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
        p.RasterizerState.DepthClipEnable = TRUE;
        p.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
        p.NumRenderTargets = 1;
        p.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
        p.SampleDesc.Count = 1;
        CHECK_HR(device->CreateGraphicsPipelineState(&p, IID_PPV_ARGS(&pipeline.p)));
        create(D3D12_HEAP_TYPE_DEFAULT, image(width, height, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET),
               D3D12_RESOURCE_STATE_RENDER_TARGET, target);
        create(D3D12_HEAP_TYPE_READBACK, buffer_desc(height * row_pitch), D3D12_RESOURCE_STATE_COPY_DEST,
               readback);
        create(D3D12_HEAP_TYPE_UPLOAD, buffer_desc(4 * row_pitch), D3D12_RESOURCE_STATE_GENERIC_READ, upload);
        heap(D3D12_DESCRIPTOR_HEAP_TYPE_RTV, false, rt_heap);
        rtv.ptr = SIZE_T(handle(rt_heap.p, 9));
        device->CreateRenderTargetView(target.p, nullptr, rtv);
        heap(D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER, true, sampler);
        D3D12_SAMPLER_DESC s{};
        s.Filter = D3D12_FILTER_MIN_MAG_MIP_POINT;
        s.AddressU = s.AddressV = s.AddressW = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
        s.ComparisonFunc = D3D12_COMPARISON_FUNC_NEVER;
        s.MaxAnisotropy = 1;
        s.MaxLOD = D3D12_FLOAT32_MAX;
        device->CreateSampler(&s, {SIZE_T(handle(sampler.p, 9))});
    }

    void enter(unsigned i) {
        chunks[i] = std::make_unique<Chunk>();
        auto &chunk = chunks[i];
        const auto &neighbor = chunks[1 - i];
        if (neighbor) {
            chunk->texture.p = neighbor->texture.p;
            chunk->texture->AddRef();
        } else {
            ++allocations;
            create(D3D12_HEAP_TYPE_DEFAULT, image(4, 4), D3D12_RESOURCE_STATE_COPY_DEST, chunk->texture);
            std::uint8_t *bytes = nullptr;
            CHECK_HR(upload->Map(0, nullptr, reinterpret_cast<void **>(&bytes)));
            for (UINT y = 0; y < 4; ++y)
                for (UINT x = 0; x < 4; ++x) {
                    const auto color = texel(allocations, x, y);
                    std::memcpy(bytes + y * row_pitch + x * 4, color.data(), 4);
                }
            upload->Unmap(0, nullptr);
            D3D12_TEXTURE_COPY_LOCATION dst{}, src{};
            dst.pResource = chunk->texture.p;
            dst.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
            src.pResource = upload.p;
            src.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
            src.PlacedFootprint = {0, {DXGI_FORMAT_R8G8B8A8_UNORM, 4, 4, 1, row_pitch}};
            list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
            auto ready = transition(chunk->texture.p, D3D12_RESOURCE_STATE_COPY_DEST,
                                    D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
            list->ResourceBarrier(1, &ready);
        }
        heap(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, true, chunk->heap);
        D3D12_SHADER_RESOURCE_VIEW_DESC view{};
        view.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        view.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
        view.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        view.Texture2D.MipLevels = 1;
        device->CreateShaderResourceView(chunk->texture.p, &view, {SIZE_T(handle(chunk->heap.p, 9))});
        constexpr std::uint16_t indices[]{0, 1, 2, 2, 1, 3};
        create(D3D12_HEAP_TYPE_UPLOAD, buffer_desc(sizeof(indices)), D3D12_RESOURCE_STATE_GENERIC_READ,
               chunk->indices);
        void *data = nullptr;
        CHECK_HR(chunk->indices->Map(0, nullptr, &data));
        std::memcpy(data, indices, sizeof(indices));
        chunk->indices->Unmap(0, nullptr);
    }

    void leave(unsigned i) {
        auto &c = chunks[i];
        c->heap.reset();
        ID3D12Resource *texture = c->texture.p;
        c->texture.p = nullptr;
        c.reset();
        const ULONG remaining = texture->Release();
        if (!chunks[1 - i]) {
            require(remaining == 0, "last chunk left a texture reference");
            ++final_releases;
        } else {
            require(remaining != 0, "west destroyed the east chunk's texture");
        }
    }

    void frame(unsigned frame_number, unsigned mask) {
        CHECK_HR(allocator->Reset());
        CHECK_HR(list->Reset(allocator.p, pipeline.p));
        for (unsigned i = 0; i != 2; ++i) {
            const bool wanted = (mask & (1u << i)) != 0;
            if (wanted && !chunks[i])
                enter(i);
            if (!wanted && chunks[i])
                leave(i);
        }
        const float background[]{0, 0, 0, 1};
        list->ClearRenderTargetView(rtv, background, 0, nullptr);
        list->SetGraphicsRootSignature(root.p);
        list->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        list->OMSetRenderTargets(1, &rtv, FALSE, nullptr);
        for (unsigned i = 0; i != 2; ++i) {
            if (!chunks[i])
                continue;
            Chunk &c = *chunks[i];
            ID3D12DescriptorHeap *heaps[]{c.heap.p, sampler.p};
            list->SetDescriptorHeaps(2, heaps);
            list->SetGraphicsRootDescriptorTable(0, {handle(c.heap.p, 10)});
            list->SetGraphicsRootDescriptorTable(1, {handle(sampler.p, 10)});
            const D3D12_INDEX_BUFFER_VIEW iv{c.indices->GetGPUVirtualAddress(), 12, DXGI_FORMAT_R16_UINT};
            list->IASetIndexBuffer(&iv);
            const D3D12_VIEWPORT viewport{float(i * width / 2), 0, float(width / 2), float(height), 0, 1};
            const D3D12_RECT scissor{LONG(i * width / 2), 0, LONG((i + 1) * width / 2), LONG(height)};
            list->RSSetViewports(1, &viewport);
            list->RSSetScissorRects(1, &scissor);
            list->DrawIndexedInstanced(6, 1, 0, 0, 0);
        }
        auto to_copy = transition(target.p, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_COPY_SOURCE);
        list->ResourceBarrier(1, &to_copy);
        D3D12_TEXTURE_COPY_LOCATION dst{}, src{};
        dst.pResource = readback.p;
        dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        dst.PlacedFootprint = {0, {DXGI_FORMAT_R8G8B8A8_UNORM, width, height, 1, row_pitch}};
        src.pResource = target.p;
        src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
        auto to_render =
            transition(target.p, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_RENDER_TARGET);
        list->ResourceBarrier(1, &to_render);
        CHECK_HR(list->Close());
        ID3D12CommandList *lists[]{list.p};
        queue->ExecuteCommandLists(1, lists);
        CHECK_HR(queue->Signal(fence.p, frame_number + 1));
        CHECK_HR(fence->SetEventOnCompletion(frame_number + 1, nullptr));
        std::uint8_t *pixels = nullptr;
        CHECK_HR(readback->Map(0, nullptr, reinterpret_cast<void **>(&pixels)));
        // Generation and occupancy are specified by the five-frame script, independently of the guest objects.
        for (UINT y = 0; y < height; ++y)
            for (UINT x = 0; x < width; ++x) {
                const bool drawn = (mask & (x < width / 2 ? west : east)) != 0;
                const Color expected =
                    drawn ? texel(frame_number == 4 ? 2 : 1, (x % (width / 2)) / 8, y / 8) : clear;
                if (std::memcmp(pixels + y * row_pitch + x * 4, expected.data(), 4)) {
                    std::fprintf(stderr, "frame %u pixel (%u,%u) differs\n", frame_number, x, y);
                    readback->Unmap(0, nullptr);
                    throw std::runtime_error("texture or chunk coverage mismatch");
                }
            }
        readback->Unmap(0, nullptr);
    }
};
} // namespace

int main(int argc, char **argv) {
    try {
        require(argc == 2, "usage: tiny_chunk_texture <M2 fixture folder>");
        Scene scene;
        scene.init(argv[1]);
        constexpr unsigned masks[]{west, west | east, east, empty, west};
        for (unsigned i = 0; i != 5; ++i)
            scene.frame(i, masks[i]);
        scene.leave(0);
        require(scene.allocations == 2 && scene.final_releases == 2, "texture allocations and final releases");
        std::puts("tiny_chunk_texture: PASS 5 frames, 10240 pixels, 2 texture allocations and final releases");
        return 0;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "tiny_chunk_texture: %s\n", e.what());
        return 1;
    }
}
