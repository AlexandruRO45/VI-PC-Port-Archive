// D3D-393: the replay skips a draw's bind or set when the same value is still bound in the command buffer and forgets
// what is bound where a command may disturb it (src/vulkan_replay.h, detail::BoundDrawState). These scenarios make
// each skip and each forgetting visible: every draw changes some state (pipeline, root arguments, viewport, scissor,
// vertex and index buffers, topology, blend factor, root signature) and keeps the rest, and between draws come clears
// (whole-view, and rectangles in a rendering scope of their own), a copy and a barrier, which keep the bound state; a
// compute dispatch and a binary-occlusion resolve, whose dispatches bind their own pipeline and push constants; a
// bundle; and a pipeline without vertex input, after which the vertex buffers must be bound again (their strides are
// invalid, which the validation layer reports). A draw that kept a stale value, or lost one it should have kept,
// lands in another cell or colour.
// The same scenarios run on the layer (a Vulkan device, the shaders d3d12_bundle_test and d3d12_draw_test use, and
// cs_user_data.comp) and on Windows' own D3D12 (HLSL from D3DCompile, the default adapter and WARP); each is read
// back as an 8x8 grid of letters: B blue (the clear), R, G, Y the draw's colour, C a green-only write over blue, K
// black (a cleared rectangle). The expected grids are what both host adapters produced; the layer is held to them,
// and so are the hosts. Set LEONIDA_VK_VALIDATION=1 to run under VK_LAYER_KHRONOS_validation with synchronization
// validation: any error fails the test. No game data.
#include "test_temp_dir.h"
#include "vk_test_device.h"
#include "d3d12_test_shaders.h"
#include "host_wine.h"
#include "../shims/d3d12_com.h"
#include "../src/shader_library.h"

#include <d3dcompiler.h>
#include <dxgi1_4.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <functional>
#include <string>
#include <vector>

#if !defined(LEONIDA_TEST_VS_SPV) || !defined(LEONIDA_TEST_PS_SPV) || !defined(LEONIDA_TEST_CS_SPV) || \
    !defined(LEONIDA_TEST_VS_FULLSCREEN_SPV)
#error "the LEONIDA_TEST_*_SPV macros must name the compiled test shaders"
#endif

#define CHECK(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return false; } \
} while (0)

#define REQUIRE(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return 1; } \
} while (0)

using namespace d3d12test;

namespace {

constexpr UINT target_px = 64;
constexpr UINT grid_cells = 8;

struct Vertex {
    float x, y;
    std::uint8_t r, g, b, a;
};

// Upload-heap layout, in bytes.
constexpr UINT64 vb_quad = 0;        // the viewport's top-left quarter as a strip: (-1,1) (0,1) (-1,0) (0,0)
constexpr UINT64 vb_other = 64;      // the same shape in the bottom-left quarter
constexpr UINT64 ib_quad = 128;      // 0 1 2 2 1 3: both triangles of the quad as a list
constexpr UINT64 ib_flip = 144;      // 3 1 2 0: as a list, only the lower-right half; as a strip, the whole quad
constexpr UINT64 tint_red = 256;     // (1, 0, 1): the red vertices stay red
constexpr UINT64 tint_green = 512;   // (0, 1, 0)
constexpr UINT64 tint_yellow = 768;  // (1, 1, 1)

constexpr auto strip = D3D_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP;
constexpr auto tlist = D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;

#define BLANK "BBBBBBBB"

using Recorder = std::function<void(struct Rig&, ID3D12GraphicsCommandList*)>;

struct Shaders {
    std::vector<std::uint8_t> vs, ps, cs, fs;  // fs: a vertex shader without inputs, one viewport-sized triangle
};

struct Rig {
    ID3D12Device* device = nullptr;
    // root1: two constants (an offset, s0 s1) and a CBV (the tint). root2 adds a third constant: another layout.
    // croot: two constants and a root UAV, for the compute pipeline.
    Ref<ID3D12RootSignature> root1, root2, croot;
    // a: all channels; b: green and alpha only; a2: a on root2; f: the output times the blend factor; fs: no vertex
    // input, a yellow triangle over the whole viewport; cs: compute.
    Ref<ID3D12PipelineState> pso_a, pso_b, pso_a2, pso_f, pso_fs, cs;
    Ref<ID3D12CommandQueue> queue;
    Ref<ID3D12Fence> fence;
    Ref<ID3D12Resource> upload, target, readback, results, uav;
    Ref<ID3D12DescriptorHeap> rtv_heap;
    Ref<ID3D12QueryHeap> queries;
    D3D12_CPU_DESCRIPTOR_HANDLE rtv{};
    D3D12_GPU_VIRTUAL_ADDRESS upload_va = 0;
    UINT64 fence_value = 0;
    std::vector<Ref<ID3D12CommandAllocator>> bundle_allocators;
    std::vector<Ref<ID3D12GraphicsCommandList>> bundles;
    Shaders shaders;

    bool make_root(UINT count, Ref<ID3D12RootSignature>& root) {
        D3D12_ROOT_PARAMETER params[3] = {};
        params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
        params[0].Constants = {0, 0, 2};
        params[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
        params[1].Descriptor = {1, 0};
        params[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
        params[2].Constants = {2, 0, 1};
        D3D12_VERSIONED_ROOT_SIGNATURE_DESC root_desc{};
        root_desc.Version = D3D_ROOT_SIGNATURE_VERSION_1_0;
        root_desc.Desc_1_0 = {count, params, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT};
        const auto blob = serialize_root_signature(root_desc);
        CHECK(!blob.empty());
        CHECK(SUCCEEDED(device->CreateRootSignature(0, blob.data(), blob.size(), IID_PPV_ARGS(&root.p))));
        return true;
    }

    bool make_pso(ID3D12RootSignature* root, UINT8 write_mask, bool blend_factor, Ref<ID3D12PipelineState>& pso,
                  bool fullscreen = false) {
        static const D3D12_INPUT_ELEMENT_DESC elements[2] = {
            {"POSITION", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 0, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
            {"COLOR", 0, DXGI_FORMAT_R8G8B8A8_UNORM, 0, D3D12_APPEND_ALIGNED_ELEMENT,
             D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0}};
        D3D12_GRAPHICS_PIPELINE_STATE_DESC desc{};
        desc.pRootSignature = root;
        desc.VS = fullscreen ? D3D12_SHADER_BYTECODE{shaders.fs.data(), shaders.fs.size()}
                             : D3D12_SHADER_BYTECODE{shaders.vs.data(), shaders.vs.size()};
        desc.PS = {shaders.ps.data(), shaders.ps.size()};
        auto& blend = desc.BlendState.RenderTarget[0];
        blend.RenderTargetWriteMask = write_mask;
        if (blend_factor) {  // colour = output * blend factor, alpha = output alpha
            blend.BlendEnable = TRUE;
            blend.SrcBlend = D3D12_BLEND_BLEND_FACTOR;
            blend.DestBlend = D3D12_BLEND_ZERO;
            blend.BlendOp = D3D12_BLEND_OP_ADD;
            blend.SrcBlendAlpha = D3D12_BLEND_ONE;
            blend.DestBlendAlpha = D3D12_BLEND_ZERO;
            blend.BlendOpAlpha = D3D12_BLEND_OP_ADD;
        }
        desc.SampleMask = 0xffffffff;
        desc.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
        desc.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
        desc.RasterizerState.DepthClipEnable = TRUE;
        desc.InputLayout = fullscreen ? D3D12_INPUT_LAYOUT_DESC{nullptr, 0} : D3D12_INPUT_LAYOUT_DESC{elements, 2};
        desc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
        desc.NumRenderTargets = 1;
        desc.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
        desc.SampleDesc.Count = 1;
        CHECK(SUCCEEDED(device->CreateGraphicsPipelineState(&desc, IID_PPV_ARGS(&pso.p))));
        return true;
    }

    bool init(ID3D12Device* d, const Shaders& s) {
        device = d;
        shaders = s;
        CHECK(make_root(2, root1) && make_root(3, root2));
        constexpr UINT8 all = D3D12_COLOR_WRITE_ENABLE_ALL;
        CHECK(make_pso(root1.p, all, false, pso_a));
        CHECK(make_pso(root1.p, D3D12_COLOR_WRITE_ENABLE_GREEN | D3D12_COLOR_WRITE_ENABLE_ALPHA, false, pso_b));
        CHECK(make_pso(root2.p, all, false, pso_a2));
        CHECK(make_pso(root1.p, all, true, pso_f));
        CHECK(make_pso(root1.p, all, false, pso_fs, true));
        {
            D3D12_ROOT_PARAMETER params[2] = {};
            params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
            params[0].Constants = {0, 0, 2};
            params[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
            params[1].Descriptor = {0, 0};
            D3D12_VERSIONED_ROOT_SIGNATURE_DESC root_desc{};
            root_desc.Version = D3D_ROOT_SIGNATURE_VERSION_1_0;
            root_desc.Desc_1_0 = {2, params, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_NONE};
            const auto blob = serialize_root_signature(root_desc);
            CHECK(!blob.empty());
            CHECK(SUCCEEDED(device->CreateRootSignature(0, blob.data(), blob.size(), IID_PPV_ARGS(&croot.p))));
            D3D12_COMPUTE_PIPELINE_STATE_DESC desc{};
            desc.pRootSignature = croot.p;
            desc.CS = {shaders.cs.data(), shaders.cs.size()};
            CHECK(SUCCEEDED(device->CreateComputePipelineState(&desc, IID_PPV_ARGS(&cs.p))));
        }

        const D3D12_HEAP_PROPERTIES upload_heap{D3D12_HEAP_TYPE_UPLOAD};
        const D3D12_HEAP_PROPERTIES default_heap{D3D12_HEAP_TYPE_DEFAULT};
        const D3D12_HEAP_PROPERTIES readback_heap{D3D12_HEAP_TYPE_READBACK};
        auto buffer = buffer_desc(4096);
        CHECK(SUCCEEDED(device->CreateCommittedResource(&upload_heap, D3D12_HEAP_FLAG_NONE, &buffer,
                                                        D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
                                                        IID_PPV_ARGS(&upload.p))));
        CHECK(SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &buffer,
                                                        D3D12_RESOURCE_STATE_COMMON, nullptr,
                                                        IID_PPV_ARGS(&results.p))));
        buffer = buffer_desc(4096, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        CHECK(SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &buffer,
                                                        D3D12_RESOURCE_STATE_COMMON, nullptr,
                                                        IID_PPV_ARGS(&uav.p))));
        buffer = buffer_desc(target_px * 256);
        CHECK(SUCCEEDED(device->CreateCommittedResource(&readback_heap, D3D12_HEAP_FLAG_NONE, &buffer,
                                                        D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                        IID_PPV_ARGS(&readback.p))));
        D3D12_RESOURCE_DESC texture{};
        texture.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        texture.Width = texture.Height = target_px;
        texture.DepthOrArraySize = texture.MipLevels = 1;
        texture.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        texture.SampleDesc.Count = 1;
        texture.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
        CHECK(SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &texture,
                                                        D3D12_RESOURCE_STATE_COMMON, nullptr,
                                                        IID_PPV_ARGS(&target.p))));
        const Vertex quad[4] = {{-1, 1, 255, 0, 0, 255}, {0, 1, 255, 0, 0, 255}, {-1, 0, 255, 0, 0, 255},
                                {0, 0, 255, 0, 0, 255}};
        const Vertex other[4] = {{-1, 0, 255, 0, 0, 255}, {0, 0, 255, 0, 0, 255}, {-1, -1, 255, 0, 0, 255},
                                 {0, -1, 255, 0, 0, 255}};
        const std::uint16_t list[6] = {0, 1, 2, 2, 1, 3};
        const std::uint16_t flip[4] = {3, 1, 2, 0};
        const float tints[3][3] = {{1, 0, 1}, {0, 1, 0}, {1, 1, 1}};
        std::uint8_t* up = nullptr;
        CHECK(SUCCEEDED(upload->Map(0, nullptr, reinterpret_cast<void**>(&up))));
        std::memset(up, 0, 4096);
        std::memcpy(up + vb_quad, quad, sizeof(quad));
        std::memcpy(up + vb_other, other, sizeof(other));
        std::memcpy(up + ib_quad, list, sizeof(list));
        std::memcpy(up + ib_flip, flip, sizeof(flip));
        std::memcpy(up + tint_red, tints[0], sizeof(tints[0]));
        std::memcpy(up + tint_green, tints[1], sizeof(tints[1]));
        std::memcpy(up + tint_yellow, tints[2], sizeof(tints[2]));
        upload->Unmap(0, nullptr);
        upload_va = upload->GetGPUVirtualAddress();

        const D3D12_DESCRIPTOR_HEAP_DESC heap_desc{D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 1, D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0};
        CHECK(SUCCEEDED(device->CreateDescriptorHeap(&heap_desc, IID_PPV_ARGS(&rtv_heap.p))));
        rtv = rtv_heap->GetCPUDescriptorHandleForHeapStart();
        device->CreateRenderTargetView(target.p, nullptr, rtv);
        const D3D12_QUERY_HEAP_DESC query_desc{D3D12_QUERY_HEAP_TYPE_OCCLUSION, 1, 0};
        CHECK(SUCCEEDED(device->CreateQueryHeap(&query_desc, IID_PPV_ARGS(&queries.p))));
        D3D12_COMMAND_QUEUE_DESC queue_desc{};
        CHECK(SUCCEEDED(device->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&queue.p))));
        CHECK(SUCCEEDED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence.p))));
        return true;
    }

    // A closed bundle created with `initial` as its pipeline state, kept alive until the rig goes.
    template<class F>
    ID3D12GraphicsCommandList* bundle(ID3D12PipelineState* initial, F&& record) {
        bundle_allocators.emplace_back();
        bundles.emplace_back();
        if (FAILED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_BUNDLE,
                                                  IID_PPV_ARGS(&bundle_allocators.back().p))) ||
            FAILED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_BUNDLE, bundle_allocators.back().p, initial,
                                             IID_PPV_ARGS(&bundles.back().p))))
            return nullptr;
        record(bundles.back().p);
        if (FAILED(bundles.back()->Close())) return nullptr;
        return bundles.back().p;
    }

    // A viewport of size x size pixels at (x, y): the quad fills its top-left quarter, a cell when size is 16, an
    // offset of 1 moves it by size / 2 pixels, and what an offset moves out of the viewport is clipped.
    void viewport(ID3D12GraphicsCommandList* list, float x, float y, float size) {
        const D3D12_VIEWPORT v{x, y, size, size, 0, 1};
        list->RSSetViewports(1, &v);
    }
    void scissor(ID3D12GraphicsCommandList* list, LONG right) {
        const D3D12_RECT rect{0, 0, right, LONG(target_px)};
        list->RSSetScissorRects(1, &rect);
    }
    void vertices(ID3D12GraphicsCommandList* list, UINT64 offset) {
        const D3D12_VERTEX_BUFFER_VIEW view{upload_va + offset, 4 * sizeof(Vertex), sizeof(Vertex)};
        list->IASetVertexBuffers(0, 1, &view);
    }
    void indices(ID3D12GraphicsCommandList* list, UINT64 offset, UINT count) {
        const D3D12_INDEX_BUFFER_VIEW view{upload_va + offset, count * 2, DXGI_FORMAT_R16_UINT};
        list->IASetIndexBuffer(&view);
    }
    void offset(ID3D12GraphicsCommandList* list, float x, float y) {
        const float xy[2] = {x, y};
        list->SetGraphicsRoot32BitConstants(0, 2, xy, 0);
    }
    void tint(ID3D12GraphicsCommandList* list, UINT64 which) {
        list->SetGraphicsRootConstantBufferView(1, upload_va + which);
    }
    // The first draw's state: pso_a, root1, a strip of vb_quad, no offset, red, a cell-sized viewport at (0, 0).
    void start(ID3D12GraphicsCommandList* list, float size = 16) {
        list->OMSetRenderTargets(1, &rtv, FALSE, nullptr);
        scissor(list, LONG(target_px));
        viewport(list, 0, 0, size);
        list->SetGraphicsRootSignature(root1.p);
        list->SetPipelineState(pso_a.p);
        list->IASetPrimitiveTopology(strip);
        vertices(list, vb_quad);
        offset(list, 0, 0);
        tint(list, tint_red);
    }
    // A compute dispatch whose root constants, if a draw read them, would move it by (1, -1).
    void dispatch(ID3D12GraphicsCommandList* list) {
        const float constants[2] = {1, -1};
        list->SetComputeRootSignature(croot.p);
        list->SetComputeRoot32BitConstants(0, 2, constants, 0);
        list->SetComputeRootUnorderedAccessView(1, uav->GetGPUVirtualAddress());
        list->SetPipelineState(cs.p);
        list->Dispatch(1, 1, 1);
        D3D12_RESOURCE_BARRIER written{};  // the next dispatch writes the same words
        written.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
        written.UAV.pResource = uav.p;
        list->ResourceBarrier(1, &written);
    }
    // Clears `cell` (column, row) black.
    void clear_cell(ID3D12GraphicsCommandList* list, LONG column, LONG row) {
        const FLOAT black[4] = {0, 0, 0, 1};
        const D3D12_RECT rect{column * 8, row * 8, column * 8 + 8, row * 8 + 8};
        list->ClearRenderTargetView(rtv, black, 1, &rect);
    }

    // Clears the target blue, runs `record` on a direct list and reads the target back as letters.
    bool render(const Recorder& record, std::string& out, HRESULT& close) {
        Ref<ID3D12CommandAllocator> allocator;
        Ref<ID3D12GraphicsCommandList> list;
        CHECK(SUCCEEDED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator.p))));
        CHECK(SUCCEEDED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.p, nullptr,
                                                  IID_PPV_ARGS(&list.p))));
        D3D12_RESOURCE_BARRIER to_target = transition(target.p, D3D12_RESOURCE_STATE_COMMON,
                                                      D3D12_RESOURCE_STATE_RENDER_TARGET);
        list->ResourceBarrier(1, &to_target);
        const FLOAT blue[4] = {0, 0, 1, 1};
        list->ClearRenderTargetView(rtv, blue, 0, nullptr);
        record(*this, list.p);
        D3D12_RESOURCE_BARRIER to_copy = transition(target.p, D3D12_RESOURCE_STATE_RENDER_TARGET,
                                                    D3D12_RESOURCE_STATE_COPY_SOURCE);
        list->ResourceBarrier(1, &to_copy);
        D3D12_TEXTURE_COPY_LOCATION dst{}, src{};
        dst.pResource = readback.p;
        dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        dst.PlacedFootprint = {0, {DXGI_FORMAT_R8G8B8A8_UNORM, target_px, target_px, 1, 256}};
        src.pResource = target.p;
        src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
        close = list->Close();
        if (FAILED(close)) return true;  // reported by the caller
        ID3D12CommandList* lists[] = {list.p};
        queue->ExecuteCommandLists(1, lists);
        CHECK(SUCCEEDED(queue->Signal(fence.p, ++fence_value)));
        CHECK(SUCCEEDED(fence->SetEventOnCompletion(fence_value, nullptr)));
        const std::uint8_t* pixels = nullptr;
        const HRESULT mapped = readback->Map(0, nullptr, const_cast<void**>(reinterpret_cast<const void* const*>(&pixels)));
        if (FAILED(mapped)) {
            char text[96];
            std::snprintf(text, sizeof(text), "Map 0x%08lx, device removed 0x%08lx", static_cast<unsigned long>(mapped),
                          static_cast<unsigned long>(device->GetDeviceRemovedReason()));
            out = text;
            return true;
        }
        out.clear();
        for (UINT row = 0; row < grid_cells; ++row) {
            for (UINT column = 0; column < grid_cells; ++column) {
                // Pixel (3, 3) of the cell: inside the upper-left half of a cell-sized quad, outside the lower-right.
                const std::uint8_t* p = pixels + (row * 8 + 3) * 256 + (column * 8 + 3) * 4;
                char letter = '?';
                if (p[3] == 255) {
                    if (p[0] == 0 && p[1] == 0 && p[2] == 255) letter = 'B';
                    else if (p[0] == 255 && p[1] == 0 && p[2] == 0) letter = 'R';
                    else if (p[0] == 0 && p[1] == 255 && p[2] == 0) letter = 'G';
                    else if (p[0] == 255 && p[1] == 255 && p[2] == 0) letter = 'Y';
                    else if (p[0] == 0 && p[1] == 255 && p[2] == 255) letter = 'C';
                    else if (p[0] == 0 && p[1] == 0 && p[2] == 0) letter = 'K';
                }
                out += letter;
            }
            if (row + 1 < grid_cells) out += ' ';
        }
        readback->Unmap(0, nullptr);
        return true;
    }
};

struct Scenario {
    const char* name;
    Recorder record;
    const char* expected;  // null: print only
};

std::vector<Scenario> scenarios() {
    std::vector<Scenario> s;
    // Draw by draw; each comment names what changed and the cell (column, row) the quad lands in.
    s.push_back({"draw to draw: each draw changes some state and keeps the rest",
                 [](Rig& r, ID3D12GraphicsCommandList* l) {
                     r.start(l);
                     l->DrawInstanced(4, 1, 0, 0);  // (0, 0) R
                     r.offset(l, 1, 0);
                     l->DrawInstanced(4, 1, 0, 0);  // root constants: (1, 0) R
                     r.tint(l, tint_green);
                     l->DrawInstanced(4, 1, 0, 0);  // root CBV: (1, 0) G
                     r.viewport(l, 16, 0, 16);
                     l->DrawInstanced(4, 1, 0, 0);  // viewport: (3, 0) G
                     r.vertices(l, vb_other);
                     l->DrawInstanced(4, 1, 0, 0);  // vertex buffer: (3, 1) G
                     l->SetPipelineState(r.pso_b.p);
                     r.tint(l, tint_yellow);
                     r.vertices(l, vb_quad);
                     r.viewport(l, 32, 16, 16);
                     l->DrawInstanced(4, 1, 0, 0);  // pipeline, CBV, vertex buffer, viewport: (5, 2) C
                     l->SetPipelineState(r.pso_a.p);
                     l->DrawInstanced(4, 1, 0, 0);  // pipeline: (5, 2) Y
                     r.scissor(l, 32);
                     r.viewport(l, 32, 24, 16);
                     l->DrawInstanced(4, 1, 0, 0);  // scissor, viewport: (5, 3) cut away
                     r.scissor(l, LONG(target_px));
                     l->DrawInstanced(4, 1, 0, 0);  // scissor: (5, 3) Y
                     r.indices(l, ib_flip, 4);
                     r.viewport(l, 48, 32, 16);
                     l->DrawIndexedInstanced(4, 1, 0, 0, 0);  // indexed strip: (7, 4) Y
                     l->IASetPrimitiveTopology(tlist);
                     r.viewport(l, 48, 40, 16);
                     l->DrawIndexedInstanced(4, 1, 0, 0, 0);  // topology: a list, (7, 5) only the lower-right half
                     r.indices(l, ib_quad, 6);
                     r.viewport(l, 48, 48, 16);
                     l->DrawIndexedInstanced(6, 1, 0, 0, 0);  // index buffer: (7, 6) Y
                     l->SetPipelineState(r.pso_f.p);
                     const FLOAT red[4] = {1, 0, 0, 1}, green[4] = {0, 1, 0, 1};
                     l->OMSetBlendFactor(red);
                     r.viewport(l, 0, 32, 16);
                     l->DrawIndexedInstanced(6, 1, 0, 0, 0);  // pipeline, blend factor, viewport: (1, 4) R
                     l->OMSetBlendFactor(green);
                     r.viewport(l, 0, 40, 16);
                     l->DrawIndexedInstanced(6, 1, 0, 0, 0);  // blend factor, viewport: (1, 5) G
                     r.viewport(l, 0, 48, 16);
                     l->DrawIndexedInstanced(6, 1, 0, 0, 0);  // viewport: (1, 6) G
                     l->SetGraphicsRootSignature(r.root2.p);  // the arguments go: the same values again
                     l->SetPipelineState(r.pso_a2.p);
                     r.offset(l, 1, 0);
                     r.tint(l, tint_yellow);
                     const UINT zero = 0;
                     l->SetGraphicsRoot32BitConstants(2, 1, &zero, 0);
                     r.viewport(l, 16, 48, 16);
                     l->DrawIndexedInstanced(6, 1, 0, 0, 0);  // root signature, pipeline, viewport: (3, 6) Y
                 },
                 "RGBGBBBB BBBGBBBB BBBBBYBB BBBBBYBB BRBBBBBY BGBBBBBB BGBYBBBY " BLANK});
    // The quad covers 2x2 cells here. Clears, a copy and a barrier bind nothing: the draws after them set nothing,
    // and must still draw.
    s.push_back({"clears, a copy and a barrier between draws keep the state",
                 [](Rig& r, ID3D12GraphicsCommandList* l) {
                     r.start(l, 32);
                     l->DrawInstanced(4, 1, 0, 0);
                     const FLOAT blue[4] = {0, 0, 1, 1};
                     l->ClearRenderTargetView(r.rtv, blue, 0, nullptr);
                     l->CopyBufferRegion(r.results.p, 0, r.upload.p, 0, 256);
                     D3D12_RESOURCE_BARRIER uav{};
                     uav.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
                     l->ResourceBarrier(1, &uav);
                     l->DrawInstanced(4, 1, 0, 0);
                     r.clear_cell(l, 0, 0);
                     l->DrawInstanced(4, 1, 0, 0);
                     r.clear_cell(l, 1, 1);
                 },
                 "RRBBBBBB RKBBBBBB " BLANK " " BLANK " " BLANK " " BLANK " " BLANK " " BLANK});
    // A dispatch binds a compute pipeline and pushes its own constants: (1, -1) would move the draw two cells right
    // and down. The draw after it, with the same root arguments, must still draw where its own say.
    s.push_back({"a compute dispatch between draws",
                 [](Rig& r, ID3D12GraphicsCommandList* l) {
                     r.start(l, 32);
                     l->SetPipelineState(r.pso_b.p);
                     r.tint(l, tint_yellow);
                     l->DrawInstanced(4, 1, 0, 0);  // (0..1, 0..1) C
                     r.dispatch(l);
                     l->SetPipelineState(r.pso_a.p);
                     l->DrawInstanced(4, 1, 0, 0);  // the same arguments: (0..1, 0..1) Y
                     r.dispatch(l);
                     l->SetPipelineState(r.pso_a.p);
                     r.viewport(l, 32, 32, 32);
                     l->DrawInstanced(4, 1, 0, 0);  // the same pipeline and arguments: (4..5, 4..5) Y
                 },
                 "YYBBBBBB YYBBBBBB " BLANK " " BLANK " BBBBYYBB BBBBYYBB " BLANK " " BLANK});
    // A binary-occlusion resolve runs a dispatch of the layer's own between the two draws.
    s.push_back({"a binary-occlusion resolve between draws",
                 [](Rig& r, ID3D12GraphicsCommandList* l) {
                     r.start(l, 32);
                     l->SetPipelineState(r.pso_b.p);
                     r.tint(l, tint_yellow);
                     l->BeginQuery(r.queries.p, D3D12_QUERY_TYPE_BINARY_OCCLUSION, 0);
                     l->DrawInstanced(4, 1, 0, 0);  // (0..1, 0..1) C
                     l->EndQuery(r.queries.p, D3D12_QUERY_TYPE_BINARY_OCCLUSION, 0);
                     l->ResolveQueryData(r.queries.p, D3D12_QUERY_TYPE_BINARY_OCCLUSION, 0, 1, r.results.p, 0);
                     l->SetPipelineState(r.pso_a.p);
                     l->DrawInstanced(4, 1, 0, 0);  // (0..1, 0..1) Y
                     r.viewport(l, 32, 32, 32);
                     l->DrawInstanced(4, 1, 0, 0);  // (4..5, 4..5) Y
                 },
                 "YYBBBBBB YYBBBBBB " BLANK " " BLANK " BBBBYYBB BBBBYYBB " BLANK " " BLANK});
    // A bundle's draw runs on the parent's root arguments and viewport with its own pipeline, and hands that back.
    s.push_back({"a bundle between draws",
                 [](Rig& r, ID3D12GraphicsCommandList* l) {
                     auto* b = r.bundle(r.pso_b.p, [](ID3D12GraphicsCommandList* b) {
                         b->IASetPrimitiveTopology(strip);
                         b->DrawInstanced(4, 1, 0, 0);
                     });
                     r.start(l);
                     r.tint(l, tint_yellow);
                     l->DrawInstanced(4, 1, 0, 0);  // (0, 0) Y
                     r.offset(l, 1, 0);
                     if (b) l->ExecuteBundle(b);   // (1, 0) C
                     r.viewport(l, 16, 0, 16);
                     l->DrawInstanced(4, 1, 0, 0);  // pso_b, handed back; viewport: (3, 0) C
                     l->SetPipelineState(r.pso_a.p);
                     r.viewport(l, 32, 0, 16);
                     l->DrawInstanced(4, 1, 0, 0);  // pipeline, viewport: (5, 0) Y
                 },
                 "YCBCBYBB " BLANK " " BLANK " " BLANK " " BLANK " " BLANK " " BLANK " " BLANK});
    // A pipeline without vertex input leaves the vertex strides static: binding it invalidates the ones the vertex
    // buffer was bound with, so the next draw with vertex input binds the unchanged buffer again.
    s.push_back({"a pipeline without vertex input between draws",
                 [](Rig& r, ID3D12GraphicsCommandList* l) {
                     r.start(l);
                     l->DrawInstanced(4, 1, 0, 0);  // (0, 0) R
                     l->SetPipelineState(r.pso_fs.p);
                     r.tint(l, tint_green);
                     r.viewport(l, 16, 0, 16);
                     l->DrawInstanced(3, 1, 0, 0);  // the whole viewport: (2..3, 0..1) G
                     l->SetPipelineState(r.pso_a.p);
                     r.viewport(l, 32, 0, 16);
                     l->DrawInstanced(4, 1, 0, 0);  // the same vertex buffer: (4, 0) G
                 },
                 "RBGGGBBB BBGGBBBB " BLANK " " BLANK " " BLANK " " BLANK " " BLANK " " BLANK});
    return s;
}

bool compile_hlsl(const char* source, const char* entry, const char* profile, std::vector<std::uint8_t>& out) {
    const HMODULE compiler = LoadLibraryExA("d3dcompiler_47.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
    const auto compile = compiler ? reinterpret_cast<pD3DCompile>(GetProcAddress(compiler, "D3DCompile")) : nullptr;
    if (!compile) return false;
    ID3DBlob* blob = nullptr;
    ID3DBlob* errors = nullptr;
    const HRESULT hr = compile(source, std::strlen(source), "draw_state.hlsl", nullptr, nullptr, entry, profile, 0, 0,
                               &blob, &errors);
    if (FAILED(hr) && errors) std::fprintf(stderr, "%s\n", static_cast<const char*>(errors->GetBufferPointer()));
    if (errors) errors->Release();
    if (FAILED(hr) || !blob) return false;
    const auto* p = static_cast<const std::uint8_t*>(blob->GetBufferPointer());
    out.assign(p, p + blob->GetBufferSize());
    blob->Release();
    return true;
}

// The layer's shaders, for the host: test/glsl/vs_draw.vert and ps_draw.frag (s0, s1 offset the vertex, the root CBV
// tints), vs_fullscreen.vert, and a compute shader that, like cs_user_data.comp, writes base + step * i into the
// root UAV.
constexpr char hlsl[] =
    "cbuffer Offset : register(b0) { float2 offset; };\n"
    "cbuffer Tint : register(b1) { float3 tint; };\n"
    "struct V { float4 position : SV_Position; float4 color : COLOR; };\n"
    "V vs(float2 position : POSITION, float4 color : COLOR) {\n"
    "    V v; v.position = float4(position + offset, 0, 1); v.color = color; return v; }\n"
    "float4 ps(V v) : SV_Target { return float4(v.color.r * tint.x, tint.y, v.color.b * tint.z, 1); }\n"
    "V fs(uint id : SV_VertexID) {\n"
    "    V v; v.position = float4(float2(id & 2, (id << 1) & 2) * 2 - 1, 0, 1); v.color = float4(1, 1, 0, 1);\n"
    "    return v; }\n"
    "cbuffer Step : register(b0) { uint base; uint step; };\n"
    "RWByteAddressBuffer result : register(u0);\n"
    "[numthreads(64, 1, 1)] void cs(uint3 id : SV_DispatchThreadID) { result.Store(id.x * 4, base + step * id.x); }\n";

// Runs every scenario and stores the grids; false on a harness failure. A host scenario gets a device of its own.
bool run_all(ID3D12Device* shared, bool own_device, IDXGIAdapter* adapter, const Shaders& shaders,
             const std::vector<Scenario>& list, std::vector<std::string>& grids, std::vector<HRESULT>& closes) {
    for (const Scenario& scenario : list) {
        Rig rig;
        Ref<ID3D12Device> own;
        if (own_device) D3D12CreateDevice(adapter, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&own.p));
        ID3D12Device* device = own_device ? own.p : shared;
        if (!device || !rig.init(device, shaders)) {
            grids.push_back("device unusable");
            closes.push_back(E_FAIL);
            continue;
        }
        std::string grid_text;
        HRESULT close = S_OK;
        if (!rig.render(scenario.record, grid_text, close)) return false;
        grids.push_back(grid_text);
        closes.push_back(close);
    }
    return true;
}

}  // namespace

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    vktest::Device vk;
    const auto device_status = vk.initialize();
    if (device_status == vktest::DeviceStatus::unavailable) {
        std::fprintf(stderr, "SKIP: no usable Vulkan 1.3 device\n");
        return 77;
    }
    REQUIRE(device_status == vktest::DeviceStatus::ready);
    const auto list = scenarios();

    // Layer shaders: FSSM-shaped containers found by hash in a translator-style index.
    const auto vs_program = fssm_program({0x56530101, 0xbf810000});
    const auto ps_program = fssm_program({0x50530101, 0xbf810000});
    const auto cs_program = fssm_program({0x43530101, 0xbf810000});
    const auto fs_program = fssm_program({0x46530101, 0xbf810000});
    const Shaders layer_shaders{shader_container(vs_program), shader_container(ps_program),
                                shader_container(cs_program), shader_container(fs_program)};
    const testfiles::ScratchDir scratch("leonida_draw_state_test");
    const auto& library = scratch.path();
    REQUIRE(testfiles::copy_file(LEONIDA_TEST_VS_SPV, library / "vs_draw.spv"));
    REQUIRE(testfiles::copy_file(LEONIDA_TEST_PS_SPV, library / "ps_draw.spv"));
    REQUIRE(testfiles::copy_file(LEONIDA_TEST_CS_SPV, library / "cs_user_data.spv"));
    REQUIRE(testfiles::copy_file(LEONIDA_TEST_VS_FULLSCREEN_SPV, library / "vs_fullscreen.spv"));
    {
        std::ofstream index(library / "index.json", std::ios::binary);
        index << "{\"vs_draw\": {\"stage\": \"vs\", \"fssm_sha256\": \"" << leonida::sha256_hex(vs_program)
              << "\", \"bindings\": [], \"push_constant_bytes\": 144, \"input_signature\": ["
                 "{\"semantic\": \"COLOR\", \"semantic_index\": 0, \"register\": 0, \"mask\": 15, \"rw_mask\": 15, "
                 "\"system_value\": 0, \"component_type\": \"float\"}, "
                 "{\"semantic\": \"POSITION\", \"semantic_index\": 0, \"register\": 1, \"mask\": 3, \"rw_mask\": 3, "
                 "\"system_value\": 0, \"component_type\": \"float\"}]},\n"
                 " \"vs_fullscreen\": {\"stage\": \"vs\", \"fssm_sha256\": \"" << leonida::sha256_hex(fs_program)
              << "\", \"bindings\": [], \"push_constant_bytes\": 144, \"input_signature\": []},\n"
                 " \"ps_draw\": {\"stage\": \"ps\", \"fssm_sha256\": \"" << leonida::sha256_hex(ps_program)
              << "\", \"bindings\": [], \"push_constant_bytes\": 144, \"outputs\": [0]},\n"
                 " \"cs_user_data\": {\"stage\": \"cs\", \"fssm_sha256\": \"" << leonida::sha256_hex(cs_program)
              << "\", \"bindings\": [], \"push_constant_bytes\": 144, \"local_size\": [64, 1, 1], "
                 "\"uses_code_base\": true, \"wave64\": false}}";
    }
    const unsigned errors_before = vk.validation_errors();
    std::vector<std::string> layer_grids;
    std::vector<HRESULT> layer_closes;
    {
        rt::d3d12::VulkanDeviceInfo info{};
        info.get_instance_proc = vk.instance_proc;
        info.instance = vk.instance;
        info.physical = vk.physical;
        info.device = vk.device;
        info.queue = vk.queue;
        info.queue_family = vk.family;
        info.buffer_device_address = true;
        info.subgroup = vk.subgroup;
        info.shader_dir = library.string();
        Ref<ID3D12Device> ours;
        REQUIRE(SUCCEEDED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&ours.p))));
        REQUIRE(run_all(ours.p, false, nullptr, layer_shaders, list, layer_grids, layer_closes));
    }

    // The host: Windows' D3D12 on the default adapter and on WARP.
    Shaders host_shaders;
    const bool wine = hostwine::host_is_wine();
    const bool compiled = !wine && compile_hlsl(hlsl, "vs", "vs_5_0", host_shaders.vs) &&
                          compile_hlsl(hlsl, "ps", "ps_5_0", host_shaders.ps) &&
                          compile_hlsl(hlsl, "cs", "cs_5_0", host_shaders.cs) &&
                          compile_hlsl(hlsl, "fs", "vs_5_0", host_shaders.fs);
    struct HostRun {
        const char* name = "";
        bool ran = false;
        std::vector<std::string> grids;
        std::vector<HRESULT> closes;
    } hosts[2];
    hosts[0].name = "the default adapter";
    hosts[1].name = "WARP";
    if (compiled) {
        Ref<IDXGIFactory4> factory;
        REQUIRE(SUCCEEDED(CreateDXGIFactory1(IID_PPV_ARGS(&factory.p))));
        for (int i = 0; i < 2; ++i) {
            Ref<IDXGIAdapter> adapter;
            Ref<ID3D12Device> host;
            if (i == 1 && FAILED(factory->EnumWarpAdapter(IID_PPV_ARGS(&adapter.p)))) continue;
            if (FAILED(D3D12CreateDevice(adapter.p, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&host.p)))) continue;
            REQUIRE(run_all(nullptr, true, adapter.p, host_shaders, list, hosts[i].grids, hosts[i].closes));
            hosts[i].ran = true;
        }
    } else if (!wine) {
        std::fprintf(stderr, "no HLSL compiler: the host is not run\n");
    }

    unsigned failures = 0, compared = 0;
    const auto verdict = [&](const char* who, const Scenario& scenario, const std::string& grid, HRESULT close) {
        if (grid == scenario.expected && close == S_OK) return;
        std::fprintf(stderr, "FAIL %s on %s: Close 0x%08lx\n  got      %s\n  expected %s\n", scenario.name, who,
                     static_cast<unsigned long>(close), grid.c_str(), scenario.expected);
        ++failures;
    };
    for (std::size_t i = 0; i < list.size(); ++i) {
        if (!list[i].expected) {
            std::printf("%s | layer %s", list[i].name, layer_grids[i].c_str());
            for (const auto& host : hosts)
                if (host.ran) std::printf(" | %s %s", host.name, host.grids[i].c_str());
            std::printf("\n");
            ++failures;
            continue;
        }
        verdict("the layer", list[i], layer_grids[i], layer_closes[i]);
        for (const auto& host : hosts) {
            if (!host.ran) continue;
            verdict(host.name, list[i], host.grids[i], host.closes[i]);
            ++compared;
        }
    }
    const unsigned validation_errors = vk.validation_errors() - errors_before;
    if (validation_errors) {
        std::fprintf(stderr, "%u Vulkan validation errors\n", validation_errors);
        ++failures;
    }
    if (!hosts[0].ran && !hosts[1].ran && !wine)
        std::fprintf(stderr, "no D3D12 reference device (default adapter or WARP)\n");
    std::printf("draw state: %zu scenarios on the layer, %u host comparisons, %u failures\n", list.size(), compared,
                failures);
    return failures ? 1 : 0;
}
