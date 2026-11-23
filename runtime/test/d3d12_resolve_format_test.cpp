// ResolveSubresource with the Format parameter choosing how the samples are averaged (D3D-411), through the
// Xbox-ABI proxy over the Vulkan device. Each case clears a 4x target to blue and draws a full-screen green quad
// from a PSO with SampleMask 0x1, so every pixel holds one green sample and three blue ones. Its resolve tells the
// colour space the samples were averaged in:
//   - as UNORM:                   green 63.75 -> 64,  blue 191.25 -> 191;
//   - as sRGB, averaged linearly: green srgb(0.25) -> 137, blue srgb(0.75) -> 225 (the samples hold 0 or 255, which
//     are 0.0 and 1.0 in both encodings).
// The cases follow D3D12's rules for the Format parameter (ID3D12GraphicsCommandList::ResolveSubresource remarks):
//   typed_srgb       R8G8B8A8_UNORM_SRGB to R8G8B8A8_UNORM_SRGB, as UNORM_SRGB (the typed formats must all agree);
//   typeless_srgb    R8G8B8A8_TYPELESS to R8G8B8A8_TYPELESS, as UNORM_SRGB (both typeless: Format picks the member);
//   typeless_unorm   the same two textures, as UNORM;
//   mixed_srgb       R8G8B8A8_TYPELESS to R8G8B8A8_UNORM_SRGB, as UNORM_SRGB (one typed: Format is its format).
// The region_ cases resolve one rectangle with ResolveSubresourceRegion (AVERAGE) into a destination cleared to red
// first: the rectangle holds the average and every other pixel stays red.
// typeless_srgb_no_rt (a destination without ALLOW_RENDER_TARGET, so no attachment usage) and region_shifted_srgb
// (a rectangle moved to other coordinates) can't be resolved in place: they go through a one-sample scratch image and
// a copy. Native D3D12 and WARP give 137/225 for an sRGB resolve whether or not the destination is a render target.
// Shaders: test/glsl/vs_draw.vert, ps_draw.frag. No game data.
// Set LEONIDA_VK_VALIDATION=1 to run under VK_LAYER_KHRONOS_validation.
#include "shader_fixture_path.h"
#include "test_temp_dir.h"
#include "vk_test_device.h"
#include "d3d12_test_shaders.h"
#include "../shims/d3d12_com.h"
#include "../shims/d3d12_xs_device.h"
#include "../src/shader_library.h"

#include <array>
#include <cstdlib>
#include <cstring>

#if !defined(LEONIDA_TEST_VS_SPV) || !defined(LEONIDA_TEST_PS_SPV)
#error "LEONIDA_TEST_VS_SPV and LEONIDA_TEST_PS_SPV must name the compiled test shaders"
#endif

#define EXPECT(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return 1; } \
} while (0)

using namespace d3d12test;
using XboxHeapHandleFn = std::uint64_t(WINAPI*)(void* heap);  // heap slot 9: handle in RAX

namespace {

struct Vertex {
    float x, y;
    std::uint8_t r, g, b, a;
};

D3D12_CPU_DESCRIPTOR_HANDLE heap_start(ID3D12DescriptorHeap* heap) {
    return {SIZE_T(reinterpret_cast<XboxHeapHandleFn>(slot(heap, 9))(heap))};
}

struct Case {
    const char* name;
    DXGI_FORMAT target;   // the 4x texture
    DXGI_FORMAT view;     // its render-target view (and the PSO's RTV format)
    DXGI_FORMAT dest;     // the single-sampled texture
    DXGI_FORMAT resolve;  // ResolveSubresource's Format
    bool srgb;            // expect the linear sRGB average
    bool dest_rt;         // the destination is created with ALLOW_RENDER_TARGET (D3D12 needs no flag)
    bool region = false;  // ResolveSubresourceRegion of `rect` to (dst_x, dst_y), into a destination cleared red
    D3D12_RECT rect{};
    UINT dst_x = 0, dst_y = 0;
};

constexpr Case kCases[] = {
    {"typed_srgb", DXGI_FORMAT_R8G8B8A8_UNORM_SRGB, DXGI_FORMAT_R8G8B8A8_UNORM_SRGB, DXGI_FORMAT_R8G8B8A8_UNORM_SRGB,
     DXGI_FORMAT_R8G8B8A8_UNORM_SRGB, true, true},
    {"typeless_srgb", DXGI_FORMAT_R8G8B8A8_TYPELESS, DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_FORMAT_R8G8B8A8_TYPELESS,
     DXGI_FORMAT_R8G8B8A8_UNORM_SRGB, true, true},
    {"typeless_unorm", DXGI_FORMAT_R8G8B8A8_TYPELESS, DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_FORMAT_R8G8B8A8_TYPELESS,
     DXGI_FORMAT_R8G8B8A8_UNORM, false, true},
    {"mixed_srgb", DXGI_FORMAT_R8G8B8A8_TYPELESS, DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_FORMAT_R8G8B8A8_UNORM_SRGB,
     DXGI_FORMAT_R8G8B8A8_UNORM_SRGB, true, true},
    {"typeless_srgb_no_rt", DXGI_FORMAT_R8G8B8A8_TYPELESS, DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_FORMAT_R8G8B8A8_TYPELESS,
     DXGI_FORMAT_R8G8B8A8_UNORM_SRGB, true, false},
    {"region_typeless_srgb", DXGI_FORMAT_R8G8B8A8_TYPELESS, DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_FORMAT_R8G8B8A8_TYPELESS,
     DXGI_FORMAT_R8G8B8A8_UNORM_SRGB, true, true, true, {4, 4, 12, 10}, 4, 4},
    {"region_mixed_srgb", DXGI_FORMAT_R8G8B8A8_TYPELESS, DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_FORMAT_R8G8B8A8_UNORM_SRGB,
     DXGI_FORMAT_R8G8B8A8_UNORM_SRGB, true, true, true, {2, 6, 9, 15}, 2, 6},
    // A rectangle moved to other coordinates: a rendering scope resolves in place, so this one goes through the
    // scratch image and a copy, like a destination without ALLOW_RENDER_TARGET.
    {"region_shifted_srgb", DXGI_FORMAT_R8G8B8A8_TYPELESS, DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_FORMAT_R8G8B8A8_TYPELESS,
     DXGI_FORMAT_R8G8B8A8_UNORM_SRGB, true, true, true, {0, 0, 8, 8}, 8, 8},
};

}  // namespace

int main() {
    vktest::Device d;
    const auto device_status = d.initialize();
    if (device_status == vktest::DeviceStatus::unavailable) return 77;  // no Vulkan GPU (hosted CI): ctest skips
    EXPECT(device_status == vktest::DeviceStatus::ready);

    const auto vs_program = fssm_program({0x56530001, 0xbf810000});
    const auto ps_program = fssm_program({0x50530001, 0xbf810000});
    const auto vs_container = shader_container(vs_program), ps_container = shader_container(ps_program);
    const testfiles::ScratchDir scratch("leonida_resolve_format_test");
    const auto& library = scratch.path();
    EXPECT(testfiles::copy_file(vktest::shader_fixture_path(LEONIDA_TEST_VS_SPV), library / "vs_draw.spv"));
    EXPECT(testfiles::copy_file(vktest::shader_fixture_path(LEONIDA_TEST_PS_SPV), library / "ps_draw.spv"));
    {
        std::ofstream index(library / "index.json", std::ios::binary);
        index << "{\"vs_draw\": {\"stage\": \"vs\", \"fssm_sha256\": \"" << leonida::sha256_hex(vs_program)
              << "\", \"bindings\": [], \"push_constant_bytes\": 144, \"input_signature\": ["
                 "{\"semantic\": \"COLOR\", \"semantic_index\": 0, \"register\": 0, \"mask\": 15, \"rw_mask\": 15, "
                 "\"system_value\": 0, \"component_type\": \"float\"}, "
                 "{\"semantic\": \"POSITION\", \"semantic_index\": 0, \"register\": 1, \"mask\": 3, \"rw_mask\": 3, "
                 "\"system_value\": 0, \"component_type\": \"float\"}]},\n"
                 " \"ps_draw\": {\"stage\": \"ps\", \"fssm_sha256\": \"" << leonida::sha256_hex(ps_program)
              << "\", \"bindings\": [], \"push_constant_bytes\": 144, \"outputs\": [0]}}";
    }

    constexpr UINT size = 16;
    constexpr UINT samples = 4;
    constexpr std::size_t count = sizeof(kCases) / sizeof(kCases[0]);
    std::array<leonida::ReplayResult, count> results{};
    std::array<std::vector<std::uint8_t>, count> images;  // size x size RGBA8, tightly packed
    {
        rt::d3d12::VulkanDeviceInfo info{};
        info.get_instance_proc = d.instance_proc;
        info.instance = d.instance;
        info.physical = d.physical;
        info.device = d.device;
        info.queue = d.queue;
        info.queue_family = d.family;
        info.buffer_device_address = true;
        info.shader_dir = library.string();
        ID3D12Device* inner = nullptr;
        EXPECT(SUCCEEDED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&inner))));
        Ref<ID3D12Device> device;
        EXPECT(SUCCEEDED(rt::d3d12_xs::create_device_proxy(inner, __uuidof(ID3D12Device),
                                                           reinterpret_cast<void**>(&device.p))));
        rt::d3d12::set_device_outer(inner, device.p);
        if (rt::d3d12::device_push_constants_need_fallback(inner)) {
            std::printf("d3d12_resolve_format_test: skipped, the PSOs need 144 push-constant bytes and this GPU has "
                        "%u\n", rt::d3d12::device_max_push_constants_size(inner));
            return 77;
        }

        D3D12_ROOT_PARAMETER params[2] = {};
        params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
        params[0].Constants = {0, 0, 2};
        params[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
        params[1].Descriptor = {1, 0};
        D3D12_VERSIONED_ROOT_SIGNATURE_DESC root_desc{};
        root_desc.Version = D3D_ROOT_SIGNATURE_VERSION_1_0;
        root_desc.Desc_1_0 = {2, params, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT};
        const auto root_blob = serialize_root_signature(root_desc);
        EXPECT(!root_blob.empty());
        Ref<ID3D12RootSignature> root;
        EXPECT(SUCCEEDED(device->CreateRootSignature(0, root_blob.data(), root_blob.size(), IID_PPV_ARGS(&root.p))));

        const D3D12_INPUT_ELEMENT_DESC elements[2] = {
            {"POSITION", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 0, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
            {"COLOR", 0, DXGI_FORMAT_R8G8B8A8_UNORM, 0, D3D12_APPEND_ALIGNED_ELEMENT,
             D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0}};
        // One sample of four written: SampleMask 0x1. No depth.
        const auto make_pso = [&](DXGI_FORMAT rtv, Ref<ID3D12PipelineState>& out) {
            D3D12_GRAPHICS_PIPELINE_STATE_DESC pso_desc{};
            pso_desc.pRootSignature = root.p;
            pso_desc.VS = {vs_container.data(), vs_container.size()};
            pso_desc.PS = {ps_container.data(), ps_container.size()};
            pso_desc.BlendState.RenderTarget[0].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
            pso_desc.SampleMask = 0x1;
            pso_desc.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
            pso_desc.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
            pso_desc.RasterizerState.DepthClipEnable = TRUE;
            pso_desc.InputLayout = {elements, 2};
            pso_desc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
            pso_desc.NumRenderTargets = 1;
            pso_desc.RTVFormats[0] = rtv;
            pso_desc.SampleDesc = {samples, 0};
            return SUCCEEDED(device->CreateGraphicsPipelineState(&pso_desc, IID_PPV_ARGS(&out.p)));
        };
        Ref<ID3D12PipelineState> pso_unorm, pso_srgb;
        EXPECT(make_pso(DXGI_FORMAT_R8G8B8A8_UNORM, pso_unorm));
        EXPECT(make_pso(DXGI_FORMAT_R8G8B8A8_UNORM_SRGB, pso_srgb));

        // A full-screen quad; the tint makes it pure green (ps_draw: r * t0, t1, b * t2).
        const Vertex vertices[4] = {
            {-1, 1, 0, 0, 0, 255}, {1, 1, 0, 0, 0, 255}, {-1, -1, 0, 0, 0, 255}, {1, -1, 0, 0, 0, 255}};
        const std::uint16_t indices[6] = {0, 1, 2, 2, 1, 3};
        const float green[3] = {0, 1, 0};

        const D3D12_HEAP_PROPERTIES default_heap{D3D12_HEAP_TYPE_DEFAULT};
        const D3D12_HEAP_PROPERTIES upload_heap{D3D12_HEAP_TYPE_UPLOAD};
        const D3D12_HEAP_PROPERTIES readback_heap{D3D12_HEAP_TYPE_READBACK};
        Ref<ID3D12Resource> upload, readback;
        const auto upload_desc = buffer_desc(4096);
        EXPECT(SUCCEEDED(device->CreateCommittedResource(&upload_heap, D3D12_HEAP_FLAG_NONE, &upload_desc,
                                                         D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
                                                         IID_PPV_ARGS(&upload.p))));
        const auto readback_desc = buffer_desc(size * 256);
        EXPECT(SUCCEEDED(device->CreateCommittedResource(&readback_heap, D3D12_HEAP_FLAG_NONE, &readback_desc,
                                                         D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                         IID_PPV_ARGS(&readback.p))));
        std::uint8_t* up = nullptr;
        EXPECT(SUCCEEDED(upload->Map(0, nullptr, reinterpret_cast<void**>(&up))));
        std::memcpy(up, vertices, sizeof(vertices));
        std::memcpy(up + 256, indices, sizeof(indices));
        std::memcpy(up + 512, green, sizeof(green));
        const D3D12_GPU_VIRTUAL_ADDRESS upload_va = upload->GetGPUVirtualAddress();

        const D3D12_DESCRIPTOR_HEAP_DESC rtv_heap_desc{D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 2,
                                                       D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0};
        Ref<ID3D12DescriptorHeap> rtv_heap;
        EXPECT(SUCCEEDED(device->CreateDescriptorHeap(&rtv_heap_desc, IID_PPV_ARGS(&rtv_heap.p))));
        const D3D12_CPU_DESCRIPTOR_HANDLE rtv = heap_start(rtv_heap.p);
        const D3D12_CPU_DESCRIPTOR_HANDLE dest_rtv{rtv.ptr + rt::d3d12::descriptor_size};

        D3D12_COMMAND_QUEUE_DESC queue_desc{};
        Ref<ID3D12CommandQueue> queue;
        Ref<ID3D12CommandAllocator> allocator;
        Ref<ID3D12GraphicsCommandList> list;
        Ref<ID3D12Fence> fence;
        EXPECT(SUCCEEDED(device->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&queue.p))));
        EXPECT(SUCCEEDED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
                                                        IID_PPV_ARGS(&allocator.p))));
        EXPECT(SUCCEEDED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.p, pso_unorm.p,
                                                   IID_PPV_ARGS(&list.p))));
        EXPECT(list->Close() == S_OK);
        EXPECT(SUCCEEDED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence.p))));
        UINT64 fence_value = 0;

        for (std::size_t c = 0; c < count; ++c) {
            const Case& k = kCases[c];
            D3D12_RESOURCE_DESC rt_desc{};
            rt_desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
            rt_desc.Width = rt_desc.Height = size;
            rt_desc.DepthOrArraySize = 1;
            rt_desc.MipLevels = 1;
            rt_desc.Format = k.target;
            rt_desc.SampleDesc = {samples, 0};
            rt_desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
            Ref<ID3D12Resource> target, resolved;
            EXPECT(SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &rt_desc,
                                                             D3D12_RESOURCE_STATE_RENDER_TARGET, nullptr,
                                                             IID_PPV_ARGS(&target.p))));
            D3D12_RESOURCE_DESC single_desc = rt_desc;
            single_desc.Format = k.dest;
            single_desc.SampleDesc = {1, 0};
            single_desc.Flags = k.dest_rt ? D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET : D3D12_RESOURCE_FLAG_NONE;
            EXPECT(SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &single_desc,
                                                             D3D12_RESOURCE_STATE_RESOLVE_DEST, nullptr,
                                                             IID_PPV_ARGS(&resolved.p))));
            D3D12_RENDER_TARGET_VIEW_DESC ms_view{};
            ms_view.Format = k.view;
            ms_view.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2DMS;
            device->CreateRenderTargetView(target.p, &ms_view, rtv);

            EXPECT(SUCCEEDED(allocator->Reset()));
            ID3D12PipelineState* pso = k.view == DXGI_FORMAT_R8G8B8A8_UNORM_SRGB ? pso_srgb.p : pso_unorm.p;
            EXPECT(SUCCEEDED(list->Reset(allocator.p, pso)));
            if (k.region) {  // the destination red first, so the pixels outside the rectangle can be checked
                D3D12_RENDER_TARGET_VIEW_DESC dest_view{};
                dest_view.Format = k.dest == DXGI_FORMAT_R8G8B8A8_TYPELESS ? DXGI_FORMAT_R8G8B8A8_UNORM : k.dest;
                dest_view.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2D;
                device->CreateRenderTargetView(resolved.p, &dest_view, dest_rtv);
                D3D12_RESOURCE_BARRIER to_rt = transition(resolved.p, D3D12_RESOURCE_STATE_RESOLVE_DEST,
                                                          D3D12_RESOURCE_STATE_RENDER_TARGET);
                list->ResourceBarrier(1, &to_rt);
                const FLOAT red[4] = {1, 0, 0, 1};
                list->ClearRenderTargetView(dest_rtv, red, 0, nullptr);
                D3D12_RESOURCE_BARRIER back = transition(resolved.p, D3D12_RESOURCE_STATE_RENDER_TARGET,
                                                         D3D12_RESOURCE_STATE_RESOLVE_DEST);
                list->ResourceBarrier(1, &back);
            }
            const FLOAT blue[4] = {0, 0, 1, 1};
            list->ClearRenderTargetView(rtv, blue, 0, nullptr);
            list->SetGraphicsRootSignature(root.p);
            list->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            const D3D12_VERTEX_BUFFER_VIEW vbv{upload_va, sizeof(vertices), sizeof(Vertex)};
            list->IASetVertexBuffers(0, 1, &vbv);
            const D3D12_INDEX_BUFFER_VIEW ibv{upload_va + 256, sizeof(indices), DXGI_FORMAT_R16_UINT};
            list->IASetIndexBuffer(&ibv);
            const D3D12_VIEWPORT viewport{0, 0, float(size), float(size), 0, 1};
            const D3D12_RECT scissor{0, 0, LONG(size), LONG(size)};
            list->RSSetViewports(1, &viewport);
            list->RSSetScissorRects(1, &scissor);
            list->OMSetRenderTargets(1, &rtv, FALSE, nullptr);
            const float origin[2] = {0, 0};
            list->SetGraphicsRoot32BitConstants(0, 2, origin, 0);
            list->SetGraphicsRootConstantBufferView(1, upload_va + 512);
            list->DrawIndexedInstanced(6, 1, 0, 0, 0);

            D3D12_RESOURCE_BARRIER to_resolve = transition(target.p, D3D12_RESOURCE_STATE_RENDER_TARGET,
                                                           D3D12_RESOURCE_STATE_RESOLVE_SOURCE);
            list->ResourceBarrier(1, &to_resolve);
            if (k.region) {
                Ref<ID3D12GraphicsCommandList1> list1;
                EXPECT(SUCCEEDED(list->QueryInterface(IID_PPV_ARGS(&list1.p))));
                D3D12_RECT rect = k.rect;
                list1->ResolveSubresourceRegion(resolved.p, 0, k.dst_x, k.dst_y, target.p, 0, &rect, k.resolve,
                                                D3D12_RESOLVE_MODE_AVERAGE);
            } else {
                list->ResolveSubresource(resolved.p, 0, target.p, 0, k.resolve);
            }
            D3D12_RESOURCE_BARRIER to_copy = transition(resolved.p, D3D12_RESOURCE_STATE_RESOLVE_DEST,
                                                        D3D12_RESOURCE_STATE_COPY_SOURCE);
            list->ResourceBarrier(1, &to_copy);
            D3D12_TEXTURE_COPY_LOCATION dst{};
            dst.pResource = readback.p;
            dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
            // The bytes as they are: a typeless destination is copied out through its UNORM member.
            const DXGI_FORMAT footprint =
                k.dest == DXGI_FORMAT_R8G8B8A8_TYPELESS ? DXGI_FORMAT_R8G8B8A8_UNORM : k.dest;
            dst.PlacedFootprint = {0, {footprint, size, size, 1, 256}};
            D3D12_TEXTURE_COPY_LOCATION src{};
            src.pResource = resolved.p;
            src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
            list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
            EXPECT(list->Close() == S_OK);
            ID3D12CommandList* lists[] = {list.p};
            queue->ExecuteCommandLists(1, lists);
            results[c] = rt::d3d12::last_execute_result(queue.p);
            EXPECT(SUCCEEDED(queue->Signal(fence.p, ++fence_value)));
            EXPECT(SUCCEEDED(fence->SetEventOnCompletion(fence_value, nullptr)));
            std::uint8_t* mapped = nullptr;
            EXPECT(SUCCEEDED(readback->Map(0, nullptr, reinterpret_cast<void**>(&mapped))));
            images[c].resize(std::size_t(size) * size * 4);
            for (UINT row = 0; row < size; ++row)
                std::memcpy(images[c].data() + std::size_t(row) * size * 4, mapped + row * 256, size * 4);
            std::memset(mapped, 0xee, size * 256);  // the next case can't pass on this case's bytes
            readback->Unmap(0, nullptr);
        }

        for (auto* r : {&upload, &readback}) r->reset();
        pso_unorm.reset();
        pso_srgb.reset();
        root.reset();
        rtv_heap.reset();
        list.reset();
        allocator.reset();
        queue.reset();
        fence.reset();
        EXPECT(rt::d3d12_xs::device_proxy_ref_count(device.p) == 1);
    }

    // One green sample and three blue ones: UNORM 64/191; sRGB averaged in linear space 137/225 (+-2 for the
    // implementation-defined precision Vulkan allows).
    unsigned failures = 0;
    for (std::size_t c = 0; c < count; ++c) {
        const Case& k = kCases[c];
        const int want_g = k.srgb ? 137 : 64, want_b = k.srgb ? 225 : 191;
        // Inside the resolved area: the average. Outside it (region cases only): the red the destination was
        // cleared to.
        const UINT w = k.region ? UINT(k.rect.right - k.rect.left) : size;
        const UINT h = k.region ? UINT(k.rect.bottom - k.rect.top) : size;
        unsigned wrong = 0, inside = 0;
        std::array<std::uint8_t, 4> sample{};  // the first pixel inside the area
        if (results[c] == leonida::ReplayResult::success && !images[c].empty()) {
            for (UINT row = 0; row < size; ++row) {
                for (UINT col = 0; col < size; ++col) {
                    const std::uint8_t* px = images[c].data() + (std::size_t(row) * size + col) * 4;
                    const bool in = col >= k.dst_x && col < k.dst_x + w && row >= k.dst_y && row < k.dst_y + h;
                    if (in && inside++ == 0) std::memcpy(sample.data(), px, 4);
                    const bool good = in ? px[0] == 0 && std::abs(int(px[1]) - want_g) <= 2 &&
                                               std::abs(int(px[2]) - want_b) <= 2 && px[3] == 255
                                         : px[0] == 255 && px[1] == 0 && px[2] == 0 && px[3] == 255;
                    if (!good) ++wrong;
                }
            }
        }
        const bool ok = results[c] == leonida::ReplayResult::success && inside == w * h && !wrong;
        std::printf("%-20s replay %d, pixel %u,%u,%u,%u, %u wrong of %u, want 0,%d,%d,255 (%s)%s: %s\n", k.name,
                    int(results[c]), sample[0], sample[1], sample[2], sample[3], wrong, size * size, want_g,
                    want_b, k.srgb ? "sRGB, linear average" : "UNORM", k.region ? ", red outside" : "",
                    ok ? "ok" : "MISMATCH");
        if (!ok) ++failures;
    }
    if (d.validation_errors()) {
        std::fprintf(stderr, "%u validation errors\n", d.validation_errors());
        return 1;
    }
    EXPECT(!failures);
    return 0;
}
