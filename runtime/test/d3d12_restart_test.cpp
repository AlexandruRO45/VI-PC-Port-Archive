// Primitive restart (IBStripCutValue) through the Xbox-ABI proxy over the Vulkan device. Each case is one draw of
// two quads, vertices 0-3 on the left and 4-7 on the right of a 64x64 target, with a cut-valued index between them
// (test/glsl/vs_restart.vert). An index that does not cut is drawn as a vertex in the middle, so an uncut strip
// bridges the gap between the quads. Triangles are checked pixel by pixel, lines by the columns they reach.
// The expectations are the host PC D3D12 runtime's, measured with the same geometry on a Radeon 680M (driver
// 31.0.14084.6002), an RTX 3050 Laptop GPU (32.0.16.2101) and WARP, which all agree on these:
//   - a strip is cut where an index equals the cut value: 0xFFFF in 16-bit indices, 0xFFFFFFFF in 32-bit ones;
//     0xFFFFFFFF never cuts 16-bit indices, and 0xFFFF leaves a 32-bit 0xFFFFFFFF uncut;
//   - nothing is cut without a cut value, or in a draw without an index buffer;
//   - line strips are cut as triangle strips are.
// Two kinds of case pin what the layer does where it cannot follow every driver (docs/d3d12_vulkan.md):
//   - a list with a cut value draws the cut index as a vertex, as AMD's driver does; NVIDIA's and WARP end the list;
//   - 0xFFFF with 32-bit indices: D3D12 cuts at 0x0000FFFF, Vulkan restarts only at 0xFFFFFFFF, so it stays uncut.
// A cut value outside the enum is refused with E_INVALIDARG, as the host runtime refuses it.
// A device with less than 144 bytes of push constants refuses every PSO until #224's uniform-buffer tail exists;
// there the test says so and draws nothing. The shaders are GLSL written to the translator's contract; no game
// data. LEONIDA_VK_VALIDATION=1 runs the test under VK_LAYER_KHRONOS_validation.
#include "vk_test_device.h"
#include "d3d12_test_shaders.h"
#include "../shims/d3d12_com.h"
#include "../shims/d3d12_xs_device.h"
#include "../src/shader_library.h"

#include <algorithm>
#include <cstring>
#include <vector>

#if !defined(LEONIDA_TEST_RESTART_VS_SPV) || !defined(LEONIDA_TEST_RESTART_PS_SPV)
#error "LEONIDA_TEST_RESTART_VS_SPV and LEONIDA_TEST_RESTART_PS_SPV must name the compiled test shaders"
#endif

#define EXPECT(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return 1; } \
} while (0)

using namespace d3d12test;
using XboxHeapHandleFn = std::uint64_t(WINAPI*)(void* heap);  // heap slot 9: handle in RAX

namespace {

constexpr UINT kSize = 64;    // the render target, in pixels
constexpr UINT kPitch = 256;  // readback row pitch
constexpr std::uint32_t C16 = 0xFFFF, C32 = 0xFFFFFFFF;

struct Case {
    const char* name;
    D3D12_PRIMITIVE_TOPOLOGY_TYPE type;
    D3D12_INDEX_BUFFER_STRIP_CUT_VALUE cut;
    D3D_PRIMITIVE_TOPOLOGY topology;
    DXGI_FORMAT format;  // DXGI_FORMAT_UNKNOWN: DrawInstanced of vertices 0-7, no index buffer
    std::vector<std::uint32_t> indices;
    bool cut_expected;
};

// Pixels inside one of the two quads.
bool in_quad(UINT x, UINT y) { return y >= 8 && y < 56 && ((x >= 8 && x < 24) || (x >= 40 && x < 56)); }

}  // namespace

int main() {
    vktest::Device d;
    EXPECT(d.create());
    VkPhysicalDeviceProperties properties{};
    d.instance_fn<PFN_vkGetPhysicalDeviceProperties>("vkGetPhysicalDeviceProperties")(d.physical, &properties);
    const std::uint32_t max_index = properties.limits.maxDrawIndexedIndexValue;

    // Shader programs and the index that maps their hashes to the compiled test modules.
    const auto vs_program = fssm_program({0x56530010, 0xbf810000});
    const auto ps_program = fssm_program({0x50530010, 0xbf810000});
    const auto library = std::filesystem::temp_directory_path() / "leonida_restart_test";
    std::filesystem::create_directories(library);
    const auto copy = std::filesystem::copy_options::overwrite_existing;
    std::filesystem::copy_file(LEONIDA_TEST_RESTART_VS_SPV, library / "vs_restart.spv", copy);
    std::filesystem::copy_file(LEONIDA_TEST_RESTART_PS_SPV, library / "ps_restart.spv", copy);
    {
        std::ofstream index(library / "index.json", std::ios::binary);
        index << "{\"vs_restart\": {\"stage\": \"vs\", \"fssm_sha256\": \"" << leonida::sha256_hex(vs_program)
              << "\", \"bindings\": [], \"push_constant_bytes\": 144, \"input_signature\": []},\n"
                 " \"ps_restart\": {\"stage\": \"ps\", \"fssm_sha256\": \"" << leonida::sha256_hex(ps_program)
              << "\", \"bindings\": [], \"push_constant_bytes\": 144, \"outputs\": [0]}}";
    }
    const auto vs = shader_container(vs_program), ps = shader_container(ps_program);

    unsigned failures = 0, skipped = 0;
    bool draws_refused = false;
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

        // An empty root signature, and a PSO per topology type and cut value. No input layout: the VS reads
        // SV_VertexID only.
        D3D12_VERSIONED_ROOT_SIGNATURE_DESC root_desc{};
        root_desc.Version = D3D_ROOT_SIGNATURE_VERSION_1_0;
        const auto root_blob = serialize_root_signature(root_desc);
        EXPECT(!root_blob.empty());
        Ref<ID3D12RootSignature> root;
        EXPECT(SUCCEEDED(device->CreateRootSignature(0, root_blob.data(), root_blob.size(), IID_PPV_ARGS(&root.p))));
        D3D12_GRAPHICS_PIPELINE_STATE_DESC pso_desc{};
        pso_desc.pRootSignature = root.p;
        pso_desc.VS = {vs.data(), vs.size()};
        pso_desc.PS = {ps.data(), ps.size()};
        pso_desc.BlendState.RenderTarget[0].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
        pso_desc.SampleMask = 0xffffffff;
        pso_desc.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
        pso_desc.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
        pso_desc.RasterizerState.DepthClipEnable = TRUE;
        pso_desc.NumRenderTargets = 1;
        pso_desc.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
        pso_desc.SampleDesc.Count = 1;
        constexpr D3D12_PRIMITIVE_TOPOLOGY_TYPE types[2] = {D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
                                                            D3D12_PRIMITIVE_TOPOLOGY_TYPE_LINE};
        Ref<ID3D12PipelineState> psos[2][3];  // by type, then by IBStripCutValue
        for (int t = 0; t < 2 && !draws_refused; ++t) {
            for (int cut = 0; cut < 3; ++cut) {
                pso_desc.PrimitiveTopologyType = types[t];
                pso_desc.IBStripCutValue = D3D12_INDEX_BUFFER_STRIP_CUT_VALUE(cut);
                const HRESULT created = device->CreateGraphicsPipelineState(&pso_desc, IID_PPV_ARGS(&psos[t][cut].p));
                if (rt::d3d12::device_push_constants_need_fallback(device.p)) {
                    EXPECT(FAILED(created));
                    draws_refused = true;
                    break;
                }
                EXPECT(SUCCEEDED(created));
            }
        }
        if (draws_refused) {
            // Below 144 bytes of push constants every PSO is refused until #224's uniform-buffer tail exists.
            std::printf("maxPushConstantsSize %u: PSOs are refused below 144 bytes (#224), so no draws on this "
                        "device\n", rt::d3d12::device_max_push_constants_size(device.p));
            root.reset();
            EXPECT(rt::d3d12_xs::device_proxy_ref_count(device.p) == 1);
        } else {
            Ref<ID3D12PipelineState> invalid;
            pso_desc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
            pso_desc.IBStripCutValue = D3D12_INDEX_BUFFER_STRIP_CUT_VALUE(3);
            EXPECT(device->CreateGraphicsPipelineState(&pso_desc, IID_PPV_ARGS(&invalid.p)) == E_INVALIDARG &&
                   !invalid.p);

            const D3D12_HEAP_PROPERTIES default_heap{D3D12_HEAP_TYPE_DEFAULT};
            const D3D12_HEAP_PROPERTIES upload_heap{D3D12_HEAP_TYPE_UPLOAD};
            const D3D12_HEAP_PROPERTIES readback_heap{D3D12_HEAP_TYPE_READBACK};
            Ref<ID3D12Resource> target, readback, indices;
            D3D12_RESOURCE_DESC rt_desc{};
            rt_desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
            rt_desc.Width = rt_desc.Height = kSize;
            rt_desc.DepthOrArraySize = 1;
            rt_desc.MipLevels = 1;
            rt_desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
            rt_desc.SampleDesc.Count = 1;
            rt_desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
            EXPECT(SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &rt_desc,
                                                             D3D12_RESOURCE_STATE_RENDER_TARGET, nullptr,
                                                             IID_PPV_ARGS(&target.p))));
            const auto readback_desc = buffer_desc(kSize * kPitch);
            EXPECT(SUCCEEDED(device->CreateCommittedResource(&readback_heap, D3D12_HEAP_FLAG_NONE, &readback_desc,
                                                             D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                             IID_PPV_ARGS(&readback.p))));
            const auto index_desc = buffer_desc(256);
            EXPECT(SUCCEEDED(device->CreateCommittedResource(&upload_heap, D3D12_HEAP_FLAG_NONE, &index_desc,
                                                             D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
                                                             IID_PPV_ARGS(&indices.p))));
            std::uint8_t* up = nullptr;
            EXPECT(SUCCEEDED(indices->Map(0, nullptr, reinterpret_cast<void**>(&up))));
            void* mapped = nullptr;
            EXPECT(SUCCEEDED(readback->Map(0, nullptr, &mapped)));
            const auto* down = static_cast<const std::uint8_t*>(mapped);

            const D3D12_DESCRIPTOR_HEAP_DESC heap_desc{D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 1,
                                                       D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0};
            Ref<ID3D12DescriptorHeap> rtv_heap;
            EXPECT(SUCCEEDED(device->CreateDescriptorHeap(&heap_desc, IID_PPV_ARGS(&rtv_heap.p))));
            const D3D12_CPU_DESCRIPTOR_HANDLE rtv{
                SIZE_T(reinterpret_cast<XboxHeapHandleFn>(slot(rtv_heap.p, 9))(rtv_heap.p))};
            device->CreateRenderTargetView(target.p, nullptr, rtv);

            D3D12_COMMAND_QUEUE_DESC queue_desc{};
            Ref<ID3D12CommandQueue> queue;
            Ref<ID3D12CommandAllocator> allocator;
            Ref<ID3D12GraphicsCommandList> list;
            Ref<ID3D12Fence> fence;
            EXPECT(SUCCEEDED(device->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&queue.p))));
            EXPECT(SUCCEEDED(
                device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator.p))));
            EXPECT(SUCCEEDED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.p, nullptr,
                                                       IID_PPV_ARGS(&list.p))));
            EXPECT(list->Close() == S_OK);
            EXPECT(SUCCEEDED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence.p))));
            UINT64 fence_value = 0;

            const auto T = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE, L = D3D12_PRIMITIVE_TOPOLOGY_TYPE_LINE;
            const auto none = D3D12_INDEX_BUFFER_STRIP_CUT_VALUE_DISABLED;
            const auto cut16 = D3D12_INDEX_BUFFER_STRIP_CUT_VALUE_0xFFFF;
            const auto cut32 = D3D12_INDEX_BUFFER_STRIP_CUT_VALUE_0xFFFFFFFF;
            const auto strip = D3D_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP, tri_list = D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
            const auto line_strip = D3D_PRIMITIVE_TOPOLOGY_LINESTRIP;
            const auto r16 = DXGI_FORMAT_R16_UINT, r32 = DXGI_FORMAT_R32_UINT;
            const std::vector<std::uint32_t> strip16 = {0, 1, 2, 3, C16, 4, 5, 6, 7};
            const std::vector<std::uint32_t> strip32 = {0, 1, 2, 3, C32, 4, 5, 6, 7};
            const std::vector<std::uint32_t> list16 = {0, 1, 2, 1, 2, 3, C16, 4, 5, 4, 5, 6, 5, 6, 7};
            const std::vector<std::uint32_t> list32 = {0, 1, 2, 1, 2, 3, C32, 4, 5, 4, 5, 6, 5, 6, 7};
            const Case cases[] = {
                {"triangle strip, 16-bit, cut 0xFFFF: cut at 0xFFFF", T, cut16, strip, r16, strip16, true},
                {"triangle strip, 16-bit, no cut value: 0xFFFF drawn", T, none, strip, r16, strip16, false},
                {"triangle strip, 16-bit, cut 0xFFFFFFFF: 0xFFFF drawn", T, cut32, strip, r16, strip16, false},
                {"triangle strip, 32-bit, cut 0xFFFFFFFF: cut at 0xFFFFFFFF", T, cut32, strip, r32, strip32, true},
                {"triangle strip, 32-bit, no cut value: 0xFFFFFFFF drawn", T, none, strip, r32, strip32, false},
                {"triangle strip, 32-bit, cut 0xFFFF: 0xFFFFFFFF drawn", T, cut16, strip, r32, strip32, false},
                {"triangle strip, 32-bit, cut 0xFFFFFFFF: 0x0000FFFF drawn", T, cut32, strip, r32, strip16, false},
                {"triangle strip, 32-bit, cut 0xFFFF: 0x0000FFFF drawn (D3D12 cuts; Vulkan cannot)", T, cut16, strip,
                 r32, strip16, false},
                {"triangle list, 16-bit, cut 0xFFFF: 0xFFFF drawn (as AMD; NVIDIA and WARP cut)", T, cut16, tri_list,
                 r16, list16, false},
                {"triangle list, 32-bit, cut 0xFFFFFFFF: 0xFFFFFFFF drawn (as AMD; NVIDIA and WARP cut)", T, cut32,
                 tri_list, r32, list32, false},
                {"triangle strip without an index buffer, cut 0xFFFF: not cut", T, cut16, strip, DXGI_FORMAT_UNKNOWN,
                 {}, false},
                {"line strip, 16-bit, cut 0xFFFF: cut at 0xFFFF", L, cut16, line_strip, r16, {0, 3, C16, 4, 7}, true},
                {"line strip, 16-bit, no cut value: 0xFFFF drawn", L, none, line_strip, r16, {0, 3, C16, 4, 7}, false},
                {"line strip, 32-bit, cut 0xFFFFFFFF: cut at 0xFFFFFFFF", L, cut32, line_strip, r32,
                 {0, 3, C32, 4, 7}, true},
            };

            const D3D12_VIEWPORT viewport{0, 0, float(kSize), float(kSize), 0, 1};
            const D3D12_RECT scissor{0, 0, LONG(kSize), LONG(kSize)};
            for (const Case& c : cases) {
                // A 32-bit 0xFFFFFFFF drawn as a vertex needs the full index range.
                const bool full_range = c.format == r32 && !c.cut_expected &&
                                        std::find(c.indices.begin(), c.indices.end(), C32) != c.indices.end();
                if (full_range && max_index < C32) {
                    std::printf("  skipped (maxDrawIndexedIndexValue %u): %s\n", max_index, c.name);
                    ++skipped;
                    continue;
                }
                const UINT stride = c.format == r16 ? 2 : 4;
                for (std::size_t i = 0; i < c.indices.size(); ++i) {
                    if (stride == 2) {
                        const std::uint16_t value = std::uint16_t(c.indices[i]);
                        std::memcpy(up + i * 2, &value, 2);
                    } else {
                        std::memcpy(up + i * 4, &c.indices[i], 4);
                    }
                }

                EXPECT(SUCCEEDED(allocator->Reset()));
                EXPECT(SUCCEEDED(list->Reset(allocator.p, psos[c.type == L][c.cut].p)));
                list->SetGraphicsRootSignature(root.p);
                list->IASetPrimitiveTopology(c.topology);
                list->RSSetViewports(1, &viewport);
                list->RSSetScissorRects(1, &scissor);
                list->OMSetRenderTargets(1, &rtv, FALSE, nullptr);
                const FLOAT clear[4] = {0, 0, 0, 0};  // alpha 0: not drawn
                list->ClearRenderTargetView(rtv, clear, 0, nullptr);
                if (c.format != DXGI_FORMAT_UNKNOWN) {
                    const D3D12_INDEX_BUFFER_VIEW view{indices->GetGPUVirtualAddress(),
                                                       UINT(c.indices.size() * stride), c.format};
                    list->IASetIndexBuffer(&view);
                    list->DrawIndexedInstanced(UINT(c.indices.size()), 1, 0, 0, 0);
                } else {
                    list->DrawInstanced(8, 1, 0, 0);
                }
                D3D12_RESOURCE_BARRIER barrier =
                    transition(target.p, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_COPY_SOURCE);
                list->ResourceBarrier(1, &barrier);
                D3D12_TEXTURE_COPY_LOCATION dst{};
                dst.pResource = readback.p;
                dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
                dst.PlacedFootprint = {0, {DXGI_FORMAT_R8G8B8A8_UNORM, kSize, kSize, 1, kPitch}};
                D3D12_TEXTURE_COPY_LOCATION src{};
                src.pResource = target.p;
                src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
                list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
                barrier = transition(target.p, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_RENDER_TARGET);
                list->ResourceBarrier(1, &barrier);
                EXPECT(list->Close() == S_OK);
                ID3D12CommandList* lists[] = {list.p};
                queue->ExecuteCommandLists(1, lists);
                EXPECT(rt::d3d12::last_execute_result(queue.p) == leonida::ReplayResult::success);
                EXPECT(SUCCEEDED(queue->Signal(fence.p, ++fence_value)));
                EXPECT(SUCCEEDED(fence->SetEventOnCompletion(fence_value, nullptr)));

                const auto covered = [&](UINT x, UINT y) { return down[y * kPitch + x * 4 + 3] == 255; };
                bool ok = true;
                if (c.type == T) {
                    // Cut: the two quads and nothing else. Not cut: the quads, and the gap between them reached
                    // (36, 32 lies in the triangle from the middle vertex to the right quad's left edge).
                    unsigned bad = 0;
                    for (UINT y = 0; y < kSize; ++y) {
                        for (UINT x = 0; x < kSize; ++x) {
                            if (covered(x, y) == in_quad(x, y) || (!c.cut_expected && covered(x, y))) continue;
                            if (bad++ < 4)
                                std::fprintf(stderr, "%s: pixel (%u, %u) %s\n", c.name, x, y,
                                             covered(x, y) ? "drawn outside the quads" : "not drawn");
                        }
                    }
                    if (!c.cut_expected && !covered(36, 32)) {
                        std::fprintf(stderr, "%s: nothing drawn between the quads at (36, 32)\n", c.name);
                        ++bad;
                    }
                    ok = bad == 0;
                } else {
                    // Lines from (8, 8) to (24, 56) and from (40, 8) to (56, 56); uncut, two more reach (32, 32).
                    unsigned left = 0, gap = 0, right = 0;
                    for (UINT y = 0; y < kSize; ++y) {
                        for (UINT x = 0; x < kSize; ++x) {
                            if (!covered(x, y)) continue;
                            ++(x < 26 ? left : x < 38 ? gap : right);
                        }
                    }
                    ok = left && right && (c.cut_expected ? gap == 0 : gap != 0);
                    if (!ok)
                        std::fprintf(stderr, "%s: %u pixels left of x 26, %u from x 26 to 37, %u right of it\n",
                                     c.name, left, gap, right);
                }
                std::printf("  %s: %s\n", ok ? "ok" : "FAILED", c.name);
                if (!ok) ++failures;
            }

            for (auto* r : {&target, &readback, &indices}) r->reset();
            for (auto& by_type : psos)
                for (auto& pso : by_type) pso.reset();
            root.reset();
            rtv_heap.reset();
            list.reset();
            allocator.reset();
            queue.reset();
            fence.reset();
            EXPECT(rt::d3d12_xs::device_proxy_ref_count(device.p) == 1);
        }
    }
    std::filesystem::remove_all(library);

    if (failures) {
        std::fprintf(stderr, "%u cases failed\n", failures);
        return 1;
    }
    if (vktest::validation_error_count()) {
        std::fprintf(stderr, "%u validation errors\n", vktest::validation_error_count());
        return 1;
    }
    if (draws_refused) {
        std::printf("primitive restart: no draws, because this device refuses PSOs (#224)\n");
        return 0;
    }
    std::printf("primitive restart: every case matched%s\n", skipped ? " (some skipped, see above)" : "");
    return 0;
}
