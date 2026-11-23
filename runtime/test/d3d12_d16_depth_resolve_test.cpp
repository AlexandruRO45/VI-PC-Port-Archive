// Regression test for D16_UNORM / R16_TYPELESS depth ResolveSubresourceRegion using the depth plane's SRV format
// DXGI_FORMAT_R16_UNORM (MIN, MAX, AVERAGE) as well as DXGI_FORMAT_D16_UNORM (MIN, MAX), both whole subresource and
// sub-rectangle, plus an edge-draw check where multisampled pixels have differing sample depths.
// Compares the Vulkan layer against expected 16-bit UNORM clear values and against Windows D3D12 (layer's adapter
// and WARP) with Vulkan validation enabled.
#include "vk_test_device.h"
#include "host_adapter.h"
#include "host_wine.h"
#include "../shims/d3d12_com.h"
#if defined(LEONIDA_DEPTH_EDGE_VS_SPV)
#include "d3d12_test_shaders.h"
#include "../src/shader_library.h"

#include <d3dcompiler.h>
#endif
#include <dxgi1_4.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#define CHECK(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return false; } \
} while (0)

namespace {
template<class T>
struct Ref {
    T* p = nullptr;
    ~Ref() { if (p) p->Release(); }
    T* operator->() const { return p; }
};
struct Event {
    HANDLE handle = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    ~Event() { if (handle) CloseHandle(handle); }
};

constexpr UINT kWidth = 8, kHeight = 6;
constexpr D3D12_RECT kClearA{1, 1, 5, 4}, kClearB{4, 3, 8, 6}, kSourceRect{2, 1, 7, 5};

struct Case {
    const char* name;
    DXGI_FORMAT format;
    D3D12_RESOLVE_MODE mode;
    bool rect;
    DXGI_FORMAT resource;
    DXGI_FORMAT view;
    bool edge = false;
    bool host_refuses = false;
};

struct EdgeDraw {
    const std::vector<std::uint8_t>* root = nullptr;
    const std::vector<std::uint8_t>* vs = nullptr;
};

struct Result {
    HRESULT close = E_FAIL;
    std::vector<std::uint16_t> depth;
};

const char* mode_name(D3D12_RESOLVE_MODE mode) {
    switch (mode) {
    case D3D12_RESOLVE_MODE_MIN: return "MIN";
    case D3D12_RESOLVE_MODE_MAX: return "MAX";
    case D3D12_RESOLVE_MODE_AVERAGE: return "AVERAGE";
    default: return "?";
    }
}

std::uint16_t quantize_d16(float value) {
    return static_cast<std::uint16_t>(double(value) * 65535.0 + 0.5);
}

bool resolve_and_read(ID3D12Device* device, const Case& c, Result& out, bool run_invalid,
                      const EdgeDraw* edge = nullptr) {
    Ref<ID3D12CommandQueue> queue;
    Ref<ID3D12CommandAllocator> allocator;
    Ref<ID3D12GraphicsCommandList> list;
    Ref<ID3D12GraphicsCommandList1> list1;
    Ref<ID3D12Fence> fence;
    D3D12_COMMAND_QUEUE_DESC queue_desc{};
    CHECK(SUCCEEDED(device->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&queue.p))));
    CHECK(SUCCEEDED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator.p))));
    CHECK(SUCCEEDED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.p, nullptr,
                                             IID_PPV_ARGS(&list.p))));
    CHECK(SUCCEEDED(list->QueryInterface(IID_PPV_ARGS(&list1.p))));
    CHECK(SUCCEEDED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence.p))));

    D3D12_HEAP_PROPERTIES heap_props{};
    heap_props.Type = D3D12_HEAP_TYPE_DEFAULT;
    D3D12_RESOURCE_DESC tex{};
    tex.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    tex.Width = kWidth;
    tex.Height = kHeight;
    tex.DepthOrArraySize = tex.MipLevels = 1;
    tex.Format = c.resource;
    tex.SampleDesc.Count = 4;
    tex.Flags = D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;
    Ref<ID3D12Resource> source, target, readback;
    CHECK(SUCCEEDED(device->CreateCommittedResource(&heap_props, D3D12_HEAP_FLAG_NONE, &tex,
                                                    D3D12_RESOURCE_STATE_DEPTH_WRITE, nullptr,
                                                    IID_PPV_ARGS(&source.p))));
    tex.SampleDesc.Count = 1;
    CHECK(SUCCEEDED(device->CreateCommittedResource(&heap_props, D3D12_HEAP_FLAG_NONE, &tex,
                                                    D3D12_RESOURCE_STATE_DEPTH_WRITE, nullptr,
                                                    IID_PPV_ARGS(&target.p))));

    D3D12_PLACED_SUBRESOURCE_FOOTPRINT placed{};
    UINT64 total = 0;
    device->GetCopyableFootprints(&tex, 0, 1, 0, &placed, nullptr, nullptr, &total);
    D3D12_RESOURCE_DESC buf{};
    buf.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    buf.Width = total;
    buf.Height = buf.SampleDesc.Count = 1;
    buf.DepthOrArraySize = buf.MipLevels = 1;
    buf.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    heap_props.Type = D3D12_HEAP_TYPE_READBACK;
    CHECK(SUCCEEDED(device->CreateCommittedResource(&heap_props, D3D12_HEAP_FLAG_NONE, &buf,
                                                    D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                    IID_PPV_ARGS(&readback.p))));

    Ref<ID3D12DescriptorHeap> heap;
    const D3D12_DESCRIPTOR_HEAP_DESC heap_desc{D3D12_DESCRIPTOR_HEAP_TYPE_DSV, 2, D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0};
    CHECK(SUCCEEDED(device->CreateDescriptorHeap(&heap_desc, IID_PPV_ARGS(&heap.p))));
    auto source_view = heap->GetCPUDescriptorHandleForHeapStart();
    auto target_view = source_view;
    target_view.ptr += device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_DSV);
    D3D12_DEPTH_STENCIL_VIEW_DESC dsv{c.view, D3D12_DSV_DIMENSION_TEXTURE2DMS};
    device->CreateDepthStencilView(source.p, &dsv, source_view);
    dsv.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2D;
    device->CreateDepthStencilView(target.p, &dsv, target_view);

    // Avoid 0.5 in clears so there is no rounding tie on 16-bit UNORM.
    list->ClearDepthStencilView(source_view, D3D12_CLEAR_FLAG_DEPTH, 0.25f, 0, 0, nullptr);
    list->ClearDepthStencilView(source_view, D3D12_CLEAR_FLAG_DEPTH, 0.75f, 0, 1, &kClearA);
    list->ClearDepthStencilView(source_view, D3D12_CLEAR_FLAG_DEPTH, 1.0f, 0, 1, &kClearB);
    list->ClearDepthStencilView(target_view, D3D12_CLEAR_FLAG_DEPTH, 0.0f, 0, 0, nullptr);

    Ref<ID3D12RootSignature> root;
    Ref<ID3D12PipelineState> pso;
    if (c.edge) {
        CHECK(edge && edge->root && edge->vs);
        CHECK(SUCCEEDED(device->CreateRootSignature(0, edge->root->data(), edge->root->size(), IID_PPV_ARGS(&root.p))));
        D3D12_GRAPHICS_PIPELINE_STATE_DESC pd{};
        pd.pRootSignature = root.p;
        pd.VS = {edge->vs->data(), edge->vs->size()};
        pd.SampleMask = UINT_MAX;
        pd.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
        pd.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
        pd.RasterizerState.DepthClipEnable = TRUE;
        pd.DepthStencilState.DepthEnable = TRUE;
        pd.DepthStencilState.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ALL;
        pd.DepthStencilState.DepthFunc = D3D12_COMPARISON_FUNC_ALWAYS;
        pd.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
        pd.DSVFormat = c.view;
        pd.SampleDesc = {4, 0};
        CHECK(SUCCEEDED(device->CreateGraphicsPipelineState(&pd, IID_PPV_ARGS(&pso.p))));
        const D3D12_VIEWPORT viewport{0, 0, float(kWidth), float(kHeight), 0, 1};
        const D3D12_RECT scissor{0, 0, LONG(kWidth), LONG(kHeight)};
        list->SetGraphicsRootSignature(root.p);
        list->SetPipelineState(pso.p);
        list->OMSetRenderTargets(0, nullptr, FALSE, &source_view);
        list->RSSetViewports(1, &viewport);
        list->RSSetScissorRects(1, &scissor);
        list->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        list->DrawInstanced(3, 1, 0, 0);
    }

    D3D12_RESOURCE_BARRIER barriers[2]{};
    for (auto& b : barriers) {
        b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        b.Transition.StateBefore = D3D12_RESOURCE_STATE_DEPTH_WRITE;
    }
    barriers[0].Transition.pResource = source.p;
    barriers[0].Transition.StateAfter = D3D12_RESOURCE_STATE_RESOLVE_SOURCE;
    barriers[1].Transition.pResource = target.p;
    barriers[1].Transition.StateAfter = D3D12_RESOURCE_STATE_RESOLVE_DEST;
    list->ResourceBarrier(2, barriers);

    D3D12_RECT rect = kSourceRect;
    const UINT x = c.rect ? UINT(rect.left) : 0;
    const UINT y = c.rect ? UINT(rect.top) : 0;
    list1->ResolveSubresourceRegion(target.p, 0, x, y, source.p, 0, c.rect ? &rect : nullptr, c.format, c.mode);

    barriers[1].Transition.StateBefore = D3D12_RESOURCE_STATE_RESOLVE_DEST;
    barriers[1].Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
    list->ResourceBarrier(1, &barriers[1]);

    D3D12_TEXTURE_COPY_LOCATION src_loc{}, dst_loc{};
    src_loc.pResource = target.p;
    src_loc.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    src_loc.SubresourceIndex = 0;
    dst_loc.pResource = readback.p;
    dst_loc.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    dst_loc.PlacedFootprint = placed;
    list->CopyTextureRegion(&dst_loc, 0, 0, 0, &src_loc, nullptr);

    out.close = list->Close();
    if (FAILED(out.close) && !run_invalid) return true;
    ID3D12CommandList* lists[] = {list.p};
    queue->ExecuteCommandLists(1, lists);
    CHECK(SUCCEEDED(queue->Signal(fence.p, 1)));
    Event event;
    CHECK(event.handle);
    CHECK(SUCCEEDED(fence->SetEventOnCompletion(1, event.handle)));
    CHECK(WaitForSingleObject(event.handle, 30000) == WAIT_OBJECT_0);

    void* mapped = nullptr;
    const D3D12_RANGE range{0, SIZE_T(total)};
    CHECK(SUCCEEDED(readback->Map(0, &range, &mapped)) && mapped);
    const auto* base = static_cast<const unsigned char*>(mapped);
    out.depth.resize(kWidth * kHeight);
    for (UINT row = 0; row < kHeight; ++row)
        std::memcpy(out.depth.data() + row * kWidth, base + placed.Offset + row * placed.Footprint.RowPitch,
                    kWidth * sizeof(std::uint16_t));
    const D3D12_RANGE none{};
    readback->Unmap(0, &none);
    return true;
}

std::vector<std::uint16_t> expected_depth(const Case& c, bool resolved) {
    std::vector<std::uint16_t> out;
    out.reserve(kWidth * kHeight);
    const auto in = [](const D3D12_RECT& r, UINT x, UINT y) {
        return LONG(x) >= r.left && LONG(x) < r.right && LONG(y) >= r.top && LONG(y) < r.bottom;
    };
    const D3D12_RECT area = c.rect ? kSourceRect : D3D12_RECT{0, 0, LONG(kWidth), LONG(kHeight)};
    for (UINT y = 0; y < kHeight; ++y) {
        for (UINT x = 0; x < kWidth; ++x) {
            float d = 0.25f;
            if (in(kClearA, x, y)) d = 0.75f;
            if (in(kClearB, x, y)) d = 1.0f;
            if (!resolved || !in(area, x, y)) d = 0.0f;
            out.push_back(quantize_d16(d));
        }
    }
    return out;
}
}  // namespace

int main() {
    vktest::Device vk;
    vk.request_buffer_device_address = true;
    const auto device_status = vk.initialize();
    if (device_status == vktest::DeviceStatus::unavailable) return 77;
    if (device_status != vktest::DeviceStatus::ready) return 1;
    const unsigned errors_before = vk.validation_errors();

    VkPhysicalDeviceDepthStencilResolveProperties resolve{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DEPTH_STENCIL_RESOLVE_PROPERTIES};
    VkPhysicalDeviceProperties2 props{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &resolve};
    vk.instance_fn<PFN_vkGetPhysicalDeviceProperties2>("vkGetPhysicalDeviceProperties2")(vk.physical, &props);
    std::printf("Vulkan depth resolve modes: 0x%x\n", unsigned(resolve.supportedDepthResolveModes));

    constexpr auto d16 = DXGI_FORMAT_D16_UNORM;
    constexpr auto r16 = DXGI_FORMAT_R16_UNORM;
    constexpr auto r16_typeless = DXGI_FORMAT_R16_TYPELESS;
    constexpr auto min_m = D3D12_RESOLVE_MODE_MIN;
    constexpr auto max_m = D3D12_RESOLVE_MODE_MAX;
    constexpr auto avg_m = D3D12_RESOLVE_MODE_AVERAGE;

    std::vector<Case> cases;
    for (const bool rect : {false, true}) {
        for (const auto mode : {min_m, max_m}) {
            cases.push_back({"D16 depth", d16, mode, rect, d16, d16});
            cases.push_back({"R16_TYPELESS depth as D16", d16, mode, rect, r16_typeless, d16});
        }
        for (const auto mode : {min_m, max_m, avg_m}) {
            cases.push_back({"D16 depth as R16_UNORM", r16, mode, rect, d16, d16});
            cases.push_back({"R16_TYPELESS depth as R16_UNORM", r16, mode, rect, r16_typeless, d16});
        }
    }
    // Edge draw cases: samples along the diagonal differ, exercising MIN, MAX, and AVERAGE on R16_UNORM.
    for (const auto mode : {min_m, max_m, avg_m})
        cases.push_back({"D16 edge as R16_UNORM", r16, mode, false, d16, d16, true});

    const Case refused[] = {
        {"D16 depth AVERAGE (must use R16_UNORM)", d16, avg_m, false, d16, d16, false, true},
        {"R16_TYPELESS depth AVERAGE as D16", d16, avg_m, true, r16_typeless, d16, false, true},
        {"D16 depth with mismatched R32_FLOAT format", DXGI_FORMAT_R32_FLOAT, min_m, false, d16, d16, false, true},
    };

    std::vector<std::uint8_t> root_blob, edge_container, edge_host_vs;
    rt::d3d12::VulkanDeviceInfo info{};
#if defined(LEONIDA_DEPTH_EDGE_VS_SPV)
    {
        const auto program = d3d12test::fssm_program({0x56530848, 0xbf810000});
        edge_container = d3d12test::shader_container(program);
        const auto library = std::filesystem::temp_directory_path() / "leonida_d16_depth_resolve_test";
        std::filesystem::create_directories(library);
        std::filesystem::copy_file(LEONIDA_DEPTH_EDGE_VS_SPV, library / "vs.spv",
                                   std::filesystem::copy_options::overwrite_existing);
        std::ofstream index(library / "index.json", std::ios::binary);
        index << "{\"vs\": {\"stage\": \"vs\", \"fssm_sha256\": \"" << leonida::sha256_hex(program)
              << "\", \"bindings\": [], \"push_constant_bytes\": 144, \"input_signature\": []}}";
        info.shader_dir = library.string();
        D3D12_VERSIONED_ROOT_SIGNATURE_DESC root_desc{};
        root_desc.Version = D3D_ROOT_SIGNATURE_VERSION_1_0;
        root_blob = d3d12test::serialize_root_signature(root_desc);
        const HMODULE compiler = LoadLibraryExA("d3dcompiler_47.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
        const auto compile = compiler ? reinterpret_cast<pD3DCompile>(reinterpret_cast<void*>(
                                            GetProcAddress(compiler, "D3DCompile")))
                                      : nullptr;
        constexpr char host_vs[] =
            "float4 main(uint id : SV_VertexID) : SV_Position {\n"
            "    const float2 corners[3] = {float2(-1, -1), float2(-1, 1), float2(1, 1)};\n"
            "    return float4(corners[id], 0.5, 1.0);\n"
            "}\n";
        ID3DBlob* code = nullptr;
        ID3DBlob* errors = nullptr;
        if (compile && SUCCEEDED(compile(host_vs, std::strlen(host_vs), nullptr, nullptr, nullptr, "main", "vs_5_0", 0,
                                         0, &code, &errors)) && code) {
            const auto* bytes = static_cast<const std::uint8_t*>(code->GetBufferPointer());
            edge_host_vs.assign(bytes, bytes + code->GetBufferSize());
        }
        if (errors) errors->Release();
        if (code) code->Release();
        if (root_blob.empty() || (edge_host_vs.empty() && !hostwine::host_is_wine())) {
            std::fprintf(stderr, "failed to prepare edge draw shader\n");
            return 1;
        }
    }
#endif
    const bool edge_draws = !edge_container.empty();
    const EdgeDraw ours_edge{&root_blob, &edge_container}, host_edge{&root_blob, &edge_host_vs};
    unsigned failures = 0, passed = 0, matched_host = 0, refused_ok = 0, references = 0;

    const auto run = [&]() -> int {
        Ref<ID3D12Device> ours;
        info.get_instance_proc = vk.instance_proc;
        info.instance = vk.instance;
        info.physical = vk.physical;
        info.device = vk.device;
        info.queue = vk.queue;
        info.queue_family = vk.family;
        info.bindless = vk.bindless;
        info.bindless_partially_bound = vk.partially_bound;
        info.host_image_copy = vk.host_image_copy;
        info.buffer_device_address = vk.buffer_device_address;
        if (FAILED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&ours.p)))) return 1;

        std::vector<Result> results(cases.size());
        for (std::size_t i = 0; i < cases.size(); ++i) {
            const Case& c = cases[i];
            if (c.edge && !edge_draws) continue;
            const VkResolveModeFlagBits bit = c.mode == min_m   ? VK_RESOLVE_MODE_MIN_BIT
                                            : c.mode == max_m   ? VK_RESOLVE_MODE_MAX_BIT
                                                                : VK_RESOLVE_MODE_AVERAGE_BIT;
            const bool can = (resolve.supportedDepthResolveModes & bit) != 0;
            if (!resolve_and_read(ours.p, c, results[i], true, &ours_edge)) return 1;
            const auto expected = expected_depth(c, can);
            if (results[i].close != (can ? S_OK : E_INVALIDARG) || (!c.edge && results[i].depth != expected)) {
                std::fprintf(stderr, "%s %s rect %d: Close 0x%08lx (expected 0x%08lx), depth %s\n", c.name,
                             mode_name(c.mode), int(c.rect), results[i].close, can ? S_OK : E_INVALIDARG,
                             results[i].depth == expected ? "matches" : "differs");
                ++failures;
            } else {
                ++passed;
            }
            if (!can) results[i].depth.clear();
        }

        // Verify the edge draw actually produced mixed samples so MIN, MAX, and AVERAGE differ.
        if (edge_draws) {
            const Result *r_min = nullptr, *r_max = nullptr, *r_avg = nullptr;
            for (std::size_t i = 0; i < cases.size(); ++i) {
                if (!cases[i].edge || results[i].depth.empty()) continue;
                if (cases[i].mode == min_m) r_min = &results[i];
                else if (cases[i].mode == max_m) r_max = &results[i];
                else if (cases[i].mode == avg_m) r_avg = &results[i];
            }
            if (r_min && r_max && r_min->depth == r_max->depth) {
                std::fprintf(stderr, "edge D16 as R16_UNORM: MIN and MAX agree everywhere\n");
                ++failures;
            }
            if (r_avg && r_min && r_max && (r_avg->depth == r_min->depth || r_avg->depth == r_max->depth)) {
                std::fprintf(stderr, "edge D16 as R16_UNORM: AVERAGE equals MIN or MAX everywhere\n");
                ++failures;
            }
        }

        for (const Case& c : refused) {
            Result r;
            if (!resolve_and_read(ours.p, c, r, true)) return 1;
            if (r.close != E_INVALIDARG || r.depth != expected_depth(c, false)) {
                std::fprintf(stderr, "refused %s %s rect %d: Close 0x%08lx, destination %s\n", c.name,
                             mode_name(c.mode), int(c.rect), r.close,
                             r.depth == expected_depth(c, false) ? "kept" : "changed");
                ++failures;
            } else {
                ++refused_ok;
            }
        }

        Ref<IDXGIFactory4> factory;
        if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory.p)))) return 1;
        for (const bool warp : {false, true}) {
            if (hostwine::host_is_wine()) break;
            Ref<IDXGIAdapter> adapter;
            Ref<ID3D12Device> host;
            std::string adapter_name;
            if (!hostadapter::reference(factory.p, vk, warp, &adapter.p, adapter_name)) continue;
            if (FAILED(D3D12CreateDevice(adapter.p, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&host.p)))) continue;
            unsigned same = 0, compared = 0, refused_alike = 0;
            for (std::size_t i = 0; i < cases.size(); ++i) {
                const Case& c = cases[i];
                // WARP removes the device on ResolveSubresourceRegion with a D* format (same quirk as D32_FLOAT).
                if (results[i].depth.empty() || (warp && c.format == d16)) continue;
                Result host_res;
                if (!resolve_and_read(host.p, c, host_res, false, &host_edge)) return 1;
                ++compared;
                if (c.edge && warp) {
                    if (SUCCEEDED(host_res.close) && host_res.depth == results[i].depth) ++same;
                    continue;
                }
                if (FAILED(host_res.close) || host_res.depth != results[i].depth) {
                    int max_diff = 0;
                    std::size_t first_idx = 0;
                    for (std::size_t k = 0; k < results[i].depth.size() && k < host_res.depth.size(); ++k) {
                        const int d = std::abs(int(host_res.depth[k]) - int(results[i].depth[k]));
                        if (d > max_diff) {
                            max_diff = d;
                            first_idx = k;
                        }
                    }
                    if (c.edge && c.mode == avg_m && SUCCEEDED(host_res.close) && max_diff <= 1) {
                        std::printf("%s %s rect %d against %s: within 1 ULP (max_diff=%d, pixel %zu: layer=%u host=%u)\n",
                                    c.name, mode_name(c.mode), int(c.rect), adapter_name.c_str(), max_diff, first_idx,
                                    unsigned(results[i].depth[first_idx]), unsigned(host_res.depth[first_idx]));
                        ++same;
                    } else {
                        std::fprintf(stderr, "%s %s rect %d against %s: host Close 0x%08lx, max_diff=%d (pixel %zu: layer=%u host=%u)\n",
                                     c.name, mode_name(c.mode), int(c.rect), adapter_name.c_str(), host_res.close,
                                     max_diff, first_idx,
                                     first_idx < results[i].depth.size() ? unsigned(results[i].depth[first_idx]) : 0u,
                                     first_idx < host_res.depth.size() ? unsigned(host_res.depth[first_idx]) : 0u);
                        ++failures;
                    }
                } else {
                    ++same;
                }
            }
            for (const Case& c : refused) {
                if (!c.host_refuses) continue;
                Result host_res;
                if (!resolve_and_read(host.p, c, host_res, false)) return 1;
                if (host_res.close != E_INVALIDARG) {
                    std::fprintf(stderr, "refused %s %s against %s: host Close 0x%08lx\n", c.name, mode_name(c.mode),
                                 adapter_name.c_str(), host_res.close);
                    ++failures;
                } else {
                    ++refused_alike;
                }
            }
            std::printf("D16 depth resolves against %s: %u of %u match, %u refusals alike\n", adapter_name.c_str(),
                        same, compared, refused_alike);
            matched_host += same;
            ++references;
        }
        return 0;
    };

    const int run_failed = run();
    const auto validation_errors = vk.validation_errors() - errors_before;
    if (validation_errors)
        std::fprintf(stderr, "d3d12_d16_depth_resolve_test: %u new Vulkan validation error(s)\n", validation_errors);
    const bool need_host = !hostwine::host_is_wine();
    if (run_failed || failures || validation_errors || (need_host && (!references || !matched_host))) return 1;
    std::printf("D16 depth resolves: %u layer cases passed, %u host comparisons matched, %u refusals kept destination\n",
                passed, matched_host, refused_ok);
    return 0;
}
