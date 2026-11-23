// Compare depth/stencil ResolveSubresourceRegion with Windows D3D12 on the layer's own adapter and on WARP, through
// clears, resolves, copies and Map on both sides: MIN and MAX of either plane, named by the depth format or the
// plane's own format, and AVERAGE of the depth plane by its own format (R32_FLOAT, R24_UNORM_X8_TYPELESS).
// Clears write whole pixels, so all samples of a pixel are equal and every mode returns that value: these cases
// check planes, formats, rectangles and layouts, not the reduction over differing samples. D32_FLOAT_S8X24_UINT takes
// the same cases as D24S8, by its own format and by its planes' (R32_FLOAT_X8X24_TYPELESS, X32_TYPELESS_G8X24_UINT)
// (TEST-847).
// Edge cases (TEST-848) then draw one triangle over the cleared source with a depth-only PSO (depth 0.5, stencil
// 0x44, test ALWAYS): its diagonal edge leaves the pixels it crosses with samples that differ, so MIN, MAX and
// AVERAGE reduce over different values. No CPU model of the coverage: the layer must match the host byte for byte,
// and MIN and MAX must differ somewhere. The layer's VS is a GLSL stand-in (test/glsl/vs_depth_edge.vert, found by
// the FSSM hash of its container), the host's HLSL compiled by d3dcompiler_47; without glslangValidator at build
// time the edge cases are skipped.
// Shifted rectangles verify the source-to-destination mapping, including disjoint and overlapping resolves in a
// list. Every pixel of both destination planes is checked. Successive lists reuse a DIRECT queue after a fence wait.
// Refused resolves must fail Close and leave the destination as it was. AVERAGE by the depth format or of stencil,
// and a whole subresource to another offset, the host refuses too (its Close fails).
// WARP, measured: a resolve with Format D32_FLOAT removes the device (DXGI_ERROR_DRIVER_INTERNAL_ERROR), and a
// D24S8 resolve of one plane writes both. So WARP skips those and is compared on the plane resolved.
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

constexpr UINT width = 8, height = 6;
constexpr D3D12_COMMAND_QUEUE_DESC direct_queue{D3D12_COMMAND_LIST_TYPE_DIRECT};
constexpr D3D12_RECT clear_a{1, 1, 5, 4}, clear_b{4, 3, 8, 6}, source_rect{2, 1, 7, 5};
constexpr D3D12_RECT whole{0, 0, LONG(width), LONG(height)};

struct ResolveRegion {
    D3D12_RECT source;
    LONG x, y;
};

struct Case {
    const char* name;
    DXGI_FORMAT format;
    bool stencil;  // the format has a stencil plane (subresource 1)
    UINT plane;    // the plane resolved
    D3D12_RESOLVE_MODE mode;
    bool rect;     // source_rect instead of the whole subresource
    LONG dst_x = -1, dst_y = -1;  // the destination offset; -1: source_rect's corner, or 0 without it
    bool host_refuses = false;    // refused cases: the host's Close fails as well
    DXGI_FORMAT resource = DXGI_FORMAT_UNKNOWN;  // the resources' format, when not `format`
    DXGI_FORMAT view = DXGI_FORMAT_UNKNOWN;      // the DSV format, when not `format`
    bool edge = false;  // a triangle drawn over the clears, so the samples of the pixels on its edge differ
    const ResolveRegion* second = nullptr;      // optional second resolve on the same resources and plane
    bool enhanced = false;
};

// The depth plane holds 24-bit UNORM (D24S8 and its typeless and plane formats) rather than a float.
bool unorm24(const Case& c) {
    switch (c.resource ? c.resource : c.format) {
    case DXGI_FORMAT_D24_UNORM_S8_UINT:
    case DXGI_FORMAT_R24G8_TYPELESS:
    case DXGI_FORMAT_R24_UNORM_X8_TYPELESS:
    case DXGI_FORMAT_X24_TYPELESS_G8_UINT: return true;
    default: return false;
    }
}

// The edge draw's PSO on one device: a depth-only pipeline over the source's DSV format, 4 samples.
struct EdgeDraw {
    const std::vector<std::uint8_t>* root = nullptr;  // a serialized empty root signature
    const std::vector<std::uint8_t>* vs = nullptr;    // the host's DXBC, or the layer's shader container
};

ResolveRegion first_region(const Case& c) {
    const auto& source = c.rect ? source_rect : whole;
    return {source, c.dst_x >= 0 ? c.dst_x : source.left, c.dst_y >= 0 ? c.dst_y : source.top};
}

struct Planes {
    HRESULT close = E_FAIL;
    std::vector<unsigned char> depth, stencil;
};

// Clears a 4x target with three overlapping values and the 1x destination to values none of them have, resolves
// `c` and reads every plane of the destination back. A list whose Close fails is executed only when `run_invalid`
// (the layer runs what it recorded; the host runtime must not be given such a list).
bool resolve_and_read(ID3D12Device* device, ID3D12CommandQueue* queue, const Case& c, Planes& out, bool run_invalid,
                      const EdgeDraw* edge = nullptr) {
    Ref<ID3D12CommandAllocator> allocator;
    Ref<ID3D12GraphicsCommandList> list;
    Ref<ID3D12GraphicsCommandList1> list1;
    Ref<ID3D12GraphicsCommandList7> list7;
    Ref<ID3D12Fence> fence;
    CHECK(SUCCEEDED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator.p))));
    CHECK(SUCCEEDED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.p, nullptr,
                                             IID_PPV_ARGS(&list.p))));
    CHECK(SUCCEEDED(list->QueryInterface(IID_PPV_ARGS(&list1.p))));
    if (c.enhanced) CHECK(SUCCEEDED(list->QueryInterface(IID_PPV_ARGS(&list7.p))));
    CHECK(SUCCEEDED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence.p))));

    D3D12_HEAP_PROPERTIES heap_properties{};
    heap_properties.Type = D3D12_HEAP_TYPE_DEFAULT;
    D3D12_RESOURCE_DESC texture{};
    texture.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    texture.Width = width;
    texture.Height = height;
    texture.DepthOrArraySize = texture.MipLevels = 1;
    texture.Format = c.resource ? c.resource : c.format;
    texture.SampleDesc.Count = 4;
    texture.Flags = D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;
    Ref<ID3D12Resource> source, target, readback;
    CHECK(SUCCEEDED(device->CreateCommittedResource(&heap_properties, D3D12_HEAP_FLAG_NONE, &texture,
        D3D12_RESOURCE_STATE_DEPTH_WRITE, nullptr, IID_PPV_ARGS(&source.p))));
    texture.SampleDesc.Count = 1;
    CHECK(SUCCEEDED(device->CreateCommittedResource(&heap_properties, D3D12_HEAP_FLAG_NONE, &texture,
        D3D12_RESOURCE_STATE_DEPTH_WRITE, nullptr, IID_PPV_ARGS(&target.p))));
    const UINT planes = c.stencil ? 2u : 1u;
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT placed[2]{};
    UINT64 total = 0;
    device->GetCopyableFootprints(&texture, 0, planes, 0, placed, nullptr, nullptr, &total);
    D3D12_RESOURCE_DESC buffer{};
    buffer.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    buffer.Width = total;
    buffer.Height = buffer.SampleDesc.Count = 1;
    buffer.DepthOrArraySize = buffer.MipLevels = 1;
    buffer.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    heap_properties.Type = D3D12_HEAP_TYPE_READBACK;
    CHECK(SUCCEEDED(device->CreateCommittedResource(&heap_properties, D3D12_HEAP_FLAG_NONE, &buffer,
        D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&readback.p))));

    Ref<ID3D12DescriptorHeap> heap;
    const D3D12_DESCRIPTOR_HEAP_DESC desc{D3D12_DESCRIPTOR_HEAP_TYPE_DSV, 2, D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0};
    CHECK(SUCCEEDED(device->CreateDescriptorHeap(&desc, IID_PPV_ARGS(&heap.p))));
    auto source_view = heap->GetCPUDescriptorHandleForHeapStart(), target_view = source_view;
    target_view.ptr += device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_DSV);
    D3D12_DEPTH_STENCIL_VIEW_DESC view{c.view ? c.view : c.format, D3D12_DSV_DIMENSION_TEXTURE2DMS};
    device->CreateDepthStencilView(source.p, &view, source_view);
    view.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2D;
    device->CreateDepthStencilView(target.p, &view, target_view);
    const auto flags = static_cast<D3D12_CLEAR_FLAGS>(D3D12_CLEAR_FLAG_DEPTH | (c.stencil ? D3D12_CLEAR_FLAG_STENCIL : 0));
    // No 0.5: it is a 24-bit rounding tie that drivers break differently (docs/d3d12_vulkan.md, D24 clears).
    list->ClearDepthStencilView(source_view, flags, 0.25f, 0x11, 0, nullptr);
    list->ClearDepthStencilView(source_view, flags, 0.75f, 0x22, 1, &clear_a);
    list->ClearDepthStencilView(source_view, flags, 1.0f, 0x33, 1, &clear_b);
    list->ClearDepthStencilView(target_view, flags, 0.0f, 0x5a, 0, nullptr);
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
        pd.DepthStencilState.StencilEnable = c.stencil;
        pd.DepthStencilState.StencilReadMask = pd.DepthStencilState.StencilWriteMask = 0xff;
        const D3D12_DEPTH_STENCILOP_DESC replace{D3D12_STENCIL_OP_KEEP, D3D12_STENCIL_OP_KEEP, D3D12_STENCIL_OP_REPLACE,
                                                 D3D12_COMPARISON_FUNC_ALWAYS};
        pd.DepthStencilState.FrontFace = pd.DepthStencilState.BackFace = replace;
        pd.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
        pd.DSVFormat = c.view ? c.view : c.format;
        pd.SampleDesc = {4, 0};
        CHECK(SUCCEEDED(device->CreateGraphicsPipelineState(&pd, IID_PPV_ARGS(&pso.p))));
        const D3D12_VIEWPORT viewport{0, 0, float(width), float(height), 0, 1};
        const D3D12_RECT scissor{0, 0, LONG(width), LONG(height)};
        list->SetGraphicsRootSignature(root.p);
        list->SetPipelineState(pso.p);
        list->OMSetRenderTargets(0, nullptr, FALSE, &source_view);
        list->OMSetStencilRef(0x44);
        list->RSSetViewports(1, &viewport);
        list->RSSetScissorRects(1, &scissor);
        list->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        list->DrawInstanced(3, 1, 0, 0);
    }

    constexpr D3D12_BARRIER_SUBRESOURCE_RANGE all{D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES};
    const auto enhanced_barrier = [&](const D3D12_TEXTURE_BARRIER* barriers, UINT count) {
        D3D12_BARRIER_GROUP group{D3D12_BARRIER_TYPE_TEXTURE, count};
        group.pTextureBarriers = barriers;
        list7->Barrier(1, &group);
    };
    D3D12_RESOURCE_BARRIER legacy[2]{};
    if (list7.p) {
        const D3D12_TEXTURE_BARRIER to_resolve[] = {
            {D3D12_BARRIER_SYNC_DEPTH_STENCIL, D3D12_BARRIER_SYNC_RESOLVE,
             D3D12_BARRIER_ACCESS_DEPTH_STENCIL_WRITE, D3D12_BARRIER_ACCESS_RESOLVE_SOURCE,
             D3D12_BARRIER_LAYOUT_DEPTH_STENCIL_WRITE, D3D12_BARRIER_LAYOUT_RESOLVE_SOURCE, source.p, all},
            {D3D12_BARRIER_SYNC_DEPTH_STENCIL, D3D12_BARRIER_SYNC_RESOLVE,
             D3D12_BARRIER_ACCESS_DEPTH_STENCIL_WRITE, D3D12_BARRIER_ACCESS_RESOLVE_DEST,
             D3D12_BARRIER_LAYOUT_DEPTH_STENCIL_WRITE, D3D12_BARRIER_LAYOUT_RESOLVE_DEST, target.p, all},
        };
        enhanced_barrier(to_resolve, 2);
    } else {
        for (auto& barrier : legacy) {
            barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
            barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
            barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_DEPTH_WRITE;
        }
        legacy[0].Transition.pResource = source.p;
        legacy[0].Transition.StateAfter = D3D12_RESOURCE_STATE_RESOLVE_SOURCE;
        legacy[1].Transition.pResource = target.p;
        legacy[1].Transition.StateAfter = D3D12_RESOURCE_STATE_RESOLVE_DEST;
        list->ResourceBarrier(2, legacy);
    }
    auto first = first_region(c);
    list1->ResolveSubresourceRegion(target.p, c.plane, UINT(first.x), UINT(first.y), source.p, c.plane,
                                    c.rect ? &first.source : nullptr, c.format, c.mode);
    if (c.second) {
        if (list7.p) {
            const D3D12_TEXTURE_BARRIER order{
                D3D12_BARRIER_SYNC_RESOLVE, D3D12_BARRIER_SYNC_RESOLVE,
                D3D12_BARRIER_ACCESS_RESOLVE_DEST, D3D12_BARRIER_ACCESS_RESOLVE_DEST,
                D3D12_BARRIER_LAYOUT_RESOLVE_DEST, D3D12_BARRIER_LAYOUT_RESOLVE_DEST, target.p, all};
            enhanced_barrier(&order, 1);
        }
        auto second = *c.second;
        list1->ResolveSubresourceRegion(target.p, c.plane, UINT(second.x), UINT(second.y), source.p, c.plane,
                                        &second.source, c.format, c.mode);
    }
    if (list7.p) {
        const D3D12_TEXTURE_BARRIER to_copy{
            D3D12_BARRIER_SYNC_RESOLVE, D3D12_BARRIER_SYNC_COPY,
            D3D12_BARRIER_ACCESS_RESOLVE_DEST, D3D12_BARRIER_ACCESS_COPY_SOURCE,
            D3D12_BARRIER_LAYOUT_RESOLVE_DEST, D3D12_BARRIER_LAYOUT_COPY_SOURCE, target.p, all};
        enhanced_barrier(&to_copy, 1);
    } else {
        legacy[1].Transition.StateBefore = D3D12_RESOURCE_STATE_RESOLVE_DEST;
        legacy[1].Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
        list->ResourceBarrier(1, &legacy[1]);
    }
    for (UINT i = 0; i < planes; ++i) {
        D3D12_TEXTURE_COPY_LOCATION src{}, dst{};
        src.pResource = target.p;
        src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        src.SubresourceIndex = i;
        dst.pResource = readback.p;
        dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        dst.PlacedFootprint = placed[i];
        list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
    }
    out.close = list->Close();
    if (FAILED(out.close) && !run_invalid) return true;
    ID3D12CommandList* lists[] = {list.p};
    queue->ExecuteCommandLists(1, lists);
    CHECK(SUCCEEDED(queue->Signal(fence.p, 1)));
    Event event;
    CHECK(event.handle);
    CHECK(SUCCEEDED(fence->SetEventOnCompletion(1, event.handle)));
    // 30 s, not 5 s: under `ctest -j 6` other GPU tests share the device, and a fence can take longer
    // than this test's idle runtime to signal (TEST-514 flake). Sibling tests already wait 15-60 s.
    CHECK(WaitForSingleObject(event.handle, 30000) == WAIT_OBJECT_0);
    void* mapped = nullptr;
    const D3D12_RANGE range{0, SIZE_T(total)};
    CHECK(SUCCEEDED(readback->Map(0, &range, &mapped)) && mapped);
    const auto* base = static_cast<unsigned char*>(mapped);
    out.depth.resize(width * height * 4);
    for (UINT row = 0; row < height; ++row)
        std::memcpy(out.depth.data() + row * width * 4, base + placed[0].Offset + row * placed[0].Footprint.RowPitch,
                    width * 4);
    if (unorm24(c))
        for (std::size_t i = 3; i < out.depth.size(); i += 4) out.depth[i] = 0;  // the X8 byte is undefined
    out.stencil.clear();
    if (c.stencil) {
        out.stencil.resize(width * height);
        for (UINT row = 0; row < height; ++row)
            std::memcpy(out.stencil.data() + row * width, base + placed[1].Offset + row * placed[1].Footprint.RowPitch,
                        width);
    }
    const D3D12_RANGE none{};
    readback->Unmap(0, &none);
    return true;
}

// What the destination holds after `c` (or, resolved = false, after a refused one): the source's clear values inside
// the resolved area of the resolved plane, its own everywhere else.
Planes expected_planes(const Case& c, bool resolved = true) {
    Planes out;
    out.close = S_OK;
    const auto in = [](const D3D12_RECT& r, LONG x, LONG y) {
        return x >= r.left && x < r.right && y >= r.top && y < r.bottom;
    };
    const auto first = first_region(c);
    for (LONG y = 0; y < LONG(height); ++y) {
        for (LONG x = 0; x < LONG(width); ++x) {
            float depth = 0.0f;
            unsigned char stencil = 0x5a;
            if (resolved) {
                for (const auto* region : {&first, c.second}) {
                    if (!region) continue;
                    const LONG sx = x - region->x + region->source.left;
                    const LONG sy = y - region->y + region->source.top;
                    if (!in(region->source, sx, sy)) continue;
                    depth = 0.25f;
                    stencil = 0x11;
                    if (in(clear_a, sx, sy)) depth = 0.75f, stencil = 0x22;
                    if (in(clear_b, sx, sy)) depth = 1.0f, stencil = 0x33;
                }
            }
            if (c.plane == 1) depth = 0.0f;
            else stencil = 0x5a;
            std::uint32_t bits = 0;
            if (unorm24(c))
                bits = std::uint32_t(double(depth) * 16777215.0 + 0.5);
            else
                std::memcpy(&bits, &depth, 4);
            for (int i = 0; i < 4; ++i) out.depth.push_back(static_cast<unsigned char>(bits >> (8 * i)));
            if (c.stencil) out.stencil.push_back(stencil);
        }
    }
    return out;
}

bool same_planes(const Planes& a, const Planes& b, bool only_resolved, UINT plane) {
    if (only_resolved) return plane ? a.stencil == b.stencil : a.depth == b.depth;
    return a.depth == b.depth && a.stencil == b.stencil;
}

const char* mode_name(D3D12_RESOLVE_MODE mode) {
    switch (mode) {
    case D3D12_RESOLVE_MODE_MIN: return "MIN";
    case D3D12_RESOLVE_MODE_MAX: return "MAX";
    case D3D12_RESOLVE_MODE_AVERAGE: return "AVERAGE";
    case D3D12_RESOLVE_MODE_DECOMPRESS: return "DECOMPRESS";
    default: return "?";
    }
}
} // namespace

int main() {
    vktest::Device vk;
    vk.request_buffer_device_address = true;  // the edge draw's graphics pipeline needs it
    const auto device_status = vk.initialize();
    if (device_status == vktest::DeviceStatus::unavailable) return 77;   // no Vulkan GPU: ctest skips
    if (device_status != vktest::DeviceStatus::ready) return 1;
    const unsigned errors_before = vk.validation_errors();
    VkPhysicalDeviceDepthStencilResolveProperties resolve{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DEPTH_STENCIL_RESOLVE_PROPERTIES};
    VkPhysicalDeviceProperties2 properties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &resolve};
    vk.instance_fn<PFN_vkGetPhysicalDeviceProperties2>("vkGetPhysicalDeviceProperties2")(vk.physical, &properties);
    std::printf("Vulkan resolve modes: depth 0x%x, stencil 0x%x\n", unsigned(resolve.supportedDepthResolveModes),
                unsigned(resolve.supportedStencilResolveModes));

    constexpr auto d32 = DXGI_FORMAT_D32_FLOAT, d24 = DXGI_FORMAT_D24_UNORM_S8_UINT;
    constexpr auto min = D3D12_RESOLVE_MODE_MIN, max = D3D12_RESOLVE_MODE_MAX, average = D3D12_RESOLVE_MODE_AVERAGE;
    constexpr auto r32 = DXGI_FORMAT_R32_FLOAT, x8 = DXGI_FORMAT_R24_UNORM_X8_TYPELESS,
                   g8 = DXGI_FORMAT_X24_TYPELESS_G8_UINT;
    constexpr auto r32_typeless = DXGI_FORMAT_R32_TYPELESS, r24g8_typeless = DXGI_FORMAT_R24G8_TYPELESS;
    constexpr auto d32s8 = DXGI_FORMAT_D32_FLOAT_S8X24_UINT, r32x8 = DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS,
                   g8x24 = DXGI_FORMAT_X32_TYPELESS_G8X24_UINT, r32g8x24_typeless = DXGI_FORMAT_R32G8X24_TYPELESS;
    std::vector<Case> cases;  // resolved, and compared with the host
    for (const bool edge : {false, true}) {
        for (const bool rect : {false, true}) {
            for (const auto mode : {min, max}) {
                cases.push_back({"D32 depth", d32, false, 0, mode, rect, -1, -1, false, {}, {}, edge});
                cases.push_back({"D24S8 depth", d24, true, 0, mode, rect, -1, -1, false, {}, {}, edge});
                cases.push_back({"D24S8 stencil", d24, true, 1, mode, rect, -1, -1, false, {}, {}, edge});
                cases.push_back({"D24S8 stencil as X24_G8", g8, true, 1, mode, rect, -1, -1, false, d24, d24, edge});
                cases.push_back({"D32S8 depth", d32s8, true, 0, mode, rect, -1, -1, false, {}, {}, edge});
                cases.push_back({"D32S8 stencil", d32s8, true, 1, mode, rect, -1, -1, false, {}, {}, edge});
                cases.push_back({"D32S8 stencil as X32_G8X24", g8x24, true, 1, mode, rect, -1, -1, false, d32s8, d32s8,
                                 edge});
            }
            for (const auto mode : {min, max, average}) {
                cases.push_back({"D32 depth as R32_FLOAT", r32, false, 0, mode, rect, -1, -1, false, d32, d32, edge});
                cases.push_back({"D24S8 depth as R24_X8", x8, true, 0, mode, rect, -1, -1, false, d24, d24, edge});
                cases.push_back({"D32S8 depth as R32_X8X24", r32x8, true, 0, mode, rect, -1, -1, false, d32s8, d32s8,
                                 edge});
            }
        }
    }
    cases.push_back({"R32_TYPELESS depth as D32", d32, false, 0, min, false, -1, -1, false, r32_typeless});
    cases.push_back({"R24G8_TYPELESS stencil as D24S8", d24, true, 1, max, true, -1, -1, false, r24g8_typeless});
    cases.push_back({"R24G8_TYPELESS depth as R24_X8", x8, true, 0, average, true, -1, -1, false, r24g8_typeless, d24});
    cases.push_back({"R32G8X24_TYPELESS stencil as D32S8", d32s8, true, 1, min, true, -1, -1, false, r32g8x24_typeless});
    cases.push_back({"R32G8X24_TYPELESS depth as R32_X8X24", r32x8, true, 0, average, false, -1, -1, false,
                     r32g8x24_typeless, d32s8});
    cases.push_back({"D32 depth to upper left", d32, false, 0, min, true, 0, 0});
    cases.push_back({"D32 depth as R32_FLOAT to lower right", r32, false, 0, max, true, 3, 2, false, d32, d32});
    cases.push_back({"D32 depth as R32_FLOAT to lower left", r32, false, 0, average, true, 0, 2, false, d32, d32});
    // Consecutive submissions on the same queue request the same temporary format, aspect and extent.
    cases.push_back({"D24S8 depth to upper right", d24, true, 0, min, true, 3, 0});
    cases.push_back({"D24S8 depth to lower left", d24, true, 0, min, true, 0, 2});
    cases.push_back({"D24S8 depth as R24_X8 to lower left", x8, true, 0, max, true, 0, 2, false, d24, d24});
    cases.push_back({"D24S8 depth as R24_X8 to lower right", x8, true, 0, average, true, 3, 2, false, d24, d24});
    cases.push_back({"D24S8 stencil to lower right", d24, true, 1, max, true, 3, 2});
    cases.push_back({"D24S8 stencil as X24_G8 to lower left", g8, true, 1, min, true, 0, 2, false, d24, d24});
    // Two rectangles in one list. The destinations are disjoint and each ends at the bottom edge.
    constexpr ResolveRegion second{{5, 0, 8, 2}, 0, 4};
    cases.push_back({"D24S8 depth, two shifted rectangles", d24, true, 0, min, true, 3, 2, false, d24, d24, false, &second});
    // The second resolve covers columns 3 and 4 of rows 2 and 3 again, replacing depth 0.75 with 0.25 there.
    constexpr ResolveRegion overlapping{{5, 0, 8, 2}, 2, 2};
    cases.push_back({"D24S8 depth, overlapping shifted rectangles", d24, true, 0, min, true, 3, 2, false, d24, d24,
                     false, &overlapping, true});
    const Case refused[] = {
        {"D32 depth", d32, false, 0, average, false, -1, -1, true},
        {"D24S8 depth", d24, true, 0, average, true, -1, -1, true},
        {"D24S8 stencil", d24, true, 1, average, false, -1, -1, true},
        {"D24S8 stencil as X24_G8", g8, true, 1, average, true, -1, -1, true, d24, d24},
        {"D32 depth, whole to another offset", d32, false, 0, max, false, 1, 0, true},
        {"D32 depth, rectangle past the right edge", d32, false, 0, min, true, 4, 2},
        {"D24S8 stencil, rectangle past the bottom edge", d24, true, 1, max, true, 3, 3},
        {"D24S8 stencil subresource as R24_X8", x8, true, 1, min, false, -1, -1, false, d24, d24},
        {"D32S8 depth", d32s8, true, 0, average, false, -1, -1, true},
        {"D32S8 stencil", d32s8, true, 1, average, true, -1, -1, true},
        {"D32S8 stencil subresource as R32_X8X24", r32x8, true, 1, max, false, -1, -1, false, d32s8, d32s8},
        {"D32 depth", d32, false, 0, D3D12_RESOLVE_MODE_DECOMPRESS, false},
    };
    unsigned shifted = 0, matched = 0, failures = 0, refusals = 0, references = 0, lacking = 0, edges = 0, edge_skipped = 0;
    // The edge draw: the layer's container and index, the host's HLSL, one empty root signature for both.
    std::vector<std::uint8_t> root_blob, edge_container, edge_host_vs;
    rt::d3d12::VulkanDeviceInfo info{};
#if defined(LEONIDA_DEPTH_EDGE_VS_SPV)
    {
        const auto program = d3d12test::fssm_program({0x56530848, 0xbf810000});
        edge_container = d3d12test::shader_container(program);
        const auto library = std::filesystem::temp_directory_path() / "leonida_depth_resolve_test";
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
            "float4 main(uint id : SV_VertexID) : SV_Position {" "\n"
            "    const float2 corners[3] = {float2(-1, -1), float2(-1, 1), float2(1, 1)};" "\n"
            "    return float4(corners[id], 0.5, 1.0);" "\n"
            "}" "\n";
        ID3DBlob* code = nullptr;
        ID3DBlob* errors = nullptr;
        if (compile && SUCCEEDED(compile(host_vs, std::strlen(host_vs), nullptr, nullptr, nullptr, "main", "vs_5_0", 0,
                                         0, &code, &errors)) && code) {
            const auto* bytes = static_cast<const std::uint8_t*>(code->GetBufferPointer());
            edge_host_vs.assign(bytes, bytes + code->GetBufferSize());
        }
        if (errors) {
            std::fprintf(stderr, "%s" "\n", static_cast<const char*>(errors->GetBufferPointer()));
            errors->Release();
        }
        if (code) code->Release();
        if (root_blob.empty() || (edge_host_vs.empty() && !hostwine::host_is_wine())) {
            std::fprintf(stderr, "the edge draw's root signature or host shader failed" "\n");
            return 1;
        }
    }
#endif
    const bool edge_draws = !edge_container.empty();
    const EdgeDraw ours_edge{&root_blob, &edge_container}, host_edge{&root_blob, &edge_host_vs};
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
        Ref<ID3D12CommandQueue> ours_queue;
        if (FAILED(ours->CreateCommandQueue(&direct_queue, IID_PPV_ARGS(&ours_queue.p)))) return 1;

        std::vector<Planes> results(cases.size());
        for (std::size_t i = 0; i < cases.size(); ++i) {
            const Case& c = cases[i];
            const VkResolveModeFlags supported =
                c.plane ? resolve.supportedStencilResolveModes : resolve.supportedDepthResolveModes;
            const bool can = supported & (c.mode == min   ? VK_RESOLVE_MODE_MIN_BIT
                                          : c.mode == max ? VK_RESOLVE_MODE_MAX_BIT
                                                          : VK_RESOLVE_MODE_AVERAGE_BIT);
            if (c.edge && !edge_draws) {
                ++edge_skipped;
                continue;
            }
            if (!resolve_and_read(ours.p, ours_queue.p, c, results[i], true, &ours_edge)) return 1;
            // A mode the device lacks is refused like the cases below. An edge case has no CPU model of its
            // coverage: only its Close is checked here, and its planes against the host below.
            const Planes expected = expected_planes(c, can);
            if (results[i].close != (can ? S_OK : E_INVALIDARG) ||
                (!c.edge && !same_planes(results[i], expected, false, c.plane))) {
                std::fprintf(stderr, "%s %s rect %d: Close 0x%08lx, destination %s the clears\n", c.name,
                             mode_name(c.mode), int(c.rect), results[i].close,
                             same_planes(results[i], expected, false, c.plane) ? "matches" : "differs from");
                ++failures;
            } else if (can && (c.dst_x >= 0 || c.dst_y >= 0)) {
                ++shifted;
            }
            if (!can) {
                results[i] = {};  // not compared with the host
                ++lacking;
            }
        }
        // The edge draw makes the samples differ: for each edge case resolved by MIN, the MAX of the same case must
        // differ from it somewhere, or the draw did not reach the resolve.
        for (std::size_t i = 0; i < cases.size(); ++i) {
            const Case& c = cases[i];
            if (!c.edge || c.mode != min || results[i].depth.empty()) continue;
            for (std::size_t j = 0; j < cases.size(); ++j) {
                const Case& o = cases[j];
                if (!o.edge || o.mode != max || std::strcmp(o.name, c.name) != 0 || o.rect != c.rect ||
                    results[j].depth.empty())
                    continue;
                ++edges;
                if (same_planes(results[i], results[j], true, c.plane)) {
                    std::fprintf(stderr, "edge %s rect %d: MIN and MAX agree everywhere; the samples did not differ" "\n",
                                 c.name, int(c.rect));
                    ++failures;
                }
            }
        }
        for (const Case& c : refused) {
            Planes p;
            if (!resolve_and_read(ours.p, ours_queue.p, c, p, true)) return 1;
            if (p.close != E_INVALIDARG || !same_planes(p, expected_planes(c, false), false, c.plane)) {
                std::fprintf(stderr, "refused %s %s rect %d: Close 0x%08lx, destination %s\n", c.name,
                             mode_name(c.mode), int(c.rect), p.close,
                             same_planes(p, expected_planes(c, false), false, c.plane) ? "kept" : "changed");
                ++failures;
            }
            ++refusals;
        }

        Ref<IDXGIFactory4> factory;
        if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory.p)))) {
            std::fprintf(stderr, "no DXGI factory\n");
            return 1;
        }
        for (const bool warp : {false, true}) {
            if (hostwine::host_is_wine()) break;  // checked against the clears above
            Ref<IDXGIAdapter> adapter;
            Ref<ID3D12Device> host;
            std::string adapter_name;
            if (!hostadapter::reference(factory.p, vk, warp, &adapter.p, adapter_name)) continue;
            if (FAILED(D3D12CreateDevice(adapter.p, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&host.p)))) continue;
            Ref<ID3D12CommandQueue> host_queue;
            if (FAILED(host->CreateCommandQueue(&direct_queue, IID_PPV_ARGS(&host_queue.p)))) return 1;
            D3D12_FEATURE_DATA_D3D12_OPTIONS12 options{};
            const bool enhanced = SUCCEEDED(host->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS12, &options,
                                                                       sizeof options)) &&
                                  options.EnhancedBarriersSupported;
            const char* name = adapter_name.c_str();
            unsigned same = 0, compared = 0, refused_alike = 0, quirks = 0;
            std::vector<const Case*> quirks_seen;  // cases without the edge where this host differs from the clears
            const auto quirky = [&](const Case& e) {
                for (const Case* q : quirks_seen)
                    if (std::strcmp(q->name, e.name) == 0 && q->mode == e.mode && q->rect == e.rect) return true;
                return false;
            };
            for (std::size_t i = 0; i < cases.size(); ++i) {
                const Case& c = cases[i];
                // See the top of the file; D32S8 by its own format is kept off WARP for the same reason.
                if (results[i].depth.empty() || (warp && (c.format == d32 || c.format == d32s8)) || (c.enhanced && !enhanced)) continue;
                Planes expected;
                if (!resolve_and_read(host.p, host_queue.p, c, expected, false, &host_edge)) return 1;
                ++compared;
                // The layer's result was checked against the clears above. A host that differs from the clears is a
                // host quirk, not a layer bug: report, don't fail. Which D24S8 resolves is the driver's: with stencil
                // MIN/MAX on both, an RTX 3070 (NVIDIA 610.74) adds "stencil as X24_G8" and "depth as R24_X8" AVERAGE,
                // a Ryzen 7 9700X iGPU (AMD 32.0.21042.62) depth MIN/MAX; whole subresources and rectangles alike.
                // Edge cases have no clear model: on WARP, or where the host had a quirk on the same case without the
                // edge, a difference from the layer is only reported.
                if (c.edge && (warp || quirky(c))) {
                    if (FAILED(expected.close) || !same_planes(expected, results[i], true, c.plane))
                        std::printf("edge %s %s rect %d: %s differs from the layer (information)\n", c.name,
                                    mode_name(c.mode), int(c.rect), name);
                    else
                        ++same;
                    continue;
                }
                if (!c.edge && SUCCEEDED(expected.close) && !same_planes(expected, expected_planes(c), warp, c.plane)) {
                    quirks_seen.push_back(&c);
                    std::printf("%s %s rect %d against %s: host quirk: the host differs from the clears; the layer "
                                "matches them\n", c.name, mode_name(c.mode), int(c.rect), name);
                    ++quirks;
                    continue;
                }
                if (FAILED(expected.close) || !same_planes(expected, results[i], warp, c.plane)) {
                    std::fprintf(stderr, "%s %s rect %d against %s: host Close 0x%08lx, host %s the clears, "
                                 "layer and host %s\n", c.name, mode_name(c.mode), int(c.rect), name, expected.close,
                                 same_planes(expected, expected_planes(c), warp, c.plane) ? "matches" : "differs from",
                                 same_planes(expected, results[i], warp, c.plane) ? "agree" : "differ");
                    ++failures;
                    continue;
                }
                ++same;
            }
            for (const Case& c : refused) {
                if (!c.host_refuses) continue;
                Planes expected;
                if (!resolve_and_read(host.p, host_queue.p, c, expected, false)) return 1;
                if (expected.close != E_INVALIDARG) {
                    std::fprintf(stderr, "refused %s %s rect %d: %s's Close 0x%08lx\n", c.name, mode_name(c.mode),
                                 int(c.rect), name, expected.close);
                    ++failures;
                    continue;
                }
                ++refused_alike;
            }
            std::printf("depth/stencil resolves against %s: %u of %u match, %u host quirks, %u refusals alike\n", name,
                        same, compared, quirks, refused_alike);
            matched += same;
            ++references;
        }
        return 0;
    }; // Destroy the D3D12 objects before checking the Vulkan validation counter.
    const int run_failed = run();
    const auto validation_errors = vk.validation_errors() - errors_before;
    if (validation_errors)
        std::fprintf(stderr, "d3d12_depth_resolve_test: %u new Vulkan validation error(s)\n", validation_errors);
    const bool need_host = !hostwine::host_is_wine();
    if (need_host && !references) std::fprintf(stderr, "no D3D12 reference device (hardware adapter or WARP)\n");
    if (run_failed || failures || validation_errors || (need_host && (!references || !matched))) return 1;
    std::printf("depth/stencil resolves: %u shifted cases matched the clears, %u host comparisons matched, %u refusals kept the destination, %u modes the "
                "device lacks were refused; %u edge MIN/MAX pairs differ%s\n", shifted, matched, refusals, lacking, edges,
                edge_skipped ? " (edge cases skipped: no glslangValidator at build time)" : "");
    return 0;
}
