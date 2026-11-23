// ID3D12GraphicsCommandList::DiscardResource on the Vulkan-backed device:
//   - Close() parity with Windows D3D12 (the layer's own adapter and WARP, skipped under Wine) across DIRECT,
//     COMPUTE, COPY and BUNDLE lists, whole/range/zero-count/partial-rect regions, and 2D RT, 2D UAV, depth-stencil,
//     3D texture and buffer resources;
//   - layer-only invalid arguments and out-of-range subresource replay rejections;
//   - GPU execution and readback with LEONIDA_VK_VALIDATION=1: subresource isolation on a 2-mip x 2-layer RT,
//     partial-rect preservation, depth/stencil plane and whole discards, 3D texture and buffer discards, and compute-
//     queue discards on both GENERAL and COLOR_ATTACHMENT_OPTIMAL-tracked textures (0 Vulkan validation errors).
// No game data or shaders.
#include "vk_test_device.h"
#include "host_adapter.h"
#include "host_wine.h"
#include "../shims/d3d12_com.h"
#include <dxgi1_4.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

namespace {
template<class T>
struct Ref {
    T* p = nullptr;
    ~Ref() { reset(); }
    T* operator->() const { return p; }
    T** put() {
        reset();
        return &p;
    }
    void reset() {
        if (p) p->Release();
        p = nullptr;
    }
};

D3D12_RESOURCE_DESC tex2d(UINT64 width, UINT height, UINT16 layers, UINT16 mips, DXGI_FORMAT format,
                          D3D12_RESOURCE_FLAGS flags) {
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    desc.Width = width;
    desc.Height = height;
    desc.DepthOrArraySize = layers;
    desc.MipLevels = mips;
    desc.Format = format;
    desc.SampleDesc.Count = 1;
    desc.Flags = flags;
    return desc;
}

D3D12_RESOURCE_DESC tex3d(UINT64 width, UINT height, UINT16 depth, UINT16 mips, DXGI_FORMAT format,
                          D3D12_RESOURCE_FLAGS flags = D3D12_RESOURCE_FLAG_NONE) {
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE3D;
    desc.Width = width;
    desc.Height = height;
    desc.DepthOrArraySize = depth;
    desc.MipLevels = mips;
    desc.Format = format;
    desc.SampleDesc.Count = 1;
    desc.Flags = flags;
    return desc;
}

D3D12_RESOURCE_DESC buffer_desc(UINT64 bytes, D3D12_RESOURCE_FLAGS flags = D3D12_RESOURCE_FLAG_NONE) {
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width = bytes;
    desc.Height = 1;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    desc.Flags = flags;
    return desc;
}

D3D12_RESOURCE_BARRIER transition(ID3D12Resource* resource, D3D12_RESOURCE_STATES before, D3D12_RESOURCE_STATES after,
                                  UINT subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES) {
    D3D12_RESOURCE_BARRIER b{};
    b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    b.Transition = {resource, subresource, before, after};
    return b;
}

HRESULT make_committed(ID3D12Device* device, D3D12_HEAP_TYPE heap_type, const D3D12_RESOURCE_DESC& desc,
                       D3D12_RESOURCE_STATES initial, ID3D12Resource** out) {
    D3D12_HEAP_PROPERTIES heap{};
    heap.Type = heap_type;
    return device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, initial, nullptr, IID_PPV_ARGS(out));
}

bool wait_queue(ID3D12CommandQueue* queue, ID3D12Fence* fence, UINT64& value) {
    ++value;
    if (FAILED(queue->Signal(fence, value))) return false;
    return SUCCEEDED(fence->SetEventOnCompletion(value, nullptr));
}

struct ParityEnv {
    ID3D12Device* device = nullptr;
    Ref<ID3D12Resource> rt, uav, ds, vol, buf;
};

bool setup_parity(ParityEnv& e) {
    constexpr auto rgba8 = DXGI_FORMAT_R8G8B8A8_UNORM;
    return SUCCEEDED(make_committed(e.device, D3D12_HEAP_TYPE_DEFAULT,
                                    tex2d(16, 16, 2, 2, rgba8, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET),
                                    D3D12_RESOURCE_STATE_RENDER_TARGET, e.rt.put())) &&
           SUCCEEDED(make_committed(e.device, D3D12_HEAP_TYPE_DEFAULT,
                                    tex2d(16, 16, 1, 1, rgba8, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS),
                                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS, e.uav.put())) &&
           SUCCEEDED(make_committed(e.device, D3D12_HEAP_TYPE_DEFAULT,
                                    tex2d(16, 16, 1, 1, DXGI_FORMAT_D24_UNORM_S8_UINT,
                                          D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL),
                                    D3D12_RESOURCE_STATE_DEPTH_WRITE, e.ds.put())) &&
           SUCCEEDED(make_committed(e.device, D3D12_HEAP_TYPE_DEFAULT, tex3d(8, 8, 4, 2, rgba8),
                                    D3D12_RESOURCE_STATE_COMMON, e.vol.put())) &&
           SUCCEEDED(make_committed(e.device, D3D12_HEAP_TYPE_DEFAULT, buffer_desc(4096),
                                    D3D12_RESOURCE_STATE_COMMON, e.buf.put()));
}

HRESULT close_result(ID3D12Device* device, D3D12_COMMAND_LIST_TYPE type,
                     const std::function<void(ID3D12GraphicsCommandList*)>& rec) {
    Ref<ID3D12CommandAllocator> allocator;
    Ref<ID3D12GraphicsCommandList> list;
    if (FAILED(device->CreateCommandAllocator(type, IID_PPV_ARGS(allocator.put())))) return 0x7fff0001;
    if (FAILED(device->CreateCommandList(0, type, allocator.p, nullptr, IID_PPV_ARGS(list.put())))) return 0x7fff0002;
    rec(list.p);
    return list->Close();
}

std::string parity_cases(ID3D12Device* device) {
    ParityEnv e{device};
    if (!setup_parity(e)) return "setup failed\n";

    const std::pair<const char*, D3D12_COMMAND_LIST_TYPE> list_types[] = {
        {"direct", D3D12_COMMAND_LIST_TYPE_DIRECT},
        {"compute", D3D12_COMMAND_LIST_TYPE_COMPUTE},
        {"copy", D3D12_COMMAND_LIST_TYPE_COPY},
        {"bundle", D3D12_COMMAND_LIST_TYPE_BUNDLE},
    };
    const std::pair<const char*, ID3D12Resource*> resources[] = {
        {"rt2d", e.rt.p},
        {"uav2d", e.uav.p},
        {"ds2d", e.ds.p},
        {"tex3d", e.vol.p},
        {"buf", e.buf.p},
    };
    const D3D12_RECT rect{2, 2, 6, 6};
    const D3D12_DISCARD_REGION range_one{0, nullptr, 0, 1};
    const D3D12_DISCARD_REGION range_zero{0, nullptr, 0, 0};
    const D3D12_DISCARD_REGION partial_rect{1, &rect, 0, 1};
    const std::pair<const char*, const D3D12_DISCARD_REGION*> regions[] = {
        {"whole", nullptr},
        {"sub0", &range_one},
        {"zero_count", &range_zero},
        {"rect", &partial_rect},
    };

    std::string out;
    for (const auto& [lt_name, lt] : list_types) {
        for (const auto& [res_name, res] : resources) {
            for (const auto& [reg_name, reg] : regions) {
                const HRESULT hr = close_result(device, lt, [&](ID3D12GraphicsCommandList* l) {
                    l->DiscardResource(res, reg);
                });
                char line[160];
                // Success or failure only: the layer reports E_INVALIDARG where the host sometimes gives E_FAIL.
                std::snprintf(line, sizeof line, "close %s %s %s: %s\n", lt_name, res_name, reg_name,
                              SUCCEEDED(hr) ? "ok" : "fails");
                out += line;
            }
        }
    }
    return out;
}

std::vector<std::string> split_lines(const std::string& s) {
    std::vector<std::string> out;
    size_t at = 0;
    while (at < s.size()) {
        const size_t nl = s.find('\n', at);
        const size_t end = nl == std::string::npos ? s.size() : nl;
        out.push_back(s.substr(at, end - at));
        at = nl == std::string::npos ? s.size() : nl + 1;
    }
    return out;
}

bool check_expected_parity_expectations(const std::string& mine) {
    unsigned bad = 0;
    for (const std::string& line : split_lines(mine)) {
        const bool is_copy_or_bundle = line.rfind("close copy ", 0) == 0 || line.rfind("close bundle ", 0) == 0;
        const bool is_direct_or_compute = line.rfind("close direct ", 0) == 0 || line.rfind("close compute ", 0) == 0;
        const bool is_tex3d_rect = line.find(" tex3d rect:") != std::string::npos;
        const bool expect_fail = is_copy_or_bundle || is_tex3d_rect;
        if (expect_fail && line.find(": fails") == std::string::npos) {
            std::fprintf(stderr, "expected failure: %s\n", line.c_str());
            ++bad;
        }
        if (is_direct_or_compute && !is_tex3d_rect && line.find(": ok") == std::string::npos) {
            std::fprintf(stderr, "expected ok: %s\n", line.c_str());
            ++bad;
        }
    }
    return bad == 0;
}

// Layer-only invalid arguments and out-of-range replay rejections (never sent to host D3D12).
bool test_invalid_arguments(ID3D12Device* device, ID3D12CommandQueue* queue) {
    unsigned failures = 0;
    Ref<ID3D12Resource> rt, buf;
    if (FAILED(make_committed(device, D3D12_HEAP_TYPE_DEFAULT,
                              tex2d(16, 16, 2, 2, DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET),
                              D3D12_RESOURCE_STATE_RENDER_TARGET, rt.put())) ||
        FAILED(make_committed(device, D3D12_HEAP_TYPE_DEFAULT, buffer_desc(4096), D3D12_RESOURCE_STATE_COMMON,
                              buf.put()))) {
        std::fprintf(stderr, "invalid args: resource creation failed\n");
        return false;
    }

    auto expect_close_invalid = [&](const char* name, const std::function<void(ID3D12GraphicsCommandList*)>& rec) {
        const HRESULT hr = close_result(device, D3D12_COMMAND_LIST_TYPE_DIRECT, rec);
        if (hr != E_INVALIDARG) {
            std::fprintf(stderr, "invalid args [%s]: Close() returned 0x%08lx, want E_INVALIDARG\n", name,
                         static_cast<unsigned long>(hr));
            ++failures;
        }
    };

    expect_close_invalid("null resource", [](ID3D12GraphicsCommandList* l) { l->DiscardResource(nullptr, nullptr); });
    expect_close_invalid("NumRects=1, pRects=nullptr", [&](ID3D12GraphicsCommandList* l) {
        const D3D12_DISCARD_REGION bad{1, nullptr, 0, 1};
        l->DiscardResource(rt.p, &bad);
    });
    expect_close_invalid("FirstSubresource=1, NumSubresources=UINT_MAX", [&](ID3D12GraphicsCommandList* l) {
        const D3D12_DISCARD_REGION bad{0, nullptr, 1, UINT_MAX};
        l->DiscardResource(rt.p, &bad);
    });

    auto expect_replay_invalid_subresource = [&](const char* name, ID3D12Resource* res,
                                                 const D3D12_DISCARD_REGION& region) {
        Ref<ID3D12CommandAllocator> alloc;
        Ref<ID3D12GraphicsCommandList> list;
        if (FAILED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(alloc.put()))) ||
            FAILED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, alloc.p, nullptr,
                                             IID_PPV_ARGS(list.put())))) {
            std::fprintf(stderr, "invalid args [%s]: list creation failed\n", name);
            ++failures;
            return;
        }
        list->DiscardResource(res, &region);
        const HRESULT hr = list->Close();
        if (hr != S_OK) {
            std::fprintf(stderr, "invalid args [%s]: Close() returned 0x%08lx, want S_OK\n", name,
                         static_cast<unsigned long>(hr));
            ++failures;
            return;
        }
        ID3D12CommandList* lists[] = {list.p};
        queue->ExecuteCommandLists(1, lists);
        const auto res_code = rt::d3d12::last_execute_result(queue);
        if (res_code != leonida::ReplayResult::invalid_subresource) {
            std::fprintf(stderr, "invalid args [%s]: replay returned %d, want invalid_subresource (%d)\n", name,
                         int(res_code), int(leonida::ReplayResult::invalid_subresource));
            ++failures;
        }
    };

    expect_replay_invalid_subresource("rt subresource 4 out of 4", rt.p, {0, nullptr, 4, 1});
    expect_replay_invalid_subresource("rt First=0 Num=UINT_MAX", rt.p, {0, nullptr, 0, UINT_MAX});
    expect_replay_invalid_subresource("buffer subresource 1", buf.p, {0, nullptr, 1, 1});

    if (!failures) std::printf("invalid arguments & replay rejections: ok\n");
    return failures == 0;
}

bool test_rt_subresources_and_partial_rect(ID3D12Device* device, ID3D12CommandQueue* queue, ID3D12Fence* fence,
                                           UINT64& fence_value) {
    // 1) 2D RT: 16x16, 2 mips x 2 layers (4 subresources).
    const D3D12_RESOURCE_DESC rt_desc =
        tex2d(16, 16, 2, 2, DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET);
    Ref<ID3D12Resource> rt;
    if (FAILED(make_committed(device, D3D12_HEAP_TYPE_DEFAULT, rt_desc, D3D12_RESOURCE_STATE_RENDER_TARGET, rt.put())))
        return false;

    D3D12_DESCRIPTOR_HEAP_DESC heap_desc{};
    heap_desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
    heap_desc.NumDescriptors = 5;
    Ref<ID3D12DescriptorHeap> rtv_heap;
    if (FAILED(device->CreateDescriptorHeap(&heap_desc, IID_PPV_ARGS(rtv_heap.put())))) return false;
    const UINT rtv_stride = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
    const D3D12_CPU_DESCRIPTOR_HANDLE rtv_base = rtv_heap->GetCPUDescriptorHandleForHeapStart();
    auto rtv_at = [&](UINT idx) {
        return D3D12_CPU_DESCRIPTOR_HANDLE{rtv_base.ptr + SIZE_T(idx) * rtv_stride};
    };

    for (UINT layer = 0; layer < 2; ++layer) {
        for (UINT mip = 0; mip < 2; ++mip) {
            const UINT sub = mip + layer * 2;
            D3D12_RENDER_TARGET_VIEW_DESC view{};
            view.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
            view.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2DARRAY;
            view.Texture2DArray.MipSlice = mip;
            view.Texture2DArray.FirstArraySlice = layer;
            view.Texture2DArray.ArraySize = 1;
            device->CreateRenderTargetView(rt.p, &view, rtv_at(sub));
        }
    }

    D3D12_PLACED_SUBRESOURCE_FOOTPRINT layouts[4]{};
    UINT rows[4]{};
    UINT64 row_bytes[4]{};
    UINT64 total = 0;
    device->GetCopyableFootprints(&rt_desc, 0, 4, 0, layouts, rows, row_bytes, &total);
    Ref<ID3D12Resource> readback;
    if (FAILED(make_committed(device, D3D12_HEAP_TYPE_READBACK, buffer_desc(total), D3D12_RESOURCE_STATE_COPY_DEST,
                              readback.put())))
        return false;

    const float colors[4][4] = {
        {1.0f, 0.0f, 0.0f, 1.0f},  // sub 0: red
        {0.0f, 1.0f, 0.0f, 1.0f},  // sub 1: green (discarded)
        {0.0f, 0.0f, 1.0f, 1.0f},  // sub 2: blue
        {1.0f, 1.0f, 0.0f, 1.0f},  // sub 3: yellow
    };
    const std::array<std::uint8_t, 4> expected[4] = {
        {255, 0, 0, 255},
        {0, 255, 0, 255},
        {0, 0, 255, 255},
        {255, 255, 0, 255},
    };
    const float magenta[4] = {1.0f, 0.0f, 1.0f, 1.0f};
    const std::array<std::uint8_t, 4> expected_magenta = {255, 0, 255, 255};

    Ref<ID3D12CommandAllocator> alloc;
    Ref<ID3D12GraphicsCommandList> list;
    if (FAILED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(alloc.put()))) ||
        FAILED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, alloc.p, nullptr,
                                         IID_PPV_ARGS(list.put()))))
        return false;

    // Clear all 4 subresources, discard subresource 1 only, copy subresources 0, 2, 3 to readback.
    for (UINT s = 0; s < 4; ++s) list->ClearRenderTargetView(rtv_at(s), colors[s], 0, nullptr);
    const D3D12_DISCARD_REGION discard_sub1{0, nullptr, 1, 1};
    list->DiscardResource(rt.p, &discard_sub1);
    const auto to_src = transition(rt.p, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_COPY_SOURCE);
    list->ResourceBarrier(1, &to_src);
    for (UINT s : {0u, 2u, 3u}) {
        D3D12_TEXTURE_COPY_LOCATION dst{readback.p, D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, {}};
        dst.PlacedFootprint = layouts[s];
        D3D12_TEXTURE_COPY_LOCATION src{rt.p, D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, {}};
        src.SubresourceIndex = s;
        list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
    }
    if (FAILED(list->Close())) return false;
    ID3D12CommandList* lists[] = {list.p};
    queue->ExecuteCommandLists(1, lists);
    if (rt::d3d12::last_execute_result(queue) != leonida::ReplayResult::success ||
        !wait_queue(queue, fence, fence_value)) {
        std::fprintf(stderr, "rt subresources: phase 1 execute failed\n");
        return false;
    }

    auto verify_sub = [&](UINT s, const std::array<std::uint8_t, 4>& want, const char* label) -> bool {
        void* mapped = nullptr;
        if (FAILED(readback->Map(0, nullptr, &mapped))) return false;
        const auto* base = static_cast<const std::uint8_t*>(mapped);
        bool ok = true;
        for (UINT y = 0; y < rows[s] && ok; ++y) {
            const auto* row = base + layouts[s].Offset + UINT64(y) * layouts[s].Footprint.RowPitch;
            for (UINT x = 0; x < layouts[s].Footprint.Width && ok; ++x) {
                if (std::memcmp(row + x * 4, want.data(), 4) != 0) {
                    std::fprintf(stderr, "%s sub %u (%u,%u): got (%u,%u,%u,%u) want (%u,%u,%u,%u)\n", label, s, x, y,
                                 row[x * 4], row[x * 4 + 1], row[x * 4 + 2], row[x * 4 + 3], want[0], want[1], want[2],
                                 want[3]);
                    ok = false;
                }
            }
        }
        readback->Unmap(0, nullptr);
        return ok;
    };

    for (UINT s : {0u, 2u, 3u})
        if (!verify_sub(s, expected[s], "rt subresource preserved")) return false;

    // Re-clear discarded subresource 1 to magenta and verify.
    if (FAILED(alloc->Reset()) || FAILED(list->Reset(alloc.p, nullptr))) return false;
    const auto to_rt = transition(rt.p, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_RENDER_TARGET);
    list->ResourceBarrier(1, &to_rt);
    list->ClearRenderTargetView(rtv_at(1), magenta, 0, nullptr);
    list->ResourceBarrier(1, &to_src);
    {
        D3D12_TEXTURE_COPY_LOCATION dst{readback.p, D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, {}};
        dst.PlacedFootprint = layouts[1];
        D3D12_TEXTURE_COPY_LOCATION src{rt.p, D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, {}};
        src.SubresourceIndex = 1;
        list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
    }
    if (FAILED(list->Close())) return false;
    queue->ExecuteCommandLists(1, lists);
    if (rt::d3d12::last_execute_result(queue) != leonida::ReplayResult::success ||
        !wait_queue(queue, fence, fence_value)) {
        std::fprintf(stderr, "rt subresources: phase 2 execute failed\n");
        return false;
    }
    if (!verify_sub(1, expected_magenta, "rt subresource re-cleared")) return false;
    std::printf("rt 2-mip x 2-layer subresource discard + re-clear: ok\n");

    // 2) Partial rect discard on 16x16 2D RT: pixels outside [4,12) x [4,12) must remain preserved.
    const D3D12_RESOURCE_DESC rect_rt_desc =
        tex2d(16, 16, 1, 1, DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET);
    Ref<ID3D12Resource> rect_rt;
    if (FAILED(make_committed(device, D3D12_HEAP_TYPE_DEFAULT, rect_rt_desc, D3D12_RESOURCE_STATE_RENDER_TARGET,
                              rect_rt.put())))
        return false;
    device->CreateRenderTargetView(rect_rt.p, nullptr, rtv_at(4));
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT rect_fp{};
    UINT rect_rows = 0;
    UINT64 rect_row_bytes = 0, rect_total = 0;
    device->GetCopyableFootprints(&rect_rt_desc, 0, 1, 0, &rect_fp, &rect_rows, &rect_row_bytes, &rect_total);
    Ref<ID3D12Resource> rect_rb;
    if (FAILED(make_committed(device, D3D12_HEAP_TYPE_READBACK, buffer_desc(rect_total),
                              D3D12_RESOURCE_STATE_COPY_DEST, rect_rb.put())))
        return false;

    const float teal[4] = {0.0f, 1.0f, 1.0f, 1.0f};
    const std::array<std::uint8_t, 4> expected_teal = {0, 255, 255, 255};
    const D3D12_RECT inner{4, 4, 12, 12};
    const D3D12_DISCARD_REGION inner_region{1, &inner, 0, 1};
    if (FAILED(alloc->Reset()) || FAILED(list->Reset(alloc.p, nullptr))) return false;
    list->ClearRenderTargetView(rtv_at(4), teal, 0, nullptr);
    list->DiscardResource(rect_rt.p, &inner_region);
    const auto rect_to_src =
        transition(rect_rt.p, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_COPY_SOURCE);
    list->ResourceBarrier(1, &rect_to_src);
    {
        D3D12_TEXTURE_COPY_LOCATION dst{rect_rb.p, D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, {}};
        dst.PlacedFootprint = rect_fp;
        D3D12_TEXTURE_COPY_LOCATION src{rect_rt.p, D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, {}};
        src.SubresourceIndex = 0;
        list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
    }
    if (FAILED(list->Close())) return false;
    queue->ExecuteCommandLists(1, lists);
    if (rt::d3d12::last_execute_result(queue) != leonida::ReplayResult::success ||
        !wait_queue(queue, fence, fence_value)) {
        std::fprintf(stderr, "partial rect discard: execute failed\n");
        return false;
    }

    void* mapped = nullptr;
    if (FAILED(rect_rb->Map(0, nullptr, &mapped))) return false;
    const auto* base = static_cast<const std::uint8_t*>(mapped);
    bool outside_ok = true;
    for (UINT y = 0; y < 16 && outside_ok; ++y) {
        const auto* row = base + rect_fp.Offset + UINT64(y) * rect_fp.Footprint.RowPitch;
        for (UINT x = 0; x < 16 && outside_ok; ++x) {
            if (x >= 4 && x < 12 && y >= 4 && y < 12) continue;
            if (std::memcmp(row + x * 4, expected_teal.data(), 4) != 0) {
                std::fprintf(stderr, "partial rect (%u,%u): got (%u,%u,%u,%u) want (0,255,255,255)\n", x, y,
                             row[x * 4], row[x * 4 + 1], row[x * 4 + 2], row[x * 4 + 3]);
                outside_ok = false;
            }
        }
    }
    rect_rb->Unmap(0, nullptr);
    if (!outside_ok) return false;
    std::printf("rt partial-rect discard preserves outside pixels: ok\n");
    return true;
}

bool test_depth_stencil_discard(ID3D12Device* device, ID3D12CommandQueue* queue, ID3D12Fence* fence,
                                UINT64& fence_value) {
    const D3D12_RESOURCE_DESC ds_desc =
        tex2d(16, 16, 1, 1, DXGI_FORMAT_D32_FLOAT_S8X24_UINT, D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL);
    Ref<ID3D12Resource> ds;
    if (FAILED(make_committed(device, D3D12_HEAP_TYPE_DEFAULT, ds_desc, D3D12_RESOURCE_STATE_DEPTH_WRITE, ds.put())))
        return false;

    D3D12_DESCRIPTOR_HEAP_DESC heap_desc{};
    heap_desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_DSV;
    heap_desc.NumDescriptors = 1;
    Ref<ID3D12DescriptorHeap> dsv_heap;
    if (FAILED(device->CreateDescriptorHeap(&heap_desc, IID_PPV_ARGS(dsv_heap.put())))) return false;
    const D3D12_CPU_DESCRIPTOR_HANDLE dsv = dsv_heap->GetCPUDescriptorHandleForHeapStart();
    device->CreateDepthStencilView(ds.p, nullptr, dsv);

    Ref<ID3D12CommandAllocator> alloc;
    Ref<ID3D12GraphicsCommandList> list;
    if (FAILED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(alloc.put()))) ||
        FAILED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, alloc.p, nullptr,
                                         IID_PPV_ARGS(list.put()))))
        return false;

    const D3D12_DISCARD_REGION depth_plane{0, nullptr, 0, 1};
    const D3D12_DISCARD_REGION stencil_plane{0, nullptr, 1, 1};
    list->ClearDepthStencilView(dsv, D3D12_CLEAR_FLAG_DEPTH | D3D12_CLEAR_FLAG_STENCIL, 0.5f, 0x2a, 0, nullptr);
    list->DiscardResource(ds.p, &depth_plane);
    list->DiscardResource(ds.p, &stencil_plane);
    list->DiscardResource(ds.p, nullptr);
    list->ClearDepthStencilView(dsv, D3D12_CLEAR_FLAG_DEPTH | D3D12_CLEAR_FLAG_STENCIL, 1.0f, 0x00, 0, nullptr);
    if (FAILED(list->Close())) return false;

    ID3D12CommandList* lists[] = {list.p};
    queue->ExecuteCommandLists(1, lists);
    if (rt::d3d12::last_execute_result(queue) != leonida::ReplayResult::success ||
        !wait_queue(queue, fence, fence_value)) {
        std::fprintf(stderr, "depth-stencil discard: execute failed\n");
        return false;
    }
    std::printf("depth-stencil plane & whole discard + re-clear: ok\n");
    return true;
}

bool test_3d_and_buffer_discard(ID3D12Device* device, ID3D12CommandQueue* queue, ID3D12Fence* fence,
                                UINT64& fence_value) {
    // 3D texture: 8x8x4, 2 mips.
    const D3D12_RESOURCE_DESC vol_desc = tex3d(8, 8, 4, 2, DXGI_FORMAT_R8G8B8A8_UNORM);
    Ref<ID3D12Resource> vol;
    if (FAILED(make_committed(device, D3D12_HEAP_TYPE_DEFAULT, vol_desc, D3D12_RESOURCE_STATE_COPY_DEST, vol.put())))
        return false;
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT fp[2]{};
    UINT rows[2]{};
    UINT64 row_bytes[2]{};
    UINT64 total = 0;
    device->GetCopyableFootprints(&vol_desc, 0, 2, 0, fp, rows, row_bytes, &total);

    Ref<ID3D12Resource> upload, readback;
    if (FAILED(make_committed(device, D3D12_HEAP_TYPE_UPLOAD, buffer_desc(total), D3D12_RESOURCE_STATE_GENERIC_READ,
                              upload.put())) ||
        FAILED(make_committed(device, D3D12_HEAP_TYPE_READBACK, buffer_desc(total), D3D12_RESOURCE_STATE_COPY_DEST,
                              readback.put())))
        return false;

    auto fill_upload = [&](std::uint8_t seed) -> bool {
        void* p = nullptr;
        if (FAILED(upload->Map(0, nullptr, &p))) return false;
        auto* bytes = static_cast<std::uint8_t*>(p);
        for (UINT s = 0; s < 2; ++s) {
            for (UINT z = 0; z < fp[s].Footprint.Depth; ++z) {
                for (UINT y = 0; y < rows[s]; ++y) {
                    auto* row = bytes + fp[s].Offset + (UINT64(z) * rows[s] + y) * fp[s].Footprint.RowPitch;
                    for (UINT64 x = 0; x < row_bytes[s]; ++x)
                        row[x] = static_cast<std::uint8_t>(seed + s * 53 + z * 19 + y * 7 + x * 3);
                }
            }
        }
        upload->Unmap(0, nullptr);
        return true;
    };
    auto check_readback_sub = [&](UINT s, std::uint8_t seed) -> bool {
        void* p = nullptr;
        if (FAILED(readback->Map(0, nullptr, &p))) return false;
        const auto* bytes = static_cast<const std::uint8_t*>(p);
        bool ok = true;
        for (UINT z = 0; z < fp[s].Footprint.Depth && ok; ++z) {
            for (UINT y = 0; y < rows[s] && ok; ++y) {
                const auto* row = bytes + fp[s].Offset + (UINT64(z) * rows[s] + y) * fp[s].Footprint.RowPitch;
                for (UINT64 x = 0; x < row_bytes[s] && ok; ++x) {
                    const auto want = static_cast<std::uint8_t>(seed + s * 53 + z * 19 + y * 7 + x * 3);
                    if (row[x] != want) {
                        std::fprintf(stderr, "3D sub %u (z=%u,y=%u,x=%llu): got 0x%02x want 0x%02x\n", s, z, y,
                                     static_cast<unsigned long long>(x), row[x], want);
                        ok = false;
                    }
                }
            }
        }
        readback->Unmap(0, nullptr);
        return ok;
    };

    if (!fill_upload(0x11)) return false;
    Ref<ID3D12CommandAllocator> alloc;
    Ref<ID3D12GraphicsCommandList> list;
    if (FAILED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(alloc.put()))) ||
        FAILED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, alloc.p, nullptr,
                                         IID_PPV_ARGS(list.put()))))
        return false;

    for (UINT s = 0; s < 2; ++s) {
        D3D12_TEXTURE_COPY_LOCATION dst{vol.p, D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, {}};
        dst.SubresourceIndex = s;
        D3D12_TEXTURE_COPY_LOCATION src{upload.p, D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, {}};
        src.PlacedFootprint = fp[s];
        list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
    }
    const D3D12_DISCARD_REGION discard_mip1{0, nullptr, 1, 1};
    list->DiscardResource(vol.p, &discard_mip1);
    const auto to_src = transition(vol.p, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COPY_SOURCE);
    list->ResourceBarrier(1, &to_src);
    {
        D3D12_TEXTURE_COPY_LOCATION dst{readback.p, D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, {}};
        dst.PlacedFootprint = fp[0];
        D3D12_TEXTURE_COPY_LOCATION src{vol.p, D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, {}};
        src.SubresourceIndex = 0;
        list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
    }
    if (FAILED(list->Close())) return false;
    ID3D12CommandList* lists[] = {list.p};
    queue->ExecuteCommandLists(1, lists);
    if (rt::d3d12::last_execute_result(queue) != leonida::ReplayResult::success ||
        !wait_queue(queue, fence, fence_value))
        return false;
    if (!check_readback_sub(0, 0x11)) return false;

    // Whole 3D discard, then upload new pattern and read back both mips.
    if (!fill_upload(0x47)) return false;
    if (FAILED(alloc->Reset()) || FAILED(list->Reset(alloc.p, nullptr))) return false;
    list->DiscardResource(vol.p, nullptr);
    const auto to_dst = transition(vol.p, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COPY_DEST);
    list->ResourceBarrier(1, &to_dst);
    for (UINT s = 0; s < 2; ++s) {
        D3D12_TEXTURE_COPY_LOCATION dst{vol.p, D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, {}};
        dst.SubresourceIndex = s;
        D3D12_TEXTURE_COPY_LOCATION src{upload.p, D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, {}};
        src.PlacedFootprint = fp[s];
        list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
    }
    list->ResourceBarrier(1, &to_src);
    for (UINT s = 0; s < 2; ++s) {
        D3D12_TEXTURE_COPY_LOCATION dst{readback.p, D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, {}};
        dst.PlacedFootprint = fp[s];
        D3D12_TEXTURE_COPY_LOCATION src{vol.p, D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, {}};
        src.SubresourceIndex = s;
        list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
    }
    if (FAILED(list->Close())) return false;
    queue->ExecuteCommandLists(1, lists);
    if (rt::d3d12::last_execute_result(queue) != leonida::ReplayResult::success ||
        !wait_queue(queue, fence, fence_value))
        return false;
    if (!check_readback_sub(0, 0x47) || !check_readback_sub(1, 0x47)) return false;

    // Default buffer: whole discard and single-subresource discard, followed by copy + readback.
    Ref<ID3D12Resource> def_buf, buf_up, buf_rb;
    if (FAILED(make_committed(device, D3D12_HEAP_TYPE_DEFAULT, buffer_desc(256), D3D12_RESOURCE_STATE_COPY_DEST,
                              def_buf.put())) ||
        FAILED(make_committed(device, D3D12_HEAP_TYPE_UPLOAD, buffer_desc(256), D3D12_RESOURCE_STATE_GENERIC_READ,
                              buf_up.put())) ||
        FAILED(make_committed(device, D3D12_HEAP_TYPE_READBACK, buffer_desc(256), D3D12_RESOURCE_STATE_COPY_DEST,
                              buf_rb.put())))
        return false;
    {
        void* p = nullptr;
        if (FAILED(buf_up->Map(0, nullptr, &p))) return false;
        auto* b = static_cast<std::uint8_t*>(p);
        for (UINT i = 0; i < 256; ++i) b[i] = static_cast<std::uint8_t>(i ^ 0x5a);
        buf_up->Unmap(0, nullptr);
    }
    const D3D12_DISCARD_REGION buf_sub0{0, nullptr, 0, 1};
    if (FAILED(alloc->Reset()) || FAILED(list->Reset(alloc.p, nullptr))) return false;
    list->DiscardResource(def_buf.p, nullptr);
    list->DiscardResource(def_buf.p, &buf_sub0);
    list->CopyBufferRegion(def_buf.p, 0, buf_up.p, 0, 256);
    const auto b_to_src = transition(def_buf.p, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COPY_SOURCE);
    list->ResourceBarrier(1, &b_to_src);
    list->CopyBufferRegion(buf_rb.p, 0, def_buf.p, 0, 256);
    if (FAILED(list->Close())) return false;
    queue->ExecuteCommandLists(1, lists);
    if (rt::d3d12::last_execute_result(queue) != leonida::ReplayResult::success ||
        !wait_queue(queue, fence, fence_value))
        return false;
    {
        void* p = nullptr;
        if (FAILED(buf_rb->Map(0, nullptr, &p))) return false;
        const auto* b = static_cast<const std::uint8_t*>(p);
        bool ok = true;
        for (UINT i = 0; i < 256; ++i) {
            if (b[i] != static_cast<std::uint8_t>(i ^ 0x5a)) {
                std::fprintf(stderr, "buffer byte %u: got 0x%02x want 0x%02x\n", i, b[i],
                             static_cast<std::uint8_t>(i ^ 0x5a));
                ok = false;
                break;
            }
        }
        buf_rb->Unmap(0, nullptr);
        if (!ok) return false;
    }
    std::printf("3D texture & default buffer discard + readback: ok\n");
    return true;
}

bool test_compute_queue_discard(ID3D12Device* device, ID3D12CommandQueue* direct_queue, ID3D12Fence* fence,
                                UINT64& fence_value) {
    D3D12_COMMAND_QUEUE_DESC cq_desc{};
    cq_desc.Type = D3D12_COMMAND_LIST_TYPE_COMPUTE;
    Ref<ID3D12CommandQueue> compute_queue;
    if (FAILED(device->CreateCommandQueue(&cq_desc, IID_PPV_ARGS(compute_queue.put())))) return false;

    // 1) UAV texture in UNORDERED_ACCESS (tracked as GENERAL) discarded on compute queue.
    Ref<ID3D12Resource> uav;
    if (FAILED(make_committed(device, D3D12_HEAP_TYPE_DEFAULT,
                              tex2d(16, 16, 1, 1, DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS),
                              D3D12_RESOURCE_STATE_UNORDERED_ACCESS, uav.put())))
        return false;

    // 2) RT+UAV texture cleared on direct queue (tracked as COLOR_ATTACHMENT_OPTIMAL), then discarded on compute
    //    queue: must execute cleanly with 0 Vulkan validation errors (exercises detail::discard_layout_allowed).
    Ref<ID3D12Resource> rt_uav;
    if (FAILED(make_committed(
            device, D3D12_HEAP_TYPE_DEFAULT,
            tex2d(16, 16, 1, 1, DXGI_FORMAT_R8G8B8A8_UNORM,
                  D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET | D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS),
            D3D12_RESOURCE_STATE_RENDER_TARGET, rt_uav.put())))
        return false;

    D3D12_DESCRIPTOR_HEAP_DESC heap_desc{};
    heap_desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
    heap_desc.NumDescriptors = 1;
    Ref<ID3D12DescriptorHeap> rtv_heap;
    if (FAILED(device->CreateDescriptorHeap(&heap_desc, IID_PPV_ARGS(rtv_heap.put())))) return false;
    const D3D12_CPU_DESCRIPTOR_HANDLE rtv = rtv_heap->GetCPUDescriptorHandleForHeapStart();
    device->CreateRenderTargetView(rt_uav.p, nullptr, rtv);

    {
        Ref<ID3D12CommandAllocator> d_alloc;
        Ref<ID3D12GraphicsCommandList> d_list;
        if (FAILED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(d_alloc.put()))) ||
            FAILED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, d_alloc.p, nullptr,
                                             IID_PPV_ARGS(d_list.put()))))
            return false;
        const float clear_color[4] = {0.25f, 0.5f, 0.75f, 1.0f};
        d_list->ClearRenderTargetView(rtv, clear_color, 0, nullptr);
        if (FAILED(d_list->Close())) return false;
        ID3D12CommandList* d_lists[] = {d_list.p};
        direct_queue->ExecuteCommandLists(1, d_lists);
        if (rt::d3d12::last_execute_result(direct_queue) != leonida::ReplayResult::success ||
            !wait_queue(direct_queue, fence, fence_value)) {
            std::fprintf(stderr, "compute discard: direct clear failed\n");
            return false;
        }
    }

    {
        Ref<ID3D12CommandAllocator> c_alloc;
        Ref<ID3D12GraphicsCommandList> c_list;
        if (FAILED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE, IID_PPV_ARGS(c_alloc.put()))) ||
            FAILED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COMPUTE, c_alloc.p, nullptr,
                                             IID_PPV_ARGS(c_list.put()))))
            return false;
        c_list->DiscardResource(uav.p, nullptr);
        c_list->DiscardResource(rt_uav.p, nullptr);
        if (FAILED(c_list->Close())) return false;
        ID3D12CommandList* c_lists[] = {c_list.p};
        compute_queue->ExecuteCommandLists(1, c_lists);
        if (rt::d3d12::last_execute_result(compute_queue.p) != leonida::ReplayResult::success ||
            !wait_queue(compute_queue.p, fence, fence_value)) {
            std::fprintf(stderr, "compute discard: compute execute failed\n");
            return false;
        }
    }
    std::printf("compute queue discard (GENERAL and COLOR_ATTACHMENT_OPTIMAL-tracked): ok\n");
    return true;
}
}  // namespace

int main() {
    vktest::Device vk;
    vk.all_queue_types = true;
    const auto status = vk.initialize();
    if (status == vktest::DeviceStatus::unavailable) {
        std::printf("SKIP: no Vulkan device\n");
        return 77;
    }
    if (status != vktest::DeviceStatus::ready) return 1;

    unsigned failures = 0;
    std::string mine;
    {
        Ref<ID3D12Device> ours;
        rt::d3d12::VulkanDeviceInfo info{};
        info.get_instance_proc = vk.instance_proc;
        info.instance = vk.instance;
        info.physical = vk.physical;
        info.device = vk.device;
        info.queue = vk.queue;
        info.queue_family = vk.family;
        if (vk.compute_queue) {
            info.compute_family = vk.compute_family;
            info.compute_queues = {vk.compute_queue};
        }
        if (vk.copy_queue) {
            info.copy_family = vk.copy_family;
            info.copy_queues = {vk.copy_queue};
        }
        if (FAILED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(ours.put())))) {
            std::fprintf(stderr, "create_vulkan_device failed\n");
            return 1;
        }

        mine = parity_cases(ours.p);
        if (!check_expected_parity_expectations(mine)) ++failures;

        D3D12_COMMAND_QUEUE_DESC qd{};
        qd.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
        Ref<ID3D12CommandQueue> queue;
        Ref<ID3D12Fence> fence;
        if (FAILED(ours->CreateCommandQueue(&qd, IID_PPV_ARGS(queue.put()))) ||
            FAILED(ours->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(fence.put())))) {
            std::fprintf(stderr, "queue/fence creation failed\n");
            return 1;
        }
        UINT64 fence_value = 0;
        if (!test_invalid_arguments(ours.p, queue.p)) ++failures;
        if (!test_rt_subresources_and_partial_rect(ours.p, queue.p, fence.p, fence_value)) ++failures;
        if (!test_depth_stencil_discard(ours.p, queue.p, fence.p, fence_value)) ++failures;
        if (!test_3d_and_buffer_discard(ours.p, queue.p, fence.p, fence_value)) ++failures;
        if (!test_compute_queue_discard(ours.p, queue.p, fence.p, fence_value)) ++failures;
    }

    if (!hostwine::host_is_wine()) {
        Ref<IDXGIFactory4> factory;
        if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(factory.put())))) {
            std::fprintf(stderr, "CreateDXGIFactory1 failed\n");
            return 1;
        }
        unsigned references = 0;
        for (const bool warp : {false, true}) {
            Ref<IDXGIAdapter> adapter;
            Ref<ID3D12Device> host;
            std::string adapter_name;
            if (!hostadapter::reference(factory.p, vk, warp, adapter.put(), adapter_name)) continue;
            if (FAILED(D3D12CreateDevice(adapter.p, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(host.put())))) continue;
            const auto a = split_lines(mine);
            const auto b = split_lines(parity_cases(host.p));
            unsigned differ = 0;
            for (size_t i = 0; i < a.size() || i < b.size(); ++i) {
                const std::string x = i < a.size() ? a[i] : "";
                const std::string y = i < b.size() ? b[i] : "";
                if (x != y) {
                    ++differ;
                    std::fprintf(stderr, "[%s] ours: %s | host: %s\n", adapter_name.c_str(), x.c_str(), y.c_str());
                }
            }
            std::printf("DiscardResource Close() parity against %s: %zu cases, %u differ\n", adapter_name.c_str(),
                        a.size(), differ);
            failures += differ;
            ++references;
        }
        if (!references) {
            std::fprintf(stderr, "no D3D12 reference device (matching adapter or WARP)\n");
            ++failures;
        }
    } else {
        std::printf("DiscardResource: host comparison skipped under Wine\n");
    }

    if (vk.validation_errors()) {
        std::fprintf(stderr, "Vulkan validation errors: %u\n", vk.validation_errors());
        return 1;
    }
    return failures ? 1 : 0;
}
