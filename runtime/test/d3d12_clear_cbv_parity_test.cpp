// Parity and validation test for:
//   1. ClearRenderTargetView / ClearDepthStencilView with NumRects > 0 and pRects == nullptr:
//      must fail Close() with E_INVALIDARG and must NOT wipe previously cleared target pixels.
//   2. Valid rectangle clears (NumRects == 1, valid pRects):
//      must succeed, preserve outside pixels, and match native Windows WARP byte-for-byte.
//   3. CreateConstantBufferView:
//      valid 256B-aligned BufferLocation and 256..65536B 256B-aligned SizeInBytes produce valid GFX10 V# descriptors;
//      null desc, zero/misaligned BufferLocation, or zero/misaligned/oversized (> 64 KiB) SizeInBytes produce null V#.
// No game data or shaders. Run with LEONIDA_VK_VALIDATION=1 to check for 0 Vulkan validation errors.
#include "vk_test_device.h"
#include "../shims/d3d12_com.h"
#include "../src/gfx10_descriptors.h"

#include <dxgi1_4.h>

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#define CHECK(condition)                                                                      \
    do {                                                                                      \
        if (!(condition)) {                                                                   \
            std::fprintf(stderr, "d3d12_clear_cbv_parity_test line %d: %s\n", __LINE__,       \
                         #condition);                                                         \
            return false;                                                                     \
        }                                                                                     \
    } while (0)

namespace {

template<class T>
struct Ref {
    T* p = nullptr;
    ~Ref() {
        if (p) p->Release();
    }
    T* operator->() const { return p; }
};

struct Event {
    HANDLE handle = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    ~Event() {
        if (handle) CloseHandle(handle);
    }
};

bool execute_and_wait(ID3D12CommandQueue* queue, ID3D12GraphicsCommandList* list, ID3D12Fence* fence,
                      UINT64 fence_value) {
    CHECK(SUCCEEDED(list->Close()));
    ID3D12CommandList* lists[] = {list};
    queue->ExecuteCommandLists(1, lists);
    CHECK(SUCCEEDED(queue->Signal(fence, fence_value)));
    Event event;
    CHECK(event.handle != nullptr);
    CHECK(SUCCEEDED(fence->SetEventOnCompletion(fence_value, event.handle)));
    CHECK(WaitForSingleObject(event.handle, 5000) == WAIT_OBJECT_0);
    return true;
}

D3D12_RESOURCE_DESC make_buffer_desc(UINT64 width) {
    D3D12_RESOURCE_DESC d{};
    d.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    d.Width = width;
    d.Height = 1;
    d.DepthOrArraySize = 1;
    d.MipLevels = 1;
    d.SampleDesc.Count = 1;
    d.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    return d;
}

D3D12_RESOURCE_DESC make_tex2d_desc(UINT width, UINT height, DXGI_FORMAT format, D3D12_RESOURCE_FLAGS flags) {
    D3D12_RESOURCE_DESC d{};
    d.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    d.Width = width;
    d.Height = height;
    d.DepthOrArraySize = 1;
    d.MipLevels = 1;
    d.Format = format;
    d.SampleDesc.Count = 1;
    d.Flags = flags;
    return d;
}

bool readback_rtv_rect_clear(ID3D12Device* device, std::vector<std::uint8_t>& pixels) {
    Ref<ID3D12CommandQueue> queue;
    Ref<ID3D12CommandAllocator> alloc;
    Ref<ID3D12GraphicsCommandList> list;
    Ref<ID3D12Fence> fence;
    D3D12_COMMAND_QUEUE_DESC qd{};
    CHECK(SUCCEEDED(device->CreateCommandQueue(&qd, IID_PPV_ARGS(&queue.p))));
    CHECK(SUCCEEDED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&alloc.p))));
    CHECK(SUCCEEDED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, alloc.p, nullptr,
                                              IID_PPV_ARGS(&list.p))));
    CHECK(SUCCEEDED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence.p))));

    const auto tex_desc = make_tex2d_desc(4, 4, DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET);
    D3D12_HEAP_PROPERTIES default_heap{};
    default_heap.Type = D3D12_HEAP_TYPE_DEFAULT;
    Ref<ID3D12Resource> rt;
    CHECK(SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &tex_desc,
                                                    D3D12_RESOURCE_STATE_RENDER_TARGET, nullptr,
                                                    IID_PPV_ARGS(&rt.p))));

    D3D12_PLACED_SUBRESOURCE_FOOTPRINT fp{};
    UINT64 total = 0;
    device->GetCopyableFootprints(&tex_desc, 0, 1, 0, &fp, nullptr, nullptr, &total);

    const auto rb_desc = make_buffer_desc(total);
    D3D12_HEAP_PROPERTIES rb_heap{};
    rb_heap.Type = D3D12_HEAP_TYPE_READBACK;
    Ref<ID3D12Resource> rb;
    CHECK(SUCCEEDED(device->CreateCommittedResource(&rb_heap, D3D12_HEAP_FLAG_NONE, &rb_desc,
                                                    D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&rb.p))));

    Ref<ID3D12DescriptorHeap> rtv_heap;
    const D3D12_DESCRIPTOR_HEAP_DESC hd{D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 1, D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0};
    CHECK(SUCCEEDED(device->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&rtv_heap.p))));
    const auto rtv = rtv_heap->GetCPUDescriptorHandleForHeapStart();
    device->CreateRenderTargetView(rt.p, nullptr, rtv);

    const FLOAT red[4] = {1.0f, 0.0f, 0.0f, 1.0f};
    const FLOAT green[4] = {0.0f, 1.0f, 0.0f, 1.0f};
    const D3D12_RECT rect{1, 1, 3, 3};
    list->ClearRenderTargetView(rtv, red, 0, nullptr);
    list->ClearRenderTargetView(rtv, green, 1, &rect);

    D3D12_RESOURCE_BARRIER b{};
    b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    b.Transition.pResource = rt.p;
    b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    b.Transition.StateBefore = D3D12_RESOURCE_STATE_RENDER_TARGET;
    b.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
    list->ResourceBarrier(1, &b);

    D3D12_TEXTURE_COPY_LOCATION src{}, dst{};
    src.pResource = rt.p;
    src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    src.SubresourceIndex = 0;
    dst.pResource = rb.p;
    dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    dst.PlacedFootprint = fp;
    list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);

    CHECK(execute_and_wait(queue.p, list.p, fence.p, 1));

    void* mapped = nullptr;
    const D3D12_RANGE range{0, SIZE_T(total)};
    CHECK(SUCCEEDED(rb->Map(0, &range, &mapped)) && mapped);
    pixels.clear();
    for (UINT y = 0; y < 4; ++y) {
        const auto* row = static_cast<const std::uint8_t*>(mapped) + fp.Offset + y * fp.Footprint.RowPitch;
        pixels.insert(pixels.end(), row, row + 4 * 4);
    }
    const D3D12_RANGE none{};
    rb->Unmap(0, &none);
    return true;
}

bool readback_dsv_rect_clear(ID3D12Device* device, std::vector<std::uint8_t>& depth_pixels,
                             std::vector<std::uint8_t>& stencil_pixels) {
    Ref<ID3D12CommandQueue> queue;
    Ref<ID3D12CommandAllocator> alloc;
    Ref<ID3D12GraphicsCommandList> list;
    Ref<ID3D12Fence> fence;
    D3D12_COMMAND_QUEUE_DESC qd{};
    CHECK(SUCCEEDED(device->CreateCommandQueue(&qd, IID_PPV_ARGS(&queue.p))));
    CHECK(SUCCEEDED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&alloc.p))));
    CHECK(SUCCEEDED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, alloc.p, nullptr,
                                              IID_PPV_ARGS(&list.p))));
    CHECK(SUCCEEDED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence.p))));

    const auto tex_desc =
        make_tex2d_desc(4, 4, DXGI_FORMAT_D32_FLOAT_S8X24_UINT, D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL);
    D3D12_HEAP_PROPERTIES default_heap{};
    default_heap.Type = D3D12_HEAP_TYPE_DEFAULT;
    Ref<ID3D12Resource> ds;
    CHECK(SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &tex_desc,
                                                    D3D12_RESOURCE_STATE_DEPTH_WRITE, nullptr,
                                                    IID_PPV_ARGS(&ds.p))));

    D3D12_PLACED_SUBRESOURCE_FOOTPRINT fp[2]{};
    UINT64 total = 0;
    device->GetCopyableFootprints(&tex_desc, 0, 2, 0, fp, nullptr, nullptr, &total);

    const auto rb_desc = make_buffer_desc(total);
    D3D12_HEAP_PROPERTIES rb_heap{};
    rb_heap.Type = D3D12_HEAP_TYPE_READBACK;
    Ref<ID3D12Resource> rb;
    CHECK(SUCCEEDED(device->CreateCommittedResource(&rb_heap, D3D12_HEAP_FLAG_NONE, &rb_desc,
                                                    D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&rb.p))));

    Ref<ID3D12DescriptorHeap> dsv_heap;
    const D3D12_DESCRIPTOR_HEAP_DESC hd{D3D12_DESCRIPTOR_HEAP_TYPE_DSV, 1, D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0};
    CHECK(SUCCEEDED(device->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&dsv_heap.p))));
    const auto dsv = dsv_heap->GetCPUDescriptorHandleForHeapStart();
    device->CreateDepthStencilView(ds.p, nullptr, dsv);

    const D3D12_CLEAR_FLAGS flags = D3D12_CLEAR_FLAG_DEPTH | D3D12_CLEAR_FLAG_STENCIL;
    const D3D12_RECT rect{1, 1, 3, 3};
    list->ClearDepthStencilView(dsv, flags, 0.75f, 0x5Au, 0, nullptr);
    list->ClearDepthStencilView(dsv, flags, 0.25f, 0x22u, 1, &rect);

    D3D12_RESOURCE_BARRIER b{};
    b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    b.Transition.pResource = ds.p;
    b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    b.Transition.StateBefore = D3D12_RESOURCE_STATE_DEPTH_WRITE;
    b.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
    list->ResourceBarrier(1, &b);

    D3D12_TEXTURE_COPY_LOCATION src{}, dst{};
    src.pResource = ds.p;
    src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    dst.pResource = rb.p;
    dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    for (UINT plane = 0; plane < 2; ++plane) {
        src.SubresourceIndex = plane;
        dst.PlacedFootprint = fp[plane];
        list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
    }

    CHECK(execute_and_wait(queue.p, list.p, fence.p, 1));

    void* mapped = nullptr;
    const D3D12_RANGE range{0, SIZE_T(total)};
    CHECK(SUCCEEDED(rb->Map(0, &range, &mapped)) && mapped);
    depth_pixels.clear();
    stencil_pixels.clear();
    for (UINT y = 0; y < 4; ++y) {
        const auto* drow = static_cast<const std::uint8_t*>(mapped) + fp[0].Offset + y * fp[0].Footprint.RowPitch;
        depth_pixels.insert(depth_pixels.end(), drow, drow + 4 * 4);
        const auto* srow = static_cast<const std::uint8_t*>(mapped) + fp[1].Offset + y * fp[1].Footprint.RowPitch;
        stencil_pixels.insert(stencil_pixels.end(), srow, srow + 4);
    }
    const D3D12_RANGE none{};
    rb->Unmap(0, &none);
    return true;
}

bool test_invalid_clear_rects_do_not_wipe(ID3D12Device* device) {
    Ref<ID3D12CommandQueue> queue;
    Ref<ID3D12CommandAllocator> alloc;
    Ref<ID3D12GraphicsCommandList> list;
    Ref<ID3D12Fence> fence;
    D3D12_COMMAND_QUEUE_DESC qd{};
    CHECK(SUCCEEDED(device->CreateCommandQueue(&qd, IID_PPV_ARGS(&queue.p))));
    CHECK(SUCCEEDED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&alloc.p))));
    CHECK(SUCCEEDED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, alloc.p, nullptr,
                                              IID_PPV_ARGS(&list.p))));
    CHECK(SUCCEEDED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence.p))));

    D3D12_HEAP_PROPERTIES default_heap{};
    default_heap.Type = D3D12_HEAP_TYPE_DEFAULT;
    D3D12_HEAP_PROPERTIES rb_heap{};
    rb_heap.Type = D3D12_HEAP_TYPE_READBACK;

    // --- RTV: seed to red, attempt invalid (1, nullptr) clear -> Close() == E_INVALIDARG, pixels stay red ---
    const auto rt_desc = make_tex2d_desc(4, 4, DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET);
    Ref<ID3D12Resource> rt;
    CHECK(SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &rt_desc,
                                                    D3D12_RESOURCE_STATE_RENDER_TARGET, nullptr,
                                                    IID_PPV_ARGS(&rt.p))));
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT rt_fp{};
    UINT64 rt_total = 0;
    device->GetCopyableFootprints(&rt_desc, 0, 1, 0, &rt_fp, nullptr, nullptr, &rt_total);
    const auto rt_rb_desc = make_buffer_desc(rt_total);
    Ref<ID3D12Resource> rt_rb;
    CHECK(SUCCEEDED(device->CreateCommittedResource(&rb_heap, D3D12_HEAP_FLAG_NONE, &rt_rb_desc,
                                                    D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                    IID_PPV_ARGS(&rt_rb.p))));
    Ref<ID3D12DescriptorHeap> rtv_heap;
    const D3D12_DESCRIPTOR_HEAP_DESC rtv_hd{D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 1, D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0};
    CHECK(SUCCEEDED(device->CreateDescriptorHeap(&rtv_hd, IID_PPV_ARGS(&rtv_heap.p))));
    const auto rtv = rtv_heap->GetCPUDescriptorHandleForHeapStart();
    device->CreateRenderTargetView(rt.p, nullptr, rtv);

    const FLOAT red[4] = {1.0f, 0.0f, 0.0f, 1.0f};
    const FLOAT blue[4] = {0.0f, 0.0f, 1.0f, 1.0f};
    const FLOAT green[4] = {0.0f, 1.0f, 0.0f, 1.0f};
    list->ClearRenderTargetView(rtv, red, 0, nullptr);
    CHECK(execute_and_wait(queue.p, list.p, fence.p, 1));

    // Invalid clear alone must fail Close() with E_INVALIDARG.
    CHECK(SUCCEEDED(alloc->Reset()));
    CHECK(SUCCEEDED(list->Reset(alloc.p, nullptr)));
    list->ClearRenderTargetView(rtv, blue, 1, nullptr);
    CHECK(list->Close() == E_INVALIDARG);

    // Valid clear followed by invalid clear on the same list must also fail Close() with E_INVALIDARG.
    CHECK(SUCCEEDED(alloc->Reset()));
    CHECK(SUCCEEDED(list->Reset(alloc.p, nullptr)));
    list->ClearRenderTargetView(rtv, green, 0, nullptr);
    list->ClearRenderTargetView(rtv, blue, 2, nullptr);
    CHECK(list->Close() == E_INVALIDARG);

    // Read back the RTV and verify it was NOT wiped.
    CHECK(SUCCEEDED(alloc->Reset()));
    CHECK(SUCCEEDED(list->Reset(alloc.p, nullptr)));
    D3D12_RESOURCE_BARRIER rt_b{};
    rt_b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    rt_b.Transition.pResource = rt.p;
    rt_b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    rt_b.Transition.StateBefore = D3D12_RESOURCE_STATE_RENDER_TARGET;
    rt_b.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
    list->ResourceBarrier(1, &rt_b);
    D3D12_TEXTURE_COPY_LOCATION rt_src{}, rt_dst{};
    rt_src.pResource = rt.p;
    rt_src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    rt_dst.pResource = rt_rb.p;
    rt_dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    rt_dst.PlacedFootprint = rt_fp;
    list->CopyTextureRegion(&rt_dst, 0, 0, 0, &rt_src, nullptr);
    CHECK(execute_and_wait(queue.p, list.p, fence.p, 2));

    void* mapped = nullptr;
    const D3D12_RANGE rt_range{0, SIZE_T(rt_total)};
    CHECK(SUCCEEDED(rt_rb->Map(0, &rt_range, &mapped)) && mapped);
    const std::uint8_t expected_red[4] = {255, 0, 0, 255};
    for (UINT y = 0; y < 4; ++y) {
        const auto* row = static_cast<const std::uint8_t*>(mapped) + rt_fp.Offset + y * rt_fp.Footprint.RowPitch;
        for (UINT x = 0; x < 4; ++x) CHECK(std::memcmp(row + x * 4, expected_red, 4) == 0);
    }
    const D3D12_RANGE none{};
    rt_rb->Unmap(0, &none);

    // --- DSV: seed to (0.75, 0x5A), attempt invalid (1, nullptr) clear -> Close() == E_INVALIDARG, pixels intact ---
    const auto ds_desc =
        make_tex2d_desc(4, 4, DXGI_FORMAT_D32_FLOAT_S8X24_UINT, D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL);
    Ref<ID3D12Resource> ds;
    CHECK(SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &ds_desc,
                                                    D3D12_RESOURCE_STATE_DEPTH_WRITE, nullptr,
                                                    IID_PPV_ARGS(&ds.p))));
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT ds_fp[2]{};
    UINT64 ds_total = 0;
    device->GetCopyableFootprints(&ds_desc, 0, 2, 0, ds_fp, nullptr, nullptr, &ds_total);
    const auto ds_rb_desc = make_buffer_desc(ds_total);
    Ref<ID3D12Resource> ds_rb;
    CHECK(SUCCEEDED(device->CreateCommittedResource(&rb_heap, D3D12_HEAP_FLAG_NONE, &ds_rb_desc,
                                                    D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                    IID_PPV_ARGS(&ds_rb.p))));
    Ref<ID3D12DescriptorHeap> dsv_heap;
    const D3D12_DESCRIPTOR_HEAP_DESC dsv_hd{D3D12_DESCRIPTOR_HEAP_TYPE_DSV, 1, D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0};
    CHECK(SUCCEEDED(device->CreateDescriptorHeap(&dsv_hd, IID_PPV_ARGS(&dsv_heap.p))));
    const auto dsv = dsv_heap->GetCPUDescriptorHandleForHeapStart();
    device->CreateDepthStencilView(ds.p, nullptr, dsv);

    const D3D12_CLEAR_FLAGS ds_flags = D3D12_CLEAR_FLAG_DEPTH | D3D12_CLEAR_FLAG_STENCIL;
    CHECK(SUCCEEDED(alloc->Reset()));
    CHECK(SUCCEEDED(list->Reset(alloc.p, nullptr)));
    list->ClearDepthStencilView(dsv, ds_flags, 0.75f, 0x5Au, 0, nullptr);
    CHECK(execute_and_wait(queue.p, list.p, fence.p, 3));

    CHECK(SUCCEEDED(alloc->Reset()));
    CHECK(SUCCEEDED(list->Reset(alloc.p, nullptr)));
    list->ClearDepthStencilView(dsv, ds_flags, 0.25f, 0x11u, 1, nullptr);
    CHECK(list->Close() == E_INVALIDARG);

    CHECK(SUCCEEDED(alloc->Reset()));
    CHECK(SUCCEEDED(list->Reset(alloc.p, nullptr)));
    list->ClearDepthStencilView(dsv, D3D12_CLEAR_FLAG_DEPTH, 0.5f, 0x33u, 3, nullptr);
    CHECK(list->Close() == E_INVALIDARG);

    CHECK(SUCCEEDED(alloc->Reset()));
    CHECK(SUCCEEDED(list->Reset(alloc.p, nullptr)));
    D3D12_RESOURCE_BARRIER ds_b{};
    ds_b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    ds_b.Transition.pResource = ds.p;
    ds_b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    ds_b.Transition.StateBefore = D3D12_RESOURCE_STATE_DEPTH_WRITE;
    ds_b.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
    list->ResourceBarrier(1, &ds_b);
    D3D12_TEXTURE_COPY_LOCATION ds_src{}, ds_dst{};
    ds_src.pResource = ds.p;
    ds_src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    ds_dst.pResource = ds_rb.p;
    ds_dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    for (UINT plane = 0; plane < 2; ++plane) {
        ds_src.SubresourceIndex = plane;
        ds_dst.PlacedFootprint = ds_fp[plane];
        list->CopyTextureRegion(&ds_dst, 0, 0, 0, &ds_src, nullptr);
    }
    CHECK(execute_and_wait(queue.p, list.p, fence.p, 4));

    mapped = nullptr;
    const D3D12_RANGE ds_range{0, SIZE_T(ds_total)};
    CHECK(SUCCEEDED(ds_rb->Map(0, &ds_range, &mapped)) && mapped);
    for (UINT y = 0; y < 4; ++y) {
        const auto* drow = static_cast<const std::uint8_t*>(mapped) + ds_fp[0].Offset + y * ds_fp[0].Footprint.RowPitch;
        const auto* srow = static_cast<const std::uint8_t*>(mapped) + ds_fp[1].Offset + y * ds_fp[1].Footprint.RowPitch;
        for (UINT x = 0; x < 4; ++x) {
            float depth = 0.0f;
            std::memcpy(&depth, drow + x * 4, sizeof(depth));
            CHECK(depth == 0.75f);
            CHECK(srow[x] == 0x5Au);
        }
    }
    ds_rb->Unmap(0, &none);
    return true;
}

bool test_cbv_validation(ID3D12Device* device) {
    const UINT inc = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    CHECK(inc == 32);

    D3D12_HEAP_PROPERTIES default_heap{};
    default_heap.Type = D3D12_HEAP_TYPE_DEFAULT;
    const auto cb_desc = make_buffer_desc(131072);  // 128 KiB
    Ref<ID3D12Resource> cb;
    CHECK(SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &cb_desc,
                                                    D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&cb.p))));
    const D3D12_GPU_VIRTUAL_ADDRESS va = cb->GetGPUVirtualAddress();
    CHECK(va != 0 && (va & 255u) == 0);

    Ref<ID3D12DescriptorHeap> heap;
    const D3D12_DESCRIPTOR_HEAP_DESC hd{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 4, D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0};
    CHECK(SUCCEEDED(device->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&heap.p))));
    const auto slot = heap->GetCPUDescriptorHandleForHeapStart();

    // Null destination handle must be a safe no-op.
    device->CreateConstantBufferView(nullptr, D3D12_CPU_DESCRIPTOR_HANDLE{0});

    const std::uint8_t zeros[32]{};
    const auto expect_null_cbv = [&](const D3D12_CONSTANT_BUFFER_VIEW_DESC* desc) -> bool {
        std::memset(reinterpret_cast<void*>(slot.ptr), 0xA5, inc);
        device->CreateConstantBufferView(desc, slot);
        CHECK(std::memcmp(reinterpret_cast<const void*>(slot.ptr), zeros, sizeof(zeros)) == 0);
        return true;
    };
    const auto expect_valid_cbv = [&](D3D12_GPU_VIRTUAL_ADDRESS addr, UINT size) -> bool {
        std::memset(reinterpret_cast<void*>(slot.ptr), 0xA5, inc);
        const D3D12_CONSTANT_BUFFER_VIEW_DESC d{addr, size};
        device->CreateConstantBufferView(&d, slot);
        leonida::gfx10::BufferWords raw{};
        std::memcpy(raw.data(), reinterpret_cast<const void*>(slot.ptr), sizeof(raw));
        leonida::gfx10::Buffer decoded{};
        CHECK(leonida::gfx10::decode(raw, decoded));
        CHECK(decoded.address == addr);
        CHECK(decoded.stride == 0);
        CHECK(decoded.num_records == size);
        CHECK(decoded.format == leonida::gfx10::format_from_vulkan(VK_FORMAT_R32_UINT));
        CHECK(decoded.swizzle == leonida::gfx10::identity_swizzle);
        CHECK(decoded.oob_select == leonida::gfx10::OobSelect::raw);
        CHECK(std::memcmp(reinterpret_cast<const std::uint8_t*>(slot.ptr) + 16, zeros, 16) == 0);
        return true;
    };

    // Valid CBVs: 256B-aligned address, 256..65536B 256B-aligned size.
    constexpr UINT kMaxCbv = D3D12_REQ_CONSTANT_BUFFER_ELEMENT_COUNT * 16u;
    static_assert(kMaxCbv == 65536u);
    CHECK(expect_valid_cbv(va, 256u));
    CHECK(expect_valid_cbv(va + 256u, 512u));
    CHECK(expect_valid_cbv(va, kMaxCbv));
    CHECK(expect_valid_cbv(va + 65536u, kMaxCbv));

    // Invalid CBVs: must all produce an all-zero null descriptor.
    CHECK(expect_null_cbv(nullptr));
    const D3D12_CONSTANT_BUFFER_VIEW_DESC invalid_cases[] = {
        {0, 256u},                  // zero BufferLocation
        {va + 1u, 256u},            // misaligned BufferLocation (+1)
        {va + 16u, 256u},           // misaligned BufferLocation (+16)
        {va + 128u, 256u},          // misaligned BufferLocation (+128)
        {va + 255u, 256u},          // misaligned BufferLocation (+255)
        {va, 0u},                   // zero SizeInBytes
        {va, 1u},                   // misaligned SizeInBytes (1)
        {va, 16u},                  // misaligned SizeInBytes (16)
        {va, 128u},                 // misaligned SizeInBytes (128)
        {va, 255u},                 // misaligned SizeInBytes (255)
        {va, 257u},                 // misaligned SizeInBytes (257)
        {va, 65535u},               // misaligned SizeInBytes (65535)
        {va, kMaxCbv + 256u},       // oversized SizeInBytes (65792)
        {va, 131072u},              // oversized SizeInBytes (128 KiB)
        {va, 0xFFFFFF00u},          // oversized SizeInBytes (~4 GiB, 256B-aligned)
    };
    for (const auto& c : invalid_cases) CHECK(expect_null_cbv(&c));
    return true;
}

bool create_warp_device(Ref<ID3D12Device>& out) {
    Ref<IDXGIFactory4> factory;
    if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory.p)))) return false;
    Ref<IDXGIAdapter> warp;
    if (FAILED(factory->EnumWarpAdapter(IID_PPV_ARGS(&warp.p)))) return false;
    return SUCCEEDED(D3D12CreateDevice(warp.p, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&out.p)));
}

}  // namespace

int main() {
    vktest::Device vk;
    const auto status = vk.initialize();
    if (status == vktest::DeviceStatus::unavailable) return 77;
    if (status != vktest::DeviceStatus::ready) return 1;

    const auto errors_before = vk.validation_errors();
    const auto run = [&]() -> int {
        Ref<ID3D12Device> ours;
        rt::d3d12::VulkanDeviceInfo info{};
        info.get_instance_proc = vk.instance_proc;
        info.instance = vk.instance;
        info.physical = vk.physical;
        info.device = vk.device;
        info.queue = vk.queue;
        info.queue_family = vk.family;
        info.buffer_device_address = true;
        if (FAILED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&ours.p)))) {
            std::fprintf(stderr, "d3d12_clear_cbv_parity_test: create_vulkan_device failed\n");
            return 1;
        }

        if (!test_invalid_clear_rects_do_not_wipe(ours.p)) return 1;
        if (!test_cbv_validation(ours.p)) return 1;

        std::vector<std::uint8_t> rtv_ours, dsv_depth_ours, dsv_stencil_ours;
        if (!readback_rtv_rect_clear(ours.p, rtv_ours)) return 1;
        if (!readback_dsv_rect_clear(ours.p, dsv_depth_ours, dsv_stencil_ours)) return 1;

        // Verify expected pixels directly on Vulkan.
        for (UINT y = 0; y < 4; ++y) {
            for (UINT x = 0; x < 4; ++x) {
                const bool inside = (x >= 1 && x < 3 && y >= 1 && y < 3);
                const std::uint8_t expected_rgba[4] = {
                    inside ? std::uint8_t(0) : std::uint8_t(255),
                    inside ? std::uint8_t(255) : std::uint8_t(0),
                    0,
                    255,
                };
                const std::size_t off = (y * 4 + x) * 4;
                if (std::memcmp(rtv_ours.data() + off, expected_rgba, 4) != 0) {
                    std::fprintf(stderr, "d3d12_clear_cbv_parity_test: RTV rect clear pixel (%u,%u) mismatch\n", x, y);
                    return 1;
                }
                float depth = 0.0f;
                std::memcpy(&depth, dsv_depth_ours.data() + off, sizeof(depth));
                const float expected_depth = inside ? 0.25f : 0.75f;
                const std::uint8_t expected_stencil = inside ? 0x22u : 0x5Au;
                if (depth != expected_depth || dsv_stencil_ours[y * 4 + x] != expected_stencil) {
                    std::fprintf(stderr, "d3d12_clear_cbv_parity_test: DSV rect clear pixel (%u,%u) mismatch\n", x, y);
                    return 1;
                }
            }
        }

        // Cross-check valid rect clears against native Windows WARP when available.
        Ref<ID3D12Device> warp;
        if (create_warp_device(warp)) {
            std::vector<std::uint8_t> rtv_warp, dsv_depth_warp, dsv_stencil_warp;
            if (!readback_rtv_rect_clear(warp.p, rtv_warp)) return 1;
            if (!readback_dsv_rect_clear(warp.p, dsv_depth_warp, dsv_stencil_warp)) return 1;
            if (rtv_ours != rtv_warp) {
                std::fprintf(stderr, "d3d12_clear_cbv_parity_test: RTV rect clear differs from WARP\n");
                return 1;
            }
            if (dsv_depth_ours != dsv_depth_warp || dsv_stencil_ours != dsv_stencil_warp) {
                std::fprintf(stderr, "d3d12_clear_cbv_parity_test: DSV rect clear differs from WARP\n");
                return 1;
            }
        }
        return 0;
    };

    const int rc = run();
    const auto validation_errors = vk.validation_errors() - errors_before;
    if (validation_errors) {
        std::fprintf(stderr, "d3d12_clear_cbv_parity_test: %llu new Vulkan validation error(s)\n",
                     static_cast<unsigned long long>(validation_errors));
        return 1;
    }
    if (rc != 0) return rc;
    std::printf("d3d12_clear_cbv_parity_test: invalid clear rects rejected (targets preserved), valid rect clears "
                "match WARP, and CBV alignment/size validation passed (0 Vulkan validation errors)\n");
    return 0;
}
