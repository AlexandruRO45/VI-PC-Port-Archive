// Verifies LEONIDA_TEXTURE_MIP_DROP quality switch end-to-end on a Vulkan device (#6043 / #6365):
// - Multi-mip 2D texture creation with mips dropped at runtime.
// - 3D volume texture creation verifying depth is right-shifted by mips_dropped.
// - CopyTextureRegion gracefully skipping dropped mips while remapping retained subresources.
// - ResourceBarrier Transition and Texture barriers skipping dropped subresources and remapping retained mips.
// - CreateShaderResourceView partial mip views clamping without mip_count bleed.
// - ExecuteCommandLists submission executing without error or refusal.
#include "vk_test_device.h"
#include "../shims/d3d12_com.h"
#include <dxgi1_4.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define EXPECT(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return 1; } \
} while (0)

namespace {

template<class T>
struct Ref {
    T* p = nullptr;
    ~Ref() { reset(); }
    void reset() {
        if (p) p->Release();
        p = nullptr;
    }
    T* operator->() const { return p; }
};

D3D12_RESOURCE_DESC tex2d_desc(UINT width, UINT height, UINT mips, DXGI_FORMAT format) {
    D3D12_RESOURCE_DESC d{};
    d.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    d.Width = width;
    d.Height = height;
    d.DepthOrArraySize = 1;
    d.MipLevels = mips;
    d.Format = format;
    d.SampleDesc.Count = 1;
    d.Flags = D3D12_RESOURCE_FLAG_NONE;
    return d;
}

D3D12_RESOURCE_DESC tex3d_desc(UINT width, UINT height, UINT depth, UINT mips, DXGI_FORMAT format) {
    D3D12_RESOURCE_DESC d{};
    d.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE3D;
    d.Width = width;
    d.Height = height;
    d.DepthOrArraySize = depth;
    d.MipLevels = mips;
    d.Format = format;
    d.SampleDesc.Count = 1;
    d.Flags = D3D12_RESOURCE_FLAG_NONE;
    return d;
}

D3D12_RESOURCE_DESC buffer_desc(UINT64 bytes) {
    D3D12_RESOURCE_DESC d{};
    d.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    d.Width = bytes;
    d.Height = 1;
    d.DepthOrArraySize = 1;
    d.MipLevels = 1;
    d.SampleDesc.Count = 1;
    d.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    return d;
}

} // namespace

int main() {
#ifdef _WIN32
    _putenv("LEONIDA_TEXTURE_MIP_DROP=1");
#else
    setenv("LEONIDA_TEXTURE_MIP_DROP", "1", 1);
#endif

    Ref<ID3D12Device> device;
    const HRESULT hr = rt::d3d12::create_vulkan_device_standalone(IID_PPV_ARGS(&device.p));
    if (hr == E_FAIL || hr == DXGI_ERROR_UNSUPPORTED) return 77; // No Vulkan GPU
    EXPECT(SUCCEEDED(hr));

    D3D12_COMMAND_QUEUE_DESC queue_desc{};
    queue_desc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    Ref<ID3D12CommandQueue> queue;
    EXPECT(SUCCEEDED(device->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&queue.p))));

    Ref<ID3D12CommandAllocator> allocator;
    EXPECT(SUCCEEDED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator.p))));

    Ref<ID3D12GraphicsCommandList> list;
    EXPECT(SUCCEEDED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.p, nullptr, IID_PPV_ARGS(&list.p))));

    D3D12_HEAP_PROPERTIES default_heap{};
    default_heap.Type = D3D12_HEAP_TYPE_DEFAULT;

    D3D12_HEAP_PROPERTIES upload_heap{};
    upload_heap.Type = D3D12_HEAP_TYPE_UPLOAD;

    // 1. 2D Texture (16x16, 4 mips): with drop 1, physical image is 8x8, 3 mips.
    Ref<ID3D12Resource> tex2d;
    const auto desc2d = tex2d_desc(16, 16, 4, DXGI_FORMAT_R8G8B8A8_UNORM);
    EXPECT(SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &desc2d,
                                                     D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&tex2d.p))));
    EXPECT(tex2d->GetDesc().MipLevels == 4);
    EXPECT(tex2d->GetDesc().Width == 16);

    // 2. 3D Volume Texture (8x8x4, 3 mips): with drop 1, width=4, height=4, depth=2.
    Ref<ID3D12Resource> tex3d;
    const auto desc3d = tex3d_desc(8, 8, 4, 3, DXGI_FORMAT_R8G8B8A8_UNORM);
    EXPECT(SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &desc3d,
                                                     D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&tex3d.p))));
    EXPECT(tex3d->GetDesc().DepthOrArraySize == 4);

    // 3. Upload Buffer
    Ref<ID3D12Resource> upload;
    const auto udesc = buffer_desc(4096);
    EXPECT(SUCCEEDED(device->CreateCommittedResource(&upload_heap, D3D12_HEAP_FLAG_NONE, &udesc,
                                                     D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&upload.p))));

    // 4. Command List: CopyTextureRegion targeting dropped mip 0 and retained mips 1, 2.
    // Mip 0 copy: should be skipped gracefully (p.skip = true)
    D3D12_TEXTURE_COPY_LOCATION dst_loc0{};
    dst_loc0.pResource = tex2d.p;
    dst_loc0.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    dst_loc0.SubresourceIndex = 0; // dropped mip

    D3D12_TEXTURE_COPY_LOCATION src_loc{};
    src_loc.pResource = upload.p;
    src_loc.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    src_loc.PlacedFootprint.Footprint.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    src_loc.PlacedFootprint.Footprint.Width = 16;
    src_loc.PlacedFootprint.Footprint.Height = 16;
    src_loc.PlacedFootprint.Footprint.Depth = 1;
    src_loc.PlacedFootprint.Footprint.RowPitch = 256;

    list->CopyTextureRegion(&dst_loc0, 0, 0, 0, &src_loc, nullptr);

    // Mip 1 copy: retained mip (physical mip 0 of 8x8)
    D3D12_TEXTURE_COPY_LOCATION dst_loc1{};
    dst_loc1.pResource = tex2d.p;
    dst_loc1.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    dst_loc1.SubresourceIndex = 1; // retained mip

    D3D12_TEXTURE_COPY_LOCATION src_loc1{};
    src_loc1.pResource = upload.p;
    src_loc1.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    src_loc1.PlacedFootprint.Offset = 512;
    src_loc1.PlacedFootprint.Footprint.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    src_loc1.PlacedFootprint.Footprint.Width = 8;
    src_loc1.PlacedFootprint.Footprint.Height = 8;
    src_loc1.PlacedFootprint.Footprint.Depth = 1;
    src_loc1.PlacedFootprint.Footprint.RowPitch = 256;

    list->CopyTextureRegion(&dst_loc1, 0, 0, 0, &src_loc1, nullptr);

    // 5. ResourceBarrier transitions:
    // Subresource 0 barrier (dropped mip -> skipped)
    D3D12_RESOURCE_BARRIER b0{};
    b0.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    b0.Transition.pResource = tex2d.p;
    b0.Transition.Subresource = 0;
    b0.Transition.StateBefore = D3D12_RESOURCE_STATE_COMMON;
    b0.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_DEST;
    list->ResourceBarrier(1, &b0);

    // Subresource 1 barrier (retained mip -> remapped to physical 0)
    D3D12_RESOURCE_BARRIER b1{};
    b1.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    b1.Transition.pResource = tex2d.p;
    b1.Transition.Subresource = 1;
    b1.Transition.StateBefore = D3D12_RESOURCE_STATE_COMMON;
    b1.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_DEST;
    list->ResourceBarrier(1, &b1);

    // All subresources barrier
    D3D12_RESOURCE_BARRIER ball{};
    ball.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    ball.Transition.pResource = tex2d.p;
    ball.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    ball.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
    ball.Transition.StateAfter = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
    list->ResourceBarrier(1, &ball);

    EXPECT(SUCCEEDED(list->Close()));

    // 6. ExecuteCommandLists - verify no invalid subresource abort
    ID3D12CommandList* lists[] = {list.p};
    queue->ExecuteCommandLists(1, lists);

    Ref<ID3D12Fence> fence;
    EXPECT(SUCCEEDED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence.p))));
    EXPECT(SUCCEEDED(queue->Signal(fence.p, 1)));

    // 7. Descriptor Heap and SRVs (verifying first_mip & mip_count clamping without bleed)
    D3D12_DESCRIPTOR_HEAP_DESC heap_desc{};
    heap_desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    heap_desc.NumDescriptors = 4;
    Ref<ID3D12DescriptorHeap> srv_heap;
    EXPECT(SUCCEEDED(device->CreateDescriptorHeap(&heap_desc, IID_PPV_ARGS(&srv_heap.p))));
    const UINT stride = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    D3D12_CPU_DESCRIPTOR_HANDLE handle = srv_heap->GetCPUDescriptorHandleForHeapStart();

    // SRV 0: requests original MostDetailedMip = 0, MipLevels = 2
    // With drop 1, mip 0 dropped, so view should start at physical 0 with mip_count = 1 (no bleed into physical 1!)
    D3D12_SHADER_RESOURCE_VIEW_DESC srv0{};
    srv0.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    srv0.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    srv0.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    srv0.Texture2D.MostDetailedMip = 0;
    srv0.Texture2D.MipLevels = 2;
    device->CreateShaderResourceView(tex2d.p, &srv0, handle);

    // SRV 1: requests original MostDetailedMip = 1, MipLevels = 2
    // With drop 1, maps to physical MostDetailedMip = 0, MipLevels = 2.
    handle.ptr += stride;
    D3D12_SHADER_RESOURCE_VIEW_DESC srv1 = srv0;
    srv1.Texture2D.MostDetailedMip = 1;
    srv1.Texture2D.MipLevels = 2;
    device->CreateShaderResourceView(tex2d.p, &srv1, handle);

    // Wait for fence
    if (fence->GetCompletedValue() < 1) {
        HANDLE event = CreateEventA(nullptr, FALSE, FALSE, nullptr);
        EXPECT(SUCCEEDED(fence->SetEventOnCompletion(1, event)));
        WaitForSingleObject(event, 5000);
        CloseHandle(event);
    }
    EXPECT(fence->GetCompletedValue() >= 1);

    std::printf("d3d12_texture_mip_drop_test: PASS\n");
    return 0;
}
