#include "vk_test_device.h"
#include "../shims/d3d12_com.h"
#include "../shims/d3d12_xs_device.h"
#include "../src/command_list.h"

#include <cstdint>
#include <cstdio>

#define EXPECT(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return 1; } \
} while (0)

namespace {

using XboxHeapHandleFn = std::uint64_t(WINAPI*)(void* heap);

void* slot(void* object, unsigned index) { return (*reinterpret_cast<void***>(object))[index]; }

template<class T>
struct Ref {
    T* p = nullptr;
    ~Ref() { if (p) p->Release(); }
    T* operator->() const { return p; }
};

D3D12_HEAP_PROPERTIES default_heap() {
    D3D12_HEAP_PROPERTIES p{};
    p.Type = D3D12_HEAP_TYPE_DEFAULT;
    p.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
    p.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
    p.CreationNodeMask = 1;
    p.VisibleNodeMask = 1;
    return p;
}

bool view_fields(ID3D12Device* device, ID3D12Resource* resource, const D3D12_DEPTH_STENCIL_VIEW_DESC& dsv,
                 std::uint32_t mip, std::uint32_t first, std::uint32_t count) {
    D3D12_DESCRIPTOR_HEAP_DESC hd{};
    hd.Type = D3D12_DESCRIPTOR_HEAP_TYPE_DSV;
    hd.NumDescriptors = 1;
    Ref<ID3D12DescriptorHeap> heap;
    if (FAILED(device->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&heap.p)))) return false;
    D3D12_CPU_DESCRIPTOR_HANDLE handle{};
    handle.ptr = reinterpret_cast<XboxHeapHandleFn>(slot(heap.p, 9))(heap.p);
    device->CreateDepthStencilView(resource, &dsv, handle);
    const auto got = *reinterpret_cast<const leonida::TargetView*>(handle.ptr);
    if (got.mip != mip || got.first_layer != first || got.layer_count != count) {
        std::fprintf(stderr, "DSV: got mip=%u first=%u count=%u, want %u/%u/%u\n", got.mip, got.first_layer,
                     got.layer_count, mip, first, count);
        return false;
    }
    return true;
}

}  // namespace

int main() {
    vktest::Device d;
    if (!d.create()) { std::printf("SKIP (no Vulkan device)\n"); return 0; }

    rt::d3d12::VulkanDeviceInfo info{};
    info.get_instance_proc = d.instance_proc;
    info.instance = d.instance;
    info.physical = d.physical;
    info.device = d.device;
    info.queue = d.queue;
    info.queue_family = d.family;
    ID3D12Device* inner = nullptr;
    EXPECT(SUCCEEDED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&inner))));
    Ref<ID3D12Device> device;
    EXPECT(SUCCEEDED(rt::d3d12_xs::create_device_proxy(inner, __uuidof(ID3D12Device),
                                                      reinterpret_cast<void**>(&device.p))));
    rt::d3d12::set_device_outer(inner, device.p);

    D3D12_RESOURCE_DESC rd{};
    rd.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    rd.Width = 8;
    rd.Height = 8;
    rd.DepthOrArraySize = 4;
    rd.MipLevels = 2;
    rd.Format = DXGI_FORMAT_D32_FLOAT;
    rd.SampleDesc.Count = 1;
    rd.Flags = D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;
    const auto hp = default_heap();
    Ref<ID3D12Resource> resource;
    if (FAILED(device->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd, D3D12_RESOURCE_STATE_COMMON,
                                               nullptr, IID_PPV_ARGS(&resource.p)))) {
        std::printf("SKIP (no depth texture)\n");
        return 0;
    }

    D3D12_DEPTH_STENCIL_VIEW_DESC dsv{};
    dsv.Format = DXGI_FORMAT_D32_FLOAT;
    dsv.Flags = D3D12_DSV_FLAG_NONE;

    dsv.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE1D;
    dsv.Texture1D.MipSlice = 1;
    EXPECT(view_fields(device.p, resource.p, dsv, 1, 0, 1));

    dsv.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE1DARRAY;
    dsv.Texture1DArray.MipSlice = 1;
    dsv.Texture1DArray.FirstArraySlice = 2;
    dsv.Texture1DArray.ArraySize = 2;
    EXPECT(view_fields(device.p, resource.p, dsv, 1, 2, 2));

    dsv.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2DMSARRAY;
    dsv.Texture2DMSArray.FirstArraySlice = 1;
    dsv.Texture2DMSArray.ArraySize = 3;
    EXPECT(view_fields(device.p, resource.p, dsv, 0, 1, 3));

    std::printf("dsv_dimensions: PASS (1D, 1D-array, 2DMS-array descriptor encoding)\n");
    return 0;
}
