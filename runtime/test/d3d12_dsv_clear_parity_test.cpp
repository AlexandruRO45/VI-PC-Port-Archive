// Synthetic Close and GPU submission parity for DSV clear flags and out-of-range depth: WARP versus Vulkan COM.
// No game data, device debug layer or proprietary shaders.
#include "vk_test_device.h"
#include "../shims/d3d12_com.h"
#include <dxgi1_4.h>
#include <cmath>
#include <cstdio>
#include <limits>

namespace {
template<class T> struct Ref {
    T* p = nullptr;
    ~Ref() { if (p) p->Release(); }
    T* operator->() const { return p; }
};

int check(ID3D12Device* device, const char* name) {
    const D3D12_HEAP_PROPERTIES heap{D3D12_HEAP_TYPE_DEFAULT};
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    desc.Width = desc.Height = 64;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.Format = DXGI_FORMAT_D32_FLOAT_S8X24_UINT;
    desc.SampleDesc.Count = 1;
    desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;
    const D3D12_CLEAR_VALUE clear{DXGI_FORMAT_D32_FLOAT_S8X24_UINT, {1.0f, 0}};
    Ref<ID3D12Resource> texture;
    if (FAILED(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc,
                                              D3D12_RESOURCE_STATE_DEPTH_WRITE, &clear,
                                              IID_PPV_ARGS(&texture.p)))) return 1;
    const D3D12_DESCRIPTOR_HEAP_DESC dsv_desc{D3D12_DESCRIPTOR_HEAP_TYPE_DSV, 1, D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0};
    Ref<ID3D12DescriptorHeap> dsv_heap;
    if (FAILED(device->CreateDescriptorHeap(&dsv_desc, IID_PPV_ARGS(&dsv_heap.p)))) return 1;
    const auto handle = dsv_heap->GetCPUDescriptorHandleForHeapStart();
    device->CreateDepthStencilView(texture.p, nullptr, handle);
    Ref<ID3D12CommandQueue> queue;
    D3D12_COMMAND_QUEUE_DESC queue_desc{};
    queue_desc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    if (FAILED(device->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&queue.p)))) return 1;
    Ref<ID3D12Fence> fence;
    if (FAILED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence.p)))) return 1;
    HANDLE event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    if (!event) return 1;

    struct Case { const char* label; D3D12_CLEAR_FLAGS flags; float depth; };
    const Case cases[] = {
        {"flags zero", D3D12_CLEAR_FLAGS(0), 1.0f},
        {"unknown flag bits", D3D12_CLEAR_FLAGS(7), 0.5f},
        {"depth above range", D3D12_CLEAR_FLAG_DEPTH, 1.5f},
        {"depth below range", D3D12_CLEAR_FLAG_DEPTH, -0.5f},
        {"depth NaN", D3D12_CLEAR_FLAG_DEPTH, std::numeric_limits<float>::quiet_NaN()},
        {"valid depth", D3D12_CLEAR_FLAG_DEPTH, 0.25f},
    };
    for (const auto& c : cases) {
        Ref<ID3D12CommandAllocator> allocator;
        Ref<ID3D12GraphicsCommandList> list;
        if (FAILED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator.p))) ||
            FAILED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.p, nullptr,
                                             IID_PPV_ARGS(&list.p)))) return 1;
        list->ClearDepthStencilView(handle, c.flags, c.depth, 7, 0, nullptr);
        const HRESULT hr = list->Close();
        if (FAILED(hr)) {
            std::fprintf(stderr, "%s %s: Close failed %08x\n", name, c.label, unsigned(hr));
            CloseHandle(event);
            return 1;
        }
        ID3D12CommandList* lists[] = {list.p};
        queue->ExecuteCommandLists(1, lists);
        const UINT64 value = UINT64(&c - cases) + 1;
        if (FAILED(queue->Signal(fence.p, value)) || FAILED(fence->SetEventOnCompletion(value, event)) ||
            WaitForSingleObject(event, 5000) != WAIT_OBJECT_0) {
            std::fprintf(stderr, "%s %s: GPU submission did not complete\n", name, c.label);
            CloseHandle(event);
            return 1;
        }
    }
    CloseHandle(event);
    std::printf("%s: six DSV clear Close and GPU submissions passed\n", name);
    return 0;
}
}

int main() {
    Ref<IDXGIFactory4> factory;
    Ref<IDXGIAdapter> warp;
    Ref<ID3D12Device> host;
    if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory.p))) ||
        FAILED(factory->EnumWarpAdapter(IID_PPV_ARGS(&warp.p))) ||
        FAILED(D3D12CreateDevice(warp.p, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&host.p)))) return 77;
    if (check(host.p, "native WARP")) return 1;
    vktest::Device vk;
    const auto status = vk.initialize();
    if (status == vktest::DeviceStatus::unavailable) return 77;
    if (status != vktest::DeviceStatus::ready) return 1;
    rt::d3d12::VulkanDeviceInfo info{};
    info.get_instance_proc = vk.instance_proc;
    info.instance = vk.instance;
    info.physical = vk.physical;
    info.device = vk.device;
    info.queue = vk.queue;
    info.queue_family = vk.family;
    Ref<ID3D12Device> layer;
    if (FAILED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&layer.p)))) return 1;
    if (check(layer.p, "Leonida Vulkan")) return 1;
    if (vk.validation_errors()) {
        std::fprintf(stderr, "%u Vulkan validation errors\n", vk.validation_errors());
        return 1;
    }
    return 0;
}
