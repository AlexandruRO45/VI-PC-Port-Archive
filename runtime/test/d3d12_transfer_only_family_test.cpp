// The transfer-only family is known to the replay on every GPU, not only on one with VK_EXT_conditional_rendering.
// The layer keeps the bindless set 0 flush and the D24/RGB96 conversion passes off a COPY queue on a transfer-only
// family (vulkan_backend.h: transfer_only_family_); a family it does not know as transfer-only gets that work, which a
// queue without graphics or compute cannot run. Here a second D3D12 device is made over the same Vulkan device with
// conditional_rendering cleared (as on a GPU without the extension), and its first submission is a buffer copy on a
// COPY queue, which is where the first bindless flush lands. No game data. LEONIDA_VK_VALIDATION=1 checks Vulkan
// usage; the ctest fails on any VUID.
#include "../shims/d3d12_com.h"

#include <wrl/client.h>

#include <cstdint>
#include <cstdio>
#include <cstring>

namespace {

using Microsoft::WRL::ComPtr;

struct Event {
    HANDLE handle = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    ~Event() { if (handle) CloseHandle(handle); }
};

bool buffer(ID3D12Device& device, D3D12_HEAP_TYPE type, D3D12_RESOURCE_STATES state, ComPtr<ID3D12Resource>& out) {
    const D3D12_HEAP_PROPERTIES heap{type, D3D12_CPU_PAGE_PROPERTY_UNKNOWN, D3D12_MEMORY_POOL_UNKNOWN, 1, 1};
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width = 256;
    desc.Height = desc.DepthOrArraySize = desc.MipLevels = 1;
    desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    return SUCCEEDED(device.CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, state, nullptr,
                                                     IID_PPV_ARGS(&out)));
}

// A buffer copy on a COPY queue, as the device's first submission; true when it ran and copied the bytes.
bool copy_on_copy_queue(ID3D12Device& device) {
    ComPtr<ID3D12Resource> upload, readback;
    if (!buffer(device, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ, upload) ||
        !buffer(device, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST, readback))
        return false;
    void* mapped = nullptr;
    const D3D12_RANGE none{0, 0};
    if (FAILED(upload->Map(0, &none, &mapped))) return false;
    for (std::uint32_t i = 0; i < 64; ++i) std::memcpy(static_cast<std::uint8_t*>(mapped) + i * 4, &i, 4);
    upload->Unmap(0, nullptr);

    const D3D12_COMMAND_QUEUE_DESC desc{D3D12_COMMAND_LIST_TYPE_COPY};
    ComPtr<ID3D12CommandQueue> queue;
    ComPtr<ID3D12CommandAllocator> allocator;
    ComPtr<ID3D12GraphicsCommandList> list;
    ComPtr<ID3D12Fence> fence;
    if (FAILED(device.CreateCommandQueue(&desc, IID_PPV_ARGS(&queue))) ||
        FAILED(device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COPY, IID_PPV_ARGS(&allocator))) ||
        FAILED(device.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COPY, allocator.Get(), nullptr,
                                        IID_PPV_ARGS(&list))) ||
        FAILED(device.CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence))))
        return false;
    list->CopyBufferRegion(readback.Get(), 0, upload.Get(), 0, 256);
    if (FAILED(list->Close())) return false;
    ID3D12CommandList* batch[] = {list.Get()};
    queue->ExecuteCommandLists(1, batch);
    const Event event;
    if (!event.handle || FAILED(queue->Signal(fence.Get(), 1)) || FAILED(fence->SetEventOnCompletion(1, event.handle)) ||
        WaitForSingleObject(event.handle, 15000) != WAIT_OBJECT_0 || FAILED(device.GetDeviceRemovedReason()))
        return false;
    if (rt::d3d12::last_execute_result(queue.Get()) != leonida::ReplayResult::success) {
        std::fprintf(stderr, "the COPY queue refused a plain buffer copy\n");
        return false;
    }
    if (FAILED(readback->Map(0, nullptr, &mapped))) return false;
    bool same = true;
    for (std::uint32_t i = 0; i < 64 && same; ++i) {
        std::uint32_t value = 0;
        std::memcpy(&value, static_cast<const std::uint8_t*>(mapped) + i * 4, 4);
        same = value == i;
    }
    readback->Unmap(0, &none);
    if (!same) std::fprintf(stderr, "the COPY queue's buffer copy read back other bytes\n");
    return same;
}

}  // namespace

int main() {
    ComPtr<ID3D12Device> standalone;
    if (FAILED(rt::d3d12::create_vulkan_device_standalone(IID_PPV_ARGS(&standalone)))) {
        std::puts("Vulkan standalone device unavailable");
        return 77;
    }
    const auto* info = rt::d3d12::vulkan_device_info(standalone.Get());
    if (!info || info->copy_queues.empty() || info->copy_family == info->queue_family) {
        std::puts("No separate Vulkan copy family: nothing to check");
        return 77;
    }
    VkQueueFlags flags = 0;
    if (const auto families = reinterpret_cast<PFN_vkGetPhysicalDeviceQueueFamilyProperties>(
            info->get_instance_proc(info->instance, "vkGetPhysicalDeviceQueueFamilyProperties"))) {
        std::uint32_t count = 0;
        families(info->physical, &count, nullptr);
        VkQueueFamilyProperties queues[64]{};
        count = count > 64 ? 64 : count;
        families(info->physical, &count, queues);
        if (info->copy_family < count) flags = queues[info->copy_family].queueFlags;
    }
    if (flags & (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT)) {
        std::puts("The copy family is not transfer-only: nothing to check");
        return 77;
    }
    // As on a GPU without VK_EXT_conditional_rendering: the same Vulkan device and queues, without SetPredication.
    rt::d3d12::VulkanDeviceInfo without = *info;
    without.conditional_rendering = false;
    ComPtr<ID3D12Device> device;
    if (FAILED(rt::d3d12::create_vulkan_device(without, IID_PPV_ARGS(&device)))) {
        std::fputs("create_vulkan_device without conditional rendering failed\n", stderr);
        return 1;
    }
    if (!copy_on_copy_queue(*device.Get())) return 1;
    std::printf("COPY queue on transfer-only family %u, device without conditional rendering: first submission ran\n",
                info->copy_family);
    return 0;
}
