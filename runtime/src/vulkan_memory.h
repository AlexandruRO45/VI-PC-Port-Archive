#pragma once

// Memory-type policy for the D3D12 layer that depends only on VkPhysicalDeviceMemoryProperties, so it can be checked
// without a GPU (d3d12_state_map_test) on the heap layouts real cards report.

#ifndef VK_NO_PROTOTYPES
#define VK_NO_PROTOTYPES
#endif
#include <vulkan/vulkan.h>

#include <algorithm>
#include <cstdint>
#include <vector>

namespace leonida {

// Without Resizable BAR the CPU sees at most a 256 MiB window of VRAM: the DEVICE_LOCAL | HOST_VISIBLE type sits in a
// heap of that size or smaller (256 MiB on AMD and on RADV, 214 MiB on an RTX 3090 with driver 616.56). Anything
// larger is Resizable BAR (or Smart Access Memory), or a GPU whose memory is all device-local and host-visible.
inline constexpr VkDeviceSize kBarWindowBytes = VkDeviceSize(256) << 20;

// True when the device has DEVICE_LOCAL | HOST_VISIBLE | HOST_COHERENT memory in a device-local heap larger than
// the 256 MiB BAR window (D3D-96): then CPU-written memory that the GPU reads, such as an upload heap, can live in
// VRAM without exhausting a small window that every such allocation would share.
inline bool resizable_bar(const VkPhysicalDeviceMemoryProperties& memory) {
    constexpr VkMemoryPropertyFlags wanted = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT |
                                             VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                             VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    for (std::uint32_t i = 0; i < memory.memoryTypeCount && i < VK_MAX_MEMORY_TYPES; ++i) {
        const VkMemoryType& type = memory.memoryTypes[i];
        if ((type.propertyFlags & wanted) != wanted || type.heapIndex >= memory.memoryHeapCount) continue;
        const VkMemoryHeap& heap = memory.memoryHeaps[type.heapIndex];
        if ((heap.flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) && heap.size > kBarWindowBytes) return true;
    }
    return false;
}

// True when `index` is a DEVICE_LOCAL | HOST_VISIBLE memory type backed by a small (<= 256 MiB) device-local BAR
// window. Without Resizable BAR, placing staging or general GPU buffers there ahead of system RAM or main VRAM quickly
// exhausts the tiny 214-256 MiB window.
inline bool small_bar_type(const VkPhysicalDeviceMemoryProperties& memory, std::uint32_t index) {
    if (index >= memory.memoryTypeCount || index >= VK_MAX_MEMORY_TYPES) return false;
    const VkMemoryType& type = memory.memoryTypes[index];
    constexpr VkMemoryPropertyFlags bar_flags =
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
    if ((type.propertyFlags & bar_flags) != bar_flags || type.heapIndex >= memory.memoryHeapCount) return false;
    const VkMemoryHeap& heap = memory.memoryHeaps[type.heapIndex];
    return (heap.flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) != 0 && heap.size <= kBarWindowBytes;
}

// The memory types to try for an allocation of `size` bytes, best first: those with `required | preferred`, then
// those with only `required`. Within each tier, non-small-BAR types come before small-BAR (<= 256 MiB) types so host
// staging buffers avoid the 214-256 MiB BAR1 window even when a driver lists BAR1 first, and GPU buffers prefer main
// VRAM over BAR1. A type whose heap is smaller than the allocation is left out, as the allocation would break
// VUID-vkAllocateMemory-pAllocateInfo-01713 (some drivers still return VK_SUCCESS). A heap that is big enough but
// already partly used is left to the driver: a failed vkAllocateMemory moves on to the next type.
inline std::vector<std::uint32_t> memory_type_order(const VkPhysicalDeviceMemoryProperties& memory,
                                                    std::uint32_t type_bits, VkMemoryPropertyFlags required,
                                                    VkMemoryPropertyFlags preferred, VkDeviceSize size) {
    std::vector<std::uint32_t> order;
    for (const VkMemoryPropertyFlags flags : {required | preferred, required}) {
        for (const bool allow_small_bar : {false, true}) {
            for (std::uint32_t i = 0; i < memory.memoryTypeCount && i < VK_MAX_MEMORY_TYPES; ++i) {
                const VkMemoryType& type = memory.memoryTypes[i];
                if (!(type_bits & (1u << i)) || (type.propertyFlags & flags) != flags) continue;
                if (type.heapIndex >= memory.memoryHeapCount || memory.memoryHeaps[type.heapIndex].size < size) continue;
                if (small_bar_type(memory, i) != allow_small_bar) continue;
                if (std::find(order.begin(), order.end(), i) == order.end()) order.push_back(i);
            }
        }
    }
    return order;
}

// Allocates device memory for `requirements`, trying each candidate from memory_type_order() until vkAllocateMemory
// succeeds. Returns VK_NULL_HANDLE when no candidate fits or succeeds. When non-null, `chosen_flags` receives the
// selected type's propertyFlags and `chosen_type` receives its memoryTypeIndex. `p_next`, when non-null, is attached
// to VkMemoryAllocateInfo::pNext (for example VkMemoryAllocateFlagsInfo with VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT).
inline VkDeviceMemory allocate_memory(PFN_vkAllocateMemory vk_allocate_memory, VkDevice device,
                                      const VkPhysicalDeviceMemoryProperties& memory,
                                      const VkMemoryRequirements& requirements, VkMemoryPropertyFlags required,
                                      VkMemoryPropertyFlags preferred, const void* p_next = nullptr,
                                      VkMemoryPropertyFlags* chosen_flags = nullptr,
                                      std::uint32_t* chosen_type = nullptr) {
    if (!vk_allocate_memory) return VK_NULL_HANDLE;
    for (const std::uint32_t i :
         memory_type_order(memory, requirements.memoryTypeBits, required, preferred, requirements.size)) {
        const VkMemoryAllocateInfo info{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, p_next, requirements.size, i};
        VkDeviceMemory out = VK_NULL_HANDLE;
        if (vk_allocate_memory(device, &info, nullptr, &out) == VK_SUCCESS) {
            if (chosen_flags) *chosen_flags = memory.memoryTypes[i].propertyFlags;
            if (chosen_type) *chosen_type = i;
            return out;
        }
    }
    return VK_NULL_HANDLE;
}

// D3D-299: true when an allocation that preferred DEVICE_LOCAL without needing host access (a texture, a default-heap
// buffer outside the arena) got a type without it, so the GPU now reads it from system memory. A host-visible
// allocation (the arena, an upload heap) in system memory is the memory type policy without Resizable BAR, not this.
inline bool spilled_to_host(VkMemoryPropertyFlags required, VkMemoryPropertyFlags preferred,
                            VkMemoryPropertyFlags chosen) {
    return (preferred & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) && !(required & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) &&
           !(chosen & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
}

}  // namespace leonida
