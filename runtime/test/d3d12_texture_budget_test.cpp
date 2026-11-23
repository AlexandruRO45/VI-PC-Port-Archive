// Run D3D12 committed texture creation with deterministic budgets and real Vulkan allocations.
#include "vk_test_device.h"
#include "../shims/d3d12_com.h"
#include "../src/vulkan_memory.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <wrl/client.h>

namespace {

using Microsoft::WRL::ComPtr;
constexpr UINT kExtent = 64;
constexpr auto kLocal = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

PFN_vkGetInstanceProcAddr g_real_proc = nullptr;
VkPhysicalDeviceMemoryProperties g_memory{};
bool g_local_full = false;

VKAPI_ATTR void VKAPI_CALL test_memory_properties(VkPhysicalDevice, VkPhysicalDeviceMemoryProperties* memory) {
    *memory = g_memory;
}

VKAPI_ATTR void VKAPI_CALL test_memory_properties2(VkPhysicalDevice, VkPhysicalDeviceMemoryProperties2* memory) {
    memory->memoryProperties = g_memory;
    auto* extension = static_cast<VkBaseOutStructure*>(memory->pNext);
    if (!extension || extension->sType != VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT) return;
    auto* budget = reinterpret_cast<VkPhysicalDeviceMemoryBudgetPropertiesEXT*>(extension);
    for (std::uint32_t i = 0; i < g_memory.memoryHeapCount; ++i) {
        const auto& heap = g_memory.memoryHeaps[i];
        budget->heapBudget[i] = heap.size;
        budget->heapUsage[i] = g_local_full && (heap.flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) ? heap.size : 0;
    }
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL test_instance_proc(VkInstance instance, const char* name) {
    if (std::strcmp(name, "vkGetPhysicalDeviceMemoryProperties") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(test_memory_properties);
    if (std::strcmp(name, "vkGetPhysicalDeviceMemoryProperties2") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(test_memory_properties2);
    return g_real_proc(instance, name);
}

bool expect_placement(ID3D12Device* device, UINT16 layers, D3D12_RESOURCE_FLAGS flags,
                      bool device_local, const char* label) {
    const D3D12_HEAP_PROPERTIES heap{D3D12_HEAP_TYPE_DEFAULT};
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    desc.Width = desc.Height = kExtent;
    desc.DepthOrArraySize = layers;
    desc.MipLevels = 1;
    desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    desc.SampleDesc.Count = 1;
    desc.Flags = flags;
    ComPtr<ID3D12Resource> texture;
    const HRESULT result = device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc,
                                                            D3D12_RESOURCE_STATE_COMMON, nullptr,
                                                            IID_PPV_ARGS(&texture));
    if (FAILED(result)) {
        std::fprintf(stderr, "%s: texture creation failed (0x%08lx)\n", label, static_cast<unsigned long>(result));
        return false;
    }
    const bool actual_local = (rt::d3d12::vulkan_memory_flags(texture.Get()) & kLocal) != 0;
    std::printf("%s: %s (%s memory)\n", label, actual_local == device_local ? "pass" : "fail",
                actual_local ? "device-local" : "system");
    return actual_local == device_local;
}

}  // namespace

int main() {
    vktest::Device vk;
    const auto status = vk.initialize();
    if (status == vktest::DeviceStatus::unavailable) return 77;
    if (status != vktest::DeviceStatus::ready) return 1;

    const VkImage probe = vk.image(VK_FORMAT_R8G8B8A8_UNORM, kExtent, kExtent, 2, VK_IMAGE_USAGE_SAMPLED_BIT);
    if (!probe) return 1;
    VkMemoryRequirements requirements{};
    vk.fn<PFN_vkGetImageMemoryRequirements>("vkGetImageMemoryRequirements")(vk.device, probe, &requirements);
    g_real_proc = vk.instance_proc;
    vk.instance_fn<PFN_vkGetPhysicalDeviceMemoryProperties>("vkGetPhysicalDeviceMemoryProperties")(vk.physical,
                                                                                                  &g_memory);
    bool local = false;
    bool non_local = false;
    for (std::uint32_t i = 0; i < g_memory.memoryTypeCount; ++i) {
        if (!(requirements.memoryTypeBits & (1u << i))) continue;
        if (leonida::is_device_local_memory_type(g_memory, i)) {
            local = true;
        } else {
            // Some drivers offer non-local image memory without HOST_VISIBLE. Model a host-visible choice in the
            // policy's inputs only. Every allocation and bind uses the real compatible index and never maps it.
            g_memory.memoryTypes[i].propertyFlags |= VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
            non_local = true;
        }
    }
    if (!local || !non_local) {
        std::puts("skip: no compatible device-local and non-local memory types for optimal RGBA8 images");
        return 77;
    }

    rt::d3d12::VulkanDeviceInfo info{};
    info.get_instance_proc = test_instance_proc;
    info.instance = vk.instance;
    info.physical = vk.physical;
    info.device = vk.device;
    info.queue = vk.queue;
    info.queue_family = vk.family;
    info.memory_budget = true;
    ComPtr<ID3D12Device> device;
    if (FAILED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&device)))) return 1;

    if (!expect_placement(device.Get(), 1, D3D12_RESOURCE_FLAG_NONE, true, "within budget")) return 1;
    g_local_full = true;
    if (!expect_placement(device.Get(), 1, D3D12_RESOURCE_FLAG_NONE, false, "sampled 2D over budget") ||
        !expect_placement(device.Get(), 2, D3D12_RESOURCE_FLAG_NONE, false, "sampled 2D array over budget") ||
        !expect_placement(device.Get(), 1, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET, true, "render target over budget") ||
        !expect_placement(device.Get(), 1, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, true, "UAV over budget"))
        return 1;
    return vk.validation_errors() == 0 ? 0 : 1;
}
