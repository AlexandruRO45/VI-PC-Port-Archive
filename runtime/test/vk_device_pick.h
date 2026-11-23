// Chooses among enumerated Vulkan physical devices for the test harnesses.
//
// Windows can expose Mesa's Dozen (a Vulkan-on-D3D12 translation layer installed by the "OpenCL, OpenGL and Vulkan
// Compatibility Pack") next to the real GPU. Dozen reports VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU, so the old
// "the last discrete device wins" loop picked the wrapper over the native driver and produced failures that looked
// like translator bugs (#915). Prefer a native ICD, then discrete > integrated > virtual > CPU > other, and keep the
// first device on a tie.
#pragma once

#include <vulkan/vulkan.h>

#include "../src/vulkan_device_filter.h"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace leonida::vk_test {

// Microsoft is the vendor of the D3D12 translation layers Windows adds (Dozen and the Basic Render Driver); no
// native GPU driver uses it. Checking the vendor avoids depending on VkPhysicalDeviceDriverProperties.
inline constexpr std::uint32_t microsoft_vendor_id = leonida::microsoft_vulkan_vendor_id;

static_assert(VK_PHYSICAL_DEVICE_TYPE_OTHER == 0 && VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU == 1 &&
                  VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU == 2 && VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU == 3 &&
                  VK_PHYSICAL_DEVICE_TYPE_CPU == 4,
              "vulkan_device_rank takes VkPhysicalDeviceType by value");

// The runtime's rank (runtime/src/vulkan_device_filter.h), so the harnesses and the runtime take the same GPU.
inline int device_rank(VkPhysicalDeviceType type, std::uint32_t vendor_id) {
    return leonida::vulkan_device_rank(static_cast<int>(type), vendor_id);
}

// The Vulkan version the tools that chain VkPhysicalDeviceVulkan11/12Features into VkPhysicalDeviceFeatures2 or
// VkDeviceCreateInfo need. A 1.0 or 1.1 device does not know those structures (a validation error, or a rejected
// vkCreateDevice), and a Vulkan 1.0 instance hands out vkGetPhysicalDeviceFeatures2 as null. The Vulkan 1.3
// structure is chained only for a device at 1.3 (props.apiVersion >= VK_API_VERSION_1_3), as
// leonida::Vulkan13Features does; a 1.2 device gets the 1.2 structures alone.
inline constexpr std::uint32_t tool_min_api_version = VK_API_VERSION_1_2;

// Index of the best candidate, or candidates.size() when there are none. Ties keep the earlier device. A device
// whose apiVersion is below min_api_version is not a candidate (0, the default, accepts every device).
inline std::size_t pick_device(const std::vector<VkPhysicalDeviceProperties>& candidates,
                               std::uint32_t min_api_version = 0) {
    std::size_t best = candidates.size();
    int best_rank = -1;
    for (std::size_t i = 0; i < candidates.size(); ++i) {
        if (!leonida::vulkan_device_wanted(candidates[i].deviceName)) continue;
        if (candidates[i].apiVersion < min_api_version) continue;
        const int rank = device_rank(candidates[i].deviceType, candidates[i].vendorID);
        if (rank > best_rank) {
            best = i;
            best_rank = rank;
        }
    }
    return best;
}

// Why no device was chosen, naming LEONIDA_VK_DEVICE when its filter excluded every device (otherwise the list was
// empty). The tools call this instead of dereferencing a null device.
inline void report_no_device() {
    const std::string filter = leonida::vulkan_device_filter();
    if (!filter.empty()) {
        std::fprintf(stderr, "LEONIDA_VK_DEVICE=%s: no Vulkan physical device matches\n", filter.c_str());
    } else {
        std::fprintf(stderr, "no Vulkan physical device found\n");
    }
}

// The exit code of a tool whose pick_device(candidates, min_api_version) found nothing, after printing why: 77
// (the device cannot run the test: a skip, as for a missing feature) when a device LEONIDA_VK_DEVICE leaves in play
// is older than min_api_version, otherwise 1 as report_no_device() above.
inline int report_no_device(const std::vector<VkPhysicalDeviceProperties>& candidates, std::uint32_t min_api_version) {
    std::uint32_t newest = 0;
    bool too_old = false;
    for (const auto& c : candidates) {
        if (!leonida::vulkan_device_wanted(c.deviceName)) continue;
        too_old = true;
        if (c.apiVersion > newest) newest = c.apiVersion;
    }
    if (!too_old || newest >= min_api_version) {
        report_no_device();
        return 1;
    }
    std::fprintf(stderr, "unsupported: no Vulkan %u.%u device (the newest is %u.%u)\n",
                 VK_API_VERSION_MAJOR(min_api_version), VK_API_VERSION_MINOR(min_api_version),
                 VK_API_VERSION_MAJOR(newest), VK_API_VERSION_MINOR(newest));
    return 77;
}

// A Vulkan 1.0 instance has no core 1.1 entry points: vkGetPhysicalDeviceFeatures2 and vkGetPhysicalDeviceProperties2
// come back null from vkGetInstanceProcAddr, and calling one crashes. Prints why and returns 77 (a skip).
inline int report_no_core11_entry_points() {
    std::fprintf(stderr, "unsupported: the Vulkan instance has no vkGetPhysicalDeviceFeatures2 (a Vulkan 1.0 loader)\n");
    return 77;
}

}  // namespace leonida::vk_test
