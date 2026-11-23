#pragma once

// COM wrappers for IDXGIFactory7, IDXGIAdapter3, IDXGIOutput6, and IDXGISwapChain3 backed by VulkanSwapchain (dxgi_swapchain.h).
// Keeps the host dxgi.dll out of the Vulkan presentation path (docs/d3d12_vulkan.md section 3.1 & 4).

#include <windows.h>
#include <dxgi1_6.h>
#include <d3d12.h>

#include <memory>

#include "dxgi_swapchain.h"

namespace rt::dxgi {

class VideoMemoryBudgetMonitor;
std::shared_ptr<VideoMemoryBudgetMonitor> make_video_memory_budget_monitor(
    PFN_vkGetInstanceProcAddr gipa, VkInstance instance, VkPhysicalDevice physical);

// Creates an IDXGIFactory4 COM object whose CreateSwapChainForHwnd creates a VulkanSwapchain
// and registers its back-buffer images with `context` (keyed by the returned ID3D12Resource* pointer).
HRESULT create_factory(leonida::VulkanContext& context, PFN_vkGetInstanceProcAddr gipa, VkInstance instance,
                       VkPhysicalDevice physical, PFN_vkGetDeviceProcAddr gdpa, VkQueue queue,
                       std::uint32_t queue_family, const std::shared_ptr<VideoMemoryBudgetMonitor>& budget,
                       REFIID riid, void** out);

}  // namespace rt::dxgi
