#pragma once

// COM wrappers for IDXGIFactory7, IDXGIAdapter3, IDXGIOutput6, and IDXGISwapChain3 backed by VulkanSwapchain (dxgi_swapchain.h).
// Keeps the host dxgi.dll out of the Vulkan presentation path (docs/d3d12_vulkan.md section 3.1 & 4).

#include <windows.h>
#include <dxgi1_6.h>
#include <d3d12.h>

#include "dxgi_swapchain.h"

namespace rt::dxgi {

// The adapter is enumerated from this instance. Swapchains use the supplied command queue's Vulkan objects.
HRESULT create_factory(IUnknown* instance_owner, PFN_vkGetInstanceProcAddr gipa, VkInstance instance,
                       VkPhysicalDevice physical, REFIID riid, void** out);

}  // namespace rt::dxgi
