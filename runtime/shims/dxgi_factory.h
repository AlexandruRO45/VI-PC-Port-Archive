// The DXGI factory and adapter the game sees instead of the host's (docs/d3d12_vulkan.md section 4.5): one factory
// with one adapter that describes the Vulkan device the layer runs on, and swapchains created on the layer's own
// command queues.
//
// Factory (IDXGIFactory5): EnumAdapters/EnumAdapters1 (adapter 0 only), EnumAdapterByLuid, CreateSwapChain,
// CreateSwapChainForHwnd, MakeWindowAssociation/GetWindowAssociation, IsCurrent, IsWindowedStereoEnabled (FALSE),
// GetCreationFlags, CheckFeatureSupport (tearing: FALSE, the swapchain presents FIFO).
// Adapter (IDXGIAdapter1): GetDesc/GetDesc1 from the Vulkan device (name, vendor/device ID, LUID, device-local and
// shared memory), EnumOutputs (none: windowed only), CheckInterfaceSupport (unsupported, as for D3D12 on Windows),
// GetParent (the factory).
// Everything else is a generated stub that reports its name once. A fullscreen request is logged and presented
// windowed.
#pragma once

#include <windows.h>
#include <dxgi1_5.h>

#include "../src/vulkan_swapchain.h"

namespace rt::dxgi {

// The host side the factory describes and presents on. The instance must have VK_KHR_surface and
// VK_KHR_win32_surface; the context's device VK_KHR_swapchain.
struct Host {
    leonida::VulkanContext* context = nullptr;
    PFN_vkGetInstanceProcAddr get_instance_proc = nullptr;
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physical = VK_NULL_HANDLE;
};

// CreateDXGIFactory2(flags, riid, out) over `host`. The host must outlive every object the factory makes.
HRESULT create_factory(const Host& host, UINT flags, REFIID riid, void** out);

}  // namespace rt::dxgi
