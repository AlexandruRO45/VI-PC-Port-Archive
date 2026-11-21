// IDXGISwapChain3 over leonida::VulkanSwapchain, for D3D12: what IDXGIFactory2::CreateSwapChainForHwnd returns
// when its "device" is one of the layer's command queues (docs/d3d12_vulkan.md section 4.5, milestone M0).
//
// Implemented: Present/Present1, GetBuffer, GetCurrentBackBufferIndex, GetDesc/GetDesc1, GetHwnd,
// GetFullscreenState (always windowed), GetLastPresentCount, GetDevice (the queue), GetParent (the factory),
// private data. Every other
// method is a generated stub that reports its name once. Known differences from DXGI, each logged when hit:
// - the present mode is FIFO, so every sync interval presents with vsync;
// - the buffers take the window's client size (the Win32 surface fixes it), and GetDesc1 reports that size;
// - ResizeBuffers is not implemented, so a resized window fails the next Present;
// - creation fails if the driver returns more images than BufferCount.
#pragma once

#include <windows.h>
#include <d3d12.h>
#include <dxgi1_4.h>

#include "../src/vulkan_swapchain.h"

namespace rt::dxgi {

// The Vulkan format for a flip-model DXGI back-buffer format; VK_FORMAT_UNDEFINED when DXGI would refuse it.
VkFormat vulkan_format(DXGI_FORMAT format);

// IDXGIFactory2::CreateSwapChainForHwnd for a queue made by rt::d3d12::create_command_queue on `context`.
// `target.queue_family` must be that queue's family. `parent` is what GetParent returns (the factory), if any.
HRESULT create_swap_chain_for_hwnd(leonida::VulkanContext& context, const leonida::PresentTarget& target,
                                   ID3D12CommandQueue* queue, HWND window, const DXGI_SWAP_CHAIN_DESC1& desc,
                                   REFIID riid, void** out, IUnknown* parent = nullptr);

// For tests and traces: the Vulkan image behind a back buffer (VK_NULL_HANDLE if not ours or out of range).
VkImage vulkan_image(IDXGISwapChain* swapchain, UINT buffer);

}  // namespace rt::dxgi
