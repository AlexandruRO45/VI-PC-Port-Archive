#pragma once

// A Vulkan swapchain for one window, without COM: the presentation side of the D3D12 layer (docs/d3d12_vulkan.md
// section 4.5). The DXGI swapchain in shims/dxgi_swapchain.cpp is a thin wrapper over it.
//
// Acquire is synchronous: acquire_next() waits on the CPU (a VkFence) until the presentation engine hands an image
// back, so the back-buffer index is known before the game records a frame, and the game's own submissions need no
// acquire semaphore. That costs latency, not correctness; an asynchronous acquire is a later optimisation.
// Present waits on one binary semaphore per image, which the queue signals after everything submitted before the
// Present (VulkanQueue::present). A semaphore is reused only once its image has been acquired again, which is
// what the presentation engine guarantees safe.
//
// Needs VK_KHR_surface and VK_KHR_win32_surface on the instance and VK_KHR_swapchain on the device.

#ifndef VK_NO_PROTOTYPES
#define VK_NO_PROTOTYPES
#endif
#include "vulkan_backend.h"

#include <windows.h>
#include <vulkan/vulkan_win32.h>

#include <algorithm>
#include <cstdint>
#include <vector>

namespace leonida {

// Where presentation happens: the instance, the device's physical device, and the queue family the D3D12 queue
// runs on (it must be able to present to the window).
struct PresentTarget {
    PFN_vkGetInstanceProcAddr get_instance_proc = nullptr;
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physical = VK_NULL_HANDLE;
    std::uint32_t queue_family = 0;
};

struct SwapchainDesc {
    std::uint32_t width = 0;  // used only when the surface leaves the size to the swapchain; 0: the client size
    std::uint32_t height = 0;
    VkFormat format = VK_FORMAT_B8G8R8A8_UNORM;
    std::uint32_t buffer_count = 2;
};

class VulkanSwapchain {
public:
    VulkanSwapchain(VulkanContext& context, const PresentTarget& target, HWND window, const SwapchainDesc& desc)
        : context_(context), instance_(target.instance) {
        create(target, window, desc);
    }

    // The caller makes the queue idle first: nothing may still wait on the semaphores.
    ~VulkanSwapchain() {
        const VkDevice device = context_.device();
        for (VkSemaphore semaphore : ready_) context_.vk.vkDestroySemaphore(device, semaphore, nullptr);
        if (fence_) destroy_fence_(device, fence_, nullptr);
        if (swapchain_) destroy_swapchain_(device, swapchain_, nullptr);
        if (surface_) destroy_surface_(instance_, surface_, nullptr);
    }

    VulkanSwapchain(const VulkanSwapchain&) = delete;
    VulkanSwapchain& operator=(const VulkanSwapchain&) = delete;

    bool ok() const { return error_ == nullptr; }
    const char* error() const { return error_; }  // why creation failed

    std::uint32_t image_count() const { return std::uint32_t(images_.size()); }
    VkImage image(std::uint32_t index) const { return images_[index]; }
    VkFormat format() const { return format_; }
    VkExtent2D extent() const { return extent_; }
    VkImageUsageFlags usage() const { return usage_; }
    VkSwapchainKHR handle() const { return swapchain_; }
    PFN_vkQueuePresentKHR queue_present() const { return queue_present_; }

    // The image the next frame renders to, and the semaphore its Present waits on.
    std::uint32_t current() const { return current_; }
    VkSemaphore ready(std::uint32_t index) const { return ready_[index]; }

    // Takes the next image from the presentation engine and waits until it may be written.
    VkResult acquire_next() {
        std::uint32_t index = 0;
        const VkResult result = acquire_(context_.device(), swapchain_, UINT64_MAX, VK_NULL_HANDLE, fence_, &index);
        if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) return result;
        const VkResult waited = wait_fences_(context_.device(), 1, &fence_, VK_TRUE, UINT64_MAX);
        reset_fences_(context_.device(), 1, &fence_);
        if (waited != VK_SUCCESS) return waited;
        current_ = index;
        return result;
    }

private:
    void create(const PresentTarget& target, HWND window, const SwapchainDesc& desc) {
        const auto instance_fn = [&](const char* name) { return target.get_instance_proc(instance_, name); };
        const auto create_surface =
            reinterpret_cast<PFN_vkCreateWin32SurfaceKHR>(instance_fn("vkCreateWin32SurfaceKHR"));
        destroy_surface_ = reinterpret_cast<PFN_vkDestroySurfaceKHR>(instance_fn("vkDestroySurfaceKHR"));
        const auto supported = reinterpret_cast<PFN_vkGetPhysicalDeviceSurfaceSupportKHR>(
            instance_fn("vkGetPhysicalDeviceSurfaceSupportKHR"));
        const auto capabilities_of = reinterpret_cast<PFN_vkGetPhysicalDeviceSurfaceCapabilitiesKHR>(
            instance_fn("vkGetPhysicalDeviceSurfaceCapabilitiesKHR"));
        const auto formats_of = reinterpret_cast<PFN_vkGetPhysicalDeviceSurfaceFormatsKHR>(
            instance_fn("vkGetPhysicalDeviceSurfaceFormatsKHR"));
        if (!create_surface || !destroy_surface_ || !supported || !capabilities_of || !formats_of) {
            error_ = "the instance lacks VK_KHR_surface or VK_KHR_win32_surface";
            return;
        }
        const auto get_device_proc =
            reinterpret_cast<PFN_vkGetDeviceProcAddr>(instance_fn("vkGetDeviceProcAddr"));
        const VkDevice device = context_.device();
        const auto device_fn = [&](const char* name) { return get_device_proc(device, name); };
        const auto create_swapchain = reinterpret_cast<PFN_vkCreateSwapchainKHR>(device_fn("vkCreateSwapchainKHR"));
        destroy_swapchain_ = reinterpret_cast<PFN_vkDestroySwapchainKHR>(device_fn("vkDestroySwapchainKHR"));
        const auto images_of = reinterpret_cast<PFN_vkGetSwapchainImagesKHR>(device_fn("vkGetSwapchainImagesKHR"));
        acquire_ = reinterpret_cast<PFN_vkAcquireNextImageKHR>(device_fn("vkAcquireNextImageKHR"));
        queue_present_ = reinterpret_cast<PFN_vkQueuePresentKHR>(device_fn("vkQueuePresentKHR"));
        const auto create_fence = reinterpret_cast<PFN_vkCreateFence>(device_fn("vkCreateFence"));
        destroy_fence_ = reinterpret_cast<PFN_vkDestroyFence>(device_fn("vkDestroyFence"));
        wait_fences_ = reinterpret_cast<PFN_vkWaitForFences>(device_fn("vkWaitForFences"));
        reset_fences_ = reinterpret_cast<PFN_vkResetFences>(device_fn("vkResetFences"));
        if (!create_swapchain || !destroy_swapchain_ || !images_of || !acquire_ || !queue_present_ ||
            !create_fence || !destroy_fence_ || !wait_fences_ || !reset_fences_) {
            error_ = "the device lacks VK_KHR_swapchain";
            return;
        }

        VkWin32SurfaceCreateInfoKHR surface_info{VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR};
        surface_info.hinstance = GetModuleHandleW(nullptr);
        surface_info.hwnd = window;
        if (create_surface(instance_, &surface_info, nullptr, &surface_) != VK_SUCCESS) {
            error_ = "vkCreateWin32SurfaceKHR failed";
            return;
        }
        VkBool32 can_present = VK_FALSE;
        supported(target.physical, target.queue_family, surface_, &can_present);
        if (!can_present) {
            error_ = "the queue family cannot present to this window";
            return;
        }

        VkSurfaceCapabilitiesKHR caps{};
        if (capabilities_of(target.physical, surface_, &caps) != VK_SUCCESS) {
            error_ = "vkGetPhysicalDeviceSurfaceCapabilitiesKHR failed";
            return;
        }
        // On Win32 the surface fixes the extent to the client area; otherwise use the requested size.
        if (caps.currentExtent.width != UINT32_MAX) {
            extent_ = caps.currentExtent;
        } else {
            RECT client{};
            GetClientRect(window, &client);
            extent_.width = desc.width ? desc.width : std::uint32_t(client.right - client.left);
            extent_.height = desc.height ? desc.height : std::uint32_t(client.bottom - client.top);
            extent_.width = std::clamp(extent_.width, caps.minImageExtent.width, caps.maxImageExtent.width);
            extent_.height = std::clamp(extent_.height, caps.minImageExtent.height, caps.maxImageExtent.height);
        }
        if (!extent_.width || !extent_.height) {
            error_ = "the window has no client area (minimised?)";
            return;
        }

        std::uint32_t count = 0;
        formats_of(target.physical, surface_, &count, nullptr);
        std::vector<VkSurfaceFormatKHR> formats(count);
        formats_of(target.physical, surface_, &count, formats.data());
        const bool has_format = std::any_of(formats.begin(), formats.end(), [&](const VkSurfaceFormatKHR& f) {
            return f.format == desc.format && f.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
        });
        if (!has_format) {
            error_ = "the surface does not support the requested format";
            return;
        }
        format_ = desc.format;

        // Clears go through vkCmdClearColorImage, so TRANSFER_DST is required; TRANSFER_SRC is for read-back.
        constexpr VkImageUsageFlags required = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        if ((caps.supportedUsageFlags & required) != required) {
            error_ = "the surface does not allow colour-attachment and transfer-destination images";
            return;
        }
        usage_ = required | (caps.supportedUsageFlags & VK_IMAGE_USAGE_TRANSFER_SRC_BIT);

        std::uint32_t min_images = std::max(desc.buffer_count, caps.minImageCount);
        if (caps.maxImageCount) min_images = std::min(min_images, caps.maxImageCount);
        VkCompositeAlphaFlagBitsKHR alpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
        if (!(caps.supportedCompositeAlpha & alpha)) {
            for (std::uint32_t bit = 1; bit; bit <<= 1)
                if (caps.supportedCompositeAlpha & bit) {
                    alpha = VkCompositeAlphaFlagBitsKHR(bit);
                    break;
                }
        }

        VkSwapchainCreateInfoKHR info{VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR};
        info.surface = surface_;
        info.minImageCount = min_images;
        info.imageFormat = format_;
        info.imageColorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
        info.imageExtent = extent_;
        info.imageArrayLayers = 1;
        info.imageUsage = usage_;
        info.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
        info.preTransform = caps.currentTransform;
        info.compositeAlpha = alpha;
        info.presentMode = VK_PRESENT_MODE_FIFO_KHR;  // always available; DXGI sync interval 1
        info.clipped = VK_TRUE;
        if (create_swapchain(device, &info, nullptr, &swapchain_) != VK_SUCCESS) {
            error_ = "vkCreateSwapchainKHR failed";
            return;
        }
        images_of(device, swapchain_, &count, nullptr);
        images_.resize(count);
        images_of(device, swapchain_, &count, images_.data());

        VkSemaphoreCreateInfo semaphore_info{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};  // binary
        ready_.resize(count, VK_NULL_HANDLE);
        for (VkSemaphore& semaphore : ready_) {
            if (context_.vk.vkCreateSemaphore(device, &semaphore_info, nullptr, &semaphore) != VK_SUCCESS) {
                error_ = "vkCreateSemaphore failed";
                return;
            }
        }
        VkFenceCreateInfo fence_info{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        if (create_fence(device, &fence_info, nullptr, &fence_) != VK_SUCCESS) {
            error_ = "vkCreateFence failed";
            return;
        }
        const VkResult first = acquire_next();
        if (first != VK_SUCCESS && first != VK_SUBOPTIMAL_KHR) error_ = "the first vkAcquireNextImageKHR failed";
    }

    VulkanContext& context_;
    VkInstance instance_;
    const char* error_ = nullptr;
    VkSurfaceKHR surface_ = VK_NULL_HANDLE;
    VkSwapchainKHR swapchain_ = VK_NULL_HANDLE;
    VkFence fence_ = VK_NULL_HANDLE;
    std::vector<VkImage> images_;
    std::vector<VkSemaphore> ready_;
    VkFormat format_ = VK_FORMAT_UNDEFINED;
    VkExtent2D extent_{};
    VkImageUsageFlags usage_ = 0;
    std::uint32_t current_ = 0;

    PFN_vkDestroySurfaceKHR destroy_surface_ = nullptr;
    PFN_vkDestroySwapchainKHR destroy_swapchain_ = nullptr;
    PFN_vkAcquireNextImageKHR acquire_ = nullptr;
    PFN_vkQueuePresentKHR queue_present_ = nullptr;
    PFN_vkDestroyFence destroy_fence_ = nullptr;
    PFN_vkWaitForFences wait_fences_ = nullptr;
    PFN_vkResetFences reset_fences_ = nullptr;
};

}  // namespace leonida
