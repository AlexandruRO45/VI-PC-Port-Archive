// Swapchain presentation on the renderer's own Vulkan objects (shared-device WSI).
//
// The D3D12 layer renders on one VkDevice; this presenter builds the VkSurfaceKHR and VkSwapchainKHR on that same
// instance and device, so a rendered image can be moved into the swapchain with vkCmdCopyImage (or vkCmdBlitImage
// when the formats differ) and presented without anything travelling through the host. The instance must have
// VK_KHR_surface and VK_KHR_win32_surface enabled and the device VK_KHR_swapchain, which is what
// create_vulkan_device_standalone(..., presentable = true) sets up.
//
// The source image's layout is the caller's business: present() moves it to TRANSFER_SRC_OPTIMAL and back to the
// layout it was given, so the layer's layout tracker stays correct (it cannot see barriers issued outside the layer).
//
// Every Vulkan entry point is resolved from the instance/device the caller passes, so nothing here links against
// vulkan-1.lib: the layer loads Vulkan at run time, and a game process must not gain a static import.
#pragma once

#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <vulkan/vulkan.h>
#include <vulkan/vulkan_win32.h>  // VkWin32SurfaceCreateInfoKHR and its PFN types

#include <cstddef>
#include <cstdint>
#include <vector>

namespace leonida::wsi {

// What a presenter shares with the renderer.
struct Gpu {
    PFN_vkGetInstanceProcAddr get_instance_proc = nullptr;
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physical = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;  // the device's graphics queue, which the D3D12 queues also submit to
    std::uint32_t queue_family = 0;
};

class Presenter {
public:
    Presenter() = default;
    ~Presenter();
    Presenter(const Presenter&) = delete;
    Presenter& operator=(const Presenter&) = delete;

    // Creates the surface for `window` and a FIFO swapchain. `format` is used when the surface offers it, otherwise
    // the surface's first format is taken; `extent` is used when the surface's current extent is undefined (a window
    // the driver does not own). Fails (after logging) when the queue family cannot present on the surface.
    bool create(const Gpu& gpu, HWND window, VkFormat format, VkExtent2D extent);
    bool ok() const { return swapchain_ != VK_NULL_HANDLE; }

    VkFormat format() const { return format_; }
    VkExtent2D extent() const { return extent_; }
    std::uint32_t image_count() const { return std::uint32_t(images_.size()); }
    int frames_presented() const { return frames_; }
    VkDevice device() const { return gpu_.device; }
    VkQueue queue() const { return gpu_.queue; }

    // Copies `source` (same device, in `source_layout`) into the next swapchain image and presents it. Equal format
    // and size are a plain copy; anything else is blitted with nearest filtering, cropped to the common area.
    bool present(VkImage source, VkFormat source_format, VkExtent2D source_extent, VkImageLayout source_layout);

    // One plane of a composite present (Xbox PresentX): a source image, the part of it to show and where it goes.
    struct Plane {
        VkImage image = VK_NULL_HANDLE;
        VkFormat format = VK_FORMAT_UNDEFINED;
        VkExtent2D extent{};
        VkImageLayout layout = VK_IMAGE_LAYOUT_UNDEFINED;
        VkRect2D source{};       // a zero size means the whole image
        VkRect2D destination{};  // a zero size means the whole swapchain
    };

    // Presents several planes in order. A single plane that covers the whole image and the whole swapchain with the
    // swapchain's own format is a plain copy; everything else is blitted with nearest filtering, which is also what
    // makes a source/destination rectangle work. Each source is left in the layout it was given.
    bool present(const Plane* planes, std::size_t count);


    // Debug aid for the window test: the next present also copies the finished swapchain image into `buffer` (RGBA8,
    // row pitch = extent().width * 4, offset 0) before presenting it, so the exact pixels that reach the screen can be
    // checked on the CPU. One frame only; the buffer must have TRANSFER_DST usage and be host visible.
    void capture_next_frame(VkBuffer buffer) { capture_ = buffer; }

private:
    // Resolved from the caller's instance/device, never linked: the layer loads Vulkan at run time.
    struct Functions {
#define LEONIDA_WSI_INSTANCE_FUNCTION(name) PFN_##name name = nullptr;
        LEONIDA_WSI_INSTANCE_FUNCTION(vkCreateWin32SurfaceKHR)
        LEONIDA_WSI_INSTANCE_FUNCTION(vkDestroySurfaceKHR)
        LEONIDA_WSI_INSTANCE_FUNCTION(vkGetPhysicalDeviceSurfaceSupportKHR)
        LEONIDA_WSI_INSTANCE_FUNCTION(vkGetPhysicalDeviceSurfaceCapabilitiesKHR)
        LEONIDA_WSI_INSTANCE_FUNCTION(vkGetPhysicalDeviceSurfaceFormatsKHR)
#undef LEONIDA_WSI_INSTANCE_FUNCTION
#define LEONIDA_WSI_DEVICE_FUNCTION(name) PFN_##name name = nullptr;
        LEONIDA_WSI_DEVICE_FUNCTION(vkCreateSwapchainKHR)
        LEONIDA_WSI_DEVICE_FUNCTION(vkDestroySwapchainKHR)
        LEONIDA_WSI_DEVICE_FUNCTION(vkGetSwapchainImagesKHR)
        LEONIDA_WSI_DEVICE_FUNCTION(vkAcquireNextImageKHR)
        LEONIDA_WSI_DEVICE_FUNCTION(vkQueuePresentKHR)
        LEONIDA_WSI_DEVICE_FUNCTION(vkQueueSubmit)
        LEONIDA_WSI_DEVICE_FUNCTION(vkQueueWaitIdle)
        LEONIDA_WSI_DEVICE_FUNCTION(vkDeviceWaitIdle)
        LEONIDA_WSI_DEVICE_FUNCTION(vkCreateCommandPool)
        LEONIDA_WSI_DEVICE_FUNCTION(vkDestroyCommandPool)
        LEONIDA_WSI_DEVICE_FUNCTION(vkAllocateCommandBuffers)
        LEONIDA_WSI_DEVICE_FUNCTION(vkFreeCommandBuffers)
        LEONIDA_WSI_DEVICE_FUNCTION(vkResetCommandBuffer)
        LEONIDA_WSI_DEVICE_FUNCTION(vkBeginCommandBuffer)
        LEONIDA_WSI_DEVICE_FUNCTION(vkEndCommandBuffer)
        LEONIDA_WSI_DEVICE_FUNCTION(vkCmdPipelineBarrier)
        LEONIDA_WSI_DEVICE_FUNCTION(vkCmdCopyImage)
        LEONIDA_WSI_DEVICE_FUNCTION(vkCmdBlitImage)
        LEONIDA_WSI_DEVICE_FUNCTION(vkCmdCopyImageToBuffer)
        LEONIDA_WSI_DEVICE_FUNCTION(vkCreateSemaphore)
        LEONIDA_WSI_DEVICE_FUNCTION(vkDestroySemaphore)
        LEONIDA_WSI_DEVICE_FUNCTION(vkCreateFence)
        LEONIDA_WSI_DEVICE_FUNCTION(vkDestroyFence)
        LEONIDA_WSI_DEVICE_FUNCTION(vkWaitForFences)
        LEONIDA_WSI_DEVICE_FUNCTION(vkResetFences)
#undef LEONIDA_WSI_DEVICE_FUNCTION
    };

    void destroy();

    Gpu gpu_{};
    Functions fn_{};
    VkSurfaceKHR surface_ = VK_NULL_HANDLE;
    VkSwapchainKHR swapchain_ = VK_NULL_HANDLE;
    VkFormat format_ = VK_FORMAT_UNDEFINED;
    VkExtent2D extent_{};
    std::vector<VkImage> images_;
    VkCommandPool pool_ = VK_NULL_HANDLE;
    VkCommandBuffer command_ = VK_NULL_HANDLE;
    VkSemaphore image_available_ = VK_NULL_HANDLE;
    // One "rendering finished" binary semaphore per swapchain image: the presentation engine consumes the one an
    // image was presented with before that image can be acquired again, which is exactly when it may be re-signalled.
    std::vector<VkSemaphore> render_finished_;
    VkFence in_flight_ = VK_NULL_HANDLE;
    VkBuffer capture_ = VK_NULL_HANDLE;
    int frames_ = 0;
};

}  // namespace leonida::wsi
