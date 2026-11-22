// Swapchain presentation on the renderer's Vulkan objects (see wsi_presenter.h). No Vulkan entry point is linked:
// everything is resolved from the caller's instance and device, so a game process keeps loading Vulkan at run time.

#include "wsi_presenter.h"

#include <vulkan/vulkan_win32.h>

#include <algorithm>
#include <cstdio>

namespace {

void log(const char* what) { std::fprintf(stderr, "[wsi] %s\n", what); }

const char* result_name(VkResult result) {
    switch (result) {
        case VK_SUCCESS: return "success";
        case VK_SUBOPTIMAL_KHR: return "suboptimal";
        case VK_ERROR_OUT_OF_DATE_KHR: return "out of date";
        case VK_ERROR_SURFACE_LOST_KHR: return "surface lost";
        case VK_ERROR_DEVICE_LOST: return "device lost";
        default: return "error";
    }
}

}  // namespace

namespace leonida::wsi {

Presenter::~Presenter() { destroy(); }

void Presenter::destroy() {
    if (gpu_.device) {
        if (fn_.vkDeviceWaitIdle) fn_.vkDeviceWaitIdle(gpu_.device);
        if (command_ && fn_.vkFreeCommandBuffers) fn_.vkFreeCommandBuffers(gpu_.device, pool_, 1, &command_);
        if (pool_ && fn_.vkDestroyCommandPool) fn_.vkDestroyCommandPool(gpu_.device, pool_, nullptr);
        if (image_available_ && fn_.vkDestroySemaphore)
            fn_.vkDestroySemaphore(gpu_.device, image_available_, nullptr);
        for (VkSemaphore semaphore : render_finished_)
            if (semaphore && fn_.vkDestroySemaphore) fn_.vkDestroySemaphore(gpu_.device, semaphore, nullptr);
        if (in_flight_ && fn_.vkDestroyFence) fn_.vkDestroyFence(gpu_.device, in_flight_, nullptr);
        if (swapchain_ && fn_.vkDestroySwapchainKHR) fn_.vkDestroySwapchainKHR(gpu_.device, swapchain_, nullptr);
    }
    if (surface_ && gpu_.instance && fn_.vkDestroySurfaceKHR) fn_.vkDestroySurfaceKHR(gpu_.instance, surface_, nullptr);
    surface_ = VK_NULL_HANDLE;
    swapchain_ = VK_NULL_HANDLE;
    command_ = VK_NULL_HANDLE;
    pool_ = VK_NULL_HANDLE;
    image_available_ = VK_NULL_HANDLE;
    render_finished_.clear();
    in_flight_ = VK_NULL_HANDLE;
    images_.clear();
    frames_ = 0;
    capture_ = VK_NULL_HANDLE;
    fn_ = Functions{};
    gpu_ = Gpu{};
}

bool Presenter::create(const Gpu& gpu, HWND window, VkFormat format, VkExtent2D extent) {
    destroy();
    if (!window) {
        log("no window to present into");
        return false;
    }
    if (!gpu.get_instance_proc || !gpu.instance || !gpu.physical || !gpu.device || !gpu.queue) {
        log("the renderer's Vulkan objects are incomplete");
        return false;
    }
    gpu_ = gpu;

    const PFN_vkGetDeviceProcAddr get_device =
        reinterpret_cast<PFN_vkGetDeviceProcAddr>(gpu_.get_instance_proc(gpu_.instance, "vkGetDeviceProcAddr"));
    if (!get_device) {
        log("vkGetDeviceProcAddr is not available on the instance");
        destroy();
        return false;
    }
#define LEONIDA_WSI_LOAD_INSTANCE(name)                                                                        \
    fn_.name = reinterpret_cast<PFN_##name>(gpu_.get_instance_proc(gpu_.instance, #name));
#define LEONIDA_WSI_LOAD_DEVICE(name) fn_.name = reinterpret_cast<PFN_##name>(get_device(gpu_.device, #name));
#define LEONIDA_WSI_INSTANCE_FUNCTIONS(X)                                                                      \
    X(vkCreateWin32SurfaceKHR) X(vkDestroySurfaceKHR) X(vkGetPhysicalDeviceSurfaceSupportKHR)                  \
    X(vkGetPhysicalDeviceSurfaceCapabilitiesKHR) X(vkGetPhysicalDeviceSurfaceFormatsKHR)
#define LEONIDA_WSI_DEVICE_FUNCTIONS(X)                                                                        \
    X(vkCreateSwapchainKHR) X(vkDestroySwapchainKHR) X(vkGetSwapchainImagesKHR) X(vkAcquireNextImageKHR)       \
    X(vkQueuePresentKHR) X(vkQueueSubmit) X(vkQueueWaitIdle) X(vkDeviceWaitIdle) X(vkCreateCommandPool)        \
    X(vkDestroyCommandPool) X(vkAllocateCommandBuffers) X(vkFreeCommandBuffers) X(vkResetCommandBuffer)        \
    X(vkBeginCommandBuffer) X(vkEndCommandBuffer) X(vkCmdPipelineBarrier) X(vkCmdCopyImage) X(vkCmdBlitImage)   \
    X(vkCmdCopyImageToBuffer) X(vkCreateSemaphore) X(vkDestroySemaphore) X(vkCreateFence) X(vkDestroyFence)    \
    X(vkWaitForFences) X(vkResetFences)
    LEONIDA_WSI_INSTANCE_FUNCTIONS(LEONIDA_WSI_LOAD_INSTANCE)
    LEONIDA_WSI_DEVICE_FUNCTIONS(LEONIDA_WSI_LOAD_DEVICE)
#undef LEONIDA_WSI_LOAD_INSTANCE
#undef LEONIDA_WSI_LOAD_DEVICE
#undef LEONIDA_WSI_INSTANCE_FUNCTIONS
#undef LEONIDA_WSI_DEVICE_FUNCTIONS
    if (!fn_.vkCreateWin32SurfaceKHR || !fn_.vkCreateSwapchainKHR || !fn_.vkQueuePresentKHR ||
        !fn_.vkAcquireNextImageKHR) {
        log("the instance/device does not expose the surface or swapchain entry points (create it presentable)");
        destroy();
        return false;
    }

    VkWin32SurfaceCreateInfoKHR surface_info{VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR};
    surface_info.hinstance = GetModuleHandleA(nullptr);
    surface_info.hwnd = window;
    if (fn_.vkCreateWin32SurfaceKHR(gpu_.instance, &surface_info, nullptr, &surface_) != VK_SUCCESS) {
        log("vkCreateWin32SurfaceKHR failed");
        destroy();
        return false;
    }

    VkBool32 present_supported = VK_FALSE;
    fn_.vkGetPhysicalDeviceSurfaceSupportKHR(gpu_.physical, gpu_.queue_family, surface_, &present_supported);
    if (!present_supported) {
        log("the renderer's queue family cannot present on this surface");
        destroy();
        return false;
    }

    VkSurfaceCapabilitiesKHR capabilities{};
    if (fn_.vkGetPhysicalDeviceSurfaceCapabilitiesKHR(gpu_.physical, surface_, &capabilities) != VK_SUCCESS) {
        log("vkGetPhysicalDeviceSurfaceCapabilitiesKHR failed");
        destroy();
        return false;
    }

    // Prefer the renderer's format, in a nonlinear colour space; otherwise take what the surface offers first.
    std::uint32_t format_count = 0;
    fn_.vkGetPhysicalDeviceSurfaceFormatsKHR(gpu_.physical, surface_, &format_count, nullptr);
    std::vector<VkSurfaceFormatKHR> formats(format_count);
    if (format_count) fn_.vkGetPhysicalDeviceSurfaceFormatsKHR(gpu_.physical, surface_, &format_count, formats.data());
    format_ = formats.empty() ? VK_FORMAT_B8G8R8A8_UNORM : formats[0].format;
    for (const auto& offered : formats) {
        if (offered.format == format && offered.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
            format_ = format;
            break;
        }
    }
    for (const auto& offered : formats) {
        if (offered.format == format) {
            format_ = format;
            break;
        }
    }

    extent_ = extent;
    if (capabilities.currentExtent.width != UINT32_MAX) extent_ = capabilities.currentExtent;
    extent_.width = std::clamp(extent_.width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
    extent_.height = std::clamp(extent_.height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height);
    if (!extent_.width || !extent_.height) {
        log("the surface has no area to present into");
        destroy();
        return false;
    }

    VkImageUsageFlags usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    if (capabilities.supportedUsageFlags & VK_IMAGE_USAGE_TRANSFER_SRC_BIT)
        usage |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;  // the present can be captured for tests

    VkCompositeAlphaFlagBitsKHR composite = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    if (!(capabilities.supportedCompositeAlpha & composite)) {
        for (std::uint32_t bit = 1; bit <= VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR; bit <<= 1) {
            if (capabilities.supportedCompositeAlpha & bit) {
                composite = VkCompositeAlphaFlagBitsKHR(bit);
                break;
            }
        }
    }

    VkSwapchainCreateInfoKHR swap_info{VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR};
    swap_info.surface = surface_;
    swap_info.minImageCount = capabilities.minImageCount + 1;
    swap_info.imageFormat = format_;
    swap_info.imageColorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
    swap_info.imageExtent = extent_;
    swap_info.imageArrayLayers = 1;
    swap_info.imageUsage = usage;
    swap_info.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;  // the same queue family renders and presents
    swap_info.preTransform = capabilities.currentTransform;
    swap_info.compositeAlpha = composite;
    swap_info.presentMode = VK_PRESENT_MODE_FIFO_KHR;
    swap_info.clipped = VK_TRUE;
    const VkResult created = fn_.vkCreateSwapchainKHR(gpu_.device, &swap_info, nullptr, &swapchain_);
    if (created != VK_SUCCESS) {
        std::fprintf(stderr, "[wsi] vkCreateSwapchainKHR: %s\n", result_name(created));
        destroy();
        return false;
    }

    std::uint32_t image_count = 0;
    fn_.vkGetSwapchainImagesKHR(gpu_.device, swapchain_, &image_count, nullptr);
    images_.resize(image_count);
    fn_.vkGetSwapchainImagesKHR(gpu_.device, swapchain_, &image_count, images_.data());

    VkCommandPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pool_info.queueFamilyIndex = gpu_.queue_family;
    if (fn_.vkCreateCommandPool(gpu_.device, &pool_info, nullptr, &pool_) != VK_SUCCESS) {
        log("vkCreateCommandPool failed");
        destroy();
        return false;
    }
    VkCommandBufferAllocateInfo alloc{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    alloc.commandPool = pool_;
    alloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    alloc.commandBufferCount = 1;
    if (fn_.vkAllocateCommandBuffers(gpu_.device, &alloc, &command_) != VK_SUCCESS) {
        log("vkAllocateCommandBuffers failed");
        destroy();
        return false;
    }
    VkSemaphoreCreateInfo semaphore_info{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    VkFenceCreateInfo fence_info{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    fence_info.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    if (fn_.vkCreateSemaphore(gpu_.device, &semaphore_info, nullptr, &image_available_) != VK_SUCCESS ||
        fn_.vkCreateFence(gpu_.device, &fence_info, nullptr, &in_flight_) != VK_SUCCESS) {
        log("presentation synchronisation objects could not be created");
        destroy();
        return false;
    }
    render_finished_.resize(images_.size(), VK_NULL_HANDLE);
    for (VkSemaphore& semaphore : render_finished_) {
        if (fn_.vkCreateSemaphore(gpu_.device, &semaphore_info, nullptr, &semaphore) != VK_SUCCESS) {
            log("presentation synchronisation objects could not be created");
            destroy();
            return false;
        }
    }
    return true;
}

bool Presenter::present(VkImage source, VkFormat source_format, VkExtent2D source_extent,
                        VkImageLayout source_layout) {
    if (!ok() || !source) return false;
    if (fn_.vkWaitForFences(gpu_.device, 1, &in_flight_, VK_TRUE, UINT64_MAX) != VK_SUCCESS) return false;

    std::uint32_t index = 0;
    const VkResult acquired =
        fn_.vkAcquireNextImageKHR(gpu_.device, swapchain_, UINT64_MAX, image_available_, VK_NULL_HANDLE, &index);
    if (acquired != VK_SUCCESS && acquired != VK_SUBOPTIMAL_KHR) {
        std::fprintf(stderr, "[wsi] vkAcquireNextImageKHR: %s\n", result_name(acquired));
        return false;
    }
    fn_.vkResetFences(gpu_.device, 1, &in_flight_);
    fn_.vkResetCommandBuffer(command_, 0);
    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    fn_.vkBeginCommandBuffer(command_, &begin);

    const auto barrier = [&](VkImage image, VkImageLayout from, VkImageLayout to, VkAccessFlags src_access,
                             VkAccessFlags dst_access, VkPipelineStageFlags src_stage, VkPipelineStageFlags dst_stage) {
        VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        b.srcAccessMask = src_access;
        b.dstAccessMask = dst_access;
        b.oldLayout = from;
        b.newLayout = to;
        b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.image = image;
        b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        fn_.vkCmdPipelineBarrier(command_, src_stage, dst_stage, 0, 0, nullptr, 0, nullptr, 1, &b);
    };

    // The rendered image goes to TRANSFER_SRC and back again, so the layer's tracker keeps the layout it left.
    barrier(source, source_layout, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_ACCESS_MEMORY_WRITE_BIT,
            VK_ACCESS_TRANSFER_READ_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
    barrier(images_[index], VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 0,
            VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);

    const std::uint32_t width = std::min(source_extent.width, extent_.width);
    const std::uint32_t height = std::min(source_extent.height, extent_.height);
    if (source_format == format_ && source_extent.width == extent_.width && source_extent.height == extent_.height) {
        VkImageCopy region{};
        region.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        region.dstSubresource = region.srcSubresource;
        region.extent = {width, height, 1};
        fn_.vkCmdCopyImage(command_, source, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, images_[index],
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
    } else {
        VkImageBlit region{};
        region.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        region.dstSubresource = region.srcSubresource;
        region.srcOffsets[1] = {std::int32_t(width), std::int32_t(height), 1};
        region.dstOffsets[1] = {std::int32_t(width), std::int32_t(height), 1};
        fn_.vkCmdBlitImage(command_, source, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, images_[index],
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region, VK_FILTER_NEAREST);
    }

    if (capture_) {
        barrier(images_[index], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT);
        VkBufferImageCopy copy{};
        copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        copy.imageExtent = {extent_.width, extent_.height, 1};
        fn_.vkCmdCopyImageToBuffer(command_, images_[index], VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, capture_, 1, &copy);
        barrier(images_[index], VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                VK_ACCESS_TRANSFER_READ_BIT, 0, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);
        capture_ = VK_NULL_HANDLE;
    } else {
        barrier(images_[index], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                VK_ACCESS_TRANSFER_WRITE_BIT, 0, VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);
    }
    barrier(source, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, source_layout, VK_ACCESS_TRANSFER_READ_BIT,
            VK_ACCESS_MEMORY_WRITE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);

    fn_.vkEndCommandBuffer(command_);
    // The image's own semaphore: its previous signal was consumed by the present that used this image, and the image
    // can only be acquired again after that present finished.
    VkSemaphore& finished = render_finished_[index];
    const VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submit.waitSemaphoreCount = 1;
    submit.pWaitSemaphores = &image_available_;
    submit.pWaitDstStageMask = &wait_stage;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &command_;
    submit.signalSemaphoreCount = 1;
    submit.pSignalSemaphores = &finished;
    if (fn_.vkQueueSubmit(gpu_.queue, 1, &submit, in_flight_) != VK_SUCCESS) {
        log("vkQueueSubmit failed");
        return false;
    }
    VkPresentInfoKHR present{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
    present.waitSemaphoreCount = 1;
    present.pWaitSemaphores = &finished;
    present.swapchainCount = 1;
    present.pSwapchains = &swapchain_;
    present.pImageIndices = &index;
    const VkResult presented = fn_.vkQueuePresentKHR(gpu_.queue, &present);
    if (presented != VK_SUCCESS && presented != VK_SUBOPTIMAL_KHR) {
        std::fprintf(stderr, "[wsi] vkQueuePresentKHR: %s\n", result_name(presented));
        return false;
    }
    ++frames_;
    return true;
}

}  // namespace leonida::wsi
