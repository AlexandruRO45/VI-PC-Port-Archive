// Swapchain presentation on the renderer's Vulkan objects (see wsi_presenter.h), plus the PE harness's animated
// window, which renders a rotating triangle into an offscreen image of the same device and presents it through the
// same path.
//
// The window is the only thing this file creates that the renderer does not own: the surface and swapchain belong to
// the caller's instance/device and are destroyed with the presenter.

#include "wsi_presenter.h"

#include <vulkan/vulkan_win32.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <vector>

#include "animated_shaders.h"
#include "d3d12_com.h"

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
        vkDeviceWaitIdle(gpu_.device);
        if (command_) vkFreeCommandBuffers(gpu_.device, pool_, 1, &command_);
        if (pool_) vkDestroyCommandPool(gpu_.device, pool_, nullptr);
        if (image_available_) vkDestroySemaphore(gpu_.device, image_available_, nullptr);
        for (VkSemaphore semaphore : render_finished_)
            if (semaphore) vkDestroySemaphore(gpu_.device, semaphore, nullptr);
        if (in_flight_) vkDestroyFence(gpu_.device, in_flight_, nullptr);
        if (swapchain_) vkDestroySwapchainKHR(gpu_.device, swapchain_, nullptr);
    }
    if (surface_ && gpu_.instance && gpu_.get_instance_proc) {
        const auto destroy_surface = reinterpret_cast<PFN_vkDestroySurfaceKHR>(
            gpu_.get_instance_proc(gpu_.instance, "vkDestroySurfaceKHR"));
        if (destroy_surface) destroy_surface(gpu_.instance, surface_, nullptr);
    }
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
    gpu_ = Gpu{};
}

bool Presenter::create(const Gpu& gpu, HWND window, VkFormat format, VkExtent2D extent) {
    destroy();
    if (!window) {
        log("no window to present into");
        return false;
    }
    if (!gpu.instance || !gpu.physical || !gpu.device || !gpu.queue || !gpu.get_instance_proc) {
        log("the renderer's Vulkan objects are incomplete");
        return false;
    }
    gpu_ = gpu;
    const auto create_surface = reinterpret_cast<PFN_vkCreateWin32SurfaceKHR>(
        gpu_.get_instance_proc(gpu_.instance, "vkCreateWin32SurfaceKHR"));
    if (!create_surface) {
        log("VK_KHR_win32_surface is not enabled on the instance (create the device presentable)");
        destroy();
        return false;
    }
    VkWin32SurfaceCreateInfoKHR surface_info{VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR};
    surface_info.hinstance = GetModuleHandleA(nullptr);
    surface_info.hwnd = window;
    if (create_surface(gpu_.instance, &surface_info, nullptr, &surface_) != VK_SUCCESS) {
        log("vkCreateWin32SurfaceKHR failed");
        destroy();
        return false;
    }

    VkBool32 present_supported = VK_FALSE;
    vkGetPhysicalDeviceSurfaceSupportKHR(gpu_.physical, gpu_.queue_family, surface_, &present_supported);
    if (!present_supported) {
        log("the renderer's queue family cannot present on this surface");
        destroy();
        return false;
    }

    VkSurfaceCapabilitiesKHR capabilities{};
    if (vkGetPhysicalDeviceSurfaceCapabilitiesKHR(gpu_.physical, surface_, &capabilities) != VK_SUCCESS) {
        log("vkGetPhysicalDeviceSurfaceCapabilitiesKHR failed");
        destroy();
        return false;
    }

    // Prefer the renderer's format, in a nonlinear colour space; otherwise take what the surface offers first.
    std::uint32_t format_count = 0;
    vkGetPhysicalDeviceSurfaceFormatsKHR(gpu_.physical, surface_, &format_count, nullptr);
    std::vector<VkSurfaceFormatKHR> formats(format_count);
    if (format_count) vkGetPhysicalDeviceSurfaceFormatsKHR(gpu_.physical, surface_, &format_count, formats.data());
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
    const VkResult created = vkCreateSwapchainKHR(gpu_.device, &swap_info, nullptr, &swapchain_);
    if (created != VK_SUCCESS) {
        std::fprintf(stderr, "[wsi] vkCreateSwapchainKHR: %s\n", result_name(created));
        destroy();
        return false;
    }

    std::uint32_t image_count = 0;
    vkGetSwapchainImagesKHR(gpu_.device, swapchain_, &image_count, nullptr);
    images_.resize(image_count);
    vkGetSwapchainImagesKHR(gpu_.device, swapchain_, &image_count, images_.data());

    VkCommandPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pool_info.queueFamilyIndex = gpu_.queue_family;
    if (vkCreateCommandPool(gpu_.device, &pool_info, nullptr, &pool_) != VK_SUCCESS) {
        log("vkCreateCommandPool failed");
        destroy();
        return false;
    }
    VkCommandBufferAllocateInfo alloc{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    alloc.commandPool = pool_;
    alloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    alloc.commandBufferCount = 1;
    if (vkAllocateCommandBuffers(gpu_.device, &alloc, &command_) != VK_SUCCESS) {
        log("vkAllocateCommandBuffers failed");
        destroy();
        return false;
    }
    VkSemaphoreCreateInfo semaphore_info{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    VkFenceCreateInfo fence_info{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    fence_info.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    if (vkCreateSemaphore(gpu_.device, &semaphore_info, nullptr, &image_available_) != VK_SUCCESS ||
        vkCreateFence(gpu_.device, &fence_info, nullptr, &in_flight_) != VK_SUCCESS) {
        log("presentation synchronisation objects could not be created");
        destroy();
        return false;
    }
    render_finished_.resize(images_.size(), VK_NULL_HANDLE);
    for (VkSemaphore& semaphore : render_finished_) {
        if (vkCreateSemaphore(gpu_.device, &semaphore_info, nullptr, &semaphore) != VK_SUCCESS) {
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
    if (vkWaitForFences(gpu_.device, 1, &in_flight_, VK_TRUE, UINT64_MAX) != VK_SUCCESS) return false;

    std::uint32_t index = 0;
    const VkResult acquired =
        vkAcquireNextImageKHR(gpu_.device, swapchain_, UINT64_MAX, image_available_, VK_NULL_HANDLE, &index);
    if (acquired != VK_SUCCESS && acquired != VK_SUBOPTIMAL_KHR) {
        std::fprintf(stderr, "[wsi] vkAcquireNextImageKHR: %s\n", result_name(acquired));
        return false;
    }
    vkResetFences(gpu_.device, 1, &in_flight_);
    vkResetCommandBuffer(command_, 0);
    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(command_, &begin);

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
        vkCmdPipelineBarrier(command_, src_stage, dst_stage, 0, 0, nullptr, 0, nullptr, 1, &b);
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
        vkCmdCopyImage(command_, source, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, images_[index],
                       VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
    } else {
        VkImageBlit region{};
        region.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        region.dstSubresource = region.srcSubresource;
        region.srcOffsets[1] = {std::int32_t(width), std::int32_t(height), 1};
        region.dstOffsets[1] = {std::int32_t(width), std::int32_t(height), 1};
        vkCmdBlitImage(command_, source, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, images_[index],
                       VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region, VK_FILTER_NEAREST);
    }

    if (capture_) {
        barrier(images_[index], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT);
        VkBufferImageCopy copy{};
        copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        copy.imageExtent = {extent_.width, extent_.height, 1};
        vkCmdCopyImageToBuffer(command_, images_[index], VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, capture_, 1, &copy);
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

    vkEndCommandBuffer(command_);
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
    if (vkQueueSubmit(gpu_.queue, 1, &submit, in_flight_) != VK_SUCCESS) {
        log("vkQueueSubmit failed");
        return false;
    }
    VkPresentInfoKHR present{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
    present.waitSemaphoreCount = 1;
    present.pWaitSemaphores = &finished;
    present.swapchainCount = 1;
    present.pSwapchains = &swapchain_;
    present.pImageIndices = &index;
    const VkResult presented = vkQueuePresentKHR(gpu_.queue, &present);
    if (presented != VK_SUCCESS && presented != VK_SUBOPTIMAL_KHR) {
        std::fprintf(stderr, "[wsi] vkQueuePresentKHR: %s\n", result_name(presented));
        return false;
    }
    ++frames_;
    return true;
}

}  // namespace leonida::wsi

// --- the PE harness's animated window ----------------------------------------------------------------

// The D3D12 device the shim DLL created for the harness (set through leonida_set_shared_device below), so the
// animation renders on the same VkDevice the game would (shared-device WSI). Null until D3D12CreateDevice was called.
static ID3D12Device* g_shared_device = nullptr;

extern "C" void leonida_set_shared_device(ID3D12Device* device) { g_shared_device = device; }

namespace {

using leonida::wsi::Presenter;

VkShaderModule make_module(VkDevice device, const std::uint32_t* words, std::size_t bytes) {
    VkShaderModuleCreateInfo info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    info.codeSize = bytes;
    info.pCode = words;
    VkShaderModule module = VK_NULL_HANDLE;
    if (vkCreateShaderModule(device, &info, nullptr, &module) != VK_SUCCESS) return VK_NULL_HANDLE;
    return module;
}

// One offscreen RGBA8 image the rotating triangle renders into; the presenter copies it into the swapchain.
struct Offscreen {
    VkImage image = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    VkFormat format = VK_FORMAT_R8G8B8A8_UNORM;
    VkExtent2D extent{};
};

bool create_offscreen(const leonida::wsi::Gpu& gpu, Offscreen& out) {
    VkImageCreateInfo info{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    info.imageType = VK_IMAGE_TYPE_2D;
    info.format = out.format;
    info.extent = {out.extent.width, out.extent.height, 1};
    info.mipLevels = 1;
    info.arrayLayers = 1;
    info.samples = VK_SAMPLE_COUNT_1_BIT;
    info.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    if (vkCreateImage(gpu.device, &info, nullptr, &out.image) != VK_SUCCESS) return false;
    VkMemoryRequirements requirements{};
    vkGetImageMemoryRequirements(gpu.device, out.image, &requirements);
    VkPhysicalDeviceMemoryProperties memory{};
    vkGetPhysicalDeviceMemoryProperties(gpu.physical, &memory);
    VkMemoryAllocateInfo allocate{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    allocate.allocationSize = requirements.size;
    for (std::uint32_t i = 0; i < memory.memoryTypeCount; ++i) {
        if ((requirements.memoryTypeBits & (1u << i)) &&
            (memory.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) {
            allocate.memoryTypeIndex = i;
            break;
        }
    }
    if (vkAllocateMemory(gpu.device, &allocate, nullptr, &out.memory) != VK_SUCCESS) return false;
    if (vkBindImageMemory(gpu.device, out.image, out.memory, 0) != VK_SUCCESS) return false;
    VkImageViewCreateInfo view{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    view.image = out.image;
    view.viewType = VK_IMAGE_VIEW_TYPE_2D;
    view.format = out.format;
    view.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    return vkCreateImageView(gpu.device, &view, nullptr, &out.view) == VK_SUCCESS;
}

void destroy_offscreen(const leonida::wsi::Gpu& gpu, Offscreen& image) {
    if (image.view) vkDestroyImageView(gpu.device, image.view, nullptr);
    if (image.image) vkDestroyImage(gpu.device, image.image, nullptr);
    if (image.memory) vkFreeMemory(gpu.device, image.memory, nullptr);
    image = Offscreen{};
}

// The rotating-triangle pipeline: dynamic rendering, one RGBA8 attachment, the time as a push constant.
VkPipeline create_animation_pipeline(const leonida::wsi::Gpu& gpu, VkFormat format, VkPipelineLayout& layout) {
    const VkShaderModule vert = make_module(gpu.device, first_animation::anim_vert_spv,
                                            sizeof(first_animation::anim_vert_spv));
    const VkShaderModule frag = make_module(gpu.device, first_animation::anim_frag_spv,
                                            sizeof(first_animation::anim_frag_spv));
    if (!vert || !frag) return VK_NULL_HANDLE;
    VkPushConstantRange push{};
    push.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    push.size = sizeof(float);
    VkPipelineLayoutCreateInfo layout_info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    layout_info.pushConstantRangeCount = 1;
    layout_info.pPushConstantRanges = &push;
    if (vkCreatePipelineLayout(gpu.device, &layout_info, nullptr, &layout) != VK_SUCCESS) return VK_NULL_HANDLE;

    VkPipelineShaderStageCreateInfo stages[2]{};
    stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vert;
    stages[0].pName = "main";
    stages[1] = stages[0];
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = frag;
    VkPipelineVertexInputStateCreateInfo vertex_input{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    VkPipelineInputAssemblyStateCreateInfo assembly{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    VkPipelineViewportStateCreateInfo viewport_state{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    viewport_state.viewportCount = 1;
    viewport_state.scissorCount = 1;
    VkPipelineRasterizationStateCreateInfo raster{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    raster.polygonMode = VK_POLYGON_MODE_FILL;
    raster.cullMode = VK_CULL_MODE_NONE;
    raster.lineWidth = 1.0f;
    VkPipelineMultisampleStateCreateInfo multisample{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    VkPipelineColorBlendAttachmentState blend_attachment{};
    blend_attachment.colorWriteMask = 0xf;
    VkPipelineColorBlendStateCreateInfo blend{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    blend.attachmentCount = 1;
    blend.pAttachments = &blend_attachment;
    const VkDynamicState states[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dynamic{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
    dynamic.dynamicStateCount = 2;
    dynamic.pDynamicStates = states;
    VkPipelineRenderingCreateInfo rendering{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
    rendering.colorAttachmentCount = 1;
    rendering.pColorAttachmentFormats = &format;
    VkGraphicsPipelineCreateInfo pipeline{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    pipeline.pNext = &rendering;
    pipeline.stageCount = 2;
    pipeline.pStages = stages;
    pipeline.pVertexInputState = &vertex_input;
    pipeline.pInputAssemblyState = &assembly;
    pipeline.pViewportState = &viewport_state;
    pipeline.pRasterizationState = &raster;
    pipeline.pMultisampleState = &multisample;
    pipeline.pColorBlendState = &blend;
    pipeline.pDynamicState = &dynamic;
    pipeline.layout = layout;
    VkPipeline created = VK_NULL_HANDLE;
    if (vkCreateGraphicsPipelines(gpu.device, VK_NULL_HANDLE, 1, &pipeline, nullptr, &created) != VK_SUCCESS)
        created = VK_NULL_HANDLE;
    vkDestroyShaderModule(gpu.device, vert, nullptr);
    vkDestroyShaderModule(gpu.device, frag, nullptr);
    return created;
}

}  // namespace

// Renders the rotating triangle on `gpu` (one offscreen image) and presents every frame through the shared-device
// presenter. Returns the number of frames presented, or a negative error code. The device is the caller's.
static int run_animation_on(const leonida::wsi::Gpu& gpu, HWND hwnd) {
    using leonida::wsi::Presenter;

    Presenter presenter;
    if (!presenter.create(gpu, hwnd, VK_FORMAT_R8G8B8A8_UNORM, VkExtent2D{640, 480})) return -3;

    Offscreen offscreen;
    offscreen.extent = presenter.extent();
    if (!create_offscreen(gpu, offscreen)) return -4;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    const VkPipeline pipeline = create_animation_pipeline(gpu, offscreen.format, layout);
    if (!pipeline) return -5;

    VkCommandPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pool_info.queueFamilyIndex = gpu.queue_family;
    VkCommandPool pool = VK_NULL_HANDLE;
    if (vkCreateCommandPool(gpu.device, &pool_info, nullptr, &pool) != VK_SUCCESS) return -6;
    VkCommandBufferAllocateInfo alloc{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    alloc.commandPool = pool;
    alloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    alloc.commandBufferCount = 1;
    VkCommandBuffer command = VK_NULL_HANDLE;
    if (vkAllocateCommandBuffers(gpu.device, &alloc, &command) != VK_SUCCESS) return -7;

    int frames = 0;
    const ULONGLONG start = GetTickCount64();
    bool running = true;
    while (running) {
        MSG msg{};
        while (PeekMessageA(&msg, nullptr, 0, 0, PM_REMOVE)) {
            if (msg.message == WM_QUIT) running = false;
            TranslateMessage(&msg);
            DispatchMessageA(&msg);
        }
        if (GetTickCount64() - start > 4000) break;

        const float time = float(GetTickCount64() - start) / 1000.0f;
        vkResetCommandBuffer(command, 0);
        VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(command, &begin);
        VkImageMemoryBarrier to_color{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        to_color.srcAccessMask = 0;
        to_color.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        to_color.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        to_color.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        to_color.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        to_color.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        to_color.image = offscreen.image;
        to_color.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                             0, 0, nullptr, 0, nullptr, 1, &to_color);
        VkRenderingAttachmentInfo color{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
        color.imageView = offscreen.view;
        color.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        color.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        color.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        color.clearValue = {{{0.05f, 0.05f, 0.08f, 1.0f}}};
        VkRenderingInfo render{VK_STRUCTURE_TYPE_RENDERING_INFO};
        render.renderArea = {{0, 0}, offscreen.extent};
        render.layerCount = 1;
        render.colorAttachmentCount = 1;
        render.pColorAttachments = &color;
        vkCmdBeginRendering(command, &render);
        vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
        VkViewport viewport{0, 0, float(offscreen.extent.width), float(offscreen.extent.height), 0, 1};
        VkRect2D scissor{{0, 0}, offscreen.extent};
        vkCmdSetViewport(command, 0, 1, &viewport);
        vkCmdSetScissor(command, 0, 1, &scissor);
        vkCmdPushConstants(command, layout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(float), &time);
        vkCmdDraw(command, 3, 1, 0, 0);
        vkCmdEndRendering(command);
        VkImageMemoryBarrier to_present{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        to_present.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        to_present.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        to_present.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        to_present.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        to_present.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        to_present.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        to_present.image = offscreen.image;
        to_present.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                             0, nullptr, 0, nullptr, 1, &to_present);
        vkEndCommandBuffer(command);
        VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &command;
        if (vkQueueSubmit(gpu.queue, 1, &submit, VK_NULL_HANDLE) != VK_SUCCESS) break;
        vkQueueWaitIdle(gpu.queue);
        // present() moves the image to TRANSFER_SRC and back to the layout it was given, so the next frame's
        // UNDEFINED-first barrier stays legal.
        if (!presenter.present(offscreen.image, offscreen.format, offscreen.extent,
                               VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL))
            break;
        ++frames;
    }
    vkDeviceWaitIdle(gpu.device);

    std::FILE* log_file = std::fopen("leonida_window.log", "w");
    if (log_file) {
        // wsi=1 keeps the harness contract; shared_device=1 says the presenter ran on the device the harness created.
        std::fprintf(log_file, "wsi=1 shared_device=1 frames=%d width=%u height=%u\n", frames,
                     presenter.extent().width, presenter.extent().height);
        std::fclose(log_file);
    }

    vkDestroyCommandPool(gpu.device, pool, nullptr);
    vkDestroyPipeline(gpu.device, pipeline, nullptr);
    vkDestroyPipelineLayout(gpu.device, layout, nullptr);
    destroy_offscreen(gpu, offscreen);
    return frames;  // presenter's destructor waits for the device and destroys the swapchain
}

// The harness's window: render on the D3D12 device the shim DLL created (made here if the harness did not create
// one), so the animation behaves exactly like a frame of the game would: one VkDevice for rendering and presenting.
extern "C" int __cdecl leonida_run_animation(void* hwnd_ptr) {
    HWND hwnd = static_cast<HWND>(hwnd_ptr);
    using leonida::wsi::Gpu;

    ID3D12Device* device = g_shared_device;
    bool owned_device = false;
    if (!device) {
        if (FAILED(rt::d3d12::create_vulkan_device_standalone(__uuidof(ID3D12Device),
                                                              reinterpret_cast<void**>(&device), true)))
            return -1;
        owned_device = true;
    }
    const rt::d3d12::VulkanDeviceInfo info = rt::d3d12::vulkan_device_info(device);
    if (!info.device) {
        if (owned_device) device->Release();
        return -2;
    }
    const Gpu gpu{info.get_instance_proc, info.instance, info.physical, info.device, info.queue, info.queue_family};
    const int frames = run_animation_on(gpu, hwnd);
    if (owned_device) device->Release();
    return frames;
}
