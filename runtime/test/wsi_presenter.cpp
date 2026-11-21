// Real Vulkan WSI presenter: VK_KHR_win32_surface + VK_KHR_swapchain, a rotating triangle, one frame in flight.
// Runs the message loop, presents ~180 frames and logs the count so the PE-loader test can assert animation.

#include "animated_shaders.h"

#include <windows.h>
#include <vulkan/vulkan.h>
#include <vulkan/vulkan_win32.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

constexpr int kWidth = 640;
constexpr int kHeight = 480;

struct Wsi {
    VkInstance instance = VK_NULL_HANDLE;
    VkSurfaceKHR surface = VK_NULL_HANDLE;
    VkPhysicalDevice physical = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    std::uint32_t family = 0;
    VkSwapchainKHR swapchain = VK_NULL_HANDLE;
    VkFormat format = VK_FORMAT_B8G8R8A8_UNORM;
    VkExtent2D extent{kWidth, kHeight};
    std::vector<VkImage> images;
    std::vector<VkImageView> views;
    VkCommandPool pool = VK_NULL_HANDLE;
    VkCommandBuffer command = VK_NULL_HANDLE;
    VkSemaphore image_available = VK_NULL_HANDLE;
    VkSemaphore render_finished = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkShaderModule vert = VK_NULL_HANDLE;
    VkShaderModule frag = VK_NULL_HANDLE;
};

VkShaderModule make_module(VkDevice device, const std::uint32_t* words, std::size_t size) {
    VkShaderModuleCreateInfo info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    info.codeSize = size;
    info.pCode = words;
    VkShaderModule module = VK_NULL_HANDLE;
    if (vkCreateShaderModule(device, &info, nullptr, &module) != VK_SUCCESS) return VK_NULL_HANDLE;
    return module;
}

bool create_swapchain(Wsi& w, HWND hwnd) {
    VkWin32SurfaceCreateInfoKHR surface_info{VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR};
    surface_info.hinstance = GetModuleHandleA(nullptr);
    surface_info.hwnd = hwnd;
    if (vkCreateWin32SurfaceKHR(w.instance, &surface_info, nullptr, &w.surface) != VK_SUCCESS) return false;

    std::uint32_t count = 0;
    vkEnumeratePhysicalDevices(w.instance, &count, nullptr);
    std::vector<VkPhysicalDevice> devices(count);
    vkEnumeratePhysicalDevices(w.instance, &count, devices.data());
    for (VkPhysicalDevice candidate : devices) {
        std::uint32_t families = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(candidate, &families, nullptr);
        std::vector<VkQueueFamilyProperties> props(families);
        vkGetPhysicalDeviceQueueFamilyProperties(candidate, &families, props.data());
        for (std::uint32_t i = 0; i < families; ++i) {
            VkBool32 present = VK_FALSE;
            vkGetPhysicalDeviceSurfaceSupportKHR(candidate, i, w.surface, &present);
            if ((props[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) && present) {
                w.physical = candidate;
                w.family = i;
                break;
            }
        }
        if (w.physical) break;
    }
    if (!w.physical) return false;

    VkPhysicalDeviceVulkan13Features features13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    VkPhysicalDeviceFeatures2 features{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &features13};
    vkGetPhysicalDeviceFeatures2(w.physical, &features);
    if (!features13.dynamicRendering) return false;
    VkPhysicalDeviceVulkan13Features enable13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    enable13.dynamicRendering = VK_TRUE;

    const char* extensions[] = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};
    const float priority = 1.0f;
    VkDeviceQueueCreateInfo queue_info{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    queue_info.queueFamilyIndex = w.family;
    queue_info.queueCount = 1;
    queue_info.pQueuePriorities = &priority;
    VkDeviceCreateInfo device_info{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, &enable13};
    device_info.queueCreateInfoCount = 1;
    device_info.pQueueCreateInfos = &queue_info;
    device_info.enabledExtensionCount = 1;
    device_info.ppEnabledExtensionNames = extensions;
    if (vkCreateDevice(w.physical, &device_info, nullptr, &w.device) != VK_SUCCESS) return false;
    vkGetDeviceQueue(w.device, w.family, 0, &w.queue);

    VkSurfaceCapabilitiesKHR capabilities{};
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(w.physical, w.surface, &capabilities);
    std::uint32_t format_count = 0;
    vkGetPhysicalDeviceSurfaceFormatsKHR(w.physical, w.surface, &format_count, nullptr);
    std::vector<VkSurfaceFormatKHR> formats(format_count);
    vkGetPhysicalDeviceSurfaceFormatsKHR(w.physical, w.surface, &format_count, formats.data());
    w.format = formats.empty() ? VK_FORMAT_B8G8R8A8_UNORM : formats[0].format;
    for (const auto& format : formats)
        if (format.format == VK_FORMAT_B8G8R8A8_UNORM && format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
            w.format = format.format;

    RECT rect{};
    GetClientRect(hwnd, &rect);
    w.extent.width = std::uint32_t(std::max<LONG>(rect.right - rect.left, 1));
    w.extent.height = std::uint32_t(std::max<LONG>(rect.bottom - rect.top, 1));
    if (capabilities.currentExtent.width != UINT32_MAX) w.extent = capabilities.currentExtent;

    VkSwapchainCreateInfoKHR swap_info{VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR};
    swap_info.surface = w.surface;
    swap_info.minImageCount = capabilities.minImageCount + 1;
    swap_info.imageFormat = w.format;
    swap_info.imageColorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
    swap_info.imageExtent = w.extent;
    swap_info.imageArrayLayers = 1;
    swap_info.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    swap_info.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    swap_info.preTransform = capabilities.currentTransform;
    swap_info.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    swap_info.presentMode = VK_PRESENT_MODE_FIFO_KHR;
    swap_info.clipped = VK_TRUE;
    if (vkCreateSwapchainKHR(w.device, &swap_info, nullptr, &w.swapchain) != VK_SUCCESS) return false;

    std::uint32_t image_count = 0;
    vkGetSwapchainImagesKHR(w.device, w.swapchain, &image_count, nullptr);
    w.images.resize(image_count);
    vkGetSwapchainImagesKHR(w.device, w.swapchain, &image_count, w.images.data());
    w.views.resize(image_count, VK_NULL_HANDLE);
    for (std::uint32_t i = 0; i < image_count; ++i) {
        VkImageViewCreateInfo view_info{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        view_info.image = w.images[i];
        view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
        view_info.format = w.format;
        view_info.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        if (vkCreateImageView(w.device, &view_info, nullptr, &w.views[i]) != VK_SUCCESS) return false;
    }
    return true;
}

bool create_pipeline(Wsi& w) {
    w.vert = make_module(w.device, first_animation::anim_vert_spv, sizeof(first_animation::anim_vert_spv));
    w.frag = make_module(w.device, first_animation::anim_frag_spv, sizeof(first_animation::anim_frag_spv));
    if (!w.vert || !w.frag) return false;
    VkPushConstantRange push{};
    push.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    push.size = sizeof(float);
    VkPipelineLayoutCreateInfo layout_info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    layout_info.pushConstantRangeCount = 1;
    layout_info.pPushConstantRanges = &push;
    if (vkCreatePipelineLayout(w.device, &layout_info, nullptr, &w.layout) != VK_SUCCESS) return false;

    VkPipelineShaderStageCreateInfo stages[2]{};
    stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = w.vert;
    stages[0].pName = "main";
    stages[1] = stages[0];
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = w.frag;
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
    rendering.pColorAttachmentFormats = &w.format;
    VkGraphicsPipelineCreateInfo info{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    info.pNext = &rendering;
    info.stageCount = 2;
    info.pStages = stages;
    info.pVertexInputState = &vertex_input;
    info.pInputAssemblyState = &assembly;
    info.pViewportState = &viewport_state;
    info.pRasterizationState = &raster;
    info.pMultisampleState = &multisample;
    info.pColorBlendState = &blend;
    info.pDynamicState = &dynamic;
    info.layout = w.layout;
    return vkCreateGraphicsPipelines(w.device, VK_NULL_HANDLE, 1, &info, nullptr, &w.pipeline) == VK_SUCCESS;
}

void draw_frame(Wsi& w, std::uint32_t index, float time) {
    VkCommandBuffer command = w.command;
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
    to_color.image = w.images[index];
    to_color.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                         0, 0, nullptr, 0, nullptr, 1, &to_color);
    VkRenderingAttachmentInfo color{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
    color.imageView = w.views[index];
    color.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    color.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    color.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    color.clearValue = {{{0.05f, 0.05f, 0.08f, 1.0f}}};
    VkRenderingInfo render{VK_STRUCTURE_TYPE_RENDERING_INFO};
    render.renderArea = {{0, 0}, {w.extent.width, w.extent.height}};
    render.layerCount = 1;
    render.colorAttachmentCount = 1;
    render.pColorAttachments = &color;
    vkCmdBeginRendering(command, &render);
    vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_GRAPHICS, w.pipeline);
    VkViewport viewport{0, 0, float(w.extent.width), float(w.extent.height), 0, 1};
    VkRect2D scissor{{0, 0}, {w.extent.width, w.extent.height}};
    vkCmdSetViewport(command, 0, 1, &viewport);
    vkCmdSetScissor(command, 0, 1, &scissor);
    vkCmdPushConstants(command, w.layout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(float), &time);
    vkCmdDraw(command, 3, 1, 0, 0);
    vkCmdEndRendering(command);
    VkImageMemoryBarrier to_present{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    to_present.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    to_present.dstAccessMask = 0;
    to_present.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    to_present.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    to_present.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    to_present.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    to_present.image = w.images[index];
    to_present.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                         0, 0, nullptr, 0, nullptr, 1, &to_present);
    vkEndCommandBuffer(command);
}

}  // namespace

extern "C" int __cdecl leonida_run_animation(void* hwnd_ptr) {
    HWND hwnd = static_cast<HWND>(hwnd_ptr);
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.apiVersion = VK_API_VERSION_1_3;
    const char* extensions[] = {VK_KHR_SURFACE_EXTENSION_NAME, "VK_KHR_win32_surface"};
    const bool validation = std::getenv("LEONIDA_VK_VALIDATION") != nullptr;
    const char* layers[] = {"VK_LAYER_KHRONOS_validation"};
    VkInstanceCreateInfo instance_info{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    instance_info.pApplicationInfo = &app;
    instance_info.enabledExtensionCount = 2;
    instance_info.ppEnabledExtensionNames = extensions;
    if (validation) {
        instance_info.enabledLayerCount = 1;
        instance_info.ppEnabledLayerNames = layers;
    }
    Wsi w;
    if (vkCreateInstance(&instance_info, nullptr, &w.instance) != VK_SUCCESS) return -1;
    if (!create_swapchain(w, hwnd)) return -2;
    if (!create_pipeline(w)) return -3;

    VkCommandPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pool_info.queueFamilyIndex = w.family;
    if (vkCreateCommandPool(w.device, &pool_info, nullptr, &w.pool) != VK_SUCCESS) return -4;
    VkCommandBufferAllocateInfo alloc{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    alloc.commandPool = w.pool;
    alloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    alloc.commandBufferCount = 1;
    if (vkAllocateCommandBuffers(w.device, &alloc, &w.command) != VK_SUCCESS) return -5;
    VkSemaphoreCreateInfo semaphore_info{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    VkFenceCreateInfo fence_info{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    fence_info.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    vkCreateSemaphore(w.device, &semaphore_info, nullptr, &w.image_available);
    vkCreateSemaphore(w.device, &semaphore_info, nullptr, &w.render_finished);
    vkCreateFence(w.device, &fence_info, nullptr, &w.fence);

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

        vkWaitForFences(w.device, 1, &w.fence, VK_TRUE, UINT64_MAX);
        vkResetFences(w.device, 1, &w.fence);
        std::uint32_t index = 0;
        const VkResult acquired = vkAcquireNextImageKHR(w.device, w.swapchain, UINT64_MAX, w.image_available,
                                                        VK_NULL_HANDLE, &index);
        if (acquired != VK_SUCCESS) break;
        const float time = float(GetTickCount64() - start) / 1000.0f;
        draw_frame(w, index, time);
        VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        submit.waitSemaphoreCount = 1;
        submit.pWaitSemaphores = &w.image_available;
        submit.pWaitDstStageMask = &wait_stage;
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &w.command;
        submit.signalSemaphoreCount = 1;
        submit.pSignalSemaphores = &w.render_finished;
        if (vkQueueSubmit(w.queue, 1, &submit, w.fence) != VK_SUCCESS) break;
        VkPresentInfoKHR present{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
        present.waitSemaphoreCount = 1;
        present.pWaitSemaphores = &w.render_finished;
        present.swapchainCount = 1;
        present.pSwapchains = &w.swapchain;
        present.pImageIndices = &index;
        if (vkQueuePresentKHR(w.queue, &present) != VK_SUCCESS) break;
        vkQueueWaitIdle(w.queue);  // the signalling semaphore must be unsignalled before the next frame
        ++frames;
    }
    vkDeviceWaitIdle(w.device);
    std::FILE* log = std::fopen("leonida_window.log", "w");
    if (log) {
        std::fprintf(log, "wsi=1 frames=%d width=%u height=%u\n", frames, w.extent.width, w.extent.height);
        std::fclose(log);
    }
    vkDestroyFence(w.device, w.fence, nullptr);
    vkDestroySemaphore(w.device, w.image_available, nullptr);
    vkDestroySemaphore(w.device, w.render_finished, nullptr);
    vkDestroyPipeline(w.device, w.pipeline, nullptr);
    vkDestroyPipelineLayout(w.device, w.layout, nullptr);
    vkDestroyShaderModule(w.device, w.vert, nullptr);
    vkDestroyShaderModule(w.device, w.frag, nullptr);
    for (VkImageView view : w.views) vkDestroyImageView(w.device, view, nullptr);
    vkDestroySwapchainKHR(w.device, w.swapchain, nullptr);
    vkDestroyCommandPool(w.device, w.pool, nullptr);
    vkDestroyDevice(w.device, nullptr);
    vkDestroySurfaceKHR(w.instance, w.surface, nullptr);
    vkDestroyInstance(w.instance, nullptr);
    return frames;
}
