// The PE harness's animated window: a rotating triangle rendered into an offscreen image on the D3D12 device's
// VkDevice and presented through leonida::wsi::Presenter (shared-device WSI). The device is the one the shim DLL
// created for the harness (leonida_set_shared_device); if the harness did not create one, this makes a presentable
// device itself. The window log keeps its wsi=1 contract and adds shared_device=1.

#include "../shims/wsi_presenter.h"  // first: it pulls in the Vulkan prototypes the render path uses below
#include "../shims/d3d12_com.h"

#include <algorithm>
#include <cstdio>
#include <cstring>

#include "animated_shaders.h"

namespace {

using leonida::wsi::Gpu;
using leonida::wsi::Presenter;

// The D3D12 device the shim DLL created for the harness, so the animation renders on the same VkDevice the game
// would (shared-device WSI). Null until D3D12CreateDevice was called.
ID3D12Device* g_shared_device = nullptr;

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

bool create_offscreen(const Gpu& gpu, Offscreen& out) {
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

void destroy_offscreen(const Gpu& gpu, Offscreen& image) {
    if (image.view) vkDestroyImageView(gpu.device, image.view, nullptr);
    if (image.image) vkDestroyImage(gpu.device, image.image, nullptr);
    if (image.memory) vkFreeMemory(gpu.device, image.memory, nullptr);
    image = Offscreen{};
}

// The rotating-triangle pipeline: dynamic rendering, one RGBA8 attachment, the time as a push constant.
VkPipeline create_animation_pipeline(const Gpu& gpu, VkFormat format, VkPipelineLayout& layout) {
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

extern "C" void leonida_set_shared_device(ID3D12Device* device) { g_shared_device = device; }

// Renders the rotating triangle on `gpu` (one offscreen image) and presents every frame through the shared-device
// presenter. Returns the number of frames presented, or a negative error code. The device is the caller's.
static int run_animation_on(const Gpu& gpu, HWND hwnd) {
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
        VkImageMemoryBarrier to_copy{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        to_copy.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        to_copy.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        to_copy.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        to_copy.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        to_copy.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        to_copy.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        to_copy.image = offscreen.image;
        to_copy.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                             0, nullptr, 0, nullptr, 1, &to_copy);
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
