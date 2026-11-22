// OMSetRenderTargets through the command-list IR (PR C, step 1+2).
//
// The list records OMSetRenderTargets with one colour view and one depth-stencil view, a pipeline and a draw; prepare()
// resolves the views and emit() has to open dynamic rendering before the draw and close it afterwards. The render
// targets are registered with the layout tracker at UNDEFINED, so emit() also has to zero-fill and transition them
// first (a D3D12 committed resource starts zeroed). The triangle comes from first_draw_shaders.h, the pixels are read
// back from an offscreen RGBA8 image and the depth plane from a D32 image: the depth plane must still read zero,
// because the pipeline in this test draws no depth and the attachment only had to be initialised.
//
// A list that names a buffer as a render target has to be rejected by prepare(), before anything is written.
//
// No game data, no keys. Exit 77 means the machine has no usable Vulkan device (the other GPU tests skip the same way).
// Set LEONIDA_VK_VALIDATION=1 to run under VK_LAYER_KHRONOS_validation; any validation error fails the test.

#include "vk_test_device.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iterator>
#include <map>
#include <tuple>
#include <vector>

#include "../src/vulkan_replay.h"
#include "first_draw_shaders.h"
#include "vulkan_bindless.h"

namespace {

constexpr std::uint32_t kSize = 64;
constexpr leonida::ResourceId kColor = 0x1000;   // the colour render target
constexpr leonida::ResourceId kDepth = 0x1001;   // the depth-stencil render target
constexpr leonida::ResourceId kBuffer = 0x1002;  // a buffer, so prepare() can be shown rejecting it
constexpr std::uintptr_t kPso = 0x2000;
constexpr std::uint64_t kVertexVa = 0x300000;
constexpr std::uint64_t kVertexBaseVa = 0x300000;

int fail(const char* what) {
    std::fprintf(stderr, "render_targets_test: %s\n", what);
    return 1;
}

// The views emit() asks for, keyed like VulkanContext keys them, so a repeated request reuses one view.
struct Views {
    vktest::Device& device;
    std::map<std::tuple<VkImage, VkImageAspectFlags, std::uint32_t, std::uint32_t, std::uint32_t>, VkImageView> views;
    ~Views() {
        const auto destroy = device.fn<PFN_vkDestroyImageView>("vkDestroyImageView");
        for (const auto& [key, view] : views) destroy(device.device, view, nullptr);
    }
    VkImageView operator()(const leonida::ResourceBinding& r, const VkImageSubresourceRange& range) {
        auto& view = views[{r.image, range.aspectMask, range.baseMipLevel, range.baseArrayLayer, range.layerCount}];
        if (view) return view;
        VkImageViewCreateInfo info{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        info.image = r.image;
        info.viewType = VK_IMAGE_VIEW_TYPE_2D;
        info.format = r.format;
        info.subresourceRange = range;
        if (device.fn<PFN_vkCreateImageView>("vkCreateImageView")(device.device, &info, nullptr, &view) != VK_SUCCESS)
            view = VK_NULL_HANDLE;
        return view;
    }
};

// The layout is the one the caller passes as bindless_layout; the lookup only resolves the pipeline.
VkPipeline lookup_pipeline(std::uintptr_t id, VkPipelineLayout*, void* user) {
    return id == kPso ? *static_cast<VkPipeline*>(user) : VK_NULL_HANDLE;
}

VkBuffer lookup_buffer(std::uint64_t va, std::uint64_t* base_va, void* user) {
    if (va != kVertexVa) return VK_NULL_HANDLE;
    *base_va = kVertexBaseVa;
    return *static_cast<VkBuffer*>(user);
}

VkShaderModule make_shader(vktest::Device& device, const std::uint32_t* words, std::size_t count) {
    VkShaderModuleCreateInfo info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    info.codeSize = count * sizeof(std::uint32_t);
    info.pCode = words;
    VkShaderModule module = VK_NULL_HANDLE;
    if (device.fn<PFN_vkCreateShaderModule>("vkCreateShaderModule")(device.device, &info, nullptr, &module) != VK_SUCCESS)
        return VK_NULL_HANDLE;
    return module;
}

}  // namespace

int main() {
    vktest::Device device;
    if (!device.create()) {
        std::printf("render_targets_test: no Vulkan device\n");
        return 77;
    }

    VkPhysicalDeviceMemoryProperties memory{};
    device.instance_fn<PFN_vkGetPhysicalDeviceMemoryProperties>("vkGetPhysicalDeviceMemoryProperties")(
        device.physical, &memory);

    leonida::BindlessSet bindless(device.device_proc, device.device, memory);
    if (!bindless.ok()) return fail("BindlessSet creation failed");

    const VkImage color_image = device.image(VK_FORMAT_R8G8B8A8_UNORM, kSize, kSize, 1,
                                             VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT);
    const VkImage depth_image =
        device.image(VK_FORMAT_D32_SFLOAT, kSize, kSize, 1, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT);
    if (!color_image || !depth_image) return fail("render target creation failed");

    void* color_mapped = nullptr;
    void* depth_mapped = nullptr;
    const VkBuffer readback =
        device.buffer(VkDeviceSize(kSize) * kSize * 4 * 2, VK_BUFFER_USAGE_TRANSFER_DST_BIT, &color_mapped);
    if (!readback) return fail("readback buffer creation failed");
    depth_mapped = static_cast<std::uint8_t*>(color_mapped) + VkDeviceSize(kSize) * kSize * 4;

    const VkShaderModule vert =
        make_shader(device, first_draw::triangle_vert_spv, std::size(first_draw::triangle_vert_spv));
    const VkShaderModule frag =
        make_shader(device, first_draw::triangle_frag_spv, std::size(first_draw::triangle_frag_spv));
    if (!vert || !frag) return fail("shader module creation failed");

    const VkDescriptorSetLayout set_layout = bindless.layout();
    VkPipelineLayoutCreateInfo layout_info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    layout_info.setLayoutCount = 1;
    layout_info.pSetLayouts = &set_layout;
    VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
    if (device.fn<PFN_vkCreatePipelineLayout>("vkCreatePipelineLayout")(device.device, &layout_info, nullptr,
                                                                        &pipeline_layout) != VK_SUCCESS)
        return fail("pipeline layout creation failed");

    VkPipelineShaderStageCreateInfo stages[2]{};
    stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vert;
    stages[0].pName = "main";
    stages[1] = stages[0];
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = frag;

    VkVertexInputBindingDescription vertex_binding{};
    vertex_binding.binding = 0;
    vertex_binding.stride = 2 * sizeof(float);
    vertex_binding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
    VkVertexInputAttributeDescription vertex_attribute{};
    vertex_attribute.location = 0;
    vertex_attribute.binding = 0;
    vertex_attribute.format = VK_FORMAT_R32G32_SFLOAT;
    VkPipelineVertexInputStateCreateInfo vertex_input{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    vertex_input.vertexBindingDescriptionCount = 1;
    vertex_input.pVertexBindingDescriptions = &vertex_binding;
    vertex_input.vertexAttributeDescriptionCount = 1;
    vertex_input.pVertexAttributeDescriptions = &vertex_attribute;
    VkPipelineInputAssemblyStateCreateInfo input_assembly{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    input_assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    VkPipelineViewportStateCreateInfo viewport_state{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    viewport_state.viewportCount = 1;
    viewport_state.scissorCount = 1;
    VkPipelineRasterizationStateCreateInfo raster{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    raster.polygonMode = VK_POLYGON_MODE_FILL;
    raster.cullMode = VK_CULL_MODE_NONE;
    raster.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    raster.lineWidth = 1.0f;
    VkPipelineMultisampleStateCreateInfo multisample{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    VkPipelineColorBlendAttachmentState blend_attachment{};
    blend_attachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                                      VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    VkPipelineColorBlendStateCreateInfo blend{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    blend.attachmentCount = 1;
    blend.pAttachments = &blend_attachment;
    VkPipelineDepthStencilStateCreateInfo depth_state{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    depth_state.depthTestEnable = VK_FALSE;  // the attachment is bound and initialised, but nothing writes depth
    depth_state.depthWriteEnable = VK_FALSE;
    depth_state.depthCompareOp = VK_COMPARE_OP_ALWAYS;
    const VkDynamicState dynamic_states[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dynamic{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
    dynamic.dynamicStateCount = 2;
    dynamic.pDynamicStates = dynamic_states;

    const VkFormat color_format = VK_FORMAT_R8G8B8A8_UNORM;
    const VkFormat depth_format = VK_FORMAT_D32_SFLOAT;
    VkPipelineRenderingCreateInfo rendering{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
    rendering.colorAttachmentCount = 1;
    rendering.pColorAttachmentFormats = &color_format;
    rendering.depthAttachmentFormat = depth_format;

    VkGraphicsPipelineCreateInfo pipeline_info{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    pipeline_info.pNext = &rendering;
    pipeline_info.stageCount = 2;
    pipeline_info.pStages = stages;
    pipeline_info.pVertexInputState = &vertex_input;
    pipeline_info.pInputAssemblyState = &input_assembly;
    pipeline_info.pViewportState = &viewport_state;
    pipeline_info.pRasterizationState = &raster;
    pipeline_info.pMultisampleState = &multisample;
    pipeline_info.pColorBlendState = &blend;
    pipeline_info.pDepthStencilState = &depth_state;
    pipeline_info.pDynamicState = &dynamic;
    pipeline_info.layout = pipeline_layout;
    VkPipeline pipeline = VK_NULL_HANDLE;
    if (device.fn<PFN_vkCreateGraphicsPipelines>("vkCreateGraphicsPipelines")(
            device.device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &pipeline) != VK_SUCCESS)
        return fail("graphics pipeline creation failed");

    const float vertices[6] = {-0.8f, -0.8f, 0.8f, -0.8f, 0.0f, 0.8f};
    void* vertex_data = nullptr;
    VkBuffer vertex_buffer = device.buffer(sizeof(vertices), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, &vertex_data);
    if (!vertex_buffer) return fail("vertex buffer creation failed");
    std::memcpy(vertex_data, vertices, sizeof(vertices));

    // The resources as the device would register them: an image, a depth image and a buffer.
    leonida::ResourceBinding color_binding;
    color_binding.image = color_image;
    color_binding.format = color_format;
    color_binding.aspects = VK_IMAGE_ASPECT_COLOR_BIT;
    color_binding.width = color_binding.height = kSize;
    leonida::ResourceBinding depth_binding;
    depth_binding.image = depth_image;
    depth_binding.format = depth_format;
    depth_binding.aspects = VK_IMAGE_ASPECT_DEPTH_BIT;
    depth_binding.width = depth_binding.height = kSize;
    leonida::ResourceBinding buffer_binding;
    buffer_binding.buffer = readback;
    buffer_binding.size = VkDeviceSize(kSize) * kSize * 4 * 2;

    const auto resolve = [&](leonida::ResourceId id) -> const leonida::ResourceBinding* {
        if (id == kColor) return &color_binding;
        if (id == kDepth) return &depth_binding;
        if (id == kBuffer) return &buffer_binding;
        return nullptr;
    };

    // A list that names a buffer as a colour render target is rejected before anything is emitted.
    {
        leonida::CommandList bad;
        const leonida::TargetView buffer_view{kBuffer, 0, 0, 1};
        const leonida::TargetView buffer_views[] = {buffer_view};
        bad.set_render_targets(buffer_views);
        bad.set_pipeline_state(kPso);
        bad.draw_instanced(3);
        if (!bad.close()) return fail("recording the rejected list failed");
        leonida::ReplayFunctions replay{};
        replay.pipeline_barrier = device.fn<PFN_vkCmdPipelineBarrier>("vkCmdPipelineBarrier");
        replay.clear_color_image = device.fn<PFN_vkCmdClearColorImage>("vkCmdClearColorImage");
        replay.clear_depth_stencil_image = device.fn<PFN_vkCmdClearDepthStencilImage>("vkCmdClearDepthStencilImage");
        replay.begin_rendering = device.fn<PFN_vkCmdBeginRendering>("vkCmdBeginRendering");
        replay.end_rendering = device.fn<PFN_vkCmdEndRendering>("vkCmdEndRendering");
        replay.image_view = [&](const leonida::ResourceBinding&, const VkImageSubresourceRange&) {
            return VkImageView{reinterpret_cast<VkImageView>(1)};
        };
        leonida::PreparedList prepared;
        const auto result = leonida::prepare(bad, resolve, replay, prepared);
        if (result != leonida::ReplayResult::invalid_binding) {
            std::fprintf(stderr, "render_targets_test: a buffer render target returned %d\n", int(result));
            return fail("a buffer as a render target was not rejected");
        }
    }

    // The real list: bind both attachments, a pipeline, a vertex buffer and one triangle.
    leonida::CommandList list;
    const leonida::TargetView color_view{kColor, 0, 0, 1};
    const leonida::TargetView depth_view{kDepth, 0, 0, 1};
    const leonida::TargetView color_views[] = {color_view};
    if (!list.set_render_targets(color_views, &depth_view)) return fail("OMSetRenderTargets rejected");
    const leonida::VertexBufferBinding binding{kVertexVa, 2 * sizeof(float), 0};
    if (!list.set_vertex_buffers(0, {&binding, 1})) return fail("IASetVertexBuffers rejected");
    if (!list.set_pipeline_state(kPso)) return fail("SetPipelineState rejected");
    if (!list.draw_instanced(3)) return fail("DrawInstanced rejected");
    if (!list.close()) return fail("Close failed");

    Views views{device};
    leonida::ReplayFunctions replay{};
    replay.pipeline_barrier = device.fn<PFN_vkCmdPipelineBarrier>("vkCmdPipelineBarrier");
    replay.clear_color_image = device.fn<PFN_vkCmdClearColorImage>("vkCmdClearColorImage");
    replay.clear_depth_stencil_image = device.fn<PFN_vkCmdClearDepthStencilImage>("vkCmdClearDepthStencilImage");
    replay.begin_rendering = device.fn<PFN_vkCmdBeginRendering>("vkCmdBeginRendering");
    replay.end_rendering = device.fn<PFN_vkCmdEndRendering>("vkCmdEndRendering");
    replay.image_view = [&](const leonida::ResourceBinding& r, const VkImageSubresourceRange& range) {
        return views(r, range);
    };
    replay.bind_pipeline = device.fn<PFN_vkCmdBindPipeline>("vkCmdBindPipeline");
    replay.bind_descriptor_sets = device.fn<PFN_vkCmdBindDescriptorSets>("vkCmdBindDescriptorSets");
    replay.bind_vertex_buffers = device.fn<PFN_vkCmdBindVertexBuffers>("vkCmdBindVertexBuffers");
    replay.draw = device.fn<PFN_vkCmdDraw>("vkCmdDraw");
    replay.set_viewport = device.fn<PFN_vkCmdSetViewport>("vkCmdSetViewport");
    replay.set_scissor = device.fn<PFN_vkCmdSetScissor>("vkCmdSetScissor");
    replay.pipeline_lookup = lookup_pipeline;
    replay.pipeline_user = &pipeline;
    replay.buffer_lookup = lookup_buffer;
    replay.buffer_user = &vertex_buffer;
    replay.bindless_layout = pipeline_layout;
    replay.bindless_set = bindless.set();
    replay.viewport = {0.0f, 0.0f, float(kSize), float(kSize), 0.0f, 1.0f};
    replay.scissor = {{0, 0}, {kSize, kSize}};

    leonida::LayoutTracker tracker(false);
    tracker.track(kColor, color_binding, VK_IMAGE_LAYOUT_UNDEFINED);
    tracker.track(kDepth, depth_binding, VK_IMAGE_LAYOUT_UNDEFINED);

    bool emitted = false;
    const bool recorded = device.run([&](VkCommandBuffer command_buffer) {
        bindless.record_fallback_init(command_buffer);
        leonida::PreparedList prepared;
        if (leonida::prepare(list, resolve, replay, prepared) != leonida::ReplayResult::success) return false;
        leonida::emit(prepared, command_buffer, replay, &tracker);
        emitted = true;

        // emit() must have closed the rendering instance: a copy is not allowed inside one, and the tracker must have
        // the attachments left in the layout dynamic rendering used.
        if (tracker.layout(kColor, 0) != VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL ||
            tracker.layout(kDepth, 0) != VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
            return false;

        VkImageMemoryBarrier to_copy[2]{};
        for (auto& barrier : to_copy) {
            barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            barrier.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        }
        to_copy[0].oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        to_copy[0].newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        to_copy[0].image = color_image;
        to_copy[1].oldLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        to_copy[1].newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        to_copy[1].image = depth_image;
        to_copy[1].subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1};
        device.fn<PFN_vkCmdPipelineBarrier>("vkCmdPipelineBarrier")(
            command_buffer, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 2, to_copy);

        device.read_back(command_buffer, color_image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_ASPECT_COLOR_BIT,
                         kSize, kSize, 1, readback, 0);
        device.read_back(command_buffer, depth_image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_ASPECT_DEPTH_BIT,
                         kSize, kSize, 1, readback, VkDeviceSize(kSize) * kSize * 4);
        return true;
    });
    if (!recorded) return fail("command buffer recording/submit failed");
    if (!emitted) return fail("the list was not emitted");

    const auto* color_pixels = static_cast<const std::uint8_t*>(color_mapped);
    const auto* depth_pixels = static_cast<const std::uint8_t*>(depth_mapped);
    std::size_t non_black = 0, orange = 0, depth_non_zero = 0;
    for (std::uint32_t i = 0; i < kSize * kSize; ++i) {
        const std::uint8_t r = color_pixels[i * 4], g = color_pixels[i * 4 + 1], b = color_pixels[i * 4 + 2];
        if (r || g || b) ++non_black;
        if (r > 200 && g > 30 && g < 120 && b < 40) ++orange;  // the shader's vec4(1.0, 0.25, 0.0, 1.0)
        float depth;
        std::memcpy(&depth, depth_pixels + i * 4, sizeof(depth));
        if (depth != 0.0f) ++depth_non_zero;
    }
    std::printf("render_targets_test: %zu/%u pixels lit, %zu orange, %zu depth non-zero\n", non_black, kSize * kSize,
                orange, depth_non_zero);

    device.fn<PFN_vkDestroyPipeline>("vkDestroyPipeline")(device.device, pipeline, nullptr);
    device.fn<PFN_vkDestroyPipelineLayout>("vkDestroyPipelineLayout")(device.device, pipeline_layout, nullptr);
    device.fn<PFN_vkDestroyShaderModule>("vkDestroyShaderModule")(device.device, vert, nullptr);
    device.fn<PFN_vkDestroyShaderModule>("vkDestroyShaderModule")(device.device, frag, nullptr);

    if (!non_black) return fail("the rendered image is black: the draw did not reach the attachment");
    if (!orange) return fail("no pixel has the shader's orange");
    if (depth_non_zero) return fail("the depth attachment was not zero-initialised");
    std::printf("render_targets_test: all tests passed\n");
    return 0;
}
