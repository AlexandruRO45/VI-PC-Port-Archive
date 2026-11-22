// A draw through VulkanQueue::execute (PR C, step 3): the same path ID3D12CommandQueue::ExecuteCommandLists takes in
// the COM layer, without COM.
//
// The list is recorded in the D3D12 IR (OMSetRenderTargets, IASetVertexBuffers, SetPipelineState, DrawInstanced) and
// the queue's VulkanContext supplies what a draw needs: the command entry points loaded by VulkanDeviceFunctions, the
// pipeline registry (the test registers its VkPipeline by address, exactly as PipelineState does for CreatePipelineState
// objects), set 0 and the console VA -> VkBuffer lookup through set_draw_resources(), and the fallback initialisation
// of the bindless set recorded by the context at the start of the submission.
//
// The render target is registered with the layout tracker at UNDEFINED, so the first use also has to zero-fill and
// transition it. The triangle comes from first_draw_shaders.h. The pixels travel back through the IR as well: a second
// list transitions the render target from the layout the first list left it in and copies it into a host-visible
// buffer, so the tracker state has to survive between submissions. The queue's own timeline semaphore is waited on
// through VulkanFence, as a D3D12 fence wait would be.
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

#include "../src/vulkan_backend.h"
#include "first_draw_shaders.h"
#include "vulkan_bindless.h"

namespace {

constexpr std::uint32_t kSize = 64;
constexpr leonida::ResourceId kColor = 0x4000;     // the colour render target
constexpr leonida::ResourceId kReadback = 0x4001;  // the buffer the second list copies the image into
constexpr std::uintptr_t kPso = 0x5000;
constexpr std::uintptr_t kUnknownPso = 0x5001;
constexpr std::uint64_t kVertexVa = 0x600000;

int fail(const char* what) {
    std::fprintf(stderr, "queue_draw_test: %s\n", what);
    return 1;
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

// The console VA -> VkBuffer lookup the device installs, with one buffer at one address.
struct BufferLookup {
    std::uint64_t va = kVertexVa;
    VkBuffer buffer = VK_NULL_HANDLE;
};

VkBuffer lookup_buffer(std::uint64_t va, std::uint64_t* base_va, void* user) {
    const auto* lookup = static_cast<const BufferLookup*>(user);
    if (va != lookup->va) return VK_NULL_HANDLE;
    if (base_va) *base_va = lookup->va;
    return lookup->buffer;
}

}  // namespace

int main() {
    using namespace leonida;
    vktest::Device device;
    if (!device.create()) {
        std::printf("queue_draw_test: no Vulkan device\n");
        return 77;
    }

    VkPhysicalDeviceMemoryProperties memory{};
    device.instance_fn<PFN_vkGetPhysicalDeviceMemoryProperties>("vkGetPhysicalDeviceMemoryProperties")(
        device.physical, &memory);

    BindlessSet bindless(device.device_proc, device.device, memory);
    if (!bindless.ok()) return fail("BindlessSet creation failed");

    // The queue shares this context: its command entry points, layout tracker and pipeline registry.
    VulkanContext context(device.device_proc, device.device, true);
    if (!context.loaded()) return fail("VulkanContext did not load the device entry points");
    VulkanQueue queue(context, device.queue, device.family);
    if (!queue.ok()) return fail("VulkanQueue creation failed");

    const VkImage color_image =
        device.image(VK_FORMAT_R8G8B8A8_UNORM, kSize, kSize, 1, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT);
    if (!color_image) return fail("render target creation failed");
    void* readback_data = nullptr;
    const VkBuffer readback =
        device.buffer(VkDeviceSize(kSize) * kSize * 4, VK_BUFFER_USAGE_TRANSFER_DST_BIT, &readback_data);
    if (!readback) return fail("readback buffer creation failed");

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
    blend_attachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT |
                                      VK_COLOR_COMPONENT_A_BIT;
    VkPipelineColorBlendStateCreateInfo blend{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    blend.attachmentCount = 1;
    blend.pAttachments = &blend_attachment;
    // Only the viewport and scissor are dynamic: the test relies on emit() setting them from the attachments.
    const VkDynamicState dynamic_states[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dynamic{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
    dynamic.dynamicStateCount = 2;
    dynamic.pDynamicStates = dynamic_states;

    const VkFormat color_format = VK_FORMAT_R8G8B8A8_UNORM;
    VkPipelineRenderingCreateInfo rendering{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
    rendering.colorAttachmentCount = 1;
    rendering.pColorAttachmentFormats = &color_format;

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
    pipeline_info.pDynamicState = &dynamic;
    pipeline_info.layout = pipeline_layout;
    VkPipeline pipeline = VK_NULL_HANDLE;
    if (device.fn<PFN_vkCreateGraphicsPipelines>("vkCreateGraphicsPipelines")(
            device.device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &pipeline) != VK_SUCCESS)
        return fail("graphics pipeline creation failed");

    const float vertices[6] = {-0.8f, -0.8f, 0.8f, -0.8f, 0.0f, 0.8f};
    void* vertex_data = nullptr;
    const VkBuffer vertex_buffer = device.buffer(sizeof(vertices), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, &vertex_data);
    if (!vertex_buffer) return fail("vertex buffer creation failed");
    std::memcpy(vertex_data, vertices, sizeof(vertices));

    // Register the resources and the pipeline the way the COM device does: the image in its D3D12 initial state
    // (UNDEFINED: never used, so the first command zero-fills it), the buffer untracked, the VkPipeline by address.
    ResourceBinding color_binding;
    color_binding.image = color_image;
    color_binding.format = color_format;
    color_binding.aspects = VK_IMAGE_ASPECT_COLOR_BIT;
    color_binding.width = color_binding.height = kSize;
    ResourceBinding readback_binding;
    readback_binding.buffer = readback;
    readback_binding.size = VkDeviceSize(kSize) * kSize * 4;
    context.add_resource(kColor, color_binding, VK_IMAGE_LAYOUT_UNDEFINED);
    context.add_resource(kReadback, readback_binding);
    context.add_pipeline(kPso, pipeline, pipeline_layout);
    if (context.pipeline(kPso).pipeline != pipeline) return fail("the pipeline registry did not return the registered pipeline");
    if (context.pipeline(kPso).layout != pipeline_layout) return fail("the registry lost the pipeline's layout");
    if (context.pipeline(kUnknownPso).pipeline != VK_NULL_HANDLE) return fail("a state that was never registered resolved");

    BufferLookup lookup{};
    lookup.buffer = vertex_buffer;
    context.set_draw_resources(bindless.set(), &lookup_buffer, &lookup, [&bindless](VkCommandBuffer command_buffer) {
        if (!bindless.fallbacks_ready()) bindless.record_fallback_init(command_buffer);
    });

    // List 1: bind the colour attachment, a vertex buffer, the pipeline and draw one triangle.
    CommandList draw_list;
    const TargetView color_view{kColor, 0, 0, 1};
    const TargetView color_views[] = {color_view};
    if (!draw_list.set_render_targets(color_views)) return fail("OMSetRenderTargets rejected");
    const VertexBufferBinding binding{kVertexVa, 2 * sizeof(float), 0};
    if (!draw_list.set_vertex_buffers(0, {&binding, 1})) return fail("IASetVertexBuffers rejected");
    if (!draw_list.set_pipeline_state(kPso)) return fail("SetPipelineState rejected");
    if (!draw_list.draw_instanced(3)) return fail("DrawInstanced rejected");
    if (!draw_list.close()) return fail("Close failed");

    {
        const CommandList* lists[] = {&draw_list};
        const ReplayResult result = queue.execute(lists);
        if (result != ReplayResult::success) {
            std::fprintf(stderr, "queue_draw_test: the draw submission returned %d\n", int(result));
            return fail("the draw list was not submitted");
        }
    }
    if (context.layout(kColor, 0) != VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL)
        return fail("the render target is not left in the attachment layout");

    // List 2: the readback goes through the IR too, starting from the layout the first submission left behind.
    CommandList read_list;
    if (!read_list.transition({kColor, all_subresources, d3d12_state::render_target, d3d12_state::copy_source}))
        return fail("the transition from the layout the draw left was rejected");
    const CopyLocation destination{kReadback, true, 0, 0, kSize, kSize, 1, kSize * 4};
    const CopyLocation source{kColor, false, 0, 0, 0, 0, 0, 0};
    if (!read_list.copy_texture({destination, 0, 0, 0, source, false, {}}))
        return fail("the readback copy was rejected");
    if (!read_list.close()) return fail("Close failed");
    {
        const CommandList* lists[] = {&read_list};
        const ReplayResult result = queue.execute(lists);
        if (result != ReplayResult::success) {
            std::fprintf(stderr, "queue_draw_test: the readback submission returned %d\n", int(result));
            return fail("the readback list was not submitted");
        }
    }
    if (context.layout(kColor, 0) != VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL)
        return fail("the readback did not move the render target to copy_source");

    // Wait for both submissions through the queue's timeline semaphore, as a D3D12 fence wait does.
    VulkanFence fence(context, 0);
    if (!queue.signal(fence.semaphore(), 1)) return fail("signalling the queue fence failed");
    if (!fence.wait(1, 5'000'000'000ull)) return fail("the queue fence was not reached");

    const auto* pixels = static_cast<const std::uint8_t*>(readback_data);
    std::size_t non_black = 0, orange = 0, unwritten = 0;
    for (std::uint32_t i = 0; i < kSize * kSize; ++i) {
        const std::uint8_t r = pixels[i * 4], g = pixels[i * 4 + 1], b = pixels[i * 4 + 2];
        if (r || g || b) ++non_black;
        if (r > 200 && g > 30 && g < 120 && b < 40) ++orange;  // the shader's vec4(1.0, 0.25, 0.0, 1.0)
        if (r == 0xcd && g == 0xcd && b == 0xcd) ++unwritten;
    }
    std::printf("queue_draw_test: %zu/%u pixels lit, %zu orange, %zu untouched\n", non_black, kSize * kSize, orange,
                unwritten);

    // A state that is not in the registry resolves to nothing, and an unregistered draw is skipped, not half-bound.
    context.remove_pipeline(kPso);
    if (context.pipeline(kPso).pipeline != VK_NULL_HANDLE) return fail("removing a pipeline left it in the registry");

    context.clear_pipelines();
    device.fn<PFN_vkDestroyPipeline>("vkDestroyPipeline")(device.device, pipeline, nullptr);
    device.fn<PFN_vkDestroyPipelineLayout>("vkDestroyPipelineLayout")(device.device, pipeline_layout, nullptr);
    device.fn<PFN_vkDestroyShaderModule>("vkDestroyShaderModule")(device.device, vert, nullptr);
    device.fn<PFN_vkDestroyShaderModule>("vkDestroyShaderModule")(device.device, frag, nullptr);

    if (unwritten) return fail("part of the render target was never written: the readback did not cover the image");
    if (!non_black) return fail("the rendered image is black: the draw did not reach the attachment");
    if (!orange) return fail("no pixel has the shader's orange");
    if (device.validation_errors()) {
        std::fprintf(stderr, "%u validation errors\n", device.validation_errors());
        return 1;
    }
    std::printf("queue_draw_test: all tests passed\n");
    return 0;
}
