// End-to-end textured quad: bindless texture2D[]/sampler[] + push-constant index + IR input assembly and draw.
//
// The 2x2 checkerboard (red/green) is uploaded, sampled through the bindless set, the index comes from a push
// constant, and the quad is drawn by leonida::emit() from a CommandList that records IASetVertexBuffers,
// IASetIndexBuffer, SetPipelineState and DrawIndexedInstanced. The result is read back and both colours must appear.
//
// No game data. Exit 77: no Vulkan device.

#include "vk_test_device.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iterator>

#include "command_list.h"
#include "first_texture_shaders.h"
#include "vulkan_bindless.h"
#include "vulkan_replay.h"

namespace {

constexpr std::uint32_t kSize = 64;
constexpr std::uint64_t kVertexVa = 0x100000000ull;
constexpr std::uint64_t kIndexVa = 0x200000000ull;

struct BufferPair {
    VkBuffer vb = VK_NULL_HANDLE;
    VkBuffer ib = VK_NULL_HANDLE;
};

VkBuffer lookup_buffer(std::uint64_t va, std::uint64_t* base, void* user) {
    const auto* pair = static_cast<const BufferPair*>(user);
    if (va == kVertexVa) {
        *base = kVertexVa;
        return pair->vb;
    }
    if (va == kIndexVa) {
        *base = kIndexVa;
        return pair->ib;
    }
    return VK_NULL_HANDLE;
}

VkPipeline lookup_pipeline(std::uintptr_t, void* user) { return *static_cast<VkPipeline*>(user); }

int fail(const char* what) {
    std::fprintf(stderr, "first_texture_test: %s\n", what);
    return 1;
}

}  // namespace

int main() {
    vktest::Device device;
    if (!device.create()) {
        std::printf("first_texture_test: no Vulkan device\n");
        return 77;
    }

    VkPhysicalDeviceMemoryProperties memory{};
    device.instance_fn<PFN_vkGetPhysicalDeviceMemoryProperties>("vkGetPhysicalDeviceMemoryProperties")(
        device.physical, &memory);
    leonida::BindlessSet bindless(device.device_proc, device.device, memory);
    if (!bindless.ok()) return fail("BindlessSet creation failed");

    // 2x2 checkerboard: red, green, green, red.
    const std::uint8_t texels[16] = {255, 0, 0, 255, 0, 255, 0, 255, 0, 255, 0, 255, 255, 0, 0, 255};
    void* staging = nullptr;
    const VkBuffer upload = device.buffer(sizeof(texels), VK_BUFFER_USAGE_TRANSFER_SRC_BIT, &staging);
    if (!upload) return fail("staging buffer creation failed");
    std::memcpy(staging, texels, sizeof(texels));

    const VkImage texture = device.image(VK_FORMAT_R8G8B8A8_UNORM, 2, 2, 1, VK_IMAGE_USAGE_SAMPLED_BIT);
    if (!texture) return fail("texture creation failed");
    VkImageViewCreateInfo view_info{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    view_info.image = texture;
    view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
    view_info.format = VK_FORMAT_R8G8B8A8_UNORM;
    view_info.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    VkImageView texture_view = VK_NULL_HANDLE;
    if (device.fn<PFN_vkCreateImageView>("vkCreateImageView")(device.device, &view_info, nullptr, &texture_view) !=
        VK_SUCCESS)
        return fail("texture view creation failed");

    VkSamplerCreateInfo sampler_info{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    sampler_info.magFilter = VK_FILTER_NEAREST;
    sampler_info.minFilter = VK_FILTER_NEAREST;
    sampler_info.addressModeU = sampler_info.addressModeV = sampler_info.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    VkSampler sampler = VK_NULL_HANDLE;
    if (device.fn<PFN_vkCreateSampler>("vkCreateSampler")(device.device, &sampler_info, nullptr, &sampler) != VK_SUCCESS)
        return fail("sampler creation failed");

    const std::uint32_t image_index = bindless.add_image(leonida::ImageDimension::dim_2d, texture_view,
                                                         VK_IMAGE_LAYOUT_GENERAL);
    const std::uint32_t sampler_index = bindless.add_sampler(sampler);
    if (!image_index || !sampler_index) return fail("bindless registration failed");
    if (!bindless.flush()) return fail("bindless flush failed");

    const VkShaderModule vert = [&] {
        VkShaderModuleCreateInfo info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        info.codeSize = sizeof(first_texture::quad_vert_spv);
        info.pCode = first_texture::quad_vert_spv;
        VkShaderModule m = VK_NULL_HANDLE;
        device.fn<PFN_vkCreateShaderModule>("vkCreateShaderModule")(device.device, &info, nullptr, &m);
        return m;
    }();
    const VkShaderModule frag = [&] {
        VkShaderModuleCreateInfo info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        info.codeSize = sizeof(first_texture::quad_frag_spv);
        info.pCode = first_texture::quad_frag_spv;
        VkShaderModule m = VK_NULL_HANDLE;
        device.fn<PFN_vkCreateShaderModule>("vkCreateShaderModule")(device.device, &info, nullptr, &m);
        return m;
    }();
    if (!vert || !frag) return fail("shader module creation failed");

    VkPushConstantRange push{};
    push.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    push.offset = 0;
    push.size = 144;
    const VkDescriptorSetLayout set_layout = bindless.layout();
    VkPipelineLayoutCreateInfo layout_info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    layout_info.setLayoutCount = 1;
    layout_info.pSetLayouts = &set_layout;
    layout_info.pushConstantRangeCount = 1;
    layout_info.pPushConstantRanges = &push;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    if (device.fn<PFN_vkCreatePipelineLayout>("vkCreatePipelineLayout")(device.device, &layout_info, nullptr, &layout) !=
        VK_SUCCESS)
        return fail("pipeline layout creation failed");

    VkPipelineShaderStageCreateInfo stages[2]{};
    stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vert;
    stages[0].pName = "main";
    stages[1] = stages[0];
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = frag;

    VkVertexInputBindingDescription binding{0, 4 * sizeof(float), VK_VERTEX_INPUT_RATE_VERTEX};
    VkVertexInputAttributeDescription attributes[2]{};
    attributes[0] = {0, 0, VK_FORMAT_R32G32_SFLOAT, 0};
    attributes[1] = {1, 0, VK_FORMAT_R32G32_SFLOAT, 2 * sizeof(float)};
    VkPipelineVertexInputStateCreateInfo vertex_input{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    vertex_input.vertexBindingDescriptionCount = 1;
    vertex_input.pVertexBindingDescriptions = &binding;
    vertex_input.vertexAttributeDescriptionCount = 2;
    vertex_input.pVertexAttributeDescriptions = attributes;
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
    blend_attachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT |
                                      VK_COLOR_COMPONENT_A_BIT;
    VkPipelineColorBlendStateCreateInfo blend{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    blend.attachmentCount = 1;
    blend.pAttachments = &blend_attachment;
    const VkDynamicState dynamic_states[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dynamic{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
    dynamic.dynamicStateCount = 2;
    dynamic.pDynamicStates = dynamic_states;

    VkFormat color_format = VK_FORMAT_R8G8B8A8_UNORM;
    VkPipelineRenderingCreateInfo rendering{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
    rendering.colorAttachmentCount = 1;
    rendering.pColorAttachmentFormats = &color_format;
    VkGraphicsPipelineCreateInfo pipeline_info{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    pipeline_info.pNext = &rendering;
    pipeline_info.stageCount = 2;
    pipeline_info.pStages = stages;
    pipeline_info.pVertexInputState = &vertex_input;
    pipeline_info.pInputAssemblyState = &assembly;
    pipeline_info.pViewportState = &viewport_state;
    pipeline_info.pRasterizationState = &raster;
    pipeline_info.pMultisampleState = &multisample;
    pipeline_info.pColorBlendState = &blend;
    pipeline_info.pDynamicState = &dynamic;
    pipeline_info.layout = layout;
    VkPipeline pipeline = VK_NULL_HANDLE;
    if (device.fn<PFN_vkCreateGraphicsPipelines>("vkCreateGraphicsPipelines")(device.device, VK_NULL_HANDLE, 1,
                                                                            &pipeline_info, nullptr, &pipeline) !=
        VK_SUCCESS)
        return fail("graphics pipeline creation failed");

    // Quad: two triangles, positions and uvs, indexed.
    const float vertices[16] = {-1, -1, 0, 1, 1, -1, 1, 1, 1, 1, 1, 0, -1, 1, 0, 0};
    const std::uint16_t indices[6] = {0, 1, 2, 2, 3, 0};
    void* vertex_data = nullptr;
    void* index_data = nullptr;
    const VkBuffer vertex_buffer = device.buffer(sizeof(vertices), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, &vertex_data);
    const VkBuffer index_buffer = device.buffer(sizeof(indices), VK_BUFFER_USAGE_INDEX_BUFFER_BIT, &index_data);
    if (!vertex_buffer || !index_buffer) return fail("vertex/index buffer creation failed");
    std::memcpy(vertex_data, vertices, sizeof(vertices));
    std::memcpy(index_data, indices, sizeof(indices));
    BufferPair buffers{vertex_buffer, index_buffer};

    const VkImage target = device.image(VK_FORMAT_R8G8B8A8_UNORM, kSize, kSize, 1, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT);
    if (!target) return fail("render target creation failed");
    VkImageViewCreateInfo target_view_info = view_info;
    target_view_info.image = target;
    VkImageView target_view = VK_NULL_HANDLE;
    if (device.fn<PFN_vkCreateImageView>("vkCreateImageView")(device.device, &target_view_info, nullptr,
                                                              &target_view) != VK_SUCCESS)
        return fail("render target view creation failed");

    void* mapped = nullptr;
    const VkBuffer readback = device.buffer(VkDeviceSize(kSize) * kSize * 4, VK_BUFFER_USAGE_TRANSFER_DST_BIT, &mapped);
    if (!readback) return fail("readback buffer creation failed");

    // The IR: IA through the command list, then a pipeline and one indexed draw.
    leonida::CommandList list;
    const leonida::VertexBufferBinding vertex_binding{kVertexVa, 4 * sizeof(float), 0};
    list.set_vertex_buffers(0, {&vertex_binding, 1});
    list.set_index_buffer(kIndexVa, 0, 57 /*DXGI_FORMAT_R16_UINT*/);
    list.set_pipeline_state(0x1000);
    list.draw_indexed_instanced(6);
    list.close();

    const bool recorded = device.run([&](VkCommandBuffer command_buffer) {
        bindless.record_fallback_init(command_buffer);
        // Upload the checkerboard: UNDEFINED -> TRANSFER_DST -> GENERAL (the layout the bindless descriptor names).
        device.initial_layout(command_buffer, texture, VK_IMAGE_ASPECT_COLOR_BIT, 1, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        VkBufferImageCopy copy{};
        copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        copy.imageExtent = {2, 2, 1};
        device.fn<PFN_vkCmdCopyBufferToImage>("vkCmdCopyBufferToImage")(command_buffer, upload, texture,
                                                                      VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);
        VkImageMemoryBarrier to_sampled{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        to_sampled.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        to_sampled.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        to_sampled.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        to_sampled.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        to_sampled.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        to_sampled.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        to_sampled.image = texture;
        to_sampled.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        device.fn<PFN_vkCmdPipelineBarrier>("vkCmdPipelineBarrier")(
            command_buffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr, 0,
            nullptr, 1, &to_sampled);

        device.initial_layout(command_buffer, target, VK_IMAGE_ASPECT_COLOR_BIT, 1,
                              VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
        VkRenderingAttachmentInfo color{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
        color.imageView = target_view;
        color.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        color.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        color.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        color.clearValue = {{{1.0f, 1.0f, 1.0f, 1.0f}}};
        VkRenderingInfo rendering_info{VK_STRUCTURE_TYPE_RENDERING_INFO};
        rendering_info.renderArea = {{0, 0}, {kSize, kSize}};
        rendering_info.layerCount = 1;
        rendering_info.colorAttachmentCount = 1;
        rendering_info.pColorAttachments = &color;
        device.fn<PFN_vkCmdBeginRendering>("vkCmdBeginRendering")(command_buffer, &rendering_info);

        const std::uint32_t push_values[2] = {image_index, sampler_index};
        device.fn<PFN_vkCmdPushConstants>("vkCmdPushConstants")(command_buffer, layout,
                                                               VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(push_values),
                                                               push_values);

        leonida::ReplayFunctions replay{};
        replay.bind_pipeline = device.fn<PFN_vkCmdBindPipeline>("vkCmdBindPipeline");
        replay.bind_descriptor_sets = device.fn<PFN_vkCmdBindDescriptorSets>("vkCmdBindDescriptorSets");
        replay.bind_vertex_buffers = device.fn<PFN_vkCmdBindVertexBuffers>("vkCmdBindVertexBuffers");
        replay.bind_index_buffer = device.fn<PFN_vkCmdBindIndexBuffer>("vkCmdBindIndexBuffer");
        replay.set_viewport = device.fn<PFN_vkCmdSetViewport>("vkCmdSetViewport");
        replay.set_scissor = device.fn<PFN_vkCmdSetScissor>("vkCmdSetScissor");
        replay.draw_indexed = device.fn<PFN_vkCmdDrawIndexed>("vkCmdDrawIndexed");
        replay.pipeline_lookup = lookup_pipeline;
        replay.pipeline_user = &pipeline;
        replay.buffer_lookup = lookup_buffer;
        replay.buffer_user = &buffers;
        replay.bindless_layout = layout;
        replay.bindless_set = bindless.set();
        replay.viewport = {0.0f, 0.0f, float(kSize), float(kSize), 0.0f, 1.0f};
        replay.scissor = {{0, 0}, {kSize, kSize}};

        leonida::PreparedList prepared;
        prepared.list = &list;
        prepared.commands.resize(list.commands().size());
        leonida::emit(prepared, command_buffer, replay);
        device.fn<PFN_vkCmdEndRendering>("vkCmdEndRendering")(command_buffer);

        VkImageMemoryBarrier to_copy{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        to_copy.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        to_copy.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        to_copy.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        to_copy.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        to_copy.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        to_copy.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        to_copy.image = target;
        to_copy.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        device.fn<PFN_vkCmdPipelineBarrier>("vkCmdPipelineBarrier")(
            command_buffer, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0,
            nullptr, 0, nullptr, 1, &to_copy);
        VkBufferImageCopy region{};
        region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        region.imageExtent = {kSize, kSize, 1};
        device.fn<PFN_vkCmdCopyImageToBuffer>("vkCmdCopyImageToBuffer")(command_buffer, target,
                                                                      VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, readback, 1,
                                                                      &region);
        return true;
    });
    if (!recorded) return fail("command buffer recording/submit failed");

    const auto* pixels = static_cast<const std::uint8_t*>(mapped);
    std::size_t red = 0, green = 0, white = 0;
    for (std::uint32_t i = 0; i < kSize * kSize; ++i) {
        const std::uint8_t r = pixels[i * 4], g = pixels[i * 4 + 1];
        if (r > 180 && g < 90) ++red;
        else if (g > 150 && r < 90) ++green;
        else if (r > 240 && g > 240) ++white;
    }
    std::printf("first_texture_test: red=%zu green=%zu white=%zu of %u\n", red, green, white, kSize * kSize);

    device.fn<PFN_vkDestroyPipeline>("vkDestroyPipeline")(device.device, pipeline, nullptr);
    device.fn<PFN_vkDestroyPipelineLayout>("vkDestroyPipelineLayout")(device.device, layout, nullptr);
    device.fn<PFN_vkDestroyShaderModule>("vkDestroyShaderModule")(device.device, vert, nullptr);
    device.fn<PFN_vkDestroyShaderModule>("vkDestroyShaderModule")(device.device, frag, nullptr);
    device.fn<PFN_vkDestroyImageView>("vkDestroyImageView")(device.device, target_view, nullptr);
    device.fn<PFN_vkDestroyImageView>("vkDestroyImageView")(device.device, texture_view, nullptr);
    device.fn<PFN_vkDestroySampler>("vkDestroySampler")(device.device, sampler, nullptr);

    if (!red || !green) return fail("the checkerboard did not reach the render target");
    if (white) return fail("part of the quad is still the clear colour");
    std::printf("first_texture_test: all tests passed\n");
    return 0;
}
