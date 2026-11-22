// GPU test for the root-signature COM object (PR B): a serialized D3D12 root signature becomes a VkPipelineLayout,
// and a real graphics pipeline builds against it.

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include <vulkan/vulkan.h>

#include "d3d12_com.h"
#include "first_draw_shaders.h"

namespace {

int failures = 0;
#define EXPECT(condition)                                                \
    do {                                                                 \
        if (!(condition)) {                                              \
            std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); \
            ++failures;                                                  \
        }                                                                \
    } while (0)

void push32(std::vector<std::uint8_t>& data, std::uint32_t value) {
    std::uint8_t bytes[4];
    std::memcpy(bytes, &value, 4);
    data.insert(data.end(), bytes, bytes + 4);
}

VkShaderModule module(VkDevice device, const std::uint32_t* words, std::size_t bytes) {
    VkShaderModuleCreateInfo info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    info.codeSize = bytes;
    info.pCode = words;
    VkShaderModule m = VK_NULL_HANDLE;
    if (vkCreateShaderModule(device, &info, nullptr, &m) != VK_SUCCESS) return VK_NULL_HANDLE;
    return m;
}

}  // namespace

int main() {
    ID3D12Device* device = nullptr;
    if (FAILED(rt::d3d12::create_vulkan_device_standalone(__uuidof(ID3D12Device),
                                                          reinterpret_cast<void**>(&device)))) {
        std::printf("root_signature_com_test: no Vulkan device\n");
        return 77;
    }

    // 1.0 root signature: one descriptor table with one 4-descriptor SRV range at t0, no static samplers.
    std::vector<std::uint8_t> blob;
    push32(blob, 1);  // version 1.0
    push32(blob, 0);  // flags
    push32(blob, 1);  // parameters
    push32(blob, 0);  // descriptor table
    push32(blob, 0);  // all
    push32(blob, 1);  // one range
    push32(blob, 0);  // SRV
    push32(blob, 4);  // four descriptors
    push32(blob, 0);  // base register
    push32(blob, 0);  // register space
    push32(blob, 0);  // offset
    push32(blob, 0);  // static samplers

    ID3D12RootSignatureDeserializer* deserializer = nullptr;
    EXPECT(SUCCEEDED(rt::d3d12::create_root_signature_deserializer(blob.data(), blob.size(),
                                                                   __uuidof(ID3D12RootSignatureDeserializer),
                                                                   reinterpret_cast<void**>(&deserializer))));
    EXPECT(deserializer != nullptr);
    if (deserializer) deserializer->Release();

    ID3D12RootSignature* root_signature = nullptr;
    EXPECT(SUCCEEDED(device->CreateRootSignature(0, blob.data(), blob.size(), IID_PPV_ARGS(&root_signature))));
    EXPECT(root_signature != nullptr);
    if (!root_signature) {
        device->Release();
        return 1;
    }

    const VkDevice vk_device = rt::d3d12::vulkan_device(device);
    const VkPipelineLayout layout = rt::d3d12::vulkan_root_signature_layout(root_signature);
    EXPECT(vk_device != VK_NULL_HANDLE);
    EXPECT(layout != VK_NULL_HANDLE);

    if (vk_device && layout) {
        VkShaderModule vert = module(vk_device, first_draw::triangle_vert_spv,
                                     sizeof(first_draw::triangle_vert_spv));
        VkShaderModule frag = module(vk_device, first_draw::triangle_frag_spv,
                                     sizeof(first_draw::triangle_frag_spv));
        EXPECT(vert && frag);
        if (vert && frag) {
            VkPipelineShaderStageCreateInfo stages[2]{};
            stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
            stages[0].module = vert;
            stages[0].pName = "main";
            stages[1] = stages[0];
            stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
            stages[1].module = frag;
            VkVertexInputBindingDescription binding{0, 2 * sizeof(float), VK_VERTEX_INPUT_RATE_VERTEX};
            VkVertexInputAttributeDescription attribute{0, 0, VK_FORMAT_R32G32_SFLOAT, 0};
            VkPipelineVertexInputStateCreateInfo vertex_input{
                VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
            vertex_input.vertexBindingDescriptionCount = 1;
            vertex_input.pVertexBindingDescriptions = &binding;
            vertex_input.vertexAttributeDescriptionCount = 1;
            vertex_input.pVertexAttributeDescriptions = &attribute;
            VkPipelineInputAssemblyStateCreateInfo assembly{
                VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
            assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
            VkPipelineViewportStateCreateInfo viewport_state{
                VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
            viewport_state.viewportCount = 1;
            viewport_state.scissorCount = 1;
            VkPipelineRasterizationStateCreateInfo raster{
                VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
            raster.polygonMode = VK_POLYGON_MODE_FILL;
            raster.cullMode = VK_CULL_MODE_NONE;
            raster.lineWidth = 1.0f;
            VkPipelineMultisampleStateCreateInfo multisample{
                VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
            multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
            VkPipelineColorBlendAttachmentState blend_attachment{};
            blend_attachment.colorWriteMask = 0xf;
            VkPipelineColorBlendStateCreateInfo blend{
                VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
            blend.attachmentCount = 1;
            blend.pAttachments = &blend_attachment;
            const VkDynamicState states[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
            VkPipelineDynamicStateCreateInfo dynamic{
                VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
            dynamic.dynamicStateCount = 2;
            dynamic.pDynamicStates = states;
            VkFormat color_format = VK_FORMAT_R8G8B8A8_UNORM;
            VkPipelineRenderingCreateInfo rendering{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
            rendering.colorAttachmentCount = 1;
            rendering.pColorAttachmentFormats = &color_format;
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
            info.layout = layout;
            VkPipeline pipeline = VK_NULL_HANDLE;
            EXPECT(vkCreateGraphicsPipelines(vk_device, VK_NULL_HANDLE, 1, &info, nullptr, &pipeline) == VK_SUCCESS);
            if (pipeline) vkDestroyPipeline(vk_device, pipeline, nullptr);
            vkDestroyShaderModule(vk_device, vert, nullptr);
            vkDestroyShaderModule(vk_device, frag, nullptr);
        }
    }

    root_signature->Release();
    device->Release();
    if (failures) {
        std::fprintf(stderr, "root_signature_com_test: %d failure(s)\n", failures);
        return 1;
    }
    std::printf("root_signature_com_test: all tests passed\n");
    return 0;
}
