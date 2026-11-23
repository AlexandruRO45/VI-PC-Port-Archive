// A packed 4444 texture can be sampleable on a device that cannot render to its native format.
// Clear and draw into a compatible R16_UINT attachment, then fetch through an A4R4G4B4_UNORM_PACK16
// view of the same image. Compare the underlying words and sampled channels with native D3D12 WARP.
#include "vk_test_device.h"
#include "../src/spirv_words.h"
#include <fstream>

namespace {
constexpr unsigned width = 5, height = 3;
constexpr VkFormat formats[] = {VK_FORMAT_R16_UINT, VK_FORMAT_A4R4G4B4_UNORM_PACK16};
constexpr VkImageSubresourceRange color{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

struct Objects {
    vktest::Device& d;
    VkImageView target{}, sampled{};
    VkShaderModule vertex{}, fragment{}, compute{};
    VkDescriptorSetLayout set_layout{};
    VkDescriptorPool pool{};
    VkPipelineLayout layout{};
    VkPipeline draw{}, fetch{};
    ~Objects() {
        for (auto pipe : {draw, fetch}) d.fn<PFN_vkDestroyPipeline>("vkDestroyPipeline")(d.device, pipe, nullptr);
        d.fn<PFN_vkDestroyPipelineLayout>("vkDestroyPipelineLayout")(d.device, layout, nullptr);
        d.fn<PFN_vkDestroyDescriptorPool>("vkDestroyDescriptorPool")(d.device, pool, nullptr);
        d.fn<PFN_vkDestroyDescriptorSetLayout>("vkDestroyDescriptorSetLayout")(d.device, set_layout, nullptr);
        for (auto module : {vertex, fragment, compute})
            d.fn<PFN_vkDestroyShaderModule>("vkDestroyShaderModule")(d.device, module, nullptr);
        for (auto view : {target, sampled})
            d.fn<PFN_vkDestroyImageView>("vkDestroyImageView")(d.device, view, nullptr);
    }
};

#define CHECK(call) do { const VkResult result = (call); if (result != VK_SUCCESS) { \
    std::fprintf(stderr, "%s: %d\n", #call, int(result)); return 1; } } while (0)

VkResult load(vktest::Device& d, const char* path, VkShaderModule& shader) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    auto code = leonida::read_spirv_words(file, file.tellg());
    if (code.empty()) {
        std::fprintf(stderr, "cannot read shader %s\n", path);
        return VK_ERROR_INITIALIZATION_FAILED;
    }
    VkShaderModuleCreateInfo info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    info.codeSize = code.size() * sizeof(code[0]); info.pCode = code.data();
    return d.fn<PFN_vkCreateShaderModule>("vkCreateShaderModule")(d.device, &info, nullptr, &shader);
}

VkResult draw_pipeline(Objects& o) {
    VkPipelineShaderStageCreateInfo stages[] = {
        {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_VERTEX_BIT, o.vertex, "main"},
        {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_FRAGMENT_BIT, o.fragment, "main"}};
    VkPipelineVertexInputStateCreateInfo input{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    const VkViewport vp{0, 0, width, height, 0, 1};
    const VkRect2D scissor{{1, 0}, {3, 1}};
    VkPipelineViewportStateCreateInfo viewport{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    viewport.viewportCount = viewport.scissorCount = 1;
    viewport.pViewports = &vp; viewport.pScissors = &scissor;
    VkPipelineRasterizationStateCreateInfo raster{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    raster.lineWidth = 1;
    VkPipelineMultisampleStateCreateInfo samples{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    samples.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    VkPipelineColorBlendAttachmentState channel{}; channel.colorWriteMask = VK_COLOR_COMPONENT_R_BIT;
    VkPipelineColorBlendStateCreateInfo blend{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    blend.attachmentCount = 1; blend.pAttachments = &channel;
    VkPipelineRenderingCreateInfo rendering{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
    rendering.colorAttachmentCount = 1; rendering.pColorAttachmentFormats = formats;
    VkGraphicsPipelineCreateInfo pipe{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, &rendering};
    pipe.stageCount = 2; pipe.pStages = stages; pipe.pVertexInputState = &input; pipe.pInputAssemblyState = &ia;
    pipe.pViewportState = &viewport; pipe.pRasterizationState = &raster; pipe.pMultisampleState = &samples;
    pipe.pColorBlendState = &blend; pipe.layout = o.layout;
    return o.d.fn<PFN_vkCreateGraphicsPipelines>("vkCreateGraphicsPipelines")(o.d.device, VK_NULL_HANDLE, 1, &pipe, nullptr, &o.draw);
}

int check(vktest::Device& d, char** paths) {
    VkPhysicalDeviceProperties props{};
    d.instance_fn<PFN_vkGetPhysicalDeviceProperties>("vkGetPhysicalDeviceProperties")(d.physical, &props);
    if (props.apiVersion < VK_API_VERSION_1_3) {
        std::puts("SKIP: the packed view uses a Vulkan 1.3 format");
        return 77;
    }
    if (!(d.families[d.family].queueFlags & VK_QUEUE_COMPUTE_BIT)) {
        std::puts("SKIP: the graphics queue cannot dispatch the sampling shader");
        return 77;
    }
    VkFormatProperties packed{};
    d.instance_fn<PFN_vkGetPhysicalDeviceFormatProperties>("vkGetPhysicalDeviceFormatProperties")(d.physical, formats[1], &packed);
    if (!(packed.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT)) {
        std::puts("SKIP: A4R4G4B4 has no sampled-image support");
        return 77;
    }
    VkImageFormatListCreateInfo list{VK_STRUCTURE_TYPE_IMAGE_FORMAT_LIST_CREATE_INFO};
    list.viewFormatCount = 2; list.pViewFormats = formats;
    VkImageCreateInfo image_info{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    image_info.pNext = &list; image_info.flags = VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT;
    image_info.imageType = VK_IMAGE_TYPE_2D; image_info.format = formats[0]; image_info.extent = {width, height, 1};
    image_info.mipLevels = image_info.arrayLayers = 1; image_info.samples = VK_SAMPLE_COUNT_1_BIT;
    image_info.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    VkPhysicalDeviceImageFormatInfo2 query{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_FORMAT_INFO_2};
    query.pNext = &list; query.format = image_info.format; query.type = image_info.imageType;
    query.tiling = image_info.tiling; query.usage = image_info.usage; query.flags = image_info.flags;
    VkImageFormatProperties2 limits{VK_STRUCTURE_TYPE_IMAGE_FORMAT_PROPERTIES_2};
    auto support = d.instance_fn<PFN_vkGetPhysicalDeviceImageFormatProperties2>("vkGetPhysicalDeviceImageFormatProperties2")(
        d.physical, &query, &limits);
    if (support == VK_ERROR_FORMAT_NOT_SUPPORTED) {
        std::puts("SKIP: the mutable R16_UINT attachment is unsupported");
        return 77;
    }
    CHECK(support);
    const VkImage image = d.image(image_info);
    void *raw = nullptr, *rgba = nullptr;
    auto transfer = d.buffer(width * height * sizeof(std::uint16_t), VK_BUFFER_USAGE_TRANSFER_DST_BIT, &raw);
    auto output = d.buffer(width * height * sizeof(std::uint32_t), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &rgba);
    if (!image || !transfer || !output || !raw || !rgba) return 1;

    Objects o{d};
    VkImageViewUsageCreateInfo usage{VK_STRUCTURE_TYPE_IMAGE_VIEW_USAGE_CREATE_INFO};
    VkImageViewCreateInfo view{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    view.pNext = &usage; view.image = image; view.viewType = VK_IMAGE_VIEW_TYPE_2D; view.subresourceRange = color;
    view.format = formats[0]; usage.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    CHECK(d.fn<PFN_vkCreateImageView>("vkCreateImageView")(d.device, &view, nullptr, &o.target));
    view.format = formats[1]; usage.usage = VK_IMAGE_USAGE_SAMPLED_BIT;
    CHECK(d.fn<PFN_vkCreateImageView>("vkCreateImageView")(d.device, &view, nullptr, &o.sampled));

    VkDescriptorSetLayoutBinding bindings[] = {
        {0, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}};
    VkDescriptorSetLayoutCreateInfo set_info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    set_info.bindingCount = 2; set_info.pBindings = bindings;
    CHECK(d.fn<PFN_vkCreateDescriptorSetLayout>("vkCreateDescriptorSetLayout")(d.device, &set_info, nullptr, &o.set_layout));
    VkDescriptorPoolSize sizes[] = {{VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1}, {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1}};
    VkDescriptorPoolCreateInfo pool{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pool.maxSets = 1; pool.poolSizeCount = 2; pool.pPoolSizes = sizes;
    CHECK(d.fn<PFN_vkCreateDescriptorPool>("vkCreateDescriptorPool")(d.device, &pool, nullptr, &o.pool));
    VkDescriptorSetAllocateInfo alloc{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    alloc.descriptorPool = o.pool; alloc.descriptorSetCount = 1; alloc.pSetLayouts = &o.set_layout;
    VkDescriptorSet set{};
    CHECK(d.fn<PFN_vkAllocateDescriptorSets>("vkAllocateDescriptorSets")(d.device, &alloc, &set));
    VkDescriptorImageInfo src{VK_NULL_HANDLE, o.sampled, VK_IMAGE_LAYOUT_GENERAL};
    VkDescriptorBufferInfo dst{output, 0, VK_WHOLE_SIZE};
    VkWriteDescriptorSet writes[] = {
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, set, 0, 0, 1, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &src},
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, set, 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &dst}};
    d.fn<PFN_vkUpdateDescriptorSets>("vkUpdateDescriptorSets")(d.device, 2, writes, 0, nullptr);
    CHECK(load(d, paths[0], o.vertex));
    CHECK(load(d, paths[1], o.fragment));
    CHECK(load(d, paths[2], o.compute));
    VkPipelineLayoutCreateInfo pl{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pl.setLayoutCount = 1; pl.pSetLayouts = &o.set_layout;
    CHECK(d.fn<PFN_vkCreatePipelineLayout>("vkCreatePipelineLayout")(d.device, &pl, nullptr, &o.layout));
    VkComputePipelineCreateInfo pipeline{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    pipeline.layout = o.layout;
    pipeline.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_COMPUTE_BIT, o.compute, "main"};
    CHECK(d.fn<PFN_vkCreateComputePipelines>("vkCreateComputePipelines")(d.device, VK_NULL_HANDLE, 1, &pipeline, nullptr, &o.fetch));
    CHECK(draw_pipeline(o));

    if (!d.run([&](VkCommandBuffer cmd) {
        d.initial_layout(cmd, image, VK_IMAGE_ASPECT_COLOR_BIT, 1, VK_IMAGE_LAYOUT_GENERAL);
        VkRenderingAttachmentInfo attachment{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
        attachment.imageView = o.target; attachment.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR; attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        attachment.clearValue.color.uint32[0] = 0xcf63;
        VkRenderingInfo pass{VK_STRUCTURE_TYPE_RENDERING_INFO};
        pass.renderArea.extent = {width, height}; pass.layerCount = pass.colorAttachmentCount = 1;
        pass.pColorAttachments = &attachment;
        d.fn<PFN_vkCmdBeginRendering>("vkCmdBeginRendering")(cmd, &pass);
        VkClearAttachment part{}; part.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT; part.clearValue.color.uint32[0] = 0x281d;
        const VkClearRect rect{{{1, 1}, {3, 2}}, 0, 1};
        d.fn<PFN_vkCmdClearAttachments>("vkCmdClearAttachments")(cmd, 1, &part, 1, &rect);
        d.fn<PFN_vkCmdBindPipeline>("vkCmdBindPipeline")(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, o.draw);
        d.fn<PFN_vkCmdDraw>("vkCmdDraw")(cmd, 3, 1, 0, 0);
        d.fn<PFN_vkCmdEndRendering>("vkCmdEndRendering")(cmd);
        auto barrier = d.fn<PFN_vkCmdPipelineBarrier>("vkCmdPipelineBarrier");
        VkMemoryBarrier visible{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                                VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT};
        barrier(cmd, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &visible, 0, nullptr, 0, nullptr);
        d.read_back(cmd, image, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_ASPECT_COLOR_BIT, width, height, 1, transfer, 0);
        d.fn<PFN_vkCmdBindPipeline>("vkCmdBindPipeline")(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, o.fetch);
        d.fn<PFN_vkCmdBindDescriptorSets>("vkCmdBindDescriptorSets")(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, o.layout, 0, 1, &set, 0, nullptr);
        d.fn<PFN_vkCmdDispatch>("vkCmdDispatch")(cmd, width, height, 1);
        VkMemoryBarrier host{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT};
        barrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &host, 0, nullptr, 0, nullptr);
        return true;
    })) return 1;
    for (unsigned y = 0; y < height; ++y) {
        for (unsigned x = 0; x < width; ++x) {
            const bool part = y >= 1 && x >= 1 && x < 4;
            unsigned expected_word = part ? 0x281d : 0xcf63;
            unsigned expected_rgba = part ? 0x22dd1188u : 0xcc3366ffu;
            if (y == 0 && x >= 1 && x < 4) {
                expected_word = 0x90b2u | (x << 8);
                expected_rgba = 0x9922bb00u | (x * 17);
            }
            const auto i = y * width + x;
            auto word = static_cast<const std::uint16_t*>(raw)[i];
            auto channels = static_cast<const std::uint32_t*>(rgba)[i];
            if (word != expected_word || channels != expected_rgba) {
                std::fprintf(stderr, "(%u,%u): packed %04x RGBA8 %08x, expected %04x %08x\n", x, y,
                             unsigned(word), unsigned(channels), expected_word, expected_rgba);
                return 1;
            }
        }
    }
    std::printf("%s: all %ux%u packed words and sampled RGBA values matched\n", props.deviceName, width, height);
    return 0;
}
}  // namespace

int main(int argc, char** argv) {
    if (argc != 4) { std::fprintf(stderr, "usage: vk_packed_rt_test vertex.spv fragment.spv compute.spv\n"); return 1; }
    vktest::Device device;
    auto status = device.initialize();
    if (status == vktest::DeviceStatus::unavailable) return 77;
    if (status != vktest::DeviceStatus::ready) return 1;
    const int result = check(device, argv + 1);
    return device.validation_errors() ? 1 : result;
}
