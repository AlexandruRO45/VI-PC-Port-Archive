// The geometry fallback's draw path (SHADER-1092) replayed on a real device: the exact command sequence the
// D3D12 layer records for a geometry-shader PSO, against the stages its PSO path resolves. Two compute passes
// -- the vertex shader fused with its fetch (tools/gs_compute_stages.py --vs-input on vs_pair.vert: one
// invocation per input-primitive vertex slot) and the capture pass (tools/spirv_gs_to_compute.py --buffers bda
// on gs_double.geom) -- and then the record-driven draw (the generated draw VS with ps_draw.frag). Every buffer
// is addressed through the push-constant block as the layer's passes are: a console VA relative to one
// GPU-address heap, which also holds the vertex and index data, the tint the pixel shader reads through its
// root CBV, and the pairing-facts parameters block. No descriptor sets, no vertex input, no Vulkan geometry
// stage.
//
// gs_double.geom emits each triangle twice, the second copy one clip-space unit to the right. A quad drawn in
// the top-left quadrant of a 64x64 target cleared blue comes out red across the whole top half. Three index
// kinds reach the same image -- non-indexed, u16 and u32 -- and vs_pair's colour at location 1, which the
// geometry shader does not read, is the unconsumed output the fused pass stores to its scratch region: a leak
// into the capture rows would move or recolour the quad.
//
// Skipped without a Vulkan device, or without bufferDeviceAddress (ctest's SKIP_RETURN_CODE). Shaders:
// test/glsl/vs_pair.vert, gs_double.geom, ps_draw.frag. No game data.
// Set LEONIDA_VK_VALIDATION=1 to run under VK_LAYER_KHRONOS_validation.
#include "vk_test_device.h"
#include "../src/vulkan_replay.h"

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <map>
#include <memory>
#include <vector>

#if !defined(LEONIDA_TEST_VS_INPUT_SPV) || !defined(LEONIDA_TEST_CAPTURE_SPV) || !defined(LEONIDA_TEST_DRAW_VS_SPV) \
    || !defined(LEONIDA_TEST_PS_SPV)
#error "LEONIDA_TEST_VS_INPUT_SPV, LEONIDA_TEST_CAPTURE_SPV, LEONIDA_TEST_DRAW_VS_SPV and LEONIDA_TEST_PS_SPV must \
name the generated stages and the pixel shader"
#endif

#define EXPECT(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return 1; } \
} while (0)

namespace {

using namespace leonida;

std::vector<std::uint32_t> spv_words(const char* path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) return {};
    const std::vector<std::uint8_t> bytes((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    if (bytes.size() < 4 || bytes.size() % 4) return {};
    std::vector<std::uint32_t> words(bytes.size() / 4);
    if (!words.empty()) std::memcpy(words.data(), bytes.data(), bytes.size());
    return words;
}

}  // namespace

int main() {
    vktest::Device d;
    d.request_buffer_device_address = true;
    const auto device_status = d.initialize();
    if (device_status == vktest::DeviceStatus::unavailable) return 77;  // no Vulkan GPU (hosted CI): ctest skips
    EXPECT(device_status == vktest::DeviceStatus::ready);
    if (!d.buffer_device_address) {
        std::printf("SKIP: no bufferDeviceAddress on this device\n");
        return 77;
    }

    const std::uint32_t size = 64, half = size / 2;
    // The one heap every console VA is an offset into.
    constexpr std::uint64_t heap_size = 2048;
    constexpr std::uint64_t vertices_va = 0, list_va = 0x80, indices16_va = 0x200, indices32_va = 0x220,
                             tint_va = 0x240, params_va = 0x260, input_va = 0x280, record_va = 0x400,
                             meta_va = 0x580;
    void* heap_mapped = nullptr;
    const VkBuffer heap = d.buffer(heap_size, 0, &heap_mapped);
    EXPECT(heap && heap_mapped);
    std::uint8_t* heap_at = static_cast<std::uint8_t*>(heap_mapped);

    // vs_pair reads a colour at location 0 and a position at location 1, both R32G32B32A32_FLOAT in one
    // 32-byte stride: colour first, as the input layout orders them by register (COLOR 0, POSITION 1).
    const float red[4] = {1, 0, 0, 1};
    const float quad[4][2] = {{-1, 1}, {0, 1}, {-1, 0}, {0, 0}};
    for (std::uint32_t v = 0; v < 4; ++v) {
        float* out = reinterpret_cast<float*>(heap_at + vertices_va + 32 * v);
        std::memcpy(out, red, 16);
        out[4] = quad[v][0];
        out[5] = quad[v][1];
        out[6] = 0;
        out[7] = 1;
    }
    // The non-indexed triangle list: the same two triangles as the index buffers.
    for (std::uint32_t i = 0; i < 6; ++i) {
        const std::uint32_t v = i < 3 ? i : (i == 3 ? 2 : (i == 4 ? 1 : 3));
        std::memcpy(heap_at + list_va + 32 * i, heap_at + vertices_va + 32 * v, 32);
    }
    const std::uint16_t indices16[6] = {0, 1, 2, 2, 1, 3};
    std::memcpy(heap_at + indices16_va, indices16, sizeof(indices16));
    const std::uint32_t indices32[6] = {0, 1, 2, 2, 1, 3};
    std::memcpy(heap_at + indices32_va, indices32, sizeof(indices32));
    const float tint[3] = {1, 0, 1};  // red stays red, as in d3d12_geometry_test
    std::memcpy(heap_at + tint_va, tint, sizeof(tint));

    // The parameters block, as the layer writes it at PSO creation: [nin, vin, nout_vs], per output slot
    // (position first, then locations ascending) the geometry shader input slot that consumes it --
    // 0xffffffff for one none does -- then per input location, ascending, the input layout slot and offset.
    const std::uint32_t params[10] = {2, 3, 3, 0, 1, 0xFFFFFFFFu, 0, 0, 0, 16};
    std::memcpy(heap_at + params_va, params, sizeof(params));

    const std::vector<std::uint32_t> input_words = spv_words(LEONIDA_TEST_VS_INPUT_SPV);
    const std::vector<std::uint32_t> capture_words = spv_words(LEONIDA_TEST_CAPTURE_SPV);
    const std::vector<std::uint32_t> draw_vs_words = spv_words(LEONIDA_TEST_DRAW_VS_SPV);
    const std::vector<std::uint32_t> ps_words = spv_words(LEONIDA_TEST_PS_SPV);
    EXPECT(!input_words.empty() && !capture_words.empty() && !draw_vs_words.empty() && !ps_words.empty());

    // -- the pipelines, as the layer's PSO path builds them: push-constant-only layouts, no descriptor sets --
    const auto create_module = [&](const std::vector<std::uint32_t>& words) -> VkShaderModule {
        VkShaderModuleCreateInfo info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        info.codeSize = words.size() * 4;
        info.pCode = words.data();
        VkShaderModule module = VK_NULL_HANDLE;
        if (d.fn<PFN_vkCreateShaderModule>("vkCreateShaderModule")(d.device, &info, nullptr, &module) != VK_SUCCESS)
            return VK_NULL_HANDLE;
        return module;
    };
    const VkShaderModule input_module = create_module(input_words);
    const VkShaderModule capture_module = create_module(capture_words);
    const VkShaderModule draw_vs_module = create_module(draw_vs_words);
    const VkShaderModule ps_module = create_module(ps_words);
    EXPECT(input_module && capture_module && draw_vs_module && ps_module);
    const auto create_layout = [&](VkShaderStageFlags stages, std::uint32_t bytes) -> VkPipelineLayout {
        const VkPushConstantRange push{stages, 0, bytes};
        VkPipelineLayoutCreateInfo info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        info.pushConstantRangeCount = 1;
        info.pPushConstantRanges = &push;
        VkPipelineLayout layout = VK_NULL_HANDLE;
        if (d.fn<PFN_vkCreatePipelineLayout>("vkCreatePipelineLayout")(d.device, &info, nullptr, &layout) !=
            VK_SUCCESS)
            return VK_NULL_HANDLE;
        return layout;
    };
    // the fused pass reads 252 bytes (its own tail and the per-slot table), the capture 168, the draw 164
    const VkPipelineLayout input_layout = create_layout(VK_SHADER_STAGE_COMPUTE_BIT,
                                                        sizeof(ShaderUserData) + 108);
    const VkPipelineLayout capture_layout = create_layout(VK_SHADER_STAGE_COMPUTE_BIT,
                                                          sizeof(ShaderUserData) + 24);
    const VkPipelineLayout graphics_layout = create_layout(VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                                                           sizeof(ShaderUserData) + 20);
    EXPECT(input_layout && capture_layout && graphics_layout);

    ComputePipeline input_pipeline{}, capture_pipeline{};
    const auto create_compute = [&](VkShaderModule module, VkPipelineLayout layout, ComputePipeline& out) {
        VkComputePipelineCreateInfo info{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        info.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        info.stage.module = module;
        info.stage.pName = "main";
        info.layout = layout;
        return d.fn<PFN_vkCreateComputePipelines>("vkCreateComputePipelines")(d.device, VK_NULL_HANDLE, 1, &info,
                                                                              nullptr, &out.pipeline) == VK_SUCCESS;
    };
    EXPECT(create_compute(input_module, input_layout, input_pipeline));
    EXPECT(create_compute(capture_module, capture_layout, capture_pipeline));
    input_pipeline.layout = input_layout;
    capture_pipeline.layout = capture_layout;

    // The record-driven draw: no vertex input (the fused pass fetched it), a triangle list, and the dynamic
    // state the replay sets at each draw, as the layer's pipelines declare.
    const VkDynamicState dynamic_states[]{VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR,
                                          VK_DYNAMIC_STATE_PRIMITIVE_TOPOLOGY,
                                          VK_DYNAMIC_STATE_PRIMITIVE_RESTART_ENABLE,
                                          VK_DYNAMIC_STATE_STENCIL_REFERENCE, VK_DYNAMIC_STATE_BLEND_CONSTANTS};
    const VkPipelineDynamicStateCreateInfo dynamic{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO, nullptr, 0,
                                                   std::uint32_t(std::size(dynamic_states)), dynamic_states};
    const VkPipelineVertexInputStateCreateInfo vertex_input{
        VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    const VkPipelineInputAssemblyStateCreateInfo assembly{
        VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO, nullptr, 0,
        VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, VK_FALSE};
    const VkViewport one_viewport{0, 0, 1, 1, 0, 1};
    const VkRect2D one_scissor{{0, 0}, {1, 1}};
    const VkPipelineViewportStateCreateInfo viewports{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO, nullptr,
                                                      0, 1, &one_viewport, 1, &one_scissor};
    const VkPipelineRasterizationStateCreateInfo raster{
        VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO, nullptr, 0, VK_FALSE, VK_FALSE,
        VK_POLYGON_MODE_FILL, VK_CULL_MODE_NONE, VK_FRONT_FACE_CLOCKWISE, VK_FALSE, 0, 0, 0, 1};
    const VkPipelineMultisampleStateCreateInfo multisample{
        VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO, nullptr, 0, VK_SAMPLE_COUNT_1_BIT, VK_FALSE, 1,
        nullptr, VK_FALSE, VK_FALSE};
    const VkPipelineColorBlendAttachmentState blend_attachment{
        VK_FALSE, VK_BLEND_FACTOR_ONE, VK_BLEND_FACTOR_ZERO, VK_BLEND_OP_ADD, VK_BLEND_FACTOR_ONE,
        VK_BLEND_FACTOR_ZERO, VK_BLEND_OP_ADD,
        VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT};
    const VkPipelineColorBlendStateCreateInfo blend{
        VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO, nullptr, 0, VK_FALSE, VK_LOGIC_OP_CLEAR, 1,
        &blend_attachment, {1, 1, 1, 1}};
    const VkPipelineShaderStageCreateInfo stages[]{
        {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_VERTEX_BIT, draw_vs_module,
         "main", nullptr},
        {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_FRAGMENT_BIT, ps_module,
         "main", nullptr}};
    // The pipeline's rendering formats, as the layer's pipelines carry them: one R8G8B8A8 colour attachment.
    const VkFormat color_format = VK_FORMAT_R8G8B8A8_UNORM;
    const VkPipelineRenderingCreateInfo rendering{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO, nullptr, 0, 1,
                                                  &color_format, VK_FORMAT_UNDEFINED, VK_FORMAT_UNDEFINED};
    VkGraphicsPipelineCreateInfo graphics_info{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    graphics_info.pNext = &rendering;
    graphics_info.stageCount = 2;
    graphics_info.pStages = stages;
    graphics_info.pVertexInputState = &vertex_input;
    graphics_info.pInputAssemblyState = &assembly;
    graphics_info.pViewportState = &viewports;
    graphics_info.pRasterizationState = &raster;
    graphics_info.pMultisampleState = &multisample;
    graphics_info.pColorBlendState = &blend;
    graphics_info.pDynamicState = &dynamic;
    graphics_info.layout = graphics_layout;
    graphics_info.renderPass = VK_NULL_HANDLE;  // dynamic rendering
    graphics_info.subpass = 0;
    GraphicsPipeline graphics{};
    EXPECT(d.fn<PFN_vkCreateGraphicsPipelines>("vkCreateGraphicsPipelines")(d.device, VK_NULL_HANDLE, 1,
                                                                            &graphics_info, nullptr,
                                                                            &graphics.pipeline) == VK_SUCCESS);
    graphics.layout = graphics_layout;
    graphics.push_stages = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    graphics.color_formats = {VK_FORMAT_R8G8B8A8_UNORM};

    // Internal ids, as the layer mints them.
    constexpr std::uintptr_t input_id = 1, capture_id = 2, graphics_id = 3;
    const std::map<std::uintptr_t, ComputePipeline> computes{{input_id, input_pipeline},
                                                             {capture_id, capture_pipeline}};
    const std::map<std::uintptr_t, GraphicsPipeline> graphs{{graphics_id, graphics}};

    // The target, its binding, and the view the replay's attachments resolve it to.
    const VkImage rt = d.image(VK_FORMAT_R8G8B8A8_UNORM, size, size, 1, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT);
    EXPECT(rt);
    constexpr ResourceId rt_id = 1;
    ResourceBinding rt_binding{};
    rt_binding.image = rt;
    rt_binding.format = VK_FORMAT_R8G8B8A8_UNORM;
    rt_binding.aspects = VK_IMAGE_ASPECT_COLOR_BIT;
    rt_binding.width = size;
    rt_binding.height = size;
    rt_binding.mip_levels = 1;
    rt_binding.array_layers = 1;
    rt_binding.depth = 1;
    rt_binding.samples = 1;
    VkImageView rt_view = VK_NULL_HANDLE;
    {
        VkImageViewCreateInfo info{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        info.image = rt;
        info.viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
        info.format = rt_binding.format;
        info.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        EXPECT(d.fn<PFN_vkCreateImageView>("vkCreateImageView")(d.device, &info, nullptr, &rt_view) == VK_SUCCESS);
    }
    auto views = std::make_shared<DrawViews>();
    views->render_targets[0] = TargetView{rt_id, 0, 0, 1, ColorViewFormat::resource};
    views->render_target_count = 1;

    void* readback_mapped = nullptr;
    const VkBuffer readback = d.buffer(std::uint64_t(size) * size * 4, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                       &readback_mapped);
    EXPECT(readback && readback_mapped);

    // The root arguments, one 32-dword array for every push: s0:s1 move the vertex (the offset (0, 0)), and
    // s2:s3 are the tint's console VA, which the pixel shader reads at the heap base.
    std::array<std::uint32_t, 32> user_data{};
    user_data[2] = std::uint32_t(tint_va);
    user_data[3] = std::uint32_t(tint_va >> 32);

    ReplayFunctions vk{};
    vk.pipeline_barrier = d.fn<PFN_vkCmdPipelineBarrier>("vkCmdPipelineBarrier");  // attachment transitions
    vk.bind_pipeline = d.fn<PFN_vkCmdBindPipeline>("vkCmdBindPipeline");
    vk.push_constants = d.fn<PFN_vkCmdPushConstants>("vkCmdPushConstants");
    vk.dispatch = d.fn<PFN_vkCmdDispatch>("vkCmdDispatch");
    vk.pipeline_barrier2 = d.fn<PFN_vkCmdPipelineBarrier2>("vkCmdPipelineBarrier2");
    vk.begin_rendering = d.fn<PFN_vkCmdBeginRendering>("vkCmdBeginRendering");
    vk.end_rendering = d.fn<PFN_vkCmdEndRendering>("vkCmdEndRendering");
    vk.set_viewport = d.fn<PFN_vkCmdSetViewport>("vkCmdSetViewport");
    vk.set_scissor = d.fn<PFN_vkCmdSetScissor>("vkCmdSetScissor");
    vk.set_primitive_topology = d.fn<PFN_vkCmdSetPrimitiveTopology>("vkCmdSetPrimitiveTopology");
    vk.set_primitive_restart_enable = d.fn<PFN_vkCmdSetPrimitiveRestartEnable>("vkCmdSetPrimitiveRestartEnable");
    vk.set_stencil_reference = d.fn<PFN_vkCmdSetStencilReference>("vkCmdSetStencilReference");
    vk.set_blend_constants = d.fn<PFN_vkCmdSetBlendConstants>("vkCmdSetBlendConstants");
    vk.draw = d.fn<PFN_vkCmdDraw>("vkCmdDraw");
    vk.draw_indexed = d.fn<PFN_vkCmdDrawIndexed>("vkCmdDrawIndexed");
    vk.bind_vertex_buffers2 = d.fn<PFN_vkCmdBindVertexBuffers2>("vkCmdBindVertexBuffers2");
    vk.bind_index_buffer = d.fn<PFN_vkCmdBindIndexBuffer>("vkCmdBindIndexBuffer");
    vk.compute_pipeline = [&](std::uintptr_t id) {
        const auto it = computes.find(id);
        return it == computes.end() ? nullptr : &it->second;
    };
    vk.graphics_pipeline = [&](std::uintptr_t id) {
        const auto it = graphs.find(id);
        return it == graphs.end() ? nullptr : &it->second;
    };
    vk.buffer_at = [&](std::uint64_t va, std::uint64_t& bytes, VkBuffer& buffer, VkDeviceSize& offset) {
        if (va >= heap_size) return false;
        buffer = heap;
        offset = va;
        bytes = heap_size - va;
        return true;
    };
    vk.image_view = [&](const ResourceBinding&, const VkImageSubresourceRange&) { return rt_view; };
    vk.heap_base = d.device_address(heap);
    vk.separate_depth_stencil_layouts = true;

    // The game's draw: two triangles, so the fused pass runs six vertex slots and the capture two primitives.
    // gs_double emits a strip of up to six vertices: 3 * (6 - 2) = 12 draw vertices per primitive.
    constexpr std::uint32_t prims = 2, slots = 6, per_prim = 12, draw_count = prims * per_prim;
    static_assert(per_prim == 3 * (6 - 2), "the draw count follows gs_double's OutputVertices");

    // One draw through the fallback: the two passes, then the record-driven draw, as the D3D12 layer's
    // recorder emits them. Returns the target's pixels, or an empty vector on a failed expectation.
    const auto draw_case = [&](std::uint32_t kind, std::uint64_t index_va, std::uint64_t vertex_va) {
        const bool ok = [&] {
            CommandList list;
            GsDrawCommands gs{};
            gs.input_id = input_id;
            gs.capture_id = capture_id;
            gs.slots = slots;
            gs.prims = prims;
            // the fused pass's tail: the parameters, input and index VAs, the count, the kind, the base, then
            // the per-slot vertex-buffer table
            gs.input_tail[0] = std::uint32_t(params_va);
            gs.input_tail[1] = std::uint32_t(params_va >> 32);
            gs.input_tail[2] = std::uint32_t(input_va);
            gs.input_tail[3] = std::uint32_t(input_va >> 32);
            gs.input_tail[4] = std::uint32_t(index_va);
            gs.input_tail[5] = std::uint32_t(index_va >> 32);
            gs.input_tail[6] = 6;
            gs.input_tail[7] = kind;
            gs.input_tail[8] = 0;
            gs.input_tail[9] = std::uint32_t(vertex_va);  // slot 0: the vertex buffer's console VA
            gs.input_tail[10] = std::uint32_t(vertex_va >> 32);
            gs.input_tail[11] = 32;
            gs.capture_tail[0] = gs.input_tail[2];
            gs.capture_tail[1] = gs.input_tail[3];
            gs.capture_tail[2] = std::uint32_t(record_va);
            gs.capture_tail[3] = std::uint32_t(record_va >> 32);
            gs.capture_tail[4] = std::uint32_t(meta_va);
            gs.capture_tail[5] = std::uint32_t(meta_va >> 32);
            gs.draw_tail[0] = gs.capture_tail[2];
            gs.draw_tail[1] = gs.capture_tail[3];
            gs.draw_tail[2] = gs.capture_tail[4];
            gs.draw_tail[3] = gs.capture_tail[5];
            gs.draw_tail[4] = prims;
            gs.draw_count = draw_count;

            Dispatch input{};
            input.pipeline = gs.input_id;
            input.user_data = user_data;
            input.tail = gs.input_tail;
            input.tail_words = 27;
            input.x = gs.slots;
            input.y = input.z = 1;
            if (!list.dispatch(std::move(input))) return false;
            if (!list.barrier(GlobalBarrier{d3d12_sync::compute_shading, d3d12_sync::compute_shading,
                                            d3d12_access::common, d3d12_access::common}))
                return false;
            Dispatch capture{};
            capture.pipeline = gs.capture_id;
            capture.user_data = user_data;
            capture.tail = gs.capture_tail;
            capture.tail_words = 6;
            capture.x = gs.prims;
            capture.y = capture.z = 1;
            if (!list.dispatch(std::move(capture))) return false;
            if (!list.barrier(GlobalBarrier{d3d12_sync::compute_shading, d3d12_sync::vertex_shading,
                                            d3d12_access::common, d3d12_access::common}))
                return false;
            Draw draw{};
            draw.pipeline = graphics_id;
            draw.user_data = user_data;
            draw.views = views;
            draw.topology = 4;  // D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST
            draw.viewport = {0, 0, float(size), float(size), 0, 1};
            draw.scissor = {0, 0, int(size), int(size)};
            draw.count = gs.draw_count;
            draw.instances = 1;
            draw.tail = gs.draw_tail;
            draw.tail_words = 5;
            if (!list.draw(std::move(draw))) return false;
            if (!list.close()) return false;

            LayoutTracker tracker(true);
            tracker.track(rt_id, rt_binding, VK_IMAGE_LAYOUT_GENERAL);
            const auto resolve = [&](ResourceId id) -> const ResourceBinding* {
                return id == rt_id ? &rt_binding : nullptr;
            };
            const VkClearColorValue blue{{0, 0, 1, 1}};
            return d.run([&](VkCommandBuffer cb) {
                d.initial_layout(cb, rt, VK_IMAGE_ASPECT_COLOR_BIT, 1, VK_IMAGE_LAYOUT_GENERAL);
                const VkImageSubresourceRange whole{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
                d.fn<PFN_vkCmdClearColorImage>("vkCmdClearColorImage")(cb, rt, VK_IMAGE_LAYOUT_GENERAL, &blue, 1,
                                                                        &whole);
                const ReplayResult replayed = replay(list, cb, resolve, vk, &tracker);
                if (replayed != ReplayResult::success) return false;
                VkImageMemoryBarrier to_copy{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
                to_copy.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
                to_copy.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
                to_copy.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
                to_copy.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
                to_copy.srcQueueFamilyIndex = to_copy.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                to_copy.image = rt;
                to_copy.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
                d.fn<PFN_vkCmdPipelineBarrier>("vkCmdPipelineBarrier")(
                    cb, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0,
                    nullptr, 0, nullptr, 1, &to_copy);
                d.read_back(cb, rt, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_ASPECT_COLOR_BIT, size, size, 1,
                            readback, 0);
                return true;
            });
        }();
        if (!ok) return std::vector<std::uint8_t>{};
        std::vector<std::uint8_t> pixels(std::size_t(size) * size * 4);
        std::memcpy(pixels.data(), readback_mapped, pixels.size());
        return pixels;
    };

    // Three index kinds, three draws: non-indexed over the six-vertex list, and the u16/u32 buffers over the
    // four-vertex one. A non-indexed draw's index VA only has to be readable (the loads always run).
    const std::vector<std::uint8_t> non_indexed = draw_case(2, input_va, list_va);
    EXPECT(!non_indexed.empty());
    const std::vector<std::uint8_t> u16 = draw_case(0, indices16_va, vertices_va);
    EXPECT(!u16.empty());
    const std::vector<std::uint8_t> u32 = draw_case(1, indices32_va, vertices_va);
    EXPECT(!u32.empty());

    unsigned mismatches = 0;
    const auto check = [&](const char* what, const std::vector<std::uint8_t>& pixels) {
        for (std::uint32_t row = 0; row < size; ++row) {
            for (std::uint32_t col = 0; col < size; ++col) {
                // gs_double doubles the quad: the top-left copy and its clip-space +1 twin fill the top half.
                const std::array<std::uint8_t, 4> want = row < half ? std::array<std::uint8_t, 4>{255, 0, 0, 255}
                                                                    : std::array<std::uint8_t, 4>{0, 0, 255, 255};
                const std::uint8_t* got = pixels.data() + std::size_t(row) * size * 4 + col * 4;
                if (std::equal(want.begin(), want.end(), got)) continue;
                if (mismatches++ < 8)
                    std::fprintf(stderr, "%s pixel (%u, %u): got %u,%u,%u,%u want %u,%u,%u,%u\n", what, col, row,
                                 got[0], got[1], got[2], got[3], want[0], want[1], want[2], want[3]);
            }
        }
    };
    check("non-indexed", non_indexed);
    check("u16 indices", u16);
    check("u32 indices", u32);
    EXPECT(u16 == non_indexed);
    EXPECT(u32 == non_indexed);
    EXPECT(!mismatches);
    // Destroy what was created, so a validation run ends with nothing leaked.
    const auto destroy_pipeline = d.fn<PFN_vkDestroyPipeline>("vkDestroyPipeline");
    const auto destroy_layout = d.fn<PFN_vkDestroyPipelineLayout>("vkDestroyPipelineLayout");
    const auto destroy_module = d.fn<PFN_vkDestroyShaderModule>("vkDestroyShaderModule");
    destroy_pipeline(d.device, graphics.pipeline, nullptr);
    destroy_pipeline(d.device, input_pipeline.pipeline, nullptr);
    destroy_pipeline(d.device, capture_pipeline.pipeline, nullptr);
    destroy_layout(d.device, graphics_layout, nullptr);
    destroy_layout(d.device, input_layout, nullptr);
    destroy_layout(d.device, capture_layout, nullptr);
    for (const VkShaderModule module : {input_module, capture_module, draw_vs_module, ps_module})
        destroy_module(d.device, module, nullptr);
    d.fn<PFN_vkDestroyImageView>("vkDestroyImageView")(d.device, rt_view, nullptr);
    // rt, heap and readback stay alive here: d.image()/d.buffer() registered them, and ~Device destroys
    // every registered object once. Destroying them here too left dangling handles for that loop.
    if (d.validation_errors()) {
        std::fprintf(stderr, "%u validation errors\n", d.validation_errors());
        return 1;
    }
    std::printf("non-indexed, u16 and u32 all drew the doubled quad: %ux%u pixels exact\n", size, size);
    return 0;
}
