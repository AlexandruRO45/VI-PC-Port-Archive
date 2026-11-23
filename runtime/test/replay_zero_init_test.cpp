// CPU unit tests for vulkan_replay.h lazy UNDEFINED zero-init gating and vkCmdFillBuffer alignment.
// Needs only Vulkan headers (no loader, GPU, shaders, or game data).

#include "../src/vulkan_replay.h"

#include <cstdint>
#include <cstdio>
#include <memory>
#include <vector>

namespace {

using namespace leonida;

int failures = 0;

void expect(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++failures;
    }
}

struct FillCall {
    VkDeviceSize offset = 0;
    VkDeviceSize size = 0;
};

std::vector<FillCall> fill_calls;
unsigned copy_b2i_calls = 0;
unsigned copy_img_calls = 0;
unsigned draw_calls = 0;
unsigned barrier2_calls = 0;
unsigned clear_color_calls = 0;
unsigned clear_depth_calls = 0;

VKAPI_ATTR void VKAPI_CALL mock_barrier(VkCommandBuffer, VkPipelineStageFlags, VkPipelineStageFlags, VkDependencyFlags,
    std::uint32_t, const VkMemoryBarrier*, std::uint32_t, const VkBufferMemoryBarrier*,
    std::uint32_t, const VkImageMemoryBarrier*) {}
VKAPI_ATTR void VKAPI_CALL mock_barrier2(VkCommandBuffer, const VkDependencyInfo*) { ++barrier2_calls; }
VKAPI_ATTR void VKAPI_CALL mock_clear_color(VkCommandBuffer, VkImage, VkImageLayout, const VkClearColorValue*,
    std::uint32_t, const VkImageSubresourceRange*) { ++clear_color_calls; }
VKAPI_ATTR void VKAPI_CALL mock_clear_depth(VkCommandBuffer, VkImage, VkImageLayout, const VkClearDepthStencilValue*,
    std::uint32_t, const VkImageSubresourceRange*) { ++clear_depth_calls; }
VKAPI_ATTR void VKAPI_CALL mock_fill_buffer(VkCommandBuffer, VkBuffer, VkDeviceSize offset, VkDeviceSize size,
    std::uint32_t) { fill_calls.push_back({offset, size}); }
VKAPI_ATTR void VKAPI_CALL mock_copy_buf(VkCommandBuffer, VkBuffer, VkBuffer, std::uint32_t, const VkBufferCopy*) {}
VKAPI_ATTR void VKAPI_CALL mock_copy_b2i(VkCommandBuffer, VkBuffer, VkImage, VkImageLayout, std::uint32_t,
    const VkBufferImageCopy*) { ++copy_b2i_calls; }
VKAPI_ATTR void VKAPI_CALL mock_copy_i2b(VkCommandBuffer, VkImage, VkImageLayout, VkBuffer, std::uint32_t,
    const VkBufferImageCopy*) {}
VKAPI_ATTR void VKAPI_CALL mock_copy_img(VkCommandBuffer, VkImage, VkImageLayout, VkImage, VkImageLayout,
    std::uint32_t, const VkImageCopy*) { ++copy_img_calls; }
VKAPI_ATTR void VKAPI_CALL mock_begin_rendering(VkCommandBuffer, const VkRenderingInfo*) {}
VKAPI_ATTR void VKAPI_CALL mock_end_rendering(VkCommandBuffer) {}
VKAPI_ATTR void VKAPI_CALL mock_bind_pipeline(VkCommandBuffer, VkPipelineBindPoint, VkPipeline) {}
VKAPI_ATTR void VKAPI_CALL mock_bind_vb2(VkCommandBuffer, std::uint32_t, std::uint32_t, const VkBuffer*,
    const VkDeviceSize*, const VkDeviceSize*, const VkDeviceSize*) {}
VKAPI_ATTR void VKAPI_CALL mock_bind_ib(VkCommandBuffer, VkBuffer, VkDeviceSize, VkIndexType) {}
VKAPI_ATTR void VKAPI_CALL mock_set_viewport(VkCommandBuffer, std::uint32_t, std::uint32_t, const VkViewport*) {}
VKAPI_ATTR void VKAPI_CALL mock_set_scissor(VkCommandBuffer, std::uint32_t, std::uint32_t, const VkRect2D*) {}
VKAPI_ATTR void VKAPI_CALL mock_set_topology(VkCommandBuffer, VkPrimitiveTopology) {}
VKAPI_ATTR void VKAPI_CALL mock_set_restart(VkCommandBuffer, VkBool32) {}
VKAPI_ATTR void VKAPI_CALL mock_set_stencil(VkCommandBuffer, VkStencilFaceFlags, std::uint32_t) {}
VKAPI_ATTR void VKAPI_CALL mock_set_blend(VkCommandBuffer, const float[4]) {}
VKAPI_ATTR void VKAPI_CALL mock_push_constants(VkCommandBuffer, VkPipelineLayout, VkShaderStageFlags,
    std::uint32_t, std::uint32_t, const void*) {}
VKAPI_ATTR void VKAPI_CALL mock_draw(VkCommandBuffer, std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t) {
    ++draw_calls;
}
VKAPI_ATTR void VKAPI_CALL mock_draw_indexed(VkCommandBuffer, std::uint32_t, std::uint32_t, std::uint32_t,
    std::int32_t, std::uint32_t) {}

ResourceBinding make_image(std::uintptr_t handle, VkFormat format, VkImageAspectFlags aspects,
                           std::uint32_t w = 4, std::uint32_t h = 4) {
    ResourceBinding b{};
    b.image = reinterpret_cast<VkImage>(handle);
    b.format = format;
    b.aspects = aspects;
    b.width = w;
    b.height = h;
    b.depth = b.mip_levels = b.array_layers = b.samples = 1;
    b.color_attachment = (aspects == VK_IMAGE_ASPECT_COLOR_BIT);
    return b;
}

ReplayFunctions base_draw_functions(const GraphicsPipeline& pipeline) {
    ReplayFunctions vk{};
    vk.pipeline_barrier = mock_barrier;
    vk.pipeline_barrier2 = mock_barrier2;
    vk.begin_rendering = mock_begin_rendering;
    vk.end_rendering = mock_end_rendering;
    vk.bind_pipeline = mock_bind_pipeline;
    vk.bind_vertex_buffers2 = mock_bind_vb2;
    vk.bind_index_buffer = mock_bind_ib;
    vk.set_viewport = mock_set_viewport;
    vk.set_scissor = mock_set_scissor;
    vk.set_primitive_topology = mock_set_topology;
    vk.set_primitive_restart_enable = mock_set_restart;
    vk.set_stencil_reference = mock_set_stencil;
    vk.set_blend_constants = mock_set_blend;
    vk.push_constants = mock_push_constants;
    vk.draw = mock_draw;
    vk.draw_indexed = mock_draw_indexed;
    vk.buffer_at = [](std::uint64_t, std::uint64_t&, VkBuffer&, VkDeviceSize&) { return false; };
    vk.image_view = [](const auto&, const auto&) { return reinterpret_cast<VkImageView>(std::uintptr_t{1}); };
    vk.graphics_pipeline = [&pipeline](std::uintptr_t) { return &pipeline; };
    return vk;
}

ReplayFunctions base_copy_functions() {
    ReplayFunctions vk{};
    vk.pipeline_barrier = mock_barrier;
    vk.copy_buffer = mock_copy_buf;
    vk.copy_buffer_to_image = mock_copy_b2i;
    vk.copy_image_to_buffer = mock_copy_i2b;
    vk.copy_image = mock_copy_img;
    vk.fill_buffer = mock_fill_buffer;
    vk.queue_stages = detail::queue_stages_copy;
    return vk;
}

void reset_counters() {
    fill_calls.clear();
    copy_b2i_calls = copy_img_calls = draw_calls = barrier2_calls = clear_color_calls = clear_depth_calls = 0;
}

void test_draw_attachments_gated_without_clear_fn() {
    GraphicsPipeline pso{};
    pso.pipeline = reinterpret_cast<VkPipeline>(std::uintptr_t{1});
    pso.layout = reinterpret_cast<VkPipelineLayout>(std::uintptr_t{1});
    pso.push_stages = VK_SHADER_STAGE_ALL_GRAPHICS;
    pso.color_formats = {VK_FORMAT_R8G8B8A8_UNORM};
    pso.depth_format = VK_FORMAT_D32_SFLOAT;
    pso.depth_aspects = VK_IMAGE_ASPECT_DEPTH_BIT;

    const ResourceBinding rt = make_image(1, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_ASPECT_COLOR_BIT);
    const ResourceBinding ds = make_image(2, VK_FORMAT_D32_SFLOAT, VK_IMAGE_ASPECT_DEPTH_BIT);
    auto resolve = [&](ResourceId id) -> const ResourceBinding* {
        if (id == 1) return &rt;
        if (id == 2) return &ds;
        return nullptr;
    };

    CommandList list;
    Draw d{};
    d.pipeline = 1;
    auto views = std::make_shared<DrawViews>();
    views->render_targets[0] = TargetView{1, 0, 0, 1};
    views->render_target_count = 1;
    d.views = views;
    d.depth = TargetView{2, 0, 0, 1};
    d.topology = 4;  // D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST
    d.count = 3;
    d.instances = 1;
    d.viewport = Viewport{0, 0, 4, 4, 0, 1};
    d.scissor = Rect{0, 0, 4, 4};
    expect(list.draw(d) && list.close(), "record draw with UNDEFINED RT and DS");

    // Missing clear_color_image -> must return unsupported rather than dereferencing null in emit().
    {
        LayoutTracker tracker;
        tracker.track(1, rt, VK_IMAGE_LAYOUT_UNDEFINED);
        tracker.track(2, ds, VK_IMAGE_LAYOUT_UNDEFINED);
        auto vk = base_draw_functions(pso);
        vk.clear_depth_stencil_image = mock_clear_depth;
        reset_counters();
        expect(replay(list, VK_NULL_HANDLE, resolve, vk, &tracker) == ReplayResult::unsupported,
               "Draw to UNDEFINED color RT without clear_color_image is refused as unsupported");
        expect(draw_calls == 0 && clear_depth_calls == 0, "no commands emitted on refusal");
    }

    // Missing clear_depth_stencil_image -> must return unsupported rather than dereferencing null in emit().
    {
        LayoutTracker tracker;
        tracker.track(1, rt, VK_IMAGE_LAYOUT_UNDEFINED);
        tracker.track(2, ds, VK_IMAGE_LAYOUT_UNDEFINED);
        auto vk = base_draw_functions(pso);
        vk.clear_color_image = mock_clear_color;
        reset_counters();
        expect(replay(list, VK_NULL_HANDLE, resolve, vk, &tracker) == ReplayResult::unsupported,
               "Draw to UNDEFINED depth target without clear_depth_stencil_image is refused as unsupported");
        expect(draw_calls == 0 && clear_color_calls == 0, "no commands emitted on refusal");
    }

    // Both present -> succeeds and clears both attachments before drawing.
    {
        LayoutTracker tracker;
        tracker.track(1, rt, VK_IMAGE_LAYOUT_UNDEFINED);
        tracker.track(2, ds, VK_IMAGE_LAYOUT_UNDEFINED);
        auto vk = base_draw_functions(pso);
        vk.clear_color_image = mock_clear_color;
        vk.clear_depth_stencil_image = mock_clear_depth;
        reset_counters();
        expect(replay(list, VK_NULL_HANDLE, resolve, vk, &tracker) == ReplayResult::success,
               "Draw to UNDEFINED RT and DS succeeds when both clear functions are set");
        expect(clear_color_calls == 1 && clear_depth_calls == 1 && draw_calls == 1,
               "both attachments zero-cleared and draw emitted");
    }
}

void test_copy_queue_and_discard_barrier_not_falsely_refused() {
    const ResourceBinding dst = make_image(1, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_ASPECT_COLOR_BIT);
    const ResourceBinding src = make_image(2, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_ASPECT_COLOR_BIT);
    auto resolve = [&](ResourceId id) -> const ResourceBinding* {
        if (id == 1) return &dst;
        if (id == 2) return &src;
        return nullptr;
    };

    // Copy-queue CopyResource to an UNDEFINED color image uses zero_compressed_image(), never clear_color_image.
    {
        CommandList list;
        expect(list.copy_resource(CopyResource{1, 2}) && list.close(), "record copy_resource");
        LayoutTracker tracker;
        tracker.track(1, dst, VK_IMAGE_LAYOUT_UNDEFINED);
        tracker.track(2, src, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);

        auto vk = base_copy_functions();
        vk.copy_scratch = [](VkDeviceSize, VkBuffer& buf, VkDeviceSize& off) {
            buf = reinterpret_cast<VkBuffer>(std::uintptr_t{0x10});
            off = 0;
            return true;
        };
        reset_counters();
        expect(replay(list, VK_NULL_HANDLE, resolve, vk, &tracker) == ReplayResult::success,
               "COPY queue CopyResource to UNDEFINED color image succeeds without clear_color_image");
        expect(copy_b2i_calls == 1 && copy_img_calls == 1, "copy-queue zero-init and image copy emitted");
    }

    // Discarding / from-undefined TextureBarrier skips initialize() in emit(), so clear_color_image is not needed.
    {
        CommandList list;
        TextureBarrier tb{};
        tb.resource = 1;
        tb.layout_before = d3d12_layout::undefined;
        tb.layout_after = d3d12_layout::render_target;
        tb.sync_before = d3d12_sync::none;
        tb.sync_after = d3d12_sync::render_target;
        tb.access_before = d3d12_access::no_access;
        tb.access_after = d3d12_access::render_target;
        tb.flags = texture_barrier_discard;
        expect(list.barrier(tb) && list.close(), "record discarding TextureBarrier");

        LayoutTracker tracker;
        tracker.track(1, dst, VK_IMAGE_LAYOUT_UNDEFINED);
        ReplayFunctions vk{};
        vk.pipeline_barrier = mock_barrier;
        vk.pipeline_barrier2 = mock_barrier2;
        reset_counters();
        expect(replay(list, VK_NULL_HANDLE, resolve, vk, &tracker) == ReplayResult::success,
               "discarding TextureBarrier from UNDEFINED succeeds without clear_color_image");
        expect(barrier2_calls == 1, "TextureBarrier emitted");
    }
}

void test_zero_compressed_image_fill_buffer_alignment() {
    auto run_case = [](VkFormat format, std::uint32_t w, std::uint32_t h, VkDeviceSize scratch_off,
                       const char* label) {
        const ResourceBinding dst = make_image(1, format, VK_IMAGE_ASPECT_COLOR_BIT, w, h);
        const ResourceBinding src = make_image(2, format, VK_IMAGE_ASPECT_COLOR_BIT, w, h);
        auto resolve = [&](ResourceId id) -> const ResourceBinding* {
            if (id == 1) return &dst;
            if (id == 2) return &src;
            return nullptr;
        };
        CommandList list;
        expect(list.copy_resource(CopyResource{1, 2}) && list.close(), label);
        LayoutTracker tracker;
        tracker.track(1, dst, VK_IMAGE_LAYOUT_UNDEFINED);
        tracker.track(2, src, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);

        auto vk = base_copy_functions();
        vk.copy_scratch = [scratch_off](VkDeviceSize, VkBuffer& buf, VkDeviceSize& off) {
            buf = reinterpret_cast<VkBuffer>(std::uintptr_t{0x20});
            off = scratch_off;
            return true;
        };
        reset_counters();
        expect(replay(list, VK_NULL_HANDLE, resolve, vk, &tracker) == ReplayResult::success, label);
        expect(fill_calls.size() == 1 && copy_b2i_calls == 1, label);
        if (!fill_calls.empty()) {
            expect(fill_calls[0].offset % 4 == 0, "vkCmdFillBuffer dstOffset is a multiple of 4");
            expect(fill_calls[0].size > 0 && fill_calls[0].size % 4 == 0,
                   "vkCmdFillBuffer size is a positive multiple of 4");
        }
    };

    run_case(VK_FORMAT_R8_UNORM, 3, 1, 0, "R8_UNORM 3x1 scratch_off=0");
    run_case(VK_FORMAT_R8G8_UNORM, 3, 1, 0, "R8G8_UNORM 3x1 scratch_off=0");
    run_case(VK_FORMAT_R8G8_UNORM, 2, 1, 2, "R8G8_UNORM 2x1 scratch_off=2");
    run_case(VK_FORMAT_R8_UNORM, 5, 1, 1, "R8_UNORM 5x1 scratch_off=1");
}

}  // namespace

int main() {
    test_draw_attachments_gated_without_clear_fn();
    test_copy_queue_and_discard_barrier_not_falsely_refused();
    test_zero_compressed_image_fill_buffer_alignment();
    if (failures) {
        std::fprintf(stderr, "replay_zero_init_test: %d failure(s)\n", failures);
        return 1;
    }
    std::printf("replay_zero_init_test: ok\n");
    return 0;
}
