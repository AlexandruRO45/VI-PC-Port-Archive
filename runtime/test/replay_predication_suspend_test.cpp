// Conditional rendering (SetPredication) must suspend around internal lazy zero-initialization (detail::initialize)
// and unconditional XDXR ray-tracing acceleration-structure dispatches (BuildAccelerationStructure,
// CopyAccelerationStructure, AccelerationStructurePostbuild), restoring the active predicate for subsequent
// predicated draws, clears, and dispatches. No GPU or game data.
#include "../src/vulkan_replay.h"

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace {

using namespace leonida;

struct Trace {
    bool cond_active = false;
    unsigned cond_begins = 0;
    unsigned cond_ends = 0;
    unsigned color_clears = 0;
    unsigned color_clears_while_cond = 0;
    unsigned depth_clears = 0;
    unsigned depth_clears_while_cond = 0;
    unsigned fill_bufs = 0;
    unsigned fill_bufs_while_cond = 0;
    unsigned copy_b2i = 0;
    unsigned copy_b2i_while_cond = 0;
    unsigned clear_att = 0;
    unsigned clear_att_while_cond = 0;
    unsigned draws = 0;
    unsigned draws_while_cond = 0;
    unsigned dispatches = 0;
    unsigned dispatches_while_cond = 0;
    unsigned xdxr_builds = 0;
    unsigned xdxr_builds_while_cond = 0;
    unsigned xdxr_copies = 0;
    unsigned xdxr_copies_while_cond = 0;
    unsigned xdxr_postbuilds = 0;
    unsigned xdxr_postbuilds_while_cond = 0;
    std::vector<std::string> events;
};

Trace* g_trace = nullptr;

template<class T>
T handle(std::uintptr_t n) {
    return reinterpret_cast<T>(n);
}

VKAPI_ATTR void VKAPI_CALL mock_barrier(VkCommandBuffer, VkPipelineStageFlags, VkPipelineStageFlags, VkDependencyFlags,
                                        std::uint32_t, const VkMemoryBarrier*, std::uint32_t,
                                        const VkBufferMemoryBarrier*, std::uint32_t, const VkImageMemoryBarrier*) {}
VKAPI_ATTR void VKAPI_CALL mock_barrier2(VkCommandBuffer, const VkDependencyInfo*) {}

VKAPI_ATTR void VKAPI_CALL mock_begin_cond(VkCommandBuffer, const VkConditionalRenderingBeginInfoEXT*) {
    g_trace->cond_active = true;
    ++g_trace->cond_begins;
    g_trace->events.push_back("begin_cond");
}

VKAPI_ATTR void VKAPI_CALL mock_end_cond(VkCommandBuffer) {
    g_trace->cond_active = false;
    ++g_trace->cond_ends;
    g_trace->events.push_back("end_cond");
}

VKAPI_ATTR void VKAPI_CALL mock_clear_color(VkCommandBuffer, VkImage, VkImageLayout, const VkClearColorValue*,
                                            std::uint32_t, const VkImageSubresourceRange*) {
    ++g_trace->color_clears;
    if (g_trace->cond_active) ++g_trace->color_clears_while_cond;
    g_trace->events.push_back("clear_color");
}

VKAPI_ATTR void VKAPI_CALL mock_clear_depth(VkCommandBuffer, VkImage, VkImageLayout, const VkClearDepthStencilValue*,
                                            std::uint32_t, const VkImageSubresourceRange*) {
    ++g_trace->depth_clears;
    if (g_trace->cond_active) ++g_trace->depth_clears_while_cond;
    g_trace->events.push_back("clear_depth");
}

VKAPI_ATTR void VKAPI_CALL mock_fill_buffer(VkCommandBuffer, VkBuffer, VkDeviceSize, VkDeviceSize, std::uint32_t) {
    ++g_trace->fill_bufs;
    if (g_trace->cond_active) ++g_trace->fill_bufs_while_cond;
    g_trace->events.push_back("fill_buffer");
}

VKAPI_ATTR void VKAPI_CALL mock_copy_b2i(VkCommandBuffer, VkBuffer, VkImage, VkImageLayout, std::uint32_t,
                                         const VkBufferImageCopy*) {
    ++g_trace->copy_b2i;
    if (g_trace->cond_active) ++g_trace->copy_b2i_while_cond;
    g_trace->events.push_back("copy_b2i");
}

VKAPI_ATTR void VKAPI_CALL mock_copy_i2b(VkCommandBuffer, VkImage, VkImageLayout, VkBuffer, std::uint32_t,
                                         const VkBufferImageCopy*) {}
VKAPI_ATTR void VKAPI_CALL mock_copy_buffer(VkCommandBuffer, VkBuffer, VkBuffer, std::uint32_t, const VkBufferCopy*) {}
VKAPI_ATTR void VKAPI_CALL mock_copy_image(VkCommandBuffer, VkImage, VkImageLayout, VkImage, VkImageLayout,
                                           std::uint32_t, const VkImageCopy*) {}
VKAPI_ATTR void VKAPI_CALL mock_resolve_image(VkCommandBuffer, VkImage, VkImageLayout, VkImage, VkImageLayout,
                                              std::uint32_t, const VkImageResolve*) {}
VKAPI_ATTR void VKAPI_CALL mock_begin_rendering(VkCommandBuffer, const VkRenderingInfo*) {}
VKAPI_ATTR void VKAPI_CALL mock_end_rendering(VkCommandBuffer) {}

VKAPI_ATTR void VKAPI_CALL mock_clear_attachments(VkCommandBuffer, std::uint32_t, const VkClearAttachment*,
                                                  std::uint32_t, const VkClearRect*) {
    ++g_trace->clear_att;
    if (g_trace->cond_active) ++g_trace->clear_att_while_cond;
    g_trace->events.push_back("clear_att");
}

VKAPI_ATTR void VKAPI_CALL mock_bind_pipeline(VkCommandBuffer, VkPipelineBindPoint, VkPipeline) {}
VKAPI_ATTR void VKAPI_CALL mock_push_constants(VkCommandBuffer, VkPipelineLayout, VkShaderStageFlags, std::uint32_t,
                                               std::uint32_t, const void*) {}
VKAPI_ATTR void VKAPI_CALL mock_set_viewport(VkCommandBuffer, std::uint32_t, std::uint32_t, const VkViewport*) {}
VKAPI_ATTR void VKAPI_CALL mock_set_scissor(VkCommandBuffer, std::uint32_t, std::uint32_t, const VkRect2D*) {}
VKAPI_ATTR void VKAPI_CALL mock_set_topology(VkCommandBuffer, VkPrimitiveTopology) {}
VKAPI_ATTR void VKAPI_CALL mock_set_restart(VkCommandBuffer, VkBool32) {}
VKAPI_ATTR void VKAPI_CALL mock_set_stencil(VkCommandBuffer, VkStencilFaceFlags, std::uint32_t) {}
VKAPI_ATTR void VKAPI_CALL mock_set_blend(VkCommandBuffer, const float[4]) {}
VKAPI_ATTR void VKAPI_CALL mock_bind_vb2(VkCommandBuffer, std::uint32_t, std::uint32_t, const VkBuffer*,
                                         const VkDeviceSize*, const VkDeviceSize*, const VkDeviceSize*) {}
VKAPI_ATTR void VKAPI_CALL mock_bind_ib(VkCommandBuffer, VkBuffer, VkDeviceSize, VkIndexType) {}

VKAPI_ATTR void VKAPI_CALL mock_draw(VkCommandBuffer, std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t) {
    ++g_trace->draws;
    if (g_trace->cond_active) ++g_trace->draws_while_cond;
    g_trace->events.push_back("draw");
}

VKAPI_ATTR void VKAPI_CALL mock_draw_indexed(VkCommandBuffer, std::uint32_t, std::uint32_t, std::uint32_t, std::int32_t,
                                             std::uint32_t) {}

VKAPI_ATTR void VKAPI_CALL mock_dispatch(VkCommandBuffer, std::uint32_t, std::uint32_t, std::uint32_t) {
    ++g_trace->dispatches;
    if (g_trace->cond_active) ++g_trace->dispatches_while_cond;
    g_trace->events.push_back("dispatch");
}

int failures = 0;
void expect(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++failures;
    }
}

ResourceBinding make_buf(std::uintptr_t id, std::uint64_t size, std::uint64_t va = 0) {
    ResourceBinding b{};
    b.buffer = handle<VkBuffer>(id);
    b.size = size;
    b.va = va;
    return b;
}

ResourceBinding make_img(std::uintptr_t id, VkFormat fmt, VkImageAspectFlags aspects, std::uint32_t w = 8,
                         std::uint32_t h = 8, std::uint32_t samples = 1, bool color_att = true, bool not_zeroed = false) {
    ResourceBinding b{};
    b.image = handle<VkImage>(id);
    b.format = fmt;
    b.aspects = aspects;
    b.width = w;
    b.height = h;
    b.depth = 1;
    b.mip_levels = 1;
    b.array_layers = 1;
    b.samples = samples;
    b.color_attachment = color_att;
    b.not_zeroed = not_zeroed;
    return b;
}

}  // namespace

int main() {
    constexpr std::uintptr_t kPredBuf = 1;
    constexpr std::uintptr_t kScratchBuf = 2;
    constexpr std::uintptr_t kColorInit = 3;
    constexpr std::uintptr_t kDepthInit = 4;
    constexpr std::uintptr_t kBcInit = 5;
    constexpr std::uintptr_t kMsaaInit = 6;
    constexpr std::uintptr_t kDstColor = 7;
    constexpr std::uintptr_t kDstDepth = 8;
    constexpr std::uintptr_t kNotZeroed = 9;
    constexpr std::uintptr_t kFootprintBuf = 10;
    constexpr std::uintptr_t kXdxrBuf = 11;
    constexpr std::uint64_t kXdxrVa = 0x100000;
    constexpr std::uintptr_t kMsaaDepthInit = 12;
    constexpr std::uintptr_t kCompPipe = 100;
    constexpr std::uintptr_t kGfxPipe = 101;

    const ResourceBinding pred_buf = make_buf(kPredBuf, 64);
    const ResourceBinding scratch_buf = make_buf(kScratchBuf, 65536);
    const ResourceBinding footprint_buf = make_buf(kFootprintBuf, 65536);
    const ResourceBinding xdxr_buf = make_buf(kXdxrBuf, 65536, kXdxrVa);

    const ResourceBinding color_img = make_img(kColorInit, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_ASPECT_COLOR_BIT);
    const ResourceBinding depth_img = make_img(kDepthInit, VK_FORMAT_D32_SFLOAT, VK_IMAGE_ASPECT_DEPTH_BIT, 8, 8, 1, false);
    const ResourceBinding bc_img = make_img(kBcInit, VK_FORMAT_BC1_RGBA_UNORM_BLOCK, VK_IMAGE_ASPECT_COLOR_BIT, 8, 8, 1, false);
    const ResourceBinding msaa_img = make_img(kMsaaInit, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_ASPECT_COLOR_BIT, 8, 8, 4, true);
    const ResourceBinding msaa_depth_img =
        make_img(kMsaaDepthInit, VK_FORMAT_D32_SFLOAT, VK_IMAGE_ASPECT_DEPTH_BIT, 8, 8, 4, false);
    const ResourceBinding dst_color_img = make_img(kDstColor, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_ASPECT_COLOR_BIT);
    const ResourceBinding dst_depth_img = make_img(kDstDepth, VK_FORMAT_D32_SFLOAT, VK_IMAGE_ASPECT_DEPTH_BIT, 8, 8, 1, false);
    const ResourceBinding nz_img = make_img(kNotZeroed, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_ASPECT_COLOR_BIT, 8, 8, 1, true, true);

    const auto resolve = [&](ResourceId id) -> const ResourceBinding* {
        switch (id) {
        case kPredBuf: return &pred_buf;
        case kFootprintBuf: return &footprint_buf;
        case kXdxrBuf: return &xdxr_buf;
        case kColorInit: return &color_img;
        case kDepthInit: return &depth_img;
        case kBcInit: return &bc_img;
        case kMsaaInit: return &msaa_img;
        case kMsaaDepthInit: return &msaa_depth_img;
        case kDstColor: return &dst_color_img;
        case kDstDepth: return &dst_depth_img;
        case kNotZeroed: return &nz_img;
        default: return nullptr;
        }
    };

    ComputePipeline comp_pso{};
    comp_pso.pipeline = handle<VkPipeline>(200);
    comp_pso.layout = handle<VkPipelineLayout>(201);

    GraphicsPipeline gfx_pso{};
    gfx_pso.pipeline = handle<VkPipeline>(202);
    gfx_pso.layout = handle<VkPipelineLayout>(203);
    gfx_pso.color_formats = {VK_FORMAT_R8G8B8A8_UNORM};
    gfx_pso.depth_format = VK_FORMAT_D32_SFLOAT;
    gfx_pso.depth_aspects = VK_IMAGE_ASPECT_DEPTH_BIT;
    gfx_pso.samples = 1;

    ReplayFunctions vk{};
    vk.pipeline_barrier = mock_barrier;
    vk.pipeline_barrier2 = mock_barrier2;
    vk.clear_color_image = mock_clear_color;
    vk.clear_depth_stencil_image = mock_clear_depth;
    vk.fill_buffer = mock_fill_buffer;
    vk.copy_buffer = mock_copy_buffer;
    vk.copy_buffer_to_image = mock_copy_b2i;
    vk.copy_image_to_buffer = mock_copy_i2b;
    vk.copy_image = mock_copy_image;
    vk.resolve_image = mock_resolve_image;
    vk.begin_rendering = mock_begin_rendering;
    vk.end_rendering = mock_end_rendering;
    vk.clear_attachments = mock_clear_attachments;
    vk.bind_pipeline = mock_bind_pipeline;
    vk.push_constants = mock_push_constants;
    vk.set_viewport = mock_set_viewport;
    vk.set_scissor = mock_set_scissor;
    vk.set_primitive_topology = mock_set_topology;
    vk.set_primitive_restart_enable = mock_set_restart;
    vk.set_stencil_reference = mock_set_stencil;
    vk.set_blend_constants = mock_set_blend;
    vk.bind_vertex_buffers2 = mock_bind_vb2;
    vk.bind_index_buffer = mock_bind_ib;
    vk.draw = mock_draw;
    vk.draw_indexed = mock_draw_indexed;
    vk.dispatch = mock_dispatch;
    vk.image_view = [](const ResourceBinding&, const VkImageSubresourceRange&) { return handle<VkImageView>(300); };
    vk.compute_pipeline = [&](std::uintptr_t id) { return id == kCompPipe ? &comp_pso : nullptr; };
    vk.graphics_pipeline = [&](std::uintptr_t id) { return id == kGfxPipe ? &gfx_pso : nullptr; };
    vk.begin_conditional_rendering = mock_begin_cond;
    vk.end_conditional_rendering = mock_end_cond;
    vk.prepare_predicate = [](VkBuffer source, VkDeviceSize offset, PredicateCapture& out) {
        out = {source, offset, handle<VkBuffer>(400), 0, handle<VkDescriptorSet>(401)};
        return true;
    };
    vk.capture_predicate = [](VkCommandBuffer, const PredicateCapture&) { g_trace->events.push_back("capture"); };
    vk.buffer_at = [&](std::uint64_t va, std::uint64_t& bytes, VkBuffer& out_buf, VkDeviceSize& out_off) {
        if (va < kXdxrVa || va - kXdxrVa >= xdxr_buf.size) return false;
        const std::uint64_t avail = xdxr_buf.size - (va - kXdxrVa);
        bytes = std::min(bytes, avail);
        out_buf = xdxr_buf.buffer;
        out_off = va - kXdxrVa;
        return bytes > 0;
    };
    vk.copy_scratch = [&](std::uint64_t, VkBuffer& out_buf, VkDeviceSize& out_off) {
        out_buf = scratch_buf.buffer;
        out_off = 0;
        return true;
    };
    vk.xdxr_check = [](const BuildAccelerationStructure&) { return true; };
    vk.xdxr_record = [](VkCommandBuffer, const BuildAccelerationStructure&) {
        ++g_trace->xdxr_builds;
        if (g_trace->cond_active) ++g_trace->xdxr_builds_while_cond;
        g_trace->events.push_back("xdxr_build");
    };
    vk.xdxr_copy = [](VkCommandBuffer, std::uint64_t, std::uint64_t, std::uint64_t) {
        ++g_trace->xdxr_copies;
        if (g_trace->cond_active) ++g_trace->xdxr_copies_while_cond;
        g_trace->events.push_back("xdxr_copy");
    };
    vk.xdxr_postbuild = [](VkCommandBuffer, std::uint64_t, std::uint64_t) {
        ++g_trace->xdxr_postbuilds;
        if (g_trace->cond_active) ++g_trace->xdxr_postbuilds_while_cond;
        g_trace->events.push_back("xdxr_postbuild");
    };
    vk.separate_depth_stencil_layouts = true;

    const auto make_tracker = [&] {
        LayoutTracker t(true);
        t.track(kColorInit, color_img, VK_IMAGE_LAYOUT_UNDEFINED);
        t.track(kDepthInit, depth_img, VK_IMAGE_LAYOUT_UNDEFINED);
        t.track(kBcInit, bc_img, VK_IMAGE_LAYOUT_UNDEFINED);
        t.track(kMsaaInit, msaa_img, VK_IMAGE_LAYOUT_UNDEFINED);
        t.track(kMsaaDepthInit, msaa_depth_img, VK_IMAGE_LAYOUT_UNDEFINED);
        t.track(kDstColor, dst_color_img, VK_IMAGE_LAYOUT_UNDEFINED);
        t.track(kDstDepth, dst_depth_img, VK_IMAGE_LAYOUT_UNDEFINED);
        t.track(kNotZeroed, nz_img, VK_IMAGE_LAYOUT_UNDEFINED);
        return t;
    };

    // 1. TransitionBarrier and TextureBarrier lazy zero-fill run outside conditional rendering, while subsequent
    //    predicated ClearRenderTarget runs inside conditional rendering.
    {
        Trace trace;
        g_trace = &trace;
        LayoutTracker tracker = make_tracker();
        CommandList list;
        expect(list.set_predication({kPredBuf, 0, d3d12_predication::equal_zero}), "1: pred");
        expect(list.transition({kColorInit, all_subresources, d3d12_state::common, d3d12_state::copy_source}), "1: trans");
        TextureBarrier tb{};
        tb.sync_before = d3d12_sync::none;
        tb.sync_after = d3d12_sync::copy;
        tb.access_before = d3d12_access::no_access;
        tb.access_after = d3d12_access::copy_source;
        tb.layout_before = d3d12_layout::common;
        tb.layout_after = d3d12_layout::copy_source;
        tb.resource = kDepthInit;
        tb.range = {all_subresources, 0, 0, 0, 0, 0};
        expect(list.barrier(tb), "1: texture barrier");
        const Rect rect{0, 0, 4, 4};
        expect(list.clear_render_target({kColorInit, 0, 0, 1}, {1, 0, 0, 1}, std::span<const Rect>(&rect, 1)),
               "1: rect clear");
        expect(list.close(), "1: close");
        expect(replay(list, VK_NULL_HANDLE, resolve, vk, &tracker) == ReplayResult::success, "1: replay");
        expect(trace.color_clears == 1 && trace.color_clears_while_cond == 0, "1: init color clear unpredicated");
        expect(trace.depth_clears == 1 && trace.depth_clears_while_cond == 0, "1: init depth clear unpredicated");
        expect(trace.clear_att == 1 && trace.clear_att_while_cond == 1, "1: user rect clear predicated");
        expect(trace.cond_begins == trace.cond_ends && !trace.cond_active, "1: cond balanced");
    }

    // 2. ClearRenderTarget, ClearDepthStencil, and ClearUnorderedAccess on UNDEFINED images suspend conditional
    //    rendering for the initial zero-fill and restore it for the actual clear.
    {
        Trace trace;
        g_trace = &trace;
        LayoutTracker tracker = make_tracker();
        CommandList list;
        expect(list.set_predication({kPredBuf, 0, d3d12_predication::equal_zero}), "2: pred");
        expect(list.clear_render_target({kColorInit, 0, 0, 1}, {1, 0, 0, 1}), "2: rtv clear");
        expect(list.clear_depth_stencil({kDepthInit, 0, 0, 1}, clear_depth, 0.5f, 0), "2: dsv clear");
        expect(list.clear_unordered_access({kDstColor, false, VK_FORMAT_R8G8B8A8_UINT, 0, 0, 0, 0, 0, 1}, false,
                                           {1, 2, 3, 4}),
               "2: uav clear");
        expect(list.close(), "2: close");
        expect(replay(list, VK_NULL_HANDLE, resolve, vk, &tracker) == ReplayResult::success, "2: replay");
        expect(trace.color_clears == 2 && trace.color_clears_while_cond == 0,
               "2: 2 color init clears outside conditional rendering");
        expect(trace.depth_clears == 1 && trace.depth_clears_while_cond == 0,
               "2: 1 depth init clear outside conditional rendering");
        expect(trace.clear_att == 3 && trace.clear_att_while_cond == 3,
               "2: 3 predicated user attachment clears inside conditional rendering");
        const std::vector<std::string> want{
            "capture",     "begin_cond",  "end_cond",   "clear_color", "begin_cond",  "clear_att",
            "end_cond",    "clear_depth", "begin_cond", "clear_att",   "end_cond",    "clear_color",
            "begin_cond",  "clear_att",   "end_cond",
        };
        expect(trace.events == want, "2: exact suspend/resume event sequence");
        expect(trace.cond_begins == trace.cond_ends && !trace.cond_active, "2: cond balanced");
    }

    // 3. CopyTexture (including BC1 scratch zero-fill via vkCmdFillBuffer + vkCmdCopyBufferToImage), CopyResource, and
    //    ResolveSubresource (color and depth MIN) suspend conditional rendering around lazy initialize().
    {
        Trace trace;
        g_trace = &trace;
        LayoutTracker tracker = make_tracker();
        CommandList list;
        expect(list.set_predication({kPredBuf, 0, d3d12_predication::equal_zero}), "3: pred");
        CopyTexture bc_copy{};
        bc_copy.dst = {kFootprintBuf, true, 0, 0, 8, 8, 1, 256};
        bc_copy.src = {kBcInit, false, 0, 0, 0, 0, 0, 0};
        expect(list.copy_texture(bc_copy), "3: copy bc");
        expect(list.copy_resource({kDstColor, kColorInit}), "3: copy res");
        ResolveSubresource res_depth{};
        res_depth.dst = kDstDepth;
        res_depth.src = kMsaaDepthInit;
        res_depth.region = ResolveRegion{0, 0, std::nullopt, d3d12_resolve_mode::minimum};
        expect(list.resolve_subresource(res_depth), "3: resolve depth");
        expect(list.resolve_subresource({kColorInit, 0, kMsaaInit, 0}), "3: resolve color");
        const Rect rect{0, 0, 2, 2};
        expect(list.clear_render_target({kDstColor, 0, 0, 1}, {0, 1, 0, 1}, std::span<const Rect>(&rect, 1)),
               "3: trailing predicated clear");
        expect(list.close(), "3: close");
        expect(replay(list, VK_NULL_HANDLE, resolve, vk, &tracker) == ReplayResult::success, "3: replay");
        expect(trace.fill_bufs == 1 && trace.fill_bufs_while_cond == 0, "3: BC scratch fill unpredicated");
        expect(trace.copy_b2i == 1 && trace.copy_b2i_while_cond == 0, "3: BC scratch b2i unpredicated");
        expect(trace.color_clears == 3 && trace.color_clears_while_cond == 0, "3: color copy/resolve inits unpredicated");
        expect(trace.depth_clears == 2 && trace.depth_clears_while_cond == 0, "3: depth resolve inits unpredicated");
        expect(trace.clear_att == 1 && trace.clear_att_while_cond == 1, "3: trailing rect clear predicated");
        expect(trace.cond_begins == trace.cond_ends && !trace.cond_active, "3: cond balanced");
    }

    // 4. Draw (detail::to_attachment) with UNDEFINED color + depth attachments zero-fills outside conditional
    //    rendering and runs vkCmdDraw inside conditional rendering.
    {
        Trace trace;
        g_trace = &trace;
        LayoutTracker tracker = make_tracker();
        CommandList list;
        expect(list.set_predication({kPredBuf, 0, d3d12_predication::equal_zero}), "4: pred");
        Draw d{};
        d.pipeline = kGfxPipe;
        d.render_target_count = 1;
        d.render_targets[0] = {kColorInit, 0, 0, 1};
        d.depth = {kDepthInit, 0, 0, 1};
        d.viewport = {0, 0, 8, 8, 0, 1};
        d.scissor = {0, 0, 8, 8};
        d.count = 3;
        d.instances = 1;
        expect(list.draw(d), "4: draw");
        expect(list.close(), "4: close");
        expect(replay(list, VK_NULL_HANDLE, resolve, vk, &tracker) == ReplayResult::success, "4: replay");
        expect(trace.color_clears == 1 && trace.color_clears_while_cond == 0, "4: color attachment init unpredicated");
        expect(trace.depth_clears == 1 && trace.depth_clears_while_cond == 0, "4: depth attachment init unpredicated");
        expect(trace.draws == 1 && trace.draws_while_cond == 1, "4: draw itself predicated");
        expect(trace.cond_begins == trace.cond_ends && !trace.cond_active, "4: cond balanced");
    }

    // 5. Already-initialized and not_zeroed UNDEFINED images emit no unnecessary end/begin toggles.
    {
        Trace trace;
        g_trace = &trace;
        LayoutTracker tracker = make_tracker();
        tracker.set(kColorInit, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}, VK_IMAGE_LAYOUT_GENERAL);
        CommandList list;
        expect(list.set_predication({kPredBuf, 0, d3d12_predication::equal_zero}), "5: pred");
        expect(list.transition({kColorInit, all_subresources, d3d12_state::common, d3d12_state::render_target}),
               "5: initialized transition");
        expect(list.transition({kNotZeroed, all_subresources, d3d12_state::common, d3d12_state::copy_source}),
               "5: not_zeroed transition");
        const Rect rect{0, 0, 2, 2};
        expect(list.clear_render_target({kColorInit, 0, 0, 1}, {1, 1, 1, 1}, std::span<const Rect>(&rect, 1)),
               "5: rect clear");
        expect(list.close(), "5: close");
        expect(replay(list, VK_NULL_HANDLE, resolve, vk, &tracker) == ReplayResult::success, "5: replay");
        expect(trace.color_clears == 0, "5: no init clears");
        expect(trace.cond_begins == 1 && trace.cond_ends == 1 && !trace.cond_active, "5: single cond scope");
        expect(trace.clear_att == 1 && trace.clear_att_while_cond == 1, "5: rect clear predicated");
    }

    // 6. XDXR Build/Copy/Postbuild suspend conditional rendering and restore it for following predicated work.
    {
        Trace trace;
        g_trace = &trace;
        CommandList list;
        expect(list.set_predication({kPredBuf, 0, d3d12_predication::equal_zero}), "6: pred");
        BuildAccelerationStructure build{};
        build.dst_va = kXdxrVa + 0x1000;
        build.scratch_va = kXdxrVa + 0x4000;
        expect(list.build_acceleration_structure(build), "6: build AS");
        CopyAccelerationStructure copy{};
        copy.src_va = kXdxrVa + 0x1000;
        copy.dst_va = kXdxrVa + 0x2000;
        expect(list.copy_acceleration_structure(copy), "6: copy AS");
        AccelerationStructurePostbuild info{};
        info.dest_va = kXdxrVa + 0x8000;
        info.type = 3;
        info.sources = {kXdxrVa + 0x1000, kXdxrVa + 0x2000};
        expect(list.acceleration_structure_postbuild(info), "6: postbuild AS");
        Dispatch dp{};
        dp.pipeline = kCompPipe;
        dp.x = dp.y = dp.z = 1;
        expect(list.dispatch(dp), "6: trailing dispatch");
        expect(list.close(), "6: close");
        expect(replay(list, VK_NULL_HANDLE, resolve, vk) == ReplayResult::success, "6: replay");
        expect(trace.xdxr_builds == 1 && trace.xdxr_builds_while_cond == 0, "6: xdxr build unpredicated");
        expect(trace.xdxr_copies == 1 && trace.xdxr_copies_while_cond == 0, "6: xdxr copy unpredicated");
        expect(trace.xdxr_postbuilds == 2 && trace.xdxr_postbuilds_while_cond == 0, "6: xdxr postbuild unpredicated");
        expect(trace.dispatches == 1 && trace.dispatches_while_cond == 1, "6: trailing dispatch predicated");
        expect(trace.cond_begins == 4 && trace.cond_ends == 4 && !trace.cond_active, "6: cond balanced");
    }

    // 7. Unpredicated execution of initialize() and XDXR callbacks never calls begin/end conditional rendering.
    {
        Trace trace;
        g_trace = &trace;
        LayoutTracker tracker = make_tracker();
        CommandList list;
        expect(list.clear_render_target({kColorInit, 0, 0, 1}, {1, 0, 0, 1}), "7: unpred clear");
        BuildAccelerationStructure build{};
        build.dst_va = kXdxrVa + 0x1000;
        build.scratch_va = kXdxrVa + 0x4000;
        expect(list.build_acceleration_structure(build), "7: unpred build AS");
        CopyAccelerationStructure copy{};
        copy.src_va = kXdxrVa + 0x1000;
        copy.dst_va = kXdxrVa + 0x2000;
        expect(list.copy_acceleration_structure(copy), "7: unpred copy AS");
        AccelerationStructurePostbuild info{};
        info.dest_va = kXdxrVa + 0x8000;
        info.type = 3;
        info.sources = {kXdxrVa + 0x1000};
        expect(list.acceleration_structure_postbuild(info), "7: unpred postbuild AS");
        expect(list.close(), "7: close");
        expect(replay(list, VK_NULL_HANDLE, resolve, vk, &tracker) == ReplayResult::success, "7: replay");
        expect(trace.color_clears == 2 && trace.xdxr_builds == 1 && trace.xdxr_copies == 1 && trace.xdxr_postbuilds == 1,
               "7: unpred ops executed");
        expect(trace.cond_begins == 0 && trace.cond_ends == 0 && !trace.cond_active, "7: no cond calls");
    }

    if (failures) return 1;
    std::puts("replay_predication_suspend_test: detail::initialize and XDXR dispatches suspend conditional rendering");
    return 0;
}
