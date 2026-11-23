// D3D-393: emit skips a draw's bind or set when the same value is still bound in the command buffer, and forgets what
// is bound at every command that may disturb it (detail::BoundDrawState). Every Vulkan entry point is a mock that
// counts its calls. The lists change one piece of state at a time, switch between pipelines whose layouts are and
// are not compatible for set 0 and push constants and to one without vertex input, and put between two draws a
// dispatch, an indirect dispatch, a predicate capture and a query resolve (helpers that bind), and barriers, clears,
// a copy, a marker, a query and an indirect draw (commands that bind nothing); a bundle's draws, and a list emitted
// twice into one command buffer. No Vulkan loader, GPU, shaders or game data.
#include "../src/vulkan_replay.h"

#include <cstdint>
#include <cstdio>
#include <memory>
#include <type_traits>
#include <unordered_map>

namespace {

using namespace leonida;

struct Calls {
    unsigned graphics_pipeline = 0, compute_pipeline = 0;
    unsigned graphics_set = 0, compute_set = 0;
    unsigned graphics_push = 0, compute_push = 0;
    unsigned viewport = 0, scissor = 0, topology = 0, restart = 0, stencil_front = 0, stencil_back = 0, blend = 0;
    unsigned vertex = 0, index = 0;
    unsigned draws = 0, mesh_draws = 0, dispatches = 0, scopes = 0, helpers = 0;
};
Calls g;

VKAPI_ATTR void VKAPI_CALL bind_pipeline(VkCommandBuffer, VkPipelineBindPoint point, VkPipeline) {
    ++(point == VK_PIPELINE_BIND_POINT_GRAPHICS ? g.graphics_pipeline : g.compute_pipeline);
}
VKAPI_ATTR void VKAPI_CALL bind_sets(VkCommandBuffer, VkPipelineBindPoint point, VkPipelineLayout, std::uint32_t,
                                     std::uint32_t, const VkDescriptorSet*, std::uint32_t, const std::uint32_t*) {
    ++(point == VK_PIPELINE_BIND_POINT_GRAPHICS ? g.graphics_set : g.compute_set);
}
VKAPI_ATTR void VKAPI_CALL push(VkCommandBuffer, VkPipelineLayout, VkShaderStageFlags stages, std::uint32_t,
                                std::uint32_t, const void*) {
    ++(stages == VK_SHADER_STAGE_COMPUTE_BIT ? g.compute_push : g.graphics_push);
}
VKAPI_ATTR void VKAPI_CALL viewport(VkCommandBuffer, std::uint32_t, std::uint32_t, const VkViewport*) { ++g.viewport; }
VKAPI_ATTR void VKAPI_CALL scissor(VkCommandBuffer, std::uint32_t, std::uint32_t, const VkRect2D*) { ++g.scissor; }
VKAPI_ATTR void VKAPI_CALL topology(VkCommandBuffer, VkPrimitiveTopology) { ++g.topology; }
VKAPI_ATTR void VKAPI_CALL restart(VkCommandBuffer, VkBool32) { ++g.restart; }
VKAPI_ATTR void VKAPI_CALL stencil(VkCommandBuffer, VkStencilFaceFlags face, std::uint32_t) {
    ++(face == VK_STENCIL_FACE_FRONT_BIT ? g.stencil_front : g.stencil_back);
}
VKAPI_ATTR void VKAPI_CALL blend(VkCommandBuffer, const float*) { ++g.blend; }
VKAPI_ATTR void VKAPI_CALL vertex(VkCommandBuffer, std::uint32_t, std::uint32_t, const VkBuffer*, const VkDeviceSize*,
                                  const VkDeviceSize*, const VkDeviceSize*) {
    ++g.vertex;
}
VKAPI_ATTR void VKAPI_CALL index(VkCommandBuffer, VkBuffer, VkDeviceSize, VkIndexType) { ++g.index; }
VKAPI_ATTR void VKAPI_CALL draw(VkCommandBuffer, std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t) {
    ++g.draws;
}
VKAPI_ATTR void VKAPI_CALL draw_indexed(VkCommandBuffer, std::uint32_t, std::uint32_t, std::uint32_t, std::int32_t,
                                        std::uint32_t) {
    ++g.draws;
}
VKAPI_ATTR void VKAPI_CALL dispatch(VkCommandBuffer, std::uint32_t, std::uint32_t, std::uint32_t) { ++g.dispatches; }
VKAPI_ATTR void VKAPI_CALL dispatch_indirect(VkCommandBuffer, VkBuffer, VkDeviceSize) { ++g.dispatches; }
VKAPI_ATTR void VKAPI_CALL begin_rendering(VkCommandBuffer, const VkRenderingInfo*) { ++g.scopes; }

// Any other vkCmd* signature: a no-op.
template<typename F>
struct Nop;
template<typename R, typename... A>
struct Nop<R(A...)> {
    static R VKAPI_CALL call(A...) {
        if constexpr (!std::is_void_v<R>) return R{};
    }
};
template<typename PFN>
PFN nop() {
    return &Nop<std::remove_pointer_t<PFN>>::call;
}

template<typename H>
H handle(std::uintptr_t value) {
    return reinterpret_cast<H>(value);
}

enum : ResourceId { color = 1, upload, predicate, results };
constexpr QueryHeapId occlusion_heap = 100;
constexpr std::uint64_t geometry = 0x100000000ull;  // a 1 MiB vertex and index arena

// p1: two vertex slots, set 0, VS+FS push constants. p2: the same with another layout handle (compatible for set 0
// and push constants). p3: no vertex input and no set 0. p4: another push-constant range (incompatible). p5: p1 with
// a 32-bit strip cut.
enum : std::uintptr_t { p1 = 0x7001, p2, p3, p4, p5, compute = 0x7100 };

struct Rig {
    std::unordered_map<ResourceId, ResourceBinding> resources;
    std::unordered_map<std::uintptr_t, GraphicsPipeline> pipelines;
    ComputePipeline cs;
    QueryPool queries;
    ReplayFunctions vk;

    Rig() {
        ResourceBinding image{};
        image.image = handle<VkImage>(0x1001);
        image.format = VK_FORMAT_R8G8B8A8_UNORM;
        image.aspects = VK_IMAGE_ASPECT_COLOR_BIT;
        image.width = image.height = 64;
        image.depth = image.mip_levels = image.array_layers = image.samples = 1;
        image.color_attachment = true;
        resources[color] = image;
        for (const ResourceId id : {upload, predicate, results}) {
            ResourceBinding b{};
            b.buffer = handle<VkBuffer>(0x2000 + id);
            b.size = 1 << 20;
            resources[id] = b;
        }
        queries = {handle<VkQueryPool>(0x3000), d3d12_query_heap::occlusion, 16, false, std::vector<bool>(16)};

        const VkDescriptorSet set0 = handle<VkDescriptorSet>(0x5000);
        const VkShaderStageFlags vs_fs = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
        const auto pipeline = [&](std::uintptr_t id, VkShaderStageFlags stages, VkDescriptorSet set,
                                  std::uint32_t vertex_slots, std::uint32_t strip_cut) {
            GraphicsPipeline& p = pipelines[id];
            p.pipeline = handle<VkPipeline>(0x6000 + id);
            p.layout = handle<VkPipelineLayout>(0x8000 + id);
            p.push_stages = stages;
            p.bindless_set = set;
            p.color_formats = {VK_FORMAT_R8G8B8A8_UNORM};
            p.vertex_slots = vertex_slots;
            p.strip_cut = strip_cut;
        };
        pipeline(p1, vs_fs, set0, 0b11, 0);
        pipeline(p2, vs_fs, set0, 0b11, 0);
        pipeline(p3, vs_fs, VK_NULL_HANDLE, 0, 0);
        pipeline(p4, vs_fs | VK_SHADER_STAGE_GEOMETRY_BIT, set0, 0b11, 0);
        pipeline(p5, vs_fs, set0, 0b11, 2);
        cs = {handle<VkPipeline>(0x6100), handle<VkPipelineLayout>(0x8100), 0, set0};

        vk.pipeline_barrier = nop<PFN_vkCmdPipelineBarrier>();
        vk.clear_color_image = nop<PFN_vkCmdClearColorImage>();
        vk.clear_depth_stencil_image = nop<PFN_vkCmdClearDepthStencilImage>();
        vk.pipeline_barrier2 = nop<PFN_vkCmdPipelineBarrier2>();
        vk.begin_rendering = begin_rendering;
        vk.end_rendering = nop<PFN_vkCmdEndRendering>();
        vk.clear_attachments = nop<PFN_vkCmdClearAttachments>();
        vk.copy_buffer = nop<PFN_vkCmdCopyBuffer>();
        vk.copy_buffer_to_image = nop<PFN_vkCmdCopyBufferToImage>();
        vk.copy_image_to_buffer = nop<PFN_vkCmdCopyImageToBuffer>();
        vk.copy_image = nop<PFN_vkCmdCopyImage>();
        vk.resolve_image = nop<PFN_vkCmdResolveImage>();
        vk.fill_buffer = nop<PFN_vkCmdFillBuffer>();
        vk.update_buffer = nop<PFN_vkCmdUpdateBuffer>();
        vk.begin_conditional_rendering = nop<PFN_vkCmdBeginConditionalRenderingEXT>();
        vk.end_conditional_rendering = nop<PFN_vkCmdEndConditionalRenderingEXT>();
        vk.prepare_predicate = [](VkBuffer source, VkDeviceSize offset, PredicateCapture& out) {
            out = {source, offset, handle<VkBuffer>(0x2100), 0, VK_NULL_HANDLE};
            return true;
        };
        vk.capture_predicate = [](VkCommandBuffer, const PredicateCapture&) { ++g.helpers; };
        vk.reset_query_pool = nop<PFN_vkCmdResetQueryPool>();
        vk.begin_query = nop<PFN_vkCmdBeginQuery>();
        vk.end_query = nop<PFN_vkCmdEndQuery>();
        vk.write_timestamp = nop<PFN_vkCmdWriteTimestamp>();
        vk.copy_query_pool_results = nop<PFN_vkCmdCopyQueryPoolResults>();
        vk.resolve_binary_occlusion = [](VkCommandBuffer, VkQueryPool, std::uint32_t, std::uint32_t, VkBuffer,
                                         VkDeviceSize) { ++g.helpers; };
        vk.query_pool = [this](QueryHeapId id) { return id == occlusion_heap ? &queries : nullptr; };
        vk.insert_label = nop<PFN_vkCmdInsertDebugUtilsLabelEXT>();
        vk.begin_label = nop<PFN_vkCmdBeginDebugUtilsLabelEXT>();
        vk.end_label = nop<PFN_vkCmdEndDebugUtilsLabelEXT>();
        vk.draw_indirect = nop<PFN_vkCmdDrawIndirect>();
        vk.draw_indexed_indirect = nop<PFN_vkCmdDrawIndexedIndirect>();
        vk.dispatch_indirect = dispatch_indirect;
        vk.image_view = [](const ResourceBinding&, const VkImageSubresourceRange&) {
            return handle<VkImageView>(0x4000);
        };
        vk.bind_pipeline = bind_pipeline;
        vk.push_constants = push;
        vk.bind_descriptor_sets = bind_sets;
        vk.dispatch = dispatch;
        vk.compute_pipeline = [this](std::uintptr_t id) { return id == compute ? &cs : nullptr; };
        vk.bind_vertex_buffers2 = vertex;
        vk.bind_index_buffer = index;
        vk.set_viewport = viewport;
        vk.set_scissor = scissor;
        vk.set_primitive_topology = topology;
        vk.set_primitive_restart_enable = restart;
        vk.set_stencil_reference = stencil;
        vk.set_blend_constants = blend;
        vk.draw = draw;
        vk.draw_indexed = draw_indexed;
        vk.graphics_pipeline = [this](std::uintptr_t id) -> const GraphicsPipeline* {
            const auto it = pipelines.find(id);
            return it == pipelines.end() ? nullptr : &it->second;
        };
        vk.buffer_at = [](std::uint64_t va, std::uint64_t& size, VkBuffer& buffer, VkDeviceSize& offset) {
            if (va < geometry || va >= geometry + (1 << 20)) return false;
            buffer = handle<VkBuffer>(0x2200);
            offset = va - geometry;
            size = std::min<std::uint64_t>(size, geometry + (1 << 20) - va);
            return true;
        };
        vk.zero_buffer = handle<VkBuffer>(0x2300);
        vk.zero_size = 65536;
    }

    const ResourceBinding* resolve(ResourceId id) const {
        const auto it = resources.find(id);
        return it == resources.end() ? nullptr : &it->second;
    }
};

// A draw's state; each field is one piece the draw sets.
struct Spec {
    std::uintptr_t pipeline = p1;
    std::uint32_t s0 = 0;
    float viewport_x = 0;
    std::int32_t scissor_right = 64;
    std::uint32_t topology = 4;  // D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST
    std::uint32_t stencil_front = 0, stencil_back = 0;
    float blend = 1;
    std::uint64_t vertex1 = geometry + 4096;  // slot 1; slot 0 is always at `geometry`
    bool indexed = true;
    std::uint64_t index = geometry + 8192;
    std::uint32_t index_format = 57;  // DXGI_FORMAT_R16_UINT
};

Draw make(const Spec& s) {
    Draw d{};
    d.pipeline = s.pipeline;
    d.user_data[0] = s.s0;
    auto views = std::make_shared<DrawViews>();
    views->render_targets[0] = {color, 0, 0, 1};
    views->render_target_count = 1;
    views->vertex_buffers[0] = {geometry, 4096, 12};
    views->vertex_buffers[1] = {s.vertex1, 4096, 8};
    d.views = std::move(views);
    d.index = {s.index, 4096, s.index_format};
    d.topology = s.topology;
    d.viewport = {s.viewport_x, 0, 64, 64, 0, 1};
    d.scissor = {0, 0, s.scissor_right, 64};
    d.blend_factor = {s.blend, 1, 1, 1};
    d.stencil_front = s.stencil_front;
    d.stencil_back = s.stencil_back;
    d.indexed = s.indexed;
    d.count = 3;
    d.instances = 1;
    return d;
}

int failures = 0;

// Replays `list` (closed here) and compares every count with `want`.
void check(const char* name, Rig& rig, CommandList& list, const Calls& want, int emits = 1) {
    g = {};
    if (!list.close()) {
        std::fprintf(stderr, "FAIL %s: the list did not close\n", name);
        ++failures;
        return;
    }
    const auto resolve = [&rig](ResourceId id) { return rig.resolve(id); };
    PreparedList prepared;
    if (const auto r = prepare(list, resolve, rig.vk, prepared); r != ReplayResult::success) {
        std::fprintf(stderr, "FAIL %s: prepare returned %d\n", name, int(r));
        ++failures;
        return;
    }
    QueryReplay queries;
    if (queries.plan(prepared) != ReplayResult::success) {
        std::fprintf(stderr, "FAIL %s: the queries were not planned\n", name);
        ++failures;
        return;
    }
    for (int i = 0; i < emits; ++i) emit(prepared, handle<VkCommandBuffer>(0x10), rig.vk);
    queries.commit();
    bool ok = true;
    const auto field = [&](const char* what, unsigned got, unsigned expected) {
        if (got == expected) return;
        std::fprintf(stderr, "FAIL %s: %s %u, expected %u\n", name, what, got, expected);
        ok = false;
    };
    field("graphics pipeline binds", g.graphics_pipeline, want.graphics_pipeline);
    field("compute pipeline binds", g.compute_pipeline, want.compute_pipeline);
    field("graphics set binds", g.graphics_set, want.graphics_set);
    field("compute set binds", g.compute_set, want.compute_set);
    field("graphics pushes", g.graphics_push, want.graphics_push);
    field("compute pushes", g.compute_push, want.compute_push);
    field("viewports", g.viewport, want.viewport);
    field("scissors", g.scissor, want.scissor);
    field("topologies", g.topology, want.topology);
    field("primitive restarts", g.restart, want.restart);
    field("front stencil references", g.stencil_front, want.stencil_front);
    field("back stencil references", g.stencil_back, want.stencil_back);
    field("blend constants", g.blend, want.blend);
    field("vertex-buffer binds", g.vertex, want.vertex);
    field("index-buffer binds", g.index, want.index);
    field("draws", g.draws, want.draws);
    field("mesh draws", g.mesh_draws, want.mesh_draws);
    field("dispatches", g.dispatches, want.dispatches);
    field("rendering scopes", g.scopes, want.scopes);
    field("helpers", g.helpers, want.helpers);
    if (!ok) ++failures;
}

// The counts of `draws` draws that set every piece once: what the first draw of a list costs.
Calls once(unsigned draws) {
    Calls c;
    c.graphics_pipeline = c.graphics_set = c.graphics_push = 1;
    c.viewport = c.scissor = c.topology = c.restart = c.stencil_front = c.stencil_back = c.blend = 1;
    c.vertex = 2;
    c.index = 1;
    c.draws = c.scopes = draws;
    return c;
}

}  // namespace

int main() {
    Rig rig;
    const Spec base;
    // The same draw three times: the second and third set nothing.
    {
        CommandList l;
        for (int i = 0; i < 3; ++i) l.draw(make(base));
        check("unchanged", rig, l, once(3));
    }
    // Mesh pipelines have no input assembly. Returning to the same indexed draw must restore
    // topology and restart once; the following identical draw can reuse both.
    {
        constexpr std::uintptr_t mesh_id = 0x7010;
        auto mesh_pipeline = rig.pipelines.at(p3);
        mesh_pipeline.pipeline = handle<VkPipeline>(mesh_id);
        mesh_pipeline.push_stages = VK_SHADER_STAGE_MESH_BIT_EXT | VK_SHADER_STAGE_FRAGMENT_BIT;
        mesh_pipeline.mesh = true;
        rig.pipelines.emplace(mesh_id, mesh_pipeline);
        rig.vk.draw_mesh_tasks = [](VkCommandBuffer, std::uint32_t, std::uint32_t, std::uint32_t) {
            if (g.topology != 1 || g.restart != 1) {
                std::fprintf(stderr, "FAIL mesh draw changed input assembly state\n");
                ++failures;
            }
            ++g.mesh_draws;
        };
        CommandList l;
        l.draw(make(base));
        auto mesh = make(base);
        mesh.pipeline = mesh_id;
        mesh.mesh = true;
        mesh.indexed = false;
        mesh.count = mesh.instances = mesh.first = 1;
        l.draw(mesh);
        l.draw(make(base));
        l.draw(make(base));
        Calls want = once(4);
        want.graphics_pipeline = want.graphics_push = 3;
        want.draws = 3;
        want.mesh_draws = 1;
        want.vertex = 4;
        want.viewport = want.scissor = want.topology = want.restart = 2;
        want.stencil_front = want.stencil_back = want.blend = 2;
        check("classic after mesh", rig, l, want);
    }
    // Each draw changes one more piece than the one before: only that piece is set again.
    {
        CommandList l;
        Spec s = base;
        l.draw(make(s));
        s.s0 = 1;  // push constants
        l.draw(make(s));
        s.viewport_x = 8;
        l.draw(make(s));
        s.scissor_right = 32;
        l.draw(make(s));
        s.topology = 5;  // strip; restart stays off (no strip cut)
        l.draw(make(s));
        s.stencil_front = 1;
        l.draw(make(s));
        s.stencil_back = 2;
        l.draw(make(s));
        s.blend = 0.5f;
        l.draw(make(s));
        s.vertex1 = geometry + 12288;  // slot 1 only
        l.draw(make(s));
        s.index = geometry + 16384;
        l.draw(make(s));
        s.index_format = 42;  // DXGI_FORMAT_R32_UINT
        l.draw(make(s));
        s.pipeline = p5;  // a 32-bit strip cut: restart on
        l.draw(make(s));
        Calls want = once(12);
        want.graphics_pipeline = want.graphics_push = want.viewport = want.scissor = want.topology = want.restart =
            want.stencil_front = want.stencil_back = want.blend = 2;
        want.vertex = 3;
        want.index = 3;
        check("one change at a time", rig, l, want);
    }
    // Set 0 and push constants stay across a layout compatible for them (p2), not across another push range (p4).
    {
        CommandList l;
        for (const std::uintptr_t p : {p1, p2, p4, p1}) {
            Spec s = base;
            s.pipeline = p;
            l.draw(make(s));
        }
        Calls want = once(4);
        want.graphics_pipeline = 4;
        want.graphics_set = want.graphics_push = 3;
        check("layouts", rig, l, want);
    }
    // A pipeline without vertex input leaves the strides static: the vertex buffers are bound again after it. Set 0
    // stays bound through it.
    {
        CommandList l;
        for (const std::uintptr_t p : {p1, p3, p1}) {
            Spec s = base;
            s.pipeline = p;
            l.draw(make(s));
        }
        Calls want = once(3);
        want.graphics_pipeline = 3;
        want.vertex = 4;
        check("no vertex input", rig, l, want);
    }
    // A dispatch, direct or indirect, binds a compute pipeline, set 0 and push constants: everything is set again.
    {
        CommandList l;
        l.draw(make(base));
        l.dispatch({compute, {}, 1, 1, 1});
        l.draw(make(base));
        ExecuteIndirect indirect;
        indirect.kind = IndirectKind::dispatch;
        indirect.stride = 12;
        indirect.max_count = 1;
        indirect.argument_buffer = upload;
        indirect.pipeline = compute;
        l.execute_indirect(indirect);
        l.draw(make(base));
        Calls want = once(3);
        want.graphics_pipeline = want.graphics_set = want.graphics_push = 3;
        want.viewport = want.scissor = want.topology = want.restart = want.stencil_front = want.stencil_back =
            want.blend = 3;
        want.vertex = 6;
        want.index = 3;
        want.compute_pipeline = want.compute_set = want.compute_push = 2;
        want.dispatches = 2;
        check("dispatches", rig, l, want);
    }
    // Barriers, clears (one in a rendering scope of its own), a copy, a marker, a query and an indirect draw bind
    // nothing: the draw after them sets nothing.
    {
        CommandList l;
        l.draw(make(base));
        l.transition({upload, all_subresources, d3d12_state::copy_source, d3d12_state::copy_dest});
        l.barrier(BufferBarrier{d3d12_sync::copy, d3d12_sync::pixel_shading, d3d12_access::copy_dest,
                                d3d12_access::shader_resource, upload});
        const Rect rect{0, 0, 16, 16};
        l.clear_render_target({color, 0, 0, 1}, {0, 0, 0, 1}, {&rect, 1});
        l.clear_render_target({color, 0, 0, 1}, {0, 0, 0, 1});
        l.copy_buffer({results, 0, upload, 0, 256});
        l.marker(DebugMarker::Kind::set, "between");
        l.begin_query({occlusion_heap, d3d12_query::occlusion, 0});
        l.end_query({occlusion_heap, d3d12_query::occlusion, 0});
        ExecuteIndirect indirect;
        indirect.kind = IndirectKind::draw;
        indirect.stride = 16;
        indirect.max_count = 1;
        indirect.argument_buffer = upload;
        l.execute_indirect(indirect);
        l.draw(make(base));
        Calls want = once(2);
        want.scopes = 3;
        check("commands that bind nothing", rig, l, want);
    }
    // SetPredication on a buffer captures the predicate with a helper dispatch; turning it off runs nothing.
    {
        CommandList l;
        l.draw(make(base));
        l.set_predication({predicate, 0, d3d12_predication::equal_zero});
        l.draw(make(base));
        l.set_predication({0, 0, 0});
        l.draw(make(base));
        Calls want = once(3);
        want.graphics_pipeline = want.graphics_set = want.graphics_push = 2;
        want.viewport = want.scissor = want.topology = want.restart = want.stencil_front = want.stencil_back =
            want.blend = 2;
        want.vertex = 4;
        want.index = 2;
        want.helpers = 1;
        check("predication", rig, l, want);
    }
    // A binary-occlusion resolve runs a helper dispatch.
    {
        CommandList l;
        l.begin_query({occlusion_heap, d3d12_query::binary_occlusion, 1});
        l.draw(make(base));
        l.end_query({occlusion_heap, d3d12_query::binary_occlusion, 1});
        l.resolve_queries({occlusion_heap, d3d12_query::binary_occlusion, 1, 1, results, 0});
        l.draw(make(base));
        Calls want = once(2);
        want.graphics_pipeline = want.graphics_set = want.graphics_push = 2;
        want.viewport = want.scissor = want.topology = want.restart = want.stencil_front = want.stencil_back =
            want.blend = 2;
        want.vertex = 4;
        want.index = 2;
        want.helpers = 1;
        check("query resolve", rig, l, want);
    }
    // An index buffer stays bound through a draw that reads none.
    {
        CommandList l;
        Spec plain = base;
        plain.indexed = false;
        l.draw(make(base));
        l.draw(make(plain));
        l.draw(make(base));
        check("non-indexed between indexed", rig, l, once(3));
    }
    // A bundle's draws are the parent's (flattened): the same state is not set again across it.
    {
        CommandList bundle, l;
        bundle.draw(make(base));
        bundle.close();
        l.draw(make(base));
        l.execute_bundle(bundle);
        l.draw(make(base));
        check("bundle", rig, l, once(3));
    }
    // Each emit starts with nothing known, even into the command buffer the last one wrote.
    {
        CommandList l;
        l.draw(make(base));
        l.draw(make(base));
        Calls want = once(4);
        want.graphics_pipeline = want.graphics_set = want.graphics_push = 2;
        want.viewport = want.scissor = want.topology = want.restart = want.stencil_front = want.stencil_back =
            want.blend = 2;
        want.vertex = 4;
        want.index = 2;
        check("two emits", rig, l, want, 2);
    }
    std::printf("replay draw state: %s\n", failures ? "FAILED" : "ok");
    return failures ? 1 : 0;
}
