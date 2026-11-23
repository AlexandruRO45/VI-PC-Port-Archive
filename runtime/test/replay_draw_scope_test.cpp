// Draws share a rendering scope (D3D-324, runtime/src/vulkan_replay.h DrawScope): consecutive draws with the same
// attachments are recorded in one vkCmdBeginRendering scope, any other command ends it first, and state a draw sets is
// only set again when it changes. The Vulkan functions are mocks that record what they are called with. No GPU or game
// data.
#include "../src/vulkan_replay.h"

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace {

using namespace leonida;

struct Trace {
    bool rendering = false;
    unsigned begins = 0, ends = 0, barriers = 0, barriers_inside = 0, nested = 0, stray_ends = 0;
    unsigned draws = 0, draws_outside = 0, pipelines = 0, sets = 0, pushes = 0, viewports = 0, scissors = 0;
    unsigned topologies = 0, restarts = 0, stencils = 0, blends = 0, vertex_binds = 0, index_binds = 0;
    unsigned cond_begins = 0, cond_ends = 0, cond_ends_inside = 0;
    VkDescriptorSet last_set = VK_NULL_HANDLE;  // the last set 0 bound
    std::vector<std::string> events;  // begin, end, barrier, draw, bind
};

Trace* g = nullptr;

template<class T>
T handle(std::uintptr_t n) {
    return reinterpret_cast<T>(n);
}

VKAPI_ATTR void VKAPI_CALL barrier(VkCommandBuffer, VkPipelineStageFlags, VkPipelineStageFlags, VkDependencyFlags,
                                   std::uint32_t, const VkMemoryBarrier*, std::uint32_t, const VkBufferMemoryBarrier*,
                                   std::uint32_t, const VkImageMemoryBarrier*) {
    ++g->barriers;
    if (g->rendering) ++g->barriers_inside;
    g->events.push_back("barrier");
}
VKAPI_ATTR void VKAPI_CALL begin_rendering(VkCommandBuffer, const VkRenderingInfo*) {
    if (g->rendering) ++g->nested;
    g->rendering = true;
    ++g->begins;
    g->events.push_back("begin");
}
VKAPI_ATTR void VKAPI_CALL end_rendering(VkCommandBuffer) {
    if (!g->rendering) ++g->stray_ends;
    g->rendering = false;
    ++g->ends;
    g->events.push_back("end");
}
VKAPI_ATTR void VKAPI_CALL bind_pipeline(VkCommandBuffer, VkPipelineBindPoint, VkPipeline) {
    ++g->pipelines;
    g->events.push_back("bind");
}
VKAPI_ATTR void VKAPI_CALL bind_sets(VkCommandBuffer, VkPipelineBindPoint, VkPipelineLayout, std::uint32_t, std::uint32_t,
                                     const VkDescriptorSet* sets, std::uint32_t, const std::uint32_t*) {
    ++g->sets;
    g->last_set = sets[0];
}
VKAPI_ATTR void VKAPI_CALL push(VkCommandBuffer, VkPipelineLayout, VkShaderStageFlags, std::uint32_t, std::uint32_t,
                                const void*) {
    ++g->pushes;
}
VKAPI_ATTR void VKAPI_CALL viewport(VkCommandBuffer, std::uint32_t, std::uint32_t, const VkViewport*) { ++g->viewports; }
VKAPI_ATTR void VKAPI_CALL scissor(VkCommandBuffer, std::uint32_t, std::uint32_t, const VkRect2D*) { ++g->scissors; }
VKAPI_ATTR void VKAPI_CALL topology(VkCommandBuffer, VkPrimitiveTopology) { ++g->topologies; }
VKAPI_ATTR void VKAPI_CALL restart(VkCommandBuffer, VkBool32) { ++g->restarts; }
VKAPI_ATTR void VKAPI_CALL stencil(VkCommandBuffer, VkStencilFaceFlags, std::uint32_t) { ++g->stencils; }
VKAPI_ATTR void VKAPI_CALL blend(VkCommandBuffer, const float[4]) { ++g->blends; }
VKAPI_ATTR void VKAPI_CALL vertex(VkCommandBuffer, std::uint32_t, std::uint32_t, const VkBuffer*, const VkDeviceSize*,
                                  const VkDeviceSize*, const VkDeviceSize*) {
    ++g->vertex_binds;
}
VKAPI_ATTR void VKAPI_CALL index(VkCommandBuffer, VkBuffer, VkDeviceSize, VkIndexType) { ++g->index_binds; }
void count_draw() {
    ++g->draws;
    if (!g->rendering) ++g->draws_outside;
    g->events.push_back("draw");
}
VKAPI_ATTR void VKAPI_CALL draw(VkCommandBuffer, std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t) {
    count_draw();
}
VKAPI_ATTR void VKAPI_CALL draw_indexed(VkCommandBuffer, std::uint32_t, std::uint32_t, std::uint32_t, std::int32_t,
                                        std::uint32_t) {
    count_draw();
}
VKAPI_ATTR void VKAPI_CALL begin_cond(VkCommandBuffer, const VkConditionalRenderingBeginInfoEXT*) {
    ++g->cond_begins;
    g->events.push_back("begin_cond");
}
VKAPI_ATTR void VKAPI_CALL end_cond(VkCommandBuffer) {
    ++g->cond_ends;
    if (g->rendering) ++g->cond_ends_inside;
    g->events.push_back("end_cond");
}

int failures = 0;
void expect(bool ok, const char* what) {
    if (ok) return;
    std::fprintf(stderr, "FAIL: %s\n", what);
    ++failures;
}

// Every scope ends, none nests, and nothing but draws (and the state they set) is recorded inside one.
void expect_well_formed(const Trace& t, const char* what) {
    if (t.begins != t.ends || t.nested || t.stray_ends || t.barriers_inside || t.draws_outside || t.rendering ||
        t.cond_ends_inside) {
        std::fprintf(stderr, "FAIL: %s: begins %u ends %u nested %u stray %u barriers inside %u draws outside %u\n", what,
                     t.begins, t.ends, t.nested, t.stray_ends, t.barriers_inside, t.draws_outside);
        ++failures;
    }
}

enum : ResourceId { color_a = 1, color_b, depth_id, vertex_id, index_id, predicate_id };
enum : std::uintptr_t { pso_a = 100, pso_b, pso_stencil, pso_fullscreen, pso_after_flush };

}  // namespace

int main() {
    const auto image = [](std::uintptr_t n, VkFormat format, VkImageAspectFlags aspects) {
        ResourceBinding b{};
        b.image = handle<VkImage>(n);
        b.format = format;
        b.aspects = aspects;
        b.width = b.height = 64;
        b.color_attachment = aspects == VK_IMAGE_ASPECT_COLOR_BIT;
        return b;
    };
    const ResourceBinding rt_a = image(1, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_ASPECT_COLOR_BIT);
    const ResourceBinding rt_b = image(2, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_ASPECT_COLOR_BIT);
    const ResourceBinding ds =
        image(3, VK_FORMAT_D32_SFLOAT_S8_UINT, VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT);
    ResourceBinding vb{};
    vb.buffer = handle<VkBuffer>(4);
    vb.size = 4096;
    vb.va = 0x10000;
    ResourceBinding ib = vb;
    ib.buffer = handle<VkBuffer>(5);
    ib.va = 0x20000;
    ResourceBinding pred{};
    pred.buffer = handle<VkBuffer>(6);
    pred.size = 16;
    const auto resolve = [&](ResourceId id) -> const ResourceBinding* {
        switch (id) {
        case color_a: return &rt_a;
        case color_b: return &rt_b;
        case depth_id: return &ds;
        case vertex_id: return &vb;
        case index_id: return &ib;
        case predicate_id: return &pred;
        default: return nullptr;
        }
    };

    const auto pipeline = [](std::uintptr_t n, VkImageAspectFlags depth_aspects) {
        GraphicsPipeline p;
        p.pipeline = handle<VkPipeline>(n);
        p.layout = handle<VkPipelineLayout>(1);
        p.push_stages = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
        p.bindless_set = handle<VkDescriptorSet>(1);
        p.color_formats = {VK_FORMAT_R8G8B8A8_UNORM};
        p.depth_format = VK_FORMAT_D32_SFLOAT_S8_UINT;
        p.depth_aspects = depth_aspects;
        p.vertex_slots = 1;
        return p;
    };
    const GraphicsPipeline a = pipeline(pso_a, VK_IMAGE_ASPECT_DEPTH_BIT);
    const GraphicsPipeline b = pipeline(pso_b, VK_IMAGE_ASPECT_DEPTH_BIT);
    const GraphicsPipeline with_stencil =
        pipeline(pso_stencil, VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT);
    GraphicsPipeline fullscreen = pipeline(pso_fullscreen, VK_IMAGE_ASPECT_DEPTH_BIT);
    fullscreen.vertex_slots = 0;  // vertices from SV_VertexID: no vertex input, so no dynamic stride state
    GraphicsPipeline after_flush = pipeline(pso_after_flush, VK_IMAGE_ASPECT_DEPTH_BIT);
    after_flush.bindless_set = handle<VkDescriptorSet>(2);  // made after a flush replaced the bindless set (D3D-314)

    ReplayFunctions vk{};
    vk.pipeline_barrier = barrier;
    vk.begin_rendering = begin_rendering;
    vk.end_rendering = end_rendering;
    vk.bind_pipeline = bind_pipeline;
    vk.bind_descriptor_sets = bind_sets;
    vk.push_constants = push;
    vk.set_viewport = viewport;
    vk.set_scissor = scissor;
    vk.set_primitive_topology = topology;
    vk.set_primitive_restart_enable = restart;
    vk.set_stencil_reference = stencil;
    vk.set_blend_constants = blend;
    vk.bind_vertex_buffers2 = vertex;
    vk.bind_index_buffer = index;
    vk.draw = draw;
    vk.draw_indexed = draw_indexed;
    vk.graphics_pipeline = [&](std::uintptr_t id) -> const GraphicsPipeline* {
        if (id == pso_a) return &a;
        if (id == pso_b) return &b;
        if (id == pso_stencil) return &with_stencil;
        if (id == pso_fullscreen) return &fullscreen;
        if (id == pso_after_flush) return &after_flush;
        return nullptr;
    };
    vk.image_view = [](const ResourceBinding& r, const VkImageSubresourceRange& range) {
        return handle<VkImageView>(reinterpret_cast<std::uintptr_t>(r.image) * 16 + range.aspectMask);
    };
    vk.buffer_at = [&](std::uint64_t va, std::uint64_t& size, VkBuffer& buffer, VkDeviceSize& offset) {
        for (const ResourceBinding* r : {&vb, &ib}) {
            if (va < r->va || va - r->va >= r->size) continue;
            buffer = r->buffer;
            offset = va - r->va;
            size = std::min<std::uint64_t>(size, r->size - offset);
            return true;
        }
        return false;
    };
    vk.begin_conditional_rendering = begin_cond;
    vk.end_conditional_rendering = end_cond;
    vk.prepare_predicate = [](VkBuffer source, VkDeviceSize offset, PredicateCapture& out) {
        out = {source, offset, handle<VkBuffer>(7), 0, handle<VkDescriptorSet>(8)};
        return true;
    };
    vk.capture_predicate = [](VkCommandBuffer, const PredicateCapture&) {};

    const auto make_draw = [](std::uintptr_t pso, ResourceId target, std::uint32_t seed = 0) {
        Draw d{};
        d.pipeline = pso;
        d.user_data[0] = seed;
        DrawViews views;  // render_target_count stays 8: slots 1-7 are unbound (resource 0)
        views.render_targets = {TargetView{target, 0, 0, 1}};
        views.vertex_buffers = {VertexBufferView{0x10000, 1024, 16}};
        d.views = std::make_shared<const DrawViews>(views);
        d.depth = TargetView{depth_id, 0, 0, 1};
        d.index = IndexBufferView{0x20000, 1024, 57};
        d.topology = 4;  // triangle list
        d.viewport = {0, 0, 64, 64, 0, 1};
        d.scissor = {0, 0, 64, 64};
        d.blend_factor = {1, 1, 1, 1};
        d.indexed = true;
        d.count = 3;
        d.instances = 1;
        return d;
    };
    const auto run = [&](CommandList& list, Trace& trace, detail::DrawScope* scope = nullptr,
                         LayoutTracker* tracker = nullptr) {
        g = &trace;
        expect(list.close(), "close the list");
        PreparedList prepared;
        expect(prepare(list, resolve, vk, prepared) == ReplayResult::success, "prepare the list");
        emit(prepared, VK_NULL_HANDLE, vk, tracker, nullptr, scope);
    };

    // 1. Eight draws to the same targets with the same state: one scope, attachments moved once, state set once.
    {
        Trace t;
        CommandList list;
        for (int i = 0; i < 8; ++i) expect(list.draw(make_draw(pso_a, color_a)), "record a draw");
        detail::DrawScope scope;
        run(list, t, &scope);
        expect_well_formed(t, "same targets");
        expect(t.begins == 1 && t.draws == 8, "same targets: eight draws in one scope");
        expect(scope.scopes == 1 && scope.draws == 8, "same targets: the scope counts one scope and eight draws");
        expect(t.barriers == 2, "same targets: one barrier per attachment, when the scope opens");
        expect(t.pipelines == 1 && t.sets == 1 && t.pushes == 1, "same targets: pipeline, set and constants once");
        expect(t.viewports == 1 && t.scissors == 1 && t.topologies == 1 && t.restarts == 1 && t.stencils == 2 &&
                   t.blends == 1,
               "same targets: dynamic state once");
        expect(t.vertex_binds == 1 && t.index_binds == 1, "same targets: vertex and index buffers once");
    }

    // 2. What changes between draws is set again, in the same scope.
    {
        Trace t;
        CommandList list;
        expect(list.draw(make_draw(pso_a, color_a, 1)), "record a draw");
        Draw moved = make_draw(pso_a, color_a, 2);  // other root constants
        moved.viewport = {0, 0, 32, 32, 0, 1};
        moved.stencil_front = moved.stencil_back = 7;
        expect(list.draw(moved), "record a draw with another viewport and stencil reference");
        expect(list.draw(make_draw(pso_b, color_a, 2)), "record a draw with another pipeline");
        run(list, t);
        expect_well_formed(t, "changing state");
        expect(t.begins == 1 && t.draws == 3, "changing state: one scope");
        // The third draw binds another pipeline with the same layout and the same constants: no push.
        expect(t.pushes == 2, "changing state: constants pushed again only when they or the layout change");
        expect(t.pipelines == 2 && t.sets == 1, "changing state: the second pipeline bound, the set kept");
        expect(t.viewports == 3 && t.stencils == 6, "changing state: viewport and stencil set again when they change");
        expect(t.scissors == 1 && t.blends == 1, "changing state: unchanged scissor and blend kept");
    }

    // 3. Another render target, or another depth/stencil attachment set, opens a new scope.
    {
        Trace t;
        CommandList list;
        expect(list.draw(make_draw(pso_a, color_a)), "record a draw to A");
        expect(list.draw(make_draw(pso_a, color_b)), "record a draw to B");
        expect(list.draw(make_draw(pso_a, color_b)), "record a second draw to B");
        expect(list.draw(make_draw(pso_stencil, color_b)), "record a draw that also attaches stencil");
        run(list, t);
        expect_well_formed(t, "other targets");
        expect(t.begins == 3 && t.draws == 4, "other targets: a scope for A, one for B, one with stencil");
        expect(t.pipelines == 3, "other targets: the pipeline is bound again in each new scope");
        expect(t.vertex_binds == 3 && t.index_binds == 3,
               "other targets: vertex and index buffers are bound again in each new scope");
    }

    // 4. Any other command ends the scope before it: a UAV barrier between draws.
    {
        Trace t;
        CommandList list;
        expect(list.draw(make_draw(pso_a, color_a)), "record a draw");
        expect(list.draw(make_draw(pso_a, color_a)), "record a draw");
        expect(list.uav_barrier({0}), "record a UAV barrier");
        expect(list.draw(make_draw(pso_a, color_a)), "record a draw");
        run(list, t);
        expect_well_formed(t, "barrier between draws");
        expect(t.begins == 2, "barrier between draws: two scopes");
        const std::vector<std::string> order{"barrier", "barrier", "begin", "bind", "draw", "draw", "end", "barrier",
                                             "barrier", "barrier", "begin", "bind", "draw", "end"};
        expect(t.events == order, "barrier between draws: the scope ends before the barrier");
    }

    // 5. A debug marker between draws also ends the scope (markers run outside one, as before).
    {
        Trace t;
        CommandList list;
        expect(list.draw(make_draw(pso_a, color_a)), "record a draw");
        expect(list.marker(DebugMarker::Kind::set, "between"), "record a marker");
        expect(list.draw(make_draw(pso_a, color_a)), "record a draw");
        run(list, t);
        expect_well_formed(t, "marker between draws");
        expect(t.begins == 2, "marker between draws: two scopes");
    }

    // 6. Predication: the scope opens and ends inside conditional rendering, and ends before it does.
    {
        Trace t;
        CommandList list;
        expect(list.set_predication({predicate_id, 0, d3d12_predication::equal_zero}), "record predication");
        expect(list.draw(make_draw(pso_a, color_a)), "record a predicated draw");
        expect(list.draw(make_draw(pso_a, color_a)), "record a predicated draw");
        run(list, t);
        expect_well_formed(t, "predicated draws");
        expect(t.begins == 1 && t.cond_begins == 1 && t.cond_ends == 1, "predicated draws: one scope");
        expect(t.events.size() >= 2 && t.events[t.events.size() - 2] == "end" && t.events.back() == "end_cond",
               "predicated draws: the scope ends before conditional rendering");
    }

    // 7. With a layout tracker: attachments move to their layouts once, and the tracker records it.
    {
        Trace t;
        LayoutTracker tracker;
        tracker.track(color_a, rt_a, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        tracker.track(depth_id, ds, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
        CommandList list;
        for (int i = 0; i < 4; ++i) expect(list.draw(make_draw(pso_a, color_a)), "record a draw");
        run(list, t, nullptr, &tracker);
        expect_well_formed(t, "tracked");
        expect(t.begins == 1 && t.barriers == 2, "tracked: one transition and one depth barrier, once");
        expect(tracker.layout(color_a, 0) == VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
               "tracked: the render target is recorded in COLOR_ATTACHMENT_OPTIMAL");
    }

    // 8. A pipeline without vertex input leaves the stride out of its dynamic states, so binding it invalidates the
    // strides: the next draw that reads vertex buffers binds them again, even when they did not change.
    {
        Trace t;
        CommandList list;
        expect(list.draw(make_draw(pso_a, color_a)), "record a draw with vertex input");
        Draw fs = make_draw(pso_fullscreen, color_a);
        fs.indexed = false;
        expect(list.draw(fs), "record a full-screen draw without vertex input");
        expect(list.draw(make_draw(pso_a, color_a)), "record the first draw again");
        run(list, t);
        expect_well_formed(t, "pipeline without vertex input");
        expect(t.begins == 1 && t.draws == 3, "pipeline without vertex input: one scope");
        expect(t.pipelines == 3, "pipeline without vertex input: three pipeline binds");
        expect(t.vertex_binds == 2, "pipeline without vertex input: the vertex buffer is bound again after it");
    }

    // 9. Set 0 comes from the device's current bindless generation when replay has one (D3D-314), and the scope caches
    // the set it bound: two pipelines made before and after a flush (other set handles, same layout) share it.
    {
        CommandList list;
        expect(list.draw(make_draw(pso_a, color_a)), "record a draw");
        expect(list.draw(make_draw(pso_after_flush, color_a)), "record a draw with a pipeline made after a flush");
        expect(list.close(), "close the list");
        PreparedList prepared;
        expect(prepare(list, resolve, vk, prepared) == ReplayResult::success, "prepare the list");

        Trace own;  // no current generation: each pipeline's own set
        g = &own;
        emit(prepared, VK_NULL_HANDLE, vk);
        expect_well_formed(own, "pipelines' own sets");
        expect(own.sets == 2 && own.last_set == handle<VkDescriptorSet>(2),
               "pipelines' own sets: each pipeline's set is bound");

        Trace current;
        g = &current;
        vk.bindless_set = handle<VkDescriptorSet>(9);
        emit(prepared, VK_NULL_HANDLE, vk);
        vk.bindless_set = VK_NULL_HANDLE;
        expect_well_formed(current, "current generation");
        expect(current.begins == 1 && current.draws == 2, "current generation: one scope");
        expect(current.sets == 1 && current.last_set == handle<VkDescriptorSet>(9),
               "current generation: the device's set is bound once, for both pipelines");
    }

    if (failures) {
        std::fprintf(stderr, "%d failure(s)\n", failures);
        return 1;
    }
    std::puts("draws share a rendering scope while their attachments match; other commands end it first");
    return 0;
}
