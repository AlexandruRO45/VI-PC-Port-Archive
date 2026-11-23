// A COPY queue keeps the CLEAR stage in its barriers but never clears an image. The barrier mask of a COPY queue
// (detail::queue_stages_copy, applied by restrict_to_queue) must keep VK_PIPELINE_STAGE_2_CLEAR_BIT:
// WriteBufferImmediate is recorded as vkCmdUpdateBuffer and synchronization2 runs it in the CLEAR stage, which a
// transfer-only queue supports. Whether a queue may call vkCmdClearColorImage is a separate test in
// detail::initialize: that command needs a compute or graphics queue, so a never-used colour image on a COPY queue
// is zeroed by a buffer copy instead. Every barrier and command replay emits is captured through mocked Vulkan
// functions. No Vulkan loader, GPU or game data.
#include "../src/vulkan_replay.h"
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace {

using namespace leonida;

// One emitted command; barriers keep their stage and access masks (one entry per Vk*MemoryBarrier*).
struct Event {
    std::string what;
    VkPipelineStageFlags2 src_stages = 0, dst_stages = 0;
    VkAccessFlags2 src_access = 0, dst_access = 0;
};
std::vector<Event> events;

VKAPI_ATTR void VKAPI_CALL barrier(VkCommandBuffer, VkPipelineStageFlags src, VkPipelineStageFlags dst,
                                   VkDependencyFlags, std::uint32_t memory_count, const VkMemoryBarrier* memory,
                                   std::uint32_t buffer_count, const VkBufferMemoryBarrier* buffers,
                                   std::uint32_t image_count, const VkImageMemoryBarrier* images) {
    for (std::uint32_t i = 0; i < memory_count; ++i)
        events.push_back({"barrier memory", src, dst, memory[i].srcAccessMask, memory[i].dstAccessMask});
    for (std::uint32_t i = 0; i < buffer_count; ++i)
        events.push_back({"barrier buffer", src, dst, buffers[i].srcAccessMask, buffers[i].dstAccessMask});
    for (std::uint32_t i = 0; i < image_count; ++i)
        events.push_back({"barrier image", src, dst, images[i].srcAccessMask, images[i].dstAccessMask});
    if (!memory_count && !buffer_count && !image_count) events.push_back({"barrier execution", src, dst});
}
VKAPI_ATTR void VKAPI_CALL barrier2(VkCommandBuffer, const VkDependencyInfo* info) {
    for (std::uint32_t i = 0; i < info->memoryBarrierCount; ++i) {
        const auto& b = info->pMemoryBarriers[i];
        events.push_back({"barrier2 memory", b.srcStageMask, b.dstStageMask, b.srcAccessMask, b.dstAccessMask});
    }
    for (std::uint32_t i = 0; i < info->bufferMemoryBarrierCount; ++i) {
        const auto& b = info->pBufferMemoryBarriers[i];
        events.push_back({"barrier2 buffer", b.srcStageMask, b.dstStageMask, b.srcAccessMask, b.dstAccessMask});
    }
    for (std::uint32_t i = 0; i < info->imageMemoryBarrierCount; ++i) {
        const auto& b = info->pImageMemoryBarriers[i];
        events.push_back({"barrier2 image", b.srcStageMask, b.dstStageMask, b.srcAccessMask, b.dstAccessMask});
    }
}
VKAPI_ATTR void VKAPI_CALL update_buffer(VkCommandBuffer, VkBuffer, VkDeviceSize, VkDeviceSize, const void*) {
    events.push_back({"update_buffer"});
}
VKAPI_ATTR void VKAPI_CALL copy_buffer(VkCommandBuffer, VkBuffer, VkBuffer, std::uint32_t, const VkBufferCopy*) {
    events.push_back({"copy_buffer"});
}
VKAPI_ATTR void VKAPI_CALL copy_buffer_to_image(VkCommandBuffer, VkBuffer, VkImage, VkImageLayout, std::uint32_t,
                                                const VkBufferImageCopy*) {
    events.push_back({"copy_buffer_to_image"});
}
VKAPI_ATTR void VKAPI_CALL copy_image_to_buffer(VkCommandBuffer, VkImage, VkImageLayout, VkBuffer, std::uint32_t,
                                                const VkBufferImageCopy*) {
    events.push_back({"copy_image_to_buffer"});
}
VKAPI_ATTR void VKAPI_CALL copy_image(VkCommandBuffer, VkImage, VkImageLayout, VkImage, VkImageLayout, std::uint32_t,
                                      const VkImageCopy*) {
    events.push_back({"copy_image"});
}
VKAPI_ATTR void VKAPI_CALL fill_buffer(VkCommandBuffer, VkBuffer, VkDeviceSize, VkDeviceSize, std::uint32_t) {
    events.push_back({"fill_buffer"});
}
VKAPI_ATTR void VKAPI_CALL clear_color(VkCommandBuffer, VkImage, VkImageLayout, const VkClearColorValue*,
                                       std::uint32_t, const VkImageSubresourceRange*) {
    events.push_back({"clear_color_image"});
}
VKAPI_ATTR void VKAPI_CALL clear_depth(VkCommandBuffer, VkImage, VkImageLayout, const VkClearDepthStencilValue*,
                                       std::uint32_t, const VkImageSubresourceRange*) {
    events.push_back({"clear_depth_stencil_image"});
}

int failures = 0;
void expect(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++failures;
    }
}

template<class T>
T handle(std::uintptr_t n) {
    return reinterpret_cast<T>(n);
}

void print(const char* title) {
    std::printf("%s:\n", title);
    for (const auto& e : events) {
        if (e.what.rfind("barrier", 0) == 0)
            std::printf("  %s src 0x%" PRIx64 "/0x%" PRIx64 " -> dst 0x%" PRIx64 "/0x%" PRIx64 "\n", e.what.c_str(),
                        std::uint64_t(e.src_stages), std::uint64_t(e.src_access), std::uint64_t(e.dst_stages),
                        std::uint64_t(e.dst_access));
        else
            std::printf("  %s\n", e.what.c_str());
    }
}

unsigned count(const char* what) {
    unsigned n = 0;
    for (const auto& e : events) n += e.what == what;
    return n;
}

ReplayFunctions functions(VkPipelineStageFlags2 queue_stages) {
    ReplayFunctions vk{};
    vk.pipeline_barrier = barrier;
    vk.pipeline_barrier2 = barrier2;
    vk.update_buffer = update_buffer;
    vk.copy_buffer = copy_buffer;
    vk.copy_buffer_to_image = copy_buffer_to_image;
    vk.copy_image_to_buffer = copy_image_to_buffer;
    vk.copy_image = copy_image;
    vk.fill_buffer = fill_buffer;
    vk.clear_color_image = clear_color;
    vk.clear_depth_stencil_image = clear_depth;
    vk.zero_buffer = handle<VkBuffer>(9);
    vk.zero_size = 64 * 1024;
    vk.queue_stages = queue_stages;
    return vk;
}

struct Queue {
    const char* name;
    VkPipelineStageFlags2 stages;  // ReplayFunctions::queue_stages, as VulkanQueue sets it
};
constexpr Queue queues[] = {{"DIRECT", 0},
                            {"COMPUTE", detail::queue_stages_compute},
                            {"COPY", detail::queue_stages_copy}};

}  // namespace

int main() {
    constexpr VkPipelineStageFlags2 clear = VK_PIPELINE_STAGE_2_CLEAR_BIT;
    // Stages whose first scope contains vkCmdUpdateBuffer under synchronization2.
    constexpr VkPipelineStageFlags2 covers_clear =
        clear | VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT | VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;

    // 1. The masks. COMPUTE keeps the stages it had; COPY gains CLEAR, which Vulkan allows on transfer-only queues.
    expect(detail::queue_stages_compute ==
               (VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT | VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT |
                VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT | VK_PIPELINE_STAGE_2_HOST_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT |
                VK_PIPELINE_STAGE_2_COPY_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
                VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT | clear),
           "the COMPUTE barrier mask is unchanged");
    expect((detail::queue_stages_copy & clear) && !(detail::queue_stages_copy & VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT),
           "the COPY barrier mask has CLEAR and no shader stage");

    // 2. restrict_to_queue keeps CLEAR in a COPY | CLEAR scope on every queue kind.
    for (const auto& q : queues) {
        auto vk = functions(q.stages);
        VkPipelineStageFlags2 stages = VK_PIPELINE_STAGE_2_COPY_BIT | clear;
        VkAccessFlags2 access = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        detail::restrict_to_queue(vk, stages, access);
        char what[96];
        std::snprintf(what, sizeof what, "%s: restrict_to_queue keeps COPY | CLEAR and TRANSFER_WRITE", q.name);
        expect(stages == (VK_PIPELINE_STAGE_2_COPY_BIT | clear) && access == VK_ACCESS_2_TRANSFER_WRITE_BIT, what);
    }

    // 3. A DEFAULT WriteBufferImmediate, an enhanced SYNC_COPY buffer barrier (COPY_DEST -> COPY_SOURCE), then a
    //    copy of the written word: the barrier's first scope must contain the write on every queue kind. This needs
    //    SYNC_COPY to map to COPY | CLEAR (#8187); this change keeps CLEAR from being masked away on COPY queues.
    constexpr std::uint64_t va = 0x10000;
    ResourceBinding src{}, dst{};
    src.buffer = handle<VkBuffer>(1);
    src.size = 256;
    src.va = va;
    dst.buffer = handle<VkBuffer>(2);
    dst.size = 256;
    for (const auto& q : queues) {
        auto vk = functions(q.stages);
        vk.buffer_at = [&](std::uint64_t at, std::uint64_t length, VkBuffer& buffer, VkDeviceSize& offset) {
            if (at < va || at - va >= src.size || length > src.size - (at - va)) return false;
            buffer = src.buffer;
            offset = at - va;
            return true;
        };
        const auto resolve = [&](ResourceId id) -> const ResourceBinding* {
            return id == 1 ? &src : id == 2 ? &dst : nullptr;
        };
        namespace y = d3d12_sync;
        namespace a = d3d12_access;
        CommandList list;
        expect(list.write_immediate({va + 8, 0x5eed1234, write_immediate_mode::normal}) &&
                   list.barrier(BufferBarrier{y::copy, y::copy, a::copy_dest, a::copy_source, 1}) &&
                   list.copy_buffer({2, 0, 1, 8, 4}) && list.close(),
               "record write, SYNC_COPY barrier, copy");
        events.clear();
        expect(replay(list, VK_NULL_HANDLE, resolve, vk) == ReplayResult::success, "replay write, barrier, copy");
        char title[96];
        std::snprintf(title, sizeof title, "%s queue: WriteBufferImmediate, SYNC_COPY barrier, CopyBufferRegion", q.name);
        print(title);
        const bool shape = events.size() == 3 && events[0].what == "update_buffer" &&
                           events[1].what.rfind("barrier2", 0) == 0 && events[2].what == "copy_buffer";
        std::snprintf(title, sizeof title, "%s: exactly update_buffer, one barrier, copy_buffer", q.name);
        expect(shape, title);
        if (!shape) continue;
        std::snprintf(title, sizeof title, "%s: the SYNC_COPY barrier's source stages contain the CLEAR stage", q.name);
        expect(events[1].src_stages & covers_clear, title);
        std::snprintf(title, sizeof title, "%s: the barrier makes TRANSFER_WRITE available", q.name);
        expect(events[1].src_access & (VK_ACCESS_2_TRANSFER_WRITE_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT), title);
    }

    // 4. A never-used (UNDEFINED) image first touched by a transition: DIRECT clears colour and depth, COMPUTE
    //    clears colour and only moves depth, COPY clears neither. Its colour image is zeroed by copying the zero
    //    buffer and its depth image only moved, since vkCmdClearColorImage needs compute or graphics and
    //    vkCmdClearDepthStencilImage graphics.
    ResourceBinding color{};
    color.image = handle<VkImage>(3);
    color.format = VK_FORMAT_R8G8B8A8_UNORM;
    color.aspects = VK_IMAGE_ASPECT_COLOR_BIT;
    color.width = color.height = 4;
    color.depth = color.mip_levels = color.array_layers = color.samples = 1;
    ResourceBinding depth = color;
    depth.image = handle<VkImage>(4);
    depth.format = VK_FORMAT_D32_SFLOAT;
    depth.aspects = VK_IMAGE_ASPECT_DEPTH_BIT;
    struct Want {
        unsigned clear_color, clear_depth, zero_copies;
    };
    const Want want_color[] = {{1, 0, 0}, {1, 0, 0}, {0, 0, 1}};
    const Want want_depth[] = {{0, 1, 0}, {0, 0, 0}, {0, 0, 0}};
    for (std::size_t i = 0; i < std::size(queues); ++i) {
        for (const bool is_color : {true, false}) {
            const ResourceBinding& image = is_color ? color : depth;
            const Want want = is_color ? want_color[i] : want_depth[i];
            auto vk = functions(queues[i].stages);
            LayoutTracker tracker;
            tracker.track(1, image, VK_IMAGE_LAYOUT_UNDEFINED);
            CommandList list;
            expect(list.transition({1, 0, d3d12_state::common, d3d12_state::copy_dest}) && list.close(),
                   "record a transition");
            events.clear();
            expect(replay(list, VK_NULL_HANDLE, [&](ResourceId) { return &image; }, vk, &tracker) ==
                       ReplayResult::success,
                   "replay a transition of a never-used image");
            char title[128];
            std::snprintf(title, sizeof title, "%s queue: first transition of a never-used %s image", queues[i].name,
                          is_color ? "colour" : "depth");
            print(title);
            std::snprintf(title, sizeof title, "%s, %s image: %u colour clears, %u depth clears, %u zero copies",
                          queues[i].name, is_color ? "colour" : "depth", want.clear_color, want.clear_depth,
                          want.zero_copies);
            expect(count("clear_color_image") == want.clear_color &&
                       count("clear_depth_stencil_image") == want.clear_depth &&
                       count("copy_buffer_to_image") == want.zero_copies,
                   title);
        }
    }

    if (failures) return 1;
    std::puts("replay_copy_queue_clear_stage_test: COPY barriers keep CLEAR; COPY queues never clear an image");
    return 0;
}
