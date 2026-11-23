// The barriers replay records for WriteBufferImmediate (WriteImmediate), which the written values cannot show:
//   - DEFAULT adds none; MARKER_IN waits for every earlier command (ALL_COMMANDS, no access) and MARKER_OUT waits for
//     them and their writes (ALL_COMMANDS, MEMORY_WRITE), both before the vkCmdUpdateBuffer (TRANSFER).
//   - On the DIRECT and COMPUTE queues an enhanced SYNC_COPY barrier after a DEFAULT write waits for the CLEAR stage,
//     where synchronization2 runs vkCmdUpdateBuffer, so a copy after the barrier reads the written word. The COPY
//     queue's mask (queue_stages_copy) drops that stage; OPTIONS3 does not advertise COPY lists until it keeps it.
// vkCmdPipelineBarrier, vkCmdPipelineBarrier2, vkCmdUpdateBuffer and vkCmdCopyBuffer are functions that record their
// calls. No Vulkan loader, GPU or game data.
#ifndef VK_NO_PROTOTYPES
#define VK_NO_PROTOTYPES
#endif
#include "../src/vulkan_replay.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {
using namespace leonida;

// Every recorded call in order: a legacy barrier, the memory barriers of a vkCmdPipelineBarrier2, an update or a copy.
struct Call {
    enum Kind { barrier, barrier2, update, copy } kind;
    VkPipelineStageFlags2 src_stages = 0, dst_stages = 0;
    VkAccessFlags2 src_access = 0, dst_access = 0;
    std::uint32_t value = 0;
};
std::vector<Call> g_calls;

void VKAPI_CALL record_barrier(VkCommandBuffer, VkPipelineStageFlags src, VkPipelineStageFlags dst, VkDependencyFlags,
                               std::uint32_t count, const VkMemoryBarrier* barriers, std::uint32_t,
                               const VkBufferMemoryBarrier*, std::uint32_t, const VkImageMemoryBarrier*) {
    Call call{Call::barrier, src, dst};
    for (std::uint32_t i = 0; i < count; ++i) {
        call.src_access |= barriers[i].srcAccessMask;
        call.dst_access |= barriers[i].dstAccessMask;
    }
    g_calls.push_back(call);
}

void VKAPI_CALL record_barrier2(VkCommandBuffer, const VkDependencyInfo* info) {
    for (std::uint32_t i = 0; i < info->memoryBarrierCount; ++i) {
        const VkMemoryBarrier2& b = info->pMemoryBarriers[i];
        g_calls.push_back({Call::barrier2, b.srcStageMask, b.dstStageMask, b.srcAccessMask, b.dstAccessMask});
    }
}

void VKAPI_CALL record_update(VkCommandBuffer, VkBuffer, VkDeviceSize, VkDeviceSize size, const void* data) {
    Call call{Call::update};
    if (size == sizeof call.value) std::memcpy(&call.value, data, sizeof call.value);
    g_calls.push_back(call);
}

void VKAPI_CALL record_copy(VkCommandBuffer, VkBuffer, VkBuffer, std::uint32_t, const VkBufferCopy*) {
    g_calls.push_back({Call::copy});
}

// Replay needs every copy function before it takes a copy; these lists copy only buffers.
void VKAPI_CALL no_buffer_to_image(VkCommandBuffer, VkBuffer, VkImage, VkImageLayout, std::uint32_t,
                                   const VkBufferImageCopy*) {}
void VKAPI_CALL no_image_to_buffer(VkCommandBuffer, VkImage, VkImageLayout, VkBuffer, std::uint32_t,
                                   const VkBufferImageCopy*) {}
void VKAPI_CALL no_image_copy(VkCommandBuffer, VkImage, VkImageLayout, VkImage, VkImageLayout, std::uint32_t,
                              const VkImageCopy*) {}

int failures = 0;
void expect(bool ok, const char* what) {
    std::printf("%s %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok) ++failures;
}
}  // namespace

int main() {
    ResourceBinding written{}, copied{};
    written.buffer = reinterpret_cast<VkBuffer>(std::uintptr_t(1));
    written.size = 256;
    written.va = 0x10000;
    copied.buffer = reinterpret_cast<VkBuffer>(std::uintptr_t(2));
    copied.size = 256;
    copied.va = 0x20000;
    const auto resolve = [&](ResourceId id) -> const ResourceBinding* {
        return id == 1 ? &written : id == 2 ? &copied : nullptr;
    };

    ReplayFunctions vk{};
    vk.pipeline_barrier = record_barrier;
    vk.pipeline_barrier2 = record_barrier2;
    vk.update_buffer = record_update;
    vk.copy_buffer = record_copy;
    vk.copy_buffer_to_image = no_buffer_to_image;
    vk.copy_image_to_buffer = no_image_to_buffer;
    vk.copy_image = no_image_copy;
    vk.buffer_at = [&](std::uint64_t va, std::uint64_t, VkBuffer& buffer, VkDeviceSize& offset) {
        if (va < written.va || va - written.va >= written.size) return false;
        buffer = written.buffer;
        offset = va - written.va;
        return true;
    };
    const VkCommandBuffer cb = reinterpret_cast<VkCommandBuffer>(std::uintptr_t(0x10));

    // 1. Each mode alone: its barrier, then the update with the value.
    {
        namespace m = write_immediate_mode;
        struct Mode {
            std::uint32_t mode;
            const char* what;
            bool barrier;
            VkAccessFlags2 src_access, dst_access;
        };
        const Mode modes[] = {
            {m::normal, "DEFAULT adds no barrier", false, 0, 0},
            {m::marker_in, "MARKER_IN waits for every earlier command (ALL_COMMANDS -> TRANSFER, no access)", true, 0,
             0},
            {m::marker_out, "MARKER_OUT waits for every earlier command and its writes (MEMORY_WRITE -> TRANSFER_WRITE)",
             true, VK_ACCESS_MEMORY_WRITE_BIT, VK_ACCESS_TRANSFER_WRITE_BIT},
        };
        for (const Mode& mode : modes) {
            CommandList list;
            const bool recorded = list.write_immediate({written.va + 8, 0x57420000u | mode.mode, mode.mode}) &&
                                  list.close();
            g_calls.clear();
            const bool replayed = recorded && replay(list, cb, resolve, vk) == ReplayResult::success;
            const std::size_t barriers = mode.barrier ? 1 : 0;
            bool ok = replayed && g_calls.size() == barriers + 1 && g_calls.back().kind == Call::update &&
                      g_calls.back().value == (0x57420000u | mode.mode);
            if (ok && mode.barrier) {
                const Call& b = g_calls.front();
                ok = b.kind == Call::barrier && b.src_stages == VK_PIPELINE_STAGE_ALL_COMMANDS_BIT &&
                     b.dst_stages == VK_PIPELINE_STAGE_TRANSFER_BIT && b.src_access == mode.src_access &&
                     b.dst_access == mode.dst_access;
            }
            expect(ok, mode.what);
        }
    }

    // 2. A DEFAULT write, an enhanced SYNC_COPY buffer barrier (COPY_DEST -> COPY_SOURCE) and a copy of the word, on
    //    the DIRECT queue (no mask) and the COMPUTE queue (queue_stages_compute): between the update and the copy there
    //    is a barrier whose first scope holds the update's CLEAR stage and its write.
    {
        const struct {
            const char* what;
            VkPipelineStageFlags2 queue_stages;
        } queues[] = {
            {"on the DIRECT queue a SYNC_COPY barrier waits for a DEFAULT write (CLEAR stage, transfer write)", 0},
            {"on the COMPUTE queue a SYNC_COPY barrier waits for a DEFAULT write (CLEAR stage, transfer write)",
             detail::queue_stages_compute},
        };
        constexpr VkPipelineStageFlags2 clear_scope =
            VK_PIPELINE_STAGE_2_CLEAR_BIT | VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT | VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
        constexpr VkAccessFlags2 write_scope = VK_ACCESS_2_TRANSFER_WRITE_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT;
        for (const auto& queue : queues) {
            vk.queue_stages = queue.queue_stages;
            CommandList list;
            const bool recorded =
                list.write_immediate({written.va, 0x434f5059, write_immediate_mode::normal}) &&
                list.barrier(BufferBarrier{d3d12_sync::copy, d3d12_sync::copy, d3d12_access::copy_dest,
                                           d3d12_access::copy_source, 1}) &&
                list.copy_buffer({2, 0, 1, 0, 4}) && list.close();
            g_calls.clear();
            const bool replayed = recorded && replay(list, cb, resolve, vk) == ReplayResult::success;
            // The calls in order: the update, then a barrier holding it, then the copy.
            bool update_seen = false, barrier_seen = false, ordered = false;
            for (const Call& call : g_calls) {
                if (call.kind == Call::update) update_seen = true;
                if (call.kind == Call::barrier2 && update_seen && (call.src_stages & clear_scope) &&
                    (call.src_access & write_scope))
                    barrier_seen = true;
                if (call.kind == Call::copy) {
                    ordered = barrier_seen;
                    break;
                }
            }
            if (replayed && !ordered)
                for (const Call& call : g_calls)
                    if (call.kind == Call::barrier2)
                        std::fprintf(stderr, "  barrier src stages 0x%llx access 0x%llx\n",
                                     static_cast<unsigned long long>(call.src_stages),
                                     static_cast<unsigned long long>(call.src_access));
            expect(replayed && ordered, queue.what);
        }
        vk.queue_stages = 0;
    }

    std::printf("replay_write_immediate_barriers_test: %s\n", failures ? "FAILED" : "OK");
    return failures ? 1 : 0;
}
