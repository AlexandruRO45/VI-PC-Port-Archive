// A held sparse mapping that submits no work must pass its GPU wait to the next submission.
// Raw waits test the submitted-signal path. Logical waits test GPU completion and queue destruction.
#include "../src/vulkan_backend.h"

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <memory>
#include <utility>

namespace {

using namespace leonida;

template<class T> T handle(std::uintptr_t value) {
    return reinterpret_cast<T>(value);
}

struct Trace {
    std::uintptr_t next_semaphore = 1;
    unsigned binds = 0;
    unsigned submits = 0;
    unsigned wait_count = 0;
    VkSemaphore wait_semaphore = VK_NULL_HANDLE;
    std::uint64_t wait_value = 0;
    VkSemaphore signal_semaphore = VK_NULL_HANDLE;
    std::uint64_t signal_value = 0;
    VkResult next_submit_result = VK_SUCCESS;
    VkSemaphore producer = VK_NULL_HANDLE;
    std::uint64_t producer_completed = 0;
    unsigned host_waits = 0;
    VkResult bind_result = VK_ERROR_OUT_OF_HOST_MEMORY;
    std::function<void()> on_idle;
};

Trace trace;

VulkanDeviceFunctions fake_driver() {
    VulkanDeviceFunctions vk;
    vk.vkCreateCommandPool = +[](VkDevice, const VkCommandPoolCreateInfo*, const VkAllocationCallbacks*,
                                 VkCommandPool* pool) {
        *pool = handle<VkCommandPool>(1);
        return VK_SUCCESS;
    };
    vk.vkDestroyCommandPool = +[](VkDevice, VkCommandPool, const VkAllocationCallbacks*) {};
    vk.vkCreateSemaphore = +[](VkDevice, const VkSemaphoreCreateInfo*, const VkAllocationCallbacks*,
                               VkSemaphore* semaphore) {
        *semaphore = handle<VkSemaphore>(trace.next_semaphore++);
        return VK_SUCCESS;
    };
    vk.vkDestroySemaphore = +[](VkDevice, VkSemaphore, const VkAllocationCallbacks*) {};
    vk.vkGetSemaphoreCounterValue = +[](VkDevice, VkSemaphore semaphore, std::uint64_t* value) {
        *value = semaphore == trace.producer ? trace.producer_completed : 0;
        return VK_SUCCESS;
    };
    vk.vkWaitSemaphores = +[](VkDevice, const VkSemaphoreWaitInfo* info, std::uint64_t) {
        if (info->semaphoreCount != 1 || info->pSemaphores[0] != trace.producer) return VK_ERROR_UNKNOWN;
        trace.producer_completed = info->pValues[0];
        ++trace.host_waits;
        return VK_SUCCESS;
    };
    vk.vkDestroyBuffer = +[](VkDevice, VkBuffer, const VkAllocationCallbacks*) {};
    vk.vkQueueWaitIdle = +[](VkQueue) {
        if (trace.on_idle) trace.on_idle();
        return VK_SUCCESS;
    };
    vk.vkQueueBindSparse = +[](VkQueue, std::uint32_t count, const VkBindSparseInfo*, VkFence) {
        trace.binds += count;
        return trace.bind_result;
    };
    vk.vkQueueSubmit = +[](VkQueue, std::uint32_t count, const VkSubmitInfo* info, VkFence) {
        trace.submits += count;
        if (count != 1) return VK_ERROR_UNKNOWN;
        trace.wait_count = info->waitSemaphoreCount;
        const auto* timeline = static_cast<const VkTimelineSemaphoreSubmitInfo*>(info->pNext);
        if (!timeline || timeline->sType != VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO ||
            timeline->waitSemaphoreValueCount != trace.wait_count || !info->signalSemaphoreCount)
            return VK_ERROR_UNKNOWN;
        trace.signal_semaphore = info->pSignalSemaphores[0];
        trace.signal_value = timeline->pSignalSemaphoreValues[0];
        if (trace.wait_count == 1) {
            trace.wait_semaphore = info->pWaitSemaphores[0];
            trace.wait_value = timeline->pWaitSemaphoreValues[0];
        }
        return std::exchange(trace.next_submit_result, VK_SUCCESS);
    };
    return vk;
}

PFN_vkVoidFunction VKAPI_CALL device_proc(VkDevice, const char* name) {
    static const auto driver = fake_driver();
#define MOCK_DEVICE_FUNCTION(n) \
    if (std::strcmp(name, #n) == 0) return reinterpret_cast<PFN_vkVoidFunction>(driver.n);
    LEONIDA_VK_DEVICE_FUNCTIONS(MOCK_DEVICE_FUNCTION)
#undef MOCK_DEVICE_FUNCTION
    return nullptr;
}

struct Case {
    const char* label;
    bool reject_bind;
    bool queued_signal;
    bool higher_following_wait;
    bool logical_gate = false;
    bool release_queue = false;
};

bool run(const Case& test) {
    const auto expect = [&](bool ok, const char* check) {
        if (!ok) std::fprintf(stderr, "%s: %s\n", test.label, check);
        return ok;
    };
    trace = {};
    constexpr std::uint64_t first_wait = 3;
    constexpr std::uint64_t producer_ticket = 11;
    VulkanContext context(device_proc, handle<VkDevice>(1), true);
    VulkanFence gate(context, 0), done(context, 0);
    auto queue = std::make_unique<VulkanQueue>(context, handle<VkQueue>(1), 0);
    if (!expect(queue->ok() && gate.semaphore() && done.semaphore(), "fake queue and fences initialized"))
        return false;

    constexpr ResourceId resource = 1;
    auto buffer = std::make_shared<SparseResource>(context.device(), context.vk);
    buffer->buffer = handle<VkBuffer>(1);
    buffer->tile_count = 1;
    buffer->memory_types = 1;
    ResourceBinding binding{};
    binding.buffer = buffer->buffer;
    binding.size = tile_size_in_bytes;
    binding.sparse_buffer = true;
    context.add_sparse_resource(resource, binding, buffer);
    const auto heap = test.reject_bind
        ? std::make_shared<const TileHeapAllocation>(TileHeapAllocation{handle<VkDeviceMemory>(1), 0,
                                                                       tile_size_in_bytes, 0})
        : nullptr;
    const std::array regions{TileRegion{0, 1}};
    const std::array ranges{TileRange{test.reject_bind ? TileRangeMode::sequential : TileRangeMode::skip, 0, 1}};

    auto logical = std::make_shared<LogicalFence>(context, 0);
    std::weak_ptr<int> retained_wait_pin;
    if (test.logical_gate) {
        auto progress = std::make_shared<QueueProgress>(context.device(), handle<VkQueue>(2),
                                                        context.vk.vkDestroySemaphore);
        trace.producer = progress->semaphore = handle<VkSemaphore>(99);
        if (!expect(logical->submit_signal({progress, producer_ticket}, first_wait,
                                           [](const FencePoint&) { return VK_SUCCESS; }) == VK_SUCCESS,
                    "accepted producer signal"))
            return false;
        queue->wait_fence(logical, first_wait);
    } else {
        const auto wait_pin = std::make_shared<int>(0);
        retained_wait_pin = wait_pin;
        queue->wait(gate.semaphore(), first_wait, [&] { return wait_pin; });
    }
    if (!expect(queue->update_tiles(resource, regions, ranges, [&](bool, std::uint32_t) { return heap; }),
                "accepted the mapping while the producer was unsignaled"))
        return false;
    queue->marker(DebugMarker::Kind::set, "marker after mapping");
    const auto expected_wait = first_wait + test.higher_following_wait;
    if (test.higher_following_wait) queue->wait(gate.semaphore(), expected_wait);
    if (test.queued_signal &&
        !expect(queue->signal(done.semaphore(), 1), "accepted the following signal while the gate was unsignaled"))
        return false;
    if (!expect(trace.submits == 0 && trace.binds == 0, "nothing submitted before the producer signal"))
        return false;

    if (test.release_queue) {
        queue.reset();
    } else if (test.logical_gate) {
        trace.producer_completed = producer_ticket;
        if (!expect(logical->completed() == first_wait, "producer completion updates the logical fence"))
            return false;
    } else {
        context.holds().signaled(gate.semaphore(), expected_wait);
        context.holds().pump_ready();
    }
    if (!test.queued_signal && !expect(queue->signal(done.semaphore(), 1), "accepted the later signal"))
        return false;
    const VkSemaphore expected_semaphore = test.logical_gate ? trace.producer : gate.semaphore();
    const auto expected_ticket = test.logical_gate ? producer_ticket : expected_wait;
    return expect(trace.host_waits == (test.release_queue ? 1u : 0u),
                  "queue destruction waits for an already submitted logical signal") &&
           expect(test.logical_gate || gate.completed() < first_wait, "raw producer has not completed on the GPU") &&
           expect(test.logical_gate || !retained_wait_pin.expired(),
                  "the forwarded raw wait stays pinned until its submission completes") &&
           expect(trace.binds == (test.reject_bind ? 1u : 0u) && buffer->mapped.empty(),
                  "the mapping was skipped or its bind failed without changing the map") &&
           expect(trace.submits == 1 && trace.wait_count == 1 && trace.wait_semaphore == expected_semaphore &&
                      trace.wait_value == expected_ticket,
                  "the following submission still carries the producer ticket exactly once");
}

bool signals_on_distinct_queues() {
    trace = {};
    VulkanContext context(device_proc, handle<VkDevice>(1), true);
    VulkanQueue first(context, handle<VkQueue>(1), 0), second(context, handle<VkQueue>(2), 0);
    const auto fence = std::make_shared<LogicalFence>(context, 0);
    const auto waiting_for_ten = fence->wait(10);
    if (!first.ok() || !second.ok() || !first.signal_fence(fence, 10) || trace.wait_count != 0) return false;
    const auto predecessor = trace.signal_semaphore;
    const auto predecessor_value = trace.signal_value;

    // The first producer is still incomplete. A rejected second submit must not become the predecessor
    // of its retry. The accepted retry waits for the first producer's progress on the other VkQueue.
    trace.next_submit_result = VK_ERROR_OUT_OF_HOST_MEMORY;
    if (second.signal_fence(fence, 4) || !second.signal_fence(fence, 4) ||
        trace.wait_count != 1 || trace.wait_semaphore != predecessor || trace.wait_value != predecessor_value) {
        std::fputs("GPU signals on distinct Vulkan queues lost the first accepted predecessor\n", stderr);
        return false;
    }
    if (fence->completed() != 0) return false;
    trace.producer = trace.signal_semaphore;
    trace.producer_completed = trace.signal_value;
    return fence->completed() == 4 &&
           waiting_for_ten->status.load() == LogicalFenceWait::Status::ready &&
           waiting_for_ten->point.queue->semaphore == predecessor;
}

bool run_lost_bind(VkResult result, bool logical) {
    trace = {};
    trace.bind_result = result;
    const bool lost = result == VK_ERROR_DEVICE_LOST;
    VulkanContext context(device_proc, handle<VkDevice>(1), true);
    VulkanFence gate(context, 0);
    auto queue = std::make_unique<VulkanQueue>(context, handle<VkQueue>(1), 0);
    if (!queue->ok() || !gate.semaphore()) return false;

    constexpr ResourceId id = 1;
    auto buffer = std::make_shared<SparseResource>(context.device(), context.vk);
    buffer->buffer = handle<VkBuffer>(1);
    buffer->tile_count = 1;
    buffer->memory_types = 1;
    auto old_heap = std::make_shared<const TileHeapAllocation>(
        TileHeapAllocation{handle<VkDeviceMemory>(1), 0, tile_size_in_bytes, 0});
    auto new_heap = std::make_shared<const TileHeapAllocation>(
        TileHeapAllocation{handle<VkDeviceMemory>(2), 0, tile_size_in_bytes, 0});
    buffer->mapped.emplace(0, TileMapping{old_heap, 0});
    ResourceBinding binding{};
    binding.buffer = buffer->buffer;
    binding.size = tile_size_in_bytes;
    binding.sparse_buffer = true;
    context.add_sparse_resource(id, binding, buffer);

    auto wait_pin = std::make_shared<int>(0);
    const std::weak_ptr<int> watched_wait = wait_pin;
    const std::weak_ptr<SparseResource> watched_buffer = buffer;
    const std::weak_ptr<const TileHeapAllocation> watched_old = old_heap, watched_new = new_heap;
    std::weak_ptr<const QueueProgress> watched_progress;
    auto logical_gate = logical ? std::make_shared<LogicalFence>(context, 0) : nullptr;
    if (logical_gate) queue->wait_fence(logical_gate, 1);
    else queue->wait(gate.semaphore(), 1, [&] { return wait_pin; });
    const std::array regions{TileRegion{0, 1}};
    const std::array ranges{TileRange{TileRangeMode::sequential, 0, 1}};
    if (!queue->update_tiles(id, regions, ranges, [&](bool, std::uint32_t) { return new_heap; })) return false;

    // The held operation must keep the resource and allocations even if the application releases all
    // of them. Eject the mock's deferred-deletion reference too so only the queue owns the buffer.
    context.remove_resource(id);
    context.drain_deleted_resources(UINT64_MAX);
    old_heap.reset();
    new_heap.reset();
    wait_pin.reset();
    buffer.reset();
    if (logical_gate) {
        VulkanQueue producer(context, handle<VkQueue>(2), 0);
        if (!producer.ok() || !producer.signal_fence(logical_gate, 1)) return false;
        watched_progress = logical_gate->submitted_point(1).queue;
        trace.producer = trace.signal_semaphore;
        trace.producer_completed = trace.signal_value;
        if (logical_gate->completed() != 1) return false;
        logical_gate->close();
        logical_gate.reset();
    } else {
        context.holds().signaled(gate.semaphore(), 1);
        context.holds().pump_ready();
    }

    const auto wait_alive = [&] { return logical ? !watched_progress.expired() : !watched_wait.expired(); };
    const auto backing_alive = [&] {
        return !watched_buffer.expired() && !watched_old.expired() && !watched_new.expired();
    };
    const auto backing_gone = [&] {
        return watched_buffer.expired() && watched_old.expired() && watched_new.expired();
    };
    const bool before_idle = wait_alive() && (lost ? backing_alive() : backing_gone());
    bool at_idle = false;
    trace.on_idle = [&] { at_idle = wait_alive() && (!lost || backing_alive()); };
    queue.reset();
    trace.on_idle = {};
    const bool ok = trace.binds == 1 && trace.submits == static_cast<unsigned>(logical) &&
                    context.is_device_lost() == lost && before_idle && at_idle && backing_gone() && !wait_alive();
    if (!ok) std::fprintf(stderr, "%s deferred sparse bind lifetime: unexpected result for VkResult %d\n",
                          logical ? "logical" : "raw", result);
    return ok;
}

} // namespace

int main() {
    const Case cases[] = {
        {"no-op mapping and queued signal", false, true, false},
        {"no-op mapping and later signal", false, false, false},
        {"rejected sparse bind and queued signal", true, true, false},
        {"no-op mapping followed by a higher wait on the same fence", false, true, true},
        {"no-op mapping with a logical fence", false, true, false, true},
        {"rejected logical mapping when the consumer queue is released", true, true, false, true, true},
    };
    bool ok = true;
    for (const auto& test : cases) ok = run(test) && ok;
    ok = signals_on_distinct_queues() && ok;
    for (const auto result : {VK_ERROR_DEVICE_LOST, VK_ERROR_OUT_OF_HOST_MEMORY}) {
        ok = run_lost_bind(result, false) && ok;
        ok = run_lost_bind(result, true) && ok;
    }
    if (ok) std::puts("Sparse queue order: preserved raw and logical waits, bindings and cross-queue signals");
    return ok ? 0 : 1;
}
