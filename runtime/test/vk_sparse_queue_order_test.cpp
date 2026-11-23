// A held sparse mapping that submits no work must pass its GPU wait to the next submission.
// Raw waits test the submitted-signal path. Logical waits test GPU completion and queue destruction.
#include "../src/vulkan_backend.h"

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>

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
    VkSemaphore producer = VK_NULL_HANDLE;
    std::uint64_t producer_completed = 0;
    unsigned host_waits = 0;
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
    vk.vkQueueWaitIdle = +[](VkQueue) { return VK_SUCCESS; };
    vk.vkQueueBindSparse = +[](VkQueue, std::uint32_t count, const VkBindSparseInfo*, VkFence) {
        trace.binds += count;
        return VK_ERROR_OUT_OF_HOST_MEMORY;
    };
    vk.vkQueueSubmit = +[](VkQueue, std::uint32_t count, const VkSubmitInfo* info, VkFence) {
        trace.submits += count;
        if (count != 1) return VK_ERROR_UNKNOWN;
        trace.wait_count = info->waitSemaphoreCount;
        const auto* timeline = static_cast<const VkTimelineSemaphoreSubmitInfo*>(info->pNext);
        if (!timeline || timeline->sType != VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO ||
            timeline->waitSemaphoreValueCount != trace.wait_count) return VK_ERROR_UNKNOWN;
        if (trace.wait_count == 1) {
            trace.wait_semaphore = info->pWaitSemaphores[0];
            trace.wait_value = timeline->pWaitSemaphoreValues[0];
        }
        return VK_SUCCESS;
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
    auto buffer = std::make_shared<SparseBuffer>(context.device(), context.vk.vkDestroyBuffer);
    buffer->buffer = handle<VkBuffer>(1);
    buffer->tile_count = 1;
    buffer->memory_types = 1;
    ResourceBinding binding{};
    binding.buffer = buffer->buffer;
    binding.size = tile_size_in_bytes;
    binding.sparse_buffer = true;
    context.add_sparse_buffer(resource, binding, buffer);
    const auto heap = test.reject_bind
        ? std::make_shared<const TileHeapAllocation>(TileHeapAllocation{handle<VkDeviceMemory>(1), 0,
                                                                       tile_size_in_bytes, 0})
        : nullptr;
    const std::array regions{BufferTileRegion{0, 1}};
    const std::array ranges{BufferTileRange{test.reject_bind ? TileRangeMode::sequential : TileRangeMode::skip, 0, 1}};

    auto logical = std::make_shared<LogicalFence>(context, 0);
    if (test.logical_gate) {
        auto progress = std::make_shared<QueueProgress>(context.device(), handle<VkQueue>(2),
                                                        context.vk.vkDestroySemaphore);
        trace.producer = progress->semaphore = handle<VkSemaphore>(99);
        if (!expect(logical->submit_signal({progress, producer_ticket}, first_wait, [] { return VK_SUCCESS; }) ==
                        VK_SUCCESS, "accepted producer signal"))
            return false;
        queue->wait_fence(logical, first_wait);
    } else {
        queue->wait(gate.semaphore(), first_wait);
    }
    if (!expect(queue->update_tiles(resource, regions, ranges, [&] { return heap; }),
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
           expect(trace.binds == (test.reject_bind ? 1u : 0u) && buffer->mapped.empty(),
                  "the mapping was skipped or its bind failed without changing the map") &&
           expect(trace.submits == 1 && trace.wait_count == 1 && trace.wait_semaphore == expected_semaphore &&
                      trace.wait_value == expected_ticket,
                  "the following submission still carries the producer ticket exactly once");
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
    if (ok) std::puts("Sparse queue order: deferred no-op and rejected mappings preserve GPU waits");
    return ok ? 0 : 1;
}
