// A held sparse mapping that submits no work must pass its GPU wait to the next submission.
// The fake Vulkan entry points keep the producer's counter at zero after its Signal is submitted.
#include "../src/vulkan_backend.h"

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
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
    vk.vkGetSemaphoreCounterValue = +[](VkDevice, VkSemaphore, std::uint64_t* value) {
        *value = 0;
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
};

bool run(const Case& test) {
    const auto expect = [&](bool ok, const char* check) {
        if (!ok) std::fprintf(stderr, "%s: %s\n", test.label, check);
        return ok;
    };
    trace = {};
    constexpr std::uint64_t first_wait = 3;
    VulkanContext context(device_proc, handle<VkDevice>(1), true);
    VulkanFence gate(context, 0), done(context, 0);
    VulkanQueue queue(context, handle<VkQueue>(1), 0);
    if (!expect(queue.ok() && gate.semaphore() && done.semaphore(), "fake queue and fences initialized"))
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

    auto wait_pin = std::make_shared<int>(0);
    const std::weak_ptr<int> retained_wait_pin = wait_pin;
    queue.wait(gate.semaphore(), first_wait, [&] { return wait_pin; });
    wait_pin.reset();
    if (!expect(queue.update_tiles(resource, regions, ranges, [&](bool, std::uint32_t) { return heap; }),
                "accepted the mapping while the producer was unsignaled"))
        return false;
    queue.marker(DebugMarker::Kind::set, "marker after mapping");
    const auto expected_wait = first_wait + test.higher_following_wait;
    if (test.higher_following_wait) queue.wait(gate.semaphore(), expected_wait);
    if (test.queued_signal &&
        !expect(queue.signal(done.semaphore(), 1), "accepted the following signal while the gate was unsignaled"))
        return false;
    if (!expect(trace.submits == 0 && trace.binds == 0, "nothing submitted before the producer signal"))
        return false;

    context.holds().signaled(gate.semaphore(), expected_wait);
    context.holds().pump_ready();
    if (!test.queued_signal && !expect(queue.signal(done.semaphore(), 1), "accepted the later signal"))
        return false;
    return expect(gate.completed() < first_wait, "producer has not completed on the GPU") &&
           expect(!retained_wait_pin.expired(), "the forwarded wait stays pinned until its submission completes") &&
           expect(trace.binds == (test.reject_bind ? 1u : 0u) && buffer->mapped.empty(),
                  "the mapping was skipped or its bind failed without changing the map") &&
           expect(trace.submits == 1 && trace.wait_count == 1 && trace.wait_semaphore == gate.semaphore() &&
                      trace.wait_value == expected_wait,
                  "the following submission still waits for the incomplete producer exactly once");
}

bool run_lost_bind(VkResult result) {
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
    queue->wait(gate.semaphore(), 1, [&] { return wait_pin; });
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
    context.holds().signaled(gate.semaphore(), 1);
    context.holds().pump_ready();
    const auto alive = [&] {
        return !watched_buffer.expired() && !watched_old.expired() && !watched_new.expired() && !watched_wait.expired();
    };
    const bool before_idle = !watched_wait.expired() && (lost ? alive() :
        watched_buffer.expired() && watched_old.expired() && watched_new.expired());
    bool at_idle = false;
    trace.on_idle = [&] { at_idle = lost ? alive() : !watched_wait.expired(); };
    queue.reset();
    trace.on_idle = {};
    const bool ok = trace.binds == 1 && trace.submits == 0 && context.is_device_lost() == lost &&
                    before_idle && at_idle && watched_buffer.expired() && watched_old.expired() &&
                    watched_new.expired() && watched_wait.expired();
    if (!ok) std::fprintf(stderr, "deferred sparse bind lifetime: unexpected result for VkResult %d\n", result);
    return ok;
}

} // namespace

int main() {
    const Case cases[] = {
        {"no-op mapping and queued signal", false, true, false},
        {"no-op mapping and later signal", false, false, false},
        {"rejected sparse bind and queued signal", true, true, false},
        {"no-op mapping followed by a higher wait on the same fence", false, true, true},
    };
    bool ok = true;
    for (const auto& test : cases) ok = run(test) && ok;
    for (const auto result : {VK_ERROR_DEVICE_LOST, VK_ERROR_OUT_OF_HOST_MEMORY})
        ok = run_lost_bind(result) && ok;
    if (ok) std::puts("Sparse queue order: forwarded GPU waits and retained device-loss bindings");
    return ok ? 0 : 1;
}
