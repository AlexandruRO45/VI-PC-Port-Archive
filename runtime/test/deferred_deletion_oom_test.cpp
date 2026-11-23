// CPU-only fake Vulkan procedures: a failed host allocation during drain must leave every queued handle available
// for a retry. No Vulkan loader, device, or game data.
#include "../src/vulkan_backend.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <initializer_list>
#include <new>

static int fail_after_new = -1;
void* operator new(std::size_t n) {
    if (fail_after_new == 0) {
        fail_after_new = -1;
        throw std::bad_alloc();
    }
    if (fail_after_new > 0) --fail_after_new;
    if (void* p = std::malloc(n ? n : 1)) return p;
    throw std::bad_alloc();
}
void operator delete(void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }

namespace {
using Queue = leonida::DeferredDeletionQueue;
using Kind = Queue::Kind;

struct Tracker {
    std::array<unsigned, 512> calls{};
    std::array<std::uintptr_t, 512> order{};
    std::size_t count = 0;
    bool invalid = false;

    void reset() { *this = {}; }
    void destroyed(std::uintptr_t id) {
        if (id >= calls.size() || count >= order.size()) {
            invalid = true;
            return;
        }
        ++calls[id];
        order[count++] = id;
        if (calls[id] != 1) invalid = true;
    }
} tracker;

Queue* reenter_queue = nullptr;
const leonida::VulkanDeviceFunctions* reenter_vk = nullptr;

void VKAPI_PTR destroy_buffer(VkDevice device, VkBuffer buffer, const VkAllocationCallbacks*) {
    const auto id = reinterpret_cast<std::uintptr_t>(buffer);
    tracker.destroyed(id);
    if (id == 300 && reenter_queue) {
        Queue* queue = reenter_queue;
        reenter_queue = nullptr;
        queue->push(5, Kind::buffer, 301);
        queue->drain(5, device, *reenter_vk);
    }
}

#define CHECK(condition)                                                                                              \
    do {                                                                                                              \
        if (!(condition)) {                                                                                           \
            std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition);                                            \
            return 1;                                                                                                 \
        }                                                                                                             \
    } while (0)
}  // namespace

int main() {
    leonida::VulkanDeviceFunctions vk{};
    vk.vkDestroyBuffer = destroy_buffer;
    const VkDevice device = reinterpret_cast<VkDevice>(1);

    for (int failure_after : {0, 1}) {
        tracker.reset();
        Queue queue(2);
        queue.push(4, Kind::buffer, 101);
        queue.push(4, Kind::buffer, 102);
        queue.push(8, Kind::buffer, 103);
        fail_after_new = failure_after;
        bool threw = false;
        try {
            queue.drain(4, device, vk);
        } catch (const std::bad_alloc&) {
            threw = true;
        }
        fail_after_new = -1;
        CHECK(threw && tracker.count == 0 && !tracker.invalid);
        queue.drain(4, device, vk);
        CHECK(tracker.count == 2 && tracker.order[0] == 101 && tracker.order[1] == 102);
        queue.drain(7, device, vk);
        CHECK(tracker.count == 2);
        queue.drain(8, device, vk);
        CHECK(tracker.count == 3 && tracker.order[2] == 103 && !tracker.invalid);
    }

    // Keep future entries in the ring while its head advances, then wrap its tail before failing a later drain.
    for (int failure_after : {1, 3}) {
        tracker.reset();
        Queue queue(2);  // minimum ring capacity is 16
        for (std::uint64_t id = 200; id < 212; ++id) queue.push(1, Kind::buffer, id);
        queue.push(3, Kind::buffer, 212);
        queue.push(3, Kind::buffer, 213);
        queue.drain(1, device, vk);
        CHECK(tracker.count == 12 && !tracker.invalid);
        tracker.reset();
        for (std::uint64_t id = 214; id < 218; ++id) queue.push(3, Kind::buffer, id);
        fail_after_new = failure_after;
        bool threw = false;
        try {
            queue.drain(3, device, vk);
        } catch (const std::bad_alloc&) {
            threw = true;
        }
        fail_after_new = -1;
        CHECK(threw && tracker.count == 0);
        queue.drain(3, device, vk);
        CHECK(tracker.count == 6 && !tracker.invalid);
        for (std::size_t i = 0; i < 6; ++i) CHECK(tracker.order[i] == 212 + i);
    }

    // Destruction callbacks must run outside the queue lock: one may enqueue and drain another handle.
    tracker.reset();
    {
        Queue queue(2);
        reenter_queue = &queue;
        reenter_vk = &vk;
        queue.push(5, Kind::buffer, 300);
        queue.drain(5, device, vk);
        CHECK(!reenter_queue && tracker.count == 2 && tracker.order[0] == 300 && tracker.order[1] == 301 &&
              !tracker.invalid);
    }
    std::puts("deferred_deletion_oom_test: all checks passed");
    return 0;
}
