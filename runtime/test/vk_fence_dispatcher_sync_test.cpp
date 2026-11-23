// FenceWaitDispatcher concurrency and device-lost regressions (runtime/src/vulkan_backend.h):
//   1. Completion callbacks in run() execute outside mutex_, so a callback can re-entrantly call
//      enqueue_fence_callback() and cancel_fence_waits() without deadlocking.
//   2. Cancelled callbacks in cancel() execute outside mutex_, so a callback can re-entrantly enqueue another wait
//      without deadlocking.
//   3. mark_device_lost() wakes a blocked vkWaitSemaphores() so pending waits complete and cancel()/stop() never hang.
//   4. vkGetSemaphoreCounterValue() returning VK_ERROR_DEVICE_LOST drains pending waits and marks device_lost_ instead
//      of spinning forever.
//   5. cancel() blocks until a callback already dispatched for that semaphore has stopped using it.
//   6. A re-entrant cancel() for a later callback in the same ready batch skips that callback.
//   7. Logical fence rewinds keep already-satisfied waits latched even when a queue advances during a counter scan.
// Skips with exit code 77 when no Vulkan 1.3 device is available. No game data.
#include "vk_test_device.h"
#include "../src/vulkan_backend.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <string_view>
#include <thread>
#include <utility>

#define EXPECT(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); std::_Exit(1); } \
} while (0)

namespace {

namespace driver {
PFN_vkGetDeviceProcAddr real_get_device = nullptr;
PFN_vkWaitSemaphores real_wait_semaphores = nullptr;
PFN_vkGetSemaphoreCounterValue real_get_counter = nullptr;
PFN_vkSignalSemaphore real_signal = nullptr;

std::atomic<bool> in_wait_semaphores{false};
std::atomic<bool> fail_counter_device_lost{false};
std::atomic<std::uint32_t> counter_calls{0};
std::atomic<std::uint32_t> wait_sem_count{0};

struct AdvanceOnSample {
    VkSemaphore sampled = VK_NULL_HANDLE;
    VkSemaphore advance = VK_NULL_HANDLE;
    std::uint64_t value = 0;
    std::uint64_t sampled_value = 0;
};
thread_local AdvanceOnSample advance_on_sample;

VkResult signal(VkDevice device, VkSemaphore semaphore, std::uint64_t value) {
    VkSemaphoreSignalInfo info{VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO};
    info.semaphore = semaphore;
    info.value = value;
    return real_signal(device, &info);
}

VKAPI_ATTR VkResult VKAPI_CALL wait_semaphores(VkDevice device, const VkSemaphoreWaitInfo* info, std::uint64_t timeout) {
    wait_sem_count.store(info->semaphoreCount, std::memory_order_release);
    in_wait_semaphores.store(true, std::memory_order_release);
    const VkResult res = real_wait_semaphores(device, info, timeout);
    in_wait_semaphores.store(false, std::memory_order_release);
    return res;
}

VKAPI_ATTR VkResult VKAPI_CALL get_semaphore_counter_value(VkDevice device, VkSemaphore sem, std::uint64_t* value) {
    counter_calls.fetch_add(1, std::memory_order_relaxed);
    if (fail_counter_device_lost.load(std::memory_order_acquire)) return VK_ERROR_DEVICE_LOST;
    if (sem == advance_on_sample.sampled) {
        const auto advance = std::exchange(advance_on_sample, AdvanceOnSample{});
        const VkResult result = signal(device, advance.advance, advance.value);
        if (result != VK_SUCCESS) return result;
        if (advance.sampled_value) {
            const VkResult sampled = signal(device, sem, advance.sampled_value);
            if (sampled != VK_SUCCESS) return sampled;
        }
    }
    return real_get_counter(device, sem, value);
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL get_device(VkDevice device, const char* name) {
    const std::string_view n = name;
    if (n == "vkWaitSemaphores") return reinterpret_cast<PFN_vkVoidFunction>(wait_semaphores);
    if (n == "vkGetSemaphoreCounterValue") return reinterpret_cast<PFN_vkVoidFunction>(get_semaphore_counter_value);
    return real_get_device(device, name);
}

void install(PFN_vkGetDeviceProcAddr device_proc, VkDevice device) {
    real_get_device = device_proc;
    real_wait_semaphores = reinterpret_cast<PFN_vkWaitSemaphores>(device_proc(device, "vkWaitSemaphores"));
    real_get_counter =
        reinterpret_cast<PFN_vkGetSemaphoreCounterValue>(device_proc(device, "vkGetSemaphoreCounterValue"));
    real_signal = reinterpret_cast<PFN_vkSignalSemaphore>(device_proc(device, "vkSignalSemaphore"));
}
}  // namespace driver

bool wait_flag(std::mutex& mu, std::condition_variable& cv, const bool& flag, int timeout_ms = 2000) {
    std::unique_lock lk(mu);
    return cv.wait_for(lk, std::chrono::milliseconds(timeout_ms), [&] { return flag; });
}

bool wait_until_in_vk_wait(int timeout_ms = 2000) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    while (!driver::in_wait_semaphores.load(std::memory_order_acquire)) {
        if (std::chrono::steady_clock::now() >= deadline) return false;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return true;
}

std::shared_ptr<leonida::QueueProgress> progress_timeline(leonida::VulkanContext& context, VkQueue physical_queue) {
    auto progress =
        std::make_shared<leonida::QueueProgress>(context.device(), physical_queue, context.vk.vkDestroySemaphore);
    VkSemaphoreTypeCreateInfo type{VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
    type.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
    const VkSemaphoreCreateInfo info{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, &type};
    EXPECT(context.vk.vkCreateSemaphore(context.device(), &info, nullptr, &progress->semaphore) == VK_SUCCESS);
    return progress;
}

void check_logical_fence_rewind(VkDevice device) {
    using namespace leonida;
    using Status = LogicalFenceWait::Status;

    VulkanContext context(driver::get_device, device, true);
    EXPECT(context.loaded());
    // These mock queue handles carry identity only. The real timeline semaphores are advanced by the test.
    const auto queue_a = reinterpret_cast<VkQueue>(std::uintptr_t{1});
    const auto queue_b = reinterpret_cast<VkQueue>(std::uintptr_t{2});
    const auto a = progress_timeline(context, queue_a), b = progress_timeline(context, queue_b);
    const auto fence = std::make_shared<LogicalFence>(context, 0);
    const auto old_wait = fence->wait(8);
    const auto expect_predecessor = [](const QueueProgress* queue, std::uint64_t ticket) {
        return [queue, ticket](const FencePoint& point) {
            EXPECT(point.queue.get() == queue && point.ticket == ticket);
            return VK_SUCCESS;
        };
    };
    const auto submitted = expect_predecessor(nullptr, 0);
    EXPECT(fence->submit_signal({a, 1}, 10, submitted) == VK_SUCCESS);
    EXPECT(fence->submit_signal({b, 1}, 6, expect_predecessor(a.get(), 1)) == VK_SUCCESS);
    EXPECT(fence->submit_signal({a, 2}, 4, expect_predecessor(b.get(), 1)) == VK_SUCCESS);

    // A is sampled at 0. The GPU reaches A1 then B1 by the time the same scan samples B. The
    // predecessor wait proves A1 has run, so its existing wait must be satisfied before publishing B1.
    driver::advance_on_sample = {b->semaphore, a->semaphore, 1, 1};
    EXPECT(fence->completed() == 6);
    EXPECT(driver::advance_on_sample.sampled == VK_NULL_HANDLE);
    EXPECT(old_wait->status.load() == Status::ready && old_wait->point.queue == a && old_wait->point.ticket == 1);
    EXPECT(driver::signal(device, a->semaphore, 2) == VK_SUCCESS);
    EXPECT(fence->completed() == 4);
    EXPECT(fence->wait(8)->status.load() == Status::pending);

    // The queue reaches 12, but no dispatcher callback runs between that completion and CPU Signal(3).
    // The host signal must first collect 12, satisfy the old waiter, and then leave the visible value at 3.
    const auto prior = fence->wait(12);
    EXPECT(fence->submit_signal({a, 3}, 12, submitted) == VK_SUCCESS);
    EXPECT(driver::signal(device, a->semaphore, 3) == VK_SUCCESS);
    EXPECT(fence->signal(3));
    EXPECT(fence->completed() == 3);
    EXPECT(prior->status.load() == Status::ready && prior->point.queue == a && prior->point.ticket == 3);
    const auto relayed = fence->wait(3);
    EXPECT(relayed && relayed->status.load() == Status::ready &&
           relayed->point.queue == a && relayed->point.ticket == 3);

    // Two D3D12 queues can share the same Vulkan queue but have distinct progress semaphores.
    // A was sampled at ticket 3. By the time C is sampled both signals have run, with A first.
    const auto c = progress_timeline(context, queue_a);
    const auto shared_queue_fence = std::make_shared<LogicalFence>(context, 0);
    const auto shared_wait = shared_queue_fence->wait(8);
    EXPECT(shared_queue_fence->submit_signal({a, 4}, 10, submitted) == VK_SUCCESS);
    EXPECT(shared_queue_fence->submit_signal({c, 1}, 4, submitted) == VK_SUCCESS);
    driver::advance_on_sample = {c->semaphore, a->semaphore, 4, 1};
    EXPECT(shared_queue_fence->completed() == 4);
    EXPECT(driver::advance_on_sample.sampled == VK_NULL_HANDLE);
    EXPECT(shared_wait->status.load() == Status::ready);
    EXPECT(shared_wait->point.queue == a && shared_wait->point.ticket == 4);
    EXPECT(shared_queue_fence->completed() == 4);

    // A winning ANY notification removes the registrations on the other fences. A late registration
    // for the same completed wait is ignored too.
    int any_owner = 0;
    bool any_done = false;
    const auto done = [&] { return any_done; };
    auto callback_data = std::make_shared<int>(0);
    const std::weak_ptr<int> callback_lifetime = callback_data;
    fence->notify(99, [data = std::move(callback_data)] { ++*data; }, &any_owner, done);
    EXPECT(!callback_lifetime.expired());
    shared_queue_fence->notify(4, [&] {
        any_done = true;
        fence->cancel_events(&any_owner);
    }, &any_owner, done);
    EXPECT(any_done && callback_lifetime.expired());
    auto late_data = std::make_shared<int>(0);
    const std::weak_ptr<int> late_lifetime = late_data;
    fence->notify(99, [data = std::move(late_data)] { ++*data; }, &any_owner, done);
    EXPECT(late_lifetime.expired());

    const auto pending = fence->wait(12);
    EXPECT(pending->status.load() == Status::pending);
    std::mutex mu;
    std::condition_variable cv;
    bool notified = false;
    fence->notify(12, [&] {
        std::lock_guard lock(mu);
        notified = true;
        cv.notify_all();
    });
    context.mark_device_lost();
    EXPECT(wait_flag(mu, cv, notified));
    fence->close();
    EXPECT(fence->completed() == UINT64_MAX);
    EXPECT(pending->status.load() == Status::cancelled);
    EXPECT(prior->status.load() == Status::ready);
    std::puts("logical fence: stable queue scan, latched waits, CPU rewind and device loss ok");
}

// Waits until run() has re-armed vkWaitSemaphores with exactly `count` semaphores (wakeup + N pending waits).
bool wait_until_wait_sem_count(std::uint32_t count, int timeout_ms = 2000) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    while (driver::wait_sem_count.load(std::memory_order_acquire) != count) {
        if (std::chrono::steady_clock::now() >= deadline) return false;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return true;
}

}  // namespace

int main() {
    using namespace leonida;

    vktest::Device d;
    if (!d.create(false)) {
        std::fprintf(stderr, "vk_fence_dispatcher_sync_test: no Vulkan 1.3 device, skipped\n");
        return 77;
    }
    driver::install(d.device_proc, d.device);

    // 1. Re-entrant completion callback in run(): callback calls enqueue_fence_callback() and cancel_fence_waits().
    {
        VulkanContext context(driver::get_device, d.device, true);
        EXPECT(context.loaded());
        VulkanFence fence_a(context, 0);
        VulkanFence fence_b(context, 0);
        VulkanFence fence_c(context, 0);
        EXPECT(fence_a.semaphore() != VK_NULL_HANDLE && fence_b.semaphore() != VK_NULL_HANDLE &&
               fence_c.semaphore() != VK_NULL_HANDLE);

        std::mutex mu;
        std::condition_variable cv;
        bool first_ran = false;
        bool chained_ran = false;
        bool c_cancelled = false;

        context.enqueue_fence_callback(fence_c.semaphore(), 99, [&] {
            std::lock_guard lk(mu);
            c_cancelled = true;
            cv.notify_all();
        });
        context.enqueue_fence_callback(fence_a.semaphore(), 1, [&] {
            context.enqueue_fence_callback(fence_b.semaphore(), 1, [&] {
                std::lock_guard lk(mu);
                chained_ran = true;
                cv.notify_all();
            });
            context.cancel_fence_waits(fence_c.semaphore());
            std::lock_guard lk(mu);
            first_ran = true;
            cv.notify_all();
        });

        EXPECT(fence_a.signal(1));
        EXPECT(wait_flag(mu, cv, first_ran));
        EXPECT(wait_flag(mu, cv, c_cancelled));
        EXPECT(fence_b.signal(1));
        EXPECT(wait_flag(mu, cv, chained_ran));

        context.cancel_fence_waits(fence_a.semaphore());
        context.cancel_fence_waits(fence_b.semaphore());
        std::printf("fence dispatcher: re-entrant completion callback in run() ok\n");

        // 2. Re-entrant callback in cancel(): cancelled callback calls enqueue_fence_callback() on another fence.
        VulkanFence fence_d(context, 0);
        VulkanFence fence_e(context, 0);
        EXPECT(fence_d.semaphore() != VK_NULL_HANDLE && fence_e.semaphore() != VK_NULL_HANDLE);

        bool cancel_returned = false;
        bool e_ran = false;
        context.enqueue_fence_callback(fence_d.semaphore(), 50, [&] {
            context.enqueue_fence_callback(fence_e.semaphore(), 1, [&] {
                std::lock_guard lk(mu);
                e_ran = true;
                cv.notify_all();
            });
        });

        std::thread cancel_thread([&] {
            context.cancel_fence_waits(fence_d.semaphore());
            std::lock_guard lk(mu);
            cancel_returned = true;
            cv.notify_all();
        });
        const bool ok_cancel = wait_flag(mu, cv, cancel_returned);
        EXPECT(ok_cancel);
        cancel_thread.join();

        EXPECT(fence_e.signal(1));
        EXPECT(wait_flag(mu, cv, e_ran));
        context.cancel_fence_waits(fence_e.semaphore());
        std::printf("fence dispatcher: re-entrant callback in cancel() ok\n");
    }

    // 3. mark_device_lost() wakes blocked vkWaitSemaphores(), completes pending waits, and stop()/cancel() never hang.
    {
        driver::in_wait_semaphores.store(false, std::memory_order_release);
        std::mutex mu;
        std::condition_variable cv;
        bool callback_fired = false;
        bool teardown_done = false;

        std::thread lost_test([&] {
            VulkanContext lost_ctx(driver::get_device, d.device, true);
            if (!lost_ctx.loaded()) std::_Exit(1);
            VulkanFence stuck(lost_ctx, 0);
            if (stuck.semaphore() == VK_NULL_HANDLE) std::_Exit(1);

            lost_ctx.enqueue_fence_callback(stuck.semaphore(), 100, [&] {
                std::lock_guard lk(mu);
                callback_fired = true;
                cv.notify_all();
            });
            if (!wait_until_in_vk_wait()) std::_Exit(1);

            lost_ctx.mark_device_lost();
            if (!wait_flag(mu, cv, callback_fired)) std::_Exit(1);

            // cancel() and ~VulkanContext() -> stop() must return promptly even after device_lost_ is set.
            lost_ctx.cancel_fence_waits(stuck.semaphore());
        });

        std::thread joiner([&] {
            lost_test.join();
            std::lock_guard lk(mu);
            teardown_done = true;
            cv.notify_all();
        });
        const bool ok_lost = wait_flag(mu, cv, teardown_done, 3000);
        EXPECT(ok_lost);
        joiner.join();
        EXPECT(callback_fired);
        std::printf("fence dispatcher: mark_device_lost() wakes blocked wait and teardown completes ok\n");
    }

    // 4. vkGetSemaphoreCounterValue() returning VK_ERROR_DEVICE_LOST drains pending waits instead of spinning forever.
    {
        driver::in_wait_semaphores.store(false, std::memory_order_release);
        driver::fail_counter_device_lost.store(false, std::memory_order_release);
        driver::counter_calls.store(0, std::memory_order_release);

        std::mutex mu;
        std::condition_variable cv;
        bool drained = false;
        bool done = false;
        bool marked_lost = false;
        std::uint32_t observed_calls = 0;

        std::thread err_test([&] {
            VulkanContext err_ctx(driver::get_device, d.device, true);
            if (!err_ctx.loaded()) std::_Exit(1);
            VulkanFence sem(err_ctx, 0);
            if (sem.semaphore() == VK_NULL_HANDLE) std::_Exit(1);

            err_ctx.enqueue_fence_callback(sem.semaphore(), 1, [&] {
                std::lock_guard lk(mu);
                drained = true;
                cv.notify_all();
            });
            if (!wait_until_in_vk_wait()) std::_Exit(1);

            driver::fail_counter_device_lost.store(true, std::memory_order_release);
            if (!sem.signal(1)) std::_Exit(1);
            if (!wait_flag(mu, cv, drained, 1500)) std::_Exit(1);

            marked_lost = err_ctx.is_device_lost();
            observed_calls = driver::counter_calls.load(std::memory_order_acquire);
            err_ctx.cancel_fence_waits(sem.semaphore());
        });

        std::thread joiner([&] {
            err_test.join();
            std::lock_guard lk(mu);
            done = true;
            cv.notify_all();
        });
        const bool ok_err = wait_flag(mu, cv, done, 3000);
        driver::fail_counter_device_lost.store(false, std::memory_order_release);
        EXPECT(ok_err);
        joiner.join();
        EXPECT(drained);
        EXPECT(marked_lost);
        EXPECT(observed_calls >= 1 && observed_calls <= 4);
        std::printf("fence dispatcher: vkGetSemaphoreCounterValue DEVICE_LOST drains without spin (%u calls) ok\n",
                    observed_calls);
    }

    // 5. cancel() must not return while run() is dispatching a callback that still uses the semaphore.
    {
        VulkanContext context(driver::get_device, d.device, true);
        EXPECT(context.loaded());
        VulkanFence fence(context, 0);
        const VkSemaphore sem = fence.semaphore();
        EXPECT(sem != VK_NULL_HANDLE);

        std::mutex mu;
        std::condition_variable cv;
        bool entered = false, used = false, release = false, cancelled = false;

        context.enqueue_fence_callback(sem, 1, [&] {
            {
                std::unique_lock lk(mu);
                entered = true;
                cv.notify_all();
                cv.wait(lk, [&] { return release; });
            }
            std::uint64_t value = 0;
            driver::real_get_counter(d.device, sem, &value);
            std::lock_guard lk(mu);
            used = true;
            cv.notify_all();
        });

        EXPECT(fence.signal(1));
        EXPECT(wait_flag(mu, cv, entered));

        std::thread cancel_thread([&] {
            context.cancel_fence_waits(sem);
            std::lock_guard lk(mu);
            cancelled = true;
            cv.notify_all();
        });
        {
            std::unique_lock lk(mu);
            EXPECT(!cv.wait_for(lk, std::chrono::milliseconds(150), [&] { return cancelled; }));
        }
        {
            std::lock_guard lk(mu);
            release = true;
        }
        cv.notify_all();
        EXPECT(wait_flag(mu, cv, cancelled));
        EXPECT(used);
        cancel_thread.join();
        std::printf("fence dispatcher: cancel waits for an in-flight callback ok\n");
    }

    // 6. A re-entrant cancel() for a later callback in the same ready batch must skip that callback instead of
    //    returning while it is still going to run with a semaphore the caller has already destroyed.
    {
        driver::in_wait_semaphores.store(false, std::memory_order_release);
        driver::wait_sem_count.store(0, std::memory_order_release);
        VulkanContext context(driver::get_device, d.device, true);
        EXPECT(context.loaded());
        VulkanFence fence_a(context, 0);
        VulkanFence fence_b(context, 0);
        const VkSemaphore sem_a = fence_a.semaphore();
        const VkSemaphore sem_b = fence_b.semaphore();
        EXPECT(sem_a != VK_NULL_HANDLE && sem_b != VK_NULL_HANDLE && sem_a != sem_b);

        std::mutex mu;
        std::condition_variable cv;
        bool a_done = false, cancel_returned = false, b_after_cancel = false;

        context.enqueue_fence_callback(sem_a, 1, [&] {
            context.cancel_fence_waits(sem_b);
            std::lock_guard lk(mu);
            // B may legitimately run inside cancel() when it was still pending; it must not run after this point.
            cancel_returned = true;
            a_done = true;
            cv.notify_all();
        });
        EXPECT(wait_until_wait_sem_count(2));  // wakeup + A
        context.enqueue_fence_callback(sem_b, 1, [&] {
            std::lock_guard lk(mu);
            if (cancel_returned) b_after_cancel = true;
            cv.notify_all();
        });
        EXPECT(wait_until_wait_sem_count(3));  // wakeup + A + B: one ready batch

        EXPECT(fence_a.signal(1));
        EXPECT(fence_b.signal(1));
        EXPECT(wait_flag(mu, cv, a_done));
        // Give the dispatcher time to walk the rest of the batch.
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        EXPECT(!b_after_cancel);

        context.cancel_fence_waits(sem_a);
        context.cancel_fence_waits(sem_b);
        std::printf("fence dispatcher: re-entrant cancel skips later callback in the same batch ok\n");
    }

    check_logical_fence_rewind(d.device);
    EXPECT(d.validation_errors() == 0);
    std::printf("vk_fence_dispatcher_sync_test: ok\n");
    return 0;
}
