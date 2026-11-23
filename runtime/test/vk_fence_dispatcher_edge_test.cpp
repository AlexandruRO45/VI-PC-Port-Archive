// FenceWaitDispatcher edge-case regressions (runtime/src/vulkan_backend.h):
//   1. wakeup_sem_ creation failure (VK_ERROR_OUT_OF_HOST_MEMORY / VK_ERROR_DEVICE_LOST):
//      on a healthy device (OOM), waits still start the worker and wait on the real target
//      semaphores with a short fallback slice (zero early hits before the target value,
//      immediate completion when reached, bounded cancel/stop, zero VK_NULL_HANDLE waits);
//      on VK_ERROR_DEVICE_LOST, device_lost_ is set immediately and waits complete promptly.
//   2. cancel(sem) on unstarted entries in a live ready batch (both cross-thread and re-entrant):
//      claims and dispatches callbacks and events synchronously before cancel() returns (so
//      FenceWaiter / held-signal bookkeeping is never dropped), while preserving dea6c38354's
//      guarantee that no callback runs after cancel() returns.
//   3. Concurrent cancel(sem) calls synchronize with each other's in-flight callbacks via
//      dispatching_, so a racing ~Fence() never returns while another cancel() still uses sem.
//   4. stop() drains remaining pending_ event/callback waits and concurrent stop() calls neither
//      double-join nor return while run() is still active.
//   5. Non-DEVICE_LOST errors (VK_ERROR_OUT_OF_HOST_MEMORY) from vkWaitSemaphores and
//      vkGetSemaphoreCounterValue back off cleanly instead of spinning at 100% CPU, keep
//      device_lost_ false, and recover once the error clears.
//   6. Claimed-but-waiting entries in cancel(S) remain visible in dispatching_ across the
//      epoch_cv_ wait, so a re-entrant cancel(S) from an active callback A for S claims and
//      completes B synchronously before returning rather than letting the other thread run B
//      after the nested cancel returns.
//   7. Concurrent stop() waits for another stop()'s entire pending_ drain (not just worker exit),
//      while a re-entrant stop() inside a drained callback still returns without deadlocking.
// Skips with exit code 77 when no Vulkan 1.3 device is available; fails on DeviceStatus::failed. No game data.
#include "vk_test_device.h"
#include "../src/vulkan_backend.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string_view>
#include <thread>

#define EXPECT(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); std::_Exit(1); } \
} while (0)

namespace {

namespace driver {
PFN_vkGetDeviceProcAddr real_get_device = nullptr;
PFN_vkCreateSemaphore real_create_sem = nullptr;
PFN_vkDestroySemaphore real_destroy_sem = nullptr;
PFN_vkWaitSemaphores real_wait_semaphores = nullptr;
PFN_vkGetSemaphoreCounterValue real_get_counter = nullptr;

std::atomic<VkResult> fail_next_create_sem{VK_SUCCESS};
std::atomic<bool> destroyed_null_or_garbage{false};
std::atomic<bool> waited_on_null_or_empty{false};
std::atomic<bool> in_wait_semaphores{false};
std::atomic<std::uint32_t> wait_sem_count{0};
std::atomic<VkResult> fail_wait_result{VK_SUCCESS};
std::atomic<std::uint32_t> wait_calls{0};
std::atomic<VkResult> fail_counter_result{VK_SUCCESS};
std::atomic<std::uint32_t> counter_calls{0};

constexpr std::uintptr_t kGarbageSem = 0xDEAD5E01u;

VKAPI_ATTR VkResult VKAPI_CALL create_semaphore(VkDevice device, const VkSemaphoreCreateInfo* info,
                                                const VkAllocationCallbacks* alloc, VkSemaphore* out) {
    const VkResult inject = fail_next_create_sem.exchange(VK_SUCCESS, std::memory_order_acq_rel);
    if (inject != VK_SUCCESS) {
        if (out) *out = reinterpret_cast<VkSemaphore>(kGarbageSem);
        return inject;
    }
    return real_create_sem(device, info, alloc, out);
}

VKAPI_ATTR void VKAPI_CALL destroy_semaphore(VkDevice device, VkSemaphore sem, const VkAllocationCallbacks* alloc) {
    if (sem == VK_NULL_HANDLE || reinterpret_cast<std::uintptr_t>(sem) == kGarbageSem) {
        destroyed_null_or_garbage.store(true, std::memory_order_release);
        return;
    }
    real_destroy_sem(device, sem, alloc);
}

VKAPI_ATTR VkResult VKAPI_CALL wait_semaphores(VkDevice device, const VkSemaphoreWaitInfo* info, std::uint64_t timeout) {
    wait_calls.fetch_add(1, std::memory_order_relaxed);
    if (!info || info->semaphoreCount == 0 || !info->pSemaphores) {
        waited_on_null_or_empty.store(true, std::memory_order_release);
    } else {
        for (std::uint32_t i = 0; i < info->semaphoreCount; ++i) {
            if (info->pSemaphores[i] == VK_NULL_HANDLE) waited_on_null_or_empty.store(true, std::memory_order_release);
        }
    }
    wait_sem_count.store(info ? info->semaphoreCount : 0u, std::memory_order_release);
    const VkResult inject = fail_wait_result.load(std::memory_order_acquire);
    if (inject != VK_SUCCESS) return inject;
    in_wait_semaphores.store(true, std::memory_order_release);
    const VkResult res = real_wait_semaphores(device, info, timeout);
    in_wait_semaphores.store(false, std::memory_order_release);
    return res;
}

VKAPI_ATTR VkResult VKAPI_CALL get_semaphore_counter_value(VkDevice device, VkSemaphore sem, std::uint64_t* value) {
    counter_calls.fetch_add(1, std::memory_order_relaxed);
    const VkResult inject = fail_counter_result.load(std::memory_order_acquire);
    if (inject != VK_SUCCESS) return inject;
    return real_get_counter(device, sem, value);
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL get_device(VkDevice device, const char* name) {
    const std::string_view n = name;
    if (n == "vkCreateSemaphore") return reinterpret_cast<PFN_vkVoidFunction>(create_semaphore);
    if (n == "vkDestroySemaphore") return reinterpret_cast<PFN_vkVoidFunction>(destroy_semaphore);
    if (n == "vkWaitSemaphores") return reinterpret_cast<PFN_vkVoidFunction>(wait_semaphores);
    if (n == "vkGetSemaphoreCounterValue") return reinterpret_cast<PFN_vkVoidFunction>(get_semaphore_counter_value);
    return real_get_device(device, name);
}

void install(PFN_vkGetDeviceProcAddr device_proc, VkDevice device) {
    real_get_device = device_proc;
    real_create_sem = reinterpret_cast<PFN_vkCreateSemaphore>(device_proc(device, "vkCreateSemaphore"));
    real_destroy_sem = reinterpret_cast<PFN_vkDestroySemaphore>(device_proc(device, "vkDestroySemaphore"));
    real_wait_semaphores = reinterpret_cast<PFN_vkWaitSemaphores>(device_proc(device, "vkWaitSemaphores"));
    real_get_counter =
        reinterpret_cast<PFN_vkGetSemaphoreCounterValue>(device_proc(device, "vkGetSemaphoreCounterValue"));
}

void reset_faults() {
    fail_next_create_sem.store(VK_SUCCESS, std::memory_order_release);
    in_wait_semaphores.store(false, std::memory_order_release);
    wait_sem_count.store(0, std::memory_order_release);
    fail_wait_result.store(VK_SUCCESS, std::memory_order_release);
    wait_calls.store(0, std::memory_order_release);
    fail_counter_result.store(VK_SUCCESS, std::memory_order_release);
    counter_calls.store(0, std::memory_order_release);
}
}  // namespace driver

struct EventCounter {
    std::mutex mu;
    std::condition_variable cv;
    int hits = 0;
};

void signal_event_counter(void* handle) {
    auto* c = static_cast<EventCounter*>(handle);
    {
        std::lock_guard lk(c->mu);
        ++c->hits;
    }
    c->cv.notify_all();
}

bool wait_flag(std::mutex& mu, std::condition_variable& cv, const bool& flag, int timeout_ms = 2000) {
    std::unique_lock lk(mu);
    return cv.wait_for(lk, std::chrono::milliseconds(timeout_ms), [&] { return flag; });
}

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
    std::setvbuf(stdout, nullptr, _IONBF, 0);

    vktest::Device d;
    const vktest::DeviceStatus status = d.initialize(false);
    if (status == vktest::DeviceStatus::failed) return 1;
    if (status == vktest::DeviceStatus::unavailable) {
        std::printf("SKIP: no suitable Vulkan 1.3 device\n");
        return 77;
    }
    driver::install(d.device_proc, d.device);
    driver::destroyed_null_or_garbage.store(false, std::memory_order_release);
    driver::waited_on_null_or_empty.store(false, std::memory_order_release);

    // 1. wakeup_sem_ creation failure: healthy OOM waits on target semaphores with a short fallback slice (never
    //    firing early before the target value, completing when signaled, and cancelling promptly); DEVICE_LOST sets
    //    device_lost_ immediately.
    {
        driver::reset_faults();
        driver::fail_next_create_sem.store(VK_ERROR_OUT_OF_HOST_MEMORY, std::memory_order_release);
        VulkanContext ctx(driver::get_device, d.device, true);
        EXPECT(ctx.loaded());
        EXPECT(!ctx.is_device_lost());

        VulkanFence fence(ctx, 0);
        EXPECT(fence.semaphore() != VK_NULL_HANDLE);

        EventCounter evt;
        std::mutex mu;
        std::condition_variable cv;
        int cb_hits = 0;
        ctx.set_signal_event_fn(signal_event_counter);
        ctx.enqueue_fence_wait(fence.semaphore(), 100, &evt);
        ctx.enqueue_fence_callback(fence.semaphore(), 100, [&] {
            {
                std::lock_guard lk(mu);
                ++cb_hits;
            }
            cv.notify_all();
        });

        // Must NOT fire immediately on a healthy device, must start the worker without wakeup_sem_
        // (2 real target entries, zero VK_NULL_HANDLE), and must still stay quiet after an intermediate signal of 99.
        {
            std::lock_guard lk(evt.mu);
            EXPECT(evt.hits == 0);
        }
        {
            std::lock_guard lk(mu);
            EXPECT(cb_hits == 0);
        }
        EXPECT(wait_until_wait_sem_count(2));
        EXPECT(fence.signal(99));
        {
            std::unique_lock lk(mu);
            EXPECT(!cv.wait_for(lk, std::chrono::milliseconds(50), [&] { return cb_hits > 0; }));
        }
        {
            std::lock_guard lk(evt.mu);
            EXPECT(evt.hits == 0);
        }
        EXPECT(!ctx.is_device_lost());

        // Reaching 100 must wake vkWaitSemaphores and complete both the event and callback.
        EXPECT(fence.signal(100));
        {
            std::unique_lock lk(evt.mu);
            EXPECT(evt.cv.wait_for(lk, std::chrono::milliseconds(2000), [&] { return evt.hits == 1; }));
        }
        {
            std::unique_lock lk(mu);
            EXPECT(cv.wait_for(lk, std::chrono::milliseconds(2000), [&] { return cb_hits == 1; }));
        }

        // Unsignaled wait must still cancel promptly via the short fallback wait slice.
        bool cancel_cb_hit = false;
        ctx.enqueue_fence_callback(fence.semaphore(), 200, [&] {
            std::lock_guard lk(mu);
            cancel_cb_hit = true;
            cv.notify_all();
        });
        EXPECT(wait_until_wait_sem_count(1));
        bool cancel_done = false;
        std::thread canceller([&] {
            ctx.cancel_fence_waits(fence.semaphore());
            std::lock_guard lk(mu);
            cancel_done = true;
            cv.notify_all();
        });
        EXPECT(wait_flag(mu, cv, cancel_done, 2000));
        canceller.join();
        EXPECT(cancel_cb_hit);
    }
    {
        driver::reset_faults();
        driver::fail_next_create_sem.store(VK_ERROR_DEVICE_LOST, std::memory_order_release);
        VulkanContext ctx(driver::get_device, d.device, true);
        EXPECT(ctx.loaded());
        EXPECT(ctx.is_device_lost());
        VulkanFence fence(ctx, 0);
        EXPECT(fence.semaphore() != VK_NULL_HANDLE);
        int lost_cb_hits = 0;
        ctx.enqueue_fence_callback(fence.semaphore(), 100, [&] { ++lost_cb_hits; });
        EXPECT(lost_cb_hits == 1);
        ctx.cancel_fence_waits(fence.semaphore());
    }
    EXPECT(!driver::destroyed_null_or_garbage.load(std::memory_order_acquire));
    EXPECT(!driver::waited_on_null_or_empty.load(std::memory_order_acquire));
    std::printf("fence dispatcher edge 1: wakeup_sem_ OOM fallback wait & DEVICE_LOST ok\n");

    // 2a. Cross-thread cancel(sem_b) on unstarted ready-batch entries completes callback + event synchronously
    //     before cancel() returns and never runs them again after cancel() returns.
    {
        driver::reset_faults();
        VulkanContext ctx(driver::get_device, d.device, true);
        EXPECT(ctx.loaded());
        ctx.set_signal_event_fn(signal_event_counter);

        VulkanFence fence_a(ctx, 0);
        VulkanFence fence_b(ctx, 0);
        VulkanFence fence_sentinel(ctx, 0);
        EXPECT(fence_a.semaphore() && fence_b.semaphore() && fence_sentinel.semaphore());

        std::mutex mu;
        std::condition_variable cv;
        bool a_entered = false, release_a = false, a_done = false;
        bool b_cancel_returned = false, b_after_cancel = false, sentinel_done = false;
        int b_cb_calls = 0;
        EventCounter b_evt;

        ctx.enqueue_fence_callback(fence_a.semaphore(), 1, [&] {
            {
                std::unique_lock lk(mu);
                a_entered = true;
                cv.notify_all();
                cv.wait(lk, [&] { return release_a; });
                a_done = true;
            }
            cv.notify_all();
        });
        EXPECT(wait_until_wait_sem_count(2));  // wakeup + A

        ctx.enqueue_fence_callback(fence_b.semaphore(), 1, [&] {
            std::lock_guard lk(mu);
            ++b_cb_calls;
            if (b_cancel_returned) b_after_cancel = true;
        });
        EXPECT(wait_until_wait_sem_count(3));  // wakeup + A + B_cb

        ctx.enqueue_fence_wait(fence_b.semaphore(), 1, &b_evt);
        EXPECT(wait_until_wait_sem_count(4));  // wakeup + A + B_cb + B_evt in one batch

        EXPECT(fence_a.signal(1));
        EXPECT(fence_b.signal(1));
        EXPECT(wait_flag(mu, cv, a_entered));

        // B_cb and B_evt are in the live ready batch behind blocked A.
        ctx.cancel_fence_waits(fence_b.semaphore());
        {
            std::lock_guard lk(mu);
            EXPECT(b_cb_calls == 1);
            b_cancel_returned = true;
        }
        {
            std::lock_guard lk(b_evt.mu);
            EXPECT(b_evt.hits == 1);
        }

        // Release A and run a sentinel wait to prove run() walked past the cancelled slots without re-running them.
        ctx.enqueue_fence_callback(fence_sentinel.semaphore(), 1, [&] {
            std::lock_guard lk(mu);
            sentinel_done = true;
            cv.notify_all();
        });
        {
            std::lock_guard lk(mu);
            release_a = true;
        }
        cv.notify_all();
        EXPECT(wait_flag(mu, cv, a_done));
        EXPECT(fence_sentinel.signal(1));
        EXPECT(wait_flag(mu, cv, sentinel_done));

        {
            std::lock_guard lk(mu);
            EXPECT(b_cb_calls == 1);
            EXPECT(!b_after_cancel);
        }
        {
            std::lock_guard lk(b_evt.mu);
            EXPECT(b_evt.hits == 1);
        }

        ctx.cancel_fence_waits(fence_a.semaphore());
        ctx.cancel_fence_waits(fence_b.semaphore());
        ctx.cancel_fence_waits(fence_sentinel.semaphore());
    }

    // 2b. Re-entrant cancel(sem_b) from A's callback in the same ready batch completes B before cancel() returns
    //     and never runs B after cancel() returns.
    {
        driver::reset_faults();
        VulkanContext ctx(driver::get_device, d.device, true);
        EXPECT(ctx.loaded());
        VulkanFence fence_a(ctx, 0);
        VulkanFence fence_b(ctx, 0);
        VulkanFence fence_sentinel(ctx, 0);
        EXPECT(fence_a.semaphore() && fence_b.semaphore() && fence_sentinel.semaphore());

        std::mutex mu;
        std::condition_variable cv;
        bool a_done = false, cancel_returned = false, b_after_cancel = false, sentinel_done = false;
        int b_calls = 0;

        ctx.enqueue_fence_callback(fence_a.semaphore(), 1, [&] {
            ctx.cancel_fence_waits(fence_b.semaphore());
            std::lock_guard lk(mu);
            EXPECT(b_calls == 1);
            cancel_returned = true;
            a_done = true;
            cv.notify_all();
        });
        EXPECT(wait_until_wait_sem_count(2));
        ctx.enqueue_fence_callback(fence_b.semaphore(), 1, [&] {
            std::lock_guard lk(mu);
            ++b_calls;
            if (cancel_returned) b_after_cancel = true;
        });
        EXPECT(wait_until_wait_sem_count(3));

        EXPECT(fence_a.signal(1));
        EXPECT(fence_b.signal(1));
        EXPECT(wait_flag(mu, cv, a_done));

        ctx.enqueue_fence_callback(fence_sentinel.semaphore(), 1, [&] {
            std::lock_guard lk(mu);
            sentinel_done = true;
            cv.notify_all();
        });
        EXPECT(fence_sentinel.signal(1));
        EXPECT(wait_flag(mu, cv, sentinel_done));

        {
            std::lock_guard lk(mu);
            EXPECT(b_calls == 1);
            EXPECT(!b_after_cancel);
        }
        ctx.cancel_fence_waits(fence_a.semaphore());
        ctx.cancel_fence_waits(fence_b.semaphore());
        ctx.cancel_fence_waits(fence_sentinel.semaphore());
    }
    std::printf("fence dispatcher edge 2: ready-batch cancel (cross-thread & re-entrant) ok\n");

    // 3. Concurrent cancel(sem) waits for another thread's cancel(sem) in-flight callback to finish.
    {
        driver::reset_faults();
        VulkanContext ctx(driver::get_device, d.device, true);
        EXPECT(ctx.loaded());
        VulkanFence fence_c(ctx, 0);
        const VkSemaphore sem_c = fence_c.semaphore();
        EXPECT(sem_c != VK_NULL_HANDLE);

        std::mutex mu;
        std::condition_variable cv;
        bool c_entered = false, release_c = false, c_used_sem = false;
        bool t1_done = false, t2_done = false;

        ctx.enqueue_fence_callback(sem_c, 99, [&] {
            {
                std::unique_lock lk(mu);
                c_entered = true;
                cv.notify_all();
                cv.wait(lk, [&] { return release_c; });
            }
            std::uint64_t val = 0;
            driver::real_get_counter(d.device, sem_c, &val);
            std::lock_guard lk(mu);
            c_used_sem = true;
            cv.notify_all();
        });
        EXPECT(wait_until_wait_sem_count(2));

        std::thread t1([&] {
            ctx.cancel_fence_waits(sem_c);
            std::lock_guard lk(mu);
            t1_done = true;
            cv.notify_all();
        });
        EXPECT(wait_flag(mu, cv, c_entered));

        std::thread t2([&] {
            ctx.cancel_fence_waits(sem_c);
            std::lock_guard lk(mu);
            t2_done = true;
            cv.notify_all();
        });
        {
            std::unique_lock lk(mu);
            EXPECT(!cv.wait_for(lk, std::chrono::milliseconds(150), [&] { return t2_done; }));
        }
        {
            std::lock_guard lk(mu);
            release_c = true;
        }
        cv.notify_all();
        EXPECT(wait_flag(mu, cv, t1_done));
        EXPECT(wait_flag(mu, cv, t2_done));
        t1.join();
        t2.join();
        EXPECT(c_used_sem);
    }
    std::printf("fence dispatcher edge 3: concurrent cancel() waits for in-flight cancel callback ok\n");

    // 4. stop() drains pending_ entries and concurrent stop() waits for worker exit without double-join.
    {
        driver::reset_faults();
        VulkanContext ctx(driver::get_device, d.device, true);
        EXPECT(ctx.loaded());
        VulkanFence fence_pending(ctx, 0);
        VulkanFence fence_busy(ctx, 0);
        EXPECT(fence_pending.semaphore() && fence_busy.semaphore());

        std::atomic<bool> lost{false};
        FenceWaitDispatcher disp(d.device, ctx.vk, lost);
        EventCounter evt;
        disp.set_signal_event_fn(signal_event_counter);

        std::mutex mu;
        std::condition_variable cv;
        bool busy_entered = false, release_busy = false;
        bool stop1_done = false, stop2_done = false;
        int pending_cb_hits = 0;

        disp.enqueue(fence_pending.semaphore(), 100, &evt);
        disp.enqueue_callback(fence_pending.semaphore(), 100, [&] {
            std::lock_guard lk(mu);
            ++pending_cb_hits;
        });
        disp.enqueue_callback(fence_busy.semaphore(), 1, [&] {
            std::unique_lock lk(mu);
            busy_entered = true;
            cv.notify_all();
            cv.wait(lk, [&] { return release_busy; });
        });
        EXPECT(wait_until_wait_sem_count(4));
        EXPECT(fence_busy.signal(1));
        EXPECT(wait_flag(mu, cv, busy_entered));

        // Two concurrent stop() calls while worker is blocked in fence_busy's callback:
        // neither may return until the worker finishes, and pending_ entries must be drained.
        std::thread s1([&] {
            disp.stop();
            std::lock_guard lk(mu);
            stop1_done = true;
            cv.notify_all();
        });
        std::thread s2([&] {
            disp.stop();
            std::lock_guard lk(mu);
            stop2_done = true;
            cv.notify_all();
        });
        {
            std::unique_lock lk(mu);
            EXPECT(!cv.wait_for(lk, std::chrono::milliseconds(150), [&] { return stop1_done || stop2_done; }));
            release_busy = true;
        }
        cv.notify_all();
        EXPECT(wait_flag(mu, cv, stop1_done));
        EXPECT(wait_flag(mu, cv, stop2_done));
        s1.join();
        s2.join();

        {
            std::lock_guard lk(evt.mu);
            EXPECT(evt.hits == 1);
        }
        {
            std::lock_guard lk(mu);
            EXPECT(pending_cb_hits == 1);
        }
    }
    std::printf("fence dispatcher edge 4: stop() drains pending_ and synchronizes concurrent stop() ok\n");

    // 5. Non-DEVICE_LOST OOM errors in vkWaitSemaphores and vkGetSemaphoreCounterValue back off instead of
    //    spinning at 100% CPU, keep device_lost_ false, and recover cleanly once cleared.
    {
        driver::reset_faults();
        VulkanContext ctx(driver::get_device, d.device, true);
        EXPECT(ctx.loaded());
        VulkanFence f_wait(ctx, 0);
        VulkanFence f_cnt(ctx, 0);
        EXPECT(f_wait.semaphore() && f_cnt.semaphore());

        std::mutex mu;
        std::condition_variable cv;
        bool wait_cb_done = false, cnt_cb_done = false;

        // 5a: vkWaitSemaphores returns VK_ERROR_OUT_OF_HOST_MEMORY transiently.
        driver::fail_wait_result.store(VK_ERROR_OUT_OF_HOST_MEMORY, std::memory_order_release);
        ctx.enqueue_fence_callback(f_wait.semaphore(), 1, [&] {
            std::lock_guard lk(mu);
            wait_cb_done = true;
            cv.notify_all();
        });
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        const std::uint32_t oom_wait_calls = driver::wait_calls.load(std::memory_order_acquire);
        driver::fail_wait_result.store(VK_SUCCESS, std::memory_order_release);
        EXPECT(oom_wait_calls >= 1 && oom_wait_calls < 200);
        EXPECT(!ctx.is_device_lost());
        EXPECT(f_wait.signal(1));
        EXPECT(wait_flag(mu, cv, wait_cb_done));

        // 5b: semaphore is already signaled (vkWaitSemaphores returns VK_SUCCESS immediately), while
        //     vkGetSemaphoreCounterValue transiently returns VK_ERROR_OUT_OF_HOST_MEMORY.
        EXPECT(f_cnt.signal(1));
        driver::counter_calls.store(0, std::memory_order_release);
        driver::fail_counter_result.store(VK_ERROR_OUT_OF_HOST_MEMORY, std::memory_order_release);
        ctx.enqueue_fence_callback(f_cnt.semaphore(), 1, [&] {
            std::lock_guard lk(mu);
            cnt_cb_done = true;
            cv.notify_all();
        });
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        const std::uint32_t oom_cnt_calls = driver::counter_calls.load(std::memory_order_acquire);
        driver::fail_counter_result.store(VK_SUCCESS, std::memory_order_release);
        EXPECT(oom_cnt_calls >= 1 && oom_cnt_calls < 200);
        EXPECT(!ctx.is_device_lost());
        EXPECT(wait_flag(mu, cv, cnt_cb_done));

        ctx.cancel_fence_waits(f_wait.semaphore());
        ctx.cancel_fence_waits(f_cnt.semaphore());
    }
    std::printf("fence dispatcher edge 5: transient OOM backoff and recovery ok\n");

    // 6. Claimed entries stay visible across cancel()'s wait: worker callback A for S enqueues B for S and blocks;
    //    another thread calls cancel(S) (claiming B and waiting on A); A then re-entrantly calls cancel(S).
    //    B must run strictly before the nested cancel(S) returns, never after.
    for (int iter = 0; iter < 5; ++iter) {
        driver::reset_faults();
        VulkanContext ctx(driver::get_device, d.device, true);
        EXPECT(ctx.loaded());
        VulkanFence fence_s(ctx, 0);
        const VkSemaphore sem_s = fence_s.semaphore();
        EXPECT(sem_s != VK_NULL_HANDLE);

        std::mutex mu;
        std::condition_variable cv;
        bool b_enqueued = false, release_a = false, a_done = false;
        bool t2_started = false, t2_done = false;
        bool nested_cancel_returned = false, b_after_nested_cancel = false;
        int b_calls = 0;

        ctx.enqueue_fence_callback(sem_s, 1, [&] {
            ctx.enqueue_fence_callback(sem_s, 99, [&] {
                std::lock_guard lk(mu);
                ++b_calls;
                if (nested_cancel_returned) b_after_nested_cancel = true;
            });
            {
                std::unique_lock lk(mu);
                b_enqueued = true;
                cv.notify_all();
                cv.wait(lk, [&] { return release_a; });
            }
            ctx.cancel_fence_waits(sem_s);
            {
                std::lock_guard lk(mu);
                nested_cancel_returned = true;
                a_done = true;
            }
            cv.notify_all();
        });
        EXPECT(wait_until_wait_sem_count(2));
        EXPECT(fence_s.signal(1));
        EXPECT(wait_flag(mu, cv, b_enqueued));

        std::thread t2([&] {
            {
                std::lock_guard lk(mu);
                t2_started = true;
            }
            cv.notify_all();
            ctx.cancel_fence_waits(sem_s);
            {
                std::lock_guard lk(mu);
                t2_done = true;
            }
            cv.notify_all();
        });
        EXPECT(wait_flag(mu, cv, t2_started));
        // Give t2 time to enter cancel(sem_s), claim B, and park on epoch_cv_ waiting for active A.
        {
            std::unique_lock lk(mu);
            EXPECT(!cv.wait_for(lk, std::chrono::milliseconds(40), [&] { return t2_done; }));
            release_a = true;
        }
        cv.notify_all();
        EXPECT(wait_flag(mu, cv, a_done));
        EXPECT(wait_flag(mu, cv, t2_done));
        t2.join();

        {
            std::lock_guard lk(mu);
            EXPECT(b_calls == 1);
            EXPECT(!b_after_nested_cancel);
        }
        ctx.cancel_fence_waits(sem_s);
    }
    std::printf("fence dispatcher edge 6: re-entrant cancel claims waiting cancel()'s entries (5/5) ok\n");

    // 7. Concurrent stop() must wait until another stop() finishes draining pending_ callbacks (after worker exit),
    //    while a re-entrant stop() inside the drained callback returns without deadlocking.
    {
        driver::reset_faults();
        VulkanContext ctx(driver::get_device, d.device, true);
        EXPECT(ctx.loaded());
        VulkanFence fence_p(ctx, 0);
        EXPECT(fence_p.semaphore() != VK_NULL_HANDLE);

        std::atomic<bool> lost{false};
        FenceWaitDispatcher disp(d.device, ctx.vk, lost);

        std::mutex mu;
        std::condition_variable cv;
        bool p_entered = false, release_p = false, reentrant_stop_ok = false, p_done = false;
        bool s1_done = false, s2_done = false;

        disp.enqueue_callback(fence_p.semaphore(), 100, [&] {
            {
                std::unique_lock lk(mu);
                p_entered = true;
                cv.notify_all();
                cv.wait(lk, [&] { return release_p; });
            }
            // Re-entrant stop() from inside the drained callback must not deadlock.
            disp.stop();
            {
                std::lock_guard lk(mu);
                reentrant_stop_ok = true;
                p_done = true;
            }
            cv.notify_all();
        });
        EXPECT(wait_until_wait_sem_count(2));

        // s1 stops the worker (worker exits), then begins draining pending_ and blocks inside P.
        std::thread s1([&] {
            disp.stop();
            std::lock_guard lk(mu);
            s1_done = true;
            cv.notify_all();
        });
        EXPECT(wait_flag(mu, cv, p_entered));

        // Now worker_alive_ is false and pending_ is empty, while s1 is still executing P.
        // A second concurrent stop() must wait until s1 finishes draining P.
        std::thread s2([&] {
            disp.stop();
            std::lock_guard lk(mu);
            s2_done = true;
            cv.notify_all();
        });
        {
            std::unique_lock lk(mu);
            EXPECT(!cv.wait_for(lk, std::chrono::milliseconds(150), [&] { return s2_done; }));
            release_p = true;
        }
        cv.notify_all();
        EXPECT(wait_flag(mu, cv, p_done));
        EXPECT(wait_flag(mu, cv, s1_done));
        EXPECT(wait_flag(mu, cv, s2_done));
        s1.join();
        s2.join();
        EXPECT(reentrant_stop_ok);
    }
    std::printf("fence dispatcher edge 7: concurrent stop() waits for drain completion & re-entrant stop ok\n");

    EXPECT(!driver::destroyed_null_or_garbage.load(std::memory_order_acquire));
    EXPECT(!driver::waited_on_null_or_empty.load(std::memory_order_acquire));
    EXPECT(d.validation_errors() == 0);
    std::printf("vk_fence_dispatcher_edge_test: ok\n");
    return 0;
}
