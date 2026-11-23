// VulkanQueue held-operation wait forwarding and device-lost signal/marker/pin-drop tests
// (runtime/src/vulkan_backend.h):
//   1. When an earlier held batch fails, its unconsumed waits must be prepended to the next held
//      operation in queue order rather than stashed into `later_waits` behind subsequent held work
//      (so a later batch and Signal still wait on the dependency on the GPU and in vkQueueSubmit).
//   2. A failed held batch followed by a held debug marker must preserve the unconsumed waits onto
//      the following held Signal.
//   3. Chained waits across a failed held batch and a still-pending held batch must merge cleanly
//      without blocking or leaking later queued unheld waits.
//   4. VulkanQueue::signal() under must_hold() && device_lost must return false without queuing a
//      held operation or pinning; marker() must not queue under device_lost; and release_ready()
//      must destroy dropped pins outside state_mutex_.
// Skips with exit code 77 when no Vulkan 1.3 device is available. No game data.
#include "vk_test_device.h"
#include "../src/vulkan_backend.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <memory>
#include <mutex>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#define EXPECT(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return 1; } \
} while (0)

namespace {

struct RecordedSubmit {
    VkQueue queue = VK_NULL_HANDLE;
    std::vector<std::pair<VkSemaphore, std::uint64_t>> waits;
    std::vector<std::pair<VkSemaphore, std::uint64_t>> signals;
};

namespace driver {
PFN_vkGetDeviceProcAddr real_get_device = nullptr;
PFN_vkQueueSubmit real_submit = nullptr;

std::mutex log_mutex;
std::vector<RecordedSubmit> submit_log;

const VkTimelineSemaphoreSubmitInfo* find_timeline(const void* pnext) {
    for (const auto* cur = static_cast<const VkBaseInStructure*>(pnext); cur; cur = cur->pNext) {
        if (cur->sType == VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO)
            return reinterpret_cast<const VkTimelineSemaphoreSubmitInfo*>(cur);
    }
    return nullptr;
}

VKAPI_ATTR VkResult VKAPI_CALL queue_submit(VkQueue queue, std::uint32_t count, const VkSubmitInfo* submits,
                                            VkFence fence) {
    const VkResult res = real_submit(queue, count, submits, fence);
    if (res == VK_SUCCESS && submits) {
        std::lock_guard lock(log_mutex);
        for (std::uint32_t s = 0; s < count; ++s) {
            RecordedSubmit rec;
            rec.queue = queue;
            const auto* tl = find_timeline(submits[s].pNext);
            for (std::uint32_t i = 0; i < submits[s].waitSemaphoreCount; ++i) {
                const VkSemaphore sem = submits[s].pWaitSemaphores ? submits[s].pWaitSemaphores[i] : VK_NULL_HANDLE;
                const std::uint64_t val =
                    (tl && i < tl->waitSemaphoreValueCount && tl->pWaitSemaphoreValues) ? tl->pWaitSemaphoreValues[i]
                                                                                         : 0;
                rec.waits.push_back({sem, val});
            }
            for (std::uint32_t i = 0; i < submits[s].signalSemaphoreCount; ++i) {
                const VkSemaphore sem = submits[s].pSignalSemaphores ? submits[s].pSignalSemaphores[i] : VK_NULL_HANDLE;
                const std::uint64_t val =
                    (tl && i < tl->signalSemaphoreValueCount && tl->pSignalSemaphoreValues)
                        ? tl->pSignalSemaphoreValues[i]
                        : 0;
                rec.signals.push_back({sem, val});
            }
            submit_log.push_back(std::move(rec));
        }
    }
    return res;
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL get_device(VkDevice device, const char* name) {
    if (std::string_view(name) == "vkQueueSubmit") return reinterpret_cast<PFN_vkVoidFunction>(queue_submit);
    return real_get_device(device, name);
}

void install(PFN_vkGetDeviceProcAddr proc, VkDevice device) {
    real_get_device = proc;
    real_submit = reinterpret_cast<PFN_vkQueueSubmit>(proc(device, "vkQueueSubmit"));
}

void clear_log() {
    std::lock_guard lock(log_mutex);
    submit_log.clear();
}

std::vector<RecordedSubmit> take_log() {
    std::lock_guard lock(log_mutex);
    return submit_log;
}
}  // namespace driver

bool has_pair(const std::vector<std::pair<VkSemaphore, std::uint64_t>>& list, VkSemaphore sem, std::uint64_t val) {
    for (const auto& [s, v] : list)
        if (s == sem && v == val) return true;
    return false;
}

bool has_sem(const std::vector<std::pair<VkSemaphore, std::uint64_t>>& list, VkSemaphore sem) {
    for (const auto& [s, v] : list)
        if (s == sem) return true;
    return false;
}

struct ScopeExit {
    std::function<void()> fn;
    ~ScopeExit() {
        if (fn) fn();
    }
};

std::atomic<unsigned> g_queue_label_calls{0};
VKAPI_ATTR void VKAPI_CALL noop_cmd_label(VkCommandBuffer, const VkDebugUtilsLabelEXT*) {}
VKAPI_ATTR void VKAPI_CALL noop_cmd_end(VkCommandBuffer) {}
VKAPI_ATTR void VKAPI_CALL count_queue_label(VkQueue, const VkDebugUtilsLabelEXT*) {
    g_queue_label_calls.fetch_add(1, std::memory_order_relaxed);
}
VKAPI_ATTR void VKAPI_CALL count_queue_end(VkQueue) {
    g_queue_label_calls.fetch_add(1, std::memory_order_relaxed);
}

}  // namespace

int main() {
    using namespace leonida;
    std::setvbuf(stdout, nullptr, _IONBF, 0);

    vktest::Device d;
    d.queue_count_wanted = 2;
    d.all_queue_types = true;
    if (!d.create(false)) {
        std::printf("SKIP: no suitable Vulkan 1.3 device\n");
        return 77;
    }
    driver::install(d.device_proc, d.device);
    EXPECT(driver::real_submit != nullptr);

    VkQueue qA = d.queue, qB = d.queue;
    std::uint32_t famA = d.family, famB = d.family;
    VulkanQueue::QueueKind kindB = VulkanQueue::QueueKind::direct;
    bool distinct_queues = false;
    if (d.queues.size() >= 2 && d.queues[1] != VK_NULL_HANDLE && d.queues[1] != d.queue) {
        qB = d.queues[1];
        distinct_queues = true;
    } else if (d.compute_queue != VK_NULL_HANDLE && d.compute_queue != d.queue) {
        qB = d.compute_queue;
        famB = d.compute_family;
        kindB = VulkanQueue::QueueKind::compute;
        distinct_queues = true;
    } else if (d.copy_queue != VK_NULL_HANDLE && d.copy_queue != d.queue) {
        qB = d.copy_queue;
        famB = d.copy_family;
        kindB = VulkanQueue::QueueKind::copy;
        distinct_queues = true;
    }
    std::printf("distinct_queues=%d (famA=%u famB=%u)\n", int(distinct_queues), famA, famB);

    void* src_ptr = nullptr;
    void* shared_ptr = nullptr;
    void* readback_ptr = nullptr;
    const VkBuffer src_buf = d.buffer(256, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, &src_ptr);
    const VkBuffer shared_buf =
        d.buffer(256, VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, &shared_ptr);
    const VkBuffer readback_buf = d.buffer(256, VK_BUFFER_USAGE_TRANSFER_DST_BIT, &readback_ptr);
    EXPECT(src_buf && shared_buf && readback_buf && src_ptr && shared_ptr && readback_ptr);

    std::uint32_t expected_val = 0xCAFEBABEu;
    std::memcpy(src_ptr, &expected_val, sizeof(expected_val));

    // Shared command lists used across cases 1-3.
    CommandList invalid_list;
    EXPECT(invalid_list.copy_buffer({ResourceId(999999), 0, ResourceId(1), 0, 4}));
    EXPECT(invalid_list.close());
    const CommandList* p_invalid = &invalid_list;

    CommandList copy_list;
    EXPECT(copy_list.copy_buffer({ResourceId(3), 0, ResourceId(2), 0, 4}));
    EXPECT(copy_list.close());
    const CommandList* p_copy = &copy_list;

    CommandList fill_list;
    EXPECT(fill_list.copy_buffer({ResourceId(2), 0, ResourceId(1), 0, 4}));
    EXPECT(fill_list.close());
    const CommandList* p_fill = &fill_list;

    // 1. Failed held batch -> valid held batch -> held Signal, with an unconsumed Wait queued after held work.
    {
        std::memset(shared_ptr, 0, 256);
        std::memset(readback_ptr, 0, 256);

        VulkanContext context(driver::get_device, d.device, true);
        EXPECT(context.loaded());
        ResourceBinding rb_src{};
        rb_src.buffer = src_buf;
        rb_src.size = 256;
        ResourceBinding rb_shared{};
        rb_shared.buffer = shared_buf;
        rb_shared.size = 256;
        rb_shared.host_readable = true;
        ResourceBinding rb_readback{};
        rb_readback.buffer = readback_buf;
        rb_readback.size = 256;
        rb_readback.host_readable = true;
        context.add_resource(1, rb_src);
        context.add_resource(2, rb_shared);
        context.add_resource(3, rb_readback);

        VulkanQueue queueA(context, qA, famA, VulkanQueue::QueueKind::direct);
        VulkanQueue queueB(context, qB, famB, kindB);
        EXPECT(queueA.ok() && queueB.ok());

        VulkanFence F1(context, 0), F2(context, 0), F_later(context, 0), gate(context, 0);
        EXPECT(F1.semaphore() && F2.semaphore() && F_later.semaphore() && gate.semaphore());

        ScopeExit cleanup;
        if (distinct_queues) {
            cleanup.fn = [&] {
                gate.signal(1);
                queueA.idle();
                queueB.idle();
            };
        }

        queueB.wait(F1.semaphore(), 1);
        EXPECT(queueB.execute({&p_invalid, 1}) == ReplayResult::success);
        EXPECT(queueB.execute({&p_copy, 1}) == ReplayResult::success);
        EXPECT(queueB.signal(F2.semaphore(), 1));
        queueB.wait(F_later.semaphore(), 1);  // stays in waits_ behind held_

        driver::clear_log();
        if (distinct_queues) {
            // Let queueA submit work gated on `gate` without holding queueA on the CPU, then Signal(F1, 1)
            // immediately on the GPU behind fill_list, releasing queueB's held work while gate == 0 on GPU.
            context.holds().signaled(gate.semaphore(), 1);
            queueA.wait(gate.semaphore(), 1);
            EXPECT(queueA.execute({&p_fill, 1}) == ReplayResult::success);
            EXPECT(queueA.signal(F1.semaphore(), 1));

            EXPECT(queueB.take_held_failure() == ReplayResult::unresolved_resource);
            const auto logs = driver::take_log();
            std::vector<RecordedSubmit> qB_logs;
            for (const auto& s : logs)
                if (s.queue == qB) qB_logs.push_back(s);
            EXPECT(qB_logs.size() >= 2);
            EXPECT(has_pair(qB_logs[0].waits, F1.semaphore(), 1));
            EXPECT(!has_sem(qB_logs[0].waits, F_later.semaphore()));
            EXPECT(!has_sem(qB_logs[1].waits, F_later.semaphore()));

            // GPU must still be gated while gate == 0.
            EXPECT(!F2.wait(1, 50'000'000ull));
            EXPECT(gate.signal(1));
            EXPECT(F2.wait(1, 5'000'000'000ull));
            cleanup.fn = {};
            EXPECT(queueA.idle() && queueB.idle());
        } else {
            EXPECT(queueA.execute({&p_fill, 1}) == ReplayResult::success);
            EXPECT(queueA.idle());
            driver::clear_log();
            EXPECT(F1.signal(1));

            EXPECT(queueB.take_held_failure() == ReplayResult::unresolved_resource);
            const auto logs = driver::take_log();
            std::vector<RecordedSubmit> qB_logs;
            for (const auto& s : logs)
                if (s.queue == qB) qB_logs.push_back(s);
            EXPECT(qB_logs.size() >= 2);
            EXPECT(has_pair(qB_logs[0].waits, F1.semaphore(), 1));
            EXPECT(!has_sem(qB_logs[0].waits, F_later.semaphore()));
            EXPECT(!has_sem(qB_logs[1].waits, F_later.semaphore()));
            EXPECT(F2.wait(1, 5'000'000'000ull));
            EXPECT(queueB.idle());
        }

        std::uint32_t got = 0;
        std::memcpy(&got, readback_ptr, sizeof(got));
        EXPECT(got == expected_val);
        EXPECT(F_later.signal(1));
        std::printf("case 1 (failed held batch -> valid held batch + signal): ok\n");
    }

    // 2. Failed held batch -> held marker -> held Signal(F2, 1) with unconsumed wait F1 preserved across marker.
    {
        VulkanContext context(driver::get_device, d.device, true);
        EXPECT(context.loaded());
        VulkanQueue queueB(context, qB, famB, kindB);
        EXPECT(queueB.ok());
        VulkanFence F1(context, 0), F2(context, 0);
        EXPECT(F1.semaphore() && F2.semaphore());

        queueB.wait(F1.semaphore(), 1);
        EXPECT(queueB.execute({&p_invalid, 1}) == ReplayResult::success);
        queueB.marker(DebugMarker::Kind::set, "between");
        EXPECT(queueB.signal(F2.semaphore(), 1));

        driver::clear_log();
        EXPECT(F1.signal(1));
        EXPECT(queueB.take_held_failure() == ReplayResult::unresolved_resource);

        const auto logs = driver::take_log();
        bool found_sig = false;
        for (const auto& s : logs) {
            if (s.queue == qB && has_pair(s.signals, F2.semaphore(), 1)) {
                EXPECT(has_pair(s.waits, F1.semaphore(), 1));
                found_sig = true;
            }
        }
        EXPECT(found_sig);
        EXPECT(F2.wait(1, 5'000'000'000ull));
        EXPECT(queueB.idle());
        std::printf("case 2 (failed held batch -> marker -> signal preserves waits): ok\n");
    }

    // 3. Chained waits across failed held batch (F1) -> second held batch (F3) -> later queued wait (F4).
    {
        std::memset(shared_ptr, 0, 256);
        std::memset(readback_ptr, 0, 256);
        VulkanContext context(driver::get_device, d.device, true);
        EXPECT(context.loaded());
        ResourceBinding rb_shared{};
        rb_shared.buffer = shared_buf;
        rb_shared.size = 256;
        ResourceBinding rb_readback{};
        rb_readback.buffer = readback_buf;
        rb_readback.size = 256;
        rb_readback.host_readable = true;
        context.add_resource(2, rb_shared);
        context.add_resource(3, rb_readback);

        VulkanQueue queueB(context, qB, famB, kindB);
        EXPECT(queueB.ok());
        VulkanFence F1(context, 0), F3(context, 0), F4(context, 0);
        EXPECT(F1.semaphore() && F3.semaphore() && F4.semaphore());

        queueB.wait(F1.semaphore(), 1);
        EXPECT(queueB.execute({&p_invalid, 1}) == ReplayResult::success);
        queueB.wait(F3.semaphore(), 1);
        EXPECT(queueB.execute({&p_copy, 1}) == ReplayResult::success);
        queueB.wait(F4.semaphore(), 1);

        driver::clear_log();
        EXPECT(F1.signal(1));  // F3 is still unsignaled, so copy_list must remain held!
        EXPECT(queueB.take_held_failure() == ReplayResult::unresolved_resource);
        {
            const auto mid_logs = driver::take_log();
            for (const auto& s : mid_logs) EXPECT(s.queue != qB);
        }
        EXPECT(context.holds().any_held());

        EXPECT(F3.signal(1));
        EXPECT(!context.holds().any_held());
        const auto end_logs = driver::take_log();
        std::size_t qB_submits = 0;
        for (const auto& s : end_logs) {
            if (s.queue != qB) continue;
            ++qB_submits;
            EXPECT(has_pair(s.waits, F1.semaphore(), 1));
            EXPECT(has_pair(s.waits, F3.semaphore(), 1));
            EXPECT(!has_sem(s.waits, F4.semaphore()));
        }
        EXPECT(qB_submits == 1);
        EXPECT(queueB.idle());
        EXPECT(F4.signal(1));
        std::printf("case 3 (chained waits F1 + F3 merged without leaking F4): ok\n");
    }

    // 4. Device-lost signal() under must_hold(), marker() under device_lost, and release_ready() pin drop
    //    outside state_mutex_.
    {
        VulkanContext context(driver::get_device, d.device, true);
        EXPECT(context.loaded());
        ResourceBinding rb_shared{};
        rb_shared.buffer = shared_buf;
        rb_shared.size = 256;
        ResourceBinding rb_readback{};
        rb_readback.buffer = readback_buf;
        rb_readback.size = 256;
        rb_readback.host_readable = true;
        context.add_resource(2, rb_shared);
        context.add_resource(3, rb_readback);

        VulkanFence F1(context, 0), F2(context, 0), F3(context, 0);
        EXPECT(F1.semaphore() && F2.semaphore() && F3.semaphore());

        // 4a & 4b: signal() under must_hold() && is_device_lost() returns false and does not call pin() or hold;
        //          marker() under is_device_lost() neither invokes queue debug-label callbacks nor pushes into held_.
        {
            g_queue_label_calls.store(0, std::memory_order_relaxed);
            DebugLabelFunctions labels{};
            labels.cmd_insert = noop_cmd_label;
            labels.cmd_begin = noop_cmd_label;
            labels.cmd_end = noop_cmd_end;
            labels.queue_insert = count_queue_label;
            labels.queue_begin = count_queue_label;
            labels.queue_end = count_queue_end;
            context.enable_debug_labels(labels);
            EXPECT(context.debug_labels());

            VulkanQueue queue(context, qB, famB, kindB);
            EXPECT(queue.ok());
            // Establish an already-held batch BEFORE marking the device lost so held_ is non-empty.
            queue.wait(F1.semaphore(), 1);
            EXPECT(queue.execute({&p_copy, 1}) == ReplayResult::success);
            EXPECT(context.holds().any_held());

            context.mark_device_lost();
            const std::uint64_t seq_before = context.holds().current_seq();
            // 4b (held_ non-empty): marker() must not push a new held entry or call debug labels.
            queue.marker(DebugMarker::Kind::set, "lost-while-held");
            EXPECT(context.holds().current_seq() == seq_before);
            EXPECT(g_queue_label_calls.load(std::memory_order_relaxed) == 0);

            // 4a (must_hold via !held_.empty()): signal() returns false without calling pin() or hold().
            bool pin_called = false;
            const bool sig_ok = queue.signal(F2.semaphore(), 1, [&]() -> std::shared_ptr<void> {
                pin_called = true;
                return std::make_shared<int>(1);
            });
            EXPECT(!sig_ok);
            EXPECT(!pin_called);
            EXPECT(context.holds().current_seq() == seq_before);
            EXPECT(!context.holds().any_held());  // signal()'s pump_ready() dropped the prior held batch

            // 4a (must_hold via pending wait alone): signal() still returns false without calling pin() or hold().
            queue.wait(F3.semaphore(), 1);
            const bool sig_ok2 = queue.signal(F2.semaphore(), 2, [&]() -> std::shared_ptr<void> {
                pin_called = true;
                return std::make_shared<int>(2);
            });
            EXPECT(!sig_ok2);
            EXPECT(!pin_called);
            EXPECT(!context.holds().any_held());

            // 4b (held_ empty): marker() must still not invoke queue debug-label callbacks on a lost device.
            queue.marker(DebugMarker::Kind::set, "lost-after-drain");
            EXPECT(g_queue_label_calls.load(std::memory_order_relaxed) == 0);
            EXPECT(!context.holds().any_held());
        }

        // 4c: release_ready() on device loss drops held op pins OUTSIDE state_mutex_.
        {
            VulkanContext context2(driver::get_device, d.device, true);
            EXPECT(context2.loaded());
            context2.add_resource(2, rb_shared);
            context2.add_resource(3, rb_readback);
            VulkanFence F_wait(context2, 0), F_sig(context2, 0);
            EXPECT(F_wait.semaphore() && F_sig.semaphore());

            struct ObsState {
                std::mutex m;
                std::condition_variable cv;
                bool finished = false;
            };
            std::shared_ptr<ObsState> obs_state = std::make_shared<ObsState>();
            std::thread observer;
            std::atomic<bool> observed_nonblocking{false};
            std::atomic<bool> observed_sig_res{true};

            VulkanQueue queue(context2, qB, famB, kindB);
            EXPECT(queue.ok());
            ScopeExit join_observer{[&] {
                if (observer.joinable()) observer.join();
            }};

            struct PinGuard {
                VulkanQueue* q;
                VkSemaphore sig_sem;
                std::shared_ptr<ObsState> obs_state;
                std::thread* obs;
                std::atomic<bool>* nonblocking;
                std::atomic<bool>* sig_res;
                PinGuard(VulkanQueue* q_, VkSemaphore s_, std::shared_ptr<ObsState> st_, std::thread* o_,
                         std::atomic<bool>* nb_, std::atomic<bool>* sr_)
                    : q(q_), sig_sem(s_), obs_state(std::move(st_)), obs(o_), nonblocking(nb_), sig_res(sr_) {}
                PinGuard(const PinGuard&) = delete;
                PinGuard& operator=(const PinGuard&) = delete;
                ~PinGuard() {
                    auto st = obs_state;
                    auto* q_ptr = q;
                    auto sem = sig_sem;
                    auto* res_ptr = sig_res;
                    *obs = std::thread([st, q_ptr, sem, res_ptr] {
                        res_ptr->store(q_ptr->signal(sem, 1));
                        {
                            std::lock_guard lk(st->m);
                            st->finished = true;
                        }
                        st->cv.notify_one();
                    });
                    std::unique_lock lk(st->m);
                    nonblocking->store(
                        st->cv.wait_for(lk, std::chrono::milliseconds(200), [&] { return st->finished; }));
                }
            };

            queue.wait(F_wait.semaphore(), 1);
            auto pin = std::make_shared<PinGuard>(
                &queue, F_sig.semaphore(), obs_state, &observer, &observed_nonblocking, &observed_sig_res);
            EXPECT(queue.execute({&p_copy, 1}, [pin]() -> std::shared_ptr<void> { return pin; }) ==
                   ReplayResult::success);
            pin.reset();

            context2.mark_device_lost();
            context2.holds().pump_ready();
            if (observer.joinable()) observer.join();
            EXPECT(observed_nonblocking.load());
            EXPECT(!observed_sig_res.load());
        }
        std::printf("case 4 (device-lost signal/marker and outside-lock pin drop): ok\n");
    }

    const unsigned val_errors = d.validation_errors();
    std::printf("Vulkan validation errors: %u\n", val_errors);
    if (val_errors != 0) return 1;
    std::printf("vk_queue_held_wait_test: ok\n");
    return 0;
}
