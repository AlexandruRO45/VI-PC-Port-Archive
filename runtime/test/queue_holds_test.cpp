// Portable CPU unit and concurrency tests for runtime/src/queue_holds.h.
// Pure C++20 standard library + Threads; no Vulkan, Windows SDK, or game files required.

#include "queue_holds.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <deque>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace {

using Holds = leonida::QueueHolds<std::uint64_t>;

int failures = 0;

void check(bool cond, const char* msg) {
    if (!cond) {
        std::fprintf(stderr, "FAIL: %s\n", msg);
        ++failures;
    }
}

struct FakeOp {
    std::uint64_t seq = 0;
    std::uint64_t wait_sem = 0;
    std::uint64_t wait_val = 0;
    std::function<void()> on_release;
};

class FakeHolder final : public Holds::Holder {
public:
    explicit FakeHolder(Holds& holds) : holds_(holds) {}

    std::uint64_t hold(std::uint64_t sem, std::uint64_t val, std::function<void()> on_release) {
        std::uint64_t seq = 0;
        {
            std::lock_guard lock(mu_);
            seq = holds_.next_seq();
            ops_.push_back(FakeOp{seq, sem, val, std::move(on_release)});
            holds_.held_changed(1);
        }
        return seq;
    }

    bool release_ready() override {
        std::vector<std::function<void()>> ready;
        {
            std::lock_guard lock(mu_);
            while (!ops_.empty() && !holds_.pending(ops_.front().wait_sem, ops_.front().wait_val)) {
                ready.push_back(std::move(ops_.front().on_release));
                ops_.pop_front();
                holds_.held_changed(-1);
            }
        }
        for (auto& fn : ready) {
            if (fn) fn();
        }
        return !ready.empty();
    }

    void discard_waiting_on(std::uint64_t sem) override {
        std::lock_guard lock(mu_);
        std::deque<FakeOp> kept;
        for (auto& op : ops_) {
            if (op.wait_sem == sem) {
                ++discarded_;
                holds_.held_changed(-1);
            } else {
                kept.push_back(std::move(op));
            }
        }
        ops_ = std::move(kept);
    }

    std::uint64_t oldest_held_seq() override {
        std::lock_guard lock(mu_);
        return ops_.empty() ? 0 : ops_.front().seq;
    }

    std::size_t discarded() const {
        std::lock_guard lock(mu_);
        return discarded_;
    }

private:
    Holds& holds_;
    mutable std::mutex mu_;
    std::deque<FakeOp> ops_;
    std::size_t discarded_ = 0;
};

void test_fence_monotonicity_and_unknown_semaphores() {
    Holds h;
    constexpr std::uint64_t kUnknown = 999;
    constexpr std::uint64_t kFence = 1;

    // Unknown semaphores are treated as already satisfied for pending(), and always signaled for would_signal().
    check(!h.pending(kUnknown, 1), "unknown semaphore is not pending");
    check(h.would_signal(kUnknown, 1), "unknown semaphore would_signal is true");
    h.signaled(kUnknown, 5);
    check(!h.pending(kUnknown, 10), "signaling unknown semaphore stays non-pending");

    h.add_fence(kFence, 2);
    check(!h.pending(kFence, 2), "initial value is satisfied");
    check(h.pending(kFence, 3), "above initial value is pending");
    check(!h.would_signal(kFence, 2), "would_signal false at current value");
    check(!h.would_signal(kFence, 1), "would_signal false below current value");
    check(h.would_signal(kFence, 3), "would_signal true above current value");

    // Signaled value advances monotonically and ignores regressions.
    h.signaled(kFence, 5);
    check(!h.pending(kFence, 5), "signaled value 5 satisfied");
    check(h.pending(kFence, 6), "value 6 still pending");
    h.signaled(kFence, 3);
    check(!h.pending(kFence, 5), "lower signal does not regress fence");
    check(!h.would_signal(kFence, 5), "would_signal false after reaching 5");
}

void test_chained_cross_queue_release_and_recursive_pump() {
    Holds h;
    constexpr std::uint64_t kF1 = 10, kF2 = 20;
    h.add_fence(kF1, 0);
    h.add_fence(kF2, 0);

    FakeHolder q1(h), q2(h);
    h.attach(&q1);
    h.attach(&q2);

    std::vector<int> order;
    // q2 waits on kF2=1, which is only signaled when q1's held op (waiting on kF1=1) releases.
    q2.hold(kF2, 1, [&] { order.push_back(2); });
    q1.hold(kF1, 1, [&] {
        order.push_back(1);
        h.signaled(kF2, 1);
        h.pump_ready();  // recursive call inside release_ready() sets dirty_ for another pass
    });

    check(h.any_held(), "both queues have held ops");
    h.pump_ready();
    check(order.empty(), "nothing releases before kF1 is signaled");

    h.signaled(kF1, 1);
    h.pump_ready();
    check(!h.any_held(), "chained release drained all held ops");
    check(order.size() == 2 && order[0] == 1 && order[1] == 2, "q1 released before q2 via recursive pump");

    h.detach(&q2);
    h.detach(&q1);
}

void test_after_held_ordering_and_cancel() {
    Holds h;
    constexpr std::uint64_t kF1 = 100, kF2 = 200;
    h.add_fence(kF1, 0);
    h.add_fence(kF2, 0);

    FakeHolder q(h);
    h.attach(&q);

    // When nothing is held, after_held runs immediately.
    int immediate = 0;
    const int owner_a = 1, owner_b = 2;
    h.after_held(&owner_a, [&] { ++immediate; });
    check(immediate == 1, "after_held runs immediately when nothing is held");

    std::vector<int> events;
    q.hold(kF1, 1, [&] { events.push_back(10); });

    // Register two callbacks for owner_a and one for owner_b while op 10 is held.
    h.after_held(&owner_a, [&] { events.push_back(101); });
    h.after_held(&owner_b, [&] { events.push_back(201); });
    h.after_held(&owner_a, [&] { events.push_back(102); });

    // Hold a newer op on kF2 AFTER the callbacks were registered: callbacks must not wait for it.
    q.hold(kF2, 1, [&] { events.push_back(20); });

    check(events.empty(), "callbacks stay queued while earlier op is held");

    // Cancel owner_b before release.
    h.cancel_after(&owner_b);

    h.signaled(kF1, 1);
    h.pump_ready();
    // op 10 releases, then owner_a's two callbacks run in order; op 20 stays held (kF2 still 0).
    check(events.size() == 3 && events[0] == 10 && events[1] == 101 && events[2] == 102,
          "after_held callbacks run in FIFO order after prior ops and ignore cancelled owner / later ops");
    check(h.any_held(), "op on kF2 remains held");

    h.signaled(kF2, 1);
    h.pump_ready();
    check(!h.any_held() && events.size() == 4 && events[3] == 20, "later op releases when kF2 signals");

    h.detach(&q);
}

void test_remove_fence_discards_waiters_and_unblocks_barriers() {
    Holds h;
    constexpr std::uint64_t kDead = 300;
    h.add_fence(kDead, 0);

    FakeHolder q(h);
    h.attach(&q);

    int op_ran = 0, barrier_ran = 0;
    const int owner = 1;
    q.hold(kDead, 1, [&] { ++op_ran; });
    h.after_held(&owner, [&] { ++barrier_ran; });

    check(h.any_held() && barrier_ran == 0, "op and barrier held before remove_fence");

    h.remove_fence(kDead);
    check(op_ran == 0, "removed fence op is discarded, not executed");
    check(q.discarded() == 1, "holder received discard_waiting_on for removed fence");
    check(!h.any_held(), "discard cleared held count");
    check(barrier_ran == 1, "removing fence pumped now-unblocked barrier");
    check(!h.pending(kDead, 1), "removed fence is no longer pending");

    h.detach(&q);
}

void test_detach_waits_for_running_pump_and_multithreaded_stress() {
    Holds h;
    constexpr std::uint64_t kF = 400;
    h.add_fence(kF, 0);

    // Part A: detach() must wait until an in-progress release_ready() finishes.
    std::atomic<bool> in_release{false};
    std::atomic<bool> allow_finish{false};
    std::atomic<bool> detach_returned{false};

    FakeHolder q(h);
    h.attach(&q);
    q.hold(kF, 1, [&] {
        in_release.store(true, std::memory_order_release);
        while (!allow_finish.load(std::memory_order_acquire))
            std::this_thread::yield();
    });

    h.signaled(kF, 1);
    std::thread pumper([&] { h.pump_ready(); });
    while (!in_release.load(std::memory_order_acquire))
        std::this_thread::yield();

    std::thread detacher([&] {
        h.detach(&q);
        detach_returned.store(true, std::memory_order_release);
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(15));
    check(!detach_returned.load(std::memory_order_acquire), "detach waits while release_ready is active");

    allow_finish.store(true, std::memory_order_release);
    pumper.join();
    detacher.join();
    check(detach_returned.load(std::memory_order_acquire), "detach finishes once release completes");

    // Part B: concurrent signalers, pumpers, and barrier callers complete every item with zero lost wakeups.
    constexpr int kSteps = 200;
    constexpr std::uint64_t kStressFence = 500;
    h.add_fence(kStressFence, 0);
    FakeHolder sq(h);
    h.attach(&sq);

    std::atomic<int> released{0};
    std::atomic<int> barriers{0};
    const int owner = 7;

    for (int i = 1; i <= kSteps; ++i) {
        sq.hold(kStressFence, std::uint64_t(i), [&] { released.fetch_add(1, std::memory_order_relaxed); });
    }

    std::thread t_sig([&] {
        for (int i = 1; i <= kSteps; ++i) {
            h.signaled(kStressFence, std::uint64_t(i));
            h.pump_ready();
        }
    });
    std::thread t_bar([&] {
        for (int i = 0; i < 64; ++i) {
            h.after_held(&owner, [&] { barriers.fetch_add(1, std::memory_order_relaxed); });
        }
    });
    t_sig.join();
    t_bar.join();
    h.pump_ready();

    check(released.load(std::memory_order_acquire) == kSteps, "all concurrent held ops released");
    check(barriers.load(std::memory_order_acquire) == 64, "all concurrent after_held callbacks ran");
    check(!h.any_held(), "no held ops remain after concurrent stress");

    h.detach(&sq);
}

}  // namespace

int main() {
    test_fence_monotonicity_and_unknown_semaphores();
    test_chained_cross_queue_release_and_recursive_pump();
    test_after_held_ordering_and_cancel();
    test_remove_fence_discards_waiters_and_unblocks_barriers();
    test_detach_waits_for_running_pump_and_multithreaded_stress();
    if (failures) {
        std::fprintf(stderr, "queue_holds_test: %d failure(s)\n", failures);
        return 1;
    }
    std::printf("queue_holds_test: ok\n");
    return 0;
}
