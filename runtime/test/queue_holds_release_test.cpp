// CPU-only race regressions: a fence released around a pending() decision must drop holds, and no new hold may
// open while that release is in progress.
#include "../src/queue_holds.h"

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <deque>
#include <mutex>
#include <thread>
#include <utility>

namespace {

using Holds = leonida::QueueHolds<int>;

// Models the queue half of QueueHolds: pending() decides to hold, hold() later counts it.
struct FakeHolder : Holds::Holder {
    Holds& holds;
    int semaphore;
    std::mutex held_mu;
    std::deque<std::pair<int, std::uint64_t>> held;
    bool dropped = false;
    bool released = false;

    // Optional hook: pause inside discard_waiting_on() when a removal is being tested.
    std::mutex* hook_mu = nullptr;
    std::condition_variable* hook_cv = nullptr;
    bool discard_entered = false;
    bool allow_discard = true;

    FakeHolder(Holds& h, int sem) : holds(h), semaphore(sem) {}

    bool release_ready() override {
        std::lock_guard own(held_mu);
        if (held.empty() || holds.pending(held.front().first, held.front().second)) return false;
        held.pop_front();
        holds.held_changed(-1);
        released = true;  // wrong after the fence is gone: the batch would use a destroyed semaphore
        return true;
    }

    void discard_waiting_on(int sem) override {
        if (hook_mu) {
            std::unique_lock lk(*hook_mu);
            discard_entered = true;
            hook_cv->notify_all();
            hook_cv->wait(lk, [this] { return allow_discard; });
        }
        std::lock_guard own(held_mu);
        std::size_t removed = 0;
        for (auto it = held.begin(); it != held.end();) {
            if (it->first != sem) {
                ++it;
                continue;
            }
            it = held.erase(it);
            ++removed;
        }
        if (removed != 0) {
            holds.held_changed(-static_cast<std::int64_t>(removed));
            dropped = true;
        }
    }

    void add_hold(int sem, std::uint64_t value) {
        std::lock_guard own(held_mu);
        held.emplace_back(sem, value);
        holds.held_changed(1);
    }
};

// 1. A release between the pending() decision and hold() must drop the hold, not submit it.
int test_decision_window() {
    Holds holds;
    const int sem = 7;
    holds.add_fence(sem, 0);
    FakeHolder fake(holds, sem);
    holds.attach(&fake);

    std::mutex mu;
    std::condition_variable cv;
    bool pending_seen = false, removal_entered = false;

    std::thread holder([&] {
        holds.begin_hold();
        const bool need_hold = holds.pending(sem, 1);
        {
            std::lock_guard lock(mu);
            pending_seen = true;
        }
        cv.notify_all();
        {
            std::unique_lock lock(mu);
            cv.wait(lock, [&] { return removal_entered; });
        }
        if (need_hold) fake.add_hold(sem, 1);
        holds.end_hold();
        holds.pump_ready();
    });

    {
        std::unique_lock lock(mu);
        cv.wait(lock, [&] { return pending_seen; });
    }
    {
        std::lock_guard lock(mu);
        removal_entered = true;
    }
    cv.notify_all();
    holds.remove_fence(sem);  // must wait for the reservation and drop the hold created inside it
    holder.join();

    if (!fake.dropped) return 1;
    if (fake.released) return 2;
    if (holds.any_held()) return 3;
    return 0;
}

// 2. After the zero-reservation wait, a new bracket must not open until the fence has been dropped and erased.
int test_post_zero_window() {
    Holds holds;
    const int sem = 9;
    holds.add_fence(sem, 0);
    FakeHolder fake(holds, sem);
    std::mutex mu;
    std::condition_variable cv;
    fake.hook_mu = &mu;
    fake.hook_cv = &cv;
    fake.allow_discard = false;
    holds.attach(&fake);

    bool first_seen = false, go = false, newcomer_entered = false, newcomer_pending = false;
    bool early_failure = false;

    std::thread first([&] {
        holds.begin_hold();
        const bool need_hold = holds.pending(sem, 1);
        {
            std::lock_guard lock(mu);
            first_seen = true;
        }
        cv.notify_all();
        {
            std::unique_lock lock(mu);
            cv.wait(lock, [&] { return go; });
        }
        if (need_hold) fake.add_hold(sem, 1);
        holds.end_hold();
        holds.pump_ready();
    });

    {
        std::unique_lock lock(mu);
        cv.wait(lock, [&] { return first_seen; });
    }
    {
        std::lock_guard lock(mu);
        go = true;
    }
    cv.notify_all();

    std::thread remover([&] { holds.remove_fence(sem); });
    {
        std::unique_lock lock(mu);
        cv.wait(lock, [&] { return fake.discard_entered; });  // removal published, old brackets drained
    }

    std::thread newcomer([&] {
        holds.begin_hold();  // must block while the removal is in progress
        {
            std::lock_guard lock(mu);
            newcomer_entered = true;
        }
        newcomer_pending = holds.pending(sem, 1);
        if (newcomer_pending) fake.add_hold(sem, 1);
        holds.end_hold();
        holds.pump_ready();
    });

    // Give the newcomer time to try; it must still be waiting for the removal to finish.
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    {
        std::lock_guard lock(mu);
        if (newcomer_entered) early_failure = true;
    }

    {
        std::lock_guard lock(mu);
        fake.allow_discard = true;
    }
    cv.notify_all();
    remover.join();
    newcomer.join();
    first.join();
    if (early_failure) return 4;

    if (newcomer_pending) return 5;
    if (!fake.dropped) return 6;
    if (fake.released) return 7;
    if (holds.any_held()) return 8;
    return 0;
}

}  // namespace

int main() {
    if (const int rc = test_decision_window()) return rc;
    if (const int rc = test_post_zero_window()) return rc;
    std::puts("queue_holds: fence release drops holds and blocks new ones until the fence is erased");
    return 0;
}
