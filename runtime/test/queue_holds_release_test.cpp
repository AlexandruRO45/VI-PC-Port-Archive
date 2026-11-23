// CPU-only race regression: a fence released between a hold decision and the hold itself must drop that hold.
#include "../src/queue_holds.h"

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
    std::deque<std::pair<int, std::uint64_t>> held;
    bool dropped = false;
    bool released = false;

    FakeHolder(Holds& h, int sem) : holds(h), semaphore(sem) {}

    bool release_ready() override {
        if (held.empty() || holds.pending(held.front().first, held.front().second)) return false;
        held.pop_front();
        holds.held_changed(-1);
        released = true;  // wrong after the fence is gone: the batch would use a destroyed semaphore
        return true;
    }

    void discard_waiting_on(int sem) override {
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
};

}  // namespace

int main() {
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
        if (need_hold) {
            fake.held.emplace_back(sem, 1);
            holds.held_changed(1);
        }
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

    std::puts("queue_holds: a fence released mid-decision drops the hold instead of submitting it");
    return 0;
}
