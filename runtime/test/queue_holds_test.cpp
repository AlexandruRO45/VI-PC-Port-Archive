// CPU-only regression for the releases QueueHolds defers while queues hold work (D3D-203, docs section 4.4): a
// release waits for every operation held when it was asked for, on any queue, whatever order those operations are
// run or dropped in, and for nothing held after it; it runs once, from run_released().
#include "../src/queue_holds.h"

#include <cstdio>

int main() {
    leonida::QueueHolds<int> holds;
    int released = 0;
    const auto release = [&] { ++released; };

    // Nothing held: the caller releases at once.
    if (holds.after_held(release) || holds.parked() != 0) return 1;

    // Queue A holds two operations (a batch naming the resource, then its Signal), queue B one; the resource is
    // released; B then holds another operation, which cannot name it.
    const auto a1 = holds.held_added();
    const auto a2 = holds.held_added();
    const auto b1 = holds.held_added();
    if (!holds.after_held(release) || holds.parked() != 1) return 2;
    const auto b2 = holds.held_added();

    // B's operation is submitted first, then A's batch: A's Signal is still held, so the release waits.
    holds.held_removed(b1);
    holds.run_released();
    if (released != 0) return 3;
    holds.held_removed(a1);
    holds.run_released();
    if (released != 0) return 4;
    // A's last operation is submitted: the release is ready but runs only in run_released (no queue lock held).
    holds.held_removed(a2);
    if (released != 0 || holds.parked() != 1) return 5;
    holds.run_released();
    if (released != 1 || holds.parked() != 0) return 6;
    holds.run_released();
    if (released != 1) return 7;  // once

    // A release that waits for b2 alone, then one behind a new hold that waits for both; c1 goes first.
    if (!holds.after_held(release)) return 8;
    const auto c1 = holds.held_added();
    if (!holds.after_held(release) || holds.parked() != 2) return 9;
    holds.held_removed(c1);
    holds.run_released();
    if (released != 1) return 10;
    // b2 is dropped (its fence released or its queue destroyed): the same as submitted for what waits on it.
    holds.held_removed(b2);
    holds.run_released();
    if (released != 3 || holds.parked() != 0 || holds.any_held()) return 11;
    if (holds.after_held(release)) return 12;

    std::puts("releases waited for the operations held before them, in any completion order, and ran once");
    return 0;
}
