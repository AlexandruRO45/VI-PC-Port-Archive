// Host-only stalls and the guest Sleep (src/host_stall.h, shims/host_stall.cpp).
//
// Part 1 checks the ledger with times passed in, so it is exact: the accounting, the per-stall limit and what a
// sleep still owes. Part 2 runs sleeps and a loop built like a title's hang check (count Sleep(1000) rounds without
// progress, start again after a round more than 2 s long) on a virtual clock, with the same ledger calls the shim
// makes: when a Sleep returns, and when the check fires under back-to-back, recurring, short and never-ending
// stalls. Part 3 runs the Sleep the registry serves on real threads: a short Sleep is not held, a long one is, and
// a stall that never ends releases it at the limit, and compiles that take turns count only their driver calls.
// Synthetic threads only: no game files, no GPU.
#include <windows.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include "../src/host_stall.h"
#include "shims.h"

namespace {

using rt::host_stall::Ledger;
using rt::host_stall::Nanoseconds;
using std::chrono::milliseconds;

int failures = 0;
int checks = 0;

void check(bool ok, const char* what) {
    ++checks;
    if (!ok) {
        ++failures;
        std::printf("FAIL %s\n", what);
    }
}

Nanoseconds ms(long long n) { return std::chrono::duration_cast<Nanoseconds>(milliseconds(n)); }

long long elapsed_ms(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration_cast<milliseconds>(std::chrono::steady_clock::now() - start).count();
}

using SleepFn = void(WINAPI*)(DWORD);

SleepFn served_sleep(const char* dll) { return reinterpret_cast<SleepFn>(rt::shims::find(dll, "Sleep")); }

// ---- Part 1: the ledger ------------------------------------------------------------------------------------------

void ledger_checks() {
    Ledger l(ms(60000));
    check(!l.open(ms(5)) && l.stalled(ms(5)) == ms(0), "an empty ledger has no stall");
    const Ledger::Id first = l.begin(ms(10));
    check(l.open(ms(10)) && l.stalled(ms(10)) == ms(0) && l.stalled(ms(25)) == ms(15), "an open stall counts up to now");
    l.end(first, ms(40));
    check(!l.open(ms(40)) && l.stalled(ms(40)) == ms(30) && l.stalled(ms(900)) == ms(30), "a closed stall stops counting");

    // Nested and overlapping stalls count the covered time once.
    const Ledger::Id outer = l.begin(ms(100));
    const Ledger::Id inner = l.begin(ms(110));
    l.end(inner, ms(120));
    check(l.open(ms(130)) && l.stalled(ms(130)) == ms(60), "the outer stall stays open when the inner one ends");
    l.end(outer, ms(150));
    check(!l.open(ms(150)) && l.stalled(ms(150)) == ms(80), "two overlapping stalls count 50 ms, not 60");

    // Two threads: the one that began first ends first.
    const Ledger::Id a = l.begin(ms(200));
    const Ledger::Id b = l.begin(ms(210));
    l.end(a, ms(220));
    check(l.open(ms(225)) && l.stalled(ms(225)) == ms(105), "the later stall stays open when the earlier one ends");
    l.end(b, ms(230));
    check(!l.open(ms(230)) && l.stalled(ms(300)) == ms(110), "crossing stalls count the covered 30 ms once");

    // An end without a begin, or a second end, changes nothing, and does not make a later begin close early.
    l.end(b, ms(240));
    l.end(12345, ms(240));
    check(!l.open(ms(250)) && l.stalled(ms(250)) == ms(110), "an unmatched end is ignored");
    const Ledger::Id late = l.begin(ms(300));
    check(l.open(ms(330)) && l.stalled(ms(330)) == ms(140), "a begin after an unmatched end opens a stall");
    l.end(late, ms(330));
}

void limit_checks() {
    // One stall counts for at most the limit; after that it is overdue, not open.
    Ledger l(ms(60000));
    const Ledger::Id stuck = l.begin(ms(0));
    check(l.open(ms(59999)) && !l.overdue(ms(59999)), "a stall is open up to its limit");
    check(!l.open(ms(60000)) && l.overdue(ms(60000)), "at its limit a stall is overdue and no longer open");
    check(l.stalled(ms(100000)) == ms(60000), "an overdue stall counts its limit and no more");

    // A stall beside the overdue one is still a stall, with a limit of its own.
    const Ledger::Id next = l.begin(ms(100000));
    check(l.open(ms(103000)) && l.stalled(ms(103000)) == ms(63000), "a new stall counts beside an overdue one");
    l.end(next, ms(105000));
    check(!l.open(ms(105000)) && l.overdue(ms(105000)) && l.stalled(ms(200000)) == ms(65000),
          "when it ends, the overdue one does not go on counting");
    l.end(stuck, ms(300000));
    check(!l.overdue(ms(300000)) && l.stalled(ms(400000)) == ms(65000), "the overdue stall's late end adds nothing");

    // The limit is per stall, not per run: an older stall that is still within its limit keeps the run going after
    // a newer one ends, up to its own limit.
    Ledger m(ms(60000));
    const Ledger::Id old_one = m.begin(ms(0));
    const Ledger::Id new_one = m.begin(ms(50000));
    m.end(new_one, ms(55000));
    check(m.open(ms(59000)) && !m.open(ms(60000)) && m.stalled(ms(90000)) == ms(60000),
          "an older stall counts to its own limit after a newer one ends");
    m.end(old_one, ms(90000));
    // And a newer one that outlives the older one's limit counts to its end.
    Ledger n(ms(60000));
    const Ledger::Id n_old = n.begin(ms(0));
    const Ledger::Id n_new = n.begin(ms(50000));
    check(n.open(ms(70000)) && n.overdue(ms(70000)), "a stall within its limit is open beside an overdue one");
    n.end(n_new, ms(80000));
    check(!n.open(ms(80000)) && n.stalled(ms(95000)) == ms(80000), "the run ends with the newer stall");
    n.end(n_old, ms(95000));
}

void owed_checks() {
    // The review's probe: a Sleep(15) under a stall from 0 to 185 ms that the host woke at 185 ms. 15 ms outside
    // the stall are still owed, so it returns at 200 ms (it used to return at 185 + 185 + 15).
    Ledger l(ms(60000));
    const Ledger::Mark start = l.mark(ms(0));
    const Ledger::Id id = l.begin(ms(0));
    check(l.owed(start, ms(15), ms(185)) == ms(15), "under an open stall the whole sleep is owed");
    check(l.look_again(ms(15), ms(185)) == ms(0) + l.limit(), "and it is looked at again at the stall's limit");
    l.end(id, ms(185));
    check(l.owed(start, ms(15), ms(185)) == ms(15) && l.look_again(ms(15), ms(185)) == ms(200),
          "a late host wake-up is not charged twice: 15 ms after the stall, not 200");
    check(l.owed(start, ms(15), ms(200)) == ms(0), "and then it owes nothing");

    // A sleep no stall overlapped owes nothing, even when the host woke it early or late.
    const Ledger::Mark later = l.mark(ms(1000));
    check(l.owed(later, ms(1000), ms(1990)) == ms(0) && l.owed(later, ms(1000), ms(2700)) == ms(0),
          "a sleep no stall overlapped returns when the host wakes it");
}

// ---- Part 2: sleeps and a hang check on a virtual clock -------------------------------------------------------------

struct Stall {
    long long begin_ms;
    long long end_ms;  // below zero: never ends
};

// The shim's steps with the time passed in: mark, the host's sleep, then owed()/look_again() until nothing is owed,
// woken early by every stall that begins or ends. Times in milliseconds; sleeps start in order. The limit is 60 s
// here unless a case passes another, whatever kStallLimit is.
class Sim {
public:
    explicit Sim(const std::vector<Stall>& stalls, Nanoseconds limit = ms(60000)) : ledger_(limit) {
        ids_.resize(stalls.size());
        for (std::size_t i = 0; i < stalls.size(); ++i) {
            events_.push_back({stalls[i].begin_ms, false, i});
            if (stalls[i].end_ms >= 0) events_.push_back({stalls[i].end_ms, true, i});
        }
        // At one instant an end comes before a begin: back-to-back stalls do not overlap.
        std::stable_sort(events_.begin(), events_.end(), [](const Event& a, const Event& b) {
            return a.at != b.at ? a.at < b.at : a.is_end && !b.is_end;
        });
    }

    // When a guest Sleep(n) called at `at` returns. `host_late`: how much later than n the host's Sleep woke.
    long long sleep(long long at, std::uint32_t n, long long host_late = 0) {
        if (!rt::host_stall::sleep_is_held(n)) return at + n + host_late;
        advance(at);
        const Ledger::Mark start = ledger_.mark(ms(at));
        long long now = at + n + host_late;
        for (;;) {
            advance(now);
            const Nanoseconds owed = ledger_.owed(start, ms(n), ms(now));
            if (owed <= Nanoseconds{0}) return now;
            const long long again = std::chrono::duration_cast<milliseconds>(ledger_.look_again(owed, ms(now))).count();
            now = next_ < events_.size() && events_[next_].at < again ? events_[next_].at : again;
        }
    }

    // A stand-in for the kind of check the retail title runs, with nothing reporting progress: every round sleeps
    // `round`; a round longer than `late` starts the count again; `limit` rounds in a row fire. Returns when it
    // fires, or -1 if it has not by `horizon`.
    long long hang_check_fires(long long from, long long horizon, std::uint32_t round = 1000, long long late = 2000,
                               int limit = 30) {
        int left = limit;
        long long now = from;
        while (now < horizon) {
            const long long before = now;
            now = sleep(now, round);
            if (now - before > late) {
                left = limit;
                continue;
            }
            if (--left == 0) return now;
        }
        return -1;
    }

private:
    struct Event {
        long long at;
        bool is_end;
        std::size_t stall;
    };

    void advance(long long to) {
        for (; next_ < events_.size() && events_[next_].at <= to; ++next_) {
            const Event& e = events_[next_];
            if (e.is_end)
                ledger_.end(ids_[e.stall], ms(e.at));
            else
                ids_[e.stall] = ledger_.begin(ms(e.at));
        }
    }

    Ledger ledger_;
    std::vector<Event> events_;
    std::vector<Ledger::Id> ids_;
    std::size_t next_ = 0;
};

void virtual_sleep_checks() {
    check(!rt::host_stall::sleep_is_held(0) && !rt::host_stall::sleep_is_held(10) && !rt::host_stall::sleep_is_held(999) &&
              rt::host_stall::sleep_is_held(1000) && rt::host_stall::sleep_is_held(5000) &&
              !rt::host_stall::sleep_is_held(INFINITE),
          "only a Sleep of a second or more is held; a yield, a poll and an endless sleep are not");
    check(Sim({{0, 6000}}).sleep(0, 10) == 10, "a Sleep(10) under a 6 s stall returns after 10 ms");

    // The review's table: Sleep(1000) at 0 s under a stall that ends at 6 s returns at 7 s, not at 12 s.
    check(Sim({{0, 6000}}).sleep(0, 1000) == 7000, "Sleep(1000) under a stall from 0 to 6 s returns at 7 s");
    // The same when the host woke the thread only when the stall ended.
    check(Sim({{0, 6000}}).sleep(0, 1000, 5000) == 7000, "and at 7 s when the host woke it at 6 s");
    check(Sim({{500, 3000}}).sleep(0, 1000) == 3500, "500 ms before the stall count: 500 ms after it are left");
    check(Sim({{200, 400}}).sleep(0, 1000) == 1200, "a 200 ms stall inside the sleep moves it by 200 ms");
    check(Sim({{200, 400}}).sleep(0, 1000, 150) == 1200, "and a host wake-up 150 ms late is part of that, not on top");
    check(Sim({{200, 400}}).sleep(0, 1000, 700) == 1700, "a host wake-up later than what is owed returns at once");
    check(Sim({}).sleep(0, 1000, 300) == 1300, "without a stall the Sleep returns when the host wakes it");
    check(Sim({{0, 500}}).sleep(600, 1000) == 1600, "a stall that closed before the Sleep started is not charged");
    check(Sim({{0, 2000}, {2500, 4000}}).sleep(0, 1000) == 4500,
          "a stall that opens while the rest is waited out is sat out too");

    // A stall that never ends counts for its limit: the Sleep returns one second after it.
    check(Sim({{0, -1}}).sleep(0, 1000) == 61000, "under a stall that never ends, Sleep(1000) returns at limit + 1 s");
    check(Sim({{0, -1}}, ms(5000)).sleep(2000, 1000) == 6000, "the limit runs from the stall's start, not the Sleep's");
}

void virtual_hang_check_checks() {
    // Nothing reports progress in any of these: the question is only when the check fires.
    check(Sim({}).hang_check_fires(0, 200000) == 30000, "without a stall the check fires after 30 rounds");

    // A cold boot: 52 compiles of 1.1 s back to back from 4 s to 61.2 s. The check does not fire while they run,
    // and fires within 30 rounds and one late round once they are over.
    {
        std::vector<Stall> compiles;
        for (int i = 0; i < 52; ++i) compiles.push_back({4000 + 1100LL * i, 4000 + 1100LL * (i + 1)});
        const long long fired = Sim(compiles).hang_check_fires(0, 400000);
        check(fired > 61200, "the check does not fire during 57.2 s of back-to-back compiles");
        check(fired != -1 && fired <= 61200 + 31000, "and fires within 31 s of the last one when nothing moves after it");
    }

    // The same with 20 ms of the title's own work between two compiles: that time counts toward the round, so the
    // round ends inside the compiles, more than 2 s long, and the count starts again.
    {
        std::vector<Stall> compiles;
        for (int i = 0; i < 52; ++i) compiles.push_back({4000 + 1120LL * i, 4000 + 1120LL * i + 1100});
        const long long last_end = 4000 + 1120LL * 51 + 1100;
        const long long fired = Sim(compiles).hang_check_fires(0, 400000);
        check(fired > last_end, "nor during compiles with 20 ms between them");
        check(fired != -1 && fired <= last_end + 31000, "and again within 31 s of the last one");
    }

    // Recurring stalls: 2.5 s of work every 4 s, 20 times, then nothing. Rounds that a stall made longer than 2 s
    // start the count again, by the title's own rule; after the last stall the check fires within 31 s.
    {
        std::vector<Stall> recurring;
        for (int i = 0; i < 20; ++i) recurring.push_back({4000LL * i, 4000LL * i + 2500});
        const long long last_end = 4000LL * 19 + 2500;
        const long long fired = Sim(recurring).hang_check_fires(0, 400000);
        check(fired != -1 && fired <= last_end + 31000, "after recurring stalls the check fires within 31 s of the last");
    }

    // Stalls too short to make a round 2 s long reset nothing: 100 ms of work every second, for ever. Each round
    // takes 1.1 or 1.2 s, and the check fires after 30 of them.
    {
        std::vector<Stall> brief;
        for (int i = 0; i < 400; ++i) brief.push_back({1000LL * i + 500, 1000LL * i + 600});
        const long long fired = Sim(brief).hang_check_fires(0, 400000);
        check(fired != -1 && fired <= 36000, "short recurring stalls do not keep the check from firing");
    }

    // A host call that never returns: the check fires 30 rounds after the stall's limit and the round it ended.
    check(Sim({{4000, -1}}).hang_check_fires(0, 400000) <= 4000 + 60000 + 31000 &&
              Sim({{4000, -1}}).hang_check_fires(0, 400000) != -1,
          "under a stall that never ends the check fires within limit + 31 s");
    // Another thread compiling meanwhile does not extend the stuck one beyond its own stalls.
    check(Sim({{4000, -1}, {70000, 73000}}).hang_check_fires(0, 400000) <= 73000 + 31000 &&
              Sim({{4000, -1}, {70000, 73000}}).hang_check_fires(0, 400000) != -1,
          "a later stall beside the stuck one delays the check only by itself");
}

// ---- Part 3: the served Sleep on real threads -------------------------------------------------------------------

void sleep_binding_checks() {
    const SleepFn api_set = served_sleep("api-ms-win-core-synch-l1-2-0.dll");
    check(api_set != nullptr, "Sleep is served under api-ms-win-core-synch-l1-2-0.dll");
    check(served_sleep("kernel32.dll") == api_set && served_sleep("kernelbase.dll") == api_set &&
              served_sleep("api-ms-win-core-synch-l1-2-1.dll") == api_set,
          "kernel32, kernelbase and the later API set serve the same Sleep");
    check(rt::shims::find("api-ms-win-core-synch-l1-2-0.dll", "SleepEx") == nullptr &&
              rt::shims::find("kernel32.dll", "SleepConditionVariableSRW") == nullptr,
          "SleepEx and the condition-variable sleeps stay the host's");
    check(!rt::shims::owns_dll("kernel32.dll") && !rt::shims::owns_dll("api-ms-win-core-synch-l1-2-0.dll"),
          "serving Sleep does not make the shims own kernel32 or the API set");
}

void sleep_without_a_stall(SleepFn guest_sleep) {
    const auto start = std::chrono::steady_clock::now();
    guest_sleep(50);
    const long long took = elapsed_ms(start);
    check(took >= 45 && took < 1000, "a Sleep(50) no stall overlaps takes about 50 ms");
    const auto yield_start = std::chrono::steady_clock::now();
    guest_sleep(0);
    check(elapsed_ms(yield_start) < 1000, "Sleep(0) yields and returns");
    const auto long_start = std::chrono::steady_clock::now();
    guest_sleep(1000);
    const long long long_took = elapsed_ms(long_start);
    check(long_took >= 950 && long_took < 1900, "a Sleep(1000) no stall overlaps takes about a second");
}

void poll_under_a_stall(SleepFn guest_sleep) {
    // The review's case: the thread that holds the stall waits for another thread that polls with Sleep(10). The
    // answer has to come while the stall is open. The wait is bounded so that a held Sleep fails the check instead
    // of hanging the test.
    auto& gate = rt::host_stall::gate();
    const Ledger::Id stall = gate.begin();
    std::atomic<bool> answered{false};
    std::thread poller([&] {
        guest_sleep(10);
        answered = true;
    });
    for (int i = 0; i < 1000 && !answered.load(); ++i) std::this_thread::sleep_for(milliseconds(5));
    check(answered.load(), "a thread that polls with Sleep(10) answers the stall's holder while the stall is open");
    gate.end(stall);
    poller.join();
}

void sleep_held_by_a_stall(SleepFn guest_sleep) {
    // A Sleep(1000) started under a stall is still asleep long after a second, and then lasts a second more.
    auto& gate = rt::host_stall::gate();
    const Ledger::Id stall = gate.begin();
    std::atomic<bool> returned{false};
    long long sleeper_ms = 0;
    const auto start = std::chrono::steady_clock::now();
    std::thread sleeper([&] {
        guest_sleep(1000);
        sleeper_ms = elapsed_ms(start);
        returned = true;
    });
    std::this_thread::sleep_for(milliseconds(1500));
    check(!returned.load(), "a Sleep(1000) does not return while a stall is open");
    {
        const rt::host_stall::Scope nested;  // a second, nested stall ends first and releases nothing
    }
    std::this_thread::sleep_for(milliseconds(200));
    check(!returned.load(), "a nested stall that ends leaves the Sleep held");
    const long long stall_ms = elapsed_ms(start);
    gate.end(stall);
    sleeper.join();
    check(returned.load() && sleeper_ms >= stall_ms + 950, "the Sleep lasts a second more once the stall is over");
    check(sleeper_ms < stall_ms + 1900, "and not two");

    // A Sleep that starts after the stall is over owes nothing.
    const auto after = std::chrono::steady_clock::now();
    guest_sleep(1000);
    check(elapsed_ms(after) < 1900, "a later Sleep is not held by a closed stall");
}

// Compiles that take turns, as behind a compile gate: a thread waits for its turn, then opens the stall around the
// driver call alone, then gives the turn back. Between two turns no stall is open and none is counted, although a
// compiler is waiting; unrelated Sleeps go on at every point.
void compiles_that_take_turns(SleepFn guest_sleep) {
    auto& gate = rt::host_stall::gate();
    std::mutex turn_mutex;
    std::condition_variable turn_changed;
    bool turn_taken = false;
    std::atomic<int> in_driver{0};  // which compiler is inside its driver call (1 or 2), 0 for none
    std::atomic<bool> driver_may_return[2] = {false, false};
    std::atomic<bool> second_may_take_turn{false};
    std::atomic<bool> second_is_waiting{false};

    const auto compile = [&](int who) {
        {
            std::unique_lock<std::mutex> lock(turn_mutex);
            if (who == 2) second_is_waiting = true;
            turn_changed.wait(lock, [&] { return !turn_taken && (who == 1 || second_may_take_turn.load()); });
            turn_taken = true;
        }
        {
            const rt::host_stall::Scope compiling;
            in_driver = who;
            while (!driver_may_return[who - 1].load()) std::this_thread::sleep_for(milliseconds(1));
            in_driver = 0;
        }
        {
            const std::lock_guard<std::mutex> lock(turn_mutex);
            turn_taken = false;
        }
        turn_changed.notify_all();
    };
    // Four unrelated threads poll with Sleep(1); how many come back within a bounded wait. The count outlives this
    // call, so a poll that was held and comes back late touches nothing that is gone.
    const auto polls_that_return = [guest_sleep] {
        const auto done = std::make_shared<std::atomic<int>>(0);
        for (int i = 0; i < 4; ++i)
            std::thread([guest_sleep, done] {
                guest_sleep(1);
                ++*done;
            }).detach();
        for (int i = 0; i < 1000 && done->load() < 4; ++i) std::this_thread::sleep_for(milliseconds(5));
        return done->load();
    };
    const auto wait_for = [](const auto& condition) {
        for (int i = 0; i < 2000 && !condition(); ++i) std::this_thread::sleep_for(milliseconds(5));
        return condition();
    };

    std::thread first(compile, 1);
    check(wait_for([&] { return in_driver.load() == 1; }), "the first compiler is in its driver call");
    std::thread second(compile, 2);
    check(wait_for([&] { return second_is_waiting.load(); }), "the second compiler waits for its turn");
    const Nanoseconds in_first = gate.mark().stalled;
    check(polls_that_return() == 4, "four Sleep(1) polls return while a compile is in the driver and another waits");
    check(gate.mark().stalled > in_first, "the driver call counts as a stall");

    // Between the turns: the first has given its turn back, the second has not been let in yet.
    driver_may_return[0] = true;
    first.join();
    const Ledger::Mark between = gate.mark();
    check(polls_that_return() == 4, "four Sleep(1) polls return between two turns");
    const auto long_start = std::chrono::steady_clock::now();
    guest_sleep(1000);
    check(elapsed_ms(long_start) < 1900, "a Sleep(1000) between two turns is not held by the waiting compiler");
    check(gate.mark().stalled == between.stalled, "no stall time is counted while a compiler only waits for its turn");

    second_may_take_turn = true;
    turn_changed.notify_all();
    check(wait_for([&] { return in_driver.load() == 2; }), "the second compiler gets its turn");
    driver_may_return[1] = true;
    second.join();
    check(gate.mark().stalled > between.stalled, "and its driver call counts as a stall again");
}

void sleep_under_a_stall_that_never_ends(SleepFn guest_sleep) {
    // The limit set to 300 ms for the test; the stall stays open across the whole Sleep. Without the limit this
    // Sleep would not return and the test would time out.
    auto& gate = rt::host_stall::gate();
    gate.set_limit(ms(300));
    const Ledger::Id stuck = gate.begin();
    const auto start = std::chrono::steady_clock::now();
    guest_sleep(1000);
    const long long took = elapsed_ms(start);
    check(took >= 1250 && took < 3000, "under a stall that never ends a Sleep(1000) returns a second after the limit");
    const auto again = std::chrono::steady_clock::now();
    guest_sleep(1000);
    check(elapsed_ms(again) < 1900, "and the next Sleep is not held by the overdue stall");
    gate.end(stuck);
    gate.set_limit(rt::host_stall::kStallLimit);
}

}  // namespace

int main() {
    ledger_checks();
    limit_checks();
    owed_checks();
    virtual_sleep_checks();
    virtual_hang_check_checks();
    sleep_binding_checks();
    if (const SleepFn guest_sleep = served_sleep("api-ms-win-core-synch-l1-2-0.dll")) {
        sleep_without_a_stall(guest_sleep);
        poll_under_a_stall(guest_sleep);
        sleep_held_by_a_stall(guest_sleep);
        compiles_that_take_turns(guest_sleep);
        sleep_under_a_stall_that_never_ends(guest_sleep);
    }
    std::printf("%d checks, %d failed\n", checks, failures);
    return failures ? 1 : 0;
}
