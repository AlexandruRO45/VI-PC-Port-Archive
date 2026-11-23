// Host-only stalls: time a title thread spends inside the layer on work the console never does, and what a guest
// Sleep does about it.
//
// The console creates a pipeline from native code in well under a millisecond. Here the same call compiles
// translated SPIR-V in the GPU driver, which takes 25 to 60 s for the 52 programs of XboxBvhBuilder.dll on a cold
// driver cache (docs/retail_boot_chain.md, "The startup hang check"). The retail title runs a hang check next to it:
// a thread that loops on Sleep(1000) and ends the process after 30 loops in which the main thread reported no
// progress. The same loop restarts its count when one Sleep returns more than 2 s late by QueryPerformanceCounter,
// which is what a suspended console looks like to it.
//
// So the layer brackets such work with a Scope, and the Sleep shim (shims/host_stall.cpp) makes a long guest Sleep
// last its length in time in which no stall was open: Sleep(n) returns once n milliseconds have passed outside
// stalls since it was called. Three limits keep that from turning into a hang of its own:
//
//   - Only a Sleep of kHeldSleepMs or more is held. A shorter one is a poll: its caller waits for something another
//     thread does, possibly the thread that holds the stall, so it stays the host's Sleep.
//   - A stall counts for at most kStallLimit. Work that has not come back by then is no longer treated as a compile:
//     held Sleeps return, the time counts as the title's again, and the shim says so once. A host call that never
//     returns therefore delays the title's hang check by kStallLimit and does not switch it off.
//   - A Sleep that no stall overlaps is the host's Sleep, unchanged, whenever the host wakes it.
//
// Nothing is skipped and no check of the title is switched off; the title sees a late wake-up and applies its own
// rule. QueryPerformanceCounter and the other waits stay the host's.
//
// Header-only and free of Windows headers, so the layer, the shim and the unit test share one definition.
#pragma once

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <vector>

namespace rt::host_stall {

using Nanoseconds = std::chrono::nanoseconds;

// The shortest guest Sleep that is held: the round of the title's hang check.
inline constexpr std::uint32_t kHeldSleepMs = 1000;
// How long one stall may count. One stall is one pipeline; the 52 of the cold run took 55.7 s together.
inline constexpr Nanoseconds kStallLimit = std::chrono::seconds(60);

// Whether a guest Sleep of this length is held. A yield and an endless sleep have no duration to stretch.
inline bool sleep_is_held(std::uint32_t milliseconds) {
    return milliseconds >= kHeldSleepMs && milliseconds != 0xFFFFFFFFu;
}

// The accounting, with the time passed in: how long at least one stall has been open and within its limit, up to
// `now`. Stalls nest and overlap (two threads compiling at once count once). `now` never goes backwards between
// calls.
class Ledger {
public:
    using Id = std::uint64_t;

    // Where a sleep started: the time, and stalled() at that time, read together.
    struct Mark {
        Nanoseconds at{0};
        Nanoseconds stalled{0};
    };

    explicit Ledger(Nanoseconds limit = kStallLimit) : limit_(limit) {}

    Nanoseconds limit() const { return limit_; }
    void set_limit(Nanoseconds limit) { limit_ = limit; }  // with no stall open

    Id begin(Nanoseconds now) {
        // A run whose stalls have all passed their limit ended there; this one starts a new run.
        if (running_ && now >= newest_limit()) close_run(newest_limit());
        open_.push_back({++last_id_, now});
        if (!running_) {
            running_ = true;
            run_from_ = now;
        }
        return last_id_;
    }

    void end(Id id, Nanoseconds now) {
        const auto it = std::find_if(open_.begin(), open_.end(), [id](const Open& o) { return o.id == id; });
        if (it == open_.end()) return;  // an end without a begin changes nothing
        const Nanoseconds counted_to = running_ ? std::min(now, newest_limit()) : Nanoseconds{0};
        const bool was_newest = it + 1 == open_.end();
        open_.erase(it);
        if (!running_) return;
        // The run goes on while an open stall is within its limit. Only the newest decides: it started last.
        if (open_.empty() || (was_newest && counted_to >= newest_limit())) close_run(counted_to);
    }

    // A stall is open and within its limit.
    bool open(Nanoseconds now) const { return running_ && now < newest_limit(); }
    // A stall is open past its limit: host-only work that did not come back.
    bool overdue(Nanoseconds now) const { return !open_.empty() && now >= open_.front().at + limit_; }
    Nanoseconds stalled(Nanoseconds now) const {
        return running_ ? closed_ + (std::min(now, newest_limit()) - run_from_) : closed_;
    }
    Mark mark(Nanoseconds now) const { return {now, stalled(now)}; }

    // What a sleep of `requested` that started at `start` still owes at `now`, after the host's sleep is over: the
    // part of `requested` that has not yet passed outside stalls. Zero or less: it may return. A sleep no stall
    // overlapped owes nothing, whenever the host woke it.
    Nanoseconds owed(const Mark& start, Nanoseconds requested, Nanoseconds now) const {
        const Nanoseconds in_stalls = stalled(now) - start.stalled;
        if (in_stalls == Nanoseconds{0}) return Nanoseconds{0};
        return requested - ((now - start.at) - in_stalls);
    }
    // When a sleep that owes `owed` at `now` is looked at again if no stall begins or ends first: when the open
    // stall reaches its limit, or when the rest has passed.
    Nanoseconds look_again(Nanoseconds owed, Nanoseconds now) const { return open(now) ? newest_limit() : now + owed; }

private:
    struct Open {
        Id id;
        Nanoseconds at;
    };

    Nanoseconds newest_limit() const { return open_.back().at + limit_; }  // with a stall open
    void close_run(Nanoseconds at) {
        closed_ += at - run_from_;
        running_ = false;
    }

    Nanoseconds limit_;
    std::vector<Open> open_;  // in the order they began
    Id last_id_ = 0;
    bool running_ = false;  // a run of counted time is going on, from run_from_ to the newest stall's end or limit
    Nanoseconds run_from_{0};
    Nanoseconds closed_{0};
};

// The process-wide ledger behind a lock, on the steady clock.
class Gate {
public:
    struct Held {
        Nanoseconds held{0};   // how long hold() blocked
        bool overdue = false;  // a stall was open past its limit meanwhile
        Nanoseconds limit{0};  // that limit
    };

    Ledger::Id begin() {
        std::lock_guard<std::mutex> lock(mutex_);
        return ledger_.begin(now());
    }
    void end(Ledger::Id id) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            ledger_.end(id, now());
        }
        changed_.notify_all();
    }

    // Read before a sleep starts; passed to hold() when the host's sleep is over.
    Ledger::Mark mark() {
        std::lock_guard<std::mutex> lock(mutex_);
        return ledger_.mark(now());
    }

    // Called when the host's sleep of `requested` has returned. Blocks until `requested` has passed outside stalls
    // since `start`; returns at once when no stall overlapped the sleep.
    Held hold(const Ledger::Mark& start, Nanoseconds requested) {
        std::unique_lock<std::mutex> lock(mutex_);
        const Nanoseconds woke = now();
        Held result;
        result.limit = ledger_.limit();
        for (;;) {
            const Nanoseconds current = now();
            result.overdue = result.overdue || ledger_.overdue(current);
            const Nanoseconds owed = ledger_.owed(start, requested, current);
            if (owed <= Nanoseconds{0}) {
                result.held = current - woke;
                return result;
            }
            // A stall that begins or ends meanwhile is seen by the next round.
            changed_.wait_for(lock, ledger_.look_again(owed, current) - current);
        }
    }

    // For tests: how long one stall may count. Call with no stall open.
    void set_limit(Nanoseconds limit) {
        std::lock_guard<std::mutex> lock(mutex_);
        ledger_.set_limit(limit);
    }

    static Nanoseconds now() {
        return std::chrono::duration_cast<Nanoseconds>(std::chrono::steady_clock::now().time_since_epoch());
    }

private:
    std::mutex mutex_;
    std::condition_variable changed_;
    Ledger ledger_;
};

inline Gate& gate() {
    static Gate instance;
    return instance;
}

// Brackets host-only work on behalf of a title call that returns at once on the console. Construct it right before
// the host call, with the call's arguments already evaluated: what runs inside counts as a stall.
class Scope {
public:
    Scope() : id_(gate().begin()) {}
    ~Scope() { gate().end(id_); }
    Scope(const Scope&) = delete;
    Scope& operator=(const Scope&) = delete;

private:
    Ledger::Id id_;
};

}  // namespace rt::host_stall
