// Hang diagnosis: the watchdog and the all-thread dump (LEONIDA_WATCHDOG, LEONIDA_THREAD_DUMP; docs/boot.md).
//
// progress() is called when the game does something that proves it is alive: a frame presented. The watchdog thread
// wakes four times a second; when no progress has come for the timeout it dumps every thread, then again after each
// further timeout, up to a count, then stays quiet until progress() is called again. Each dump says, per thread: its
// id and name, running or waiting (measured: the thread's CPU time over a 200 ms window), where it started, the image
// code that called the import it is in, and its frames. Running means the thread spins; waiting means it is blocked on
// something, and the import named is what it waits in.
//
// A thread is stopped only to copy its registers and stack (boot_diag.h, StackSample); formatting happens after it
// runs again, so a thread stopped inside the heap cannot hang the dump.
//
// The idea (a stack dump of every thread three times, five seconds apart, when no frame is presented) comes from
// xvalkyrie, src/runtime/modules/watchdog.cpp at commit 53af7df77c10 (GPL-3.0); this is a new implementation.
#pragma once

#include <chrono>
#include <memory>
#include <thread>

#include "boot_diag.h"

namespace rt::boot_diag {

inline constexpr DWORD kCpuWindowMs = 200;

namespace detail {
inline FILETIME zero_time() { return FILETIME{}; }
inline uint64_t cpu_100ns(HANDLE thread) {
    FILETIME create{}, exit_time{}, kernel{}, user{};
    if (!GetThreadTimes(thread, &create, &exit_time, &kernel, &user)) return 0;
    auto u64 = [](const FILETIME& f) { return (static_cast<uint64_t>(f.dwHighDateTime) << 32) | f.dwLowDateTime; };
    return u64(kernel) + u64(user);
}

struct ThreadEntry {
    DWORD tid = 0;
    HANDLE handle = nullptr;
    uint64_t cpu_before = 0, cpu_after = 0;
    std::unique_ptr<StackSample> sample;
};

inline std::vector<DWORD> thread_ids() {
    std::vector<DWORD> ids;
    HANDLE snap = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
    if (snap == INVALID_HANDLE_VALUE) return ids;
    THREADENTRY32 te{};
    te.dwSize = sizeof te;
    const DWORD pid = GetCurrentProcessId();
    for (BOOL ok = Thread32First(snap, &te); ok; ok = Thread32Next(snap, &te))
        if (te.th32OwnerProcessID == pid) ids.push_back(te.th32ThreadID);
    CloseHandle(snap);
    return ids;
}
}  // namespace detail

// "running" (spinning or computing), "waiting" (blocked), or "mixed", from the CPU share of the sample window.
inline const char* cpu_state(double percent) {
    return percent >= kRunningCpuPercent ? "RUNNING" : percent < kWaitingCpuPercent ? "WAITING" : "mixed";
}

// Dumps every thread but the caller's. 'why' heads the report ("no progress for 5.0 s (dump 1 of 3)").
// 'own_context', when given, is the calling thread's own context (a fault): that thread is described from it instead of
// being skipped.
inline void dump_all_threads(const std::string& why, const CONTEXT* own_context = nullptr) {
    const DWORD self = GetCurrentThreadId();
    std::vector<detail::ThreadEntry> entries;
    for (DWORD tid : detail::thread_ids()) {
        if (tid == self) continue;
        detail::ThreadEntry e;
        e.tid = tid;
        e.handle = OpenThread(THREAD_QUERY_INFORMATION | THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT, FALSE, tid);
        if (!e.handle) continue;
        e.sample = std::make_unique<StackSample>();
        e.sample->tid = tid;
        e.cpu_before = detail::cpu_100ns(e.handle);
        entries.push_back(std::move(e));
    }
    Sleep(kCpuWindowMs);
    for (auto& e : entries) e.cpu_after = detail::cpu_100ns(e.handle);

    for (auto& e : entries) {  // stopped one at a time, and only to copy
        if (SuspendThread(e.handle) == static_cast<DWORD>(-1)) continue;
        CONTEXT c{};
        c.ContextFlags = CONTEXT_CONTROL | CONTEXT_INTEGER;
        if (GetThreadContext(e.handle, &c)) sample_context(c, *e.sample);
        ResumeThread(e.handle);
    }

    std::vector<std::string> lines;
    lines.push_back("thread dump: " + why + format("; %zu thread(s) besides the reporting one", entries.size()));
    for (auto& e : entries) {
        const double pct = static_cast<double>(e.cpu_after - e.cpu_before) / (static_cast<double>(kCpuWindowMs) * 10000.0) * 100.0;
        const std::string start = thread_start_text(e.handle);
        lines.push_back(format("%s %s (cpu %.0f%% of %lu ms)%s%s", thread_label(e.tid).c_str(), cpu_state(pct), pct,
                               kCpuWindowMs, start.empty() ? "" : ", started at ", start.c_str()));
        if (!e.sample->ok)
            lines.push_back("  could not read this thread's registers");
        else
            describe_stack(*e.sample, lines);
        CloseHandle(e.handle);
    }
    if (own_context) {
        auto s = std::make_unique<StackSample>();
        s->tid = self;
        sample_context(*own_context, *s);
        lines.push_back(thread_label(self) + " (the reporting thread)");
        describe_stack(*s, lines);
    }
    report("thread_dump", lines);
}

// ---- the watchdog ---------------------------------------------------------------------------------------------------

struct WatchdogConfig {
    double timeout_seconds = 0;  // 0: no stall dumps
    int dumps_per_stall = 3;
    uint64_t dump_at_progress = 0;  // 0: none; else one dump when progress() has been called this many times
};

namespace detail {
struct Watchdog {
    std::atomic<uint64_t> progress{0};
    std::atomic<bool> stop{false};
    std::atomic<bool> dump_requested{false};
    std::atomic<int> dump_count{0};  // dumps written (tests wait on it)
    WatchdogConfig config;
    std::thread thread;
    HANDLE wake = nullptr;
};
inline Watchdog& wd() {
    static auto* w = new Watchdog;  // immortal: progress() is called from present paths until the process ends
    return *w;
}
inline std::atomic<bool> wd_running{false};

inline void watchdog_main() {
    Watchdog& w = wd();
    auto last_progress_value = w.progress.load();
    auto stall_start = std::chrono::steady_clock::now();
    auto last_dump = stall_start;
    int dumps = 0;
    while (!w.stop.load()) {
        WaitForSingleObject(w.wake, 250);
        if (w.stop.load()) break;
        const auto now = std::chrono::steady_clock::now();
        if (w.dump_requested.exchange(false)) {
            dump_all_threads(format("frame %llu reached (LEONIDA_THREAD_DUMP)", static_cast<unsigned long long>(w.config.dump_at_progress)));
            ++w.dump_count;
        }
        const uint64_t p = w.progress.load();
        if (p != last_progress_value) {
            last_progress_value = p;
            stall_start = last_dump = now;
            dumps = 0;
            continue;
        }
        if (w.config.timeout_seconds <= 0 || dumps >= w.config.dumps_per_stall) continue;
        const double stalled = std::chrono::duration<double>(now - stall_start).count();
        const double since_dump = std::chrono::duration<double>(now - last_dump).count();
        if (stalled < w.config.timeout_seconds || (dumps > 0 && since_dump < w.config.timeout_seconds)) continue;
        ++dumps;
        dump_all_threads(format("no progress for %.1f s (dump %d of %d)", stalled, dumps, w.config.dumps_per_stall));
        ++w.dump_count;
        last_dump = std::chrono::steady_clock::now();
    }
}
}  // namespace detail

// Called when the game proves it is alive (a frame presented). One relaxed add; the watchdog reads it.
inline void progress() {
    detail::Watchdog& w = detail::wd();
    const uint64_t n = w.progress.fetch_add(1, std::memory_order_relaxed) + 1;
    if (w.config.dump_at_progress && n == w.config.dump_at_progress && detail::wd_running.load(std::memory_order_acquire)) {
        w.dump_requested.store(true);
        SetEvent(w.wake);
    }
}

inline bool start_watchdog(const WatchdogConfig& config) {
    detail::Watchdog& w = detail::wd();
    if (detail::wd_running.load()) return false;
    if (config.timeout_seconds <= 0 && !config.dump_at_progress) return false;
    w.config = config;
    w.stop = false;
    w.dump_requested = false;
    w.dump_count = 0;
    w.progress = 0;
    w.wake = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    w.thread = std::thread(detail::watchdog_main);
    detail::wd_running.store(true, std::memory_order_release);
    return true;
}

inline int watchdog_dump_count() { return detail::wd().dump_count.load(); }

inline void stop_watchdog() {
    detail::Watchdog& w = detail::wd();
    if (!detail::wd_running.exchange(false)) return;
    w.stop = true;
    SetEvent(w.wake);
    if (w.thread.joinable()) w.thread.join();
    CloseHandle(w.wake);
    w.wake = nullptr;
}

}  // namespace rt::boot_diag
