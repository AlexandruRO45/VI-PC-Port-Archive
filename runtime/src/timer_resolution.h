#pragma once

#ifdef _WIN32
#include <windows.h>
#include <mmsystem.h>
#include <cstdlib>
#include <cstring>

namespace leonida {

// Scoped timer resolution for Windows runtime execution.
// By default, Windows has a system timer interrupt period of ~15.625 ms (64 Hz).
// This causes std::condition_variable::wait_until (used by FrameClock / SteadyFrameClock in
// frame_pacer.h), Sleep(), and thread synchronization to overshoot presentation deadlines
// by up to 15.6 ms, producing frame pacing jitter and missed presentation intervals.
// Requesting a 1 ms period via timeBeginPeriod(1) ensures 1.0 ms granularity.
class ScopedTimerResolution {
public:
    explicit ScopedTimerResolution(UINT period_ms = 1) : period_ms_(period_ms) {
        winmm_ = LoadLibraryW(L"winmm.dll");
        if (winmm_) {
            time_begin_ = reinterpret_cast<PfnTimePeriod>(GetProcAddress(winmm_, "timeBeginPeriod"));
            time_end_ = reinterpret_cast<PfnTimePeriod>(GetProcAddress(winmm_, "timeEndPeriod"));
            if (time_begin_ && time_begin_(period_ms_) == TIMERR_NOERROR) {
                active_ = true;
            }
        }
    }

    ~ScopedTimerResolution() {
        if (active_ && time_end_) {
            time_end_(period_ms_);
        }
        if (winmm_) {
            FreeLibrary(winmm_);
        }
    }

    ScopedTimerResolution(const ScopedTimerResolution&) = delete;
    ScopedTimerResolution& operator=(const ScopedTimerResolution&) = delete;

    bool active() const { return active_; }
    UINT period_ms() const { return period_ms_; }

private:
    using PfnTimePeriod = MMRESULT(WINAPI*)(UINT);
    HMODULE winmm_ = nullptr;
    PfnTimePeriod time_begin_ = nullptr;
    PfnTimePeriod time_end_ = nullptr;
    UINT period_ms_ = 1;
    bool active_ = false;
};

// Optionally configures process priority for runtime execution.
// Fixed-clock CPUs (such as Intel CC150 @ 3.50 GHz without Turbo Boost) lack high dynamic
// single-thread frequency scaling and are vulnerable to preemption by background OS tasks
// (indexing, updates, telemetry). Elevating process priority to ABOVE_NORMAL prevents OS
// worker threads from starving the main title/render thread while avoiding real-time lockups.
// Configurable via LEONIDA_PROCESS_PRIORITY ("normal", "above_normal", "high").
inline DWORD configure_process_priority() {
    char env[32] = {};
    std::size_t env_len = 0;
    const errno_t err = getenv_s(&env_len, env, "LEONIDA_PROCESS_PRIORITY");
    DWORD prio = ABOVE_NORMAL_PRIORITY_CLASS;
    if (err == 0 && env_len > 1) {
        if (!_stricmp(env, "normal")) prio = NORMAL_PRIORITY_CLASS;
        else if (!_stricmp(env, "high")) prio = HIGH_PRIORITY_CLASS;
        else if (!_stricmp(env, "above_normal")) prio = ABOVE_NORMAL_PRIORITY_CLASS;
        else if (!_stricmp(env, "idle") || !_stricmp(env, "below_normal")) prio = BELOW_NORMAL_PRIORITY_CLASS;
    }
    SetPriorityClass(GetCurrentProcess(), prio);
    return prio;
}

}  // namespace leonida
#else
namespace leonida {
class ScopedTimerResolution {
public:
    explicit ScopedTimerResolution(unsigned int = 1) {}
    bool active() const { return true; }
    unsigned int period_ms() const { return 1; }
};
inline unsigned long configure_process_priority() { return 0; }
}  // namespace leonida
#endif
