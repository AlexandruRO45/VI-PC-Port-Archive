#include <windows.h>
#include <ApiRefState.h>

#include <cstdio>
#include <atomic>
#include <thread>

static int failures = 0;

void Check(bool condition, const char* name)
{
    std::printf("%s %s\n",
        condition ? "PASS" : "FAIL", name);

    if (!condition)
        ++failures;
}

// Blocks until another thread stores at least `value` in `step` and notifies it. The hand-offs between the
// threads used to be `while (!flag) std::this_thread::yield();`, which costs a scheduler quantum or more per
// round when every core is busy: the test took 0.16 s idle and 221 s with 32 busy processes on 16 cores, against
// ctest's 10 s. A thread released from a wait is boosted on Windows; one spinning on yield is not.
void WaitUntil(const std::atomic<int>& step, int value)
{
    for (int seen = step.load(std::memory_order_acquire); seen < value; seen = step.load(std::memory_order_acquire))
        step.wait(seen, std::memory_order_acquire);
}

// The race's start: how long the contender spins on `go` before it blocks, and how many different Release
// delays the rounds cycle through (in YieldProcessor pauses).
constexpr int kStartSpin = 1 << 14;
constexpr int kReleaseDelaySteps = 8;

struct RaceContext
{
    ApiRefState state;
    HANDLE rundownEntered = nullptr;
    HANDLE allowRundownExit = nullptr;
};

DWORD WINAPI RundownThread(void* arg)
{
    auto& ctx = *static_cast<RaceContext*>(arg);

    if (ctx.state.AddRef() != 0)
        return 1;

    auto finalRelease = ctx.state.Release();

    if (!finalRelease.startRundown ||
        finalRelease.remaining != 0)
        return 2;

    // Simulate an internal AddRef during RundownObject().
    if (ctx.state.AddRef() != 0)
        return 3;

    SetEvent(ctx.rundownEntered);

    if (WaitForSingleObject(
        ctx.allowRundownExit, 3000) != WAIT_OBJECT_0)
        return 4;

    auto internalRelease = ctx.state.Release();

    if (internalRelease.startRundown ||
        internalRelease.remaining != 0)
        return 5;

    return 0;
}

void CheckPinWinsBeforeFinalRelease()
{
    constexpr int rounds = 200;
    int failuresLocal = 0;

    // One contender thread for every round, and each hand-off blocks in atomic::wait (see WaitUntil and
    // CheckFinalReleaseRace for why neither a yield spin nor a new thread per round holds up on a busy machine).
    // Each step of a round is numbered by the round, so no flag needs resetting between rounds.
    ApiRefState* state = nullptr;
    bool acquired = false;
    ApiRefState::ReleaseResult pinRelease{ 0, false };
    std::atomic<int> started{ 0 }, pinAttempted{ 0 }, pinReleaseAllowed{ 0 }, finished{ 0 };

    std::thread contender([&]
    {
        for (int round = 1; round <= rounds; ++round)
        {
            WaitUntil(started, round);

            acquired = state->TryAddRef();

            pinAttempted.store(round, std::memory_order_release);
            pinAttempted.notify_one();

            if (acquired)
            {
                WaitUntil(pinReleaseAllowed, round);

                pinRelease = state->Release();
            }

            finished.store(round, std::memory_order_release);
            finished.notify_one();
        }
    });

    for (int round = 1; round <= rounds; ++round)
    {
        ApiRefState roundState;
        roundState.AddRef();
        state = &roundState;
        pinRelease = { 0, false };

        started.store(round, std::memory_order_release);
        started.notify_one();
        WaitUntil(pinAttempted, round);

        const auto ownerRelease = roundState.Release();

        bool valid =
            acquired &&
            ownerRelease.remaining == 1 &&
            !ownerRelease.startRundown &&
            roundState.RefCount() == 1 &&
            !roundState.IsDying();

        pinReleaseAllowed.store(round, std::memory_order_release);
        pinReleaseAllowed.notify_one();
        WaitUntil(finished, round);

        valid =
            valid &&
            pinRelease.remaining == 0 &&
            pinRelease.startRundown &&
            roundState.RefCount() == 0 &&
            roundState.IsDying() &&
            !roundState.TryAddRef();

        if (!valid)
        {
            ++failuresLocal;
        }
    }

    contender.join();

    Check(failuresLocal == 0,
          "pin acquired before final owner Release stays valid");

    std::printf(
        "forced pin-first rounds: %d, failures: %d\n",
        rounds, failuresLocal);
}
void CheckFinalReleaseRace()
{
    constexpr int rounds = 1000;
    int badRundown = 0;
    int badFinalState = 0;
    int successfulPins = 0;

    // One contender thread for every round. A new thread per round waits for a scheduler quantum before it
    // first runs when the cores are busy (no wake-up boost, unlike a thread released from a wait), which
    // alone kept 1000 rounds past ctest's timeout. Each step of a round is numbered by the round, so no flag
    // needs resetting between rounds. The contender announces it is ready for the next round as soon as it has
    // finished this one, and reads the round's state only after `go`, so a round costs the owner one wait
    // (`finished`) and the contender one (`go`): with one CPU each wait is a context switch.
    ApiRefState* state = nullptr;
    std::atomic<int> rundownCount{ 0 };
    bool pinned = false;
    std::atomic<int> ready{ 0 }, go{ 0 }, finished{ 0 };

    std::thread contender([&]
    {
        for (int round = 1; round <= rounds; ++round)
        {
            ready.store(round, std::memory_order_release);
            ready.notify_one();

            // Spin briefly so both sides leave the start line together, then block if the owner thread
            // isn't running (a busy machine).
            for (int spin = 0; spin < kStartSpin && go.load(std::memory_order_acquire) < round; ++spin)
                YieldProcessor();
            WaitUntil(go, round);

            pinned = state->TryAddRef();

            if (pinned && state->Release().startRundown)
                rundownCount.fetch_add(1);

            finished.store(round, std::memory_order_release);
            finished.notify_one();
        }
    });

    for (int i = 0; i < rounds; ++i)
    {
        const int round = i + 1;
        ApiRefState roundState;
        roundState.AddRef();
        state = &roundState;
        rundownCount.store(0);

        WaitUntil(ready, round);

        go.store(round, std::memory_order_release);
        go.notify_one();

        // The contender has to see `go` before its TryAddRef, so a Release straight after the store always
        // wins (every round was a rejected pin). Delaying it by a different number of pauses each round
        // sweeps the window, so both orders occur: the contender's pin first, and the final Release first.
        for (int pause = 0; pause < i % kReleaseDelaySteps; ++pause)
            YieldProcessor();

        if (roundState.Release().startRundown)
            rundownCount.fetch_add(1);

        WaitUntil(finished, round);

        if (pinned)
            ++successfulPins;

        if (rundownCount.load() != 1)
            ++badRundown;

        bool unexpectedPin = roundState.TryAddRef();

        if (unexpectedPin)
            roundState.Release();

        if (roundState.RefCount() != 0 ||
            !roundState.IsDying() ||
            unexpectedPin)
        {
            ++badFinalState;
        }
    }

    contender.join();

    Check(badRundown == 0,
          "racing final Release starts rundown exactly once");

    Check(badFinalState == 0,
          "racing final Release preserves terminal state");

    std::printf(
        "race rounds: %d, successful pins: %d, rejected pins: %d\n",
        rounds, successfulPins, rounds - successfulPins);
}
int main()
{
    std::setvbuf(stdout, nullptr, _IONBF, 0);

    std::puts("api_ref_state_test");

    ApiRefState state;

    Check(!state.TryAddRef(),
          "cannot pin zero-ref object");

    Check(state.AddRef() == 0,
          "initial AddRef");

    Check(state.TryAddRef(),
          "can pin live object");

    Check(state.RefCount() == 2,
          "two references held");

    auto first = state.Release();

    Check(first.remaining == 1 &&
          !first.startRundown,
          "non-final Release");

    auto last = state.Release();

    Check(last.remaining == 0 &&
          last.startRundown,
          "final Release starts rundown");

    Check(state.IsDying(),
          "dying marker is permanent");

    Check(!state.TryAddRef(),
          "cannot pin after final Release");

    RaceContext race;

    race.rundownEntered =
        CreateEventW(nullptr, TRUE, FALSE, nullptr);

    race.allowRundownExit =
        CreateEventW(nullptr, TRUE, FALSE, nullptr);

    if (!race.rundownEntered || !race.allowRundownExit)
        return 2;

    HANDLE thread = CreateThread(
        nullptr, 0, RundownThread, &race, 0, nullptr);

    if (!thread)
        return 3;

    DWORD entered = WaitForSingleObject(
        race.rundownEntered, 3000);

    Check(entered == WAIT_OBJECT_0,
          "rundown thread reached checkpoint");

    if (entered == WAIT_OBJECT_0)
    {
        Check(race.state.IsDying(),
              "dying bit survives internal AddRef");

        Check(race.state.RefCount() == 1,
              "temporary internal reference exists");

        bool rejected = true;

        for (int i = 0; i < 10000; ++i)
        {
            if (race.state.TryAddRef())
            {
                rejected = false;
                race.state.Release();
                break;
            }
        }

        Check(rejected,
              "concurrent pins rejected during rundown");
    }

    SetEvent(race.allowRundownExit);

    DWORD joined = WaitForSingleObject(thread, 3000);

    Check(joined == WAIT_OBJECT_0,
          "rundown thread completed");

    if (joined == WAIT_OBJECT_0)
    {
        DWORD exitCode = 1;
        GetExitCodeThread(thread, &exitCode);

        Check(exitCode == 0,
              "internal Release did not restart rundown");
    }

    CloseHandle(thread);
    CloseHandle(race.rundownEntered);
    CloseHandle(race.allowRundownExit);

    CheckPinWinsBeforeFinalRelease();
    CheckFinalReleaseRace();

    std::printf("RESULT: %s (%d failures)\n",
        failures == 0 ? "PASS" : "FAIL", failures);

    return failures == 0 ? 0 : 1;
}
