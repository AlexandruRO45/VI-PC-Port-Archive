// Records initialization calls from tls_route images. It is loaded before and unloaded after
// the images it observes, so detach calls remain visible after the image is gone. Each call
// records the calling thread, so thread notifications can be attributed to one thread.
#include <windows.h>

#include <cstddef>
#include <cstdint>

struct Call {
    std::uint32_t caller;
    std::uint32_t reason;
    std::uint64_t module;
    std::uint64_t reserved;
    std::uint32_t thread;
    std::uint32_t padding;
};

static Call calls[64];
static volatile LONG count;

// One-shot hook run after the next recorded DLL_THREAD_ATTACH, on that thread. Tests use it to call
// pe_map APIs from inside a guest TLS callback without any Python on that path.
using Hook = void (*)(void*);
static void* volatile pending_hook;
static void* volatile pending_argument;
static volatile DWORD pending_thread;

extern "C" __declspec(dllexport) void sink_record(void* module, std::uint32_t reason, void* reserved,
                                                  std::uint32_t caller) {
    const LONG index = InterlockedIncrement(&count) - 1;
    if (index < static_cast<LONG>(sizeof(calls) / sizeof(calls[0])))
        calls[index] = {caller, reason, reinterpret_cast<std::uint64_t>(module),
                        reinterpret_cast<std::uint64_t>(reserved), GetCurrentThreadId(), 0};
    // Only the armed thread consumes the hook, so an unrelated thread (a pool worker) cannot take it.
    if (reason == DLL_THREAD_ATTACH && pending_thread == GetCurrentThreadId()) {
        if (auto hook = reinterpret_cast<Hook>(InterlockedExchangePointer(&pending_hook, nullptr))) {
            hook(pending_argument);  // written before the hook was published
        }
    }
}

extern "C" __declspec(dllexport) void sink_on_thread_attach(Hook hook, void* argument, DWORD thread) {
    pending_argument = argument;
    pending_thread = thread;
    InterlockedExchangePointer(&pending_hook, reinterpret_cast<void*>(hook));
}

extern "C" __declspec(dllexport) std::size_t sink_read(Call* out, std::size_t capacity) {
    constexpr std::size_t slots = sizeof(calls) / sizeof(calls[0]);
    const std::size_t recorded = static_cast<std::size_t>(count);
    const std::size_t stored = recorded < slots ? recorded : slots;
    for (std::size_t i = 0; i < stored && i < capacity; ++i) out[i] = calls[i];
    return stored;
}

extern "C" __declspec(dllexport) void sink_clear() { InterlockedExchange(&count, 0); }

// Releases one image from `threads` native threads at once. Each thread spins until every thread is
// ready, then calls release(image) and stores the result. Returns only after every thread has fully
// exited, including its DLL_THREAD_DETACH, so no notification of the race is still in flight.
using Release = int (*)(void*);
struct Race {
    Release release;
    void* image;
    volatile LONG ready;
    volatile LONG go;
    int* results;
    LONG index;
};

static DWORD WINAPI race_thread(void* parameter) {
    auto* race = static_cast<Race*>(parameter);
    const LONG slot = InterlockedIncrement(&race->index) - 1;
    InterlockedIncrement(&race->ready);
    while (!InterlockedCompareExchange(&race->go, 0, 0)) YieldProcessor();
    race->results[slot] = race->release(race->image);
    return 0;
}

extern "C" __declspec(dllexport) int sink_race_release(Release release, void* image, int threads,
                                                       int* results) {
    if (threads <= 0 || threads > MAXIMUM_WAIT_OBJECTS) return 0;
    Race race{release, image, 0, 0, results, 0};
    HANDLE handles[MAXIMUM_WAIT_OBJECTS] = {};
    int started = 0;
    for (; started < threads; ++started) {
        handles[started] = CreateThread(nullptr, 0, race_thread, &race, 0, nullptr);
        if (!handles[started]) break;
    }
    while (InterlockedCompareExchange(&race.ready, 0, 0) < started) YieldProcessor();
    InterlockedExchange(&race.go, 1);
    WaitForMultipleObjects(static_cast<DWORD>(started), handles, TRUE, INFINITE);
    for (int i = 0; i < started; ++i) CloseHandle(handles[i]);
    return started;
}
