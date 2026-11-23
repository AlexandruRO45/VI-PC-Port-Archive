// The Windows thread pool (work, timers, waits) on guest threads: every callback runs on a thread of its own, made
// with spawn_thread so it has a TEB and the DLLs see it attach.
// ponytail: one thread per callback and per armed timer or wait, no pooling; a wait polls its handle every 10 ms so a
// re-armed or closed wait never swallows a signal meant for someone else.
#include "host.h"

#include <chrono>
#include <condition_variable>
#include <mutex>

namespace host {
namespace {

using HANDLE = void*;
using Callback = void(MSABI*)(void* instance, void* ctx, void* object);
using WaitCallback = void(MSABI*)(void* instance, void* ctx, void* object, uint32_t result);
using SimpleCallback = void(MSABI*)(void* instance, void* ctx);

struct PoolObject {
    PoolObject(void* f, void* c) : fn(f), ctx(c) {}
    void* fn;
    void* ctx;
    std::mutex m;
    std::condition_variable cv;
    int running = 0;      // callbacks started and not finished
    uint64_t armed = 0;   // generation: SetThreadpoolTimer/Wait and cancels bump it
    int64_t due = 0;      // timer: FILETIME ticks (negative: relative)
    uint32_t period = 0;  // timer: ms
    HANDLE handle = nullptr;
    uint32_t timeout = 0;  // wait: ms
};
struct Job {
    PoolObject* o;
    uint64_t generation;
    enum Kind { Work, Timer, Wait, Simple } kind;
};
char g_instance;  // the callback instance: nothing calls back into it

void finish(PoolObject* o) {
    std::lock_guard<std::mutex> g(o->m);
    --o->running;
    o->cv.notify_all();
}
bool still_armed(PoolObject* o, uint64_t generation) {
    std::lock_guard<std::mutex> g(o->m);
    return o->armed == generation;
}
uint64_t filetime_now() {
    return clock_gettime_nsec_np(CLOCK_REALTIME) / 100 + 116444736000000000ull;
}

MSABI uint32_t pool_thread(void* p) {
    HostGs gs;
    Job job = *static_cast<Job*>(p);
    delete static_cast<Job*>(p);
    PoolObject* o = job.o;
    if (job.kind == Job::Work || job.kind == Job::Simple) {
        if (job.kind == Job::Work) call_guest(reinterpret_cast<Callback>(o->fn), (void*)&g_instance, o->ctx, (void*)o);
        else call_guest(reinterpret_cast<SimpleCallback>(o->fn), (void*)&g_instance, o->ctx), delete o;
        if (job.kind == Job::Work) finish(o);
        return 0;
    }
    if (job.kind == Job::Wait) {
        auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(o->timeout);
        uint32_t r;
        for (;;) {
            if (!still_armed(o, job.generation)) return finish(o), 0;
            if ((r = wait_object(o->handle, 10)) != 0x102) break;
            if (o->timeout != 0xFFFFFFFFu && std::chrono::steady_clock::now() >= deadline) break;
        }
        if (still_armed(o, job.generation))
            call_guest(reinterpret_cast<WaitCallback>(o->fn), (void*)&g_instance, o->ctx, (void*)o, r);
        return finish(o), 0;
    }
    std::unique_lock<std::mutex> g(o->m);  // a timer
    int64_t due = o->due;
    auto when = std::chrono::steady_clock::now() +
                std::chrono::microseconds((due < 0 ? -due : due > int64_t(filetime_now()) ? due - int64_t(filetime_now()) : 0) / 10);
    for (;;) {
        if (o->cv.wait_until(g, when, [&] { return o->armed != job.generation; })) break;
        g.unlock();
        call_guest(reinterpret_cast<Callback>(o->fn), (void*)&g_instance, o->ctx, (void*)o);
        g.lock();
        if (!o->period) break;
        when += std::chrono::milliseconds(o->period);
    }
    --o->running;
    o->cv.notify_all();
    return 0;
}
void start(PoolObject* o, Job::Kind kind, uint64_t generation) {  // with o->running already counted
    Object* t = spawn_thread(&pool_thread, new Job{o, generation, kind});
    if (t) close_object(t);
    else finish(o);
}

PoolObject* CreateThreadpoolWork(void* fn, void* ctx, void*) { return new PoolObject(fn, ctx); }
void SubmitThreadpoolWork(PoolObject* o) {
    {
        std::lock_guard<std::mutex> g(o->m);
        ++o->running;
    }
    start(o, Job::Work, 0);
}
int TrySubmitThreadpoolCallback(void* fn, void* ctx, void*) {
    Object* t = spawn_thread(&pool_thread, new Job{new PoolObject(fn, ctx), 0, Job::Simple});
    if (t) close_object(t);
    return t != nullptr;
}
PoolObject* CreateThreadpoolTimer(void* fn, void* ctx, void*) { return new PoolObject(fn, ctx); }
void SetThreadpoolTimer(PoolObject* o, const int64_t* due, uint32_t period, uint32_t) {
    uint64_t generation;
    {
        std::lock_guard<std::mutex> g(o->m);
        generation = ++o->armed;
        o->cv.notify_all();  // the previous arming stops
        if (!due) return;
        o->due = *due;
        o->period = period;
        ++o->running;
    }
    start(o, Job::Timer, generation);
}
int IsThreadpoolTimerSet(PoolObject* o) {
    std::lock_guard<std::mutex> g(o->m);
    return o->running > 0;
}
PoolObject* CreateThreadpoolWait(void* fn, void* ctx, void*) { return new PoolObject(fn, ctx); }
void SetThreadpoolWait(PoolObject* o, HANDLE h, const int64_t* timeout) {
    uint64_t generation;
    {
        std::lock_guard<std::mutex> g(o->m);
        generation = ++o->armed;
        if (!h) return;
        o->handle = h;
        int64_t t = timeout ? *timeout : 0;
        o->timeout = !timeout ? 0xFFFFFFFFu
                     : t < 0  ? static_cast<uint32_t>(-t / 10000)
                              : static_cast<uint32_t>(t > int64_t(filetime_now()) ? (t - int64_t(filetime_now())) / 10000 : 0);
        ++o->running;
    }
    start(o, Job::Wait, generation);
}
void wait_callbacks(PoolObject* o, int32_t cancel) {
    std::unique_lock<std::mutex> g(o->m);
    if (cancel) ++o->armed, o->cv.notify_all();
    o->cv.wait(g, [o] { return o->running == 0; });
}
void WaitForThreadpoolWorkCallbacks(PoolObject* o, int32_t cancel) { wait_callbacks(o, cancel); }
void WaitForThreadpoolTimerCallbacks(PoolObject* o, int32_t cancel) { wait_callbacks(o, cancel); }
void WaitForThreadpoolWaitCallbacks(PoolObject* o, int32_t cancel) { wait_callbacks(o, cancel); }
void close_pool(PoolObject* o) {  // Windows releases it once its callbacks are done
    wait_callbacks(o, 1);
    delete o;
}
void CloseThreadpoolWork(PoolObject* o) { close_pool(o); }
void CloseThreadpoolTimer(PoolObject* o) { close_pool(o); }
void CloseThreadpoolWait(PoolObject* o) { close_pool(o); }

}  // namespace

#define K(name) EXPORT("kernel32.dll", name)
Exports threadpool_exports() {
    return {
        K(CreateThreadpoolWork), K(SubmitThreadpoolWork), K(TrySubmitThreadpoolCallback),
        K(WaitForThreadpoolWorkCallbacks), K(CloseThreadpoolWork),
        K(CreateThreadpoolTimer), K(SetThreadpoolTimer), K(IsThreadpoolTimerSet),
        K(WaitForThreadpoolTimerCallbacks), K(CloseThreadpoolTimer),
        K(CreateThreadpoolWait), K(SetThreadpoolWait), K(WaitForThreadpoolWaitCallbacks), K(CloseThreadpoolWait),
    };
}

}  // namespace host
