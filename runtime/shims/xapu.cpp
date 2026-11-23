// xapu.dll: the console's audio processing unit queue.
//
// The executable imports five functions (gta6rt --dry-run; the whole set):
//   XApuConnect, XApuDisconnect, XApuEnqueueCommand, XApuDequeueResult, XApuGetResultCount
//
// xapu is the client side of the APU command queue the title's audio mixer drives (see chad.dll next door for the DSP
// control channel). A command is enqueued, the hardware answers, the title dequeues the result. There is no APU here,
// so commands are completed on enqueue and results are kept in a small queue the caller drains.
//
// ASSUMPTION: neither the command/result record shapes nor the return types are measured
// (docs/d3d12_vulkan.md §7 lists the exports only). HRESULT for the connect/enqueue pair and S_OK is returned;
// XApuGetResultCount answers as a count so a drain loop terminates; XApuDequeueResult consumes one entry and reports
// S_OK while any remain, so a caller that only loops on success does not spin forever on an empty queue.
#include <windows.h>

#include <atomic>
#include <cstdio>
#include <mutex>

#include "shims.h"

namespace {

std::mutex g_mutex;
uint32_t g_pending = 0;
std::atomic<bool> g_warned{false};

void warn_once() {
    if (!g_warned.exchange(true, std::memory_order_relaxed)) {
        std::fprintf(stderr,
                     "[rt] xapu: the APU command queue is not implemented; commands complete on enqueue and the "
                     "caller drains the result count (logged once)\n");
    }
}

HRESULT __stdcall XApuConnect(void** /*out*/) {
    warn_once();
    return S_OK;
}

HRESULT __stdcall XApuDisconnect(void* /*apu*/) {
    const std::lock_guard lock(g_mutex);
    g_pending = 0;
    return S_OK;
}

HRESULT __stdcall XApuEnqueueCommand(void* /*apu*/, const void* /*command*/, uint32_t /*size*/) {
    const std::lock_guard lock(g_mutex);
    ++g_pending;
    return S_OK;
}

// Consumes one completed command result. S_OK while one was taken, S_FALSE when the queue is empty.
HRESULT __stdcall XApuDequeueResult(void* /*apu*/, void* /*out*/) {
    const std::lock_guard lock(g_mutex);
    if (g_pending == 0) return S_FALSE;
    --g_pending;
    return S_OK;
}

uint32_t __stdcall XApuGetResultCount(void* /*apu*/) {
    const std::lock_guard lock(g_mutex);
    return g_pending;
}

}  // namespace

SHIM_TABLE(xapu,
    {"xapu.dll", "XApuConnect", reinterpret_cast<void*>(&XApuConnect)},
    {"xapu.dll", "XApuDisconnect", reinterpret_cast<void*>(&XApuDisconnect)},
    {"xapu.dll", "XApuEnqueueCommand", reinterpret_cast<void*>(&XApuEnqueueCommand)},
    {"xapu.dll", "XApuDequeueResult", reinterpret_cast<void*>(&XApuDequeueResult)},
    {"xapu.dll", "XApuGetResultCount", reinterpret_cast<void*>(&XApuGetResultCount)},
);
