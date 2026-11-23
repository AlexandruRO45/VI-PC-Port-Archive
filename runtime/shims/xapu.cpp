// xapu.dll: Xbox APU command interface. The console sends audio-processing commands to the custom AMD APU;
// a PC has no equivalent coprocessor. The stubs succeed silently: XApuConnect returns a handle (nonzero),
// XApuGetResultCount reports zero pending results, and the rest return 0 (success). No audio is produced
// through this path; a future XAudio2 shim would replace it.
//
// ABI hypothesis: names from the game's import table; parameter types are not measured.
#include <windows.h>

#include <atomic>
#include <cstdint>
#include <cstdio>

#include "shims.h"

namespace {

uint64_t __stdcall XApuConnect(uint64_t a, uint64_t b, uint64_t c, uint64_t d) {
    static std::atomic<bool> warned{false};
    if (!warned.exchange(true, std::memory_order_relaxed))
        std::fprintf(stderr, "[rt] xapu: XApuConnect: stub (no APU on PC)\n");
    return 1;
}

uint64_t __stdcall XApuDisconnect(uint64_t a, uint64_t b, uint64_t c, uint64_t d) {
    static std::atomic<bool> warned{false};
    if (!warned.exchange(true, std::memory_order_relaxed))
        std::fprintf(stderr, "[rt] xapu: XApuDisconnect: stub\n");
    return 0;
}

uint64_t __stdcall XApuEnqueueCommand(uint64_t a, uint64_t b, uint64_t c, uint64_t d) {
    return 0;
}

uint64_t __stdcall XApuDequeueResult(uint64_t a, uint64_t b, uint64_t c, uint64_t d) {
    return 0;
}

uint64_t __stdcall XApuGetResultCount(uint64_t a, uint64_t b, uint64_t c, uint64_t d) {
    return 0;
}

}  // namespace

SHIM_TABLE(xapu,
    {"xapu.dll", "XApuConnect", reinterpret_cast<void*>(&XApuConnect)},
    {"xapu.dll", "XApuDisconnect", reinterpret_cast<void*>(&XApuDisconnect)},
    {"xapu.dll", "XApuEnqueueCommand", reinterpret_cast<void*>(&XApuEnqueueCommand)},
    {"xapu.dll", "XApuDequeueResult", reinterpret_cast<void*>(&XApuDequeueResult)},
    {"xapu.dll", "XApuGetResultCount", reinterpret_cast<void*>(&XApuGetResultCount)},
);
