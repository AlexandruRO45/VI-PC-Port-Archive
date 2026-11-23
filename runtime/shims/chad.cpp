// chad.dll: Xbox DSP audio subsystem (XDsp*). The console sends audio mixing commands to a dedicated DSP
// processor; a PC has no equivalent hardware. The stubs log on first call and return success (0) so the
// game's audio initialisation completes without crashing. No audio is produced.
//
// ABI hypothesis: names from the game's import table; parameter counts and types are not measured. Each
// function receives up to four register arguments (x64 fastcall) and returns zero in RAX.
#include <windows.h>

#include <atomic>
#include <cstdint>
#include <cstdio>

#include "shims.h"

namespace {

uint64_t __stdcall XDspActivate(uint64_t a, uint64_t b, uint64_t c, uint64_t d) {
    static std::atomic<bool> warned{false};
    if (!warned.exchange(true, std::memory_order_relaxed))
        std::fprintf(stderr, "[rt] chad: XDspActivate(%llx, %llx, %llx, %llx): stub (no DSP on PC)\n",
                     static_cast<unsigned long long>(a), static_cast<unsigned long long>(b),
                     static_cast<unsigned long long>(c), static_cast<unsigned long long>(d));
    return 0;
}

uint64_t __stdcall XDspConnectWithMaximumStreamLimit(uint64_t a, uint64_t b, uint64_t c, uint64_t d) {
    static std::atomic<bool> warned{false};
    if (!warned.exchange(true, std::memory_order_relaxed))
        std::fprintf(stderr, "[rt] chad: XDspConnectWithMaximumStreamLimit: stub (no DSP on PC)\n");
    return 0;
}

uint64_t __stdcall XDspDeactivate(uint64_t a, uint64_t b, uint64_t c, uint64_t d) {
    static std::atomic<bool> warned{false};
    if (!warned.exchange(true, std::memory_order_relaxed))
        std::fprintf(stderr, "[rt] chad: XDspDeactivate: stub\n");
    return 0;
}

uint64_t __stdcall XDspDisconnect(uint64_t a, uint64_t b, uint64_t c, uint64_t d) {
    static std::atomic<bool> warned{false};
    if (!warned.exchange(true, std::memory_order_relaxed))
        std::fprintf(stderr, "[rt] chad: XDspDisconnect: stub\n");
    return 0;
}

uint64_t __stdcall XDspSubmitCommand(uint64_t a, uint64_t b, uint64_t c, uint64_t d) {
    static std::atomic<bool> warned{false};
    if (!warned.exchange(true, std::memory_order_relaxed))
        std::fprintf(stderr, "[rt] chad: XDspSubmitCommand: stub\n");
    return 0;
}

}  // namespace

SHIM_TABLE(chad,
    {"chad.dll", "XDspActivate", reinterpret_cast<void*>(&XDspActivate)},
    {"chad.dll", "XDspConnectWithMaximumStreamLimit", reinterpret_cast<void*>(&XDspConnectWithMaximumStreamLimit)},
    {"chad.dll", "XDspDeactivate", reinterpret_cast<void*>(&XDspDeactivate)},
    {"chad.dll", "XDspDisconnect", reinterpret_cast<void*>(&XDspDisconnect)},
    {"chad.dll", "XDspSubmitCommand", reinterpret_cast<void*>(&XDspSubmitCommand)},
);
