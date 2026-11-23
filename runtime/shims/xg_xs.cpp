// xg_xs.dll: Xbox GPU texture tiling helpers (XG library). On the console these compute optimal memory
// layouts for RDNA2's tiled addressing; on a PC the driver handles tiling and the calls are stubs.
//
// ABI hypothesis: names from the game's import table. XGCreateTextureComputer likely returns a COM-style
// HRESULT and writes an interface pointer to an out parameter; XGSuggestSwizzleMode returns a swizzle enum.
// Without measured call sites, the stubs return 0 (S_OK / linear) and log once.
#include <windows.h>

#include <atomic>
#include <cstdint>
#include <cstdio>

#include "shims.h"

namespace {

uint64_t __stdcall XGCreateTextureComputer(uint64_t a, uint64_t b, uint64_t c, uint64_t d) {
    static std::atomic<bool> warned{false};
    if (!warned.exchange(true, std::memory_order_relaxed))
        std::fprintf(stderr, "[rt] xg_xs: XGCreateTextureComputer: stub (PC driver handles tiling)\n");
    return 0;
}

uint64_t __stdcall XGSuggestSwizzleMode(uint64_t a, uint64_t b, uint64_t c, uint64_t d) {
    static std::atomic<bool> warned{false};
    if (!warned.exchange(true, std::memory_order_relaxed))
        std::fprintf(stderr, "[rt] xg_xs: XGSuggestSwizzleMode: stub returning 0 (linear)\n");
    return 0;
}

}  // namespace

SHIM_TABLE(xg_xs,
    {"xg_xs.dll", "XGCreateTextureComputer", reinterpret_cast<void*>(&XGCreateTextureComputer)},
    {"xg_xs.dll", "XGSuggestSwizzleMode", reinterpret_cast<void*>(&XGSuggestSwizzleMode)},
);
