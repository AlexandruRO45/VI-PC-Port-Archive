// xg_xs.dll: the console's texture-layout helpers.
//
// The executable imports two of them (gta6rt --dry-run; the only xg_xs imports it has):
//   XGCreateTextureComputer(...)    a per-resource helper that answers tiled layouts for one texture
//   XGSuggestSwizzleMode(...)      which swizzle mode a resource should use
//
// Both decide how a console resource is laid out in memory before it is created. A PC resource is created by the
// D3D12 layer here, which already decides its own tiling (the placed-resource path in d3d12_vulkan_device.cpp), so a
// mode that is only a hint can be answered without a computer object existing.
//
// ASSUMPTION: argument lists are not measured (docs/d3d12_vulkan.md §7 names the exports only). XGSuggestSwizzleMode
// returns the "no swizzle, linear" mode 0 and XGCreateTextureComputer returns a non-null handle the caller can keep:
// a null return would read as failure and stop a resource at creation. Both are logged once so a first run shows
// which shapes the title actually uses.
#include <windows.h>

#include <atomic>
#include <cstdio>

#include "shims.h"

namespace {

std::atomic<bool> g_warned{false};

void warn_once(const char* what) {
    if (!g_warned.exchange(true, std::memory_order_relaxed)) {
        std::fprintf(stderr,
                     "[rt] xg_xs: %s answered as linear/unspecified; the console layout is not reproduced yet "
                     "(logged once)\n",
                     what);
    }
}

// void* XGCreateTextureComputer(const void* desc) -> an opaque computer the caller passes back. Nothing is
// dereferenced here, so the handle only has to be stable and non-null.
void* __stdcall XGCreateTextureComputer(const void* /*desc*/) {
    warn_once("XGCreateTextureComputer");
    static char computer;  // stable address, never freed by the shim
    return &computer;
}

// uint32_t XGSuggestSwizzleMode(const void* desc) -> XG_SWIZZLE_MODE. 0 is the linear mode; a console title that
// gets it lays the resource out linearly, which the D3D12 layer can still create.
uint32_t __stdcall XGSuggestSwizzleMode(const void* /*desc*/) {
    warn_once("XGSuggestSwizzleMode");
    return 0;
}

}  // namespace

SHIM_TABLE(xg_xs,
    {"xg_xs.dll", "XGCreateTextureComputer", reinterpret_cast<void*>(&XGCreateTextureComputer)},
    {"xg_xs.dll", "XGSuggestSwizzleMode", reinterpret_cast<void*>(&XGSuggestSwizzleMode)},
);
