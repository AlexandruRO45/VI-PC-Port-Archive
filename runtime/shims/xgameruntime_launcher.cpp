// xgameruntime: XLauncherImpl, offline. A title asks the shell to launch another title (a companion app, a
// different game); with no shell running there is no other title to start, so the calls complete at once and
// report the launch as handed off.
//
// The class, the interface and its slot order are xodus-gaming/xgameruntime-docs' COM page (GPL-3.0; facts only,
// nothing copied): XLauncherImpl 1b339674-328d-4283-a200-3171f18d3639, interface IXLauncherImpl same id, first
// method at slot 3 (IUnknown takes 0-2): XLaunchNewGame.
//
// Measured on the executable (gdk_runtime_calls): 3 thunks reach this class, all referenced, and the runtime does
// not serve it - a title booting past its shell handoff needs a non-failing answer here or it waits on a launch
// that never reports.
//
// ASSUMPTION: XLaunchNewGame's parameter list is not measured (the public page names the function only). The call
// is accepted and logged with the pointer values, and completes synchronously with S_OK; a title that expects the
// new title's process to appear is not served.
#include <windows.h>
#include <unknwn.h>

#include <atomic>
#include <cstdio>

#include "shims.h"

namespace {

constexpr GUID kLauncherClassId = {0x1b339674, 0x328d, 0x4283, {0xa2, 0x00, 0x31, 0x71, 0xf1, 0x8d, 0x36, 0x39}};

std::atomic<bool> g_warned{false};

// A minimal IUnknown-shaped object: QueryInterface, AddRef, Release, then slot 3.
struct Launcher {
    void** vtable;
    LONG refs = 1;
};

HRESULT __stdcall launcher_query(Launcher* self, const GUID* /*iid*/, void** out) {
    if (!out) return E_POINTER;
    *out = self;
    return S_OK;
}

ULONG __stdcall launcher_add_ref(Launcher* self) {
    return static_cast<ULONG>(InterlockedIncrement(&self->refs));
}

ULONG __stdcall launcher_release(Launcher* self) {
    return static_cast<ULONG>(InterlockedDecrement(&self->refs));
}

// XLaunchNewGame(title id, argv, out) on the console. Shape unknown, so three pointer-sized words are taken and
// logged; the title-id word is the one worth reading in a run.
HRESULT __stdcall launch_new_game(Launcher* /*self*/, const void* a, const void* b, const void* c) {
    if (!g_warned.exchange(true, std::memory_order_relaxed)) {
        std::fprintf(stderr,
                     "[rt] xgameruntime: XLaunchNewGame(%p, %p, %p): no shell to launch through, reporting success "
                     "(logged once)\n",
                     a, b, c);
    }
    return S_OK;
}

void* launcher_vtable[] = {
    reinterpret_cast<void*>(&launcher_query),
    reinterpret_cast<void*>(&launcher_add_ref),
    reinterpret_cast<void*>(&launcher_release),
    reinterpret_cast<void*>(&launch_new_game),
};

Launcher g_launcher{launcher_vtable};

}  // namespace

namespace rt::shims {

bool xgameruntime_launcher_query(const GUID& class_id, REFIID interface_id, void** out, HRESULT* hr) {
    if (class_id != kLauncherClassId) return false;
    if (!out) {
        *hr = E_POINTER;
        return true;
    }
    if (interface_id != kLauncherClassId && interface_id != IID_IUnknown) {
        *out = nullptr;
        *hr = E_NOINTERFACE;
        return true;
    }
    launcher_add_ref(&g_launcher);
    *out = &g_launcher;
    *hr = S_OK;
    return true;
}

}  // namespace rt::shims
