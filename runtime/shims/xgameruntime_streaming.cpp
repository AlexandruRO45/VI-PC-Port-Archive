// xgameruntime: XLauncherImpl, the class behind XLaunchUri / XDisplayAcquireTimeoutDeferral, offline.
//
// Each thunk asks QueryApiImpl for the class and calls one slot with the function's own arguments after `this`
// (docs/runtime_imports.md, "Class layouts, measured"). Class and interface ids and slot order are
// xodus-gaming/xgameruntime-docs' "Layout" lists (GPL-3.0; facts only, nothing copied); signatures follow Microsoft's
// public GDK reference. No GDK header or library code is used.
//
// XLaunchUri has no shell to hand the URI to and fails with ERROR_NOT_SUPPORTED (the URI is not logged); the
// screen-timeout deferral hands out a handle that Close accepts. XGameStreamingImpl is served by
// xgameruntime_offline.cpp.
#include <windows.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <iterator>
#include <mutex>
#include <set>

namespace {

constexpr HRESULT kNotSupported = HRESULT_FROM_WIN32(ERROR_NOT_SUPPORTED);

constexpr GUID kIUnknown = {0x00000000, 0x0000, 0x0000, {0xc0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x46}};
constexpr GUID kLauncherClass = {0x1b339674, 0x328d, 0x4283, {0xa2, 0x00, 0x31, 0x71, 0xf1, 0x8d, 0x36, 0x39}};

std::mutex g_mutex;

// ---- XLauncherImpl ----

std::set<uintptr_t> g_deferrals;   // under g_mutex
uintptr_t g_next_deferral = 1;

HRESULT STDMETHODCALLTYPE launcher_launch_uri(void*, void* /*user*/, const char* uri) {                    // 3
    if (!uri) return E_INVALIDARG;
    static std::atomic<bool> logged{false};
    if (!logged.exchange(true)) std::fprintf(stderr, "[rt] xgameruntime: XLaunchUri ignored (no shell offline)\n");
    return kNotSupported;
}
HRESULT STDMETHODCALLTYPE launcher_acquire_deferral(void*, uintptr_t* handle) {                            // 4
    if (!handle) return E_INVALIDARG;
    std::lock_guard lock(g_mutex);
    *handle = g_next_deferral++;
    g_deferrals.insert(*handle);
    return S_OK;
}
void STDMETHODCALLTYPE launcher_close_deferral(void*, uintptr_t handle) {                                  // 5
    std::lock_guard lock(g_mutex);
    g_deferrals.erase(handle);
}

// ---- the objects ----

struct Object {
    const void* const* vtable;
    const GUID* class_id;
    const GUID* ifaces[3];   // further interface ids, null-terminated
};

HRESULT STDMETHODCALLTYPE object_query_interface(void* self, REFIID iid, void** out) {
    if (!out) return E_POINTER;
    auto* o = static_cast<Object*>(self);
    bool known = iid == kIUnknown || iid == *o->class_id;
    for (const GUID* i : o->ifaces) known = known || (i && iid == *i);
    if (known) {
        *out = self;
        return S_OK;
    }
    *out = nullptr;
    return E_NOINTERFACE;
}
// Process-wide singletons: references are not counted, the objects are never freed.
ULONG STDMETHODCALLTYPE object_add_ref(void*) { return 2; }
ULONG STDMETHODCALLTYPE object_release(void*) { return 1; }

#define SLOT(fn) reinterpret_cast<const void*>(&fn)
const void* const kLauncherVtable[] = {
    SLOT(object_query_interface), SLOT(object_add_ref), SLOT(object_release),
    SLOT(launcher_launch_uri),          // 3 XLaunchUri
    SLOT(launcher_acquire_deferral),    // 4 XDisplayAcquireTimeoutDeferral
    SLOT(launcher_close_deferral),      // 5 XDisplayCloseTimeoutDeferralHandle
};
#undef SLOT
static_assert(std::size(kLauncherVtable) == 6, "IUnknown + IXLauncherImpl's 3 methods");

Object g_launcher{kLauncherVtable, &kLauncherClass, {nullptr, nullptr, nullptr}};

}  // namespace

namespace rt::shims {

// Called by xgameruntime.cpp's QueryApiImpl for every class: true when the class is one served here, with *hr set.
bool xgameruntime_streaming_query(const GUID& class_id, REFIID interface_id, void** out, HRESULT* hr) {
    for (Object* o : {&g_launcher}) {
        if (class_id != *o->class_id) continue;
        *hr = object_query_interface(o, interface_id, out);
        return true;
    }
    return false;
}

}  // namespace rt::shims
