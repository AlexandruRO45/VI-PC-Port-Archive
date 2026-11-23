// xgameruntime: XGameStreamingImpl and XLauncherImpl, the classes behind the XGameStreaming* functions and XLaunchUri /
// XDisplayAcquireTimeoutDeferral, offline.
//
// Each thunk asks QueryApiImpl for the class and calls one slot with the function's own arguments after `this`
// (docs/runtime_imports.md, "Class layouts, measured"). Class and interface ids and slot order are
// xodus-gaming/xgameruntime-docs' "Layout" lists (GPL-3.0; facts only, nothing copied); signatures follow Microsoft's
// public GDK reference. No GDK header or library code is used.
//
// A PC running the port is not streamed to anyone, so XGameStreamingImpl answers as a console with no client:
//   - XGameStreamingIsStreaming is false, the client count 0 and GetClients returns none, every client's connection
//     state is Disconnected, and a connection-state registration succeeds with a token that never fires;
//   - every call that names a client fails with HRESULT_FROM_WIN32(ERROR_NOT_FOUND), its outputs cleared where their
//     type is known (a frame token and the display-details structure are not in the pages read, so those are left
//     untouched);
//   - touch-control layouts and state updates have nobody to send to: they succeed and do nothing;
//   - XGameStreamingSetResolution fails with HRESULT_FROM_WIN32(ERROR_NOT_SUPPORTED).
// XLauncherImpl: XLaunchUri has no shell to hand the URI to and fails with ERROR_NOT_SUPPORTED (the URI is not
// logged); the screen-timeout deferral hands out a handle that Close accepts.
//
// ASSUMPTION: the console executable is not available. The slot numbers are xgameruntime-docs' order:
// IXGameStreamingImpl is 29 methods (slots 3-31), IXGameStreamingImpl2 adds 32 and IXGameStreamingImpl3 adds 33-34.
// The package's libHttpClient.GDK.dll has thunks for slots 3-34 and four more of an interface the docs do not list
// (35-38): those answer 0 and log once. XGameStreamingClientId's size is not in the pages read; it is taken as a
// register-sized value, and on x64 a larger structure is passed by pointer in the same register, so the argument
// positions hold either way.
#include <windows.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <iterator>
#include <mutex>
#include <set>

namespace {

struct Token {   // XTaskQueueRegistrationToken: one 64-bit value
    uint64_t token;
};
using ClientId = uint64_t;

constexpr HRESULT kNoClient = HRESULT_FROM_WIN32(ERROR_NOT_FOUND);
constexpr HRESULT kNotSupported = HRESULT_FROM_WIN32(ERROR_NOT_SUPPORTED);

constexpr GUID kIUnknown = {0x00000000, 0x0000, 0x0000, {0xc0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x46}};
constexpr GUID kStreamingClass = {0x0a2192aa, 0xb2d5, 0x4d58, {0x83, 0xbe, 0x38, 0x3b, 0x6d, 0x80, 0x79, 0x9e}};
constexpr GUID kStreaming1 = {0x8aff07f5, 0xa1bf, 0x4db8, {0x80, 0xa5, 0x31, 0xcc, 0xa0, 0xde, 0x51, 0xb7}};
constexpr GUID kStreaming2 = {0x5f5e5169, 0x746c, 0x4001, {0xad, 0x1c, 0xda, 0x72, 0x8d, 0x01, 0xc9, 0xeb}};
constexpr GUID kStreaming3 = {0x57786622, 0x6605, 0x46d0, {0xb9, 0x17, 0x0f, 0x22, 0xbb, 0xcd, 0x9c, 0x52}};
constexpr GUID kLauncherClass = {0x1b339674, 0x328d, 0x4283, {0xa2, 0x00, 0x31, 0x71, 0xf1, 0x8d, 0x36, 0x39}};

std::mutex g_mutex;

// ---- XGameStreamingImpl ----

std::set<uint64_t> g_state_tokens;   // connection-state registrations not yet unregistered, under g_mutex
uint64_t g_next_token = 1;

HRESULT STDMETHODCALLTYPE gs_initialize(void*) { return S_OK; }   // 3
void STDMETHODCALLTYPE gs_uninitialize(void*) {}                  // 4
bool STDMETHODCALLTYPE gs_is_streaming(void*) { return false; }   // 5
HRESULT STDMETHODCALLTYPE gs_register_client_props(void*, ClientId, void*, void*, void*, Token* token) {   // 6
    if (token) token->token = 0;
    return kNoClient;
}
bool STDMETHODCALLTYPE gs_unregister_client_props(void*, ClientId, Token, bool) { return false; }          // 7
HRESULT STDMETHODCALLTYPE gs_physical_dimensions(void*, ClientId, uint32_t* h, uint32_t* v) {              // 8
    if (h) *h = 0;
    if (v) *v = 0;
    return kNoClient;
}
uint32_t STDMETHODCALLTYPE gs_client_count(void*) { return 0; }   // 9
HRESULT STDMETHODCALLTYPE gs_get_clients(void*, uint32_t, ClientId*, uint32_t* used) {                     // 10
    if (!used) return E_INVALIDARG;
    *used = 0;
    return S_OK;
}
uint32_t STDMETHODCALLTYPE gs_connection_state(void*, ClientId) { return 0; }   // 11 Disconnected
HRESULT STDMETHODCALLTYPE gs_register_state_changed(void*, void*, void*, void* callback, Token* token) {   // 12
    if (token) token->token = 0;
    if (!callback || !token) return E_INVALIDARG;
    std::lock_guard lock(g_mutex);
    token->token = g_next_token++;
    g_state_tokens.insert(token->token);
    return S_OK;
}
bool STDMETHODCALLTYPE gs_unregister_state_changed(void*, Token token, bool) {                             // 13
    std::lock_guard lock(g_mutex);
    return g_state_tokens.erase(token.token) != 0;
}
HRESULT STDMETHODCALLTYPE gs_added_latency(void*, ClientId, uint32_t* in, uint32_t* out, uint32_t* dev) {  // 14
    if (in) *in = 0;
    if (out) *out = 0;
    if (dev) *dev = 0;
    return kNoClient;
}
size_t STDMETHODCALLTYPE gs_location_name_size(void*) { return 0; }   // 15
HRESULT STDMETHODCALLTYPE gs_location_name(void*, size_t size, char* name) {                               // 16
    if (name && size) name[0] = '\0';
    return kNoClient;
}
void STDMETHODCALLTYPE gs_hide_touch(void*) {}                                                             // 17
void STDMETHODCALLTYPE gs_show_touch_layout(void*, const char*) {}                                         // 18
void STDMETHODCALLTYPE gs_hide_touch_on_client(void*, ClientId) {}                                         // 19
void STDMETHODCALLTYPE gs_show_touch_layout_on_client(void*, ClientId, const char*) {}                     // 20
HRESULT STDMETHODCALLTYPE gs_touch_input_enabled(void*, ClientId, bool* enabled) {                         // 21
    if (enabled) *enabled = false;
    return kNoClient;
}
HRESULT STDMETHODCALLTYPE gs_last_frame_displayed(void*, ClientId, void*) { return kNoClient; }            // 22
HRESULT STDMETHODCALLTYPE gs_associated_frame(void*, void*, void*) { return kNoClient; }                   // 23
HRESULT STDMETHODCALLTYPE gs_gamepad_physicality(void*, void*, void*) { return kNoClient; }                // 24
HRESULT STDMETHODCALLTYPE gs_update_touch_state(void*, size_t count, const void* ops) {                    // 25
    return (count && !ops) ? E_INVALIDARG : S_OK;
}
HRESULT STDMETHODCALLTYPE gs_update_touch_state_on_client(void*, ClientId, size_t, const void*) {          // 26
    return kNoClient;
}
HRESULT STDMETHODCALLTYPE gs_show_touch_with_state(void*, const char* layout, size_t count, const void* ops) {   // 27
    return (!layout || (count && !ops)) ? E_INVALIDARG : S_OK;
}
HRESULT STDMETHODCALLTYPE gs_show_touch_with_state_on_client(void*, ClientId, const char*, size_t, const void*) {   // 28
    return kNoClient;
}
size_t STDMETHODCALLTYPE gs_touch_bundle_name_size(void*, ClientId) { return 0; }                          // 29
HRESULT STDMETHODCALLTYPE gs_touch_bundle_version(void*, ClientId, void*, size_t size, char* name) {      // 30
    if (name && size) name[0] = '\0';
    return kNoClient;
}
HRESULT STDMETHODCALLTYPE gs_client_ip(void*, ClientId, size_t size, char* ip) {                           // 31
    if (ip && size) ip[0] = '\0';
    return kNoClient;
}
HRESULT STDMETHODCALLTYPE gs_session_id(void*, ClientId, size_t size, char* id, size_t* used) {           // 32
    if (id && size) id[0] = '\0';
    if (used) *used = 0;
    return kNoClient;
}
HRESULT STDMETHODCALLTYPE gs_display_details(void*, ClientId, uint32_t, float, float, void*) {             // 33
    return kNoClient;
}
HRESULT STDMETHODCALLTYPE gs_set_resolution(void*, uint32_t, uint32_t) { return kNotSupported; }           // 34

template <int Slot> uint64_t STDMETHODCALLTYPE gs_private_slot(void*) {                                    // 35-38
    static std::atomic<bool> logged{false};
    if (!logged.exchange(true))
        std::fprintf(stderr, "[rt] xgameruntime: XGameStreamingImpl slot %d (not in xgameruntime-docs) called; answering 0\n", Slot);
    return 0;
}

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
const void* const kStreamingVtable[] = {
    SLOT(object_query_interface), SLOT(object_add_ref), SLOT(object_release),
    SLOT(gs_initialize), SLOT(gs_uninitialize), SLOT(gs_is_streaming),                                        // 3-5
    SLOT(gs_register_client_props), SLOT(gs_unregister_client_props), SLOT(gs_physical_dimensions),            // 6-8
    SLOT(gs_client_count), SLOT(gs_get_clients), SLOT(gs_connection_state),                                   // 9-11
    SLOT(gs_register_state_changed), SLOT(gs_unregister_state_changed), SLOT(gs_added_latency),               // 12-14
    SLOT(gs_location_name_size), SLOT(gs_location_name),                                                      // 15-16
    SLOT(gs_hide_touch), SLOT(gs_show_touch_layout), SLOT(gs_hide_touch_on_client),                           // 17-19
    SLOT(gs_show_touch_layout_on_client), SLOT(gs_touch_input_enabled),                                       // 20-21
    SLOT(gs_last_frame_displayed), SLOT(gs_associated_frame), SLOT(gs_gamepad_physicality),                   // 22-24
    SLOT(gs_update_touch_state), SLOT(gs_update_touch_state_on_client), SLOT(gs_show_touch_with_state),       // 25-27
    SLOT(gs_show_touch_with_state_on_client), SLOT(gs_touch_bundle_name_size), SLOT(gs_touch_bundle_version), // 28-30
    SLOT(gs_client_ip),                                                                                       // 31
    SLOT(gs_session_id),                                                                                      // 32 (IXGameStreamingImpl2)
    SLOT(gs_display_details), SLOT(gs_set_resolution),                                                        // 33-34 (IXGameStreamingImpl3)
    SLOT(gs_private_slot<35>), SLOT(gs_private_slot<36>), SLOT(gs_private_slot<37>), SLOT(gs_private_slot<38>),
};
const void* const kLauncherVtable[] = {
    SLOT(object_query_interface), SLOT(object_add_ref), SLOT(object_release),
    SLOT(launcher_launch_uri),          // 3 XLaunchUri
    SLOT(launcher_acquire_deferral),    // 4 XDisplayAcquireTimeoutDeferral
    SLOT(launcher_close_deferral),      // 5 XDisplayCloseTimeoutDeferralHandle
};
#undef SLOT
static_assert(std::size(kStreamingVtable) == 39, "IUnknown + IXGameStreamingImpl's 29 + 1 + 2 + four unlisted");
static_assert(std::size(kLauncherVtable) == 6, "IUnknown + IXLauncherImpl's 3 methods");

Object g_streaming{kStreamingVtable, &kStreamingClass, {&kStreaming1, &kStreaming2, &kStreaming3}};
Object g_launcher{kLauncherVtable, &kLauncherClass, {nullptr, nullptr, nullptr}};

}  // namespace

namespace rt::shims {

// Called by xgameruntime.cpp's QueryApiImpl for every class: true when the class is one served here, with *hr set.
bool xgameruntime_streaming_query(const GUID& class_id, REFIID interface_id, void** out, HRESULT* hr) {
    for (Object* o : {&g_streaming, &g_launcher}) {
        if (class_id != *o->class_id) continue;
        *hr = object_query_interface(o, interface_id, out);
        return true;
    }
    return false;
}

}  // namespace rt::shims
