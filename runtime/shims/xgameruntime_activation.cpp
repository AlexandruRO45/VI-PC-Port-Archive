// xgameruntime: XGameProtocolImpl and XGameInviteImpl, the classes behind XGameProtocolRegisterForActivation and
// XGameInviteRegisterForEvent, offline.
//
// Each thunk asks QueryApiImpl for the class and calls one slot with the function's own arguments after `this`
// (docs/runtime_imports.md, "Class layouts, measured"). Class and interface ids and slot order are
// xodus-gaming/xgameruntime-docs' "Layout" lists (GPL-3.0; facts only, nothing copied); signatures follow Microsoft's
// public GDK reference. No GDK header or library code is used.
//
//   XGameProtocolImpl (IXGameProtocolImpl, slots 3-4):  3 Register(queue, context, callback, token*),
//                                                       4 bool Unregister(token, wait)
//   XGameInviteImpl (IXGameInviteImpl, slots 3-4; IXGameInviteImpl2 adds 5-7, "no docs"): the same two shapes.
//
// A protocol activation or an invite comes from the console shell; there is no shell here, so nothing ever fires. A
// registration succeeds, hands out a token, and is forgotten by its Unregister (true when the token was live). The
// three IXGameInviteImpl2 slots (pending-invite register / unregister / accept, added in the October 2025 GDK) have
// no public signature and answer E_NOTIMPL (false for the unregister), logged once.
//
// ASSUMPTION: the console executable is not available, so which of these slots it calls is unknown until its first
// boot; the slots are not measured against a caller.
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

constexpr GUID kIUnknown = {0x00000000, 0x0000, 0x0000, {0xc0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x46}};
constexpr GUID kProtocolClass = {0x95fd18d2, 0x74dd, 0x4d7c, {0xaa, 0x1b, 0x0b, 0x51, 0x82, 0x76, 0x65, 0xd6}};
constexpr GUID kProtocolIface = {0x026b010c, 0x06c3, 0x4cdd, {0xbb, 0xcb, 0x43, 0xf2, 0x29, 0xdb, 0x1c, 0xff}};
constexpr GUID kInviteClass = {0x0651aae2, 0x4012, 0x4077, {0xbf, 0x84, 0x8b, 0x90, 0x97, 0x09, 0x0e, 0x2c}};
constexpr GUID kInvite2Iface = {0x014d1cc3, 0xbcfe, 0x41ff, {0xb2, 0xf0, 0xe1, 0xef, 0x07, 0x15, 0x58, 0x28}};

// One registry per class: tokens handed out and not unregistered.
struct Registrations {
    std::mutex mutex;
    std::set<uint64_t> live;
    uint64_t next = 1;
};
Registrations g_protocol;
Registrations g_invite;

template <Registrations& R>
HRESULT STDMETHODCALLTYPE register_event(void*, void* /*queue*/, void* /*context*/, void* callback, Token* token) {
    if (token) token->token = 0;
    if (!callback || !token) return E_INVALIDARG;
    std::lock_guard lock(R.mutex);
    token->token = R.next++;
    R.live.insert(token->token);
    return S_OK;
}

template <Registrations& R>
bool STDMETHODCALLTYPE unregister_event(void*, Token token, bool /*wait*/) {
    std::lock_guard lock(R.mutex);
    return R.live.erase(token.token) != 0;
}

void log_once(std::atomic<bool>& flag, const char* what) {
    if (!flag.exchange(true))
        std::fprintf(stderr, "[rt] xgameruntime: %s called; no public signature, answering E_NOTIMPL\n", what);
}
HRESULT STDMETHODCALLTYPE pending_register(void*) {
    static std::atomic<bool> logged{false};
    log_once(logged, "XGameInviteRegisterForPendingEvent (IXGameInviteImpl2 slot 5)");
    return E_NOTIMPL;
}
bool STDMETHODCALLTYPE pending_unregister(void*) {
    static std::atomic<bool> logged{false};
    log_once(logged, "XGameInviteUnregisterForPendingEvent (IXGameInviteImpl2 slot 6)");
    return false;
}
HRESULT STDMETHODCALLTYPE pending_accept(void*) {
    static std::atomic<bool> logged{false};
    log_once(logged, "XGameInviteAcceptPendingInvite (IXGameInviteImpl2 slot 7)");
    return E_NOTIMPL;
}

struct Object {
    const void* const* vtable;
    const GUID* class_id;
    const GUID* iface2;   // the derived interface id, or null
    const GUID* iface1;   // the first interface when it differs from the class id, or null
};

HRESULT STDMETHODCALLTYPE object_query_interface(void* self, REFIID iid, void** out) {
    if (!out) return E_POINTER;
    auto* o = static_cast<Object*>(self);
    if (iid == kIUnknown || iid == *o->class_id || (o->iface1 && iid == *o->iface1) || (o->iface2 && iid == *o->iface2)) {
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
const void* const kProtocolVtable[] = {
    SLOT(object_query_interface), SLOT(object_add_ref), SLOT(object_release),
    SLOT(register_event<g_protocol>),     // 3 XGameProtocolRegisterForActivation
    SLOT(unregister_event<g_protocol>),   // 4 XGameProtocolUnregisterForActivation
};
const void* const kInviteVtable[] = {
    SLOT(object_query_interface), SLOT(object_add_ref), SLOT(object_release),
    SLOT(register_event<g_invite>),       // 3 XGameInviteRegisterForEvent
    SLOT(unregister_event<g_invite>),     // 4 XGameInviteUnregisterForEvent
    SLOT(pending_register),               // 5 XGameInviteRegisterForPendingEvent (IXGameInviteImpl2)
    SLOT(pending_unregister),             // 6 XGameInviteUnregisterForPendingEvent
    SLOT(pending_accept),                 // 7 XGameInviteAcceptPendingInvite
};
#undef SLOT
static_assert(std::size(kProtocolVtable) == 5, "IUnknown + IXGameProtocolImpl's 2 methods");
static_assert(std::size(kInviteVtable) == 8, "IUnknown + IXGameInviteImpl's 2 + IXGameInviteImpl2's 3");

Object g_protocol_object{kProtocolVtable, &kProtocolClass, nullptr, &kProtocolIface};
Object g_invite_object{kInviteVtable, &kInviteClass, &kInvite2Iface, nullptr};

}  // namespace

namespace rt::shims {

// Called by xgameruntime.cpp's QueryApiImpl for every class: true when the class is one served here, with *hr set.
bool xgameruntime_activation_query(const GUID& class_id, REFIID interface_id, void** out, HRESULT* hr) {
    for (Object* o : {&g_protocol_object, &g_invite_object}) {
        if (class_id != *o->class_id) continue;
        *hr = object_query_interface(o, interface_id, out);
        return true;
    }
    return false;
}

}  // namespace rt::shims
