// XGameProtocolImpl and XGameInviteImpl through QueryApiImpl, called the way xgameruntime.lib's thunks call them:
// QueryApiImpl with the class id (or an interface id), one vtable slot with `this` first. Slots follow
// xgameruntime-docs' layout; arguments and expected values follow the public GDK reference. No game data; nothing online.
#include <windows.h>

#include <cstdint>
#include <cstdio>
#include <cstring>

#include "shims.h"

namespace {

int g_checks = 0;
int g_failures = 0;

void check(bool condition, const char* what) {
    ++g_checks;
    std::printf("%s %s\n", condition ? "PASS" : "FAIL", what);
    if (!condition) ++g_failures;
}

// "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx", as xgameruntime-docs prints the ids, so the shim's byte arrays are checked
// against the text rather than against a second copy of the same bytes.
GUID parse_guid(const char* text) {
    unsigned char bytes[16] = {};   // in text order
    int n = 0;
    for (const char* c = text; *c && n < 32; ++c) {
        int v = (*c >= '0' && *c <= '9') ? *c - '0' : (*c >= 'a' && *c <= 'f') ? *c - 'a' + 10 : -1;
        if (v < 0) continue;   // the dashes
        bytes[n / 2] = static_cast<unsigned char>(bytes[n / 2] << 4 | v);
        ++n;
    }
    GUID g{};
    g.Data1 = static_cast<unsigned long>(bytes[0]) << 24 | bytes[1] << 16 | bytes[2] << 8 | bytes[3];
    g.Data2 = static_cast<unsigned short>(bytes[4] << 8 | bytes[5]);
    g.Data3 = static_cast<unsigned short>(bytes[6] << 8 | bytes[7]);
    std::memcpy(g.Data4, bytes + 8, 8);
    return g;
}

using Query = HRESULT(WINAPI*)(const GUID*, REFIID, void**);
template <class F> F at(void* obj, int index) { return reinterpret_cast<F>((*static_cast<void* const* const*>(obj))[index]); }

struct Token {
    uint64_t token;
};
using Register = HRESULT(STDMETHODCALLTYPE*)(void*, void*, void*, void*, Token*);
using Unregister = bool(STDMETHODCALLTYPE*)(void*, Token, bool);
using Slot = HRESULT(STDMETHODCALLTYPE*)(void*);
using SlotBool = bool(STDMETHODCALLTYPE*)(void*);

void callback_stub() {}

// The behaviour both classes share: slots 3 and 4.
void check_registration(void* obj, const char* name) {
    char msg[160];
    auto reg = at<Register>(obj, 3);
    auto unreg = at<Unregister>(obj, 4);
    Token t1{0}, t2{0};
    std::snprintf(msg, sizeof msg, "%s: Register succeeds and hands out a token", name);
    check(reg(obj, nullptr, nullptr, reinterpret_cast<void*>(&callback_stub), &t1) == S_OK && t1.token != 0, msg);
    std::snprintf(msg, sizeof msg, "%s: a second registration gets another token", name);
    check(reg(obj, nullptr, &t1, reinterpret_cast<void*>(&callback_stub), &t2) == S_OK && t2.token != 0 && t2.token != t1.token, msg);
    Token bad{7};
    std::snprintf(msg, sizeof msg, "%s: a null callback is E_INVALIDARG and clears the token", name);
    check(reg(obj, nullptr, nullptr, nullptr, &bad) == E_INVALIDARG && bad.token == 0, msg);
    std::snprintf(msg, sizeof msg, "%s: a null token pointer is E_INVALIDARG", name);
    check(reg(obj, nullptr, nullptr, reinterpret_cast<void*>(&callback_stub), nullptr) == E_INVALIDARG, msg);
    std::snprintf(msg, sizeof msg, "%s: Unregister of a live token is true (wait or not)", name);
    check(unreg(obj, t1, true) && unreg(obj, t2, false), msg);
    std::snprintf(msg, sizeof msg, "%s: Unregister of a token already removed, or never handed out, is false", name);
    check(!unreg(obj, t1, true) && !unreg(obj, Token{0}, false) && !unreg(obj, Token{999999}, false), msg);
}

}  // namespace

int main() {
    std::printf("xgameruntime_activation_test: XGameProtocolImpl and XGameInviteImpl through QueryApiImpl\n");
    auto query = reinterpret_cast<Query>(rt::shims::find("xgameruntime.dll", "QueryApiImpl"));
    check(query != nullptr, "QueryApiImpl is registered");
    if (!query) return 1;

    // ---- XGameProtocolImpl ----
    const GUID protocol_class = parse_guid("95fd18d2-74dd-4d7c-aa1b-0b51827665d6");
    void* protocol = nullptr;
    check(query(&protocol_class, protocol_class, &protocol) == S_OK && protocol, "XGameProtocolImpl is handed out for its class id");
    if (!protocol) return 1;
    void* face = nullptr;
    check(query(&protocol_class, parse_guid("026b010c-06c3-4cdd-bbcb-43f229db1cff"), &face) == S_OK && face == protocol,
          "IXGameProtocolImpl (its interface id differs from the class id) answers with the same object");
    void* none = reinterpret_cast<void*>(1);
    check(query(&protocol_class, parse_guid("0dd112ac-7c24-448c-b92b-3960fb5bd30c"), &none) == E_NOINTERFACE && none == nullptr,
          "an interface the class doesn't have is E_NOINTERFACE with a cleared out pointer");
    check_registration(protocol, "protocol");

    // ---- XGameInviteImpl ----
    const GUID invite_class = parse_guid("0651aae2-4012-4077-bf84-8b9097090e2c");
    void* invite = nullptr;
    check(query(&invite_class, invite_class, &invite) == S_OK && invite, "XGameInviteImpl is handed out for its class id");
    if (!invite) return 1;
    face = nullptr;
    check(query(&invite_class, parse_guid("014d1cc3-bcfe-41ff-b2f0-e1ef07155828"), &face) == S_OK && face == invite,
          "IXGameInviteImpl2 answers with the same object");
    check_registration(invite, "invite");
    check(at<Slot>(invite, 5)(invite) == E_NOTIMPL && !at<SlotBool>(invite, 6)(invite) && at<Slot>(invite, 7)(invite) == E_NOTIMPL,
          "the three undocumented IXGameInviteImpl2 slots answer E_NOTIMPL / false");

    // The two classes keep separate registrations.
    Token pt{0};
    at<Register>(protocol, 3)(protocol, nullptr, nullptr, reinterpret_cast<void*>(&callback_stub), &pt);
    check(!at<Unregister>(invite, 4)(invite, pt, false), "a protocol token is not a live invite token");
    check(at<Unregister>(protocol, 4)(protocol, pt, false), "but it is live for the protocol class");

    std::printf("%d checks, %d failures\n", g_checks, g_failures);
    return g_failures ? 1 : 0;
}
