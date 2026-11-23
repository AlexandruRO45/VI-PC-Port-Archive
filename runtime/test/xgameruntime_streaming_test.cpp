// XGameStreamingImpl and XLauncherImpl through QueryApiImpl, called the way xgameruntime.lib's thunks call them:
// QueryApiImpl with the class id (or an interface id), one vtable slot with `this` first. Slots follow
// xgameruntime-docs' layout; arguments and expected values follow the public GDK reference. No game data; nothing online.
#include <windows.h>

#include <cstddef>
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

constexpr HRESULT kNoClient = HRESULT_FROM_WIN32(ERROR_NOT_FOUND);
constexpr HRESULT kNotSupported = HRESULT_FROM_WIN32(ERROR_NOT_SUPPORTED);

struct Token {
    uint64_t token;
};
constexpr uint64_t kClient = 0x1234;   // a client id nobody handed out

using Hr0 = HRESULT(STDMETHODCALLTYPE*)(void*);
using Void0 = void(STDMETHODCALLTYPE*)(void*);
using Bool0 = bool(STDMETHODCALLTYPE*)(void*);
using U32_0 = uint32_t(STDMETHODCALLTYPE*)(void*);
using Size0 = size_t(STDMETHODCALLTYPE*)(void*);
using ClientHr = HRESULT(STDMETHODCALLTYPE*)(void*, uint64_t);
using ClientU32 = uint32_t(STDMETHODCALLTYPE*)(void*, uint64_t);
using ClientSize = size_t(STDMETHODCALLTYPE*)(void*, uint64_t);
using Dimensions = HRESULT(STDMETHODCALLTYPE*)(void*, uint64_t, uint32_t*, uint32_t*);
using Latency = HRESULT(STDMETHODCALLTYPE*)(void*, uint64_t, uint32_t*, uint32_t*, uint32_t*);
using GetClients = HRESULT(STDMETHODCALLTYPE*)(void*, uint32_t, uint64_t*, uint32_t*);
using RegisterState = HRESULT(STDMETHODCALLTYPE*)(void*, void*, void*, void*, Token*);
using UnregisterState = bool(STDMETHODCALLTYPE*)(void*, Token, bool);
using UnregisterClient = bool(STDMETHODCALLTYPE*)(void*, uint64_t, Token, bool);
using RegisterClient = HRESULT(STDMETHODCALLTYPE*)(void*, uint64_t, void*, void*, void*, Token*);
using Name = HRESULT(STDMETHODCALLTYPE*)(void*, size_t, char*);
using TouchEnabled = HRESULT(STDMETHODCALLTYPE*)(void*, uint64_t, bool*);
using UpdateState = HRESULT(STDMETHODCALLTYPE*)(void*, size_t, const void*);
using ShowState = HRESULT(STDMETHODCALLTYPE*)(void*, const char*, size_t, const void*);
using SessionId = HRESULT(STDMETHODCALLTYPE*)(void*, uint64_t, size_t, char*, size_t*);
using SetResolution = HRESULT(STDMETHODCALLTYPE*)(void*, uint32_t, uint32_t);
using LaunchUri = HRESULT(STDMETHODCALLTYPE*)(void*, void*, const char*);
using Acquire = HRESULT(STDMETHODCALLTYPE*)(void*, uintptr_t*);
using Close = void(STDMETHODCALLTYPE*)(void*, uintptr_t);

void callback_stub() {}

}  // namespace

int main() {
    std::printf("xgameruntime_streaming_test: XGameStreamingImpl and XLauncherImpl through QueryApiImpl\n");
    auto query = reinterpret_cast<Query>(rt::shims::find("xgameruntime.dll", "QueryApiImpl"));
    check(query != nullptr, "QueryApiImpl is registered");
    if (!query) return 1;

    // ---- XGameStreamingImpl: the object and its interfaces ----
    const GUID cls = parse_guid("0a2192aa-b2d5-4d58-83be-383b6d80799e");
    void* gs = nullptr;
    check(query(&cls, cls, &gs) == S_OK && gs, "XGameStreamingImpl is handed out for its class id");
    if (!gs) return 1;
    const char* const faces[] = {"8aff07f5-a1bf-4db8-80a5-31cca0de51b7", "5f5e5169-746c-4001-ad1c-da728d01c9eb",
                                 "57786622-6605-46d0-b917-0f22bbcd9c52"};
    bool all = true;
    for (const char* id : faces) {
        void* face = nullptr;
        all = all && query(&cls, parse_guid(id), &face) == S_OK && face == gs;
    }
    check(all, "IXGameStreamingImpl, 2 and 3 (ids as the docs print them) answer with the same object");
    void* none = reinterpret_cast<void*>(1);
    check(query(&cls, parse_guid("0dd112ac-7c24-448c-b92b-3960fb5bd30c"), &none) == E_NOINTERFACE && none == nullptr,
          "an interface the class doesn't have is E_NOINTERFACE with a cleared out pointer");

    // ---- not streaming, no client ----
    check(at<Hr0>(gs, 3)(gs) == S_OK, "XGameStreamingInitialize succeeds");
    at<Void0>(gs, 4)(gs);
    check(!at<Bool0>(gs, 5)(gs), "XGameStreamingIsStreaming is false");
    check(at<U32_0>(gs, 9)(gs) == 0, "XGameStreamingGetClientCount is 0");
    uint32_t used = 99;
    uint64_t ids[4] = {};
    check(at<GetClients>(gs, 10)(gs, 4, ids, &used) == S_OK && used == 0, "XGameStreamingGetClients returns no client");
    check(at<GetClients>(gs, 10)(gs, 4, ids, nullptr) == E_INVALIDARG, "XGameStreamingGetClients(null count) is E_INVALIDARG");
    check(at<ClientU32>(gs, 11)(gs, kClient) == 0, "a client's connection state is Disconnected (0)");

    // connection-state registration
    Token t1{0}, t2{0}, bad{7};
    check(at<RegisterState>(gs, 12)(gs, nullptr, nullptr, reinterpret_cast<void*>(&callback_stub), &t1) == S_OK && t1.token != 0,
          "XGameStreamingRegisterConnectionStateChanged hands out a token");
    check(at<RegisterState>(gs, 12)(gs, nullptr, &t1, reinterpret_cast<void*>(&callback_stub), &t2) == S_OK && t2.token != 0 &&
              t2.token != t1.token,
          "a second registration gets another token");
    check(at<RegisterState>(gs, 12)(gs, nullptr, nullptr, nullptr, &bad) == E_INVALIDARG && bad.token == 0,
          "a null callback is E_INVALIDARG and clears the token");
    check(at<UnregisterState>(gs, 13)(gs, t1, true) && at<UnregisterState>(gs, 13)(gs, t2, false),
          "XGameStreamingUnregisterConnectionStateChanged is true for live tokens");
    check(!at<UnregisterState>(gs, 13)(gs, t1, true) && !at<UnregisterState>(gs, 13)(gs, Token{0}, false),
          "and false for a token already removed or never handed out");

    // calls that name a client
    uint32_t a = 5, b = 5, c = 5;
    check(at<Dimensions>(gs, 8)(gs, kClient, &a, &b) == kNoClient && a == 0 && b == 0, "physical dimensions: no such client, outputs cleared");
    check(at<Latency>(gs, 14)(gs, kClient, &a, &b, &c) == kNoClient && a == 0 && b == 0 && c == 0, "added latency: no such client, outputs cleared");
    Token ct{7};
    check(at<RegisterClient>(gs, 6)(gs, kClient, nullptr, nullptr, reinterpret_cast<void*>(&callback_stub), &ct) == kNoClient && ct.token == 0,
          "RegisterClientPropertiesChanged: no such client, token cleared");
    check(!at<UnregisterClient>(gs, 7)(gs, kClient, Token{1}, true), "UnregisterClientPropertiesChanged: no such client, false");
    bool touch = true;
    check(at<TouchEnabled>(gs, 21)(gs, kClient, &touch) == kNoClient && !touch, "IsTouchInputEnabled: no such client, output false");
    check(at<ClientHr>(gs, 22)(gs, kClient) == kNoClient, "GetLastFrameDisplayed: no such client");
    check(at<ClientSize>(gs, 29)(gs, kClient) == 0, "GetTouchBundleVersionNameSize is 0");
    char text[8] = {'x', 'x', 'x'};
    size_t used_size = 5;
    check(at<SessionId>(gs, 32)(gs, kClient, sizeof text, text, &used_size) == kNoClient && text[0] == '\0' && used_size == 0,
          "GetSessionId (IXGameStreamingImpl2, slot 32): no such client, buffer emptied");
    check(at<Name>(gs, 16)(gs, sizeof text, text) == kNoClient, "GetServerLocationName: not streaming");
    check(at<Size0>(gs, 15)(gs) == 0, "GetServerLocationNameSize is 0");

    // touch controls have nobody to send to
    at<Void0>(gs, 17)(gs);
    check(at<UpdateState>(gs, 25)(gs, 0, nullptr) == S_OK && at<UpdateState>(gs, 25)(gs, 1, nullptr) == E_INVALIDARG,
          "UpdateTouchControlsState: succeeds with nothing to do; operations missing is E_INVALIDARG");
    char ops[16] = {};
    check(at<ShowState>(gs, 27)(gs, "layout", 1, ops) == S_OK && at<ShowState>(gs, 27)(gs, nullptr, 0, nullptr) == E_INVALIDARG,
          "ShowTouchControlsWithStateUpdate: succeeds; a null layout is E_INVALIDARG");

    check(at<SetResolution>(gs, 34)(gs, 1920, 1080) == kNotSupported, "XGameStreamingSetResolution (IXGameStreamingImpl3, slot 34) is not supported");
    check(at<Hr0>(gs, 35)(gs) == 0 && at<Hr0>(gs, 38)(gs) == 0, "the four unlisted slots (35-38) answer 0");

    // ---- XLauncherImpl ----
    const GUID lcls = parse_guid("1b339674-328d-4283-a200-3171f18d3639");
    void* l = nullptr;
    check(query(&lcls, lcls, &l) == S_OK && l && l != gs, "XLauncherImpl is handed out for its class id");
    if (!l) return 1;
    check(at<LaunchUri>(l, 3)(l, nullptr, "ms-settings:") == kNotSupported, "XLaunchUri has no shell: not supported");
    check(at<LaunchUri>(l, 3)(l, nullptr, nullptr) == E_INVALIDARG, "XLaunchUri(null uri) is E_INVALIDARG");
    uintptr_t h1 = 0, h2 = 0;
    check(at<Acquire>(l, 4)(l, &h1) == S_OK && h1 != 0 && at<Acquire>(l, 4)(l, &h2) == S_OK && h2 != 0 && h1 != h2,
          "XDisplayAcquireTimeoutDeferral hands out distinct handles");
    check(at<Acquire>(l, 4)(l, nullptr) == E_INVALIDARG, "XDisplayAcquireTimeoutDeferral(null) is E_INVALIDARG");
    at<Close>(l, 5)(l, h1);
    at<Close>(l, 5)(l, h2);
    at<Close>(l, 5)(l, 0);
    check(true, "XDisplayCloseTimeoutDeferralHandle accepts the handles and a null one");

    std::printf("%d checks, %d failures\n", g_checks, g_failures);
    return g_failures ? 1 : 0;
}
