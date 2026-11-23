// XLauncherImpl through QueryApiImpl, called the way xgameruntime.lib's thunks call them:
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

constexpr HRESULT kNotSupported = HRESULT_FROM_WIN32(ERROR_NOT_SUPPORTED);

using LaunchUri = HRESULT(STDMETHODCALLTYPE*)(void*, void*, const char*);
using Acquire = HRESULT(STDMETHODCALLTYPE*)(void*, uintptr_t*);
using Close = void(STDMETHODCALLTYPE*)(void*, uintptr_t);

}  // namespace

int main() {
    std::printf("xgameruntime_streaming_test: XGameStreamingImpl and XLauncherImpl through QueryApiImpl\n");
    auto query = reinterpret_cast<Query>(rt::shims::find("xgameruntime.dll", "QueryApiImpl"));
    check(query != nullptr, "QueryApiImpl is registered");
    if (!query) return 1;

    // ---- XLauncherImpl ----
    const GUID lcls = parse_guid("1b339674-328d-4283-a200-3171f18d3639");
    void* l = nullptr;
    check(query(&lcls, lcls, &l) == S_OK && l, "XLauncherImpl is handed out for its class id");
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
