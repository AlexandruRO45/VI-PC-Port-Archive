// XGameUiImpl through QueryApiImpl, called the way xgameruntime.lib's XGameUi* thunks call it: QueryApiImpl with the
// class id (or an interface id), one vtable slot with `this` first. Slots follow xgameruntime-docs' layout; arguments
// and expected values follow the public GDK reference. No game data; nothing online.
#include <windows.h>

#include <cstdint>
#include <cstdio>
#include <cstring>

#include <XAsync.h>
#include <XAsyncProvider.h>
#include <XTaskQueue.h>

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

bool finished(XAsyncBlock* async) { return XAsyncGetStatus(async, true) != E_PENDING; }

constexpr HRESULT kNetworkUnreachable = HRESULT_FROM_WIN32(ERROR_NETWORK_UNREACHABLE);

using MessageAsync = HRESULT(STDMETHODCALLTYPE*)(void*, XAsyncBlock*, const char*, const char*, const char*, const char*,
                                                 const char*, uint32_t, uint32_t);
using MessageResult = HRESULT(STDMETHODCALLTYPE*)(void*, XAsyncBlock*, uint32_t*);

// Shows a message dialog and returns the button it was answered with, or 99 when it failed.
uint32_t answer(void* ui, const char* first, const char* second, const char* third, uint32_t default_button) {
    XAsyncBlock async{};
    if (at<MessageAsync>(ui, 3)(ui, &async, "Title", "Content", first, second, third, default_button, 0) != S_OK) return 99;
    if (!finished(&async)) return 99;
    uint32_t button = 99;
    return at<MessageResult>(ui, 4)(ui, &async, &button) == S_OK ? button : 99;
}

}  // namespace

int main() {
    std::printf("xgameruntime_ui_test: XGameUiImpl through QueryApiImpl\n");
    auto query = reinterpret_cast<Query>(rt::shims::find("xgameruntime.dll", "QueryApiImpl"));
    check(query != nullptr, "QueryApiImpl is registered");
    if (!query) return 1;

    // ---- the object and its interfaces ----
    const GUID cls = parse_guid("dfcd4649-4ff8-4043-ba07-35d607df98b0");
    void* ui = nullptr;
    check(query(&cls, cls, &ui) == S_OK && ui, "QueryApiImpl hands out XGameUiImpl for its class id (as the thunks ask)");
    if (!ui) return 1;
    const char* const faces[] = {"6eeaa73e-9669-43ad-a2c7-d0da4e1f50a1", "36a03122-9ea3-4a3a-a8a4-899cfd85d7db",
                                 "ade7eba1-2093-42ce-a544-a523a66790e0", "eaf669df-5542-4590-99a3-8dc061f837cc"};
    bool all = true;
    for (const char* id : faces) {
        void* face = nullptr;
        all = all && query(&cls, parse_guid(id), &face) == S_OK && face == ui;
    }
    check(all, "IXGameUiImpl..IXGameUiImpl4 (ids as the docs print them) answer with the same object");
    void* none = reinterpret_cast<void*>(1);
    check(query(&cls, parse_guid("0dd112ac-7c24-448c-b92b-3960fb5bd30c"), &none) == E_NOINTERFACE && none == nullptr,
          "an interface the class doesn't have is E_NOINTERFACE with a cleared out pointer");

    // ---- XGameUiShowMessageDialogAsync / Result (3, 4) ----
    check(answer(ui, "OK", nullptr, nullptr, 0) == 0, "a one-button dialog is answered with First");
    check(answer(ui, "Yes", "No", nullptr, 1) == 1, "a dialog with default Second is answered with Second");
    check(answer(ui, "A", "B", "C", 2) == 2, "a three-button dialog with default Third is answered with Third");
    check(answer(ui, "Yes", "No", nullptr, 2) == 0, "a default naming a button without text falls back to First");
    XAsyncBlock no_button{};
    check(at<MessageAsync>(ui, 3)(ui, &no_button, "T", "C", nullptr, nullptr, nullptr, 0, 0) == E_INVALIDARG,
          "a dialog without a first button is E_INVALIDARG");

    // ---- XGameUiShowErrorDialogAsync / Result (14, 15) ----
    using ErrorAsync = HRESULT(STDMETHODCALLTYPE*)(void*, XAsyncBlock*, HRESULT, const char*);
    using CloseResult = HRESULT(STDMETHODCALLTYPE*)(void*, XAsyncBlock*);
    XAsyncBlock error{};
    check(at<ErrorAsync>(ui, 14)(ui, &error, E_FAIL, "synthetic error") == S_OK && finished(&error) &&
              at<CloseResult>(ui, 15)(ui, &error) == S_OK,
          "XGameUiShowErrorDialogAsync is logged and closed: Result S_OK");

    // ---- dialogs shown and closed at once (5-10, 30-31, 40-41) ----
    using UserAsync = HRESULT(STDMETHODCALLTYPE*)(void*, XAsyncBlock*, void*, uint64_t);
    struct Pair {
        int async, result;
        const char* what;
    };
    const Pair closed[] = {{5, 6, "XGameUiShowSendGameInvite"},      {7, 8, "XGameUiShowPlayerProfileCard"},
                           {9, 10, "XGameUiShowAchievements"},      {30, 31, "XGameUiShowMultiplayerActivityGameInvite (2)"},
                           {40, 41, "XGameUiShowStateShare (3)"}};
    for (const Pair& p : closed) {
        XAsyncBlock async{};
        char what[128];
        std::snprintf(what, sizeof(what), "%s: shown and closed, Result S_OK", p.what);
        check(at<UserAsync>(ui, p.async)(ui, &async, nullptr, 0) == S_OK && finished(&async) &&
                  at<CloseResult>(ui, p.result)(ui, &async) == S_OK,
              what);
    }

    // ---- XGameUiShowTextEntryAsync / ResultSize / Result (17-19) ----
    using TextAsync = HRESULT(STDMETHODCALLTYPE*)(void*, XAsyncBlock*, const char*, const char*, const char*, uint32_t, uint32_t);
    using TextSize = HRESULT(STDMETHODCALLTYPE*)(void*, XAsyncBlock*, uint32_t*);
    using TextResult = HRESULT(STDMETHODCALLTYPE*)(void*, XAsyncBlock*, uint32_t, char*, uint32_t*);
    XAsyncBlock entry{};
    uint32_t size = 0, used = 0;
    char buf[32] = {};
    check(at<TextAsync>(ui, 17)(ui, &entry, "Name", "Save name", "Save 1", 0, 16) == S_OK && finished(&entry) &&
              at<TextSize>(ui, 18)(ui, &entry, &size) == S_OK && size == 7,
          "XGameUiShowTextEntryResultSize: the default text and its terminator (7)");
    check(at<TextResult>(ui, 19)(ui, &entry, sizeof(buf), buf, &used) == S_OK && std::strcmp(buf, "Save 1") == 0 && used == 7,
          "XGameUiShowTextEntryResult: the default text (OK pressed without typing)");
    XAsyncBlock cut{};
    std::memset(buf, 0, sizeof(buf));
    check(at<TextAsync>(ui, 17)(ui, &cut, "Name", nullptr, "Save 1", 0, 4) == S_OK && finished(&cut) &&
              at<TextResult>(ui, 19)(ui, &cut, sizeof(buf), buf, &used) == S_OK && std::strcmp(buf, "Save") == 0,
          "a default text longer than maxTextLength is cut to it");
    XAsyncBlock empty{};
    check(at<TextAsync>(ui, 17)(ui, &empty, "Name", nullptr, nullptr, 0, 16) == S_OK && finished(&empty) &&
              at<TextSize>(ui, 18)(ui, &empty, &size) == S_OK && size == 1 &&
              at<TextResult>(ui, 19)(ui, &empty, sizeof(buf), buf, &used) == S_OK && buf[0] == 0 && used == 1,
          "no default text: an empty answer");
    XAsyncBlock small{};
    check(at<TextAsync>(ui, 17)(ui, &small, "Name", nullptr, "Save 1", 0, 16) == S_OK && finished(&small) &&
              FAILED(at<TextResult>(ui, 19)(ui, &small, 3, buf, &used)) && used == 0,
          "a buffer shorter than ResultSize fails");

    // ---- XGameUiSetNotificationPositionHint (16) ----
    using Position = HRESULT(STDMETHODCALLTYPE*)(void*, uint32_t);
    check(at<Position>(ui, 16)(ui, 0) == S_OK && at<Position>(ui, 16)(ui, 5) == S_OK && at<Position>(ui, 16)(ui, 6) == E_INVALIDARG,
          "XGameUiSetNotificationPositionHint: BottomCenter..TopRight accepted, anything else E_INVALIDARG");

    // ---- network-only dialogs fail before starting (11, 24) ----
    using Picker = HRESULT(STDMETHODCALLTYPE*)(void*, XAsyncBlock*, void*, const char*, uint32_t, const uint64_t*, uint32_t,
                                               const uint64_t*, uint32_t, uint32_t);
    using WebAuth = HRESULT(STDMETHODCALLTYPE*)(void*, XAsyncBlock*, void*, const char*, const char*);
    XAsyncBlock picker{}, web{};
    check(at<Picker>(ui, 11)(ui, &picker, nullptr, "Pick", 0, nullptr, 0, nullptr, 1, 1) == kNetworkUnreachable,
          "XGameUiShowPlayerPickerAsync fails before starting (no friends list offline)");
    check(at<WebAuth>(ui, 24)(ui, &web, nullptr, "https://example.invalid/", "https://example.invalid/done") ==
              kNetworkUnreachable,
          "XGameUiShowWebAuthenticationAsync fails before starting");

    // ---- E_NOTIMPL: unknown slots, in-game text entry, custom-UI callbacks (20, 34, 42) ----
    using Unknown = HRESULT(STDMETHODCALLTYPE*)(void*, void*, void*);
    void* handle = reinterpret_cast<void*>(1);
    check(at<Unknown>(ui, 20)(ui, nullptr, nullptr) == E_NOTIMPL, "slot 20 (no public signature): E_NOTIMPL");
    using TextOpen = HRESULT(STDMETHODCALLTYPE*)(void*, const void*, uint32_t, const char*, uint32_t, void**);
    check(at<TextOpen>(ui, 34)(ui, nullptr, 16, "", 0, &handle) == E_NOTIMPL,
          "XGameUiTextEntryOpen (2): E_NOTIMPL, so a title falls back to XGameUiShowTextEntryAsync");
    using SetCallbacks = HRESULT(STDMETHODCALLTYPE*)(void*, const void*, bool);
    check(at<SetCallbacks>(ui, 42)(ui, nullptr, true) == E_NOTIMPL, "XGameUiSetUiCallbacks (4): E_NOTIMPL");

    std::printf("xgameruntime_ui_test: %d of %d checks failed\n", g_failures, g_checks);
    return g_failures ? 1 : 0;
}
