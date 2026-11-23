// XAccessibilityImpl through QueryApiImpl, called the way xgameruntime.lib's XClosedCaption*, XHighContrast*,
// XSpeechToText* and XSpeechSynthesizer* thunks call it: QueryApiImpl with the class id (or an interface id), one
// vtable slot with `this` first. Slots follow xgameruntime-docs' layout; arguments and expected values follow the
// public GDK reference. No game data; nothing online.
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

// XClosedCaptionProperties as the GDK reference lists it: three colours, two enums, a float, a bool.
struct CaptionProperties {
    uint8_t background[4];
    uint8_t font[4];
    uint8_t window[4];
    uint32_t edge;
    uint32_t style;
    float scale;
    bool enabled;
};

using GetProps = HRESULT(STDMETHODCALLTYPE*)(void*, CaptionProperties*);
using SetBool = HRESULT(STDMETHODCALLTYPE*)(void*, bool);
using GetMode = HRESULT(STDMETHODCALLTYPE*)(void*, uint32_t*);
using SetU32 = HRESULT(STDMETHODCALLTYPE*)(void*, uint32_t);
using SendString = HRESULT(STDMETHODCALLTYPE*)(void*, const char*, const char*, uint32_t);
using Enumerate = HRESULT(STDMETHODCALLTYPE*)(void*, void*, BOOL(CALLBACK*)(const void*, void*));
using Create = HRESULT(STDMETHODCALLTYPE*)(void*, uintptr_t*);
using Handle = HRESULT(STDMETHODCALLTYPE*)(void*, uintptr_t);
using CustomVoice = HRESULT(STDMETHODCALLTYPE*)(void*, uintptr_t, const char*);
using Stream = HRESULT(STDMETHODCALLTYPE*)(void*, uintptr_t, const char*, uintptr_t*);
using StreamSize = HRESULT(STDMETHODCALLTYPE*)(void*, uintptr_t, size_t*);
using StreamData = HRESULT(STDMETHODCALLTYPE*)(void*, uintptr_t, size_t, void*, size_t*);
using Begin = HRESULT(STDMETHODCALLTYPE*)(void*, const char*, const char*, uint32_t, uint32_t*);
using Change = HRESULT(STDMETHODCALLTYPE*)(void*, uint32_t, const char*);

int g_voices = 0;
BOOL CALLBACK count_voice(const void*, void*) {
    ++g_voices;
    return TRUE;
}

}  // namespace

int main() {
    std::printf("xgameruntime_accessibility_test: XAccessibilityImpl through QueryApiImpl\n");
    auto query = reinterpret_cast<Query>(rt::shims::find("xgameruntime.dll", "QueryApiImpl"));
    check(query != nullptr, "QueryApiImpl is registered");
    if (!query) return 1;

    // ---- the object and its interfaces ----
    const GUID cls = parse_guid("3e241977-6237-41e9-8559-102c2d9983f1");
    void* a = nullptr;
    check(query(&cls, cls, &a) == S_OK && a, "QueryApiImpl hands out XAccessibilityImpl for its class id (as the thunks ask)");
    if (!a) return 1;
    void* a2 = nullptr;
    check(query(&cls, parse_guid("d722b373-8c4d-4692-9e51-d6ad9b37aa7d"), &a2) == S_OK && a2 == a,
          "IXAccessibilityImpl2 (id as the docs print it) answers with the same object");
    void* none = reinterpret_cast<void*>(1);
    check(query(&cls, parse_guid("0dd112ac-7c24-448c-b92b-3960fb5bd30c"), &none) == E_NOINTERFACE && none == nullptr,
          "an interface the class doesn't have is E_NOINTERFACE with a cleared out pointer");
    check(at<ULONG(STDMETHODCALLTYPE*)(void*)>(a, 1)(a) > 0 && at<ULONG(STDMETHODCALLTYPE*)(void*)>(a, 2)(a) > 0,
          "AddRef and Release answer");

    // ---- closed captions ----
    CaptionProperties p;
    std::memset(&p, 0xCD, sizeof p);
    check(at<GetProps>(a, 3)(a, &p) == S_OK, "XClosedCaptionGetProperties succeeds");
    check(!p.enabled && p.scale == 1.0f && p.edge == 0 && p.style == 0, "captions start off, scale 1, default edge and style");
    check(p.font[0] == 0xFF && p.font[1] == 0xFF && p.font[2] == 0xFF && p.font[3] == 0xFF, "the font colour is opaque white");
    uint8_t zero[4] = {};
    check(!std::memcmp(p.background, zero, 4) && !std::memcmp(p.window, zero, 4), "background and window colours are zero");
    check(at<GetProps>(a, 3)(a, nullptr) == E_INVALIDARG, "XClosedCaptionGetProperties(null) is E_INVALIDARG");
    check(at<SetBool>(a, 4)(a, true) == S_OK, "XClosedCaptionSetEnabled(true) succeeds");
    check(at<GetProps>(a, 3)(a, &p) == S_OK && p.enabled, "the next XClosedCaptionGetProperties reports captions on");
    check(at<SetBool>(a, 4)(a, false) == S_OK && at<GetProps>(a, 3)(a, &p) == S_OK && !p.enabled,
          "and off again after XClosedCaptionSetEnabled(false)");

    // ---- high contrast ----
    uint32_t mode = 99;
    check(at<GetMode>(a, 5)(a, &mode) == S_OK && mode == 0, "XHighContrastGetMode reports Off (0)");
    check(at<GetMode>(a, 5)(a, nullptr) == E_INVALIDARG, "XHighContrastGetMode(null) is E_INVALIDARG");

    // ---- speech to text ----
    bool all_positions = true;
    for (uint32_t pos = 0; pos <= 7; ++pos) all_positions = all_positions && at<SetU32>(a, 6)(a, pos) == S_OK;
    check(all_positions, "XSpeechToTextSetPositionHint accepts BottomCenter (0) through TopRight (7)");
    check(at<SetU32>(a, 6)(a, 8) == E_INVALIDARG, "a position past TopRight is E_INVALIDARG");
    check(at<SendString>(a, 7)(a, "Speaker", "text", 0) == S_OK && at<SendString>(a, 7)(a, "Speaker", "text", 1) == S_OK,
          "XSpeechToTextSendString takes the Voice and Text types");
    check(at<SendString>(a, 7)(a, "Speaker", "text", 2) == E_INVALIDARG && at<SendString>(a, 7)(a, nullptr, "text", 0) == E_INVALIDARG &&
              at<SendString>(a, 7)(a, "Speaker", nullptr, 0) == E_INVALIDARG,
          "an unknown type or a null string is E_INVALIDARG");

    uint32_t id1 = 0, id2 = 0;
    check(at<Begin>(a, 17)(a, "S", "one", 0, &id1) == S_OK && id1 != 0, "XSpeechToTextBeginHypothesisString returns an id");
    check(at<Begin>(a, 17)(a, "S", "two", 1, &id2) == S_OK && id2 != 0 && id2 != id1, "a second hypothesis gets another id");
    check(at<Change>(a, 18)(a, id1, "one more") == S_OK, "XSpeechToTextUpdateHypothesisString on a live id succeeds");
    check(at<Change>(a, 19)(a, id1, "one final") == S_OK, "XSpeechToTextFinalizeHypothesisString ends it");
    check(at<Change>(a, 18)(a, id1, "late") == E_INVALIDARG && at<Change>(a, 19)(a, id1, "late") == E_INVALIDARG,
          "a finalized id is no longer valid for Update or Finalize");
    check(at<SetU32>(a, 20)(a, id2) == S_OK && at<SetU32>(a, 20)(a, id2) == E_INVALIDARG,
          "XSpeechToTextCancelHypothesisString cancels a live id once");
    check(at<Change>(a, 18)(a, 0, "x") == E_INVALIDARG && at<Change>(a, 18)(a, id2 + 1000, "x") == E_INVALIDARG,
          "an id that was never handed out is E_INVALIDARG");
    uint32_t id3 = 7;
    check(at<Begin>(a, 17)(a, "S", "three", 5, &id3) == E_INVALIDARG && id3 == 0, "Begin with an unknown type fails and clears the id");

    // ---- speech synthesizer ----
    g_voices = 0;
    check(at<Enumerate>(a, 8)(a, nullptr, count_voice) == S_OK && g_voices == 0, "no voice is installed: the callback is never called");
    check(at<Enumerate>(a, 8)(a, nullptr, nullptr) == E_INVALIDARG, "a null callback is E_INVALIDARG");

    uintptr_t synth = 0;
    check(at<Create>(a, 9)(a, &synth) == S_OK && synth != 0, "XSpeechSynthesizerCreate returns a handle");
    check(at<Create>(a, 9)(a, nullptr) == E_INVALIDARG, "XSpeechSynthesizerCreate(null) is E_INVALIDARG");
    check(at<Handle>(a, 11)(a, synth) == S_OK, "XSpeechSynthesizerSetDefaultVoice on it succeeds");
    check(at<CustomVoice>(a, 12)(a, synth, "Some Voice") == kNotSupported, "a custom voice that is not installed is not supported");
    check(at<CustomVoice>(a, 12)(a, synth, nullptr) == E_INVALIDARG, "a null voice id is E_INVALIDARG");

    uintptr_t stream = 7;
    check(at<Stream>(a, 13)(a, synth, "hello", &stream) == kNotSupported && stream == 0,
          "XSpeechSynthesizerCreateStreamFromText is not supported (nothing is synthesized) and clears the handle");
    stream = 7;
    check(at<Stream>(a, 21)(a, synth, "<speak/>", &stream) == kNotSupported && stream == 0,
          "XSpeechSynthesizerCreateStreamFromSsml is not supported either");
    check(at<Stream>(a, 13)(a, 0, "hello", &stream) == E_INVALIDARG, "a null synthesizer is E_INVALIDARG");
    check(at<Stream>(a, 13)(a, synth, nullptr, &stream) == E_INVALIDARG, "a null text is E_INVALIDARG");
    size_t size = 5, used = 5;
    char buffer[4];
    check(at<StreamSize>(a, 15)(a, 1, &size) == E_INVALIDARG && size == 0, "no stream exists to size");
    check(at<StreamData>(a, 16)(a, 1, sizeof buffer, buffer, &used) == E_INVALIDARG && used == 0, "no stream exists to read");
    check(at<Handle>(a, 14)(a, 1) == E_INVALIDARG, "no stream exists to close");

    check(at<Handle>(a, 10)(a, synth) == S_OK, "XSpeechSynthesizerCloseHandle closes it");
    check(at<Handle>(a, 10)(a, synth) == E_INVALIDARG, "a handle closed twice is E_INVALIDARG");
    check(at<Handle>(a, 11)(a, synth) == E_INVALIDARG, "a closed synthesizer can no longer be used");
    check(at<Handle>(a, 10)(a, 0) == S_OK, "closing a null handle is accepted (_In_opt_)");

    std::printf("%d checks, %d failures\n", g_checks, g_failures);
    return g_failures ? 1 : 0;
}
