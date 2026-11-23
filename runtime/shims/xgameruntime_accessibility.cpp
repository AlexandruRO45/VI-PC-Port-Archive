// xgameruntime: XAccessibilityImpl, the class xgameruntime.lib's XClosedCaption*, XHighContrast*, XSpeechToText* and
// XSpeechSynthesizer* functions reach through QueryApiImpl, offline.
//
// Each thunk asks QueryApiImpl for XAccessibilityImpl and calls one slot with the function's own arguments after
// `this` (docs/runtime_imports.md, "Class layouts, measured"). The class and interface ids and the slot order are
// xodus-gaming/xgameruntime-docs' "Layout" list of IXAccessibilityImpl (GPL-3.0; facts only, nothing copied);
// IXAccessibilityImpl2 adds no slot the docs know of. Signatures, structure layouts and enum values follow
// Microsoft's public GDK reference (learn.microsoft.com/gaming/gdk, XAccessibility and XSpeechSynthesizer). No GDK
// header or library code is used.
//
// A PC with no console accessibility settings answers like a console with every feature off:
//   - closed captions are off (XClosedCaptionProperties::Enabled false, scale 1, white font, no background) until the
//     title flags them with XClosedCaptionSetEnabled, which the next XClosedCaptionGetProperties reports;
//   - the high contrast mode is Off;
//   - speech-to-text calls are accepted and dropped: there is no overlay to draw the text on, and the title's own text
//     is never written to the log (it can be game text). A hypothesis id is handed out by Begin and checked by
//     Update/Finalize/Cancel;
//   - no voice is installed: the enumeration calls back zero times, a synthesizer handle can be made and closed, and
//     the calls that would produce audio (CreateStreamFromText/Ssml, SetCustomVoice) fail with
//     HRESULT_FROM_WIN32(ERROR_NOT_SUPPORTED), the error QueryApiImpl gives for a class the runtime doesn't provide.
//     Nothing is synthesized and no host speech engine is used.
//
// ASSUMPTION: the console executable is not available, so which of these slots it calls is unknown until its first
// boot. The slots are not measured against a caller yet. XColor's layout is not in the pages read (an in-memory
// four-byte value is assumed); only the opaque-white font colour is written, which reads the same in any channel
// order, and the other two colours are left zero.
#include <windows.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iterator>
#include <mutex>
#include <set>

namespace {

// ---- Public GDK types (XAccessibility.h and XSpeechSynthesizer.h, as the GDK reference documents them) ----

struct XColor4 {   // XColor: four bytes (channel order not read from the reference)
    uint8_t b[4];
};
struct XClosedCaptionProperties {
    XColor4 BackgroundColor;
    XColor4 FontColor;
    XColor4 WindowColor;
    uint32_t FontEdgeAttribute;   // XClosedCaptionFontEdgeAttribute: Default = 0
    uint32_t FontStyle;           // XClosedCaptionFontStyle: Default = 0
    float FontScale;
    bool Enabled;
};
static_assert(sizeof(XClosedCaptionProperties) == 28, "three four-byte colours, two enums, a float and a bool");

constexpr uint32_t kHighContrastOff = 0;           // XHighContrastMode::Off
constexpr uint32_t kSpeechToTextTypeMax = 1;       // XSpeechToTextType: Voice = 0, Text = 1
constexpr uint32_t kSpeechToTextPositionMax = 7;   // XSpeechToTextPositionHint: BottomCenter = 0 .. TopRight = 7

constexpr HRESULT kNotSupported = HRESULT_FROM_WIN32(ERROR_NOT_SUPPORTED);

using VoicesCallback = BOOL(CALLBACK*)(const void* voice, void* context);

// ---- state ----

std::atomic<bool> g_captions_enabled{false};

std::mutex g_mutex;
std::set<uintptr_t> g_synthesizers;   // handles handed out and not closed
std::set<uint32_t> g_hypotheses;      // ids handed out by Begin and not finalized or cancelled
uintptr_t g_next_handle = 1;
uint32_t g_next_hypothesis = 1;

// ---- IXAccessibilityImpl (slots 3-21) ----

// 3: HRESULT XClosedCaptionGetProperties(XClosedCaptionProperties* properties)
HRESULT STDMETHODCALLTYPE cc_get_properties(void*, XClosedCaptionProperties* properties) {
    if (!properties) return E_INVALIDARG;
    std::memset(properties, 0, sizeof *properties);
    std::memset(properties->FontColor.b, 0xFF, sizeof properties->FontColor.b);
    properties->FontScale = 1.0f;
    properties->Enabled = g_captions_enabled.load();
    return S_OK;
}

// 4: HRESULT XClosedCaptionSetEnabled(bool enabled)
HRESULT STDMETHODCALLTYPE cc_set_enabled(void*, bool enabled) {
    g_captions_enabled = enabled;
    return S_OK;
}

// 5: HRESULT XHighContrastGetMode(XHighContrastMode* mode)
HRESULT STDMETHODCALLTYPE high_contrast_get_mode(void*, uint32_t* mode) {
    if (!mode) return E_INVALIDARG;
    *mode = kHighContrastOff;
    return S_OK;
}

// 6: HRESULT XSpeechToTextSetPositionHint(XSpeechToTextPositionHint position)
HRESULT STDMETHODCALLTYPE stt_set_position_hint(void*, uint32_t position) {
    return position <= kSpeechToTextPositionMax ? S_OK : E_INVALIDARG;
}

// 7: HRESULT XSpeechToTextSendString(const char* speakerName, const char* content, XSpeechToTextType type)
HRESULT STDMETHODCALLTYPE stt_send_string(void*, const char* speaker, const char* content, uint32_t type) {
    if (!speaker || !content || type > kSpeechToTextTypeMax) return E_INVALIDARG;
    return S_OK;
}

// 8: HRESULT XSpeechSynthesizerEnumerateInstalledVoices(void* context, XSpeechSynthesizerInstalledVoicesCallback* callback)
// The callback is called once per installed voice: none here.
HRESULT STDMETHODCALLTYPE speech_enumerate_voices(void*, void*, VoicesCallback callback) {
    return callback ? S_OK : E_INVALIDARG;
}

bool known_synthesizer(uintptr_t handle) {
    std::lock_guard lock(g_mutex);
    return g_synthesizers.count(handle) != 0;
}

// 9: HRESULT XSpeechSynthesizerCreate(XSpeechSynthesizerHandle* speechSynthesizer)
HRESULT STDMETHODCALLTYPE speech_create(void*, uintptr_t* handle) {
    if (!handle) return E_INVALIDARG;
    std::lock_guard lock(g_mutex);
    *handle = g_next_handle++;
    g_synthesizers.insert(*handle);
    return S_OK;
}

// 10: HRESULT XSpeechSynthesizerCloseHandle(XSpeechSynthesizerHandle speechSynthesizer); the handle is _In_opt_.
HRESULT STDMETHODCALLTYPE speech_close(void*, uintptr_t handle) {
    if (!handle) return S_OK;
    std::lock_guard lock(g_mutex);
    return g_synthesizers.erase(handle) ? S_OK : E_INVALIDARG;
}

// 11: HRESULT XSpeechSynthesizerSetDefaultVoice(XSpeechSynthesizerHandle speechSynthesizer)
HRESULT STDMETHODCALLTYPE speech_set_default_voice(void*, uintptr_t handle) {
    return known_synthesizer(handle) ? S_OK : E_INVALIDARG;
}

// 12: HRESULT XSpeechSynthesizerSetCustomVoice(XSpeechSynthesizerHandle speechSynthesizer, const char* voiceId)
// The voice id names an installed voice, and there is none.
HRESULT STDMETHODCALLTYPE speech_set_custom_voice(void*, uintptr_t handle, const char* voice_id) {
    if (!known_synthesizer(handle) || !voice_id) return E_INVALIDARG;
    return kNotSupported;
}

// 13 and 21: HRESULT XSpeechSynthesizerCreateStreamFromText / FromSsml(XSpeechSynthesizerHandle speechSynthesizer,
//     const char* text or ssml, XSpeechSynthesizerStreamHandle* speechSynthesisStream)
HRESULT STDMETHODCALLTYPE speech_create_stream(void*, uintptr_t handle, const char* text, uintptr_t* stream) {
    if (stream) *stream = 0;
    if (!known_synthesizer(handle) || !text || !stream) return E_INVALIDARG;
    return kNotSupported;
}

// 14: HRESULT XSpeechSynthesizerCloseStreamHandle(XSpeechSynthesizerStreamHandle speechSynthesisStream)
// No stream can exist, so any handle is unknown.
HRESULT STDMETHODCALLTYPE speech_close_stream(void*, uintptr_t) { return E_INVALIDARG; }

// 15: HRESULT XSpeechSynthesizerGetStreamDataSize(XSpeechSynthesizerStreamHandle, size_t* bufferSize)
HRESULT STDMETHODCALLTYPE speech_stream_data_size(void*, uintptr_t, size_t* size) {
    if (size) *size = 0;
    return E_INVALIDARG;
}

// 16: HRESULT XSpeechSynthesizerGetStreamData(XSpeechSynthesizerStreamHandle, size_t bufferSize, void* buffer,
//     size_t* bufferUsed)
HRESULT STDMETHODCALLTYPE speech_stream_data(void*, uintptr_t, size_t, void*, size_t* used) {
    if (used) *used = 0;
    return E_INVALIDARG;
}

// 17: HRESULT XSpeechToTextBeginHypothesisString(const char* speakerName, const char* content, XSpeechToTextType type,
//     uint32_t* hypothesisId)
HRESULT STDMETHODCALLTYPE stt_begin_hypothesis(void*, const char* speaker, const char* content, uint32_t type, uint32_t* id) {
    if (id) *id = 0;
    if (!speaker || !content || type > kSpeechToTextTypeMax || !id) return E_INVALIDARG;
    std::lock_guard lock(g_mutex);
    *id = g_next_hypothesis++;
    g_hypotheses.insert(*id);
    return S_OK;
}

// 18 and 19: HRESULT XSpeechToTextUpdateHypothesisString / FinalizeHypothesisString(uint32_t hypothesisId,
//     const char* content). Finalize ends the hypothesis.
template <bool Finalize>
HRESULT STDMETHODCALLTYPE stt_change_hypothesis(void*, uint32_t id, const char* content) {
    if (!content) return E_INVALIDARG;
    std::lock_guard lock(g_mutex);
    if (!g_hypotheses.count(id)) return E_INVALIDARG;
    if (Finalize) g_hypotheses.erase(id);
    return S_OK;
}

// 20: HRESULT XSpeechToTextCancelHypothesisString(uint32_t hypothesisId)
HRESULT STDMETHODCALLTYPE stt_cancel_hypothesis(void*, uint32_t id) {
    std::lock_guard lock(g_mutex);
    return g_hypotheses.erase(id) ? S_OK : E_INVALIDARG;
}

// ---- the object ----

constexpr GUID kIUnknown = {0x00000000, 0x0000, 0x0000, {0xc0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x46}};
constexpr GUID kClass = {0x3e241977, 0x6237, 0x41e9, {0x85, 0x59, 0x10, 0x2c, 0x2d, 0x99, 0x83, 0xf1}};    // = IXAccessibilityImpl
constexpr GUID kInterface2 = {0xd722b373, 0x8c4d, 0x4692, {0x9e, 0x51, 0xd6, 0xad, 0x9b, 0x37, 0xaa, 0x7d}};  // IXAccessibilityImpl2

struct Object {
    const void* const* vtable;
};

HRESULT STDMETHODCALLTYPE object_query_interface(void* self, REFIID iid, void** out) {
    if (!out) return E_POINTER;
    if (iid == kIUnknown || iid == kClass || iid == kInterface2) {
        *out = self;
        return S_OK;
    }
    *out = nullptr;
    return E_NOINTERFACE;
}
// A process-wide singleton: references are not counted, the object is never freed.
ULONG STDMETHODCALLTYPE object_add_ref(void*) { return 2; }
ULONG STDMETHODCALLTYPE object_release(void*) { return 1; }

#define SLOT(fn) reinterpret_cast<const void*>(&fn)
const void* const kVtable[] = {
    SLOT(object_query_interface), SLOT(object_add_ref), SLOT(object_release),
    SLOT(cc_get_properties),                   // 3  XClosedCaptionGetProperties
    SLOT(cc_set_enabled),                      // 4  XClosedCaptionSetEnabled
    SLOT(high_contrast_get_mode),              // 5  XHighContrastGetMode
    SLOT(stt_set_position_hint),               // 6  XSpeechToTextSetPositionHint
    SLOT(stt_send_string),                     // 7  XSpeechToTextSendString
    SLOT(speech_enumerate_voices),             // 8  XSpeechSynthesizerEnumerateInstalledVoices
    SLOT(speech_create),                       // 9  XSpeechSynthesizerCreate
    SLOT(speech_close),                        // 10 XSpeechSynthesizerCloseHandle
    SLOT(speech_set_default_voice),            // 11 XSpeechSynthesizerSetDefaultVoice
    SLOT(speech_set_custom_voice),             // 12 XSpeechSynthesizerSetCustomVoice
    SLOT(speech_create_stream),                // 13 XSpeechSynthesizerCreateStreamFromText
    SLOT(speech_close_stream),                 // 14 XSpeechSynthesizerCloseStreamHandle
    SLOT(speech_stream_data_size),             // 15 XSpeechSynthesizerGetStreamDataSize
    SLOT(speech_stream_data),                  // 16 XSpeechSynthesizerGetStreamData
    SLOT(stt_begin_hypothesis),                // 17 XSpeechToTextBeginHypothesisString
    SLOT(stt_change_hypothesis<false>),        // 18 XSpeechToTextUpdateHypothesisString
    SLOT(stt_change_hypothesis<true>),         // 19 XSpeechToTextFinalizeHypothesisString
    SLOT(stt_cancel_hypothesis),               // 20 XSpeechToTextCancelHypothesisString
    SLOT(speech_create_stream),                // 21 XSpeechSynthesizerCreateStreamFromSsml
};
#undef SLOT
static_assert(std::size(kVtable) == 22, "IUnknown + IXAccessibilityImpl's 19 methods");

Object g_object{kVtable};

}  // namespace

namespace rt::shims {

// Called by xgameruntime.cpp's QueryApiImpl for every class: true when the class is XAccessibilityImpl, with *hr set.
bool xgameruntime_accessibility_query(const GUID& class_id, REFIID interface_id, void** out, HRESULT* hr) {
    if (class_id != kClass) return false;
    *hr = object_query_interface(&g_object, interface_id, out);
    return true;
}

}  // namespace rt::shims
