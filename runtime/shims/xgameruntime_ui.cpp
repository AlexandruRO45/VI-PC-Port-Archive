// xgameruntime: XGameUiImpl, the class xgameruntime.lib's XGameUi* functions reach through QueryApiImpl, offline.
//
// Each XGameUi* thunk asks QueryApiImpl for XGameUiImpl and calls one slot with the function's own arguments after
// `this` (docs/runtime_imports.md, "Class layouts, measured"). The class and interface ids and the slot order are
// xodus-gaming/xgameruntime-docs' "Layout" lists (GPL-3.0; facts only, nothing copied). Signatures and enum values
// follow Microsoft's public GDK reference (learn.microsoft.com/gaming/gdk, the XGameUi pages). No GDK header or
// library code is used.
//
// There is no system UI here, so each dialog is answered at once, the way a player would answer it without reading:
//   - a message dialog is printed to the log and answered with its default button;
//   - an error dialog is printed to the log and closed;
//   - text entry is answered with the title's default text (OK pressed without typing), cut to maxTextLength;
//   - the profile card, achievements, game invite, multiplayer activity invite and state share are shown and closed
//     at once (S_OK);
//   - the player picker and web authentication need the network (friends, a sign-in page): they fail before starting
//     with HRESULT_FROM_WIN32(ERROR_NETWORK_UNREACHABLE), so no XAsyncBlock is left waiting.
// The in-game text-entry handles (IXGameUiImpl2) and the custom-UI callbacks (IXGameUiImpl4) answer E_NOTIMPL, so a
// title falls back to the system dialogs above. The point is that a title reporting a problem through a system dialog
// doesn't hang waiting for an answer, and that its message appears in the log.
//
// ASSUMPTION: the console executable is not available, so which of these slots it calls is unknown until its first
// boot. Each slot without an offline answer logs once when called, so that boot lists them.
#include <windows.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iterator>
#include <string>

#include <XAsync.h>
#include <XAsyncProvider.h>
#include <XTaskQueue.h>

namespace {

// ---- Public GDK types (XGameUI.h, as the GDK reference documents them) ----

enum class XGameUiMessageDialogButton : uint32_t { First = 0, Second = 1, Third = 2 };

constexpr HRESULT kNetworkUnreachable = HRESULT_FROM_WIN32(ERROR_NETWORK_UNREACHABLE);

// Strings from the title are printed as given; a null one as "".
const char* text(const char* s) { return s ? s : ""; }

// ---- XAsync providers ----

// A dialog shown and closed at once: completes with no payload, so its Result can be read any number of times.
struct Closed {
    const char* name;
};
const Closed kSendGameInvite{"XGameUiShowSendGameInviteAsync"};
const Closed kProfileCard{"XGameUiShowPlayerProfileCardAsync"};
const Closed kAchievements{"XGameUiShowAchievementsAsync"};
const Closed kErrorDialog{"XGameUiShowErrorDialogAsync"};
const Closed kMultiplayerInvite{"XGameUiShowMultiplayerActivityGameInviteAsync"};
const Closed kStateShare{"XGameUiShowStateShareAsync"};

HRESULT CALLBACK closed_provider(XAsyncOp op, const XAsyncProviderData* data) {
    if (op == XAsyncOp::Begin) return XAsyncSchedule(data->async, 0);
    if (op == XAsyncOp::DoWork) XAsyncComplete(data->async, S_OK, 0);
    return S_OK;
}

HRESULT begin_closed(XAsyncBlock* async, const Closed& which) {
    if (!async) return E_POINTER;
    return XAsyncBegin(async, nullptr, &which, which.name, closed_provider);
}
template <const Closed& Which> HRESULT STDMETHODCALLTYPE closed_result(void*, XAsyncBlock* async) {
    return XAsyncGetResult(async, &Which, 0, nullptr, nullptr);
}

// A message dialog: the context carries the answer; the payload is the XGameUiMessageDialogButton.
const char kMessageDialogIdentity = 0;
HRESULT CALLBACK message_provider(XAsyncOp op, const XAsyncProviderData* data) {
    switch (op) {
    case XAsyncOp::Begin:
        return XAsyncSchedule(data->async, 0);
    case XAsyncOp::DoWork:
        XAsyncComplete(data->async, S_OK, sizeof(XGameUiMessageDialogButton));
        return S_OK;
    case XAsyncOp::GetResult:
        if (!data->buffer || data->bufferSize < sizeof(XGameUiMessageDialogButton)) return E_INVALIDARG;
        *static_cast<XGameUiMessageDialogButton*>(data->buffer) =
            static_cast<XGameUiMessageDialogButton>(reinterpret_cast<uintptr_t>(data->context));
        return S_OK;
    default:
        return S_OK;
    }
}

// Text entry: the context is a heap copy of the answer, freed when the call is cleaned up; the payload is the text
// and its terminator.
const char kTextEntryIdentity = 0;
HRESULT CALLBACK text_entry_provider(XAsyncOp op, const XAsyncProviderData* data) {
    auto* answer = static_cast<std::string*>(data->context);
    switch (op) {
    case XAsyncOp::Begin:
        return XAsyncSchedule(data->async, 0);
    case XAsyncOp::DoWork:
        XAsyncComplete(data->async, S_OK, answer->size() + 1);
        return S_OK;
    case XAsyncOp::GetResult:
        if (!data->buffer || data->bufferSize < answer->size() + 1) return E_INVALIDARG;
        std::memcpy(data->buffer, answer->c_str(), answer->size() + 1);
        return S_OK;
    case XAsyncOp::Cleanup:
        delete answer;
        return S_OK;
    default:
        return S_OK;
    }
}

// ---- IXGameUiImpl (slots 3-29) ----

// 3: HRESULT XGameUiShowMessageDialogAsync(XAsyncBlock* async, const char* titleText, const char* contentText,
//        const char* firstButtonText, const char* secondButtonText, const char* thirdButtonText,
//        XGameUiMessageDialogButton defaultButton, XGameUiMessageDialogButton cancelButton)
// Answered with the default button; a default naming a button without text falls back to the first.
HRESULT STDMETHODCALLTYPE ui_message_dialog_async(void*, XAsyncBlock* async, const char* title, const char* content,
                                                  const char* first, const char* second, const char* third,
                                                  XGameUiMessageDialogButton default_button,
                                                  XGameUiMessageDialogButton cancel_button) {
    if (!async) return E_POINTER;
    if (!first) return E_INVALIDARG;   // a dialog has at least one button
    const char* labels[] = {first, second, third};
    auto answer = default_button;
    if (static_cast<uint32_t>(answer) > 2 || !labels[static_cast<uint32_t>(answer)]) answer = XGameUiMessageDialogButton::First;
    std::fprintf(stderr,
                 "[rt] xgameruntime: message dialog \"%s\": %s\n"
                 "[rt]   buttons: [%s] [%s] [%s] (default %u, cancel %u); answering \"%s\"\n",
                 text(title), text(content), text(first), text(second), text(third), static_cast<unsigned>(default_button),
                 static_cast<unsigned>(cancel_button), labels[static_cast<uint32_t>(answer)]);
    return XAsyncBegin(async, reinterpret_cast<void*>(static_cast<uintptr_t>(answer)), &kMessageDialogIdentity,
                       "XGameUiShowMessageDialogAsync", message_provider);
}

// 4: HRESULT XGameUiShowMessageDialogResult(XAsyncBlock* async, XGameUiMessageDialogButton* resultButton)
HRESULT STDMETHODCALLTYPE ui_message_dialog_result(void*, XAsyncBlock* async, XGameUiMessageDialogButton* button) {
    if (!button) return E_POINTER;
    return XAsyncGetResult(async, &kMessageDialogIdentity, sizeof(*button), button, nullptr);
}

// 5: XGameUiShowSendGameInviteAsync(async, XUserHandle requestingUser, const char* sessionConfigurationId,
//        const char* sessionTemplateName, const char* sessionId, const char* invitationText,
//        const char* customActivationContext). Shown and closed: nobody to invite offline.
HRESULT STDMETHODCALLTYPE ui_send_invite_async(void*, XAsyncBlock* async) { return begin_closed(async, kSendGameInvite); }

// 7: XGameUiShowPlayerProfileCardAsync(async, XUserHandle requestingUser, uint64_t targetPlayer)
HRESULT STDMETHODCALLTYPE ui_profile_card_async(void*, XAsyncBlock* async) { return begin_closed(async, kProfileCard); }

// 9: XGameUiShowAchievementsAsync(async, XUserHandle requestingUser, uint32_t titleId)
HRESULT STDMETHODCALLTYPE ui_achievements_async(void*, XAsyncBlock* async) { return begin_closed(async, kAchievements); }

// 14: HRESULT XGameUiShowErrorDialogAsync(XAsyncBlock* async, HRESULT errorCode, const char* context)
HRESULT STDMETHODCALLTYPE ui_error_dialog_async(void*, XAsyncBlock* async, HRESULT error, const char* context) {
    if (!async) return E_POINTER;
    std::fprintf(stderr, "[rt] xgameruntime: error dialog 0x%08lx \"%s\"; closing it\n", static_cast<unsigned long>(error),
                 text(context));
    return begin_closed(async, kErrorDialog);
}

// 16: HRESULT XGameUiSetNotificationPositionHint(XGameUiNotificationPositionHint position): no notifications are
// drawn, so any of the six documented positions is accepted.
HRESULT STDMETHODCALLTYPE ui_set_notification_position(void*, uint32_t position) {
    return position <= 5 ? S_OK : E_INVALIDARG;   // BottomCenter = 0 ... TopRight = 5
}

// 17: HRESULT XGameUiShowTextEntryAsync(XAsyncBlock* async, const char* titleText, const char* descriptionText,
//        const char* defaultText, XGameUiTextEntryInputScope inputScope, uint32_t maxTextLength)
HRESULT STDMETHODCALLTYPE ui_text_entry_async(void*, XAsyncBlock* async, const char* title, const char* description,
                                              const char* default_text, uint32_t, uint32_t max_length) {
    if (!async) return E_POINTER;
    auto* answer = new std::string(text(default_text));
    if (answer->size() > max_length) answer->resize(max_length);
    std::fprintf(stderr, "[rt] xgameruntime: text entry \"%s\" (%s); answering the default text \"%s\"\n", text(title),
                 text(description), answer->c_str());
    const HRESULT hr = XAsyncBegin(async, answer, &kTextEntryIdentity, "XGameUiShowTextEntryAsync", text_entry_provider);
    if (FAILED(hr)) delete answer;   // no Cleanup runs for a call that never began
    return hr;
}

// 18: HRESULT XGameUiShowTextEntryResultSize(XAsyncBlock* async, uint32_t* resultTextBufferSize): with the terminator.
HRESULT STDMETHODCALLTYPE ui_text_entry_result_size(void*, XAsyncBlock* async, uint32_t* size) {
    if (!size) return E_POINTER;
    *size = 0;
    size_t bytes = 0;
    const HRESULT hr = XAsyncGetResultSize(async, &bytes);
    if (SUCCEEDED(hr)) *size = static_cast<uint32_t>(bytes);
    return hr;
}

// 19: HRESULT XGameUiShowTextEntryResult(XAsyncBlock* async, uint32_t resultTextBufferSize, char* resultTextBuffer,
//                                        uint32_t* resultTextBufferUsed)
HRESULT STDMETHODCALLTYPE ui_text_entry_result(void*, XAsyncBlock* async, uint32_t size, char* buffer, uint32_t* used) {
    if (used) *used = 0;
    if (!buffer) return E_POINTER;
    size_t written = 0;
    const HRESULT hr = XAsyncGetResult(async, &kTextEntryIdentity, size, buffer, &written);
    if (SUCCEEDED(hr) && used) *used = static_cast<uint32_t>(written);
    return hr;
}

// 30: XGameUiShowMultiplayerActivityGameInviteAsync(async, XUserHandle requestingUser) (IXGameUiImpl2)
HRESULT STDMETHODCALLTYPE ui_multiplayer_invite_async(void*, XAsyncBlock* async) {
    return begin_closed(async, kMultiplayerInvite);
}
// 40: XGameUiShowStateShareAsync(async, XUserHandle requestingUser, const char* linkToken) (IXGameUiImpl3)
HRESULT STDMETHODCALLTYPE ui_state_share_async(void*, XAsyncBlock* async) { return begin_closed(async, kStateShare); }

// ---- Slots without an offline answer ----

// Slots that need the network (player picker, web authentication): the *Async function fails before starting, so
// the XAsyncBlock is never used; its Result partners return the same error. The arguments don't matter, so the
// functions take none (x64: the caller owns the argument area).
template <int Slot> HRESULT STDMETHODCALLTYPE offline_failure(void*) {
    static std::atomic<bool> s_logged{false};
    if (!s_logged.exchange(true))
        std::fprintf(stderr, "[rt] xgameruntime: XGameUiImpl slot %d needs the network; not available offline\n", Slot);
    return kNetworkUnreachable;
}

// Slots with no public signature (the docs' "unknown"), the in-game text-entry handles and the custom-UI callbacks:
// E_NOTIMPL, so a caller fails instead of reading outputs nothing wrote. Every documented function here returns an
// HRESULT, so E_NOTIMPL reads as a failure.
template <int Slot> HRESULT STDMETHODCALLTYPE not_implemented(void*) {
    static std::atomic<bool> s_logged{false};
    if (!s_logged.exchange(true))
        std::fprintf(stderr, "[rt] xgameruntime: XGameUiImpl slot %d is not implemented offline; answering E_NOTIMPL\n", Slot);
    return E_NOTIMPL;
}

// ---- The object ----

constexpr GUID kUiClass = {0xdfcd4649, 0x4ff8, 0x4043, {0xba, 0x07, 0x35, 0xd6, 0x07, 0xdf, 0x98, 0xb0}};
// Unlike XUserImpl and XStoreImpl, the class id is not IXGameUiImpl's id. IXGameUiImpl2..4 extend IXGameUiImpl, so
// each answers with the same vtable.
constexpr GUID kUiInterfaces[] = {
    {0x6eeaa73e, 0x9669, 0x43ad, {0xa2, 0xc7, 0xd0, 0xda, 0x4e, 0x1f, 0x50, 0xa1}},   // IXGameUiImpl
    {0x36a03122, 0x9ea3, 0x4a3a, {0xa8, 0xa4, 0x89, 0x9c, 0xfd, 0x85, 0xd7, 0xdb}},   // IXGameUiImpl2
    {0xade7eba1, 0x2093, 0x42ce, {0xa5, 0x44, 0xa5, 0x23, 0xa6, 0x67, 0x90, 0xe0}},   // IXGameUiImpl3
    {0xeaf669df, 0x5542, 0x4590, {0x99, 0xa3, 0x8d, 0xc0, 0x61, 0xf8, 0x37, 0xcc}},   // IXGameUiImpl4
};
constexpr GUID kIUnknown = {0x00000000, 0x0000, 0x0000, {0xc0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x46}};

struct UiObject {
    const void* const* vtable;
};
UiObject* ui_object();

// The thunks pass the class id as the interface id too (docs/runtime_imports.md), so the class id is accepted as
// an interface id alongside IUnknown and IXGameUiImpl..4.
HRESULT query(REFIID iid, void** out) {
    if (!out) return E_POINTER;
    *out = nullptr;
    bool known = iid == kIUnknown || iid == kUiClass;
    for (const GUID& id : kUiInterfaces) known = known || iid == id;
    if (!known) return E_NOINTERFACE;
    *out = ui_object();
    return S_OK;
}

// A process-wide singleton, as the other runtime classes: AddRef and Release don't free.
HRESULT STDMETHODCALLTYPE object_query_interface(void*, REFIID iid, void** out) { return query(iid, out); }
ULONG STDMETHODCALLTYPE object_add_ref(void*) { return 2; }
ULONG STDMETHODCALLTYPE object_release(void*) { return 1; }

#define FN(f) reinterpret_cast<const void*>(&f)

const void* const kUiVtable[] = {
    FN(object_query_interface), FN(object_add_ref), FN(object_release),          // 0-2  IUnknown
    FN(ui_message_dialog_async),                                                // 3  XGameUiShowMessageDialogAsync
    FN(ui_message_dialog_result),                                               // 4  ...Result
    FN(ui_send_invite_async),                                                   // 5  XGameUiShowSendGameInviteAsync
    FN(closed_result<kSendGameInvite>),                                         // 6  ...Result
    FN(ui_profile_card_async),                                                  // 7  XGameUiShowPlayerProfileCardAsync
    FN(closed_result<kProfileCard>),                                            // 8  ...Result
    FN(ui_achievements_async),                                                  // 9  XGameUiShowAchievementsAsync
    FN(closed_result<kAchievements>),                                           // 10 ...Result
    FN(offline_failure<11>),                                                    // 11 XGameUiShowPlayerPickerAsync
    FN(offline_failure<12>),                                                    // 12 ...ResultCount
    FN(offline_failure<13>),                                                    // 13 ...Result
    FN(ui_error_dialog_async),                                                  // 14 XGameUiShowErrorDialogAsync
    FN(closed_result<kErrorDialog>),                                            // 15 ...Result
    FN(ui_set_notification_position),                                           // 16 XGameUiSetNotificationPositionHint
    FN(ui_text_entry_async),                                                    // 17 XGameUiShowTextEntryAsync
    FN(ui_text_entry_result_size),                                              // 18 ...ResultSize
    FN(ui_text_entry_result),                                                   // 19 ...Result
    FN(not_implemented<20>),                                                    // 20 (unknown in the docs)
    FN(not_implemented<21>),                                                    // 21 (unknown in the docs)
    FN(not_implemented<22>),                                                    // 22 (unknown in the docs)
    FN(not_implemented<23>),                                                    // 23 (unknown in the docs)
    FN(offline_failure<24>),                                                    // 24 XGameUiShowWebAuthenticationAsync
    FN(offline_failure<25>),                                                    // 25 ...ResultSize
    FN(offline_failure<26>),                                                    // 26 ...Result
    FN(offline_failure<27>),                                                    // 27 XGameUiShowWebAuthenticationWithOptionsAsync
    FN(not_implemented<28>),                                                    // 28 (unknown in the docs)
    FN(not_implemented<29>),                                                    // 29 (unknown in the docs)
    FN(ui_multiplayer_invite_async),                                            // 30 XGameUiShowMultiplayerActivityGameInviteAsync (2)
    FN(closed_result<kMultiplayerInvite>),                                      // 31 ...Result (2)
    FN(not_implemented<32>),                                                    // 32 (unknown in the docs) (2)
    FN(not_implemented<33>),                                                    // 33 (unknown in the docs) (2)
    FN(not_implemented<34>),                                                    // 34 XGameUiTextEntryOpen (2)
    FN(not_implemented<35>),                                                    // 35 XGameUiTextEntryClose (2)
    FN(not_implemented<36>),                                                    // 36 XGameUiTextEntryGetState (2)
    FN(not_implemented<37>),                                                    // 37 XGameUiTextEntryGetExtents (2)
    FN(not_implemented<38>),                                                    // 38 XGameUiTextEntryUpdatePositionHint (2)
    FN(not_implemented<39>),                                                    // 39 XGameUiTextEntryUpdateVisibility (2)
    FN(ui_state_share_async),                                                   // 40 XGameUiShowStateShareAsync (3)
    FN(closed_result<kStateShare>),                                             // 41 ...Result (3)
    FN(not_implemented<42>),                                                    // 42 XGameUiSetUiCallbacks (4)
    FN(not_implemented<43>),                                                    // 43 XGameUiSetMessageDialogUiResponse (4)
    FN(not_implemented<44>),                                                    // 44 XGameUiSetPlayerPickerUiResponse (4)
    FN(not_implemented<45>),                                                    // 45 XGameUiSetTextEntryUiResponse (4)
    FN(not_implemented<46>),                                                    // 46 XGameUiSetPlayerProfileCardUiResponse (4)
    FN(not_implemented<47>),                                                    // 47 XGameUiSetSendGameInviteUiResponse (4)
    FN(not_implemented<48>),                                                    // 48 XGameUiSetAchievementsUiResponse (4)
    FN(not_implemented<49>),                                                    // 49 XGameUiSetMultiplayerActivityGameInviteUiResponse (4)
    FN(not_implemented<50>),                                                    // 50 XGameUiSetErrorDialogUiResponse (4)
};
static_assert(std::size(kUiVtable) == 51, "IUnknown + IXGameUiImpl's 27 methods + 10 + 2 + 9");

#undef FN

UiObject* ui_object() {
    static UiObject o{kUiVtable};
    return &o;
}

}  // namespace

namespace rt::shims {

// Called by xgameruntime.cpp's QueryApiImpl for every class: true when the class is XGameUiImpl, with *hr set.
bool xgameruntime_ui_query(const GUID& class_id, REFIID interface_id, void** out, HRESULT* hr) {
    if (class_id != kUiClass) return false;
    *hr = query(interface_id, out);
    return true;
}

}  // namespace rt::shims
