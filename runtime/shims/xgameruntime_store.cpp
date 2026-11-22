// xgameruntime: XStoreImpl, the class xgameruntime.lib's XStore* functions reach through QueryApiImpl, offline.
//
// Each XStore* thunk asks QueryApiImpl for XStoreImpl and calls one slot with the function's own arguments after
// `this` (docs/runtime_imports.md, "Class layouts, measured"). The class and interface ids and the slot order are
// xodus-gaming/xgameruntime-docs' "Layout" lists (GPL-3.0; facts only, nothing copied). Signatures, enum values and
// error codes follow Microsoft's public GDK reference (learn.microsoft.com/gaming/gdk, the XStore pages). That
// reference names but doesn't give STORE_SKU_ID_SIZE and TRIAL_UNIQUE_ID_MAX_SIZE; their values (18 and 64) are from
// minecraft-linux/xgameruntime-rs (MIT, src/xstore.rs at c523673a9e), whose XStoreGameLicense has the same field
// order as the reference. No GDK header or library code is used.
//
// Offline, the store answers what a console with the game installed and no network would answer:
//   - the game has a full license: active, not a trial, not a disc license;
//   - the installed package's content can be licensed (XStoreAcquireLicenseForPackageAsync, ...ForDurablesAsync)
//     and the license stays valid, so a registered license-lost callback is never called;
//   - there are no add-on licenses, no product listings and no package updates (each query completes with an empty
//     result);
//   - everything that needs the store service or a store UI fails synchronously before starting, so no XAsyncBlock
//     is left waiting: purchases, consumables, user collection and purchase ids, license tokens, rate-and-review,
//     redeem, gifting, product pages, downloads. XStoreCanAcquireLicenseFor*Async fails the same way: its result
//     struct's sku field is sized by SKU_ID_SIZE, which no public source gives, so nothing writes into it.
//
// ASSUMPTION: the console executable is not available, so which of these slots it calls is unknown until its first
// boot. The docs' unknown slots and the undocumented IXStoreImpl2 one log once when called, so that boot lists them;
// the unknown ones answer E_NOTIMPL.
#include <windows.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <iterator>
#include <mutex>
#include <set>
#include <unordered_map>

#include <XAsync.h>
#include <XAsyncProvider.h>
#include <XTaskQueue.h>

namespace {

// ---- Public GDK types (XStore.h, as the GDK reference documents them) ----

constexpr size_t kStoreSkuIdSize = 18;        // STORE_SKU_ID_SIZE (see the header comment)
constexpr size_t kTrialUniqueIdMaxSize = 64;  // TRIAL_UNIQUE_ID_MAX_SIZE

struct XStoreGameLicense {
    char skuStoreId[kStoreSkuIdSize];
    bool isActive;
    bool isTrialOwnedByThisUser;
    bool isDiscLicense;
    bool isTrial;
    uint32_t trialTimeRemainingInSeconds;
    char trialUniqueId[kTrialUniqueIdMaxSize];
    time_t expirationDate;
};
static_assert(offsetof(XStoreGameLicense, isActive) == 18 && offsetof(XStoreGameLicense, trialTimeRemainingInSeconds) == 24 &&
                  offsetof(XStoreGameLicense, trialUniqueId) == 28 && offsetof(XStoreGameLicense, expirationDate) == 96 &&
                  sizeof(XStoreGameLicense) == 104,
              "XStoreGameLicense, x64 natural alignment");

using XStoreGameLicenseChangedCallback = void(CALLBACK*)(void* context);
using XStorePackageLicenseLostCallback = void(CALLBACK*)(void* context);
using XStoreProductQueryCallback = bool(CALLBACK*)(const void* product, void* context);

// Documented on the XStoreAcquireLicenseForPackageAsync page: "Package for this product isn't installed".
constexpr HRESULT E_GAMEPACKAGE_NO_PACKAGE_IDENTIFIER = static_cast<HRESULT>(0x89245208);
// The store service isn't reachable: what a console without a network answers for a service call.
constexpr HRESULT kStoreUnreachable = HRESULT_FROM_WIN32(ERROR_NETWORK_UNREACHABLE);

// ---- Handles ----

// Contexts, licenses and product queries are each their own allocation, closed once by their Close function. A
// handle is valid while it is in the live map under its kind: it's looked up, never read, so a closed, foreign or
// wrong-kind pointer is refused without touching its memory, and a double close does nothing. A freed block's address
// may come back from the heap for a newer handle; a stale handle then names that one, as a stale pointer would.
enum class Kind : uint32_t { Context, License, ProductQuery };
struct StoreHandle {
    uint32_t unused;
};
std::mutex g_handles_mutex;
std::unordered_map<const void*, Kind> g_handles;

StoreHandle* new_handle(Kind kind) {
    auto* h = new StoreHandle{};
    std::lock_guard<std::mutex> lock(g_handles_mutex);
    g_handles.emplace(h, kind);
    return h;
}
bool valid(const void* h, Kind kind) {
    if (!h) return false;
    std::lock_guard<std::mutex> lock(g_handles_mutex);
    auto it = g_handles.find(h);
    return it != g_handles.end() && it->second == kind;
}
void close_handle(void* h, Kind kind) {
    {
        std::lock_guard<std::mutex> lock(g_handles_mutex);
        auto it = h ? g_handles.find(h) : g_handles.end();
        if (it == g_handles.end() || it->second != kind) return;   // closed already, not ours, or another kind
        g_handles.erase(it);
    }
    delete static_cast<StoreHandle*>(h);
}

// ---- XAsync providers ----
//
// An operation's identity is the address of its Async function's tag, so a Result call with another operation's
// XAsyncBlock fails as XAsyncGetResult documents. A handle result is created when the title collects it, so a result
// never collected leaks nothing. An empty result completes with no payload; xtaskqueue then lets XAsyncGetResult be
// called any number of times, which the ResultCount/Result pairs rely on.

enum class Payload { None, GameLicense, License, ProductQuery };
struct Operation {
    const char* name;
    Payload payload;
};

size_t payload_size(Payload p) {
    switch (p) {
    case Payload::GameLicense: return sizeof(XStoreGameLicense);
    case Payload::License:
    case Payload::ProductQuery: return sizeof(void*);
    default: return 0;
    }
}

HRESULT CALLBACK store_provider(XAsyncOp op, const XAsyncProviderData* data) {
    const auto* operation = static_cast<const Operation*>(data->context);
    switch (op) {
    case XAsyncOp::Begin:
        return XAsyncSchedule(data->async, 0);
    case XAsyncOp::DoWork:
        XAsyncComplete(data->async, S_OK, payload_size(operation->payload));
        return S_OK;
    case XAsyncOp::GetResult:
        if (!data->buffer || data->bufferSize < payload_size(operation->payload)) return E_INVALIDARG;
        switch (operation->payload) {
        case Payload::GameLicense: {
            // A full license for the installed game. The SKU and trial ids are empty (offline there is no catalog to
            // name them) and expirationDate is 0, as for a license that doesn't expire.
            auto* license = static_cast<XStoreGameLicense*>(data->buffer);
            std::memset(license, 0, sizeof(*license));
            license->isActive = true;
            return S_OK;
        }
        case Payload::License:
            *static_cast<void**>(data->buffer) = new_handle(Kind::License);
            return S_OK;
        case Payload::ProductQuery:
            *static_cast<void**>(data->buffer) = new_handle(Kind::ProductQuery);
            return S_OK;
        default:
            return S_OK;
        }
    default:
        return S_OK;
    }
}

HRESULT begin(XAsyncBlock* async, const Operation& operation) {
    if (!async) return E_POINTER;
    return XAsyncBegin(async, const_cast<Operation*>(&operation), &operation, operation.name, store_provider);
}

template <class T> HRESULT handle_result(XAsyncBlock* async, const Operation& operation, T* out) {
    if (!out) return E_POINTER;
    *out = nullptr;
    return XAsyncGetResult(async, &operation, sizeof(*out), out, nullptr);
}

// An empty list's count: 0 once the operation succeeded, its failure (or E_PENDING) otherwise.
HRESULT empty_count(XAsyncBlock* async, const Operation& operation, uint32_t* count) {
    if (!count) return E_POINTER;
    *count = 0;
    return XAsyncGetResult(async, &operation, 0, nullptr, nullptr);
}
HRESULT empty_list(XAsyncBlock* async, const Operation& operation, uint32_t count) {
    const HRESULT hr = XAsyncGetResult(async, &operation, 0, nullptr, nullptr);
    if (FAILED(hr)) return hr;
    return count == 0 ? S_OK : E_INVALIDARG;   // more entries asked for than the count gave
}

const Operation kAssociatedProducts{"XStoreQueryAssociatedProductsAsync", Payload::ProductQuery};
const Operation kProducts{"XStoreQueryProductsAsync", Payload::ProductQuery};
const Operation kEntitledProducts{"XStoreQueryEntitledProductsAsync", Payload::ProductQuery};
const Operation kProductForCurrentGame{"XStoreQueryProductForCurrentGameAsync", Payload::ProductQuery};
const Operation kProductForPackage{"XStoreQueryProductForPackageAsync", Payload::ProductQuery};
const Operation kAssociatedProductsForStoreId{"XStoreQueryAssociatedProductsForStoreIdAsync", Payload::ProductQuery};
const Operation kLicenseForPackage{"XStoreAcquireLicenseForPackageAsync", Payload::License};
const Operation kLicenseForDurables{"XStoreAcquireLicenseForDurablesAsync", Payload::License};
const Operation kGameLicense{"XStoreQueryGameLicenseAsync", Payload::GameLicense};
const Operation kAddOnLicenses{"XStoreQueryAddOnLicensesAsync", Payload::None};
const Operation kGameAndDlcUpdates{"XStoreQueryGameAndDlcPackageUpdatesAsync", Payload::None};
const Operation kPackageUpdates{"XStoreQueryPackageUpdatesAsync", Payload::None};

// ---- IXStoreImpl (slots 3-69) ----

// 3: HRESULT XStoreCreateContext(const XUserHandle user, XStoreContextHandle* storeContextHandle). The user is not
// checked: every XUserHandle names the one offline user (xgameruntime_user.cpp), and PC titles pass nullptr.
HRESULT STDMETHODCALLTYPE store_create_context(void*, void*, void** context) {
    if (!context) return E_POINTER;
    *context = new_handle(Kind::Context);
    return S_OK;
}

// 4: void XStoreCloseContextHandle(XStoreContextHandle storeContextHandle)
void STDMETHODCALLTYPE store_close_context(void*, void* context) { close_handle(context, Kind::Context); }

// 5: XStoreQueryAssociatedProductsAsync(ctx, XStoreProductKind productKinds, uint32_t maxItemsToRetrievePerPage, async)
HRESULT STDMETHODCALLTYPE store_associated_async(void*, void* ctx, uint32_t, uint32_t, XAsyncBlock* async) {
    return valid(ctx, Kind::Context) ? begin(async, kAssociatedProducts) : E_INVALIDARG;
}
// 6, 8, 10, 12, 14, 78: XStore*Result(XAsyncBlock* async, XStoreProductQueryHandle* productQueryHandle)
template <const Operation& Op> HRESULT STDMETHODCALLTYPE store_query_result(void*, XAsyncBlock* async, void** query) {
    return handle_result(async, Op, query);
}
// 7: XStoreQueryProductsAsync(ctx, productKinds, const char** storeIds, size_t storeIdsCount,
//                             const char** actionFilters, size_t actionFiltersCount, async)
HRESULT STDMETHODCALLTYPE store_products_async(void*, void* ctx, uint32_t, const char**, size_t, const char**, size_t,
                                               XAsyncBlock* async) {
    return valid(ctx, Kind::Context) ? begin(async, kProducts) : E_INVALIDARG;
}
// 9: XStoreQueryEntitledProductsAsync(ctx, productKinds, uint32_t maxItemsToRetrievePerPage, async)
HRESULT STDMETHODCALLTYPE store_entitled_async(void*, void* ctx, uint32_t, uint32_t, XAsyncBlock* async) {
    return valid(ctx, Kind::Context) ? begin(async, kEntitledProducts) : E_INVALIDARG;
}
// 11: XStoreQueryProductForCurrentGameAsync(ctx, async)
HRESULT STDMETHODCALLTYPE store_current_game_async(void*, void* ctx, XAsyncBlock* async) {
    return valid(ctx, Kind::Context) ? begin(async, kProductForCurrentGame) : E_INVALIDARG;
}
// 13: XStoreQueryProductForPackageAsync(ctx, productKinds, const char* packageIdentifier, async)
HRESULT STDMETHODCALLTYPE store_for_package_async(void*, void* ctx, uint32_t, const char* package, XAsyncBlock* async) {
    if (!valid(ctx, Kind::Context) || !package) return E_INVALIDARG;
    return begin(async, kProductForPackage);
}

// 15: HRESULT XStoreEnumerateProductsQuery(const XStoreProductQueryHandle, void* context, XStoreProductQueryCallback*)
// Offline queries list no products, so the callback is never called.
HRESULT STDMETHODCALLTYPE store_enumerate(void*, void* query, void*, XStoreProductQueryCallback callback) {
    if (!callback) return E_POINTER;
    return valid(query, Kind::ProductQuery) ? S_OK : E_INVALIDARG;
}
// 16: bool XStoreProductsQueryHasMorePages(const XStoreProductQueryHandle)
bool STDMETHODCALLTYPE store_has_more_pages(void*, void*) { return false; }
// 19: void XStoreCloseProductsQueryHandle(XStoreProductQueryHandle)
void STDMETHODCALLTYPE store_close_query(void*, void* query) { close_handle(query, Kind::ProductQuery); }

// 20: XStoreAcquireLicenseForPackageAsync(ctx, const char* packageIdentifier, async)
HRESULT STDMETHODCALLTYPE store_license_for_package_async(void*, void* ctx, const char* package, XAsyncBlock* async) {
    if (!valid(ctx, Kind::Context) || !package) return E_INVALIDARG;
    return begin(async, kLicenseForPackage);
}
// 21, 72: XStoreAcquireLicenseFor*Result(XAsyncBlock* async, XStoreLicenseHandle* storeLicenseHandle)
template <const Operation& Op> HRESULT STDMETHODCALLTYPE store_license_result(void*, XAsyncBlock* async, void** license) {
    return handle_result(async, Op, license);
}
// 22: bool XStoreIsLicenseValid(const XStoreLicenseHandle)
bool STDMETHODCALLTYPE store_is_license_valid(void*, void* license) { return valid(license, Kind::License); }
// 23: void XStoreCloseLicenseHandle(XStoreLicenseHandle)
void STDMETHODCALLTYPE store_close_license(void*, void* license) { close_handle(license, Kind::License); }

// 28: XStoreQueryGameLicenseAsync(ctx, async); 29: XStoreQueryGameLicenseResult(async, XStoreGameLicense* license)
HRESULT STDMETHODCALLTYPE store_game_license_async(void*, void* ctx, XAsyncBlock* async) {
    return valid(ctx, Kind::Context) ? begin(async, kGameLicense) : E_INVALIDARG;
}
HRESULT STDMETHODCALLTYPE store_game_license_result(void*, XAsyncBlock* async, XStoreGameLicense* license) {
    if (!license) return E_POINTER;
    return XAsyncGetResult(async, &kGameLicense, sizeof(*license), license, nullptr);
}

// 30: XStoreQueryAddOnLicensesAsync(ctx, async); 31: ...ResultCount(async, uint32_t* count);
// 32: ...Result(async, uint32_t count, XStoreAddonLicense* addOnLicenses). No add-ons offline.
// 55-57: XStoreQueryGameAndDlcPackageUpdates*, the same shape. No updates offline.
template <const Operation& Op> HRESULT STDMETHODCALLTYPE store_empty_list_async(void*, void* ctx, XAsyncBlock* async) {
    return valid(ctx, Kind::Context) ? begin(async, Op) : E_INVALIDARG;
}
template <const Operation& Op> HRESULT STDMETHODCALLTYPE store_empty_count(void*, XAsyncBlock* async, uint32_t* count) {
    return empty_count(async, Op, count);
}
template <const Operation& Op> HRESULT STDMETHODCALLTYPE store_empty_list(void*, XAsyncBlock* async, uint32_t count, void*) {
    return empty_list(async, Op, count);
}

// 65: HRESULT XStoreQueryPackageIdentifier(const char* storeId, size_t size, char* packageIdentifier). Offline there
// is no catalog mapping store ids to installed packages.
HRESULT STDMETHODCALLTYPE store_query_package_identifier(void*, const char* store_id, size_t size, char* out) {
    if (!store_id || !out || size == 0) return E_INVALIDARG;
    out[0] = 0;
    return E_GAMEPACKAGE_NO_PACKAGE_IDENTIFIER;
}

// 66: XStoreRegisterGameLicenseChanged(ctx, XTaskQueueHandle, void* context, callback*, XTaskQueueRegistrationToken*)
// 67: bool XStoreUnregisterGameLicenseChanged(ctx, XTaskQueueRegistrationToken token, bool wait)
// 68: XStoreRegisterPackageLicenseLost(license, XTaskQueueHandle, void* context, callback*, XTaskQueueRegistrationToken*)
// 69: bool XStoreUnregisterPackageLicenseLost(license, XTaskQueueRegistrationToken token, bool wait)
// Licenses never change offline, so the callbacks are kept only as tokens. Unregistering is never pending: true.
std::mutex g_tokens_mutex;
std::set<uint64_t> g_tokens;
uint64_t g_next_token = 1;

HRESULT register_token(const void* owner, Kind kind, const void* callback, XTaskQueueRegistrationToken* token) {
    if (!callback || !token) return E_POINTER;
    if (!valid(owner, kind)) return E_INVALIDARG;
    std::lock_guard<std::mutex> lock(g_tokens_mutex);
    token->token = g_next_token++;
    g_tokens.insert(token->token);
    return S_OK;
}
bool unregister_token(XTaskQueueRegistrationToken token) {
    std::lock_guard<std::mutex> lock(g_tokens_mutex);
    g_tokens.erase(token.token);
    return true;
}

HRESULT STDMETHODCALLTYPE store_register_license_changed(void*, void* ctx, XTaskQueueHandle, void*,
                                                         XStoreGameLicenseChangedCallback callback,
                                                         XTaskQueueRegistrationToken* token) {
    return register_token(ctx, Kind::Context, reinterpret_cast<const void*>(callback), token);
}
bool STDMETHODCALLTYPE store_unregister_license_changed(void*, void*, XTaskQueueRegistrationToken token, bool) {
    return unregister_token(token);
}
HRESULT STDMETHODCALLTYPE store_register_license_lost(void*, void* license, XTaskQueueHandle, void*,
                                                      XStorePackageLicenseLostCallback callback,
                                                      XTaskQueueRegistrationToken* token) {
    return register_token(license, Kind::License, reinterpret_cast<const void*>(callback), token);
}
bool STDMETHODCALLTYPE store_unregister_license_lost(void*, void*, XTaskQueueRegistrationToken token, bool) {
    return unregister_token(token);
}

// ---- IXStoreImpl3 (71-72) ----

// 71: XStoreAcquireLicenseForDurablesAsync(ctx, const char* storeId, async)
HRESULT STDMETHODCALLTYPE store_license_for_durables_async(void*, void* ctx, const char* store_id, XAsyncBlock* async) {
    if (!valid(ctx, Kind::Context) || !store_id) return E_INVALIDARG;
    return begin(async, kLicenseForDurables);
}

// ---- IXStoreImpl5 (77-81) ----

// 77: XStoreQueryAssociatedProductsForStoreIdAsync(ctx, const char* storeId, productKinds, uint32_t maxItems, async)
HRESULT STDMETHODCALLTYPE store_associated_for_store_id_async(void*, void* ctx, const char* store_id, uint32_t, uint32_t,
                                                              XAsyncBlock* async) {
    if (!valid(ctx, Kind::Context) || !store_id) return E_INVALIDARG;
    return begin(async, kAssociatedProductsForStoreId);
}
// 79: XStoreQueryPackageUpdatesAsync(ctx, const char** packageIdentifiers, size_t packageIdentifiersCount, async)
HRESULT STDMETHODCALLTYPE store_package_updates_async(void*, void* ctx, const char**, size_t, XAsyncBlock* async) {
    return valid(ctx, Kind::Context) ? begin(async, kPackageUpdates) : E_INVALIDARG;
}

// ---- Slots without an offline answer ----

// A slot with no public signature, logged once. 46-48 (the docs' "unknown") answer E_NOTIMPL, so a caller fails
// instead of reading outputs nothing wrote. 70, IXStoreImpl2's undocumented XStoreIsAvailabilityPurchasable, is by
// its name read as a bool, where E_NOTIMPL's low byte (1) would mean true, so it answers 0 (false: nothing can be
// purchased offline).
template <int Slot> HRESULT STDMETHODCALLTYPE unknown_slot(void*) {
    static std::atomic<bool> s_logged{false};
    if (!s_logged.exchange(true))
        std::fprintf(stderr, "[rt] xgameruntime: XStoreImpl slot %d has no public signature; answering E_NOTIMPL\n", Slot);
    return E_NOTIMPL;
}
template <int Slot> uint64_t STDMETHODCALLTYPE private_slot(void*) {
    static std::atomic<bool> s_logged{false};
    if (!s_logged.exchange(true))
        std::fprintf(stderr, "[rt] xgameruntime: XStoreImpl private slot %d called; answering 0\n", Slot);
    return 0;
}

// An operation that needs the store service or a store UI. The *Async function fails before starting, so the
// XAsyncBlock is never used and no callback runs; its Result/ResultSize/ResultCount partners return the same error.
// Their arguments don't matter, so the functions take none (x64: the caller owns the argument area).
template <int Slot> HRESULT STDMETHODCALLTYPE offline_failure(void*) {
    static std::atomic<bool> s_logged{false};
    if (!s_logged.exchange(true))
        std::fprintf(stderr, "[rt] xgameruntime: XStoreImpl slot %d needs the store service; not available offline\n", Slot);
    return kStoreUnreachable;
}

// ---- The object ----

constexpr GUID kStoreClass = {0x0dd112ac, 0x7c24, 0x448c, {0xb9, 0x2b, 0x39, 0x60, 0xfb, 0x5b, 0xd3, 0x0c}};
// IXStoreImpl's id is the class id; IXStoreImpl2..6 extend it, so each answers with the same vtable.
constexpr GUID kStoreInterfaces[] = {
    kStoreClass,
    {0x60b09f4e, 0x1b85, 0x45b1, {0x82, 0x6c, 0x16, 0x91, 0x18, 0xe2, 0x30, 0xe1}},   // IXStoreImpl2
    {0x2d42fea5, 0xe71d, 0x4b76, {0x97, 0xcd, 0xc5, 0x0a, 0xfb, 0xb3, 0xae, 0x5d}},   // IXStoreImpl3
    {0xde3dbdd4, 0x0b37, 0x4bdb, {0xa1, 0x0e, 0xac, 0xf3, 0xa3, 0x54, 0xd0, 0x6a}},   // IXStoreImpl4
    {0x5c48dedf, 0x0b67, 0x4492, {0xa4, 0xb5, 0x68, 0x29, 0xb8, 0xe7, 0x96, 0xe1}},   // IXStoreImpl5
    {0xb09d803c, 0x2414, 0x4a05, {0x82, 0xc6, 0x66, 0xdf, 0xdc, 0x9e, 0x9a, 0x44}},   // IXStoreImpl6
};
constexpr GUID kIUnknown = {0x00000000, 0x0000, 0x0000, {0xc0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x46}};

struct StoreObject {
    const void* const* vtable;
};
StoreObject* store_object();

HRESULT query(REFIID iid, void** out) {
    if (!out) return E_POINTER;
    *out = nullptr;
    bool known = iid == kIUnknown;
    for (const GUID& id : kStoreInterfaces) known = known || iid == id;
    if (!known) return E_NOINTERFACE;
    *out = store_object();
    return S_OK;
}

// A process-wide singleton, as the other runtime classes: AddRef and Release don't free.
HRESULT STDMETHODCALLTYPE object_query_interface(void*, REFIID iid, void** out) { return query(iid, out); }
ULONG STDMETHODCALLTYPE object_add_ref(void*) { return 2; }
ULONG STDMETHODCALLTYPE object_release(void*) { return 1; }

#define FN(f) reinterpret_cast<const void*>(&f)

const void* const kStoreVtable[] = {
    FN(object_query_interface), FN(object_add_ref), FN(object_release),          // 0-2  IUnknown
    FN(store_create_context),                                                   // 3  XStoreCreateContext
    FN(store_close_context),                                                    // 4  XStoreCloseContextHandle
    FN(store_associated_async),                                                 // 5  XStoreQueryAssociatedProductsAsync
    FN(store_query_result<kAssociatedProducts>),                                // 6  ...Result
    FN(store_products_async),                                                   // 7  XStoreQueryProductsAsync
    FN(store_query_result<kProducts>),                                          // 8  ...Result
    FN(store_entitled_async),                                                   // 9  XStoreQueryEntitledProductsAsync
    FN(store_query_result<kEntitledProducts>),                                  // 10 ...Result
    FN(store_current_game_async),                                               // 11 XStoreQueryProductForCurrentGameAsync
    FN(store_query_result<kProductForCurrentGame>),                             // 12 ...Result
    FN(store_for_package_async),                                                // 13 XStoreQueryProductForPackageAsync
    FN(store_query_result<kProductForPackage>),                                 // 14 ...Result
    FN(store_enumerate),                                                        // 15 XStoreEnumerateProductsQuery
    FN(store_has_more_pages),                                                   // 16 XStoreProductsQueryHasMorePages
    FN(offline_failure<17>),                                                    // 17 XStoreProductsQueryNextPageAsync
    FN(offline_failure<18>),                                                    // 18 ...NextPageResult
    FN(store_close_query),                                                      // 19 XStoreCloseProductsQueryHandle
    FN(store_license_for_package_async),                                        // 20 XStoreAcquireLicenseForPackageAsync
    FN(store_license_result<kLicenseForPackage>),                               // 21 ...Result
    FN(store_is_license_valid),                                                 // 22 XStoreIsLicenseValid
    FN(store_close_license),                                                    // 23 XStoreCloseLicenseHandle
    FN(offline_failure<24>),                                                    // 24 XStoreCanAcquireLicenseForStoreIdAsync
    FN(offline_failure<25>),                                                    // 25 ...Result
    FN(offline_failure<26>),                                                    // 26 XStoreCanAcquireLicenseForPackageAsync
    FN(offline_failure<27>),                                                    // 27 ...Result
    FN(store_game_license_async),                                               // 28 XStoreQueryGameLicenseAsync
    FN(store_game_license_result),                                              // 29 XStoreQueryGameLicenseResult
    FN(store_empty_list_async<kAddOnLicenses>),                                 // 30 XStoreQueryAddOnLicensesAsync
    FN(store_empty_count<kAddOnLicenses>),                                      // 31 ...ResultCount
    FN(store_empty_list<kAddOnLicenses>),                                       // 32 ...Result
    FN(offline_failure<33>),                                                    // 33 XStoreQueryConsumableBalanceRemainingAsync
    FN(offline_failure<34>),                                                    // 34 ...Result
    FN(offline_failure<35>),                                                    // 35 XStoreReportConsumableFulfillmentAsync
    FN(offline_failure<36>),                                                    // 36 ...Result
    FN(offline_failure<37>),                                                    // 37 XStoreGetUserCollectionsIdAsync
    FN(offline_failure<38>),                                                    // 38 ...ResultSize
    FN(offline_failure<39>),                                                    // 39 ...Result
    FN(offline_failure<40>),                                                    // 40 XStoreGetUserPurchaseIdAsync
    FN(offline_failure<41>),                                                    // 41 ...ResultSize
    FN(offline_failure<42>),                                                    // 42 ...Result
    FN(offline_failure<43>),                                                    // 43 XStoreQueryLicenseTokenAsync
    FN(offline_failure<44>),                                                    // 44 ...ResultSize
    FN(offline_failure<45>),                                                    // 45 ...Result
    FN(unknown_slot<46>),                                                       // 46 (unknown in the docs)
    FN(unknown_slot<47>),                                                       // 47 (unknown in the docs)
    FN(unknown_slot<48>),                                                       // 48 (unknown in the docs)
    FN(offline_failure<49>),                                                    // 49 XStoreShowPurchaseUIAsync
    FN(offline_failure<50>),                                                    // 50 ...Result
    FN(offline_failure<51>),                                                    // 51 XStoreShowRateAndReviewUIAsync
    FN(offline_failure<52>),                                                    // 52 ...Result
    FN(offline_failure<53>),                                                    // 53 XStoreShowRedeemTokenUIAsync
    FN(offline_failure<54>),                                                    // 54 ...Result
    FN(store_empty_list_async<kGameAndDlcUpdates>),                             // 55 XStoreQueryGameAndDlcPackageUpdatesAsync
    FN(store_empty_count<kGameAndDlcUpdates>),                                  // 56 ...ResultCount
    FN(store_empty_list<kGameAndDlcUpdates>),                                   // 57 ...Result
    FN(offline_failure<58>),                                                    // 58 XStoreDownloadPackageUpdatesAsync
    FN(offline_failure<59>),                                                    // 59 ...Result
    FN(offline_failure<60>),                                                    // 60 XStoreDownloadAndInstallPackageUpdatesAsync
    FN(offline_failure<61>),                                                    // 61 ...Result
    FN(offline_failure<62>),                                                    // 62 XStoreDownloadAndInstallPackagesAsync
    FN(offline_failure<63>),                                                    // 63 ...ResultCount
    FN(offline_failure<64>),                                                    // 64 ...Result
    FN(store_query_package_identifier),                                         // 65 XStoreQueryPackageIdentifier
    FN(store_register_license_changed),                                         // 66 XStoreRegisterGameLicenseChanged
    FN(store_unregister_license_changed),                                       // 67 XStoreUnregisterGameLicenseChanged
    FN(store_register_license_lost),                                            // 68 XStoreRegisterPackageLicenseLost
    FN(store_unregister_license_lost),                                          // 69 XStoreUnregisterPackageLicenseLost
    FN(private_slot<70>),                                                       // 70 XStoreIsAvailabilityPurchasable (2)
    FN(store_license_for_durables_async),                                       // 71 XStoreAcquireLicenseForDurablesAsync (3)
    FN(store_license_result<kLicenseForDurables>),                              // 72 ...Result (3)
    FN(offline_failure<73>),                                                    // 73 XStoreShowAssociatedProductsUIAsync (4)
    FN(offline_failure<74>),                                                    // 74 ...Result (4)
    FN(offline_failure<75>),                                                    // 75 XStoreShowProductPageUIAsync (4)
    FN(offline_failure<76>),                                                    // 76 ...Result (4)
    FN(store_associated_for_store_id_async),                                    // 77 XStoreQueryAssociatedProductsForStoreIdAsync (5)
    FN(store_query_result<kAssociatedProductsForStoreId>),                      // 78 ...Result (5)
    FN(store_package_updates_async),                                            // 79 XStoreQueryPackageUpdatesAsync (5)
    FN(store_empty_count<kPackageUpdates>),                                     // 80 ...ResultCount (5)
    FN(store_empty_list<kPackageUpdates>),                                      // 81 ...Result (5)
    FN(offline_failure<82>),                                                    // 82 XStoreShowGiftingUIAsync (6)
    FN(offline_failure<83>),                                                    // 83 ...Result (6)
};
static_assert(std::size(kStoreVtable) == 84, "IUnknown + IXStoreImpl's 67 methods + 1 + 2 + 4 + 5 + 2");

#undef FN

StoreObject* store_object() {
    static StoreObject o{kStoreVtable};
    return &o;
}

}  // namespace

namespace rt::shims {

// Called by xgameruntime.cpp's QueryApiImpl for every class: true when the class is XStoreImpl, with *hr set.
bool xgameruntime_store_query(const GUID& class_id, REFIID interface_id, void** out, HRESULT* hr) {
    if (class_id != kStoreClass) return false;
    *hr = query(interface_id, out);
    return true;
}

// For tests: the store context, license and product-query handles a title holds open.
long xgameruntime_store_open_handles() {
    std::lock_guard<std::mutex> lock(g_handles_mutex);
    return static_cast<long>(g_handles.size());
}

}  // namespace rt::shims
