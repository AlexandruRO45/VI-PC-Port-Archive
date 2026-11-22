// XStoreImpl through QueryApiImpl, called the way xgameruntime.lib's XStore* thunks call it: QueryApiImpl with the
// class id (or a later interface id), one vtable slot with `this` first. Slots follow xgameruntime-docs' layout;
// arguments and expected values follow the public GDK reference. No game data; nothing online.
#include <windows.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <XAsync.h>
#include <XAsyncProvider.h>
#include <XTaskQueue.h>

#include "shims.h"

namespace rt::shims {
long xgameruntime_store_open_handles();
}

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

constexpr HRESULT kStoreUnreachable = HRESULT_FROM_WIN32(ERROR_NETWORK_UNREACHABLE);
constexpr HRESULT E_GAMEPACKAGE_NO_PACKAGE_IDENTIFIER = static_cast<HRESULT>(0x89245208);

}  // namespace

int main() {
    std::printf("xgameruntime_store_test: XStoreImpl through QueryApiImpl\n");
    auto query = reinterpret_cast<Query>(rt::shims::find("xgameruntime.dll", "QueryApiImpl"));
    check(query != nullptr, "QueryApiImpl is registered");
    if (!query) return 1;

    // ---- the object and its interfaces ----
    const GUID cls = parse_guid("0dd112ac-7c24-448c-b92b-3960fb5bd30c");
    void* store = nullptr;
    check(query(&cls, cls, &store) == S_OK && store, "QueryApiImpl hands out XStoreImpl for its class id");
    if (!store) return 1;
    const char* const later[] = {"60b09f4e-1b85-45b1-826c-169118e230e1", "2d42fea5-e71d-4b76-97cd-c50afbb3ae5d",
                                 "de3dbdd4-0b37-4bdb-a10e-acf3a354d06a", "5c48dedf-0b67-4492-a4b5-6829b8e796e1",
                                 "b09d803c-2414-4a05-82c6-66dfdc9e9a44"};
    bool all = true;
    for (const char* id : later) {
        void* face = nullptr;
        all = all && query(&cls, parse_guid(id), &face) == S_OK && face == store;
    }
    check(all, "IXStoreImpl2..IXStoreImpl6 (ids as the docs print them) answer with the same object");
    void* none = reinterpret_cast<void*>(1);
    const GUID user_iid = parse_guid("01acd177-91f9-4763-a38e-ccbb55ce32e0");
    check(query(&cls, user_iid, &none) == E_NOINTERFACE && none == nullptr,
          "an interface the class doesn't have is E_NOINTERFACE with a cleared out pointer");

    // ---- XStoreCreateContext (3) ----
    using Create = HRESULT(STDMETHODCALLTYPE*)(void*, void*, void**);
    using CloseH = void(STDMETHODCALLTYPE*)(void*, void*);
    void* ctx = nullptr;
    check(at<Create>(store, 3)(store, nullptr, &ctx) == S_OK && ctx, "XStoreCreateContext hands out a context");

    // ---- XStoreQueryGameLicenseAsync / Result (28, 29) ----
    using CtxAsync = HRESULT(STDMETHODCALLTYPE*)(void*, void*, XAsyncBlock*);
    using BufResult = HRESULT(STDMETHODCALLTYPE*)(void*, XAsyncBlock*, void*);
    XAsyncBlock lic{};
    unsigned char license[104];
    std::memset(license, 0xcc, sizeof(license));
    check(at<CtxAsync>(store, 28)(store, ctx, &lic) == S_OK && finished(&lic) &&
              at<BufResult>(store, 29)(store, &lic, license) == S_OK,
          "XStoreQueryGameLicenseAsync completes and XStoreQueryGameLicenseResult fills the license");
    int64_t expiration = -1;
    uint32_t trial_left = 1;
    std::memcpy(&expiration, license + 96, 8);
    std::memcpy(&trial_left, license + 24, 4);
    check(license[18] == 1 && license[19] == 0 && license[20] == 0 && license[21] == 0,
          "isActive (byte 18) is true; isTrialOwnedByThisUser, isDiscLicense, isTrial (19-21) are false");
    check(license[0] == 0 && license[28] == 0 && trial_left == 0 && expiration == 0,
          "empty SKU and trial ids, no trial time, expirationDate 0 (a license that doesn't expire)");
    XAsyncBlock bad_ctx{};
    check(at<CtxAsync>(store, 28)(store, license, &bad_ctx) == E_INVALIDARG, "a pointer that isn't a context is E_INVALIDARG");

    // ---- XStoreAcquireLicenseForPackageAsync / Result, IsLicenseValid, license lost (20-23, 68, 69) ----
    using PkgAsync = HRESULT(STDMETHODCALLTYPE*)(void*, void*, const char*, XAsyncBlock*);
    using HandleResult = HRESULT(STDMETHODCALLTYPE*)(void*, XAsyncBlock*, void**);
    using IsValid = bool(STDMETHODCALLTYPE*)(void*, void*);
    using RegisterFn = HRESULT(STDMETHODCALLTYPE*)(void*, void*, XTaskQueueHandle, void*, void(CALLBACK*)(void*),
                                                   XTaskQueueRegistrationToken*);
    using UnregisterFn = bool(STDMETHODCALLTYPE*)(void*, void*, XTaskQueueRegistrationToken, bool);
    auto on_event = [](void*) {};
    XAsyncBlock acquire{};
    void* package_license = nullptr;
    check(at<PkgAsync>(store, 20)(store, ctx, "PackageId", &acquire) == S_OK && finished(&acquire) &&
              at<HandleResult>(store, 21)(store, &acquire, &package_license) == S_OK && package_license,
          "XStoreAcquireLicenseForPackageAsync hands out a license");
    check(at<IsValid>(store, 22)(store, package_license), "XStoreIsLicenseValid: true");
    check(!at<IsValid>(store, 22)(store, ctx), "a context is not a license: false");
    XTaskQueueRegistrationToken lost{};
    check(at<RegisterFn>(store, 68)(store, package_license, nullptr, nullptr, on_event, &lost) == S_OK && lost.token != 0 &&
              at<UnregisterFn>(store, 69)(store, package_license, lost, false),
          "XStoreRegisterPackageLicenseLost gives a token; unregistering is never pending (true)");
    XAsyncBlock acquire_null{};
    check(at<PkgAsync>(store, 20)(store, ctx, nullptr, &acquire_null) == E_INVALIDARG, "a null package identifier is E_INVALIDARG");
    XAsyncBlock durables{};
    void* durable_license = nullptr;
    check(at<PkgAsync>(store, 71)(store, ctx, "9NBLGGH4R315", &durables) == S_OK && finished(&durables) &&
              at<HandleResult>(store, 72)(store, &durables, &durable_license) == S_OK &&
              at<IsValid>(store, 22)(store, durable_license),
          "XStoreAcquireLicenseForDurablesAsync (IXStoreImpl3): a valid license");

    // ---- add-on licenses: none (30-32) ----
    using Count = HRESULT(STDMETHODCALLTYPE*)(void*, XAsyncBlock*, uint32_t*);
    using List = HRESULT(STDMETHODCALLTYPE*)(void*, XAsyncBlock*, uint32_t, void*);
    XAsyncBlock addons{};
    uint32_t count = 99;
    check(at<CtxAsync>(store, 30)(store, ctx, &addons) == S_OK && finished(&addons) &&
              at<Count>(store, 31)(store, &addons, &count) == S_OK && count == 0 &&
              at<List>(store, 32)(store, &addons, 0, nullptr) == S_OK,
          "XStoreQueryAddOnLicenses*: count 0, then Result(0) is S_OK");
    check(at<List>(store, 32)(store, &addons, 1, license) == E_INVALIDARG, "asking for more add-ons than the count is E_INVALIDARG");

    // ---- package updates: none (55-57; IXStoreImpl5 79-81) ----
    using UpdatesAsync = HRESULT(STDMETHODCALLTYPE*)(void*, void*, const char**, size_t, XAsyncBlock*);
    XAsyncBlock updates{}, package_updates{};
    count = 99;
    check(at<CtxAsync>(store, 55)(store, ctx, &updates) == S_OK && finished(&updates) &&
              at<Count>(store, 56)(store, &updates, &count) == S_OK && count == 0 &&
              at<List>(store, 57)(store, &updates, 0, nullptr) == S_OK,
          "XStoreQueryGameAndDlcPackageUpdates*: no updates");
    const char* ids[] = {"PackageId"};
    count = 99;
    check(at<UpdatesAsync>(store, 79)(store, ctx, ids, 1, &package_updates) == S_OK && finished(&package_updates) &&
              at<Count>(store, 80)(store, &package_updates, &count) == S_OK && count == 0,
          "XStoreQueryPackageUpdates* (IXStoreImpl5): no updates");

    // ---- product queries: empty (11, 12, 15, 16, 19; 5, 6) ----
    using Enumerate = HRESULT(STDMETHODCALLTYPE*)(void*, void*, void*, bool(CALLBACK*)(const void*, void*));
    using HasMore = bool(STDMETHODCALLTYPE*)(void*, void*);
    using Associated = HRESULT(STDMETHODCALLTYPE*)(void*, void*, uint32_t, uint32_t, XAsyncBlock*);
    XAsyncBlock current{};
    void* products = nullptr;
    check(at<CtxAsync>(store, 11)(store, ctx, &current) == S_OK && finished(&current) &&
              at<HandleResult>(store, 12)(store, &current, &products) == S_OK && products,
          "XStoreQueryProductForCurrentGameAsync hands out a product query");
    int seen = 0;
    auto on_product = [](const void*, void* context) -> bool { ++*static_cast<int*>(context); return true; };
    check(at<Enumerate>(store, 15)(store, products, &seen, on_product) == S_OK && seen == 0,
          "XStoreEnumerateProductsQuery: S_OK, no products offline");
    check(!at<HasMore>(store, 16)(store, products), "XStoreProductsQueryHasMorePages: false");
    XAsyncBlock associated{};
    void* associated_products = nullptr;
    check(at<Associated>(store, 5)(store, ctx, 0x1ff, 25, &associated) == S_OK && finished(&associated) &&
              at<HandleResult>(store, 6)(store, &associated, &associated_products) == S_OK && associated_products,
          "XStoreQueryAssociatedProductsAsync: an (empty) product query");

    // ---- game license changed (66, 67) ----
    using RegisterCtx = HRESULT(STDMETHODCALLTYPE*)(void*, void*, XTaskQueueHandle, void*, void(CALLBACK*)(void*),
                                                    XTaskQueueRegistrationToken*);
    XTaskQueueRegistrationToken changed{};
    check(at<RegisterCtx>(store, 66)(store, ctx, nullptr, nullptr, on_event, &changed) == S_OK && changed.token != 0 &&
              at<UnregisterFn>(store, 67)(store, ctx, changed, true),
          "XStoreRegisterGameLicenseChanged / Unregister");

    // ---- offline answers ----
    using PackageIdentifier = HRESULT(STDMETHODCALLTYPE*)(void*, const char*, size_t, char*);
    using StoreIdAsync = HRESULT(STDMETHODCALLTYPE*)(void*, void*, const char*, XAsyncBlock*);
    using Purchase = HRESULT(STDMETHODCALLTYPE*)(void*, void*, const char*, const char*, const char*, XAsyncBlock*);
    char package_id[64] = "x";
    check(at<PackageIdentifier>(store, 65)(store, "9NBLGGH4R315", sizeof(package_id), package_id) ==
                  E_GAMEPACKAGE_NO_PACKAGE_IDENTIFIER && package_id[0] == 0,
          "XStoreQueryPackageIdentifier: E_GAMEPACKAGE_NO_PACKAGE_IDENTIFIER, empty string");
    XAsyncBlock can_acquire{}, purchase{};
    check(at<StoreIdAsync>(store, 24)(store, ctx, "9NBLGGH4R315", &can_acquire) == kStoreUnreachable,
          "XStoreCanAcquireLicenseForStoreIdAsync fails before starting");
    check(at<Purchase>(store, 49)(store, ctx, "9NBLGGH4R315", nullptr, nullptr, &purchase) == kStoreUnreachable,
          "XStoreShowPurchaseUIAsync fails before starting (no store UI offline)");
    check(at<CtxAsync>(store, 82)(store, ctx, &purchase) == kStoreUnreachable, "XStoreShowGiftingUIAsync (IXStoreImpl6) as well");

    // ---- handles: all accounted for ----
    check(rt::shims::xgameruntime_store_open_handles() == 5,
          "five handles are open: a context, two licenses, two product queries");
    at<CloseH>(store, 23)(store, package_license);
    at<CloseH>(store, 23)(store, durable_license);
    at<CloseH>(store, 19)(store, products);
    at<CloseH>(store, 19)(store, associated_products);
    at<CloseH>(store, 4)(store, ctx);
    check(rt::shims::xgameruntime_store_open_handles() == 0, "the Close functions close each of them");
    at<CloseH>(store, 23)(store, package_license);   // a double close: looked up, not read, so a no-op
    check(rt::shims::xgameruntime_store_open_handles() == 0 && !at<IsValid>(store, 22)(store, package_license),
          "a closed license is not valid and closing it again does nothing");
    using Unknown = HRESULT(STDMETHODCALLTYPE*)(void*, void*, size_t, char*);
    char probe[8] = "x";
    check(at<Unknown>(store, 46)(store, nullptr, sizeof(probe), probe) == E_NOTIMPL,
          "slot 46 (no public signature) answers E_NOTIMPL rather than S_OK with nothing written");
    using IsPurchasable = bool(STDMETHODCALLTYPE*)(void*, void*);
    check(!at<IsPurchasable>(store, 70)(store, nullptr), "XStoreIsAvailabilityPurchasable (IXStoreImpl2): false");

    std::printf("xgameruntime_store_test: %d of %d checks failed\n", g_failures, g_checks);
    return g_failures ? 1 : 0;
}
