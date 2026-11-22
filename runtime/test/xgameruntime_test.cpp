// Checks the registered xgameruntime shim the way a GDK title uses it at boot.
//
// Every function is resolved through rt::shims::find("xgameruntime.dll", name), the same table the loader binds
// imports to. The sequence follows a typical title: initialize the runtime, query the device, create a queue whose
// work runs on the thread pool and whose completions the game loop dispatches by hand, run async work on it and on
// the process queue, drive a custom XAsyncBegin provider to a result, then uninitialize. No game data; nothing
// online.
#include <windows.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iterator>
#include <string>
#include <vector>

#include <XAsync.h>
#include <XAsyncProvider.h>
#include <XTaskQueue.h>

#include "file_system.h"
#include "shims.h"

namespace {

int g_checks = 0;
int g_failures = 0;

void check(bool condition, const char* what) {
    ++g_checks;
    std::printf("%s %s\n", condition ? "PASS" : "FAIL", what);
    if (!condition) ++g_failures;
}

template <class Fn>
Fn get(const char* name) {
    return reinterpret_cast<Fn>(rt::shims::find("xgameruntime.dll", name));
}

using InitFn = HRESULT(__stdcall*)();
using UninitFn = void(__stdcall*)();
using DeviceTypeFn = uint32_t(__stdcall*)();
using ConsoleIdFn = HRESULT(__stdcall*)(size_t, char*, size_t*);
using QueueCreateFn = HRESULT(__stdcall*)(XTaskQueueDispatchMode, XTaskQueueDispatchMode, XTaskQueueHandle*);
using QueueCloseFn = void(__stdcall*)(XTaskQueueHandle);
using QueueDispatchFn = bool(__stdcall*)(XTaskQueueHandle, XTaskQueuePort, uint32_t);
using QueueProcessFn = bool(__stdcall*)(XTaskQueueHandle*);
using AsyncRunFn = HRESULT(__stdcall*)(XAsyncBlock*, XAsyncWork*);
using AsyncStatusFn = HRESULT(__stdcall*)(XAsyncBlock*, bool);
using AsyncBeginFn = HRESULT(__stdcall*)(XAsyncBlock*, void*, const void*, const char*, XAsyncProvider*);
using AsyncScheduleFn = HRESULT(__stdcall*)(XAsyncBlock*, uint32_t);
using AsyncCompleteFn = void(__stdcall*)(XAsyncBlock*, HRESULT, size_t);
using AsyncResultSizeFn = HRESULT(__stdcall*)(XAsyncBlock*, size_t*);
using AsyncResultFn = HRESULT(__stdcall*)(XAsyncBlock*, const void*, size_t, void*, size_t*);

struct alignas(8) XVersion {
    uint16_t major;
    uint16_t minor;
    uint16_t build;
    uint16_t revision;
};
static_assert(sizeof(XVersion) == 8, "XVersion must be 8 bytes per GDK spec");

struct XSystemAnalyticsInfo {
    XVersion osVersion;
    XVersion hostingOsVersion;
    char family[64];
    char form[64];
};
static_assert(sizeof(XSystemAnalyticsInfo) == 144, "XSystemAnalyticsInfo must be 144 bytes per GDK spec");

struct XPackageMountHandle {
    uint64_t handle;
};

using AnalyticsInfoFn = XSystemAnalyticsInfo(__stdcall*)();
using PackageIdentifierFn = HRESULT(__stdcall*)(size_t, char*);
using PackageMountFn = HRESULT(__stdcall*)(const char*, XPackageMountHandle*);
using PackageCloseMountHandleFn = void(__stdcall*)(XPackageMountHandle);
using PackageMountPathSizeFn = HRESULT(__stdcall*)(XPackageMountHandle, size_t*);
using PackageMountPathFn = HRESULT(__stdcall*)(XPackageMountHandle, size_t, char*);
using PackagedProcessFn = bool(__stdcall*)();

const char* const kNames[] = {
    "XGameRuntimeInitialize", "XGameRuntimeUninitialize", "XSystemGetDeviceType", "XSystemGetConsoleId",
    "XSystemGetAnalyticsInfo", "XPackageGetCurrentProcessPackageIdentifier", "XPackageMount",
    "XPackageCloseMountHandle", "XPackageGetMountPathSize", "XPackageGetMountPath",
    "XPackageIsPackagedProcess",
    "XTaskQueueCreate", "XTaskQueueCreateComposite", "XTaskQueueGetPort", "XTaskQueueDuplicateHandle",
    "XTaskQueueCloseHandle", "XTaskQueueTerminate", "XTaskQueueDispatch", "XTaskQueueSubmitCallback",
    "XTaskQueueSubmitDelayedCallback", "XTaskQueueRegisterWaiter", "XTaskQueueUnregisterWaiter",
    "XTaskQueueRegisterMonitor", "XTaskQueueUnregisterMonitor", "XTaskQueueGetCurrentProcessTaskQueue",
    "XTaskQueueSetCurrentProcessTaskQueue", "XAsyncGetStatus", "XAsyncGetResultSize", "XAsyncCancel", "XAsyncRun",
    "XAsyncBegin", "XAsyncSchedule", "XAsyncComplete", "XAsyncGetResult",
};

std::atomic<int> g_work_runs{0};
std::atomic<DWORD> g_work_thread{0};
std::atomic<int> g_completions{0};
std::atomic<DWORD> g_completion_thread{0};

HRESULT CALLBACK work(XAsyncBlock* b) {
    g_work_thread = GetCurrentThreadId();
    ++g_work_runs;
    *static_cast<int*>(b->context) = 42;
    return S_OK;
}

void CALLBACK completed(XAsyncBlock*) {
    g_completion_thread = GetCurrentThreadId();
    ++g_completions;
}

// A provider that produces a 4-byte result on the queue's work port: the shape GDK async APIs use internally.
struct ProviderState {
    AsyncCompleteFn complete;
    std::atomic<int> ops{0};
};

HRESULT CALLBACK provider(XAsyncOp op, const XAsyncProviderData* data) {
    auto* st = static_cast<ProviderState*>(data->context);
    st->ops |= 1 << static_cast<int>(op);
    switch (op) {
        case XAsyncOp::DoWork:
            st->complete(data->async, S_OK, sizeof(uint32_t));
            return S_OK;
        case XAsyncOp::GetResult:
            *static_cast<uint32_t*>(data->buffer) = 0x1EA0;
            return S_OK;
        default:
            return S_OK;
    }
}

HMODULE g_teardown_rt_mod = nullptr;
FARPROC(WINAPI* g_teardown_get_proc)(HMODULE, LPCSTR) = nullptr;
BOOL(WINAPI* g_teardown_free_lib)(HMODULE) = nullptr;

// Registered with std::atexit before check_run_time_load() constructs any function-local statics in library_loader.
// LIFO atexit order means this runs after those statics would have been destroyed, simulating mapped-DLL
// DLL_PROCESS_DETACH or earlier atexit handlers calling GetProcAddress/FreeLibrary on a pseudo-handle during exit.
void check_late_teardown() {
    if (!g_teardown_rt_mod || !g_teardown_get_proc || !g_teardown_free_lib) return;
    if (g_teardown_get_proc(g_teardown_rt_mod, "UninitializeApiImpl") == nullptr) {
        std::fprintf(stderr, "FAIL late teardown GetProcAddress on pseudo-handle\n");
        std::_Exit(1);
    }
    SetLastError(0);
    if (g_teardown_get_proc(g_teardown_rt_mod, "LateTeardownMissingSymbol") != nullptr ||
        GetLastError() != ERROR_PROC_NOT_FOUND) {
        std::fprintf(stderr, "FAIL late teardown unserved GetProcAddress on pseudo-handle\n");
        std::_Exit(1);
    }
    if (g_teardown_free_lib(g_teardown_rt_mod) != TRUE) {
        std::fprintf(stderr, "FAIL late teardown FreeLibrary on pseudo-handle\n");
        std::_Exit(1);
    }
}

// ---- The classes measured with tools/gdk_runtime_calls.py ----
//
// Each is asked for exactly as its thunks in the package's libHttpClient.GDK.dll ask: QueryApiImpl with the class id
// and the measured interface id, then one slot with `this` first and the documented arguments, then Release.

using QueryFn = HRESULT(WINAPI*)(const GUID*, REFIID, void**);

template <class F> F method(void* obj, int index) { return reinterpret_cast<F>((*static_cast<void* const* const*>(obj))[index]); }
void release_object(void* obj) { method<ULONG(STDMETHODCALLTYPE*)(void*)>(obj, 2)(obj); }
void* ask(QueryFn query, const GUID& cls, const GUID& iid) {
    void* p = nullptr;
    return query(&cls, iid, &p) == S_OK ? p : nullptr;
}
// Every id answers with the same object (a derived interface answers for each one it inherits).
bool answers_all(QueryFn query, const GUID& cls, const GUID* ids, size_t n, void* expected) {
    bool all = true;
    for (size_t i = 0; i < n; ++i) {
        void* same = nullptr;
        all = all && query(&cls, ids[i], &same) == S_OK && same == expected;
        if (same) release_object(same);
    }
    return all;
}

const GUID kSystem5Id = {0x1861cf2e, 0xe18b, 0x4834, {0xa9, 0xf5, 0xb4, 0xa4, 0xe6, 0xef, 0xb4, 0xcf}};

struct PlsSpaceInfo {
    uint64_t available_free, total_free, used, total;
};

std::vector<std::wstring> list_game_root() {
    using First = HANDLE(WINAPI*)(LPCWSTR, LPWIN32_FIND_DATAW);
    using Next = BOOL(WINAPI*)(HANDLE, LPWIN32_FIND_DATAW);
    using Close = BOOL(WINAPI*)(HANDLE);
    auto first = reinterpret_cast<First>(rt::shims::find("kernel32.dll", "FindFirstFileW"));
    auto next = reinterpret_cast<Next>(rt::shims::find("kernel32.dll", "FindNextFileW"));
    auto close = reinterpret_cast<Close>(rt::shims::find("kernel32.dll", "FindClose"));
    std::vector<std::wstring> names;
    if (!first || !next || !close) return names;
    WIN32_FIND_DATAW found{};
    const HANDLE h = first(L"GAME:\\*", &found);
    if (h == INVALID_HANDLE_VALUE) return names;
    do names.push_back(found.cFileName);
    while (next(h, &found));
    close(h);
    return names;
}

// XPersistentLocalStorageImpl as IXPersistentLocalStorageImpl3 (slots 3-8), over folders in %TEMP%.
void check_persistent_local_storage(QueryFn query) {
    const GUID cls = {0xf4faf4d4, 0x2d04, 0x4fce, {0xb3, 0xe0, 0x47, 0x4a, 0x71, 0x3a, 0x3e, 0x84}};
    const GUID chain[] = {cls,
                          {0xd29411df, 0x0794, 0x4553, {0x8b, 0x27, 0x95, 0xfc, 0x02, 0xd0, 0xf7, 0x5d}},
                          {0x41a4e10c, 0x5a7e, 0x41d9, {0x8c, 0x37, 0x37, 0xbd, 0xe6, 0x2a, 0x07, 0xd6}}};
    wchar_t temp[MAX_PATH] = {};
    GetTempPathW(MAX_PATH, temp);
    const std::filesystem::path root =
        std::filesystem::path(temp) / (L"leonida_xgameruntime_test_" + std::to_wstring(GetCurrentProcessId()));
    const std::filesystem::path game = root / L"game", overlay = root / L"overlay";
    std::error_code ec;
    std::filesystem::remove_all(root, ec);
    std::filesystem::create_directories(game, ec);
    if (HANDLE f = CreateFileW((game / L"data.bin").c_str(), GENERIC_WRITE, 0, nullptr, CREATE_NEW, 0, nullptr);
        f != INVALID_HANDLE_VALUE)
        CloseHandle(f);
    rt::fs::configure(game, overlay);

    void* p = ask(query, cls, chain[2]);
    check(p != nullptr, "QueryApiImpl hands out XPersistentLocalStorageImpl for IXPersistentLocalStorageImpl3");
    if (!p) return;
    check(answers_all(query, cls, chain, std::size(chain), p),
          "IXPersistentLocalStorageImpl3 answers for IXPersistentLocalStorageImpl and IXPersistentLocalStorageImpl2");

    using PathSize = HRESULT(STDMETHODCALLTYPE*)(void*, size_t*);
    using GetPath = HRESULT(STDMETHODCALLTYPE*)(void*, size_t, char*, size_t*);
    using SpaceInfo = HRESULT(STDMETHODCALLTYPE*)(void*, PlsSpaceInfo*);
    using PromptAsync = HRESULT(STDMETHODCALLTYPE*)(void*, uint64_t, XAsyncBlock*);
    using PromptResult = HRESULT(STDMETHODCALLTYPE*)(void*, XAsyncBlock*);
    using Mount = HRESULT(STDMETHODCALLTYPE*)(void*, const char*, uint64_t*);

    const std::filesystem::path pls = (std::filesystem::absolute(overlay) / L".gta6rt-pls").lexically_normal();
    check(!std::filesystem::exists(pls), "the storage folder does not exist before first use");
    size_t size = 0;
    check(method<PathSize>(p, 3)(p, &size) == S_OK && size > 1, "slot 3 (XPersistentLocalStorageGetPathSize) succeeds");
    check(method<PathSize>(p, 3)(p, nullptr) == E_INVALIDARG, "slot 3 refuses a null pathSize (E_INVALIDARG)");
    check(std::filesystem::is_directory(pls) && !std::filesystem::exists(game / L".gta6rt-pls"),
          "first use makes <overlay>\\.gta6rt-pls; nothing is written in the package");

    std::string path(size, 'x');
    size_t used = 0;
    const std::string expected = pls.string() + "\\";
    check(method<GetPath>(p, 4)(p, size, path.data(), &used) == S_OK && used == size && path[size - 1] == '\0' &&
              std::strlen(path.c_str()) == size - 1 && _stricmp(path.c_str(), expected.c_str()) == 0,
          "slot 4 (XPersistentLocalStorageGetPath) fills exactly GetPathSize's characters: the folder, a separator, "
          "the terminator");
    std::string short_buffer(size - 1, 'x');
    used = 77;
    check(method<GetPath>(p, 4)(p, size - 1, short_buffer.data(), &used) == E_FAIL && used == 77,
          "slot 4: a buffer one character short fails with E_FAIL (Xbox devices) and leaves pathUsed alone");
    SetEnvironmentVariableA("LEONIDA_DEVICE_TYPE", "1");
    check(method<GetPath>(p, 4)(p, size - 1, short_buffer.data(), &used) == HRESULT_FROM_WIN32(ERROR_INSUFFICIENT_BUFFER),
          "slot 4: the same with device type Pc is ERROR_INSUFFICIENT_BUFFER (PC devices)");
    SetEnvironmentVariableA("LEONIDA_DEVICE_TYPE", nullptr);
    check(method<GetPath>(p, 4)(p, size, nullptr, &used) == E_INVALIDARG &&
              method<GetPath>(p, 4)(p, size, path.data(), nullptr) == S_OK,
          "slot 4 refuses a null path; pathUsed is optional");

    // The title writes through the path it was given (a host path; the file shims pass it through).
    const std::string save = path.c_str() + std::string("save.bin");
    std::vector<char> bytes(4096, 'p');
    DWORD written = 0;
    if (HANDLE f = CreateFileA(save.c_str(), GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS, 0, nullptr);
        f != INVALID_HANDLE_VALUE) {
        WriteFile(f, bytes.data(), static_cast<DWORD>(bytes.size()), &written, nullptr);
        CloseHandle(f);
    }
    PlsSpaceInfo info{1, 2, 3, 4};
    check(written == 4096 && method<SpaceInfo>(p, 5)(p, &info) == S_OK && info.used == 4096 &&
              info.available_free > 0 && info.total_free == info.available_free &&
              info.total == info.used + info.available_free,
          "slot 5 (XPersistentLocalStorageGetSpaceInfo): the folder's files are used, the volume's free bytes "
          "available, the total both");
    check(method<SpaceInfo>(p, 5)(p, nullptr) == E_INVALIDARG, "slot 5 refuses a null info");

    bool package_seen = false, storage_seen = false;
    for (const std::wstring& name : list_game_root()) {
        package_seen = package_seen || _wcsicmp(name.c_str(), L"data.bin") == 0;
        storage_seen = storage_seen || _wcsicmp(name.c_str(), L".gta6rt-pls") == 0;
    }
    check(package_seen && !storage_seen, "a GAME:\\ listing shows the package, not the storage folder");

    std::atomic<int> completions{0};
    XAsyncBlock block{};
    block.context = &completions;
    block.callback = [](XAsyncBlock* b) { static_cast<std::atomic<int>*>(b->context)->fetch_add(1); };
    check(method<PromptAsync>(p, 6)(p, 1ull << 30, &block) == S_OK,
          "slot 6 (XPersistentLocalStoragePromptUserForSpaceAsync) starts the request");
    const HRESULT status = XAsyncGetStatus(&block, true);
    for (int i = 0; i < 500 && completions.load() == 0; ++i) Sleep(2);
    check(status == E_ABORT && method<PromptResult>(p, 7)(p, &block) == E_ABORT && completions.load() == 1,
          "no prompt offline: the request completes once, with E_ABORT, which slot 7 (...PromptUserForSpaceResult) "
          "returns");
    check(method<PromptAsync>(p, 6)(p, 1, nullptr) == E_INVALIDARG && method<PromptResult>(p, 7)(p, nullptr) == E_INVALIDARG,
          "slots 6 and 7 refuse a null XAsyncBlock");

    uint64_t mount = 0x1234;
    check(method<Mount>(p, 8)(p, "AnotherTitle_8wekyb3d8bbwe", &mount) == HRESULT_FROM_WIN32(ERROR_FILE_NOT_FOUND) &&
              mount == 0,
          "slot 8 (XPersistentLocalStorageMountForPackage): no other title's storage offline, FILE_NOT_FOUND, handle "
          "cleared");
    check(method<Mount>(p, 8)(p, nullptr, &mount) == E_INVALIDARG && method<Mount>(p, 8)(p, "x", nullptr) == E_INVALIDARG,
          "slot 8 refuses a null identifier or handle");

    rt::fs::configure({}, {});
    size = 7;
    check(method<PathSize>(p, 3)(p, &size) == HRESULT_FROM_WIN32(ERROR_PATH_NOT_FOUND) && size == 0,
          "without an overlay folder, slot 3 fails with ERROR_PATH_NOT_FOUND and reports 0");
    release_object(p);
    std::filesystem::remove_all(root, ec);
}

// XGameRuntimeFeatureImpl, XGameEventImpl, XSystemAnalyticsImpl and XDisplayImpl.
void check_measured_classes(QueryFn query, XSystemAnalyticsInfo (*named_analytics)()) {
    check_persistent_local_storage(query);

    const GUID feature_cls = {0x8836fe87, 0xedb9, 0x4fe3, {0x8d, 0xad, 0x05, 0xf0, 0xd2, 0xcd, 0x5b, 0x40}};
    void* f = ask(query, feature_cls, feature_cls);
    check(f != nullptr, "QueryApiImpl hands out XGameRuntimeFeatureImpl for its class id");
    if (f) {
        using Available = bool(STDMETHODCALLTYPE*)(void*, uint32_t);  // bool, per Microsoft's reference page
        auto available = method<Available>(f, 3);
        // XGameRuntimeFeature values: XAsync 2, XAsyncProvider 3, XDisplay 4, XNetworking 10, XPackage 11,
        // XPersistentLocalStorage 12, XStore 14, XSystem 15, XTaskQueue 16, XThread 17, XError 19, XGameEvent 20.
        bool served = true;
        for (uint32_t v : {2u, 3u, 10u, 12u, 15u, 16u, 17u, 19u, 20u}) served = served && available(f, v);
        check(served, "slot 3 (XGameRuntimeIsFeatureAvailable): true for the features whose classes are served");
        check(!available(f, 4) && !available(f, 14) && !available(f, 22) && !available(f, 0xFFFFFFFFu),
              "slot 3: false for XDisplay (documented for the consoles only), XStore and values past the enum");
        // Every other feature follows whether QueryApiImpl serves its class.
        const struct {
            uint32_t feature;
            GUID cls;
        } follow[] = {
            {0, {0x3e241977, 0x6237, 0x41e9, {0x85, 0x59, 0x10, 0x2c, 0x2d, 0x99, 0x83, 0xf1}}},   // XAccessibility
            {1, {0xa4f1aee2, 0x4bf1, 0x4485, {0xb0, 0x08, 0xa7, 0xc2, 0x6d, 0x52, 0xac, 0x27}}},   // XAppCapture
            {5, {0x973a344e, 0x24bf, 0x4d0f, {0x84, 0x57, 0x56, 0xc5, 0x34, 0x89, 0x2b, 0x29}}},   // XGame
            {6, {0x0651aae2, 0x4012, 0x4077, {0xbf, 0x84, 0x8b, 0x90, 0x97, 0x09, 0x0e, 0x2c}}},   // XGameInvite
            {7, {0x704c3f58, 0xe629, 0x4cc2, {0xb1, 0x97, 0x30, 0x51, 0x1b, 0x99, 0x6f, 0xe2}}},   // XGameSave
            {8, {0xdfcd4649, 0x4ff8, 0x4043, {0xba, 0x07, 0x35, 0xd6, 0x07, 0xdf, 0x98, 0xb0}}},   // XGameUI
            {9, {0x1b339674, 0x328d, 0x4283, {0xa2, 0x00, 0x31, 0x71, 0xf1, 0x8d, 0x36, 0x39}}},   // XLauncher
            {11, {0xaf406016, 0xe850, 0x4aa8, {0xa8, 0x8d, 0x2f, 0x3d, 0xcb, 0x9d, 0xac, 0x7e}}},  // XPackage
            {13, {0x3e241977, 0x6237, 0x41e9, {0x85, 0x59, 0x10, 0x2c, 0x2d, 0x99, 0x83, 0xf1}}},  // XSpeechSynthesizer
            {18, {0x01acd177, 0x91f9, 0x4763, {0xa3, 0x8e, 0xcc, 0xbb, 0x55, 0xce, 0x32, 0xe0}}},  // XUser
            {21, {0x0a2192aa, 0xb2d5, 0x4d58, {0x83, 0xbe, 0x38, 0x3b, 0x6d, 0x80, 0x79, 0x9e}}},  // XGameStreaming
        };
        bool consistent = true;
        for (const auto& entry : follow) {
            void* obj = nullptr;
            const bool class_served = query(&entry.cls, entry.cls, &obj) == S_OK;
            if (obj) release_object(obj);
            consistent = consistent && available(f, entry.feature) == class_served;
        }
        check(consistent, "slot 3: every other feature is available exactly when QueryApiImpl serves its class");
        release_object(f);
    }

    const GUID event_cls = {0xbbfbdcc7, 0xbfe7, 0x409b, {0xa5, 0xca, 0xed, 0xf0, 0x54, 0x96, 0x0b, 0x4d}};
    void* e = ask(query, event_cls, event_cls);
    check(e != nullptr, "QueryApiImpl hands out XGameEventImpl for its class id");
    if (e) {
        using Write = HRESULT(STDMETHODCALLTYPE*)(void*, void*, const char*, const char*, const char*, const char*,
                                                  const char*);
        auto write = method<Write>(e, 3);
        int user = 0;  // stands in for an XUserHandle
        const char* scid = "00000000-0000-0000-0000-000000000000";
        const char* session = "11111111-2222-3333-4444-555555555555";
        check(write(e, &user, scid, session, "MatchEnd", "{\"mode\":\"offline\"}", "{\"score\":42}") == S_OK &&
                  write(e, &user, scid, session, "MatchStart", nullptr, nullptr) == S_OK,
              "slot 3 (XGameEventWrite) accepts an event, with or without its optional JSON fields");
        check(write(e, nullptr, scid, session, "x", nullptr, nullptr) == E_INVALIDARG &&
                  write(e, &user, nullptr, session, "x", nullptr, nullptr) == E_INVALIDARG &&
                  write(e, &user, scid, nullptr, "x", nullptr, nullptr) == E_INVALIDARG &&
                  write(e, &user, scid, session, nullptr, nullptr, nullptr) == E_INVALIDARG,
              "slot 3 refuses a null user, SCID, play session or event name (E_INVALIDARG)");
        bool all = true;
        for (int i = 0; i < 100; ++i) all = all && write(e, &user, scid, session, "Tick", nullptr, "{\"n\":1}") == S_OK;
        check(all, "slot 3 keeps accepting events once its log lines are spent");
        release_object(e);
    }

    // A member function returning a struct, called the way MSVC calls one (the result's address after `this`, the
    // same address back in rax): what the thunk does.
    struct AnalyticsView {
        virtual HRESULT STDMETHODCALLTYPE QueryInterface(REFIID, void**) = 0;
        virtual ULONG STDMETHODCALLTYPE AddRef() = 0;
        virtual ULONG STDMETHODCALLTYPE Release() = 0;
        virtual XSystemAnalyticsInfo STDMETHODCALLTYPE GetAnalyticsInfo() = 0;
    };
    const GUID analytics_cls = {0xb884675d, 0xb738, 0x4a9c, {0x81, 0x5d, 0x9a, 0x9a, 0x1e, 0x0c, 0x6c, 0x9b}};
    void* a = ask(query, analytics_cls, analytics_cls);
    check(a != nullptr, "QueryApiImpl hands out XSystemAnalyticsImpl for its class id");
    if (a) {
        const XSystemAnalyticsInfo via_slot = static_cast<AnalyticsView*>(a)->GetAnalyticsInfo();
        const XSystemAnalyticsInfo named = named_analytics();
        check(std::memcmp(&via_slot, &named, sizeof named) == 0 && std::strcmp(via_slot.family, "Xbox") == 0,
              "slot 3 (XSystemGetAnalyticsInfo) returns the named export's struct by value");
        release_object(a);
    }

    const GUID display_cls = {0x03f0fe74, 0xfdd9, 0x4e5c, {0xb6, 0x30, 0xf9, 0x33, 0x9c, 0x47, 0xac, 0xc5}};
    const GUID display_if = {0x35f07670, 0x706e, 0x4bfb, {0x94, 0x76, 0x09, 0x07, 0x98, 0xc5, 0xeb, 0xf3}};
    void* d = ask(query, display_cls, display_if);
    check(d != nullptr, "QueryApiImpl hands out XDisplayImpl for IXDisplayImpl");
    if (d) {
        struct HdrInfo {
            float min_tml, max_tml, max_fftml;
        } info{1.f, 2.f, 3.f};
        using TryHdr = uint32_t(STDMETHODCALLTYPE*)(void*, uint32_t, HdrInfo*);  // XDisplayHdrModeResult : uint32_t
        auto try_hdr = method<TryHdr>(d, 4);
        check(try_hdr(d, 0, &info) == 2 && info.min_tml == 0.f && info.max_tml == 0.f && info.max_fftml == 0.f,
              "slot 4 (XDisplayTryEnableHdrMode), PreferHdr: Disabled (2), tone-map values zeroed");
        check(try_hdr(d, 1, nullptr) == 2, "slot 4, PreferRefreshRate with no info: Disabled");
        info = {1.f, 2.f, 3.f};
        check(try_hdr(d, 7, &info) == 0 && info.max_tml == 0.f, "slot 4: a preference outside the enum is Unknown (0)");
        check(method<HRESULT(STDMETHODCALLTYPE*)(void*)>(d, 3)(d) == E_NOTIMPL,
              "slot 3 (undocumented, never called) answers E_NOTIMPL");
        void* same = ask(query, display_cls, display_cls);
        void* other = reinterpret_cast<void*>(1);
        check(same == d && query(&display_cls, kSystem5Id, &other) == E_NOINTERFACE && other == nullptr,
              "XDisplayImpl answers its class id and IXDisplayImpl, not another class's interface");
        if (same) release_object(same);
        release_object(d);
    }

    // Classes nobody serves yet are still refused, and logged, as before.
    const GUID refused[] = {
        {0x1b339674, 0x328d, 0x4283, {0xa2, 0x00, 0x31, 0x71, 0xf1, 0x8d, 0x36, 0x39}},  // XLauncherImpl
        {0x0a2192aa, 0xb2d5, 0x4d58, {0x83, 0xbe, 0x38, 0x3b, 0x6d, 0x80, 0x79, 0x9e}},  // XGameStreamingImpl
        {0x0dd112ac, 0x7c24, 0x448c, {0xb9, 0x2b, 0x39, 0x60, 0xfb, 0x5b, 0xd3, 0x0c}},  // XStoreImpl
        {0x12345678, 0x9abc, 0xdef0, {0x12, 0x34, 0x56, 0x78, 0x9a, 0xbc, 0xde, 0xf0}},  // no such class
    };
    bool all_refused = true;
    for (const GUID& cls : refused) {
        void* out = reinterpret_cast<void*>(1);
        all_refused = all_refused && query(&cls, cls, &out) == static_cast<HRESULT>(0x80070032) && out == nullptr;
    }
    check(all_refused, "XLauncherImpl, XGameStreamingImpl, XStoreImpl and an unknown class are still refused");
}

// xgameruntime.lib's thunks don't import the runtime: they LoadLibraryW(L"xgameruntime.dll") and GetProcAddress five
// names (measured in the package's libHttpClient.GDK.dll and XCurl.dll). Drive that path through the loader shims,
// bound by the api set XCurl.dll imports them from, as a mapped image would be.
void check_run_time_load() {
    using LoadW = HMODULE(WINAPI*)(LPCWSTR);
    using LoadExA = HMODULE(WINAPI*)(LPCSTR, HANDLE, DWORD);
    using GetModW = HMODULE(WINAPI*)(LPCWSTR);
    using GetProc = FARPROC(WINAPI*)(HMODULE, LPCSTR);
    using Free = BOOL(WINAPI*)(HMODULE);
    const char* set = "api-ms-win-core-libraryloader-l1-2-0.dll";
    auto load_w = reinterpret_cast<LoadW>(rt::shims::find(set, "LoadLibraryW"));
    auto load_ex_a = reinterpret_cast<LoadExA>(rt::shims::find("kernel32.dll", "LoadLibraryExA"));
    auto get_mod_w = reinterpret_cast<GetModW>(rt::shims::find(set, "GetModuleHandleW"));
    auto get_proc = reinterpret_cast<GetProc>(rt::shims::find(set, "GetProcAddress"));
    auto free_lib = reinterpret_cast<Free>(rt::shims::find(set, "FreeLibrary"));
    check(load_w && load_ex_a && get_mod_w && get_proc && free_lib, "the loader functions are registered");
    if (!(load_w && load_ex_a && get_mod_w && get_proc && free_lib)) return;

    check(rt::shims::owns_dll("xgameruntime.dll") && !rt::shims::owns_dll("kernel32.dll") &&
              !rt::shims::owns_dll(set) && !rt::shims::owns_dll("nosuch.dll"),
          "owns_dll: xgameruntime.dll yes; kernel32, api sets and unknown modules no");

    HMODULE rt_mod = load_w(L"xgameruntime.dll");
    check(rt_mod != nullptr, "LoadLibraryW(xgameruntime.dll) is served without the host runtime");
    check(load_w(L"XGameRuntime") == rt_mod && load_ex_a("C:\\Some\\Dir\\XGAMERUNTIME.DLL", nullptr, 0) == rt_mod &&
              get_mod_w(L"xgameruntime.dll") == rt_mod,
          "the same module for a bare name, a full path in any case, LoadLibraryExA and GetModuleHandleW");

    const char* entry_points[] = {"InitializeApiImpl", "InitializeApiImplEx", "InitializeApiImplEx2",
                                  "UninitializeApiImpl", "QueryApiImpl"};
    bool all = true;
    for (const char* n : entry_points)
        all = all && get_proc(rt_mod, n) &&
              reinterpret_cast<void*>(get_proc(rt_mod, n)) == rt::shims::find("xgameruntime.dll", n);
    check(all, "GetProcAddress serves the five entry points xgameruntime.lib looks up");
    SetLastError(0);
    check(get_proc(rt_mod, "NoSuchRuntimeFunction") == nullptr && GetLastError() == ERROR_PROC_NOT_FOUND,
          "an unserved name is NULL with ERROR_PROC_NOT_FOUND, as from a real module");
    SetLastError(0);
    check(get_proc(rt_mod, nullptr) == nullptr && GetLastError() == ERROR_PROC_NOT_FOUND,
          "a null proc name is NULL with ERROR_PROC_NOT_FOUND");

    // XCurl.dll's other way in: LoadLibraryExW(L"gameplatformservices.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32),
    // then GetApiVersionImpl(uint64_t*, uint64_t*) and QueryApiImpl. Served by the shims, never the PC's own.
    using LoadExW = HMODULE(WINAPI*)(LPCWSTR, HANDLE, DWORD);
    auto load_ex_w = reinterpret_cast<LoadExW>(rt::shims::find(set, "LoadLibraryExW"));
    HMODULE gps = load_ex_w ? load_ex_w(L"gameplatformservices.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32) : nullptr;
    check(rt::shims::owns_dll("gameplatformservices.dll") && gps != nullptr && gps != rt_mod,
          "LoadLibraryExW(gameplatformservices.dll, SEARCH_SYSTEM32) is served by the shims");
    using ApiVersion = HRESULT(WINAPI*)(uint64_t*, uint64_t*);
    auto api_version = gps ? reinterpret_cast<ApiVersion>(get_proc(gps, "GetApiVersionImpl")) : nullptr;
    uint64_t v1 = 0, v2 = 0, w1 = 0, w2 = 0;
    check(api_version && api_version(&v1, &v2) == S_OK && api_version(&w1, &w2) == S_OK && v1 && v2 && v1 == w1 &&
              v2 == w2,
          "GetApiVersionImpl writes the same two non-zero versions on every call");
    check(gps && reinterpret_cast<void*>(get_proc(gps, "QueryApiImpl")) ==
                     rt::shims::find("xgameruntime.dll", "QueryApiImpl"),
          "its QueryApiImpl is the offline runtime's");

    struct Options {
        uint32_t unknown;
        BOOL is_inline;
        const char* game_config;
    } options{1, FALSE, nullptr};
    using InitEx2 = HRESULT(WINAPI*)(ULONG, ULONG, char, const Options*);
    using Query = HRESULT(WINAPI*)(const GUID*, REFIID, void**);
    using Uninit = HRESULT(WINAPI*)();
    auto init_ex2 = reinterpret_cast<InitEx2>(get_proc(rt_mod, "InitializeApiImplEx2"));
    auto query = reinterpret_cast<Query>(get_proc(rt_mod, "QueryApiImpl"));
    auto uninit_api = reinterpret_cast<Uninit>(get_proc(rt_mod, "UninitializeApiImpl"));
    check(init_ex2(0x2410, 0x0, 0x02, &options) == S_OK, "InitializeApiImplEx2 succeeds offline");
    const GUID some_class = {0x973a344e, 0x24bf, 0x4d0f, {0x84, 0x57, 0x56, 0xc5, 0x34, 0x89, 0x2b, 0x29}};
    void* out = reinterpret_cast<void*>(1);
    check(query(&some_class, some_class, &out) == static_cast<HRESULT>(0x80070032) && out == nullptr,
          "QueryApiImpl refuses a class it doesn't provide yet (HRESULT 0x80070032) and clears the out pointer");

    // XThreadingImpl and XErrorImpl, driven the way xgameruntime.lib's thunks in the package's DLLs do: QueryApiImpl
    // with the class id as both ids, one vtable slot with `this` first, then Release (slot 2).
    const GUID threading = {0x073b7dcb, 0x1fcf, 0x4030, {0x94, 0xbe, 0xe3, 0xc9, 0xeb, 0x62, 0x34, 0x28}};
    const GUID error_cls = {0x8ca467f7, 0x22e8, 0x4096, {0x84, 0x56, 0xbb, 0x8a, 0xa1, 0x3f, 0x79, 0xd8}};
    using Fn = void*;
    auto slot = [](void* obj, int index) { return (*static_cast<Fn* const*>(obj))[index]; };
    auto release = [&](void* obj) { reinterpret_cast<ULONG(STDMETHODCALLTYPE*)(void*)>(slot(obj, 2))(obj); };
    auto get = [&](const GUID& cls) {
        void* p = nullptr;
        return query(&cls, cls, &p) == S_OK ? p : nullptr;
    };
    void* t = get(threading);
    check(t != nullptr, "QueryApiImpl hands out XThreadingImpl for its class id");
    if (t) {
        void* again = get(threading);
        check(again == t, "XThreadingImpl is one process-wide object");
        release(again);
        XTaskQueueHandle queue = nullptr;
        using Create = HRESULT(STDMETHODCALLTYPE*)(void*, XTaskQueueDispatchMode, XTaskQueueDispatchMode,
                                                  XTaskQueueHandle*);
        check(reinterpret_cast<Create>(slot(t, 12))(t, XTaskQueueDispatchMode::Manual, XTaskQueueDispatchMode::Manual,
                                                    &queue) == S_OK && queue,
              "slot 12 is XTaskQueueCreate");
        static std::atomic<int> s_ran{0};
        using Submit = HRESULT(STDMETHODCALLTYPE*)(void*, XTaskQueueHandle, XTaskQueuePort, void*,
                                                  XTaskQueueCallback*);
        XTaskQueueCallback* cb = [](void* ctx, bool canceled) {
            if (!canceled) static_cast<std::atomic<int>*>(ctx)->fetch_add(1);
        };
        check(reinterpret_cast<Submit>(slot(t, 18))(t, queue, XTaskQueuePort::Work, &s_ran, cb) == S_OK,
              "slot 18 is XTaskQueueSubmitCallback");
        using Dispatch = bool(STDMETHODCALLTYPE*)(void*, XTaskQueueHandle, XTaskQueuePort, uint32_t);
        check(reinterpret_cast<Dispatch>(slot(t, 16))(t, queue, XTaskQueuePort::Work, 0) && s_ran.load() == 1,
              "slot 16 is XTaskQueueDispatch: the callback ran on the work port");
        XTaskQueueHandle dup = nullptr;
        using Dup = HRESULT(STDMETHODCALLTYPE*)(void*, XTaskQueueHandle, XTaskQueueHandle*);
        check(reinterpret_cast<Dup>(slot(t, 15))(t, queue, &dup) == S_OK && dup,
              "slot 15 is XTaskQueueDuplicateHandle");
        using Close = void(STDMETHODCALLTYPE*)(void*, XTaskQueueHandle);
        reinterpret_cast<Close>(slot(t, 17))(t, dup);
        reinterpret_cast<Close>(slot(t, 17))(t, queue);
        using SetSensitive = HRESULT(STDMETHODCALLTYPE*)(void*, bool);
        using IsSensitive = bool(STDMETHODCALLTYPE*)(void*);
        check(reinterpret_cast<SetSensitive>(slot(t, 27))(t, true) == S_OK &&
                  reinterpret_cast<IsSensitive>(slot(t, 30))(t) &&
                  reinterpret_cast<SetSensitive>(slot(t, 27))(t, false) == S_OK &&
                  !reinterpret_cast<IsSensitive>(slot(t, 30))(t),
              "slots 27 and 30 set and read this thread's time-sensitive flag");
        void* unk = nullptr;
        const GUID iunknown = {0, 0, 0, {0xc0, 0, 0, 0, 0, 0, 0, 0x46}};
        using QI = HRESULT(STDMETHODCALLTYPE*)(void*, REFIID, void**);
        check(reinterpret_cast<QI>(slot(t, 0))(t, iunknown, &unk) == S_OK && unk == t &&
                  reinterpret_cast<QI>(slot(t, 0))(t, error_cls, &unk) == E_NOINTERFACE && !unk,
              "QueryInterface answers IUnknown and the class id, not another class");
        release(t);
        release(t);
    }
    void* e = get(error_cls);
    check(e != nullptr, "QueryApiImpl hands out XErrorImpl for its class id");
    if (e) {
        struct Seen {
            HRESULT hr;
            char msg[32];
        } seen{};
        using Callback = bool(CALLBACK*)(HRESULT, const char*, void*);
        Callback on_error = [](HRESULT hr, const char* msg, void* ctx) {
            auto* s = static_cast<Seen*>(ctx);
            s->hr = hr;
            strncpy_s(s->msg, msg ? msg : "", _TRUNCATE);
            return true;
        };
        using SetCallback = void(STDMETHODCALLTYPE*)(void*, Callback, void*);
        using Report = HRESULT(STDMETHODCALLTYPE*)(void*, HRESULT, const char*);
        reinterpret_cast<SetCallback>(slot(e, 4))(e, on_error, &seen);
        const HRESULT reported = reinterpret_cast<Report>(slot(e, 3))(e, E_FAIL, "offline test");
        check(reported == S_OK && seen.hr == E_FAIL && std::strcmp(seen.msg, "offline test") == 0,
              "slot 3 is XErrorReport and reaches the callback slot 4 (XErrorSetCallback) set");
        reinterpret_cast<SetCallback>(slot(e, 4))(e, nullptr, nullptr);
        using SetOptions = void(STDMETHODCALLTYPE*)(void*, uint32_t, uint32_t);  // void, per Microsoft's page
        reinterpret_cast<SetOptions>(slot(e, 5))(e, 0x01, 0x00);  // OutputDebugStringOnError / None
        release(e);
    }

    // XNetworkingImpl, asked for as XCurl.dll does: the class id with IXNetworkingImpl2's interface id.
    const GUID networking = {0x37e56907, 0x2f10, 0x41e8, {0xb7, 0x2f, 0x36, 0xed, 0xb1, 0x85, 0x33, 0x1a}};
    const GUID networking2 = {0xbf2346b2, 0x39af, 0x4658, {0xb5, 0xea, 0x44, 0x71, 0x3c, 0x7e, 0x83, 0xb3}};
    const HRESULT no_network = HRESULT_FROM_WIN32(ERROR_NETWORK_UNREACHABLE);
    void* n = nullptr;
    check(query(&networking, networking2, &n) == S_OK && n, "QueryApiImpl hands out XNetworkingImpl for IXNetworkingImpl2");
    void* n1 = nullptr;
    check(query(&networking, networking, &n1) == S_OK && n1 == n, "and the same object for IXNetworkingImpl");
    if (n1) release(n1);
    if (n) {
        struct Hint {
            uint32_t level, cost, iana;
            bool initialized, approaching, over, roaming;
        } hint{9, 9, 9, false, true, true, true};
        using GetHint = HRESULT(STDMETHODCALLTYPE*)(void*, Hint*);
        check(reinterpret_cast<GetHint>(slot(n, 15))(n, &hint) == S_OK && hint.level == 1 && hint.cost == 0 &&
                  hint.initialized && !hint.approaching && !hint.over && !hint.roaming,
              "slot 15 (XNetworkingGetConnectivityHint): initialized, no connectivity");
        XAsyncBlock block{};
        using SecurityAsync = HRESULT(STDMETHODCALLTYPE*)(void*, const wchar_t*, XAsyncBlock*);
        using ResultSize = HRESULT(STDMETHODCALLTYPE*)(void*, XAsyncBlock*, size_t*);
        size_t size = 99;
        check(reinterpret_cast<SecurityAsync>(slot(n, 11))(n, L"https://example.invalid/", &block) == S_OK &&
                  XAsyncGetStatus(&block, true) == no_network &&
                  reinterpret_cast<ResultSize>(slot(n, 12))(n, &block, &size) == no_network && size == 0,
              "slots 11/12 (UTF-16 URL security info): the async call completes with no network");
        // Slots 10/13 in Microsoft's order: (async, bufferByteCount, bufferByteCountUsed*, buffer*, information**).
        // A sentinel buffer must stay untouched; the two outputs are written where they are, not into the buffer.
        using SecurityResult = HRESULT(STDMETHODCALLTYPE*)(void*, XAsyncBlock*, size_t, size_t*, uint8_t*, void**);
        for (int s : {10, 13}) {
            uint8_t buffer[32];
            std::memset(buffer, 0xA5, sizeof(buffer));
            size_t used = 77;
            void* info = &used;
            const HRESULT hr = reinterpret_cast<SecurityResult>(slot(n, s))(n, &block, sizeof(buffer), &used, buffer, &info);
            bool untouched = true;
            for (uint8_t byte : buffer) untouched = untouched && byte == 0xA5;
            check(hr == no_network && used == 0 && info == nullptr && untouched,
                  s == 10 ? "slot 10 (ANSI result): outputs zeroed in place, the caller's buffer untouched"
                          : "slot 13 (UTF-16 result): outputs zeroed in place, the caller's buffer untouched");
        }
        using Verify = HRESULT(STDMETHODCALLTYPE*)(void*, void*, const void*);
        check(reinterpret_cast<Verify>(slot(n, 14))(n, nullptr, nullptr) == no_network,
              "slot 14 (XNetworkingVerifyServerCertificate) fails offline");
        XTaskQueueRegistrationToken token{};
        using Register = HRESULT(STDMETHODCALLTYPE*)(void*, XTaskQueueHandle, void*, void*, XTaskQueueRegistrationToken*);
        using Unregister = bool(STDMETHODCALLTYPE*)(void*, XTaskQueueRegistrationToken, bool);
        check(reinterpret_cast<Register>(slot(n, 16))(n, nullptr, nullptr, nullptr, &token) == S_OK && token.token &&
                  reinterpret_cast<Unregister>(slot(n, 17))(n, token, true),
              "slots 16/17 register and unregister a connectivity-hint callback");
        uint16_t port = 7;
        using QueryPort = HRESULT(STDMETHODCALLTYPE*)(void*, uint16_t*);
        using NotImpl = HRESULT(STDMETHODCALLTYPE*)(void*);
        check(reinterpret_cast<QueryPort>(slot(n, 3))(n, &port) == no_network && port == 0 &&
                  reinterpret_cast<NotImpl>(slot(n, 20))(n) == E_NOTIMPL,
              "slot 3 (preferred UDP port) has no answer offline; slot 20 (IXNetworkingImpl2) is E_NOTIMPL");
        release(n);
    }

    // XSystemImpl, asked for as libHttpClient.GDK.dll does: the class id with IXSystemImpl5's interface id.
    const GUID system_cls = {0xe349bd1a, 0xfc20, 0x4e40, {0xb9, 0x9c, 0x41, 0x78, 0xcc, 0x6b, 0x40, 0x9f}};
    const GUID system5 = {0x1861cf2e, 0xe18b, 0x4834, {0xa9, 0xf5, 0xb4, 0xa4, 0xe6, 0xef, 0xb4, 0xcf}};
    void* sy = nullptr;
    check(query(&system_cls, system5, &sy) == S_OK && sy, "QueryApiImpl hands out XSystemImpl for IXSystemImpl5");
    if (sy) {
        using IdQuery = HRESULT(STDMETHODCALLTYPE*)(void*, size_t, char*, size_t*);
        char buf[64] = {};
        size_t used = 0;
        check(reinterpret_cast<IdQuery>(slot(sy, 3))(sy, sizeof(buf), buf, &used) == S_OK && used == 39 &&
                  std::strlen(buf) == 38,
              "slot 3 (XSystemGetConsoleId): the 39-byte placeholder");
        check(reinterpret_cast<IdQuery>(slot(sy, 4))(sy, sizeof(buf), buf, &used) == S_OK &&
                  std::strcmp(buf, "RETAIL") == 0 && used == 7,
              "slot 4 (XSystemGetXboxLiveSandboxId): RETAIL");
        check(reinterpret_cast<IdQuery>(slot(sy, 4))(sy, 15, buf, &used) == HRESULT_FROM_WIN32(ERROR_INSUFFICIENT_BUFFER) &&
                  reinterpret_cast<IdQuery>(slot(sy, 4))(sy, 16, buf, &used) == S_OK,
              "slot 4 needs XSystemXboxLiveSandboxIdMaxBytes (16), not just the id's 7 bytes");
        const GUID chain[] = {
            system_cls,
            {0x6fd71f09, 0x7513, 0x49f0, {0x89, 0xbc, 0xbf, 0xaf, 0x5d, 0xf6, 0xf8, 0x52}},
            {0x67ce4bfc, 0xb1d1, 0x4ac7, {0xbc, 0x3a, 0xcb, 0x92, 0x19, 0xa9, 0x7a, 0x85}},
            {0xdadc2895, 0x34b0, 0x4ef5, {0xa8, 0x3e, 0x45, 0x11, 0x4d, 0x62, 0x9b, 0x80}},
            system5};
        bool all = true;
        for (const GUID& iid : chain) {
            void* same = nullptr;
            all = all && query(&system_cls, iid, &same) == S_OK && same == sy;
            if (same) release(same);
        }
        check(all, "IXSystemImpl5 answers for every interface it inherits (IXSystemImpl .. IXSystemImpl4)");
        check(reinterpret_cast<IdQuery>(slot(sy, 5))(sy, sizeof(buf), buf, &used) == S_OK && used == 45 &&
                  reinterpret_cast<IdQuery>(slot(sy, 5))(sy, 44, buf, &used) ==
                      HRESULT_FROM_WIN32(ERROR_INSUFFICIENT_BUFFER) &&
                  reinterpret_cast<IdQuery>(slot(sy, 5))(sy, 45, nullptr, &used) == E_INVALIDARG,
              "slot 5 (XSystemGetAppSpecificDeviceId): 45 bytes; a short or null buffer is refused");
        using Track = HRESULT(STDMETHODCALLTYPE*)(void*, void*, void*);
        using IsValid = bool(STDMETHODCALLTYPE*)(void*, void*);  // bool, per Microsoft's reference page
        using Bandwidth = void(STDMETHODCALLTYPE*)(void*, bool);
        int ctx = 0;
        reinterpret_cast<Bandwidth>(slot(sy, 8))(sy, true);
        check(reinterpret_cast<Track>(slot(sy, 6))(sy, &ctx, &ctx) == S_OK &&
                  reinterpret_cast<IsValid>(slot(sy, 7))(sy, &ctx) &&
                  !reinterpret_cast<IsValid>(slot(sy, 7))(sy, nullptr),
              "slots 6-8 (handle tracking, handle check, download bandwidth) answer offline");
        release(sy);
    }
    check_measured_classes(query,
                           reinterpret_cast<AnalyticsInfoFn>(rt::shims::find("xgameruntime.dll", "XSystemGetAnalyticsInfo")));
    check(uninit_api() == S_OK, "UninitializeApiImpl succeeds");

    HMODULE k32 = load_w(L"kernel32.dll");
    check(k32 == GetModuleHandleW(L"kernel32.dll"), "a host module still loads from the host");
    check(reinterpret_cast<void*>(get_proc(k32, "GetTickCount")) ==
              reinterpret_cast<void*>(GetProcAddress(k32, "GetTickCount")),
          "GetProcAddress on a host module is the host's");
    check(free_lib(rt_mod) == TRUE && load_w(L"xgameruntime.dll") == rt_mod,
          "FreeLibrary on the runtime succeeds and keeps it loaded");

    g_teardown_rt_mod = rt_mod;
    g_teardown_get_proc = get_proc;
    g_teardown_free_lib = free_lib;
}

}  // namespace

int main() {
    std::printf("xgameruntime_test: the offline GDK runtime shim, through the shim registry\n");

    int missing = 0;
    for (const char* n : kNames)
        if (!rt::shims::find("xgameruntime.dll", n)) {
            std::printf("     not registered: %s\n", n);
            ++missing;
        }
    check(missing == 0, "every xgameruntime.dll function in the table resolves through rt::shims::find");
    if (missing) {
        std::printf("xgameruntime_test: %d of %d checks failed\n", g_failures, g_checks);
        return 1;
    }

    auto init = get<InitFn>("XGameRuntimeInitialize");
    auto uninit = get<UninitFn>("XGameRuntimeUninitialize");
    check(init() == S_OK, "XGameRuntimeInitialize returns S_OK");
    check(init() == S_OK, "a second XGameRuntimeInitialize also returns S_OK (reference counted)");

    check(get<DeviceTypeFn>("XSystemGetDeviceType")() == 7, "XSystemGetDeviceType reports XboxScarlettAnaconda (7)");

    auto console_id = get<ConsoleIdFn>("XSystemGetConsoleId");
    char id[64] = {};
    size_t used = 0;
    check(console_id(8, id, &used) == HRESULT_FROM_WIN32(ERROR_INSUFFICIENT_BUFFER),
          "XSystemGetConsoleId rejects a buffer smaller than XSystemConsoleIdBytes");
    check(console_id(sizeof(id), id, &used) == S_OK && used == 39 && std::strlen(id) == 38,
          "XSystemGetConsoleId fills 39 bytes (38 characters and the terminator)");

    auto analytics_info = get<AnalyticsInfoFn>("XSystemGetAnalyticsInfo");
    XSystemAnalyticsInfo info = analytics_info();
    check(info.osVersion.major == 10 && info.osVersion.minor == 0 && info.osVersion.build == 22000 &&
              info.osVersion.revision == 0,
          "XSystemGetAnalyticsInfo returns osVersion 10.0.22000.0");
    check(info.hostingOsVersion.major == 10 && info.hostingOsVersion.minor == 0 &&
              info.hostingOsVersion.build == 22000 && info.hostingOsVersion.revision == 0,
          "XSystemGetAnalyticsInfo returns hostingOsVersion 10.0.22000.0");
    check(std::strcmp(info.family, "Xbox") == 0, "XSystemGetAnalyticsInfo returns family 'Xbox'");
    check(std::strcmp(info.form, "Xbox Series X") == 0, "XSystemGetAnalyticsInfo returns form 'Xbox Series X'");

    auto package_id = get<PackageIdentifierFn>("XPackageGetCurrentProcessPackageIdentifier");
    check(package_id(0, nullptr) == E_INVALIDARG, "XPackageGetCurrentProcessPackageIdentifier rejects null buffer and size 0");
    char pkg_id[64] = {};
    check(package_id(0, pkg_id) == E_INVALIDARG, "XPackageGetCurrentProcessPackageIdentifier rejects size 0 with buffer");
    check(package_id(10, nullptr) == E_INVALIDARG, "XPackageGetCurrentProcessPackageIdentifier rejects null pointer with size");
    check(package_id(16, pkg_id) == HRESULT_FROM_WIN32(ERROR_INSUFFICIENT_BUFFER),
          "XPackageGetCurrentProcessPackageIdentifier rejects insufficient buffer");
    check(package_id(sizeof(pkg_id), pkg_id) == S_OK && std::strlen(pkg_id) == 32,
          "XPackageGetCurrentProcessPackageIdentifier fills 33-byte package identifier");

    SetEnvironmentVariableA("LEONIDA_PACKAGE_ID", "CustomTestPackageId_123456789012");
    check(package_id(sizeof(pkg_id), pkg_id) == S_OK &&
              std::strcmp(pkg_id, "CustomTestPackageId_123456789012") == 0,
          "XPackageGetCurrentProcessPackageIdentifier respects LEONIDA_PACKAGE_ID override");
    SetEnvironmentVariableA("LEONIDA_PACKAGE_ID", nullptr);
    // pkg_id still holds the override; the mount checks below need the game's own identifier again.
    check(package_id(sizeof(pkg_id), pkg_id) == S_OK && std::strlen(pkg_id) == 32,
          "XPackageGetCurrentProcessPackageIdentifier returns the default again once the override is cleared");

    auto mount_pkg = get<PackageMountFn>("XPackageMount");
    auto close_mount = get<PackageCloseMountHandleFn>("XPackageCloseMountHandle");
    auto mount_path_size = get<PackageMountPathSizeFn>("XPackageGetMountPathSize");
    auto mount_path = get<PackageMountPathFn>("XPackageGetMountPath");

    const XPackageMountHandle kInvalidMount{0};
    const XPackageMountHandle kStaleMount{999};

    XPackageMountHandle test_out{0};
    check(mount_pkg(nullptr, nullptr) == E_INVALIDARG, "XPackageMount rejects null identifier and null out pointer");
    check(mount_pkg(pkg_id, nullptr) == E_INVALIDARG, "XPackageMount rejects null out pointer with identifier");
    check(mount_pkg(nullptr, &test_out) == E_INVALIDARG, "XPackageMount rejects null identifier with out pointer");
    check(mount_pkg(pkg_id, &test_out) == E_ACCESSDENIED,
          "XPackageMount rejects game package identifier with E_ACCESSDENIED");
    check(mount_pkg("UninstalledPackage_0000000000000", &test_out) == HRESULT_FROM_WIN32(ERROR_NOT_FOUND),
          "XPackageMount rejects uninstalled package identifier with ERROR_NOT_FOUND");

    size_t req_size = 0;
    check(mount_path_size(kInvalidMount, &req_size) == E_INVALIDARG,
          "XPackageGetMountPathSize rejects invalid mount handle 0");
    check(mount_path_size(kStaleMount, &req_size) == HRESULT_FROM_WIN32(ERROR_NOT_FOUND),
          "XPackageGetMountPathSize rejects unknown mount handle");

    char mnt_path[MAX_PATH] = {};
    check(mount_path(kInvalidMount, 0, nullptr) == E_INVALIDARG,
          "XPackageGetMountPath rejects invalid mount handle 0");
    check(mount_path(kStaleMount, sizeof(mnt_path), mnt_path) == HRESULT_FROM_WIN32(ERROR_NOT_FOUND),
          "XPackageGetMountPath rejects unknown mount handle");

    SetEnvironmentVariableA("LEONIDA_CONTENT_PACKAGES", "DLC1_123456789012345678901234567=C:\\LeonidaDLC1");
    XPackageMountHandle dlc_mount{0};
    check(mount_pkg("DLC1_123456789012345678901234567", &dlc_mount) == S_OK && dlc_mount.handle != 0,
          "XPackageMount mounts content package from LEONIDA_CONTENT_PACKAGES");
    check(mount_path_size(dlc_mount, nullptr) == E_INVALIDARG,
          "XPackageGetMountPathSize rejects null size pointer");
    check(mount_path_size(dlc_mount, &req_size) == S_OK && req_size == std::strlen("C:\\LeonidaDLC1") + 1,
          "XPackageGetMountPathSize returns required buffer size including null terminator for mounted handle");
    check(mount_path(dlc_mount, 0, mnt_path) == E_INVALIDARG,
          "XPackageGetMountPath rejects size 0 with buffer");
    check(mount_path(dlc_mount, 10, nullptr) == E_INVALIDARG,
          "XPackageGetMountPath rejects null pointer with size");
    check(mount_path(dlc_mount, 5, mnt_path) == HRESULT_FROM_WIN32(ERROR_INSUFFICIENT_BUFFER),
          "XPackageGetMountPath rejects insufficient buffer");
    check(mount_path(dlc_mount, sizeof(mnt_path), mnt_path) == S_OK &&
              std::strcmp(mnt_path, "C:\\LeonidaDLC1") == 0,
          "XPackageGetMountPath returns configured content directory for mounted handle");

    close_mount(dlc_mount);
    check(mount_path_size(dlc_mount, &req_size) == HRESULT_FROM_WIN32(ERROR_NOT_FOUND),
          "XPackageGetMountPathSize rejects closed/stale mount handle");
    check(mount_path(dlc_mount, sizeof(mnt_path), mnt_path) == HRESULT_FROM_WIN32(ERROR_NOT_FOUND),
          "XPackageGetMountPath rejects closed/stale mount handle");
    SetEnvironmentVariableA("LEONIDA_CONTENT_PACKAGES", nullptr);

    auto is_packaged = get<PackagedProcessFn>("XPackageIsPackagedProcess");
    check(is_packaged() == true, "XPackageIsPackagedProcess returns true by default");
    SetEnvironmentVariableA("LEONIDA_PACKAGED", "0");
    check(is_packaged() == false, "XPackageIsPackagedProcess respects LEONIDA_PACKAGED=0 override");
    SetEnvironmentVariableA("LEONIDA_PACKAGED", nullptr);

    // Work on the thread pool, completions dispatched by the "game loop" on this thread.
    auto create = get<QueueCreateFn>("XTaskQueueCreate");
    auto close = get<QueueCloseFn>("XTaskQueueCloseHandle");
    auto dispatch = get<QueueDispatchFn>("XTaskQueueDispatch");
    auto run = get<AsyncRunFn>("XAsyncRun");
    auto status = get<AsyncStatusFn>("XAsyncGetStatus");
    XTaskQueueHandle q = nullptr;
    check(create(XTaskQueueDispatchMode::ThreadPool, XTaskQueueDispatchMode::Manual, &q) == S_OK && q,
          "XTaskQueueCreate(ThreadPool work, Manual completion)");
    int result = 0;
    XAsyncBlock block{};
    block.queue = q;
    block.context = &result;
    block.callback = completed;
    check(run(&block, work) == S_OK, "XAsyncRun accepts work on that queue");
    for (int i = 0; i < 500 && g_completions == 0; ++i) {
        while (dispatch(q, XTaskQueuePort::Completion, 0)) {}
        if (!g_completions) Sleep(2);
    }
    check(g_work_runs == 1 && result == 42, "the work callback ran once and wrote its result");
    check(g_work_thread != GetCurrentThreadId(), "the work callback ran on a thread-pool thread");
    check(g_completions == 1 && g_completion_thread == GetCurrentThreadId(),
          "the completion callback ran once, on the thread that dispatched the completion port");
    check(status(&block, false) == S_OK, "XAsyncGetStatus reports S_OK after completion");
    close(q);

    // The process task queue, used when XAsyncBlock::queue is null.
    XTaskQueueHandle pq = nullptr;
    check(get<QueueProcessFn>("XTaskQueueGetCurrentProcessTaskQueue")(&pq) && pq,
          "XTaskQueueGetCurrentProcessTaskQueue returns a queue");
    if (pq) close(pq);
    int result2 = 0;
    XAsyncBlock block2{};
    block2.context = &result2;
    check(run(&block2, work) == S_OK && status(&block2, true) == S_OK && result2 == 42,
          "XAsyncRun with a null queue completes on the process queue");

    // A custom provider: XAsyncBegin, XAsyncSchedule, XAsyncComplete, XAsyncGetResultSize, XAsyncGetResult.
    ProviderState st{get<AsyncCompleteFn>("XAsyncComplete")};
    XAsyncBlock block3{};
    static const int kIdentity = 0;
    HRESULT hr = get<AsyncBeginFn>("XAsyncBegin")(&block3, &st, &kIdentity, "xgameruntime_test", provider);
    check(hr == S_OK, "XAsyncBegin starts a provider");
    check(get<AsyncScheduleFn>("XAsyncSchedule")(&block3, 0) == S_OK, "XAsyncSchedule queues its work");
    check(status(&block3, true) == S_OK, "the provider completes with S_OK");
    size_t size = 0;
    check(get<AsyncResultSizeFn>("XAsyncGetResultSize")(&block3, &size) == S_OK && size == sizeof(uint32_t),
          "XAsyncGetResultSize reports the 4-byte result");
    uint32_t value = 0;
    size_t got = 0;
    check(get<AsyncResultFn>("XAsyncGetResult")(&block3, &kIdentity, sizeof(value), &value, &got) == S_OK &&
              value == 0x1EA0 && got == sizeof(value),
          "XAsyncGetResult returns the provider's result");
    // Cleanup comes when the async state's last reference goes, which can be on the work thread just after the
    // result is read, so allow it a moment.
    const int cleanup_bit = 1 << static_cast<int>(XAsyncOp::Cleanup);
    for (int i = 0; i < 500 && !(st.ops & cleanup_bit); ++i) Sleep(2);
    check((st.ops & (1 << static_cast<int>(XAsyncOp::Begin))) && (st.ops & (1 << static_cast<int>(XAsyncOp::DoWork))) &&
              (st.ops & (1 << static_cast<int>(XAsyncOp::GetResult))) && (st.ops & (1 << static_cast<int>(XAsyncOp::Cleanup))),
          "the provider saw Begin, DoWork, GetResult and Cleanup");

    uninit();
    uninit();
    check(init() == S_OK, "XGameRuntimeInitialize works again after balanced uninitialize calls");
    uninit();

    std::atexit(check_late_teardown);
    check_run_time_load();

    if (g_failures) {
        std::printf("xgameruntime_test: %d of %d checks failed\n", g_failures, g_checks);
        return 1;
    }
    std::printf("xgameruntime_test: OK (%d checks)\n", g_checks);
    return 0;
}
