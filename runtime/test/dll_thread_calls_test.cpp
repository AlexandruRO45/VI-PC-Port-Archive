// DisableThreadLibraryCalls from a mapped image's DllMain (shims/library_loader.cpp).
//
// The real function only accepts a module the Windows loader knows, so for a manually mapped image it returned FALSE
// and a DllMain that returns its result made Loader::initialise() throw "DllMain returned FALSE". The shim answers
// TRUE for a mapped module and for a module the shims own (a pseudo-handle), and leaves every other handle to the
// real function.
//
// Maps dll_thread_calls.dll (test/fixtures/dll_thread_calls.cpp). xmem.cpp is linked only so the shims own a module
// (xmem.dll) to hand out a pseudo-handle for. No game files.
#include <windows.h>

#include <cstdio>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

#include "pe_loader.h"
#include "shims.h"

namespace {

int g_failures = 0;

void check(bool ok, const std::string& what) {
    std::printf("%s %s\n", ok ? "PASS" : "FAIL", what.c_str());
    std::fflush(stdout);
    if (!ok) ++g_failures;
}

int fake_host_named_shim() { return 11; }
int fake_host_ordinal_shim() { return 22; }

struct BundledProbe {
    std::vector<std::string> seen;

    static void* find_loaded(void* ctx, const std::string& dll_lower) {
        static_cast<BundledProbe*>(ctx)->seen.push_back(dll_lower);
        return nullptr;
    }
    static void* load_bundled(void* ctx, const std::string& dll_lower, uint32_t, bool& handled) {
        handled = false;
        static_cast<BundledProbe*>(ctx)->seen.push_back(dll_lower);
        return nullptr;
    }
    static void* get_proc(void*, void*, const std::string&, bool& owns_handle) {
        owns_handle = false;
        return nullptr;
    }
    static bool owns_handle(void*, void*) { return false; }
};

struct BundledGuard {
    BundledProbe* probe;
    rt::shims::BundledModuleOps ops{
        &BundledProbe::find_loaded,
        &BundledProbe::load_bundled,
        &BundledProbe::get_proc,
        &BundledProbe::owns_handle,
    };
    explicit BundledGuard(BundledProbe* p) : probe(p) { rt::shims::add_bundled_loader(probe, &ops); }
    ~BundledGuard() { rt::shims::remove_bundled_loader(probe); }
    BundledGuard(const BundledGuard&) = delete;
    BundledGuard& operator=(const BundledGuard&) = delete;
};

// Inspect addresses only. Loading System32's DLL and querying its exports never calls a Winsock function.
void test_owned_host_exports(const rt::Module& mapped) {
    using GetProcFn = FARPROC(WINAPI*)(HMODULE, LPCSTR);
    using GetModuleFn = HMODULE(WINAPI*)(LPCSTR);
    using FreeLibFn = BOOL(WINAPI*)(HMODULE);
    const auto get_proc = reinterpret_cast<GetProcFn>(rt::shims::find("kernel32.dll", "GetProcAddress"));
    const auto get_module = reinterpret_cast<GetModuleFn>(rt::shims::find("kernel32.dll", "GetModuleHandleA"));
    const auto free_lib = reinterpret_cast<FreeLibFn>(rt::shims::find("kernel32.dll", "FreeLibrary"));
    check(get_proc && get_module && free_lib, "the registry serves the host export lookup controls");
    if (!get_proc || !get_module || !free_lib) return;

    HMODULE host = ::LoadLibraryExW(L"ws2_32.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
    check(host != nullptr, "load the native Winsock module for address comparisons only");
    if (!host) return;
    const FARPROC native_connect = ::GetProcAddress(host, "connect");
    const FARPROC native_socket = ::GetProcAddress(host, "socket");
    const FARPROC native_resolver = ::GetProcAddress(host, "getaddrinfo");
    check(native_connect && native_socket && native_resolver &&
              ::GetProcAddress(host, MAKEINTRESOURCEA(4)) == native_connect &&
              ::GetProcAddress(host, MAKEINTRESOURCEA(23)) == native_socket,
          "native named and ordinal exports exist independently of the shim table");
    check(!rt::shims::owns_dll("ws2_32.dll") && get_proc(host, "socket") == native_socket &&
              get_proc(host, MAKEINTRESOURCEA(23)) == native_socket,
          "an unowned real module preserves native named and ordinal lookup");
    SetLastError(0x12345678);
    const FARPROC native_null = ::GetProcAddress(host, nullptr);
    const DWORD native_null_error = GetLastError();
    std::printf("NATIVE_NULL returned=%d error=%lu\n", native_null != nullptr,
                static_cast<unsigned long>(native_null_error));
    SetLastError(0x12345678);
    const FARPROC unowned_null = get_proc(host, nullptr);
    const DWORD unowned_null_error = GetLastError();
    check(unowned_null == native_null && unowned_null_error == native_null_error,
          "an unowned real module preserves the native null-name result");

    const rt::shims::Entry entries[] = {
        {"ws2_32.dll", "connect", reinterpret_cast<void*>(&fake_host_named_shim)},
        {"ws2_32.dll", "#4", reinterpret_cast<void*>(&fake_host_ordinal_shim)},
    };
    const rt::shims::Table table(entries, 2);
    const HMODULE owned = get_module("ws2_32.dll");
    check(rt::shims::owns_dll("ws2_32.dll") && owned && owned != host,
          "the same DLL has distinct owned pseudo and native host handles");
    for (HMODULE handle : {host, owned}) {
        if (!handle) continue;
        const std::string kind = handle == host ? "real owned handle: " : "pseudo handle: ";
        SetLastError(0x12345678);
        const FARPROC named = get_proc(handle, "connect");
        const DWORD named_error = GetLastError();
        check(named == reinterpret_cast<FARPROC>(&fake_host_named_shim) && named_error == 0x12345678,
              kind + "registered name keeps its shim and last error");
        SetLastError(0x12345678);
        const FARPROC ordinal = get_proc(handle, MAKEINTRESOURCEA(4));
        const DWORD ordinal_error = GetLastError();
        check(ordinal == reinterpret_cast<FARPROC>(&fake_host_ordinal_shim) && ordinal_error == 0x12345678,
              kind + "registered ordinal keeps its shim and last error");
        struct MissingExport { LPCSTR name; const char* label; };
        const MissingExport missing[] = {
            {"socket", "unregistered native socket name"},
            {MAKEINTRESOURCEA(23), "unregistered native socket ordinal"},
            {"getaddrinfo", "unregistered native resolver name"},
            {"leonida_missing_export", "absent name"},
            {MAKEINTRESOURCEA(65535), "absent ordinal"},
            {nullptr, "null name"},
        };
        for (const auto& item : missing) {
            SetLastError(0x12345678);
            const FARPROC result = get_proc(handle, item.name);
            const DWORD error = GetLastError();
            std::printf("EXPORT_LOOKUP %s%s returned=%d error=%lu\n", kind.c_str(), item.label,
                        result != nullptr, static_cast<unsigned long>(error));
            check(!result && error == ERROR_PROC_NOT_FOUND, kind + item.label + " is refused");
        }
    }

    // The mapped copy is still a bundled module, even though the earlier test owns its DLL name in the registry.
    const HMODULE bundled = reinterpret_cast<HMODULE>(mapped.base);
    check(get_proc(bundled, "dll_thread_calls_bogus_handle") ==
              reinterpret_cast<FARPROC>(mapped.exports.at("dll_thread_calls_bogus_handle")) &&
              get_proc(bundled, MAKEINTRESOURCEA(1)) == reinterpret_cast<FARPROC>(mapped.exports.at("#1")),
          "bundled named and ordinal exports retain precedence over host ownership");
    const LPCSTR bundled_missing[] = {"leonida_missing_export", MAKEINTRESOURCEA(65535), nullptr};
    for (LPCSTR name : bundled_missing) {
        SetLastError(0x12345678);
        const FARPROC result = get_proc(bundled, name);
        const DWORD error = GetLastError();
        check(!result && error == ERROR_PROC_NOT_FOUND, "bundled missing name, ordinal or null name is refused");
    }
    HMODULE kernel = ::GetModuleHandleW(L"kernel32.dll");
    check(!rt::shims::owns_dll("kernel32.dll") &&
              get_proc(kernel, "GetCurrentProcessId") == ::GetProcAddress(kernel, "GetCurrentProcessId"),
          "a host-preferred module still resolves an unregistered native export");
    check(free_lib(host) != FALSE, "release the native Winsock module reference");
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: dll_thread_calls_test <dll_thread_calls.dll>\n");
        return 2;
    }
    const std::filesystem::path dll = argv[1];
    try {
        rt::Loader loader(dll.parent_path());
        rt::Module& m = loader.load(dll);

        bool shimmed = false;
        for (const auto& r : m.imports)
            if (r.name == "DisableThreadLibraryCalls") shimmed = r.binding == rt::Binding::Shim;
        check(shimmed, "the DisableThreadLibraryCalls import is bound to the shim");

        bool initialised = true;
        try {
            loader.initialise();
        } catch (const std::exception& e) {
            initialised = false;
            std::printf("     initialise() threw: %s\n", e.what());
        }
        check(initialised, "initialise() succeeds when DllMain returns DisableThreadLibraryCalls(hinstance)");

        const auto bogus =
            reinterpret_cast<BOOL(__cdecl*)()>(m.exports.at("dll_thread_calls_bogus_handle"));
        SetLastError(0);
        check(bogus() == FALSE, "a handle that names no module still gets the real answer (FALSE)");

        // A shim pseudo-handle is no module the Windows loader knows, so the real function would fail on it.
        using GetModuleHandleAFn = HMODULE(WINAPI*)(LPCSTR);
        using DisableFn = BOOL(WINAPI*)(HMODULE);
        const auto get_module =
            reinterpret_cast<GetModuleHandleAFn>(rt::shims::find("kernel32.dll", "GetModuleHandleA"));
        const auto disable =
            reinterpret_cast<DisableFn>(rt::shims::find("kernel32.dll", "DisableThreadLibraryCalls"));
        check(get_module != nullptr && disable != nullptr,
              "the registry serves GetModuleHandleA and DisableThreadLibraryCalls");
        const HMODULE owned = get_module ? get_module("xmem.dll") : nullptr;
        check(owned != nullptr, "the shims hand out a pseudo-handle for the module they own (xmem.dll)");
        check(disable != nullptr && owned != nullptr && disable(owned) == TRUE,
              "DisableThreadLibraryCalls on a shim pseudo-handle answers TRUE");

        {
            using GetModAFn = HMODULE(WINAPI*)(LPCSTR);
            using GetModWFn = HMODULE(WINAPI*)(LPCWSTR);
            using LoadAFn = HMODULE(WINAPI*)(LPCSTR);
            using LoadWFn = HMODULE(WINAPI*)(LPCWSTR);
            using LoadExAFn = HMODULE(WINAPI*)(LPCSTR, HANDLE, DWORD);
            using LoadExWFn = HMODULE(WINAPI*)(LPCWSTR, HANDLE, DWORD);
            const auto get_mod_a =
                reinterpret_cast<GetModAFn>(rt::shims::find("kernel32.dll", "GetModuleHandleA"));
            const auto get_mod_w =
                reinterpret_cast<GetModWFn>(rt::shims::find("kernel32.dll", "GetModuleHandleW"));
            const auto load_a = reinterpret_cast<LoadAFn>(rt::shims::find("kernel32.dll", "LoadLibraryA"));
            const auto load_w = reinterpret_cast<LoadWFn>(rt::shims::find("kernel32.dll", "LoadLibraryW"));
            const auto load_ex_a =
                reinterpret_cast<LoadExAFn>(rt::shims::find("kernel32.dll", "LoadLibraryExA"));
            const auto load_ex_w =
                reinterpret_cast<LoadExWFn>(rt::shims::find("kernel32.dll", "LoadLibraryExW"));
            check(get_mod_a && get_mod_w && load_a && load_w && load_ex_a && load_ex_w,
                  "the registry serves all LoadLibrary and GetModuleHandle variants");
            if (get_mod_a && get_mod_w && load_a && load_w && load_ex_a && load_ex_w) {
                BundledProbe probe;
                BundledGuard guard(&probe);
                get_mod_a("");
                get_mod_w(L"");
                get_mod_a("C:\\some\\dir\\");
                get_mod_w(L"C:/some/dir/");
                get_mod_a("C:");
                load_a("");
                load_w(L"");
                load_a("C:\\some\\dir\\");
                load_w(L"C:/some/dir/");
                load_ex_a("C:\\some\\dir\\", nullptr, 0);
                load_ex_w(L"C:/some/dir/", nullptr, 0);
                check(probe.seen.empty(),
                      "empty and trailing-slash paths do not synthesize '.dll' for bundled lookup");
                check(get_mod_a("C:xmem.dll") == owned && get_mod_w(L"C:XMEM") == owned,
                      "drive-relative paths resolve owned modules");
            }
        }

        {
            const rt::shims::Entry host_entries[] = {
                {"dll_thread_calls.dll", "test_host_cache_eviction", reinterpret_cast<void*>(&fake_host_named_shim)},
                {"dll_thread_calls.dll", "#1", reinterpret_cast<void*>(&fake_host_ordinal_shim)},
            };
            const rt::shims::Table host_table(host_entries, 2);
            using GetProcFn = FARPROC(WINAPI*)(HMODULE, LPCSTR);
            using FreeLibFn = BOOL(WINAPI*)(HMODULE);
            const auto get_proc =
                reinterpret_cast<GetProcFn>(rt::shims::find("kernel32.dll", "GetProcAddress"));
            const auto free_lib =
                reinterpret_cast<FreeLibFn>(rt::shims::find("kernel32.dll", "FreeLibrary"));
            check(get_proc != nullptr && free_lib != nullptr,
                  "the registry serves GetProcAddress and FreeLibrary");

            HMODULE host_mod = ::LoadLibraryW(dll.c_str());
            check(host_mod != nullptr, "host LoadLibraryW loads dll_thread_calls.dll");
            if (host_mod && get_proc && free_lib) {
                check(get_proc(host_mod, "test_host_cache_eviction") ==
                          reinterpret_cast<FARPROC>(&fake_host_named_shim),
                      "GetProcAddress resolves named shim on live host module");
                check(get_proc(host_mod, MAKEINTRESOURCEA(1)) ==
                          reinterpret_cast<FARPROC>(&fake_host_ordinal_shim),
                      "GetProcAddress resolves ordinal shim on live host module");
                check(free_lib(host_mod) == TRUE, "FreeLibrary unloads host module");
                check(::GetModuleHandleW(dll.c_str()) == nullptr,
                      "host module is actually unloaded after FreeLibrary");
                check(get_proc(host_mod, "test_host_cache_eviction") == nullptr,
                      "GetProcAddress does not return stale cached shim after FreeLibrary");
                check(get_proc(host_mod, MAKEINTRESOURCEA(1)) == nullptr,
                      "GetProcAddress does not return stale ordinal shim after FreeLibrary");
            }
        }
        test_owned_host_exports(m);
    } catch (const std::exception& e) {
        std::printf("FAIL unexpected exception: %s\n", e.what());
        ++g_failures;
    }
    std::printf("dll_thread_calls_test: %s (%d failures)\n", g_failures ? "FAILED" : "OK", g_failures);
    return g_failures ? 1 : 0;
}
