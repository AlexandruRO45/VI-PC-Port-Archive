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

#include "pe_loader.h"
#include "shims.h"

namespace {

int g_failures = 0;

void check(bool ok, const std::string& what) {
    std::printf("%s %s\n", ok ? "PASS" : "FAIL", what.c_str());
    std::fflush(stdout);
    if (!ok) ++g_failures;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc > 2) {
        std::fprintf(stderr, "usage: dll_thread_calls_test <dll_thread_calls.dll>\n");
        return 2;
    }
    const std::filesystem::path dll = argc == 2 ? argv[1] : "fixtures/RelWithDebInfo/dll_thread_calls.dll";
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

        bool exit_shimmed = false;
        for (const auto& r : m.imports)
            if (r.name == "FreeLibraryAndExitThread") exit_shimmed = r.binding == rt::Binding::Shim;
        check(exit_shimmed, "FreeLibraryAndExitThread import binds to the shim, not the host");
        if (exit_shimmed) {
            const auto worker = reinterpret_cast<LPTHREAD_START_ROUTINE>(m.exports.at("dll_exit_thread"));
            auto run = [&](HMODULE module, const char* label) {
                HANDLE thread = CreateThread(nullptr, 0, worker, module, 0, nullptr);
                DWORD code = 0;
                const DWORD wait = thread ? WaitForSingleObject(thread, 10000) : WAIT_FAILED;
                if (thread && wait != WAIT_OBJECT_0) {
                    check(false, label);
                    std::fflush(nullptr);
                    ExitProcess(1);  // Do not destroy the loader while its worker may still run.
                }
                bool ok = thread && GetExitCodeThread(thread, &code) && code == 73;
                check(ok, label);
                if (thread) CloseHandle(thread);
            };
            run(owned, "pseudo-handle worker exits with code 73");
            run(reinterpret_cast<HMODULE>(m.base), "mapped module worker exits with code 73");
            HMODULE host = LoadLibraryW(dll.c_str());
            HMODULE second = LoadLibraryW(dll.c_str());
            check(host && second == host, "host fixture has two references");
            if (host && second) {
                run(host, "host module worker exits with code 73");
                check(GetModuleHandleW(dll.filename().c_str()) == host,
                      "host fixture remains loaded after the worker exits");
                check(FreeLibrary(second) != FALSE, "final host FreeLibrary succeeds");
                check(GetModuleHandleW(dll.filename().c_str()) == nullptr,
                      "worker released exactly one host module reference");
            }
        }
    } catch (const std::exception& e) {
        std::printf("FAIL unexpected exception: %s\n", e.what());
        ++g_failures;
    }
    std::printf("dll_thread_calls_test: %s (%d failures)\n", g_failures ? "FAILED" : "OK", g_failures);
    return g_failures ? 1 : 0;
}
