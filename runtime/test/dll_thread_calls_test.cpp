// DisableThreadLibraryCalls from a mapped image's DllMain (shims/library_loader.cpp).
//
// The real function only accepts a module the Windows loader knows, so for a manually mapped image it returned FALSE and
// a DllMain that returns its result made Loader::initialise() throw "DllMain returned FALSE". The shim answers TRUE for
// a mapped module and leaves every other handle to the real function.
//
// Maps dll_thread_calls.dll (test/fixtures/dll_thread_calls.cpp). No game files.
#include <windows.h>

#include <cstdio>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <thread>

#include "pe_loader.h"

namespace {

int g_failures = 0;

void check(bool ok, const std::string& what) {
    std::printf("%s %s\n", ok ? "PASS" : "FAIL", what.c_str());
    std::fflush(stdout);
    if (!ok) ++g_failures;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: dll_thread_calls_test <opt-out.dll> <enabled.dll>\n");
        return 2;
    }
    const std::filesystem::path dll = argv[1];
    try {
        rt::Loader loader(dll.parent_path());
        rt::Module& opted_out = loader.load(dll);
        rt::Module& enabled = loader.load(argv[2]);

        bool shimmed = false;
        for (const auto& r : opted_out.imports)
            if (r.name == "DisableThreadLibraryCalls") shimmed = r.binding == rt::Binding::Shim;
        check(shimmed, "the DisableThreadLibraryCalls import is bound to the shim");
        using Count = LONG(__cdecl*)(DWORD);
        const auto count = reinterpret_cast<Count>(enabled.exports.at("dll_thread_calls_count"));
        const auto opted_out_count = reinterpret_cast<Count>(opted_out.exports.at("dll_thread_calls_count"));
        check(enabled.tls_index == rt::kNoTlsIndex && opted_out.tls_index == rt::kNoTlsIndex,
              "neither DLL has static TLS");
        std::thread([] {}).join();
        check(count(DLL_THREAD_ATTACH) == 0 && count(DLL_THREAD_DETACH) == 0,
              "map-only DLL receives no thread notifications");

        loader.initialise();
        check(count(DLL_PROCESS_ATTACH) == 1 && opted_out_count(DLL_PROCESS_ATTACH) == 1,
              "both DLLs receive process attach, including the DLL that opts out there");
        bool started_after_attach = false;
        std::thread([&] { started_after_attach = count(DLL_THREAD_ATTACH) == 1; }).join();
        check(started_after_attach, "DllMain receives thread attach before the worker body");
        check(count(DLL_THREAD_DETACH) == 1, "DllMain receives thread detach before join returns");
        check(opted_out_count(DLL_THREAD_ATTACH) == 0 && opted_out_count(DLL_THREAD_DETACH) == 0,
              "process-attach opt-out prevents both thread notifications");

        reinterpret_cast<void(__cdecl*)()>(enabled.exports.at("dll_thread_calls_disable_on_attach"))();
        std::thread([] {}).join();
        const auto did_disable = reinterpret_cast<BOOL(__cdecl*)()>(
            enabled.exports.at("dll_thread_calls_did_disable"));
        check(did_disable() == TRUE, "DisableThreadLibraryCalls also works inside thread attach");
        std::thread([] {}).join();
        check(count(DLL_THREAD_ATTACH) == 2 && count(DLL_THREAD_DETACH) == 1,
              "a thread-attach opt-out suppresses the current detach and future notifications");

        const auto bogus =
            reinterpret_cast<BOOL(__cdecl*)()>(opted_out.exports.at("dll_thread_calls_bogus_handle"));
        check(bogus() == FALSE, "a handle that names no module still gets the real answer (FALSE)");
    } catch (const std::exception& e) {
        std::printf("FAIL unexpected exception: %s\n", e.what());
        ++g_failures;
    }
    std::printf("dll_thread_calls_test: %s (%d failures)\n", g_failures ? "FAILED" : "OK", g_failures);
    return g_failures ? 1 : 0;
}
