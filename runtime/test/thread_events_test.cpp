// DLL_THREAD_ATTACH / DLL_THREAD_DETACH for a mapped DLL's DllMain (RT-170).
//
// The loader used to send thread events only to the TLS callbacks of a module with static TLS, and to no DllMain at
// all: a DLL without a TLS directory never heard about a thread. Each fixture DLL (test/fixtures/thread_events.cpp)
// records the events it gets. For every variant the test loads the same file twice, with the Windows loader (the
// control) and with rt::Loader, runs a few threads, and requires the two logs to be identical: the same events, in the
// same order.
// It also pins what the Windows loader does (measured on Windows 11 build 26300), so that the expectation is not just
// "the same as the loader under test". Per thread:
//   plain         DllMain: attach, then (when the thread ends) detach;
//   tls           the TLS callback and DllMain both get attach and detach, callback first, on both;
//   disabled      DisableThreadLibraryCalls(hinstance) in DllMain: that DLL hears nothing;
//   disabled_tls  the same call in a DLL that has a TLS directory: Windows returns TRUE but ignores it, and the DLL
//                 gets the same events as `tls`.
// Finally, after the Loader is gone, new threads must not run anything of the unmapped image.
// Needs gta6rt_tls_host.dll next to this EXE (the loader's thread-event hook). No game files.
#include <windows.h>

#include <cstdio>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>

#include "pe_loader.h"

namespace {

using LogFn = int(__cdecl*)(char*, int);
using ResetFn = void(__cdecl*)();

int g_failures = 0;

void check(bool ok, const std::string& what) {
    std::printf("%s %s\n", ok ? "PASS" : "FAIL", what.c_str());
    std::fflush(stdout);
    if (!ok) ++g_failures;
}

void run_threads(int n) {
    for (int i = 0; i < n; ++i) std::thread([] {}).join();
}

std::string read_log(LogFn log) {
    char buf[600] = {};
    log(buf, sizeof buf);
    return buf;
}

// The events the Windows loader sends for `dll` over three short threads.
std::string windows_log(const std::filesystem::path& dll) {
    HMODULE h = LoadLibraryW(dll.c_str());
    if (!h)
        throw std::runtime_error("LoadLibrary failed for " + dll.string() + ", error " +
                                 std::to_string(GetLastError()));
    auto log = reinterpret_cast<LogFn>(GetProcAddress(h, "thread_events_log"));
    auto reset = reinterpret_cast<ResetFn>(GetProcAddress(h, "thread_events_reset"));
    reset();
    run_threads(3);
    const std::string out = read_log(log);
    FreeLibrary(h);
    return out;
}

// The same through rt::Loader.
std::string mapped_log(const std::filesystem::path& dll) {
    rt::Loader loader(dll.parent_path());
    rt::Module& m = loader.load(dll);
    loader.initialise();
    auto log = reinterpret_cast<LogFn>(m.exports.at("thread_events_log"));
    auto reset = reinterpret_cast<ResetFn>(m.exports.at("thread_events_reset"));
    reset();
    run_threads(3);
    return read_log(log);
}

std::string repeat(const std::string& s, int n) {
    std::string out;
    for (int i = 0; i < n; ++i) out += s;
    return out;
}

void variant(const std::filesystem::path& dll, const char* name, const std::string& expected) {
    std::string windows, mapped;
    try {
        windows = windows_log(dll);
    } catch (const std::exception& e) {
        std::printf("     %s\n", e.what());
    }
    try {
        mapped = mapped_log(dll);
    } catch (const std::exception& e) {
        std::printf("     the loader threw: %s\n", e.what());
    }
    std::printf("     %s: windows='%s' mapped='%s'\n", name, windows.c_str(), mapped.c_str());
    check(windows == expected, std::string(name) + ": the Windows loader sends '" + expected + "' over three threads");
    check(mapped == windows, std::string(name) + ": the mapped DLL gets exactly the same events, in the same order");
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 5) {
        std::fprintf(stderr, "usage: thread_events_test <plain.dll> <tls.dll> <disabled.dll> <disabled_tls.dll>\n");
        return 2;
    }
    try {
        variant(argv[1], "plain (no TLS directory)", repeat("DADT", 3));
        variant(argv[2], "tls (TLS callback + DllMain)", repeat("CADACTDT", 3));
        variant(argv[3], "disabled (DisableThreadLibraryCalls, no TLS)", "");
        variant(argv[4], "disabled_tls (DisableThreadLibraryCalls, with TLS)", repeat("CADACTDT", 3));

        // After the Loader is gone, nothing of the unmapped image may be called by a later thread.
        {
            rt::Loader loader(std::filesystem::path(argv[1]).parent_path());
            loader.load(argv[1]);
            loader.initialise();
        }
        run_threads(3);
        check(true, "threads after the Loader is destroyed run fine");
    } catch (const std::exception& e) {
        std::printf("FAIL unexpected exception: %s\n", e.what());
        ++g_failures;
    }
    std::printf("thread_events_test: %s (%d failures)\n", g_failures ? "FAILED" : "OK", g_failures);
    return g_failures ? 1 : 0;
}
