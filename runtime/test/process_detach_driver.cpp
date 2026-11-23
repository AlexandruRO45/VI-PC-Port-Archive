// Only the production PE loader/process shim, deliberately independent of GPU/game dependencies.
#include <windows.h>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include "fixtures/process_detach_event.h"
#include "../src/pe_loader.h"
#include "../shims/process_shim.h"
#include "../shims/shims.h"

bool early_detach_control = false;

namespace rt::shims { std::mutex& bundled_loader_test_mutex(); }
namespace rt::shims::library_loader { std::mutex& lookup_cache_test_mutex(); }
static unsigned WINAPI detach_probe_export() { return 7; }
SHIM_TABLE(process_detach_probe,
    {"process_detach_probe.dll", "detach_probe_export", reinterpret_cast<void*>(&detach_probe_export)},
);

template<class Mutex>
bool hold_lock_until_exit(Mutex& mutex) {
    struct Hold {
        Mutex* mutex;
        HANDLE ready;
        static DWORD WINAPI run(void* context) {
            const auto [mutex, ready] = *static_cast<Hold*>(context);
            std::lock_guard lock(*mutex);
            SetEvent(ready);
            Sleep(INFINITE);
            return 0;
        }
    };
    const HANDLE ready = CreateEventW(nullptr, TRUE, FALSE, nullptr);
    if (!ready) return false;
    Hold hold{&mutex, ready};
    const HANDLE worker = CreateThread(nullptr, 0, Hold::run, &hold, 0, nullptr);
    if (worker && WaitForSingleObject(ready, 5000) != WAIT_OBJECT_0) TerminateProcess(GetCurrentProcess(), 3);
    if (worker) CloseHandle(worker);
    CloseHandle(ready);
    if (worker) detach_event(3, 300, 1);  // worker acquired the actual lock and will retain it through ExitProcess
    return worker != nullptr;
}
void detach_start_guest(void* context) {
    auto& loader = *static_cast<rt::Loader*>(context);
    try { loader.initialise(); }
    catch (...) { loader.detach_initialised_modules(); throw; }
}
int wmain(int argc, wchar_t** argv) {
    SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX);
    if (argc != 2 && argc != 3) return 2;
    early_detach_control = argc == 3 && !std::wcscmp(argv[2], L"--early-detach-control");
    if (argc == 3 && !early_detach_control) return 2;
    try {
        const auto image = std::filesystem::absolute(argv[1]);
        rt::Loader loader(image.parent_path());
        auto& main = loader.load(image);
        loader.enable_process_exit();
        rt::process::set_command_line(image.filename().wstring(), {});
        char scenario[64]{};
        GetEnvironmentVariableA("LEONIDA_DETACH_CASE", scenario, sizeof(scenario));
        if (!std::strcmp(scenario, "map_only")) rt::process::finish_run(23);
        const int code = rt::process::run_entry(main.base + main.entry_rva, 8u << 20, detach_start_guest, &loader,
            [](void* context) {
                auto& loader = *static_cast<rt::Loader*>(context);
                // Negative control: correct thread/TLS, but workers can still execute during callbacks.
                if (early_detach_control) loader.detach_initialised_modules(reinterpret_cast<void*>(1));
                loader.prepare_process_exit();
            });
        // The exit caller has prepared the Loader and is parked. Take each real lookup lock on a worker
        // in that interval and let native ExitProcess stop it while it still owns the lock.
        if ((!std::strcmp(scenario, "lookup_loader") && !hold_lock_until_exit(loader.process_detach_test_mutex())) ||
            (!std::strcmp(scenario, "lookup_registry") && !hold_lock_until_exit(rt::shims::bundled_loader_test_mutex())) ||
            (!std::strcmp(scenario, "lookup_cache") && !hold_lock_until_exit(rt::shims::library_loader::lookup_cache_test_mutex())))
            return 3;
        rt::process::finish_run(code);
    } catch (const std::exception& error) {
        std::fprintf(stderr, "process detach driver: %s\n", error.what());
        return 1;
    }
}
