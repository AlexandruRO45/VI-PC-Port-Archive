// Only the production PE loader/process shim, deliberately independent of GPU/game dependencies.
#include <windows.h>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include "../src/pe_loader.h"
#include "../shims/process_shim.h"

bool early_detach_control = false;
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
        rt::process::finish_run(code);
    } catch (const std::exception& error) {
        std::fprintf(stderr, "process detach driver: %s\n", error.what());
        return 1;
    }
}
