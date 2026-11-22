// gta6rt: native runtime for the Xbox Series build.
//
//   gta6rt --dry-run <image> [--game-dir DIR] [--overlay DIR]
//       Maps the image and its bundled modules, applies relocations, binds every import, installs static TLS for
//       this thread and prints a report. No code from the mapped images
//       runs. TLS needs the runtime's own gta6rt_tls_host.dll, which reserves the module TLS index; system DLLs are
//       loaded to resolve host imports, as before.
//
//   gta6rt --init <image> [--game-dir DIR] [--overlay DIR]
//       The same mapping, then each loaded DLL's TLS callbacks and DllMain(DLL_PROCESS_ATTACH) are run.
//
//   gta6rt --run <exe> [--game-dir DIR] [--overlay DIR] [--trace-d3d12] [-- game arguments...]
//       The same, then the EXE's own TLS callbacks and its entry point, on a game thread. The game's argv and
//       command line are <exe name> plus the arguments after "--"; exit/ExitProcess from the game end the run
//       (shims/process.cpp). gta6rt prints the binding report first and returns the game's exit code.
//
//   gta6rt --splash <game-dir> [--seconds N]
//       Opens the title's window the way the console shell does before the first frame: the SplashScreenImage of
//       <game-dir>/MicrosoftGame.Config, centred on its BackgroundColor, titled after the title's display name. It
//       stays until closed (Esc or the close button) or N seconds pass. Reads the config and the image only; no game
//       code runs.
//
// --game-dir is also where a title's GAME: paths read from; anything it writes or deletes goes to the overlay
// (--overlay, default <game-dir>/overlay) so the package is never modified (shims/file_system.cpp).
#include <windows.h>

#include <algorithm>
#include <cstdio>
#include <cwchar>
#include <map>
#include <string>
#include <vector>

#include "base_reservation.h"
#include "pe_loader.h"
#include "shell_visuals.h"
#include "../shims/d3d12_trace.h"
#include "../shims/file_system.h"
#include "../shims/process_shim.h"
#include "../shims/shims.h"

namespace {

const char* binding_name(rt::Binding b) {
    switch (b) {
        case rt::Binding::Shim: return "shim";
        case rt::Binding::Host: return "host";
        case rt::Binding::Module: return "module";
        default: return "UNIMPLEMENTED";
    }
}

int usage() {
    std::fprintf(stderr,
                 "usage: gta6rt --dry-run <image> [--game-dir DIR] [--overlay DIR]\n"
                 "       gta6rt --init <image> [--game-dir DIR] [--overlay DIR] [--trace-d3d12]\n"
                 "       gta6rt --run <exe> [--game-dir DIR] [--overlay DIR] [--trace-d3d12] [-- game arguments...]\n"
                 "       gta6rt --splash <game-dir> [--seconds N]\n");
    return 2;
}

// --splash: the title's splash window, from its MicrosoftGame.Config (shell_visuals.h).
int splash_mode(int argc, wchar_t** argv) {
    std::filesystem::path dir;
    unsigned long seconds = 0;
    for (int i = 2; i < argc; ++i) {
        const std::wstring a = argv[i];
        if (a == L"--seconds" && i + 1 < argc) seconds = std::wcstoul(argv[++i], nullptr, 10);
        else if (dir.empty()) dir = a;
        else return usage();
    }
    if (dir.empty()) return usage();
    const rt::shell::ShellVisuals visuals = rt::shell::read_shell_visuals(dir);
    if (!visuals.found) {
        std::fprintf(stderr, "error: no <ShellVisuals> in %ls\\MicrosoftGame.Config\n", dir.c_str());
        return 1;
    }
    std::printf("%ls: splash %ls, background #%06x\n", visuals.display_name.c_str(),
                visuals.splash.empty() ? L"(none)" : visuals.splash.c_str(), static_cast<unsigned>(visuals.background));
    std::fflush(stdout);
    return rt::shell::show_splash(visuals, static_cast<uint32_t>(std::min<unsigned long>(seconds, 86400) * 1000));
}

}  // namespace

int wmain(int argc, wchar_t** argv) {
    if (argc >= 2 && std::wstring(argv[1]) == L"--splash") return splash_mode(argc, argv);
    std::filesystem::path image, game_dir, overlay_dir;
    bool dry_run = false, init = false, run = false, trace_d3d12 = false;
    std::vector<std::wstring> game_args;  // wide: path::string() throws on a character the code page lacks (RT-84)
    for (int i = 1; i < argc; ++i) {
        std::wstring a = argv[i];
        if (a == L"--") {
            for (++i; i < argc; ++i) game_args.push_back(argv[i]);
            break;
        }
        if (a == L"--dry-run") dry_run = true;
        else if (a == L"--init") init = true;
        else if (a == L"--run") run = true;
        else if (a == L"--trace-d3d12") trace_d3d12 = true;
        else if (a == L"--game-dir" && i + 1 < argc) game_dir = argv[++i];
        else if (a == L"--overlay" && i + 1 < argc) overlay_dir = argv[++i];
        else if (image.empty()) image = a;
        else return usage();
    }
    if (image.empty()) return usage();
    // Exactly one mode: --dry-run --run would otherwise initialise the modules and call the entry point.
    if (dry_run + init + run != 1) return usage();
    if (game_dir.empty()) game_dir = image.parent_path();
    // An image without relocations whose range this process already uses (arm64 Wine on macOS) runs in a new
    // process that has the range reserved from the start (base_reservation.h); that process does not do this again.
    if (!rt::base_reservation::inherited())
        if (const auto range = rt::base_reservation::needed(image)) return rt::base_reservation::relaunch(*range);
    // GAME: paths read under --game-dir; whatever a title writes lands in the overlay, never in the package.
    if (overlay_dir.empty()) overlay_dir = game_dir / "overlay";
    rt::fs::configure(game_dir, overlay_dir);
    rt::d3d12_trace::set_enabled(trace_d3d12);

    try {
        rt::Loader loader(game_dir);
        rt::Module& main_module = loader.load(image);  // maps, binds and installs TLS; runs no image code
        if (init || run) {
            loader.initialise();  // TLS callbacks, then DllMain(DLL_PROCESS_ATTACH) for DLLs
            std::printf("%zu module(s) initialised\n", loader.modules().size());
        }
        int unimplemented = 0, unimplemented_hresult = 0;
        for (auto* m : loader.modules()) {
            std::map<rt::Binding, int> counts;
            for (auto& r : m->imports) counts[r.binding]++;
            std::printf("%-36s base=%p%s size=%u entry=+%#x  shim=%d host=%d module=%d unimplemented=%d\n",
                        m->name.c_str(), m->base, m->relocated ? " (relocated)" : "", m->size, m->entry_rva,
                        counts[rt::Binding::Shim], counts[rt::Binding::Host], counts[rt::Binding::Module],
                        counts[rt::Binding::Unimplemented]);
            for (auto& r : m->imports)
                if (r.binding == rt::Binding::Unimplemented || r.binding == rt::Binding::Shim) {
                    const bool missing = r.binding == rt::Binding::Unimplemented;
                    const bool hresult =
                        missing && rt::shims::classify_import(r.dll, r.name) == rt::shims::ReturnClass::HResult;
                    std::printf("    %-13s %s!%s%s\n", binding_name(r.binding), r.dll.c_str(), r.name.c_str(),
                                hresult ? "  [returns HRESULT]" : "");
                    unimplemented += missing;
                    unimplemented_hresult += hresult;
                }
        }
        std::printf("%zu module(s) mapped, %d unimplemented import(s)\n", loader.modules().size(), unimplemented);
        std::printf("unimplemented by return policy: hresult=%d other=%d\n", unimplemented_hresult,
                    unimplemented - unimplemented_hresult);
        if (!run) return 0;
        if (!main_module.entry_rva) {
            std::fprintf(stderr, "error: %s has no entry point\n", main_module.name.c_str());
            return 1;
        }
        rt::process::set_command_line(image.filename().wstring(), game_args);
        std::printf("console paths: GAME: -> %ls, writes -> %ls\n", game_dir.c_str(), overlay_dir.c_str());
        std::printf("running %s\n", main_module.name.c_str());
        std::fflush(stdout);
        const int code = rt::process::run_entry(main_module.base + main_module.entry_rva);
        std::fflush(stdout);
        std::printf("game exited with code %d\n", code);
        rt::shims::report_called_unimplemented();
        std::fflush(stdout);
        std::fflush(stderr);
        // Not `return`: a normal host exit runs the CRT's atexit table, which the game shares with gta6rt (its
        // handlers would run against a game that has stopped) and whose last word decides the exit status.
        ExitProcess(static_cast<UINT>(code));
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
