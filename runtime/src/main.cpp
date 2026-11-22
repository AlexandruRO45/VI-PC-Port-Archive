// gta6rt: native runtime for the Xbox Series build.
//
//   gta6rt --dry-run <image> [--game-dir DIR] [--shims DLL]
//       Maps the image and its bundled modules, applies relocations, binds every import and prints a report.
//       No code from the image runs.
//
//   gta6rt --init <image> [--game-dir DIR] [--shims DLL]
//       The same mapping, then each loaded DLL's TLS callbacks and DllMain(DLL_PROCESS_ATTACH) are run.
//
//   gta6rt --run <image> [--game-dir DIR] [--shims DLL]
//       --init, then the image's entry point is called: the way a game starts.
//
//   gta6rt --shims <DLL>
//       Loads a shim DLL into this process before anything is mapped, so the DLL's shim table (for example the test
//       harness's kernel32/user32/d3d12/dxgi entries) is registered and the game's imports can bind to it. Real
//       targets link their shims into gta6rt itself; this flag is for the PE harness and for experiments.
//
// The unimplemented imports are printed and appended to missing_imports.log next to the working directory, so a
// game that stops on a call names the API that is still missing.
#include <windows.h>

#include <cstdio>
#include <map>
#include <string>

#include "dxgi.h"  // rt::d3d12::link_dxgi_shims
#include "pe_loader.h"
#include "shims.h"  // rt::shims::find: what the process itself provides

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
                 "usage: gta6rt --dry-run <image> [--game-dir DIR] [--shims DLL]\n"
                 "       gta6rt --init <image> [--game-dir DIR] [--shims DLL]\n"
                 "       gta6rt --run <image> [--game-dir DIR] [--shims DLL]\n");
    return 2;
}

}  // namespace

int wmain(int argc, wchar_t** argv) {
#ifdef LEONIDA_VULKAN_DEVICE
    // The DXGI shims (dxgi_shims.cpp, which keeps dxgi.cpp in the link) must be registered even though nothing here
    // calls them directly: the game's `dxgi.dll!CreateDXGIFactory1` import binds to them.
    rt::d3d12::link_dxgi_shims();
#endif

    std::filesystem::path image, game_dir, shim_dll;
    bool dry_run = false, init = false, run = false;
    for (int i = 1; i < argc; ++i) {
        std::wstring a = argv[i];
        if (a == L"--dry-run") dry_run = true;
        else if (a == L"--init") init = true;
        else if (a == L"--run") run = true;
        else if (a == L"--game-dir" && i + 1 < argc) game_dir = argv[++i];
        else if (a == L"--shims" && i + 1 < argc) shim_dll = argv[++i];
        else if (image.empty()) image = a;
        else return usage();
    }
    if (image.empty()) return usage();
    if (!dry_run && !init && !run) {
        std::fprintf(stderr,
                     "Only --dry-run, --init and --run exist so far: running game code needs the shims listed by "
                     "--dry-run first.\n");
        return 2;
    }
    if (game_dir.empty()) game_dir = image.parent_path();

    // A shim DLL registers its table at load, before the loader resolves the image's imports.
    if (!shim_dll.empty()) {
        const HMODULE module = LoadLibraryW(shim_dll.c_str());
        if (!module) {
            std::fprintf(stderr, "error: cannot load shims from %ls (0x%08lx)\n", shim_dll.c_str(), GetLastError());
            return 1;
        }
        std::printf("shims loaded: %ls\n", shim_dll.c_str());
    }

    // The shim tables linked into gta6rt (and any --shims DLL) registered themselves before main. Say so, because a
    // silently dropped table is the difference between a bound import and an unimplemented one.
    std::printf("shim tables linked: dxgi.dll=%s d3d12_xs.dll=%s\n",
                rt::shims::find("dxgi.dll", "CreateDXGIFactory1") ? "yes" : "no",
                rt::shims::find("d3d12_xs.dll", "D3D12XboxCreateDevice") ? "yes" : "no");

    try {
        rt::Loader loader(game_dir);
        loader.load(image);
        if (init || run) {
            loader.initialise();
            std::printf("%zu module(s) initialised\n", loader.modules().size());
        }
        int unimplemented = 0;
        for (auto* m : loader.modules()) {
            std::map<rt::Binding, int> counts;
            for (auto& r : m->imports) counts[r.binding]++;
            std::printf("%-36s base=%p%s size=%u entry=+%#x  shim=%d host=%d module=%d unimplemented=%d\n",
                        m->name.c_str(), m->base, m->relocated ? " (relocated)" : "", m->size, m->entry_rva,
                        counts[rt::Binding::Shim], counts[rt::Binding::Host], counts[rt::Binding::Module],
                        counts[rt::Binding::Unimplemented]);
            for (auto& r : m->imports)
                if (r.binding == rt::Binding::Unimplemented || r.binding == rt::Binding::Shim) {
                    std::printf("    %-13s %s!%s\n", binding_name(r.binding), r.dll.c_str(), r.name.c_str());
                    unimplemented += r.binding == rt::Binding::Unimplemented;
                }
        }
        std::printf("%zu module(s) mapped, %d unimplemented import(s)\n", loader.modules().size(), unimplemented);
        if (run) {
            rt::Module* entry = loader.modules().empty() ? nullptr : loader.modules().front();
            if (!entry || !entry->entry_rva) {
                std::fprintf(stderr, "error: the image has no entry point to run\n");
                return 1;
            }
            std::printf("running %s (+%#x) ...\n", entry->name.c_str(), entry->entry_rva);
            std::fflush(stdout);
            auto* start = reinterpret_cast<void (*)()>(static_cast<std::uint8_t*>(entry->base) + entry->entry_rva);
            start();
            std::printf("the image's entry point returned\n");
        }
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
