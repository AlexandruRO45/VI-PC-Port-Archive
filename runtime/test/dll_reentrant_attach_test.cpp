#include <windows.h>

#include <cstdio>
#include <exception>
#include <filesystem>

#include "pe_loader.h"

int main() {
    try {
        const auto dir = std::filesystem::current_path();
        rt::Loader loader(dir);
        auto& importer = loader.load(dir / "dll_attach_importer.dll");
        auto& a = loader.load(dir / "dll_attach_a.dll");
        loader.initialise();

        auto* b = loader.find_loaded_module("dll_attach_b.dll");
        if (!b) {
            std::puts("FAIL: DllMain did not load mapped DLL B");
            return 1;
        }
        using Count = int(WINAPI*)();
        using Peer = HMODULE(WINAPI*)();
        bool owns_b = false;
        const auto count_a = reinterpret_cast<Count>(a.exports.at("attach_count"));
        const auto count_b = reinterpret_cast<Count>(loader.resolve_module_export(b, "attach_count", owns_b));
        const auto peer_a = reinterpret_cast<Peer>(a.exports.at("peer_handle"));
        const auto peer_b = reinterpret_cast<Peer>(loader.resolve_module_export(b, "peer_handle", owns_b));
        if (!count_b || !peer_b || peer_a() != b || peer_b() != static_cast<void*>(a.base)) {
            std::puts("FAIL: both LoadLibraryW calls must return the mapped peer handle");
            return 1;
        }
        const int attaches_a = count_a();
        const int attaches_b = count_b();
        const bool ok = attaches_a == 1 && attaches_b == 1;
        std::printf("%s: DLL_PROCESS_ATTACH counts A=%d B=%d (expected A=1 B=1)\n",
                    ok ? "PASS" : "FAIL", attaches_a, attaches_b);
        using DependencyReady = BOOL(WINAPI*)();
        const auto dependency_was_ready =
            reinterpret_cast<DependencyReady>(importer.exports.at("dependency_was_ready"));
        const bool in_order = dependency_was_ready() != FALSE;
        std::printf("%s: static importer observed completed attach of A\n", in_order ? "PASS" : "FAIL");
        return ok && in_order ? 0 : 9;
    } catch (const std::exception& e) {
        std::printf("FAIL: %s\n", e.what());
        return 1;
    }
}
