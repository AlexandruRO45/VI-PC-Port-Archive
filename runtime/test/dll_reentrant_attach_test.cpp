#include <windows.h>

#include <cstdio>
#include <exception>
#include <filesystem>

#include "pe_loader.h"

namespace {
bool mutual_load(const std::filesystem::path& dir) {
    rt::Loader loader(dir);
    auto& importer = loader.load(dir / "dll_attach_importer.dll");
    auto& a = loader.load(dir / "dll_attach_a.dll");
    loader.initialise();

    auto* b = loader.find_loaded_module("dll_attach_b.dll");
    if (!b) {
        std::puts("FAIL: DllMain did not load mapped DLL B");
        return false;
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
        return false;
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
    return ok && in_order;
}

bool static_cycle(const std::filesystem::path& dir) {
    rt::Loader loader(dir);
    auto& a = loader.load(dir / "dll_attach_cycle_a.dll");
    auto& b = loader.load(dir / "dll_attach_cycle_b.dll");
    loader.initialise();
    using ObservedAttach = BOOL(WINAPI*)();
    const bool ok = reinterpret_cast<ObservedAttach>(a.exports.at("observed_peer_attach"))() &&
                    !reinterpret_cast<ObservedAttach>(b.exports.at("observed_peer_attach"))();
    std::printf("%s: static cycle attached B before A\n", ok ? "PASS" : "FAIL");
    return ok;
}

bool failed_dependency(const std::filesystem::path& dir) {
    rt::Loader loader(dir);
    auto& root = loader.load(dir / "dll_attach_failure_root.dll");
    loader.initialise();
    using FailureChecked = int(WINAPI*)();
    const bool ok = reinterpret_cast<FailureChecked>(root.exports.at("failure_checked"))() &&
                    !loader.find_loaded_module("dll_attach_failure_b.dll");
    std::printf("%s: nested load failed and startup never attached B\n", ok ? "PASS" : "FAIL");
    return ok;
}
}

int main() {
    try {
        const auto dir = std::filesystem::current_path();
        const bool mutual_ok = mutual_load(dir);
        const bool cycle_ok = static_cycle(dir);
        const bool failure_ok = failed_dependency(dir);
        return mutual_ok && cycle_ok && failure_ok ? 0 : 9;
    } catch (const std::exception& e) {
        std::printf("FAIL: %s\n", e.what());
        return 1;
    }
}
