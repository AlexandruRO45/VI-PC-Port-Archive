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

bool failed_retry(const std::filesystem::path& dir) {
    rt::Loader loader(dir);
    auto& root = loader.load(dir / "dll_attach_failure_root.dll");
    const auto original = loader.load(dir / "dll_attach_failure_c.dll").base;
    // Root's nested load of B reaches the already mapped C before startup would attach C.
    SetEnvironmentVariableW(L"LEONIDA_TEST_ATTACH_RETRY", nullptr);
    loader.initialise();
    using FailureChecked = int(WINAPI*)();
    MEMORY_BASIC_INFORMATION info{};
    const bool unloaded = reinterpret_cast<FailureChecked>(root.exports.at("failure_checked"))() &&
                          !loader.find_loaded_module("dll_attach_failure_c.dll") &&
                          !loader.owns_module_handle(original) &&
                          VirtualQuery(original, &info, sizeof info) && info.State == MEM_FREE;
    if (!SetEnvironmentVariableW(L"LEONIDA_TEST_ATTACH_RETRY", L"1")) return false;
    bool handled = false;
    const auto retry = loader.load_runtime_module("dll_attach_failure_c.dll", 0, handled);
    SetEnvironmentVariableW(L"LEONIDA_TEST_ATTACH_RETRY", nullptr);
    const bool ok = unloaded && handled && retry && retry == loader.find_loaded_module("dll_attach_failure_c.dll");
    std::printf("%s: failed pre-mapped DLL was unloaded and a fresh load succeeded\n", ok ? "PASS" : "FAIL");
    return ok;
}

bool nested_detach(const std::filesystem::path& dir) {
    int log = 0;
    {
        rt::Loader loader(dir);
        auto& a = loader.load(dir / "dll_attach_a.dll");
        loader.initialise();
        // B completes attach within A's attach, so A must detach before B.
        using SetDetachLog = void(WINAPI*)(int*);
        bool owns_b = false;
        auto set_b = reinterpret_cast<SetDetachLog>(loader.resolve_module_export(
            loader.find_loaded_module("dll_attach_b.dll"), "set_detach_log", owns_b));
        if (!set_b) return false;
        reinterpret_cast<SetDetachLog>(a.exports.at("set_detach_log"))(&log);
        set_b(&log);
    }
    std::printf("%s: nested detach order %d (expected 12)\n", log == 12 ? "PASS" : "FAIL", log);
    return log == 12;
}
}

int main() {
    try {
        const auto dir = std::filesystem::current_path();
        const bool mutual_ok = mutual_load(dir);
        const bool cycle_ok = static_cycle(dir);
        const bool failure_ok = failed_dependency(dir);
        const bool retry_ok = failed_retry(dir);
        const bool detach_ok = nested_detach(dir);
        return mutual_ok && cycle_ok && failure_ok && retry_ok && detach_ok ? 0 : 9;
    } catch (const std::exception& e) {
        std::printf("FAIL: %s\n", e.what());
        return 1;
    }
}
