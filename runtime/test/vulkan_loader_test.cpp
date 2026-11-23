// Unit tests for runtime/test/vulkan_loader.h: missing overrides, libraries lacking vkGetInstanceProcAddr (and
// unloading them on failure), Windows CWD DLL-hijacking resistance via LOAD_LIBRARY_SEARCH_SYSTEM32, explicit
// LEONIDA_VULKAN_LOADER path overrides, and load_vulkan() idempotence.
#include "vulkan_loader.h"

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <string>
#include <system_error>

namespace {

int failures = 0;

void expect(bool ok, const char* what) {
    if (ok) return;
    std::fprintf(stderr, "FAIL %s\n", what);
    ++failures;
}

#ifdef _WIN32

class CwdGuard {
public:
    CwdGuard() {
        const DWORD needed = GetCurrentDirectoryW(0, nullptr);
        if (needed > 1) {
            original_.resize(needed - 1);
            const DWORD written = GetCurrentDirectoryW(needed, original_.data());
            if (written == 0 || written >= needed) original_.clear();
        }
    }

    CwdGuard(const CwdGuard&) = delete;
    CwdGuard& operator=(const CwdGuard&) = delete;

    ~CwdGuard() { restore(); }

    bool valid() const { return !original_.empty(); }

    bool restore() {
        if (original_.empty()) return true;
        const bool ok = SetCurrentDirectoryW(original_.c_str()) != 0;
        if (ok) original_.clear();
        return ok;
    }

private:
    std::wstring original_;
};

bool to_acp_path(const std::filesystem::path& p, std::string& out) {
    const std::wstring& w = p.native();
    if (w.empty()) return false;
    BOOL used_default = FALSE;
    const int needed =
        WideCharToMultiByte(CP_ACP, WC_NO_BEST_FIT_CHARS, w.c_str(), -1, nullptr, 0, nullptr, &used_default);
    if (needed <= 1 || used_default) return false;
    std::string narrow(static_cast<std::size_t>(needed - 1), '\0');
    const int written = WideCharToMultiByte(
        CP_ACP, WC_NO_BEST_FIT_CHARS, w.c_str(), -1, narrow.data(), needed, nullptr, &used_default);
    if (written != needed || used_default) return false;
    out = std::move(narrow);
    return true;
}

void test_windows_loader_paths() {
    wchar_t sys_buf[MAX_PATH]{};
    const UINT sys_len = GetSystemDirectoryW(sys_buf, MAX_PATH);
    expect(sys_len > 0 && sys_len < MAX_PATH, "GetSystemDirectoryW succeeds");
    if (sys_len == 0 || sys_len >= MAX_PATH) return;

    const std::filesystem::path sys_dir(sys_buf);

    // 1. A real system DLL that exports no vkGetInstanceProcAddr returns nullptr and is freed if newly loaded.
    constexpr const char* kBenignDlls[] = {"cabinet.dll", "xmllite.dll", "version.dll"};
    bool tested_unloaded_system_dll = false;
    for (const char* dll : kBenignDlls) {
        std::error_code ec;
        if (!std::filesystem::exists(sys_dir / dll, ec) || ec) continue;
        const bool already_loaded = GetModuleHandleA(dll) != nullptr;
        expect(load_vulkan_from(dll) == nullptr, "system DLL without vkGetInstanceProcAddr returns nullptr");
        if (!already_loaded) {
            expect(GetModuleHandleA(dll) == nullptr, "failed bare DLL override calls FreeLibrary");
            tested_unloaded_system_dll = true;
            break;
        }
    }

    const std::filesystem::path benign_src = sys_dir / L"version.dll";
    std::error_code ec;
    expect(std::filesystem::exists(benign_src, ec) && !ec, "System32\\version.dll exists");
    if (!std::filesystem::exists(benign_src, ec) || ec) return;

    // 2. CWD hijacking resistance and explicit path overrides inside an isolated temporary directory.
    const std::filesystem::path tmp_dir =
        std::filesystem::temp_directory_path(ec) /
        ("leonida_vulkan_loader_test_" + std::to_string(GetCurrentProcessId()) + "_" +
         std::to_string(GetTickCount64()));
    expect(!ec, "temp_directory_path succeeds");
    if (ec) return;

    std::filesystem::create_directories(tmp_dir, ec);
    expect(!ec, "create temp test directory");
    if (ec) return;

    const std::filesystem::path fake_cwd_vulkan = tmp_dir / L"vulkan-1.dll";
    std::filesystem::copy_file(benign_src, fake_cwd_vulkan, std::filesystem::copy_options::overwrite_existing, ec);
    expect(!ec, "stage non-Vulkan vulkan-1.dll in temp directory");

    CwdGuard cwd;
    expect(cwd.valid(), "capture original working directory");
    if (!ec && cwd.valid() && SetCurrentDirectoryW(tmp_dir.c_str()) != 0) {
        expect(GetModuleHandleW(fake_cwd_vulkan.c_str()) == nullptr, "fake CWD vulkan-1.dll starts unloaded");

        // Explicit relative overrides to the fake DLL must fail and unload it.
        expect(load_vulkan_from("./vulkan-1.dll") == nullptr,
               "explicit ./vulkan-1.dll without vkGetInstanceProcAddr returns nullptr");
        expect(GetModuleHandleW(fake_cwd_vulkan.c_str()) == nullptr,
               "failed ./vulkan-1.dll override unloads module");

        expect(load_vulkan_from(".\\vulkan-1.dll") == nullptr,
               "explicit .\\vulkan-1.dll without vkGetInstanceProcAddr returns nullptr");
        expect(GetModuleHandleW(fake_cwd_vulkan.c_str()) == nullptr,
               "failed .\\vulkan-1.dll override unloads module");
        tested_unloaded_system_dll = true;

        // Default lookup (nullptr or empty override) must search System32 and ignore <cwd>\vulkan-1.dll.
        const std::filesystem::path sys_vulkan = sys_dir / L"vulkan-1.dll";
        const bool has_sys_vulkan = std::filesystem::exists(sys_vulkan, ec) && !ec;

        const PFN_vkGetInstanceProcAddr from_null = load_vulkan_from(nullptr);
        const PFN_vkGetInstanceProcAddr from_empty = load_vulkan_from("");
        expect(GetModuleHandleW(fake_cwd_vulkan.c_str()) == nullptr,
               "default load_vulkan_from ignores vulkan-1.dll in CWD");
        expect(from_null == from_empty, "nullptr and empty override resolve identically");

        if (has_sys_vulkan) {
            expect(from_null != nullptr, "default load_vulkan_from loads System32\\vulkan-1.dll");
            if (from_null != nullptr) {
                HMODULE owner = nullptr;
                const BOOL got_owner = GetModuleHandleExW(
                    GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                    reinterpret_cast<LPCWSTR>(reinterpret_cast<std::uintptr_t>(from_null)), &owner);
                expect(got_owner != 0, "GetModuleHandleExW resolves default vkGetInstanceProcAddr module");
                expect(owner == GetModuleHandleW(sys_vulkan.c_str()),
                       "default vkGetInstanceProcAddr comes from System32\\vulkan-1.dll");
            }

            // Explicit path override to a custom loader copy loads from that path rather than System32.
            const std::filesystem::path custom_loader = tmp_dir / L"custom_vk_loader.dll";
            std::string custom_acp;
            if (from_null != nullptr && to_acp_path(custom_loader, custom_acp)) {
                std::filesystem::copy_file(
                    sys_vulkan, custom_loader, std::filesystem::copy_options::overwrite_existing, ec);
                expect(!ec, "copy System32\\vulkan-1.dll to custom_vk_loader.dll");
                if (!ec) {
                    expect(GetModuleHandleW(custom_loader.c_str()) == nullptr, "custom_vk_loader.dll starts unloaded");
                    const PFN_vkGetInstanceProcAddr custom_fn = load_vulkan_from(custom_acp.c_str());
                    expect(custom_fn != nullptr, "explicit path override loads custom_vk_loader.dll");
                    const HMODULE custom_mod = GetModuleHandleW(custom_loader.c_str());
                    expect(custom_mod != nullptr, "custom_vk_loader.dll is mapped after override load");
                    if (custom_fn != nullptr && custom_mod != nullptr) {
                        HMODULE custom_owner = nullptr;
                        const BOOL got_custom_owner = GetModuleHandleExW(
                            GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                            reinterpret_cast<LPCWSTR>(reinterpret_cast<std::uintptr_t>(custom_fn)), &custom_owner);
                        expect(got_custom_owner != 0, "GetModuleHandleExW resolves custom vkGetInstanceProcAddr");
                        expect(custom_owner == custom_mod,
                               "override vkGetInstanceProcAddr comes from custom_vk_loader.dll");
                        expect(custom_owner != GetModuleHandleW(sys_vulkan.c_str()),
                               "override module is distinct from System32\\vulkan-1.dll");
                    }
                    if (custom_mod != nullptr) {
                        expect(FreeLibrary(custom_mod) != 0, "FreeLibrary unloads custom_vk_loader.dll");
                        expect(GetModuleHandleW(custom_loader.c_str()) == nullptr,
                               "custom_vk_loader.dll is unloaded after FreeLibrary");
                    }
                }
            }
        }
    } else {
        expect(false, "enter temporary working directory");
    }

    expect(tested_unloaded_system_dll, "verified FreeLibrary on at least one non-Vulkan DLL");
    expect(cwd.restore(), "restore original working directory before temp cleanup");

    std::filesystem::remove_all(tmp_dir, ec);
    expect(!ec && !std::filesystem::exists(tmp_dir), "temp directory cleans up with zero locked DLL files");
}

#else

void test_posix_non_vulkan_library() {
#ifdef __APPLE__
    expect(load_vulkan_from("/usr/lib/libSystem.B.dylib") == nullptr,
           "libSystem.B.dylib without vkGetInstanceProcAddr returns nullptr");
    constexpr const char* kCandidates[] = {
        "/usr/lib/libbz2.1.0.dylib",
        "/usr/lib/libz.1.dylib",
        "/usr/lib/libsqlite3.dylib",
    };
#else
    expect(load_vulkan_from("libm.so.6") == nullptr, "libm.so.6 without vkGetInstanceProcAddr returns nullptr");
    constexpr const char* kCandidates[] = {
        "libz.so.1",
        "libutil.so.1",
        "libresolv.so.2",
        "libm.so.6",
    };
#endif
    for (const char* candidate : kCandidates) {
        if (dlopen(candidate, RTLD_NOW | RTLD_NOLOAD) != nullptr) continue;
        void* probe = dlopen(candidate, RTLD_NOW | RTLD_LOCAL);
        if (!probe) continue;
        dlclose(probe);
        if (dlopen(candidate, RTLD_NOW | RTLD_NOLOAD) != nullptr) continue;

        expect(load_vulkan_from(candidate) == nullptr,
               "non-Vulkan shared library without vkGetInstanceProcAddr returns nullptr");
        expect(dlopen(candidate, RTLD_NOW | RTLD_NOLOAD) == nullptr,
               "failed shared library override calls dlclose");
        break;
    }
}

#endif

}  // namespace

int main() {
#ifdef _WIN32
    expect(load_vulkan_from("leonida_nonexistent_vulkan_loader_xyz.dll") == nullptr,
           "missing bare DLL override returns nullptr");
    expect(load_vulkan_from("./leonida_nonexistent_dir_xyz/leonida_nonexistent_vulkan_loader.dll") == nullptr,
           "missing path DLL override returns nullptr");
    test_windows_loader_paths();
#elif defined(__APPLE__)
    expect(load_vulkan_from("leonida_nonexistent_vulkan_loader_xyz.dylib") == nullptr,
           "missing bare dylib override returns nullptr");
    expect(load_vulkan_from("./leonida_nonexistent_dir_xyz/leonida_nonexistent_vulkan_loader.dylib") == nullptr,
           "missing path dylib override returns nullptr");
    test_posix_non_vulkan_library();
#else
    expect(load_vulkan_from("leonida_nonexistent_vulkan_loader_xyz.so") == nullptr,
           "missing bare shared-object override returns nullptr");
    expect(load_vulkan_from("./leonida_nonexistent_dir_xyz/leonida_nonexistent_vulkan_loader.so") == nullptr,
           "missing path shared-object override returns nullptr");
    test_posix_non_vulkan_library();
#endif

    const PFN_vkGetInstanceProcAddr first = load_vulkan();
    const PFN_vkGetInstanceProcAddr second = load_vulkan();
    expect(first == second, "load_vulkan() is idempotent across calls");

    if (failures) return 1;
    std::puts("vulkan_loader: all checks passed");
    return 0;
}
