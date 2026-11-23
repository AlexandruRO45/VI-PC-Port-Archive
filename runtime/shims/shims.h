// Shim registry: reimplementations of console-only imports.
// Add a DLL by writing a shims/<dll>.cpp with a SHIM_TABLE and listing it in CMakeLists.txt.
#pragma once
#ifdef _WIN32
#include <windows.h>
#endif
#include <atomic>
#include <cstdint>
#include <cstring>
#include <string>

namespace rt::shims {

struct Entry {
    const char* dll;   // lower-case, with .dll
    const char* name;  // export name, or "#ordinal"
    void* fn;
};

// Registers a static table at startup.
struct Table {
    Table(const Entry* entries, size_t count);
};

void* find(const std::string& dll_lower, const std::string& name);

// Whether the shims own a whole module: the registry serves names from it and it is not a host system DLL (kernel32,
// kernelbase, the CRTs, api-ms-* sets, vulkan-1) of which only a few functions are replaced. A title's
// LoadLibrary/GetModuleHandle of an owned module gets a pseudo-handle that GetProcAddress resolves through find()
// (library_loader.cpp). `dll_lower` is a lower-case file name with ".dll".
bool owns_dll(const std::string& dll_lower);

// What a call to an import that has no binding does. Chosen by the environment variable LEONIDA_UNIMPLEMENTED
// (read once, on the first call):
//   log   (default)  print the name once and return 0, in rax and xmm0
//   fail             as log, but return E_NOTIMPL for imports classified as HRESULT-returning (classify_import)
//   stop             on the first call print the import, the caller's return address as module+RVA and rcx, rdx, r8,
//                    r9 as received; then break if a debugger is attached, else end the process with exit code 3
// Out parameters are never written in any mode.
enum class UnimplementedPolicy { Log, Fail, Stop };

// Whether an import is known to return an HRESULT. Judged by DLL and name only; see registry.cpp for the list and
// where each entry comes from. Unknown imports are Other.
enum class ReturnClass { HResult, Other };

inline constexpr uint32_t kENotImpl = 0x80004001u;  // E_NOTIMPL

// "log", "fail" or "stop", any case. Empty or null is log without complaint; anything else is log with *valid = false.
UnimplementedPolicy parse_unimplemented_policy(const char* text, bool* valid = nullptr);
UnimplementedPolicy unimplemented_policy();
void set_unimplemented_policy(UnimplementedPolicy policy);  // overrides the environment (tests)
ReturnClass classify_import(const std::string& dll, const std::string& name);

// Resolves an address to a mapped module for the `stop` message. A Loader registers itself; `fn` returns true and
// fills the module's file name and the address's RVA when the address is inside one of its modules.
using ModuleLookup = bool (*)(void* ctx, uint64_t address, std::string& module, uint32_t& rva);
void add_module_lookup(void* ctx, ModuleLookup fn);
// "a.dll+0x1234" for an address inside a mapped module, "host:ntdll.dll+0x1234" inside a host module, else the bare
// address in hex. Used by the startup trace (startup_trace.h).
std::string module_offset(uint64_t address);
void remove_module_lookup(void* ctx);

// The base of the mapped module holding `address`, through the same lookups. False if no live Loader maps it.
bool mapped_image_base(uint64_t address, uint64_t& base);

// Run-time LoadLibrary / GetModuleHandle / GetProcAddress / FreeLibrary for bundled package DLLs managed by a live
// Loader. Registered in Loader::Loader and removed in Loader::~Loader.
struct BundledModuleOps {
    void* (*find_loaded)(void* ctx, const std::string& dll_lower);
    void* (*load_bundled)(void* ctx, const std::string& dll_lower, uint32_t flags, bool& handled);
    void* (*get_proc)(void* ctx, void* module, const std::string& sym, bool& owns_handle);
    bool (*owns_handle)(void* ctx, void* module);
    // The same two lookups by the module's native (wide) basename, for names the ANSI code page cannot spell
    // (library_loader's W entries use them when a name has a character outside ASCII). Either may be null.
    void* (*find_loaded_w)(void* ctx, const std::wstring& basename);
    void* (*load_bundled_w)(void* ctx, const std::wstring& basename, uint32_t flags, bool& handled);
};
void add_bundled_loader(void* ctx, const BundledModuleOps* ops);
void remove_bundled_loader(void* ctx);

void* find_loaded_bundled_module(const std::string& dll_lower);
void* load_bundled_module(const std::string& dll_lower, uint32_t flags, bool& handled);
void* find_loaded_bundled_module_w(const std::wstring& basename);
void* load_bundled_module_w(const std::wstring& basename, uint32_t flags, bool& handled);
void* get_bundled_proc_address(void* module, const std::string& sym, bool& owns_handle);
bool owns_bundled_module_handle(void* module);

// Test seam: called at the top of every unimplemented-import call with the address of rt_unimplemented's return
// address slot (so a test can check stack alignment) and from inside the call (so it can walk the stack).
using TestHook = void (*)(const void* return_address_slot);
void set_unimplemented_test_hook(TestHook hook);

// Makes a small thunk for an import with no binding; calling it follows the policy above. It passes the caller's
// return address and the first four argument registers through unchanged.
void* make_unimplemented_thunk(const std::string& dll, const std::string& name);

// Prints every unimplemented import that has been called.
void report_called_unimplemented();

// Layout of the Xbox GDK __XMemAllocationHooks__ export when present in the mapped guest module.
struct XMemAllocationHooks {
    void*(__stdcall* alloc_fn)(size_t size, uint64_t attributes);
    void(__stdcall* free_fn)(void* p, uint64_t attributes);
};

inline std::atomic<const XMemAllocationHooks*>& guest_xmem_hooks_override() {
    static std::atomic<const XMemAllocationHooks*> override_hooks{nullptr};
    return override_hooks;
}

inline void set_guest_xmem_hooks_for_testing(const XMemAllocationHooks* hooks) {
    guest_xmem_hooks_override().store(hooks, std::memory_order_release);
}

using GuestImageBaseLookupFn = bool (*)(uint64_t address, uint64_t& base);

inline std::atomic<GuestImageBaseLookupFn>& guest_image_base_lookup() {
    static std::atomic<GuestImageBaseLookupFn> fn{nullptr};
    return fn;
}

inline void register_guest_image_base_lookup(GuestImageBaseLookupFn fn) {
    guest_image_base_lookup().store(fn, std::memory_order_release);
}

// Resolves __XMemAllocationHooks__ from a mapped guest image via mapped_image_base, or nullptr.
// Requires both alloc_fn and free_fn to be non-null and caches both hits and missed image scans.
inline const XMemAllocationHooks* find_guest_xmem_hooks(uint64_t caller_addr = 0x140000000ULL) {
    if (const auto* override_hooks = guest_xmem_hooks_override().load(std::memory_order_acquire)) {
        return (override_hooks->alloc_fn && override_hooks->free_fn) ? override_hooks : nullptr;
    }
#ifdef _WIN32
    static std::atomic<const XMemAllocationHooks*> cached{nullptr};
    if (const auto* found = cached.load(std::memory_order_acquire)) return found;

    const GuestImageBaseLookupFn lookup_base = guest_image_base_lookup().load(std::memory_order_acquire);
    if (!lookup_base) return nullptr;

    static std::atomic<uint64_t> missed_base_0{0};
    static std::atomic<uint64_t> missed_base_1{0};

    const auto committed_ok = [](uintptr_t addr, size_t size) -> bool {
        MEMORY_BASIC_INFORMATION mbi{};
        if (!VirtualQuery(reinterpret_cast<const void*>(addr), &mbi, sizeof(mbi))) return false;
        if (mbi.State != MEM_COMMIT || (mbi.Protect & (PAGE_NOACCESS | PAGE_GUARD))) return false;
        const uintptr_t end = reinterpret_cast<uintptr_t>(mbi.BaseAddress) + mbi.RegionSize;
        return addr <= end && size <= end - addr;
    };

    const auto scan_image = [&](uintptr_t image_base) -> const XMemAllocationHooks* {
        if (!image_base || !committed_ok(image_base, sizeof(IMAGE_DOS_HEADER))) return nullptr;
        const auto* dos = reinterpret_cast<const IMAGE_DOS_HEADER*>(image_base);
        if (dos->e_magic != IMAGE_DOS_SIGNATURE || dos->e_lfanew < static_cast<LONG>(sizeof(IMAGE_DOS_HEADER)) ||
            dos->e_lfanew > 0x10000) {
            return nullptr;
        }
        const uintptr_t nt_addr = image_base + static_cast<uintptr_t>(dos->e_lfanew);
        if (!committed_ok(nt_addr, sizeof(IMAGE_NT_HEADERS64))) return nullptr;
        const auto* nt = reinterpret_cast<const IMAGE_NT_HEADERS64*>(nt_addr);
        if (nt->Signature != IMAGE_NT_SIGNATURE || nt->OptionalHeader.Magic != IMAGE_NT_OPTIONAL_HDR64_MAGIC) {
            return nullptr;
        }
        if (nt->OptionalHeader.NumberOfRvaAndSizes <= IMAGE_DIRECTORY_ENTRY_EXPORT) return nullptr;
        const IMAGE_DATA_DIRECTORY& exp_dir = nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_EXPORT];
        if (!exp_dir.VirtualAddress || exp_dir.Size < sizeof(IMAGE_EXPORT_DIRECTORY)) return nullptr;

        const uintptr_t exp_addr = image_base + exp_dir.VirtualAddress;
        if (!committed_ok(exp_addr, sizeof(IMAGE_EXPORT_DIRECTORY))) return nullptr;
        const auto* exp = reinterpret_cast<const IMAGE_EXPORT_DIRECTORY*>(exp_addr);
        if (!exp->NumberOfNames || !exp->AddressOfNames || !exp->AddressOfNameOrdinals || !exp->AddressOfFunctions) {
            return nullptr;
        }
        const uintptr_t names_addr = image_base + exp->AddressOfNames;
        const uintptr_t ords_addr = image_base + exp->AddressOfNameOrdinals;
        const uintptr_t funcs_addr = image_base + exp->AddressOfFunctions;
        if (!committed_ok(names_addr, exp->NumberOfNames * sizeof(DWORD)) ||
            !committed_ok(ords_addr, exp->NumberOfNames * sizeof(WORD)) ||
            !committed_ok(funcs_addr, exp->NumberOfFunctions * sizeof(DWORD))) {
            return nullptr;
        }

        const auto* names = reinterpret_cast<const DWORD*>(names_addr);
        const auto* ords = reinterpret_cast<const WORD*>(ords_addr);
        const auto* funcs = reinterpret_cast<const DWORD*>(funcs_addr);
        for (DWORD i = 0; i < exp->NumberOfNames; ++i) {
            const uintptr_t name_ptr = image_base + names[i];
            if (!committed_ok(name_ptr, 24)) continue;
            if (std::strcmp(reinterpret_cast<const char*>(name_ptr), "__XMemAllocationHooks__") != 0) continue;
            const WORD ord = ords[i];
            if (ord >= exp->NumberOfFunctions) return nullptr;
            const uintptr_t sym_addr = image_base + funcs[ord];
            if (!committed_ok(sym_addr, sizeof(XMemAllocationHooks))) return nullptr;
            const auto* hooks = reinterpret_cast<const XMemAllocationHooks*>(sym_addr);
            if (!hooks->alloc_fn || !hooks->free_fn) return nullptr;
            return hooks;
        }
        return nullptr;
    };

    const auto already_missed = [&](uint64_t base) -> bool {
        return base == missed_base_0.load(std::memory_order_acquire) ||
               base == missed_base_1.load(std::memory_order_acquire);
    };
    const auto record_miss = [&](uint64_t base) {
        uint64_t expected = 0;
        if (!missed_base_0.compare_exchange_strong(expected, base, std::memory_order_acq_rel) &&
            expected != base) {
            expected = 0;
            missed_base_1.compare_exchange_strong(expected, base, std::memory_order_acq_rel);
        }
    };

    uint64_t base = 0;
    if (caller_addr && lookup_base(caller_addr, base) && base && !already_missed(base)) {
        if (const auto* hooks = scan_image(static_cast<uintptr_t>(base))) {
            cached.store(hooks, std::memory_order_release);
            return hooks;
        }
        record_miss(base);
    }

    constexpr uint64_t kDefaultGuestAddr = 0x140000000ULL;
    uint64_t default_base = 0;
    if (caller_addr != kDefaultGuestAddr && lookup_base(kDefaultGuestAddr, default_base) && default_base &&
        default_base != base && !already_missed(default_base)) {
        if (const auto* hooks = scan_image(static_cast<uintptr_t>(default_base))) {
            cached.store(hooks, std::memory_order_release);
            return hooks;
        }
        record_miss(default_base);
    }
#endif
    return nullptr;
}

}  // namespace rt::shims

#define SHIM_TABLE(id, ...)                                                       \
    static const ::rt::shims::Entry id##_entries[] = {__VA_ARGS__};               \
    static ::rt::shims::Table id##_table(id##_entries, sizeof(id##_entries) / sizeof(id##_entries[0]))
