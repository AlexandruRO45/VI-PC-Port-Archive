// library_loader: LoadLibrary / GetModuleHandle / GetProcAddress for modules the shims own, and GetModuleHandle /
// GetModuleHandleEx / GetModuleFileName / GetProcAddress for the images the loader maps.
//
// GDK code does not always import the console runtime. The package's libHttpClient.GDK.dll and XCurl.dll have no
// import entry for xgameruntime.dll at all: their statically linked xgameruntime.lib thunks hold the wide string
// "xgameruntime.dll" and the names InitializeApiImpl, InitializeApiImplEx, InitializeApiImplEx2, QueryApiImpl and
// UninitializeApiImpl, and look them up at run time (docs/runtime_imports.md, "Run-time loaded modules"). Bound to the
// host, that LoadLibraryW would load the PC's Gaming Services runtime or fail, and never reach the offline shims.
//
// So for a module the shim registry owns (rt::shims::owns_dll: xgameruntime.dll, d3d12_xs.dll, xmem.dll, ...), these
// return a pseudo-handle, and GetProcAddress on it resolves names (and "#ordinal") through rt::shims::find. A name the
// registry doesn't serve returns NULL with ERROR_PROC_NOT_FOUND, as a real module would, and is logged once.
//
// The loader's images are VirtualAlloc'd, so Windows doesn't know them: asked for one by name, handle or address it
// fails, and asked for the process's executable (NULL) it names gta6rt.exe, from which a title would take its install
// folder. GetModuleHandle*, GetModuleHandleEx*, GetModuleFileName* and GetProcAddress answer for them from the live
// Loaders (rt::shims::find_mapped_module, mapped_image_base, get_bundled_proc_address), with Windows' name rules and
// return conventions. NULL, to all four, is the EXE `gta6rt --run` runs (Loader::set_main_module), as Windows answers
// NULL with the process's executable; without one it stays Windows' answer, gta6rt.exe.
//
// Every other module uses the same named shims as static imports, falling back to the host. Only images mapped by the
// loader see these: they are bound through the registry, the runtime's own code calls the host directly.
#include <windows.h>

#include <cstdio>
#include <cstring>
#include <cwchar>
#include <deque>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <vector>

#include "shims.h"

namespace rt::shims::library_loader {

// A module the shims own, identified by a pointer to one of these. Host modules are image bases (64 KiB aligned);
// these live in the runtime's heap, and GetProcAddress tells them apart by lookup, not by address range.
struct Pseudo {
    char tag[8] = {'L', 'E', 'O', 'S', 'H', 'I', 'M', 0};
    std::string dll;  // lower case, with .dll
};

namespace {

// Intentionally immortal: mapped-DLL DLL_PROCESS_DETACH and earlier atexit handlers can call GetProcAddress or
// FreeLibrary on pseudo-handles during process teardown, after normal statics would have been destroyed.
std::mutex& loader_mutex() {
    static auto* m = new std::mutex;
    return *m;
}

std::deque<Pseudo>& pseudos() {   // a deque never moves its elements, so handed-out handles stay valid
    static auto* d = new std::deque<Pseudo>;
    return *d;
}

std::string lower_ascii(std::string s) {
    for (auto& c : s)
        if (c >= 'A' && c <= 'Z') c = static_cast<char>(c + 32);
    return s;
}

std::string narrow(const wchar_t* w) {
    std::string s;
    for (; w && *w; ++w) s += (*w < 0x80) ? static_cast<char>(*w) : '?';
    return s;
}

// LoadLibrary's file-name rules for the module a path names: the last path component, lower case, and ".dll" added
// when there is no extension (a trailing '.' means "no extension, add nothing").
std::string module_name(std::string path) {
    const size_t slash = path.find_last_of("\\/");
    if (slash != std::string::npos) path.erase(0, slash + 1);
    path = lower_ascii(path);
    if (!path.empty() && path.back() == '.') path.pop_back();
    else if (path.find('.') == std::string::npos) path += ".dll";
    return path;
}

std::string host_module_name(HMODULE module) {
    std::lock_guard lock(loader_mutex());
    static auto* names = new std::map<HMODULE, std::string>;
    if (auto it = names->find(module); it != names->end()) return it->second;

    std::vector<wchar_t> path(MAX_PATH);
    for (;;) {
        const DWORD length = GetModuleFileNameW(module, path.data(), static_cast<DWORD>(path.size()));
        if (!length) return {};
        if (length < path.size()) {
            const std::string name = module_name(narrow(path.data()));
            names->emplace(module, name);
            return name;
        }
        if (path.size() == 32768) return {};
        path.resize(path.size() > 16384 ? 32768 : path.size() * 2);
    }
}

}  // namespace

// The pseudo-handle for an owned module, or nullptr when the shims don't own `path`'s module.
HMODULE owned_handle(const std::string& path) {
    const std::string dll = module_name(path);
    if (dll.empty() || !owns_dll(dll)) return nullptr;
    std::lock_guard lock(loader_mutex());
    for (auto& p : pseudos())
        if (p.dll == dll) return reinterpret_cast<HMODULE>(&p);
    pseudos().push_back(Pseudo{});
    pseudos().back().dll = dll;
    std::fprintf(stderr, "[rt] loader: %s is served by the shims (run-time load)\n", dll.c_str());
    return reinterpret_cast<HMODULE>(&pseudos().back());
}

// The owned module a handle names, or nullptr for a host module.
const Pseudo* pseudo(HMODULE h) {
    std::lock_guard lock(loader_mutex());
    for (auto& p : pseudos())
        if (reinterpret_cast<HMODULE>(&p) == h) return &p;
    return nullptr;
}

namespace {

// ---- Mapped images. The rules below are what Windows 11 does, measured by calling its functions on host modules;
// pe_loader_test (test_module_identity, test_proc_address) and the tiny_game fixture check the shims against them.

// GetFullPathName's answer: '/' made '\\', "." and ".." resolved, a relative path made full against the current
// folder. Empty if it fails.
std::wstring full_path(const std::wstring& name) {
    const DWORD need = GetFullPathNameW(name.c_str(), 0, nullptr, nullptr);
    if (!need) return {};
    std::wstring full(need, L'\0');
    const DWORD got = GetFullPathNameW(name.c_str(), need, full.data(), nullptr);
    if (!got || got >= need) return {};
    full.resize(got);
    return full;
}

// A relative path as Windows types one (RtlDetermineDosPathNameType_U): neither rooted ("\\x", "\\\\server\\x",
// "\\\\?\\x") nor on a drive ("C:\\x", "C:x").
bool relative_path(const std::wstring& name) {
    return !name.empty() && name[0] != L'\\' && name[0] != L'/' && !(name.size() >= 2 && name[1] == L':');
}

// The file a relative name with a path ("..\\x.dll", ".\\x.dll", "sub\\x.dll") names to GetModuleHandle. Windows
// doesn't make it full against the current folder: it looks for it along the DLL search path, as LoadLibrary would,
// and the first entry under which the name exists, as a file or a folder, decides, even when that file isn't loaded
// and one further along is. The search path is ntdll's own (LdrGetDllPath, exported since Windows 8, with no flags,
// as for LoadLibrary: the application's folder, here gta6rt.exe's, the system folders, the current folder, then PATH,
// as SetDllDirectory leaves it). Empty when no candidate exists or there is no search path. The last error is left
// alone.
std::wstring searched_path(const std::wstring& name) {
    using GetDllPath = LONG(NTAPI*)(PCWSTR, ULONG, PWSTR*, PWSTR*);
    using ReleasePath = VOID(NTAPI*)(PWSTR);
    static const HMODULE ntdll = GetModuleHandleW(L"ntdll.dll");
    static const auto get_path = reinterpret_cast<GetDllPath>(GetProcAddress(ntdll, "LdrGetDllPath"));
    static const auto release = reinterpret_cast<ReleasePath>(GetProcAddress(ntdll, "RtlReleasePath"));
    const DWORD error = GetLastError();
    PWSTR path = nullptr, unused = nullptr;  // the second output was null in every case measured
    std::wstring found;
    if (get_path && release && get_path(name.c_str(), 0, &path, &unused) >= 0 && path) {
        for (const wchar_t* entry = path; *entry && found.empty();) {
            const wchar_t* end = std::wcschr(entry, L';');
            if (!end) end = entry + std::wcslen(entry);
            std::wstring candidate(entry, end);
            entry = *end ? end + 1 : end;
            if (candidate.empty()) continue;
            if (candidate.back() != L'\\' && candidate.back() != L'/') candidate += L'\\';
            candidate += name;
            if (GetFileAttributesW(candidate.c_str()) != INVALID_FILE_ATTRIBUTES) found = full_path(candidate);
        }
        release(path);
    }
    SetLastError(error);
    return found;
}

// GetModuleHandle's name rules: ".dll" is added when the file name has no extension, and trailing dots, however many,
// mean "no extension, add nothing" and are dropped. A name with a path ('\\' or '/') names the module mapped from that
// file: a relative one is looked for along the DLL search path (searched_path), any other is made full as
// GetFullPathName makes it. Empty if it names no file.
std::wstring mapped_name(std::wstring name) {
    const size_t slash = name.find_last_of(L"\\/");
    if (!name.empty() && name.back() == L'.') {
        while (!name.empty() && name.back() == L'.') name.pop_back();
    } else if (name.find(L'.', slash == std::wstring::npos ? 0 : slash + 1) == std::wstring::npos) {
        name += L".dll";
    }
    if (slash == std::wstring::npos) return name;
    return relative_path(name) ? searched_path(name) : full_path(name);
}

// The image a live Loader maps under `name`, or with a null name its main executable; null when there's none, and
// the caller asks Windows instead. Answers before the host, so a mapped image wins over a host module of that name.
HMODULE mapped_module(const wchar_t* name) {
    ModuleQuery query;
    std::wstring key;
    if (name) {
        key = mapped_name(name);
        if (key.empty()) return nullptr;
        query.name = key.c_str();
    }
    uint64_t base = 0;
    return find_mapped_module(query, base) ? reinterpret_cast<HMODULE>(base) : nullptr;
}

// GetModuleHandleA and GetModuleHandleExA read the name in the file APIs' code page (SetFileApisToOEM switches it),
// while GetModuleFileNameA writes the ANSI one either way.
HMODULE mapped_module_a(const char* name) {
    if (!name) return mapped_module(nullptr);
    const UINT code_page = AreFileApisANSI() ? CP_ACP : CP_OEMCP;
    const int n = MultiByteToWideChar(code_page, 0, name, -1, nullptr, 0);
    if (n <= 0) return nullptr;
    std::wstring wide(static_cast<size_t>(n), L'\0');
    MultiByteToWideChar(code_page, 0, name, -1, wide.data(), n);
    wide.resize(static_cast<size_t>(n - 1));
    return mapped_module(wide.c_str());
}

// The mapped image holding `address`, as GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS finds it: any byte of the image.
HMODULE mapped_module_at(const void* address) {
    uint64_t base = 0;
    return mapped_image_base(reinterpret_cast<uint64_t>(address), base) ? reinterpret_cast<HMODULE>(base) : nullptr;
}

// Whether `module` is the base of a mapped image, or with a null module whether there is a main executable; `path`,
// when not null, gets its full path. A handle that is not an image's base (a LOAD_LIBRARY_AS_DATAFILE one has a low
// bit set) names no module, as in Windows.
bool mapped_image(HMODULE module, std::wstring* path) {
    ModuleQuery query;
    query.base = reinterpret_cast<uint64_t>(module);
    uint64_t base = 0;
    return find_mapped_module(query, base, path);
}

// GetModuleFileName's copy since Vista: the whole name and a null, returning its length with the last error set to
// ERROR_SUCCESS; or, when that doesn't fit, size - 1 characters and a null, returning size with
// ERROR_INSUFFICIENT_BUFFER. With size 0 nothing is written and 0 is returned.
template <class Char>
DWORD copy_file_name(const std::basic_string<Char>& name, Char* buffer, DWORD size) {
    if (name.size() < size) {
        std::memcpy(buffer, name.c_str(), (name.size() + 1) * sizeof(Char));
        SetLastError(ERROR_SUCCESS);
        return static_cast<DWORD>(name.size());
    }
    if (size) {
        std::memcpy(buffer, name.data(), (size - 1) * sizeof(Char));
        buffer[size - 1] = 0;
    }
    SetLastError(ERROR_INSUFFICIENT_BUFFER);
    return size;
}

// GetModuleHandleEx checks its arguments before it looks anything up: the out handle is cleared, and an unknown flag,
// PIN with UNCHANGED_REFCOUNT, FROM_ADDRESS without an address, or no out pointer fail with ERROR_INVALID_PARAMETER.
bool module_handle_ex_arguments(DWORD flags, const void* name, HMODULE* out) {
    constexpr DWORD known = GET_MODULE_HANDLE_EX_FLAG_PIN | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT |
                            GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS;
    if (out) *out = nullptr;
    const bool pin_and_unchanged =
        (flags & GET_MODULE_HANDLE_EX_FLAG_PIN) && (flags & GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT);
    if ((flags & ~known) || pin_and_unchanged || (!name && (flags & GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS)) || !out) {
        SetLastError(ERROR_INVALID_PARAMETER);
        return false;
    }
    return true;
}

constexpr DWORD kDataFileLoadFlags =
    LOAD_LIBRARY_AS_DATAFILE | LOAD_LIBRARY_AS_DATAFILE_EXCLUSIVE | LOAD_LIBRARY_AS_IMAGE_RESOURCE;

HMODULE load_shim_or_bundled(const std::string& path, DWORD flags, bool& handled) {
    handled = false;
    if (HMODULE h = owned_handle(path)) {
        handled = true;
        return h;
    }
    const std::string dll = module_name(path);
    if (!dll.empty()) {
        if (void* h = load_bundled_module(dll, static_cast<uint32_t>(flags), handled); handled)
            return reinterpret_cast<HMODULE>(h);
    }
    return nullptr;
}

HMODULE WINAPI load_library_a(LPCSTR name) {
    if (name) {
        bool handled = false;
        HMODULE h = load_shim_or_bundled(name, 0, handled);
        if (handled) return h;
    }
    return LoadLibraryA(name);
}

HMODULE WINAPI load_library_w(LPCWSTR name) {
    if (name) {
        bool handled = false;
        HMODULE h = load_shim_or_bundled(narrow(name), 0, handled);
        if (handled) return h;
    }
    return LoadLibraryW(name);
}

HMODULE WINAPI load_library_ex_a(LPCSTR name, HANDLE file, DWORD flags) {
    if (name && (flags & kDataFileLoadFlags) == 0) {
        bool handled = false;
        HMODULE h = load_shim_or_bundled(name, flags, handled);
        if (handled) return h;
    }
    return LoadLibraryExA(name, file, flags);
}

HMODULE WINAPI load_library_ex_w(LPCWSTR name, HANDLE file, DWORD flags) {
    if (name && (flags & kDataFileLoadFlags) == 0) {
        bool handled = false;
        HMODULE h = load_shim_or_bundled(narrow(name), flags, handled);
        if (handled) return h;
    }
    return LoadLibraryExW(name, file, flags);
}

// GetModuleHandle answers for an owned module as if it were loaded (the GDK may probe before it loads), then for a
// mapped image, bundled package modules a run-time LoadLibrary mapped included, and NULL is the main executable when
// a Loader has one. A hit leaves the last error alone, as Windows does.
HMODULE WINAPI get_module_handle_a(LPCSTR name) {
    if (HMODULE h = name ? owned_handle(name) : nullptr) return h;
    if (HMODULE h = mapped_module_a(name)) return h;
    return GetModuleHandleA(name);
}

HMODULE WINAPI get_module_handle_w(LPCWSTR name) {
    if (HMODULE h = name ? owned_handle(narrow(name)) : nullptr) return h;
    if (HMODULE h = mapped_module(name)) return h;
    return GetModuleHandleW(name);
}

// GetModuleHandleEx answers what GetModuleHandle does, and FROM_ADDRESS for an address inside a mapped image. Owned
// modules and mapped images are not unloaded while mapped code runs, so neither the reference count nor PIN needs
// bookkeeping: FreeLibrary on them succeeds and changes nothing (free_library).
BOOL WINAPI get_module_handle_ex_a(DWORD flags, LPCSTR name, HMODULE* out) {
    if (!module_handle_ex_arguments(flags, name, out)) return FALSE;
    HMODULE h = nullptr;
    if (flags & GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS) {
        h = mapped_module_at(name);
    } else {
        if (name) h = owned_handle(name);
        if (!h) h = mapped_module_a(name);
    }
    if (!h) return GetModuleHandleExA(flags, name, out);
    *out = h;
    return TRUE;
}

BOOL WINAPI get_module_handle_ex_w(DWORD flags, LPCWSTR name, HMODULE* out) {
    if (!module_handle_ex_arguments(flags, name, out)) return FALSE;
    HMODULE h = nullptr;
    if (flags & GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS) {
        h = mapped_module_at(name);
    } else {
        if (name) h = owned_handle(narrow(name));
        if (!h) h = mapped_module(name);
    }
    if (!h) return GetModuleHandleExW(flags, name, out);
    *out = h;
    return TRUE;
}

// GetModuleFileName names a mapped image by the full path it was mapped from, and NULL the main executable. Anything
// else, owned pseudo-handles included, is Windows' to answer.
DWORD WINAPI get_module_file_name_a(HMODULE module, LPSTR buffer, DWORD size) {
    std::wstring path;
    if (!mapped_image(module, &path)) return GetModuleFileNameA(module, buffer, size);
    const int n = WideCharToMultiByte(CP_ACP, 0, path.c_str(), -1, nullptr, 0, nullptr, nullptr);
    if (n <= 0) return 0;  // the last error is the conversion's
    std::string ansi(static_cast<size_t>(n), '\0');
    WideCharToMultiByte(CP_ACP, 0, path.c_str(), -1, ansi.data(), n, nullptr, nullptr);
    ansi.resize(static_cast<size_t>(n - 1));
    return copy_file_name(ansi, buffer, size);
}

DWORD WINAPI get_module_file_name_w(HMODULE module, LPWSTR buffer, DWORD size) {
    std::wstring path;
    if (!mapped_image(module, &path)) return GetModuleFileNameW(module, buffer, size);
    return copy_file_name(path, buffer, size);
}

// A GetProcAddress name as the registry and Module::exports key it: the name, or "#N" for an ordinal (a value below
// 0x10000, IS_INTRESOURCE, as in the real GetProcAddress).
std::string proc_key(LPCSTR name) {
    return IS_INTRESOURCE(name) ? "#" + std::to_string(reinterpret_cast<uintptr_t>(name)) : name;
}

// Whether a mapped image has an export directory, which decides GetProcAddress's error for a null name: Windows looks
// only at the export data directory's RVA (measured on patched copies of a DLL: with the size 0, or a directory that
// lists no functions, it is still ERROR_INVALID_PARAMETER; with the RVA 0 it is ERROR_INVALID_ORDINAL). `module` is
// the base of an image a live Loader maps, so its headers are there and were checked when it was mapped.
bool has_export_directory(HMODULE module) {
    const auto* base = reinterpret_cast<const BYTE*>(module);
    const auto* nt = reinterpret_cast<const IMAGE_NT_HEADERS64*>(
        base + reinterpret_cast<const IMAGE_DOS_HEADER*>(base)->e_lfanew);
    return nt->OptionalHeader.NumberOfRvaAndSizes > IMAGE_DIRECTORY_ENTRY_EXPORT &&
           nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_EXPORT].VirtualAddress != 0;
}

// GetProcAddress on an owned module resolves through the registry. On a module a live Loader maps it answers from the
// image's exports, a forwarder followed as for an import (Loader::resolve_module_export), and what the image doesn't
// export fails as in Windows: a name with ERROR_PROC_NOT_FOUND, an ordinal with ERROR_INVALID_ORDINAL, and a null
// name (ordinal 0) with ERROR_INVALID_PARAMETER when the image has an export directory, ERROR_INVALID_ORDINAL when it
// has none. A hit leaves the last error alone. A null module is the process's executable: Windows answers
// GetProcAddress(NULL, name) from the EXE's own exports (measured on Windows 11: the same address as through
// GetModuleHandle(NULL)), so here it is the EXE `gta6rt --run` runs, through the same lookup as its handle, and
// without one Windows' answer from gta6rt.exe.
FARPROC WINAPI get_proc_address(HMODULE module, LPCSTR name) {
    if (const Pseudo* p = pseudo(module)) {
        if (!name) {
            SetLastError(ERROR_PROC_NOT_FOUND);
            return nullptr;
        }
        const std::string what = proc_key(name);
        if (void* fn = find(p->dll, what)) return reinterpret_cast<FARPROC>(fn);
        {
            static auto* m = new std::mutex;
            static auto* seen = new std::set<std::string>;
            std::lock_guard lock(*m);
            if (seen->insert(p->dll + "!" + what).second)
                std::fprintf(stderr, "[rt] loader: GetProcAddress(%s, %s): not served\n", p->dll.c_str(), what.c_str());
        }
        SetLastError(ERROR_PROC_NOT_FOUND);
        return nullptr;
    }

    if (!module) module = mapped_module(nullptr);  // the main module, or still null when no Loader has one

    if (!name) {
        if (owns_bundled_module_handle(module)) {
            SetLastError(has_export_directory(module) ? ERROR_INVALID_PARAMETER : ERROR_INVALID_ORDINAL);
            return nullptr;
        }
        return GetProcAddress(module, name);
    }

    const std::string what = proc_key(name);
    bool owns_handle = false;
    void* fn = get_bundled_proc_address(module, what, owns_handle);
    if (owns_handle) {
        if (fn) return reinterpret_cast<FARPROC>(fn);
        SetLastError(IS_INTRESOURCE(name) ? ERROR_INVALID_ORDINAL : ERROR_PROC_NOT_FOUND);
        return nullptr;
    }
    if (module && !IS_INTRESOURCE(name)) {
        const DWORD error = GetLastError();
        const std::string dll = host_module_name(module);
        void* shim = dll.empty() ? nullptr : find(dll, what);
        SetLastError(error);
        if (shim) return reinterpret_cast<FARPROC>(shim);
    }
    return GetProcAddress(module, name);
}

BOOL WINAPI free_library(HMODULE module) {
    if (pseudo(module)) return TRUE;   // owned modules are never unloaded
    // Nor are mapped modules: they stay live until ~Loader detaches them, and GetModuleHandleEx counted no reference
    // to release.
    if (owns_bundled_module_handle(module)) return TRUE;
    return FreeLibrary(module);
}

// DisableThreadLibraryCalls(hinstance) from a mapped image's DllMain. The real function only accepts a module the
// Windows loader mapped, so for ours it fails, and a DllMain that returns its result (the MSVC runtime's satellite
// DLLs do) fails Loader::initialise() with "DllMain returned FALSE". Owned and mapped modules answer TRUE. The loader
// sends no thread events to a DllMain, so there is nothing further to switch off; every other handle goes to the
// real function.
BOOL WINAPI disable_thread_calls(HMODULE module) {
    if (pseudo(module)) return TRUE;
    if (owns_bundled_module_handle(module)) return TRUE;
    return DisableThreadLibraryCalls(module);
}

#define LOADER(dll)                                                                         \
    {dll, "LoadLibraryA", reinterpret_cast<void*>(&rt::shims::library_loader::load_library_a)},                        \
    {dll, "LoadLibraryW", reinterpret_cast<void*>(&rt::shims::library_loader::load_library_w)},                        \
    {dll, "LoadLibraryExA", reinterpret_cast<void*>(&rt::shims::library_loader::load_library_ex_a)},                   \
    {dll, "LoadLibraryExW", reinterpret_cast<void*>(&rt::shims::library_loader::load_library_ex_w)},                   \
    {dll, "GetModuleHandleA", reinterpret_cast<void*>(&rt::shims::library_loader::get_module_handle_a)},               \
    {dll, "GetModuleHandleW", reinterpret_cast<void*>(&rt::shims::library_loader::get_module_handle_w)},               \
    {dll, "GetModuleHandleExA", reinterpret_cast<void*>(&rt::shims::library_loader::get_module_handle_ex_a)},          \
    {dll, "GetModuleHandleExW", reinterpret_cast<void*>(&rt::shims::library_loader::get_module_handle_ex_w)},          \
    {dll, "GetModuleFileNameA", reinterpret_cast<void*>(&rt::shims::library_loader::get_module_file_name_a)},          \
    {dll, "GetModuleFileNameW", reinterpret_cast<void*>(&rt::shims::library_loader::get_module_file_name_w)},          \
    {dll, "GetProcAddress", reinterpret_cast<void*>(&rt::shims::library_loader::get_proc_address)},                    \
    {dll, "FreeLibrary", reinterpret_cast<void*>(&rt::shims::library_loader::free_library)},                           \
    {dll, "DisableThreadLibraryCalls", reinterpret_cast<void*>(&rt::shims::library_loader::disable_thread_calls)}

}  // namespace
}  // namespace rt::shims::library_loader

// The package's DLLs import the loader functions through these api sets (XCurl.dll: libraryloader-l1-2-0 and
// -l1-2-1); kernel32 and kernelbase cover code linked against the classic libraries.
SHIM_TABLE(library_loader,
    LOADER("kernel32.dll"),
    LOADER("kernelbase.dll"),
    LOADER("api-ms-win-core-libraryloader-l1-1-0.dll"),
    LOADER("api-ms-win-core-libraryloader-l1-1-1.dll"),
    LOADER("api-ms-win-core-libraryloader-l1-2-0.dll"),
    LOADER("api-ms-win-core-libraryloader-l1-2-1.dll"),
    LOADER("api-ms-win-core-libraryloader-l1-2-2.dll"),
);
