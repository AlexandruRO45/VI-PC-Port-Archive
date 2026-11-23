// Checks the registered xcrdapi.dll shim (DStorageGetFactory) and its stub COM factory vtable.
//
// Everything runs in-process through rt::shims::find and the library_loader shims; no retail files or DirectStorage
// runtime are required.
#include <windows.h>
#include <unknwn.h>

#include <cstdint>
#include <cstdio>

#include "shims.h"

namespace {

using DStorageGetFactoryFn = HRESULT(WINAPI*)(const IID*, void**);
using LoadLibraryWFn = HMODULE(WINAPI*)(LPCWSTR);
using GetModuleHandleAFn = HMODULE(WINAPI*)(LPCSTR);
using GetProcAddressFn = FARPROC(WINAPI*)(HMODULE, LPCSTR);
using FreeLibraryFn = BOOL(WINAPI*)(HMODULE);

using QueryInterfaceRawFn = HRESULT(STDMETHODCALLTYPE*)(void*, const IID*, void**);
using AddRefRawFn = ULONG(STDMETHODCALLTYPE*)(void*);
using ReleaseRawFn = ULONG(STDMETHODCALLTYPE*)(void*);
using FactorySlotFn = HRESULT(STDMETHODCALLTYPE*)(void*);

int g_checks = 0;
int g_failures = 0;

void check(bool condition, const char* what) {
    ++g_checks;
    std::printf("%s %s\n", condition ? "PASS" : "FAIL", what);
    if (!condition) ++g_failures;
}

}  // namespace

int main() {
    std::printf("xcrdapi_test: registered xcrdapi.dll shim\n");

    check(rt::shims::owns_dll("xcrdapi.dll"), "rt::shims::owns_dll(\"xcrdapi.dll\") is true");

    const auto get_factory =
        reinterpret_cast<DStorageGetFactoryFn>(rt::shims::find("xcrdapi.dll", "DStorageGetFactory"));
    check(get_factory != nullptr, "registry exposes xcrdapi.dll!DStorageGetFactory");
    if (!get_factory) return 1;

    // Resolve through library_loader (as a title calling LoadLibraryW/GetProcAddress would).
    const auto load_w = reinterpret_cast<LoadLibraryWFn>(rt::shims::find("kernel32.dll", "LoadLibraryW"));
    const auto get_mod_a = reinterpret_cast<GetModuleHandleAFn>(rt::shims::find("kernel32.dll", "GetModuleHandleA"));
    const auto get_proc = reinterpret_cast<GetProcAddressFn>(rt::shims::find("kernel32.dll", "GetProcAddress"));
    const auto free_lib = reinterpret_cast<FreeLibraryFn>(rt::shims::find("kernel32.dll", "FreeLibrary"));
    check(load_w && get_mod_a && get_proc && free_lib, "library_loader shims resolve from kernel32.dll");
    if (load_w && get_mod_a && get_proc && free_lib) {
        HMODULE mod = load_w(L"XCRDAPI.dll");
        check(mod != nullptr, "LoadLibraryW(L\"XCRDAPI.dll\") returns a shim pseudo-handle");
        check(get_mod_a("xcrdapi") == mod, "GetModuleHandleA(\"xcrdapi\") matches the pseudo-handle");
        check(reinterpret_cast<void*>(get_proc(mod, "DStorageGetFactory")) == reinterpret_cast<void*>(get_factory),
              "GetProcAddress resolves DStorageGetFactory to the registered shim");
        SetLastError(ERROR_SUCCESS);
        check(get_proc(mod, "NonExistentExport") == nullptr && GetLastError() == ERROR_PROC_NOT_FOUND,
              "GetProcAddress on an unserved export returns NULL with ERROR_PROC_NOT_FOUND");
        check(free_lib(mod) != FALSE, "FreeLibrary on the pseudo-handle returns TRUE");
    }

    // Argument validation.
    check(get_factory(nullptr, nullptr) == E_POINTER, "DStorageGetFactory(nullptr, nullptr) returns E_POINTER");

    constexpr GUID kTestIid = {0x6924ea0c, 0xc3cd, 0x4826, {0xb1, 0x0a, 0xf6, 0x4f, 0x4e, 0xd9, 0x27, 0xc1}};
    check(get_factory(&kTestIid, nullptr) == E_POINTER, "DStorageGetFactory(&iid, nullptr) returns E_POINTER");

    void* factory = reinterpret_cast<void*>(uintptr_t{0x5a5a5a5a5a5a5a5aULL});
    check(get_factory(nullptr, &factory) == E_INVALIDARG && factory == nullptr,
          "DStorageGetFactory(nullptr, &out) zeroes out and returns E_INVALIDARG");

    // Valid call returns S_OK and a non-null COM factory stub.
    check(get_factory(&kTestIid, &factory) == S_OK && factory != nullptr,
          "DStorageGetFactory(&iid, &out) returns S_OK with non-null factory");
    if (!factory) return 1;

    const void* const* vtable = *reinterpret_cast<const void* const* const*>(factory);
    check(vtable != nullptr, "factory has a non-null vtable");
    if (!vtable) return 1;

    const auto qi = reinterpret_cast<QueryInterfaceRawFn>(vtable[0]);
    const auto add_ref = reinterpret_cast<AddRefRawFn>(vtable[1]);
    const auto release = reinterpret_cast<ReleaseRawFn>(vtable[2]);

    void* qi_out = reinterpret_cast<void*>(uintptr_t{0x1234});
    check(qi(factory, &kTestIid, nullptr) == E_POINTER, "QueryInterface with null ppv returns E_POINTER");
    check(qi(factory, nullptr, &qi_out) == E_INVALIDARG && qi_out == nullptr,
          "QueryInterface with null riid zeroes ppv and returns E_INVALIDARG");
    check(qi(factory, &IID_IUnknown, &qi_out) == S_OK && qi_out == factory,
          "QueryInterface(IID_IUnknown) returns S_OK and the factory singleton");

    // Refcounting: initial DStorageGetFactory added 1, QueryInterface added 1 -> refs == 2.
    const ULONG after_add = add_ref(factory);
    check(after_add == 3, "AddRef increments reference count to 3");
    check(release(factory) == 2, "Release decrements reference count to 2");
    check(release(factory) == 1, "Release decrements reference count to 1");
    check(release(factory) == 0, "Release decrements reference count to 0");
    check(release(factory) == 0, "extra Release on static singleton saturates at 0 without underflow");

    // Remaining factory slots 3..63 return E_NOTIMPL.
    bool all_slots_notimpl = true;
    for (size_t slot = 3; slot < 64; ++slot) {
        const auto method = reinterpret_cast<FactorySlotFn>(vtable[slot]);
        if (!method || method(factory) != E_NOTIMPL) {
            all_slots_notimpl = false;
            std::printf("FAIL slot %zu did not return E_NOTIMPL\n", slot);
            break;
        }
    }
    check(all_slots_notimpl, "factory vtable slots 3..63 return E_NOTIMPL");

    std::printf("%d check(s), %d failure(s)\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
