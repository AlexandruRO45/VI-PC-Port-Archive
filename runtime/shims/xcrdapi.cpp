// XCRDAPI.dll: Xbox DirectStorage (DStorageGetFactory). The game tries to get the console's DirectStorage
// factory for async I/O. Returning E_NOTIMPL causes it to fall back to standard file I/O (CreateFile /
// ReadFile), which the file_system shim already handles.
//
// The return type is HRESULT (registry.cpp classifies it); the ABI matches the PC DirectStorage
// DStorageGetFactory(REFIID riid, void** ppv) documented on learn.microsoft.com.
#include <windows.h>

#include <atomic>
#include <cstdio>

#include "shims.h"

namespace {

HRESULT __stdcall DStorageGetFactory(const IID* /*riid*/, void** ppv) {
    static std::atomic<bool> warned{false};
    if (!warned.exchange(true, std::memory_order_relaxed))
        std::fprintf(stderr, "[rt] xcrdapi: DStorageGetFactory: returning E_NOTIMPL; "
                     "the game falls back to file I/O\n");
    if (ppv) *ppv = nullptr;
    return E_NOTIMPL;
}

}  // namespace

SHIM_TABLE(xcrdapi,
    {"xcrdapi.dll", "DStorageGetFactory", reinterpret_cast<void*>(&DStorageGetFactory)},
);
