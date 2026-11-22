// XAudio2_9 factory forwarding. The system XAudio2 implementation owns every engine/voice object returned here, so
// the shim only translates the console DLL names and keeps the host DLL loaded for the process lifetime.
#include <windows.h>
#include <xaudio2.h>

#include "shims.h"

namespace {

using XAudio2CreateFn = HRESULT(WINAPI*)(IXAudio2**, UINT32, XAUDIO2_PROCESSOR);

HMODULE host_xaudio2_module() {
    static HMODULE module = LoadLibraryExW(L"xaudio2_9.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
    return module;
}

HRESULT WINAPI create_xaudio2(IXAudio2** out_engine, UINT32 flags, XAUDIO2_PROCESSOR processor) {
    HMODULE module = host_xaudio2_module();
    if (!module) {
        const DWORD error = GetLastError();
        if (out_engine) *out_engine = nullptr;
        return error ? HRESULT_FROM_WIN32(error) : E_FAIL;
    }

    auto create = reinterpret_cast<XAudio2CreateFn>(GetProcAddress(module, "XAudio2Create"));
    if (!create) {
        const DWORD error = GetLastError();
        if (out_engine) *out_engine = nullptr;
        return error ? HRESULT_FROM_WIN32(error) : HRESULT_FROM_WIN32(ERROR_PROC_NOT_FOUND);
    }

    return create(out_engine, flags, processor);
}

}  // namespace

SHIM_TABLE(xaudio2,
           {"xaudio2_9.dll", "XAudio2Create", reinterpret_cast<void*>(&create_xaudio2)},
           {"xaudio2_9redist.dll", "XAudio2Create", reinterpret_cast<void*>(&create_xaudio2)},
);
