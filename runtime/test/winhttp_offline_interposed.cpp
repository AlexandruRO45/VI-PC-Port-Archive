// Compile the shim with native auto-proxy lookup interposed, so tests cannot start discovery.
#include <windows.h>
#include <winhttp.h>
#include <cstring>

int g_native_auto_proxy_calls = 0;

BOOL WINAPI intercepted_auto_proxy(DWORD, LPWSTR*) {
    ++g_native_auto_proxy_calls;
    SetLastError(ERROR_WINHTTP_AUTODETECTION_FAILED);
    return FALSE;
}

FARPROC WINAPI intercepted_get_proc_address(HMODULE module, LPCSTR name) {
    if (reinterpret_cast<ULONG_PTR>(name) > 0xffff &&
        std::strcmp(name, "WinHttpDetectAutoProxyConfigUrl") == 0) {
        return reinterpret_cast<FARPROC>(&intercepted_auto_proxy);
    }
    return GetProcAddress(module, name);
}

#define GetProcAddress intercepted_get_proc_address
#define WinHttpDetectAutoProxyConfigUrl intercepted_auto_proxy
#include "../shims/winhttp_offline.cpp"
