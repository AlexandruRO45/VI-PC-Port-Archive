// Two original synthetic DLLs exercise dependency rollback with live static TLS.
#include <windows.h>

namespace {
void event(const char* name, unsigned int count, unsigned int value) {
    char text[192];
    const int size = wsprintfA(text, "EVENT %s count=%u tid=%lu value=0x%02X\n",
                              name, count, GetCurrentThreadId(), value);
    DWORD written;
    WriteFile(GetStdHandle(STD_OUTPUT_HANDLE), text, size, &written, nullptr);
}
}

#if defined(ROLLBACK_DEPENDENCY_A)
namespace {
__declspec(thread) unsigned int tls_value = 0x35;
DWORD attach_thread;
unsigned int attach_count;
unsigned int detach_count;
unsigned int tls_attach_count;
unsigned int tls_detach_count;
bool callback_ok;
bool reentered;
void NTAPI tls_callback(PVOID, DWORD reason, PVOID) {
    if (reason == DLL_PROCESS_ATTACH) {
        ++tls_attach_count;
        callback_ok = tls_value == 0x35 && tls_attach_count == 1;
        tls_value = 0x6A;
        event("A_TLS_ATTACH", tls_attach_count, tls_value);
    } else if (reason == DLL_THREAD_ATTACH || reason == DLL_THREAD_DETACH) {
        event("A_THREAD_INVALID", 1, tls_value);
    } else if (reason == DLL_PROCESS_DETACH) {
        char bad[16];
        if (GetEnvironmentVariableA("LEONIDA_ROLLBACK_BAD", bad, sizeof(bad))) {
            if (lstrcmpA(bad, "count") == 0) ++tls_detach_count;
            if (lstrcmpA(bad, "state") == 0) tls_value = 0x99;
        }
        event("A_TLS_DETACH", ++tls_detach_count, tls_value);
    }
}
}
#pragma comment(linker, "/EXPORT:rollback_forward=startup_rollback_a.rollback_a_ready")
#pragma section(".CRT$XLB", read)
#pragma comment(linker, "/INCLUDE:_tls_used")
#pragma comment(linker, "/INCLUDE:rollback_tls_callback")
extern "C" {
__declspec(allocate(".CRT$XLB")) PIMAGE_TLS_CALLBACK rollback_tls_callback = tls_callback;
}
extern "C" __declspec(dllexport) bool rollback_a_ready() {
    return callback_ok && tls_value == 0x6A && attach_count == 1 && attach_thread == GetCurrentThreadId();
}
BOOL WINAPI DllMain(HINSTANCE, DWORD reason, LPVOID) {
    if (reason == DLL_PROCESS_ATTACH) {
        attach_thread = GetCurrentThreadId();
        ++attach_count;
        event(rollback_a_ready() ? "A_ATTACH" : "A_INVALID", attach_count, tls_value);
    } else if (reason == DLL_THREAD_ATTACH || reason == DLL_THREAD_DETACH) {
        event("A_THREAD_INVALID", 1, tls_value);
    } else if (reason == DLL_PROCESS_DETACH) {
        ++detach_count;
        char enabled[8];
        if (!reentered && GetEnvironmentVariableA("LEONIDA_ROLLBACK_REENTER", enabled, sizeof(enabled))) {
            reentered = true;  // Bound the reproducer even if the loader erroneously reattaches us.
            HMODULE self = LoadLibraryA("startup_rollback_a.dll");
            const DWORD error = GetLastError();
            event(!self && error == ERROR_DLL_INIT_FAILED ? "A_LOAD_BLOCKED" : "A_LOAD_INVALID", 1, tls_value);
            HMODULE loaded = GetModuleHandleA("startup_rollback_a.dll");
            event(loaded && GetProcAddress(loaded, "rollback_a_ready") ? "A_LOOKUP_OK" : "A_LOOKUP_INVALID", 1, tls_value);
            event(loaded && !GetProcAddress(loaded, "rollback_forward") ?
                  "A_FORWARD_BLOCKED" : "A_FORWARD_INVALID", 1, tls_value);
        }
        const bool valid = rollback_a_ready() && detach_count == 1 && tls_detach_count == 1;
        event(valid ? "A_DETACH" : "A_INVALID", detach_count, tls_value);
    }
    return TRUE;
}
#else
extern "C" __declspec(dllimport) bool rollback_a_ready();
extern "C" __declspec(dllexport) int rollback_b() { return 0; }
BOOL WINAPI DllMain(HINSTANCE, DWORD reason, LPVOID) {
    if (reason == DLL_PROCESS_ATTACH) {
        event(rollback_a_ready() ? "B_REJECT" : "B_INVALID", 1, 0);
        return FALSE;
    }
    if (reason == DLL_PROCESS_DETACH) event("B_DETACH", 1, 0);
    return TRUE;
}
#endif
