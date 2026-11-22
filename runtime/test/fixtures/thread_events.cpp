// A CRT-free DLL that records the thread events it is told about, in the order it gets them. thread_events_test maps it
// with the loader and loads the same file with the Windows loader as the control. No game data.
//
// Built four times from this file:
//   thread_events_plain         DllMain only: no TLS directory, like most DLLs;
//   thread_events_tls           DllMain plus a TLS callback (a callback-only TLS directory, no data block);
//   thread_events_disabled      DllMain only, and it calls DisableThreadLibraryCalls(hinstance) on attach and returns
//                               its result, as the MSVC runtime's satellite DLLs do;
//   thread_events_disabled_tls  the same call, in a DLL that also has the TLS callback.
// Every variant imports from kernel32 (SetLastError in record): the Windows loader ignores the TLS directory of a DLL
// loaded at run time that has no import table.
//
// The log is two characters per event: who ('D' = DllMain, 'C' = TLS callback) and what ('A' = DLL_THREAD_ATTACH,
// 'T' = DLL_THREAD_DETACH). Process events are not recorded.
#include <windows.h>

namespace {

char g_log[512];
volatile LONG g_length = 0;

void record(char who, DWORD reason) {
    const char what = reason == DLL_THREAD_ATTACH ? 'A' : reason == DLL_THREAD_DETACH ? 'T' : 0;
    if (!what) return;
    SetLastError(0);  // the import that makes Windows honour a TLS directory (see above)
    const LONG at = InterlockedExchangeAdd(&g_length, 2);
    if (at + 2 <= static_cast<LONG>(sizeof g_log)) {
        g_log[at] = who;
        g_log[at + 1] = what;
    }
}

#if defined(THREAD_EVENTS_TLS) || defined(THREAD_EVENTS_DISABLE_TLS)
ULONG g_tls_index = 0;

void NTAPI tls_callback(PVOID, DWORD reason, PVOID) { record('C', reason); }

PIMAGE_TLS_CALLBACK g_callbacks[] = {tls_callback, nullptr};
#endif

}  // namespace

#if defined(THREAD_EVENTS_TLS) || defined(THREAD_EVENTS_DISABLE_TLS)
// A TLS directory with no data block (start == end, no zero fill) and one callback; there is no CRT to provide one.
extern "C" const IMAGE_TLS_DIRECTORY64 _tls_used = {
    0,                                          // StartAddressOfRawData
    0,                                          // EndAddressOfRawData
    reinterpret_cast<ULONGLONG>(&g_tls_index),  // AddressOfIndex
    reinterpret_cast<ULONGLONG>(g_callbacks),   // AddressOfCallBacks
    0,                                          // SizeOfZeroFill
    0,                                          // Characteristics
};
#endif

extern "C" BOOL WINAPI thread_events_main(HINSTANCE instance, DWORD reason, LPVOID) {
#if defined(THREAD_EVENTS_DISABLE) || defined(THREAD_EVENTS_DISABLE_TLS)
    if (reason == DLL_PROCESS_ATTACH) return DisableThreadLibraryCalls(instance);
#else
    (void)instance;
#endif
    record('D', reason);
    return TRUE;
}

// Copies the log (NUL-terminated) into `out` and returns its length. The optimizer is off here: it would turn the copy
// loop into a memcpy call, and there is no CRT to provide one.
#pragma optimize("", off)
extern "C" __declspec(dllexport) int __cdecl thread_events_log(char* out, int capacity) {
    int n = static_cast<int>(g_length);
    if (n > static_cast<int>(sizeof g_log)) n = static_cast<int>(sizeof g_log);
    if (capacity <= 0) return n;
    if (n > capacity - 1) n = capacity - 1;
    for (int i = 0; i < n; ++i) out[i] = g_log[i];
    out[n] = 0;
    return n;
}
#pragma optimize("", on)

extern "C" __declspec(dllexport) void __cdecl thread_events_reset() { g_length = 0; }
