// Fixture EXE for the loader's /GA TLS rewrite. It is built like a Windows application (/GA), so its
// __declspec(thread) access reads TEB->ThreadLocalStoragePointer[0] directly instead of going through _tls_index.
// The loader maps it and calls the exported accessors; the entry point is never run. The TLS callback reads the
// thread-local value on every DLL_THREAD_ATTACH, the way the CRT's own callback does on a title's threads.
#include <windows.h>

__declspec(thread) int g_ga_value = 7;

int g_ga_bad_attach = 0;

extern "C" void __cdecl ga_tls_callback(PVOID, DWORD reason, PVOID) {
    if (reason == DLL_THREAD_ATTACH && g_ga_value != 7) ++g_ga_bad_attach;
}

#pragma section(".CRT$XLB", long, read)
extern "C" __declspec(allocate(".CRT$XLB")) PIMAGE_TLS_CALLBACK g_ga_callback_entry = ga_tls_callback;

extern "C" __declspec(dllexport) int __cdecl tiny_ga_get() { return g_ga_value; }

extern "C" __declspec(dllexport) void __cdecl tiny_ga_set(int value) { g_ga_value = value; }

extern "C" __declspec(dllexport) int __cdecl tiny_ga_bad_attach_count() { return g_ga_bad_attach; }

int main() { return 0; }
