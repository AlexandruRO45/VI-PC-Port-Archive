// Two synthetic DLLs that load each other from DLL_PROCESS_ATTACH.
#include <windows.h>

namespace {
#ifdef ATTACH_DLL_A
constexpr auto peer_name = L"dll_attach_b.dll";
#else
constexpr auto peer_name = L"dll_attach_a.dll";
#endif
int process_attaches;
HMODULE peer;
}

extern "C" BOOL WINAPI dll_attach_main(HINSTANCE, DWORD reason, LPVOID) {
    // Stop a broken loader's recursion so the test can report duplicate attaches.
    if (reason == DLL_PROCESS_ATTACH && ++process_attaches == 1) {
        peer = LoadLibraryW(peer_name);
        return peer != nullptr;
    }
    return TRUE;
}

extern "C" __declspec(dllexport) int WINAPI attach_count() { return process_attaches; }
extern "C" __declspec(dllexport) HMODULE WINAPI peer_handle() { return peer; }
