// Two DLLs that load each other during process attach, and a static importer of A.
#include <windows.h>

#ifdef ATTACH_DLL_IMPORTER
extern "C" __declspec(dllimport) BOOL WINAPI attach_complete();
#endif

namespace {
#ifdef ATTACH_DLL_IMPORTER
BOOL dependency_ready;
#else
#ifdef ATTACH_DLL_A
constexpr auto peer_name = L"dll_attach_b.dll";
constexpr int detach_id = 1;
#else
constexpr auto peer_name = L"dll_attach_a.dll";
constexpr int detach_id = 2;
#endif
int process_attaches;
HMODULE peer;
BOOL process_attach_complete;
int* detach_log;
#endif
}

extern "C" BOOL WINAPI dll_attach_main(HINSTANCE, DWORD reason, LPVOID) {
#ifdef ATTACH_DLL_IMPORTER
    if (reason == DLL_PROCESS_ATTACH) dependency_ready = attach_complete();
#else
    // Stop a broken loader's recursion so the test can report duplicate attaches.
    if (reason == DLL_PROCESS_ATTACH && ++process_attaches == 1) {
        peer = LoadLibraryW(peer_name);
        process_attach_complete = peer != nullptr;
        return process_attach_complete;
    }
    if (reason == DLL_PROCESS_DETACH && detach_log) *detach_log = *detach_log * 10 + detach_id;
#endif
    return TRUE;
}

#ifdef ATTACH_DLL_IMPORTER
extern "C" __declspec(dllexport) BOOL WINAPI dependency_was_ready() { return dependency_ready; }
#else
extern "C" __declspec(dllexport) BOOL WINAPI attach_complete() { return process_attach_complete; }
extern "C" __declspec(dllexport) int WINAPI attach_count() { return process_attaches; }
extern "C" __declspec(dllexport) HMODULE WINAPI peer_handle() { return peer; }
extern "C" __declspec(dllexport) void WINAPI set_detach_log(int* log) { detach_log = log; }
#endif
