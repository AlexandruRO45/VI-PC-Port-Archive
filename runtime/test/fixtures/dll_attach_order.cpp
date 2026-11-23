#include <windows.h>

#ifdef ATTACH_CYCLE

#ifdef ATTACH_CYCLE_A
#define OWN_ATTACHED cycle_a_attached
#define PEER_ATTACHED cycle_b_attached
#else
#define OWN_ATTACHED cycle_b_attached
#define PEER_ATTACHED cycle_a_attached
#endif
extern "C" __declspec(dllimport) BOOL WINAPI PEER_ATTACHED();
namespace {
BOOL attached;
BOOL observed_peer;
}
extern "C" BOOL WINAPI dll_attach_main(HINSTANCE, DWORD reason, LPVOID) {
    if (reason == DLL_PROCESS_ATTACH) {
        observed_peer = PEER_ATTACHED();
        attached = TRUE;
    }
    return TRUE;
}
extern "C" __declspec(dllexport) BOOL WINAPI OWN_ATTACHED() { return attached; }
extern "C" __declspec(dllexport) BOOL WINAPI observed_peer_attach() { return observed_peer; }

#elif defined(ATTACH_CYCLE_SEED)

// Export the same symbol as A so B can be linked before the real A exists.
extern "C" __declspec(dllexport) BOOL WINAPI cycle_a_attached() { return FALSE; }
extern "C" BOOL WINAPI dll_attach_main(HINSTANCE, DWORD, LPVOID) { return TRUE; }

#elif defined(ATTACH_FAILURE_ROOT)

namespace {
DWORD load_error;
int unexpected_attaches;
}
extern "C" __declspec(dllexport) void WINAPI record_unexpected_attach() { ++unexpected_attaches; }
extern "C" __declspec(dllexport) int WINAPI failure_checked() {
    return load_error == ERROR_DLL_INIT_FAILED && unexpected_attaches == 0;
}
extern "C" BOOL WINAPI dll_attach_main(HINSTANCE, DWORD reason, LPVOID) {
    if (reason == DLL_PROCESS_ATTACH && !LoadLibraryW(L"dll_attach_failure_b.dll"))
        load_error = GetLastError();
    return TRUE;
}

#elif defined(ATTACH_FAILURE_B)

extern "C" __declspec(dllimport) void WINAPI record_unexpected_attach();
extern "C" __declspec(dllimport) void WINAPI failing_dependency();
extern "C" BOOL WINAPI dll_attach_main(HINSTANCE, DWORD reason, LPVOID) {
    if (reason == DLL_PROCESS_ATTACH) {
        failing_dependency();
        record_unexpected_attach();
    }
    return TRUE;
}

#else

extern "C" __declspec(dllexport) void WINAPI failing_dependency() {}
extern "C" BOOL WINAPI dll_attach_main(HINSTANCE, DWORD reason, LPVOID) {
    return reason != DLL_PROCESS_ATTACH;
}

#endif
