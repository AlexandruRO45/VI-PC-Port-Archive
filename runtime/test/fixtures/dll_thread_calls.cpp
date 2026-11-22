// CRT-free fixture, built once with thread notifications enabled and once with process-attach opt-out.
#include <windows.h>

namespace {
volatile LONG g_counts[4]{};
// Keep a base relocation when both copies are mapped at once.
volatile LONG* volatile g_relocated_counts = g_counts;
volatile LONG g_disable_on_attach = 0;
volatile LONG g_disable_succeeded = 0;
}

extern "C" BOOL WINAPI dll_thread_calls_main(HINSTANCE instance, DWORD reason, LPVOID) {
    if (reason < 4) InterlockedIncrement(g_relocated_counts + reason);
#ifdef DLL_THREAD_CALLS_OPTOUT
    if (reason == DLL_PROCESS_ATTACH) return DisableThreadLibraryCalls(instance);
#endif
    if (reason == DLL_THREAD_ATTACH && InterlockedExchange(&g_disable_on_attach, 0))
        InterlockedExchange(&g_disable_succeeded, DisableThreadLibraryCalls(instance));
    return TRUE;
}

extern "C" __declspec(dllexport) LONG dll_thread_calls_count(DWORD reason) {
    return reason < 4 ? InterlockedCompareExchange(g_relocated_counts + reason, 0, 0) : -1;
}

extern "C" __declspec(dllexport) void dll_thread_calls_disable_on_attach() {
    InterlockedExchange(&g_disable_on_attach, 1);
}

extern "C" __declspec(dllexport) BOOL dll_thread_calls_did_disable() {
    return InterlockedCompareExchange(&g_disable_succeeded, 0, 0);
}

extern "C" __declspec(dllexport) BOOL dll_thread_calls_bogus_handle() {
    return DisableThreadLibraryCalls(reinterpret_cast<HMODULE>(static_cast<uintptr_t>(0x10000)));
}
