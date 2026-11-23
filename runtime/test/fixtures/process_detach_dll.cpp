#include "process_detach_event.h"
#include <cstring>

__declspec(thread) int dll_tls = 0x22;
void* volatile observed_thread = nullptr;
volatile LONG heartbeat = 0;
extern "C" __declspec(dllexport) void __cdecl detach_heartbeat() { InterlockedIncrement(&heartbeat); }

void NTAPI detach_dll_tls(PVOID, DWORD reason, PVOID reserved) {
    if (reason == DLL_PROCESS_ATTACH) dll_tls = 0x44;
    detach_event(2, reason, dll_tls, reserved != nullptr);
}
#pragma section(".CRT$XLB", read)
extern "C" __declspec(allocate(".CRT$XLB")) PIMAGE_TLS_CALLBACK detach_dll_pointer = detach_dll_tls;
#pragma comment(linker, "/include:_tls_used")
#pragma comment(linker, "/include:detach_dll_pointer")

extern "C" __declspec(dllexport) void __cdecl detach_touch(unsigned value, HANDLE observed) {
    dll_tls = value;
    InterlockedExchangePointer(&observed_thread, observed);
    detach_event(2, 100, dll_tls);
}
BOOL WINAPI DllMain(HINSTANCE, DWORD reason, LPVOID reserved) {
    detach_event(2, 10 + reason, dll_tls, reserved != nullptr);
    if (reason == DLL_PROCESS_DETACH) {
        char scenario[64]{};
        GetEnvironmentVariableA("LEONIDA_DETACH_CASE", scenario, sizeof(scenario));
        const HANDLE observed = InterlockedCompareExchangePointer(&observed_thread, nullptr, nullptr);
        if (observed && std::strcmp(scenario, "racing_exit"))
            detach_event(2, 200, WaitForSingleObject(observed, 0));
        if (!std::strcmp(scenario, "entry_with_worker")) {
            detach_event(2, 201, InterlockedCompareExchange(&heartbeat, 0, 0));
            Sleep(30);
            detach_event(2, 202, InterlockedCompareExchange(&heartbeat, 0, 0));
        }
        if (!std::strcmp(scenario, "reentry")) ExitProcess(23);
    }
    return TRUE;
}
