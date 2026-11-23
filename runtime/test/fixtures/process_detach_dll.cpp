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
BOOL WINAPI DllMain(HINSTANCE image, DWORD reason, LPVOID reserved) {
    detach_event(2, 10 + reason, dll_tls, reserved != nullptr);
    if (reason == DLL_PROCESS_DETACH) {
        char scenario[64]{};
        GetEnvironmentVariableA("LEONIDA_DETACH_CASE", scenario, sizeof(scenario));
        if (!std::strncmp(scenario, "lookup", 6)) {
            const auto module = GetModuleHandleW(L"process_detach_dep.dll");
            detach_event(2, 210, module == image);
            const auto proc = GetProcAddress(module, "detach_heartbeat");
            detach_event(2, 211, proc == reinterpret_cast<FARPROC>(&detach_heartbeat));
            const auto host = GetModuleHandleW(L"kernel32.dll");
            detach_event(2, 212, GetProcAddress(host, "GetCurrentProcessId") != nullptr);
            if (!std::strcmp(scenario, "lookup_cache")) {
                const auto shim = GetModuleHandleW(L"process_detach_probe.dll");
                const auto exported = reinterpret_cast<unsigned(WINAPI*)()>(GetProcAddress(shim, "detach_probe_export"));
                detach_event(2, 213, shim && exported && exported() == 7);
            }
        }
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
