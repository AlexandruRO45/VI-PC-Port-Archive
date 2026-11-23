#include "process_detach_event.h"
#include <cstdlib>
#include <cstring>
#include <process.h>

extern "C" __declspec(dllimport) void __cdecl detach_touch(unsigned, HANDLE);
extern "C" __declspec(dllimport) void __cdecl detach_heartbeat();
__declspec(thread) int exe_tls = 0x35;
HANDLE ready = nullptr, release_worker = nullptr, entry_thread = nullptr;
char exit_mode[64]{};

void NTAPI detach_exe_tls(PVOID, DWORD reason, PVOID reserved) {
    if (reason == DLL_PROCESS_ATTACH) exe_tls = 0x6A;
    detach_event(1, reason, exe_tls, reserved != nullptr);
}
#pragma section(".CRT$XLB", read)
extern "C" __declspec(allocate(".CRT$XLB")) PIMAGE_TLS_CALLBACK detach_exe_pointer = detach_exe_tls;
#pragma comment(linker, "/include:_tls_used")
#pragma comment(linker, "/include:detach_exe_pointer")

[[noreturn]] void exit_guest() {
    if (!std::strcmp(exit_mode, "_exit")) _exit(23);
    if (!std::strcmp(exit_mode, "_Exit")) std::_Exit(23);
    if (!std::strcmp(exit_mode, "quick_exit")) std::quick_exit(23);
    if (!std::strcmp(exit_mode, "exit")) std::exit(23);
    if (!std::strcmp(exit_mode, "abort")) {
        _set_abort_behavior(0, _WRITE_ABORT_MSG | _CALL_REPORTFAULT);
        std::abort();
    }
    if (!std::strcmp(exit_mode, "TerminateProcess")) TerminateProcess(GetCurrentProcess(), 23);
    ExitProcess(23);
}
DWORD WINAPI idle_worker(void*) {
    SetEvent(ready);
    for (;;) {
        detach_heartbeat();
        Sleep(1);
    }
}
DWORD WINAPI exit_worker(void*) {
    WaitForSingleObject(release_worker, INFINITE);
    exe_tls = 0x97;
    detach_event(1, 101, exe_tls);
    detach_touch(0x98, entry_thread);
    exit_guest();
}
int main() {
    SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX);
    detach_event(1, 100, exe_tls);
    exe_tls = 0x77;
    detach_event(1, 101, exe_tls);
    detach_touch(0x88, nullptr);
    GetEnvironmentVariableA("LEONIDA_DETACH_MODE", exit_mode, sizeof(exit_mode));
    char scenario[64]{};
    GetEnvironmentVariableA("LEONIDA_DETACH_CASE", scenario, sizeof(scenario));
    if (!std::strcmp(scenario, "entry_with_worker")) {
        ready = CreateEventW(nullptr, TRUE, FALSE, nullptr);
        HANDLE worker = CreateThread(nullptr, 0, idle_worker, nullptr, 0, nullptr);
        if (!ready || !worker || WaitForSingleObject(ready, 5000) != WAIT_OBJECT_0) return 99;
        detach_touch(0x88, worker);
    } else if (!std::strcmp(scenario, "worker_exit") || !std::strcmp(scenario, "racing_exit")) {
        release_worker = CreateEventW(nullptr, TRUE, FALSE, nullptr);
        if (!DuplicateHandle(GetCurrentProcess(), GetCurrentThread(), GetCurrentProcess(), &entry_thread,
                             SYNCHRONIZE, FALSE, 0)) return 99;
        HANDLE worker = CreateThread(nullptr, 0, exit_worker, nullptr, 0, nullptr);
        if (!release_worker || !worker) return 99;
        SetEvent(release_worker);
        if (!std::strcmp(scenario, "worker_exit")) Sleep(INFINITE);
        // Either caller may win; callbacks must see the TLS belonging to the winner and no live competitor.
        detach_touch(0x88, worker);
    } else if (!std::strcmp(scenario, "ExitThread")) {
        ExitThread(42);
    }
    if (!std::strcmp(exit_mode, "return")) return 23;
    exit_guest();
}
