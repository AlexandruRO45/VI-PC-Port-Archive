// Original synthetic PE startup probe: entry-thread state and a fresh worker TLS template.
#include <windows.h>

namespace {
__declspec(thread) unsigned int tls_value = 0x35;
DWORD process_thread;
DWORD worker_thread;
unsigned int process_before, process_after, process_count;
unsigned int thread_before, thread_count, thread_detach_count, thread_detach_value;
DWORD detach_thread;
unsigned int unexpected_attach;
bool inject(const char* wanted) {
    char mode[16];
    return GetEnvironmentVariableA("LEONIDA_TLS_PROBE_BAD", mode, sizeof(mode)) &&
           lstrcmpA(mode, wanted) == 0;
}
void NTAPI startup_callback(PVOID, DWORD reason, PVOID) {
    if (reason == DLL_PROCESS_ATTACH) {
        ++process_count;
        if (inject("count")) ++process_count;  // Negative control: the test must reject this.
        process_thread = GetCurrentThreadId();
        process_before = tls_value;
        tls_value = 0x6A;
        process_after = tls_value;
    } else if (reason == DLL_THREAD_ATTACH) {
        ++thread_count;
        if (GetCurrentThreadId() == process_thread) ++unexpected_attach;
        thread_before = tls_value;
        if (inject("state")) tls_value = 0x99;
    } else if (reason == DLL_THREAD_DETACH) {
        ++thread_detach_count;
        detach_thread = GetCurrentThreadId();
        thread_detach_value = tls_value;
    }
}
DWORD WINAPI worker(void*) {
    worker_thread = GetCurrentThreadId();
    if (tls_value != 0x35 || thread_before != 0x35 || thread_count != 1 ||
        worker_thread == process_thread) return 7;
    tls_value = 0x77;
    return 0;
}
}

#pragma section(".CRT$XLB", read)
#pragma comment(linker, "/INCLUDE:_tls_used")
#pragma comment(linker, "/INCLUDE:tls_startup_probe_callback")
extern "C" {
__declspec(allocate(".CRT$XLB")) PIMAGE_TLS_CALLBACK tls_startup_probe_callback = startup_callback;
}

extern "C" DWORD WINAPI tls_startup_probe_entry(void*) {
    const DWORD entry_thread = GetCurrentThreadId();
    const unsigned int entry_value = tls_value;
    DWORD code = process_count != 1 || process_before != 0x35 || process_after != 0x6A ? 2 :
                 entry_value != 0x6A ? 3 : entry_thread != process_thread ? 4 :
                 thread_count != 0 || unexpected_attach != 0 ? 5 : 0;
    if (!code) {
        HANDLE thread = CreateThread(nullptr, 0, worker, nullptr, 0, nullptr);
        if (!thread) code = 6;
        else {
            DWORD worker_code = 0;
            if (WaitForSingleObject(thread, 5000) != WAIT_OBJECT_0 ||
                !GetExitCodeThread(thread, &worker_code)) {
                // Do not return with a live worker touching fixture state.
                ExitProcess(8);
            }
            CloseHandle(thread);
            if (worker_code) code = worker_code;
            else if (tls_value != 0x6A || unexpected_attach != 0 || thread_count != 1 ||
                     thread_detach_count != 1 || detach_thread != worker_thread ||
                     thread_detach_value != 0x77) code = 9;
        }
    }
    char output[768];
    const int length = wsprintfA(output,
        "process_attach tid=%lu count=%u before=0x%02X after=0x%02X\n"
        "entry tid=%lu value=0x%02X same_thread=%u unexpected_thread_attach=%u\n"
        "worker tid=%lu attach_count=%u template=0x%02X detach_tid=%lu detach_count=%u detach_value=0x%02X\n"
        "entry_final=0x%02X exit=%lu\n",
        process_thread, process_count, process_before, process_after,
        entry_thread, entry_value, entry_thread == process_thread ? 1u : 0u, unexpected_attach,
        worker_thread, thread_count, thread_before, detach_thread, thread_detach_count, thread_detach_value,
        tls_value, code);
    DWORD written;
    WriteFile(GetStdHandle(STD_OUTPUT_HANDLE), output, static_cast<DWORD>(length), &written, nullptr);
    return code;
}