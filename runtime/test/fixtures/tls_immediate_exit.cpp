// Synthetic stand-in program for `gta6rt --run` (RT-374): PE TLS DLL_PROCESS_DETACH on immediate-exit paths.
// Test code, not game content. check_tls_immediate_exit.cmake runs it natively and through gta6rt --run for
// _exit, _Exit, quick_exit, and ExitProcess and verifies that PE TLS process-detach runs while atexit does not.
#include <windows.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {

thread_local unsigned int g_tls_value = 0x11;

void say(const char* what) {
    std::printf("tls_immediate_exit: %s\n", what);
    std::fflush(stdout);
}

void atexit_handler() { say("atexit"); }

void NTAPI tls_callback(PVOID, DWORD reason, PVOID) {
    if (reason == DLL_PROCESS_ATTACH) {
        g_tls_value = 0x22;
        say("tls_attach");
    } else if (reason == DLL_PROCESS_DETACH) {
        char buf[64];
        std::snprintf(buf, sizeof(buf), "tls_detach=0x%02x", g_tls_value);
        say(buf);
    }
}

}  // namespace

#pragma section(".CRT$XLB", read)
#pragma comment(linker, "/INCLUDE:_tls_used")
#pragma comment(linker, "/INCLUDE:tls_immediate_exit_cb")
extern "C" {
__declspec(allocate(".CRT$XLB")) PIMAGE_TLS_CALLBACK tls_immediate_exit_cb = tls_callback;
}

int main(int argc, char** argv) {
    const char* mode = argc > 1 ? argv[1] : "_exit";
    std::atexit(atexit_handler);
    g_tls_value = 0x42;
    say("main");
    if (!std::strcmp(mode, "_exit")) _exit(8);
    if (!std::strcmp(mode, "_Exit")) std::_Exit(9);
    if (!std::strcmp(mode, "quick_exit")) std::quick_exit(10);
    if (!std::strcmp(mode, "ExitProcess")) ExitProcess(11);
    return 1;
}
