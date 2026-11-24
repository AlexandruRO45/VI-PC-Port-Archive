// Second CRT-free DLL. It imports the first one and its initializer reads a_value(), so the value it
// publishes shows whether the first DLL was initialized before it. NO_INIT drops the initializer.
#include <windows.h>

#pragma runtime_checks("", off)

extern "C" __declspec(dllimport) void a_log(char event);
extern "C" __declspec(dllimport) unsigned a_value();

namespace {
volatile unsigned g_b_value;
}

extern "C" __declspec(dllexport) unsigned b_value() { return g_b_value; }

extern "C" BOOL WINAPI b_main(HINSTANCE, DWORD reason, LPVOID) {
#ifndef NO_INIT
    if (reason == DLL_PROCESS_ATTACH) {
        a_log('b');
        g_b_value = a_value() * 2;
    }
#endif
    return TRUE;
}
