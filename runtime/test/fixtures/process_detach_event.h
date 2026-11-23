#pragma once
#include <windows.h>
#include <cstdint>

// Unbuffered records survive _exit/ExitProcess and never depend on the shared CRT's stdout state.
struct DetachEvent {
    uint32_t magic, module, stage, thread, value, reserved;
};
inline void detach_event(uint32_t module, uint32_t stage, uint32_t value, bool reserved = false) {
    const DetachEvent event{0x52543337, module, stage, GetCurrentThreadId(), value, reserved ? 1u : 0u};
    DWORD written = 0;
    WriteFile(GetStdHandle(STD_OUTPUT_HANDLE), &event, sizeof(event), &written, nullptr);
}
