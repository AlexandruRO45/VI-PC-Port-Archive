// Per-thread values for the runtime, held in TlsAlloc slots instead of the EXE's static TLS block.
//
// The mapped title was linked with its static TLS at module index 0, and its CRT inlines gs:[58][0] at every
// thread-local access. gta6rt.exe is the first module in the process, so index 0 is this program's own module index:
// the loader copies the title's 5,248-byte template to the base of that block, into the room the pad in
// src/pe_loader.cpp reserves in front of the runtime's own TLS variables. Anything the runtime needs per thread
// therefore lives here, in a dynamic TLS slot, where the title's writes cannot reach it.
#pragma once

#include <windows.h>

#include <cstdint>

namespace rt {

template <typename Tag>
class HostTls {
public:
    static std::uintptr_t get() { return reinterpret_cast<std::uintptr_t>(TlsGetValue(slot())); }
    static void set(std::uintptr_t value) { TlsSetValue(slot(), reinterpret_cast<void*>(value)); }
    static bool flag() { return get() != 0; }
    static void set_flag(bool value) { set(value ? 1 : 0); }

private:
    static DWORD slot() {
        static const DWORD value = TlsAlloc();
        return value;
    }
};

struct InHandlerTag {};
struct WideFallbackTag {};
struct SingleSteppingTag {};
struct TimeSensitiveTag {};
struct SaveReadCountTag {};
struct QueuePumpTag {};
struct ThreadTlsSlotsTag {};

using t_in_handler = HostTls<InHandlerTag>;
using t_wide_fallback = HostTls<WideFallbackTag>;
using t_single_stepping = HostTls<SingleSteppingTag>;
using t_time_sensitive = HostTls<TimeSensitiveTag>;
using t_save_read_count = HostTls<SaveReadCountTag>;
using t_queue_pump = HostTls<QueuePumpTag>;
using t_thread_tls_slots = HostTls<ThreadTlsSlotsTag>;

}  // namespace rt
