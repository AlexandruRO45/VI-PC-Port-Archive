// Shared pieces of the macOS host prototype: the guest's TEB, the gs switch between guest and host code, and the
// export table the loader binds imports from.
#pragma once
#include <pthread.h>

#include <cstdint>
#include <cstdlib>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

extern "C" void _thread_set_tsd_base(uint64_t);  // private libsystem_kernel call; Wine uses it the same way

#define MSABI __attribute__((ms_abi))

namespace host {

constexpr size_t kTebSize = 0x2000;
constexpr size_t kTebStackBase = 0x08, kTebStackLimit = 0x10, kTebSelf = 0x30, kTebPid = 0x40, kTebTid = 0x48;
constexpr size_t kTebTlsPointer = 0x58, kTebPeb = 0x60, kTebLastError = 0x68, kTebTlsSlots = 0x1480;
constexpr size_t kTebHostGs = 0x1F00;  // our stash, past the end of the documented x64 TEB (0x1838)
constexpr int kTlsSlots = 64;

template <typename T> T& teb_at(uint8_t* teb, size_t off) { return *reinterpret_cast<T*>(teb + off); }

extern thread_local uint8_t* t_teb;  // only read while gs holds the pthread base
extern uint8_t* g_image;             // the mapped game image
extern std::string g_command_line;   // the guest's, as GetCommandLineA gives it
extern int g_argc;
extern char** g_argv;

uint8_t* make_teb();          // TEB and static TLS for the calling thread; runs the TLS callbacks with `reason`
void attach_thread(uint8_t* teb, uint32_t reason);
void free_teb(uint8_t* teb);
uint32_t thread_id();         // the id GetCurrentThreadId and the CRT report
void set_last_error(uint32_t e);
std::string to_host_path(const char* windows_path);
[[noreturn]] void fail(const std::string& why);

// Run guest code with gs on the thread's TEB, from host code.
template <typename R, typename... A, typename... B> R call_guest(R(MSABI* fn)(A...), B... args) {
    uint8_t* teb = t_teb;
    _thread_set_tsd_base(reinterpret_cast<uint64_t>(teb));
    if constexpr (std::is_void_v<R>) {
        fn(args...);
        _thread_set_tsd_base(teb_at<uint64_t>(teb, kTebHostGs));
    } else {
        R r = fn(args...);
        _thread_set_tsd_base(teb_at<uint64_t>(teb, kTebHostGs));
        return r;
    }
}

// Entered from guest code: gs -> pthread base for the host function, back to the TEB on return.
struct HostGs {
    uint8_t* teb;
    HostGs() {
        __asm__ volatile("movq %%gs:0x30, %0" : "=r"(teb));
        _thread_set_tsd_base(teb_at<uint64_t>(teb, kTebHostGs));
    }
    ~HostGs() { _thread_set_tsd_base(reinterpret_cast<uint64_t>(teb)); }
};

template <typename Sig, Sig* F> struct Thunk;
template <typename R, typename... A, R (*F)(A...)> struct Thunk<R(A...), F> {
    static MSABI R call(A... a) {
        HostGs g;
        return F(a...);
    }
};

// Kernel handles.
struct Object {
    enum Kind { File, Thread } kind;
    int fd = -1;
    pthread_t thread{};
    uint32_t tid = 0;
    uint32_t exit_code = 0;
    bool joined = false;
};
using ThreadProc = uint32_t(MSABI*)(void*);
Object* spawn_thread(ThreadProc proc, void* arg);  // nullptr on failure
uint32_t wait_thread(Object* o);                   // joins; returns the exit code
void close_object(Object* o);

using Exports = std::vector<std::pair<std::string, void*>>;  // "dll!name" -> ms_abi entry
Exports kernel32_exports();
Exports crt_exports();
Exports eh_exports();

}  // namespace host

#define EXPORT(dll, name) {dll "!" #name, reinterpret_cast<void*>(&host::Thunk<decltype(name), &name>::call)}
#define EXPORT_AS(dll, sym, name) {dll "!" sym, reinterpret_cast<void*>(&host::Thunk<decltype(name), &name>::call)}
