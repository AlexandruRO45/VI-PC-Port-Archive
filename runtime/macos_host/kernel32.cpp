// kernel32 on POSIX: the subset guest/guest.c and Leonida's tiny_game import. Windows types are spelled with
// fixed widths (DWORD = uint32_t, LONG = int32_t, WCHAR = char16_t), since long and wchar_t differ on macOS.
#include "host.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

#include <cstdio>
#include <cstring>
#include <map>
#include <mutex>

namespace host {
namespace {

using HANDLE = void*;
HANDLE const kInvalid = reinterpret_cast<HANDLE>(-1);
Object g_std[3] = {{Object::File, 0}, {Object::File, 1}, {Object::File, 2}};

[[noreturn]] void unsupported(const char* what) { fail(std::string("unsupported: ") + what); }

HANDLE GetStdHandle(uint32_t which) {
    switch (static_cast<int32_t>(which)) {
        case -10: return &g_std[0];
        case -11: return &g_std[1];
        case -12: return &g_std[2];
        default: return kInvalid;
    }
}
int WriteFile(HANDLE h, const void* buf, uint32_t n, uint32_t* written, void*) {
    ssize_t r = write(static_cast<Object*>(h)->fd, buf, n);
    if (written) *written = r < 0 ? 0 : static_cast<uint32_t>(r);
    return r >= 0;
}
int ReadFile(HANDLE h, void* buf, uint32_t n, uint32_t* got, void*) {
    ssize_t r = read(static_cast<Object*>(h)->fd, buf, n);
    if (got) *got = r < 0 ? 0 : static_cast<uint32_t>(r);
    return r >= 0;
}
[[noreturn]] void ExitProcess(uint32_t code) {
    std::fflush(nullptr);
    _exit(static_cast<int>(code));
}
int TerminateProcess(HANDLE, uint32_t code) { ExitProcess(code); }
uint32_t GetLastError() { return teb_at<uint32_t>(t_teb, kTebLastError); }

HANDLE GetProcessHeap() { return reinterpret_cast<HANDLE>(0x1000); }  // one heap: malloc
void* HeapAlloc(HANDLE, uint32_t flags, uint64_t size) { return (flags & 0x8) ? calloc(1, size) : malloc(size); }
int HeapFree(HANDLE, uint32_t, void* p) {
    free(p);
    return 1;
}

std::mutex g_vmem_lock;
std::map<uintptr_t, size_t> g_vmem;  // ponytail: whole reserve/commit/release only, no partial decommit
void* VirtualAlloc(void* addr, uint64_t size, uint32_t, uint32_t) {
    void* p = mmap(addr, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
    if (p == MAP_FAILED) {
        set_last_error(8 /* ERROR_NOT_ENOUGH_MEMORY */);
        return nullptr;
    }
    std::lock_guard<std::mutex> lock(g_vmem_lock);
    g_vmem[reinterpret_cast<uintptr_t>(p)] = size;
    return p;
}
int VirtualFree(void* addr, uint64_t, uint32_t type) {
    if (type != 0x8000 /* MEM_RELEASE */) return 1;
    std::lock_guard<std::mutex> lock(g_vmem_lock);
    auto it = g_vmem.find(reinterpret_cast<uintptr_t>(addr));
    if (it == g_vmem.end()) {
        set_last_error(87 /* ERROR_INVALID_PARAMETER */);
        return 0;
    }
    munmap(addr, it->second);
    g_vmem.erase(it);
    return 1;
}

std::mutex g_tls_lock;
bool g_tls_used[kTlsSlots] = {};
uint32_t TlsAlloc() {
    std::lock_guard<std::mutex> lock(g_tls_lock);
    for (int i = 0; i < kTlsSlots; ++i)
        if (!g_tls_used[i]) {
            g_tls_used[i] = true;
            return static_cast<uint32_t>(i);
        }
    return 0xFFFFFFFFu;
}
void* TlsGetValue(uint32_t i) { return i < kTlsSlots ? teb_at<void*>(t_teb, kTebTlsSlots + 8 * i) : nullptr; }
int TlsSetValue(uint32_t i, void* v) {
    if (i >= kTlsSlots) return 0;
    teb_at<void*>(t_teb, kTebTlsSlots + 8 * i) = v;
    return 1;
}

HANDLE CreateThread(void*, uint64_t, ThreadProc proc, void* arg, uint32_t, uint32_t* tid) {
    Object* o = spawn_thread(proc, arg);
    if (!o) set_last_error(8);
    else if (tid) *tid = o->tid;
    return o;
}
uint32_t WaitForSingleObject(HANDLE h, uint32_t) {  // ponytail: threads only, INFINITE only
    auto* o = static_cast<Object*>(h);
    if (o->kind != Object::Thread) return 0xFFFFFFFFu;
    wait_thread(o);
    return 0;
}
int CloseHandle(HANDLE h) {
    close_object(static_cast<Object*>(h));
    return 1;
}
void Sleep(uint32_t ms) { usleep(ms * 1000u); }
int QueryPerformanceCounter(int64_t* out) {
    *out = static_cast<int64_t>(clock_gettime_nsec_np(CLOCK_UPTIME_RAW));
    return 1;
}
void GetSystemTimeAsFileTime(uint64_t* ft) {  // 100 ns ticks since 1601-01-01
    *ft = clock_gettime_nsec_np(CLOCK_REALTIME) / 100 + 116444736000000000ull;
}

HANDLE CreateFileW(const char16_t* name, uint32_t access, uint32_t, void*, uint32_t disposition, uint32_t, HANDLE) {
    std::string narrow;  // ponytail: ASCII names only
    for (; *name; ++name) narrow.push_back(static_cast<char>(*name));
    int mode = (access & 0x40000000) ? ((access & 0x80000000) ? O_RDWR : O_WRONLY) : O_RDONLY;
    if (disposition == 1) mode |= O_CREAT | O_EXCL;        // CREATE_NEW
    else if (disposition == 2) mode |= O_CREAT | O_TRUNC;  // CREATE_ALWAYS
    else if (disposition == 4) mode |= O_CREAT;            // OPEN_ALWAYS
    else if (disposition == 5) mode |= O_TRUNC;            // TRUNCATE_EXISTING
    int fd = open(to_host_path(narrow.c_str()).c_str(), mode, 0644);
    if (fd < 0) {
        set_last_error(2 /* ERROR_FILE_NOT_FOUND */);
        return kInvalid;
    }
    return new Object{Object::File, fd};
}

const char* GetCommandLineA() { return g_command_line.c_str(); }
HANDLE GetCurrentProcess() { return kInvalid; }
uint32_t GetCurrentProcessId() { return static_cast<uint32_t>(getpid()); }
uint32_t GetCurrentThreadId() { return thread_id(); }
HANDLE GetModuleHandleW(const char16_t* name) {
    if (name) unsupported("GetModuleHandleW with a module name");
    return g_image;
}
uint32_t GetTempPathA(uint32_t size, char* buf) {
    const char* tmp = "/tmp/";
    uint32_t n = static_cast<uint32_t>(std::strlen(tmp));
    if (size <= n) return n + 1;
    std::memcpy(buf, tmp, n + 1);
    return n;
}
int SetCurrentDirectoryA(const char* path) { return chdir(to_host_path(path).c_str()) == 0; }
void InitializeSListHead(void* head) { std::memset(head, 0, 16); }
int IsDebuggerPresent() { return 0; }
int IsProcessorFeaturePresent(uint32_t feature) { return feature == 23; }  // PF_FASTFAIL_AVAILABLE
void* SetUnhandledExceptionFilter(void*) { return nullptr; }

// The CRT reaches these only on a failure path (__report_gsfailure, unhandled exceptions).
int32_t UnhandledExceptionFilter(void*) { unsupported("UnhandledExceptionFilter (the guest crashed)"); }
void RtlCaptureContext(void*) { unsupported("RtlCaptureContext (the guest reported a failure)"); }
void* RtlLookupFunctionEntry(uint64_t, uint64_t*, void*) { unsupported("RtlLookupFunctionEntry"); }
void* RtlVirtualUnwind(uint32_t, uint64_t, uint64_t, void*, void*, void**, uint64_t*, void*) {
    unsupported("RtlVirtualUnwind");
}

}  // namespace

#define K(name) EXPORT("kernel32.dll", name)
Exports kernel32_exports() {
    return {
        K(GetStdHandle), K(WriteFile), K(ReadFile), K(ExitProcess), K(TerminateProcess), K(GetLastError),
        K(GetProcessHeap), K(HeapAlloc), K(HeapFree), K(VirtualAlloc), K(VirtualFree), K(TlsAlloc),
        K(TlsGetValue), K(TlsSetValue), K(CreateThread), K(WaitForSingleObject), K(CloseHandle), K(Sleep),
        K(QueryPerformanceCounter), K(GetSystemTimeAsFileTime), K(CreateFileW), K(GetCommandLineA),
        K(GetCurrentProcess), K(GetCurrentProcessId), K(GetCurrentThreadId), K(GetModuleHandleW),
        K(GetTempPathA), K(SetCurrentDirectoryA), K(InitializeSListHead), K(IsDebuggerPresent),
        K(IsProcessorFeaturePresent), K(SetUnhandledExceptionFilter), K(UnhandledExceptionFilter),
        K(RtlCaptureContext), K(RtlLookupFunctionEntry), K(RtlVirtualUnwind),
    };
}

}  // namespace host
