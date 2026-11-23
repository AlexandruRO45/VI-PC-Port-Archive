// kernel32 on POSIX: the subset guest/guest.c and Leonida's tiny_game import. Windows types are spelled with
// fixed widths (DWORD = uint32_t, LONG = int32_t, WCHAR = char16_t), since long and wchar_t differ on macOS.
#include "host.h"

#include <fcntl.h>
#include <mach/mach.h>
#include <mach/mach_vm.h>
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
int to_prot(uint32_t page) {
    if (page == 0x01) return PROT_NONE;  // PAGE_NOACCESS
    return PROT_READ | (page & 0xCC ? PROT_WRITE : 0) | (page & 0xF0 ? PROT_EXEC : 0);
}
// Reserved memory is mapped readable and writable at once, so committing inside a reservation is a no-op.
void* VirtualAlloc(void* addr, uint64_t size, uint32_t type, uint32_t page) {
    auto a = reinterpret_cast<uintptr_t>(addr);
    std::lock_guard<std::mutex> lock(g_vmem_lock);
    if (a) {
        auto it = g_vmem.upper_bound(a);
        if (it != g_vmem.begin() && a < std::prev(it)->first + std::prev(it)->second) {
            if (!(type & 0x1000 /* MEM_COMMIT */)) return set_last_error(487 /* ERROR_INVALID_ADDRESS */), nullptr;
            void* base = reinterpret_cast<void*>(a & ~uintptr_t(4095));
            mprotect(base, size + (a & 4095), to_prot(page) | PROT_READ | PROT_WRITE);
            return base;
        }
    }
    void* p = mmap(addr, size, to_prot(page) | PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
    if (p == MAP_FAILED) return set_last_error(8 /* ERROR_NOT_ENOUGH_MEMORY */), nullptr;
    if (a && p != addr) {  // Windows gives the address asked for or nothing
        munmap(p, size);
        return set_last_error(487 /* ERROR_INVALID_ADDRESS */), nullptr;
    }
    g_vmem[reinterpret_cast<uintptr_t>(p)] = size;
    return p;
}
uint64_t VirtualQuery(const void* addr, uint8_t* info, uint64_t len) {
    if (len < 48) return set_last_error(24 /* ERROR_BAD_LENGTH */), 0;
    mach_vm_address_t start = reinterpret_cast<uintptr_t>(addr) & ~uintptr_t(4095);
    mach_vm_address_t region = start;
    mach_vm_size_t size = 0;
    vm_region_basic_info_data_64_t vi{};
    mach_msg_type_number_t count = VM_REGION_BASIC_INFO_COUNT_64;
    mach_port_t object;
    kern_return_t kr = mach_vm_region(mach_task_self(), &region, &size, VM_REGION_BASIC_INFO_64,
                                      reinterpret_cast<vm_region_info_t>(&vi), &count, &object);
    std::memset(info, 0, 48);
    uint64_t base = start, alloc = 0, rsize;
    uint32_t state, protect = 0, type = 0;
    if (kr != KERN_SUCCESS || region > start) {  // free, up to the next mapping
        rsize = kr == KERN_SUCCESS ? region - start : 0x1000;
        state = 0x10000;  // MEM_FREE
        protect = 0x01;
    } else {
        rsize = region + size - start;
        state = 0x1000;  // MEM_COMMIT
        int p = vi.protection;
        protect = p == 0 ? 0x01 : (p & VM_PROT_EXECUTE) ? ((p & VM_PROT_WRITE) ? 0x40 : 0x20) : ((p & VM_PROT_WRITE) ? 0x04 : 0x02);
        void* m = module_containing(addr);
        alloc = m ? reinterpret_cast<uint64_t>(m) : region;
        type = m ? 0x1000000 /* MEM_IMAGE */ : 0x20000 /* MEM_PRIVATE */;
        std::lock_guard<std::mutex> lock(g_vmem_lock);
        auto it = g_vmem.upper_bound(start);
        if (!m && it != g_vmem.begin() && start < std::prev(it)->first + std::prev(it)->second) alloc = std::prev(it)->first;
    }
    std::memcpy(info, &base, 8);
    std::memcpy(info + 8, &alloc, 8);
    std::memcpy(info + 16, &protect, 4);
    std::memcpy(info + 24, &rsize, 8);
    std::memcpy(info + 32, &state, 4);
    std::memcpy(info + 36, &protect, 4);
    std::memcpy(info + 40, &type, 4);
    return 48;
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

}  // namespace

#define K(name) EXPORT("kernel32.dll", name)
Exports kernel32_exports() {
    return {
        K(GetStdHandle), K(WriteFile), K(ReadFile), K(ExitProcess), K(TerminateProcess), K(GetLastError),
        K(GetProcessHeap), K(HeapAlloc), K(HeapFree), K(VirtualAlloc), K(VirtualQuery), K(VirtualFree), K(TlsAlloc),
        K(TlsGetValue), K(TlsSetValue), K(CreateThread), K(CloseHandle), K(Sleep),
        K(QueryPerformanceCounter), K(GetSystemTimeAsFileTime), K(CreateFileW), K(GetCommandLineA),
        K(GetCurrentProcess), K(GetCurrentProcessId), K(GetCurrentThreadId),
        K(GetTempPathA), K(SetCurrentDirectoryA), K(InitializeSListHead), K(IsDebuggerPresent),
        K(IsProcessorFeaturePresent), K(SetUnhandledExceptionFilter), K(UnhandledExceptionFilter),
    };
}

}  // namespace host
