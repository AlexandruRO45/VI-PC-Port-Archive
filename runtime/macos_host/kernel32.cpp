// kernel32 on POSIX: the subset guest/guest.c and Leonida's tiny_game import. Windows types are spelled with
// fixed widths (DWORD = uint32_t, LONG = int32_t, WCHAR = char16_t), since long and wchar_t differ on macOS.
#include "host.h"

#include <fcntl.h>
#include <mach/mach.h>
#include <mach/mach_vm.h>
#include <sys/mman.h>
#include <sys/sysctl.h>
#include <time.h>
#include <unistd.h>

#include <cstdio>
#include <cstring>
#include <atomic>
#include <map>
#include <mutex>
#include <string>

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
// An OVERLAPPED (Internal, InternalHigh, Offset, OffsetHigh, hEvent) names the file offset; the transfer is done
// synchronously, so the call completes at once with the status and count filled in and the event set (the low bit of hEvent only asks for no completion port).
// ponytail: no real async I/O; a reader that waits on the event or GetOverlappedResult sees it already done.
static ssize_t transfer(HANDLE h, void* buf, uint32_t n, uint8_t* ov, bool write_it) {
    const int fd = static_cast<Object*>(h)->fd;
    if (!ov) return write_it ? write(fd, buf, n) : read(fd, buf, n);
    const off_t at = static_cast<off_t>(*reinterpret_cast<uint32_t*>(ov + 16) |
                                        uint64_t{*reinterpret_cast<uint32_t*>(ov + 20)} << 32);
    const ssize_t r = write_it ? pwrite(fd, buf, n, at) : pread(fd, buf, n, at);
    if (r >= 0) lseek(fd, at + r, SEEK_SET);  // a synchronous handle moves its position past the transfer
    *reinterpret_cast<uint64_t*>(ov) = r < 0 ? 0xC0000185 /* STATUS_IO_DEVICE_ERROR */ : 0;
    *reinterpret_cast<uint64_t*>(ov + 8) = r < 0 ? 0 : static_cast<uint64_t>(r);
    if (HANDLE e = *reinterpret_cast<HANDLE*>(ov + 24)) set_signal(reinterpret_cast<HANDLE>(reinterpret_cast<uintptr_t>(e) & ~uintptr_t{1}));
    return r;
}
int WriteFile(HANDLE h, const void* buf, uint32_t n, uint32_t* written, void* ov) {
    ssize_t r = transfer(h, const_cast<void*>(buf), n, static_cast<uint8_t*>(ov), true);
    if (written) *written = r < 0 ? 0 : static_cast<uint32_t>(r);
    return r >= 0;
}
int ReadFile(HANDLE h, void* buf, uint32_t n, uint32_t* got, void* ov) {
    ssize_t r = transfer(h, buf, n, static_cast<uint8_t*>(ov), false);
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
std::map<uintptr_t, size_t> g_place;  // placeholders (VirtualAlloc2's MEM_RESERVE_PLACEHOLDER): PROT_NONE, base -> size
std::map<uintptr_t, size_t> g_views;  // views of a section (MapViewOfFile3): base -> size
constexpr uint32_t kMemReplacePlaceholder = 0x4000, kMemReservePlaceholder = 0x40000, kMemPreservePlaceholder = 0x2;
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
    std::unique_lock<std::mutex> place_lock(g_vmem_lock);
    auto pl = g_place.upper_bound(start);
    bool placeholder = pl != g_place.begin() && start < std::prev(pl)->first + std::prev(pl)->second;
    if (placeholder) --pl;
    place_lock.unlock();
    if (placeholder) {  // MEM_RESERVE, up to the end of the placeholder
        alloc = pl->first;
        rsize = pl->first + pl->second - start;
        state = 0x2000;
        type = 0x20000;  // MEM_PRIVATE
        std::memcpy(info + 8, &alloc, 8);
        uint32_t noaccess = 0x01;
        std::memcpy(info + 16, &noaccess, 4);
        std::memcpy(info, &base, 8);
        std::memcpy(info + 24, &rsize, 8);
        std::memcpy(info + 32, &state, 4);
        std::memcpy(info + 40, &type, 4);
        return 48;
    }
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
// Placeholders on mmap: a PROT_NONE anonymous mapping recorded in g_place. Splitting one is bookkeeping only (Mach
// regions split on their own), and replacing part of one maps over it with MAP_FIXED.
void* map_aligned(uint64_t size, uint64_t align, int prot) {  // anywhere, base a multiple of align
    if (align <= 0x10000) align = 0x10000;  // Windows' allocation granularity
    void* p = mmap(nullptr, size + align, prot, MAP_PRIVATE | MAP_ANON, -1, 0);
    if (p == MAP_FAILED) return nullptr;
    uintptr_t a = reinterpret_cast<uintptr_t>(p), base = (a + align - 1) & ~(align - 1);
    if (base > a) munmap(p, base - a);
    if (a + align > base) munmap(reinterpret_cast<void*>(base + size), a + align - base);
    return reinterpret_cast<void*>(base);
}
// With g_vmem_lock held: [a, a + size) cut out of the placeholder holding it, which keeps what is left either side.
bool take_placeholder(uintptr_t a, uint64_t size) {
    auto it = g_place.upper_bound(a);
    if (it == g_place.begin()) return false;
    --it;
    uintptr_t base = it->first, end = base + it->second;
    if (a + size > end) return false;
    g_place.erase(it);
    if (a > base) g_place[base] = a - base;
    if (a + size < end) g_place[a + size] = end - (a + size);
    return true;
}
// With g_vmem_lock held: [a, a + size) back to a placeholder of its own.
void restore_placeholder(uintptr_t a, uint64_t size) {
    mmap(reinterpret_cast<void*>(a), size, PROT_NONE, MAP_PRIVATE | MAP_ANON | MAP_FIXED, -1, 0);
    g_place[a] = size;
}
uint64_t alignment_of(const uint64_t* params, uint32_t count) {  // MEM_EXTENDED_PARAMETER: {Type:8 | ..., value}
    for (uint32_t i = 0; params && i < count; ++i)
        if ((params[2 * i] & 0xFF) == 1 /* MemExtendedParameterAddressRequirements */ && params[2 * i + 1])
            return reinterpret_cast<const uint64_t*>(params[2 * i + 1])[2];  // MEM_ADDRESS_REQUIREMENTS.Alignment
    return 0;
}
void* VirtualAlloc2(HANDLE, void* addr, uint64_t size, uint32_t type, uint32_t page, const uint64_t* params,
                    uint32_t count) {
    auto a = reinterpret_cast<uintptr_t>(addr);
    std::lock_guard<std::mutex> lock(g_vmem_lock);
    if (type & kMemReplacePlaceholder) {
        if (!a || !take_placeholder(a, size)) return set_last_error(487 /* ERROR_INVALID_ADDRESS */), nullptr;
        mmap(addr, size, to_prot(page) | PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON | MAP_FIXED, -1, 0);
        g_vmem[a] = size;
        return addr;
    }
    if (a) return set_last_error(87 /* ponytail: placeholders at a given address */), nullptr;
    bool placeholder = type & kMemReservePlaceholder;
    void* p = map_aligned(size, alignment_of(params, count), placeholder ? PROT_NONE : PROT_READ | PROT_WRITE);
    if (!p) return set_last_error(8 /* ERROR_NOT_ENOUGH_MEMORY */), nullptr;
    (placeholder ? g_place : g_vmem)[reinterpret_cast<uintptr_t>(p)] = size;
    return p;
}

// A section of anonymous memory (CreateFileMappingW on INVALID_HANDLE_VALUE) is a shared-memory object's fd.
HANDLE CreateFileMappingW(HANDLE file, void*, uint32_t, uint32_t high, uint32_t low, const char16_t* name) {
    if (file != kInvalid || name) unsupported("CreateFileMappingW on a file or with a name");
    static std::atomic<int> serial{0};
    std::string shm = "/leonida." + std::to_string(getpid()) + "." + std::to_string(serial++);
    int fd = shm_open(shm.c_str(), O_RDWR | O_CREAT | O_EXCL, 0600);
    if (fd < 0) return set_last_error(8), nullptr;
    shm_unlink(shm.c_str());
    if (ftruncate(fd, static_cast<off_t>((uint64_t(high) << 32) | low)) != 0) {
        close(fd);
        return set_last_error(8), nullptr;
    }
    return new Object{Object::File, fd};
}
void* MapViewOfFile3(HANDLE section, HANDLE, void* addr, uint64_t offset, uint64_t size, uint32_t type,
                     uint32_t page, const uint64_t*, uint32_t) {
    auto a = reinterpret_cast<uintptr_t>(addr);
    std::lock_guard<std::mutex> lock(g_vmem_lock);
    if ((type & kMemReplacePlaceholder) && !take_placeholder(a, size)) return set_last_error(487), nullptr;
    int flags = MAP_SHARED | (a ? MAP_FIXED : 0);
    void* p = mmap(addr, size, to_prot(page), flags, static_cast<Object*>(section)->fd, static_cast<off_t>(offset));
    if (p == MAP_FAILED) return set_last_error(8), nullptr;
    g_views[reinterpret_cast<uintptr_t>(p)] = size;
    return p;
}
int UnmapViewOfFile2(HANDLE, void* addr, uint32_t flags) {
    auto a = reinterpret_cast<uintptr_t>(addr);
    std::lock_guard<std::mutex> lock(g_vmem_lock);
    auto it = g_views.find(a);
    if (it == g_views.end()) return set_last_error(487), 0;
    if (flags & kMemPreservePlaceholder) restore_placeholder(a, it->second);
    else munmap(addr, it->second);
    g_views.erase(it);
    return 1;
}

int VirtualFree(void* addr, uint64_t size, uint32_t type) {
    if (!(type & 0x8000) /* MEM_RELEASE */) return 1;
    std::lock_guard<std::mutex> lock(g_vmem_lock);
    auto a = reinterpret_cast<uintptr_t>(addr);
    if (type & kMemPreservePlaceholder) {  // split a placeholder, or turn a replaced range back into one
        if (take_placeholder(a, size)) return g_place[a] = size, 1;
        auto v = g_vmem.find(a);
        if (v == g_vmem.end() || v->second != size) return set_last_error(87), 0;
        g_vmem.erase(v);
        return restore_placeholder(a, size), 1;
    }
    if (auto p = g_place.find(a); p != g_place.end()) {
        munmap(addr, p->second);
        g_place.erase(p);
        return 1;
    }
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
// One RelationProcessorCore entry per core (32 bytes: mask, relationship, then the union), then the package.
int GetLogicalProcessorInformation(uint8_t* buf, uint32_t* len) {
    int cores = 0, logical = 0;
    size_t n = sizeof(int);
    sysctlbyname("hw.physicalcpu", &cores, &n, nullptr, 0);
    sysctlbyname("hw.logicalcpu", &logical, &n, nullptr, 0);
    if (cores < 1 || cores > 64 || logical < cores || logical > 64) cores = logical = 1;
    uint32_t need = 32u * (cores + 1);
    if (!buf || *len < need) return *len = need, set_last_error(122 /* ERROR_INSUFFICIENT_BUFFER */), 0;
    std::memset(buf, 0, need);
    int per = logical / cores;
    for (int c = 0; c <= cores; ++c) {
        uint64_t mask = c < cores ? ((per == 64 ? ~0ull : (1ull << per) - 1) << (c * per))
                                  : (logical == 64 ? ~0ull : (1ull << logical) - 1);
        std::memcpy(buf + 32 * c, &mask, 8);
        buf[32 * c + 8] = c < cores ? 0 : 3;  // RelationProcessorCore, RelationProcessorPackage
        buf[32 * c + 16] = c < cores && per > 1;  // ProcessorCore.Flags: LTP_PC_SMT
    }
    *len = need;
    return 1;
}

// The CRT reaches these only on a failure path (__report_gsfailure, unhandled exceptions).
int32_t UnhandledExceptionFilter(void*) { unsupported("UnhandledExceptionFilter (the guest crashed)"); }

}  // namespace

#define K(name) EXPORT("kernel32.dll", name)
Exports kernel32_exports() {
    return {
        K(GetStdHandle), K(WriteFile), K(ReadFile), K(ExitProcess), K(TerminateProcess), K(GetLastError),
        K(GetProcessHeap), K(HeapAlloc), K(HeapFree), K(VirtualAlloc), K(VirtualQuery), K(VirtualFree), K(TlsAlloc),
        K(TlsGetValue), K(TlsSetValue), K(CreateThread), K(CloseHandle), K(Sleep),
        K(QueryPerformanceCounter), K(GetSystemTimeAsFileTime), EXPORT_AS("kernel32.dll", "GetSystemTimePreciseAsFileTime", GetSystemTimeAsFileTime), K(CreateFileW), K(GetCommandLineA),
        K(GetCurrentProcess), K(GetCurrentProcessId), K(GetCurrentThreadId),
        K(GetTempPathA), K(SetCurrentDirectoryA), K(InitializeSListHead), K(IsDebuggerPresent),
        K(IsProcessorFeaturePresent), K(SetUnhandledExceptionFilter), K(UnhandledExceptionFilter),
        K(CreateFileMappingW), K(GetLogicalProcessorInformation), EXPORT("kernelbase.dll", VirtualAlloc2), EXPORT("kernelbase.dll", MapViewOfFile3),
        EXPORT("kernelbase.dll", UnmapViewOfFile2),
    };
}

}  // namespace host
