// Prototype: run a Windows x64 PE natively on macOS (an x86-64 Mach-O under Rosetta 2), without Wine.
// Maps the image, applies DIR64 relocations, binds imports to host functions with the Windows x64 calling
// convention, gives every guest thread a TEB reached through gs:, and swaps gs back to the pthread base while a
// host function runs. Proof of concept for #2947; the import table covers only guest/guest.c.
#include <mach/mach.h>
#include <pthread.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <time.h>
#include <unistd.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <mutex>
#include <string>
#include <type_traits>
#include <vector>

extern "C" void _thread_set_tsd_base(uint64_t);  // private libsystem_kernel call; Wine uses it the same way

#define MSABI __attribute__((ms_abi))

namespace {

// ---- TEB and gs ----------------------------------------------------------------------------------------------
constexpr size_t kTebSize = 0x2000;
constexpr size_t kTebStackBase = 0x08, kTebStackLimit = 0x10, kTebSelf = 0x30, kTebPid = 0x40, kTebTid = 0x48;
constexpr size_t kTebPeb = 0x60, kTebLastError = 0x68, kTebTlsSlots = 0x1480;
constexpr size_t kTebHostGs = 0x1F00;  // our stash, past the end of the documented x64 TEB (0x1838)
constexpr int kTlsSlots = 64;

uint8_t* g_peb = nullptr;
thread_local uint8_t* t_teb = nullptr;  // only read while gs holds the pthread base

uint64_t pthread_gsbase() {  // what libpthread set gs to for this thread (Wine: mac_thread_gsbase)
    thread_identifier_info_data_t info;
    mach_msg_type_number_t count = THREAD_IDENTIFIER_INFO_COUNT;
    mach_port_t self = mach_thread_self();
    kern_return_t kr = thread_info(self, THREAD_IDENTIFIER_INFO, reinterpret_cast<thread_info_t>(&info), &count);
    mach_port_deallocate(mach_task_self(), self);
    if (kr != KERN_SUCCESS) { std::fprintf(stderr, "host: thread_info failed (%d)\n", kr); std::abort(); }
    return info.thread_handle;
}

template <typename T> T& teb_at(uint8_t* teb, size_t off) { return *reinterpret_cast<T*>(teb + off); }

uint8_t* make_teb() {  // for the calling thread
    auto* teb = static_cast<uint8_t*>(aligned_alloc(0x1000, kTebSize));
    if (!teb) std::abort();
    std::memset(teb, 0, kTebSize);
    pthread_t self = pthread_self();
    auto* top = static_cast<uint8_t*>(pthread_get_stackaddr_np(self));
    teb_at<void*>(teb, kTebStackBase) = top;
    teb_at<void*>(teb, kTebStackLimit) = top - pthread_get_stacksize_np(self);
    teb_at<void*>(teb, kTebSelf) = teb;
    teb_at<uint64_t>(teb, kTebPid) = static_cast<uint64_t>(getpid());
    uint64_t tid = 0;
    pthread_threadid_np(nullptr, &tid);
    teb_at<uint64_t>(teb, kTebTid) = tid;
    teb_at<void*>(teb, kTebPeb) = g_peb;
    teb_at<uint64_t>(teb, kTebHostGs) = pthread_gsbase();
    t_teb = teb;
    return teb;
}

// Run guest code with gs on the thread's TEB.
template <typename R, typename... A> R call_guest(uint8_t* teb, R(MSABI* fn)(A...), A... args) {
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

// ---- Handles -------------------------------------------------------------------------------------------------
struct Object {
    enum Kind { File, Thread } kind;
    int fd = -1;
    pthread_t thread{};
    uint32_t exit_code = 0;
    bool joined = false;
};
using HANDLE = void*;
HANDLE const kInvalid = reinterpret_cast<HANDLE>(-1);
Object g_std[3] = {{Object::File, 0}, {Object::File, 1}, {Object::File, 2}};

void set_last_error(uint32_t e) { teb_at<uint32_t>(t_teb, kTebLastError) = e; }

// ---- kernel32 ------------------------------------------------------------------------------------------------
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
uint32_t GetLastError() { return teb_at<uint32_t>(t_teb, kTebLastError); }

HANDLE GetProcessHeap() { return reinterpret_cast<HANDLE>(0x1000); }  // one heap: malloc
void* HeapAlloc(HANDLE, uint32_t flags, uint64_t size) { return (flags & 0x8) ? calloc(1, size) : malloc(size); }
int HeapFree(HANDLE, uint32_t, void* p) { free(p); return 1; }

std::mutex g_vmem_lock;
std::map<uintptr_t, size_t> g_vmem;  // ponytail: whole reserve/commit/release only, no partial decommit
void* VirtualAlloc(void* addr, uint64_t size, uint32_t, uint32_t) {
    void* p = mmap(addr, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
    if (p == MAP_FAILED) { set_last_error(8 /* ERROR_NOT_ENOUGH_MEMORY */); return nullptr; }
    std::lock_guard<std::mutex> lock(g_vmem_lock);
    g_vmem[reinterpret_cast<uintptr_t>(p)] = size;
    return p;
}
int VirtualFree(void* addr, uint64_t, uint32_t type) {
    if (type != 0x8000 /* MEM_RELEASE */) return 1;
    std::lock_guard<std::mutex> lock(g_vmem_lock);
    auto it = g_vmem.find(reinterpret_cast<uintptr_t>(addr));
    if (it == g_vmem.end()) { set_last_error(87); return 0; }
    munmap(addr, it->second);
    g_vmem.erase(it);
    return 1;
}

std::mutex g_tls_lock;
bool g_tls_used[kTlsSlots] = {};
uint32_t TlsAlloc() {
    std::lock_guard<std::mutex> lock(g_tls_lock);
    for (int i = 0; i < kTlsSlots; ++i)
        if (!g_tls_used[i]) { g_tls_used[i] = true; return static_cast<uint32_t>(i); }
    return 0xFFFFFFFFu;
}
void* TlsGetValue(uint32_t i) { return i < kTlsSlots ? teb_at<void*>(t_teb, kTebTlsSlots + 8 * i) : nullptr; }
int TlsSetValue(uint32_t i, void* v) {
    if (i >= kTlsSlots) return 0;
    teb_at<void*>(t_teb, kTebTlsSlots + 8 * i) = v;
    return 1;
}

using ThreadProc = uint32_t(MSABI*)(void*);
struct ThreadStart { Object* obj; ThreadProc proc; void* arg; };
void* thread_main(void* p) {
    auto* s = static_cast<ThreadStart*>(p);
    uint8_t* teb = make_teb();
    s->obj->exit_code = call_guest(teb, s->proc, s->arg);
    delete s;
    t_teb = nullptr;
    free(teb);
    return nullptr;
}
HANDLE CreateThread(void*, uint64_t, ThreadProc proc, void* arg, uint32_t, uint32_t* tid) {
    auto* obj = new Object{Object::Thread};
    if (pthread_create(&obj->thread, nullptr, thread_main, new ThreadStart{obj, proc, arg}) != 0) {
        delete obj;
        set_last_error(8);
        return nullptr;
    }
    if (tid) {
        uint64_t id = 0;
        pthread_threadid_np(obj->thread, &id);
        *tid = static_cast<uint32_t>(id);
    }
    return obj;
}
uint32_t WaitForSingleObject(HANDLE h, uint32_t) {  // ponytail: threads only, INFINITE only
    auto* o = static_cast<Object*>(h);
    if (o->kind != Object::Thread) return 0xFFFFFFFFu;
    if (!o->joined) { pthread_join(o->thread, nullptr); o->joined = true; }
    return 0;
}
int CloseHandle(HANDLE h) {
    auto* o = static_cast<Object*>(h);
    if (o >= g_std && o < g_std + 3) return 1;
    if (o->kind == Object::File) close(o->fd);
    if (o->kind == Object::Thread && !o->joined) pthread_detach(o->thread);
    delete o;
    return 1;
}
void Sleep(uint32_t ms) { usleep(ms * 1000u); }
int QueryPerformanceCounter(int64_t* out) {
    *out = static_cast<int64_t>(clock_gettime_nsec_np(CLOCK_UPTIME_RAW));
    return 1;
}
HANDLE CreateFileW(const char16_t* name, uint32_t access, uint32_t, void*, uint32_t disposition, uint32_t, HANDLE) {
    std::string path;  // ponytail: ASCII names only
    for (; *name; ++name) path.push_back(*name == u'\\' ? '/' : static_cast<char>(*name));
    int mode = (access & 0x40000000) ? ((access & 0x80000000) ? O_RDWR : O_WRONLY) : O_RDONLY;
    if (disposition == 1) mode |= O_CREAT | O_EXCL;       // CREATE_NEW
    else if (disposition == 2) mode |= O_CREAT | O_TRUNC; // CREATE_ALWAYS
    else if (disposition == 4) mode |= O_CREAT;           // OPEN_ALWAYS
    else if (disposition == 5) mode |= O_TRUNC;           // TRUNCATE_EXISTING
    int fd = open(path.c_str(), mode, 0644);
    if (fd < 0) { set_last_error(2 /* ERROR_FILE_NOT_FOUND */); return kInvalid; }
    return new Object{Object::File, fd};
}

#define EXPORT(name) {"kernel32.dll!" #name, reinterpret_cast<void*>(&Thunk<decltype(name), &name>::call)}
const std::map<std::string, void*> kExports = {
    EXPORT(GetStdHandle), EXPORT(WriteFile), EXPORT(ReadFile), EXPORT(ExitProcess), EXPORT(GetLastError),
    EXPORT(GetProcessHeap), EXPORT(HeapAlloc), EXPORT(HeapFree), EXPORT(VirtualAlloc), EXPORT(VirtualFree),
    EXPORT(TlsAlloc), EXPORT(TlsGetValue), EXPORT(TlsSetValue), EXPORT(CreateThread),
    EXPORT(WaitForSingleObject), EXPORT(CloseHandle), EXPORT(Sleep), EXPORT(QueryPerformanceCounter),
    EXPORT(CreateFileW),
};

// ---- PE loading ----------------------------------------------------------------------------------------------
template <typename T> T rd(const uint8_t* p) { T v; std::memcpy(&v, p, sizeof v); return v; }

[[noreturn]] void fail(const std::string& why) {
    std::fprintf(stderr, "host: %s\n", why.c_str());
    std::exit(70);
}

uint8_t* load_pe(const char* path) {
    std::ifstream f(path, std::ios::binary);
    std::vector<uint8_t> file((std::istreambuf_iterator<char>(f)), {});
    if (file.size() < 0x40 || file[0] != 'M' || file[1] != 'Z') fail("not a PE file (no MZ header)");
    uint32_t pe = rd<uint32_t>(&file[0x3c]);
    if (pe + 0x108 > file.size() || std::memcmp(&file[pe], "PE\0\0", 4) != 0) fail("no PE signature");
    const uint8_t* coff = &file[pe + 4];
    if (rd<uint16_t>(coff) != 0x8664) fail("not an x64 image");
    uint16_t nsec = rd<uint16_t>(coff + 2), optsz = rd<uint16_t>(coff + 16);
    const uint8_t* opt = coff + 20;
    if (rd<uint16_t>(opt) != 0x20b) fail("not PE32+");
    uint32_t entry_rva = rd<uint32_t>(opt + 16);
    uint64_t base = rd<uint64_t>(opt + 24);
    uint32_t image_size = rd<uint32_t>(opt + 56), header_size = rd<uint32_t>(opt + 60);
    auto dir = [&](int i) { return std::make_pair(rd<uint32_t>(opt + 112 + 8 * i), rd<uint32_t>(opt + 116 + 8 * i)); };
    if (dir(9).second) fail("TLS directory not supported yet");

    auto* image = static_cast<uint8_t*>(
        mmap(reinterpret_cast<void*>(base), image_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0));
    if (image == MAP_FAILED) fail("cannot map the image");
    std::memcpy(image, file.data(), header_size);
    const uint8_t* sec = opt + optsz;
    for (int i = 0; i < nsec; ++i, sec += 40) {
        uint32_t va = rd<uint32_t>(sec + 12), raw_size = rd<uint32_t>(sec + 16), raw = rd<uint32_t>(sec + 20);
        if (raw_size && raw + raw_size <= file.size()) std::memcpy(image + va, &file[raw], raw_size);
    }

    int64_t delta = reinterpret_cast<int64_t>(image) - static_cast<int64_t>(base);
    if (delta) {
        auto [rva, size] = dir(5);
        if (rd<uint16_t>(coff + 18) & 0x1) fail("image wants its exact base (relocations stripped)");
        for (uint32_t off = 0; off < size;) {
            uint32_t page = rd<uint32_t>(image + rva + off), block = rd<uint32_t>(image + rva + off + 4);
            if (block < 8) break;
            for (uint32_t e = 8; e < block; e += 2) {
                uint16_t ent = rd<uint16_t>(image + rva + off + e);
                if ((ent >> 12) == 10) *reinterpret_cast<int64_t*>(image + page + (ent & 0xfff)) += delta;
                else if ((ent >> 12) != 0) fail("unsupported relocation type");
            }
            off += block;
        }
    }

    std::vector<std::string> missing;
    for (uint32_t d = dir(1).first; d && rd<uint32_t>(image + d + 12); d += 20) {
        std::string dll = reinterpret_cast<char*>(image + rd<uint32_t>(image + d + 12));
        for (auto& c : dll) c = static_cast<char>(std::tolower(c));
        uint32_t lookup = rd<uint32_t>(image + d) ? rd<uint32_t>(image + d) : rd<uint32_t>(image + d + 16);
        auto* iat = reinterpret_cast<uint64_t*>(image + rd<uint32_t>(image + d + 16));
        for (auto* t = reinterpret_cast<uint64_t*>(image + lookup); *t; ++t, ++iat) {
            if (*t >> 63) { missing.push_back(dll + "!#" + std::to_string(*t & 0xffff)); continue; }
            std::string key = dll + "!" + reinterpret_cast<char*>(image + (*t & 0x7fffffff) + 2);
            auto it = kExports.find(key);
            if (it == kExports.end()) missing.push_back(key);
            else *iat = reinterpret_cast<uint64_t>(it->second);
        }
    }
    if (!missing.empty()) {
        for (auto& m : missing) std::fprintf(stderr, "host: unresolved import %s\n", m.c_str());
        fail("unresolved imports");
    }

    sec = opt + optsz;
    for (int i = 0; i < nsec; ++i, sec += 40) {
        uint32_t va = rd<uint32_t>(sec + 12), vsize = rd<uint32_t>(sec + 8), ch = rd<uint32_t>(sec + 36);
        int prot = ((ch & 0x40000000) ? PROT_READ : 0) | ((ch & 0x80000000) ? PROT_WRITE : 0) |
                   ((ch & 0x20000000) ? PROT_EXEC : 0);
        size_t len = (vsize + 0xfff) & ~size_t{0xfff};
        if (len && mprotect(image + va, len, prot) != 0) fail("mprotect failed");
    }
    std::fprintf(stderr, "host: mapped %s at %p (preferred %#llx)\n", path, static_cast<void*>(image),
                 static_cast<unsigned long long>(base));
    return image + entry_rva;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: macos_host <image.exe>\n"); return 64; }
    g_peb = static_cast<uint8_t*>(calloc(1, 0x1000));
    void* entry = load_pe(argv[1]);
    uint8_t* teb = make_teb();
    call_guest(teb, reinterpret_cast<void(MSABI*)()>(entry));
    std::fprintf(stderr, "host: entry point returned\n");
    return 0;
}
