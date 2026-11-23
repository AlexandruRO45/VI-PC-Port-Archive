// Prototype: run a Windows x64 PE natively on macOS (an x86-64 Mach-O under Rosetta 2), without Wine.
// Maps the image, applies DIR64 relocations, binds imports to host functions with the Windows x64 calling
// convention, gives every guest thread a TEB (with static TLS) reached through gs:, and swaps gs back to the
// pthread base while a host function runs. Proof of concept for #2947.
//
//   macos_host <image.exe> [guest arguments...]
#include "host.h"

#include <dlfcn.h>
#include <mach-o/dyld.h>
#include <limits.h>
#include <signal.h>
#include <strings.h>

#include <mach/mach.h>
#include <mach/mach_vm.h>
#include <sys/mman.h>
#include <unistd.h>

bool fix_illegal(ucontext_t* uc);  // illegal.cpp

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <mutex>
#include <vector>

namespace host {

thread_local uint8_t* t_teb = nullptr;
uint8_t* g_image = nullptr;
std::string g_command_line;
int g_argc = 0;
char** g_argv = nullptr;

namespace {

uint8_t* g_peb = nullptr;
std::map<std::string, void*> g_exports;

struct Module {  // a mapped PE: the image, or a DLL loaded from disk
    uint8_t* base = nullptr;
    uint8_t* entry = nullptr;
    uint32_t size = 0;
    std::string path;
    const uint8_t* tls_start = nullptr;  // IMAGE_TLS_DIRECTORY64, resolved
    size_t tls_raw = 0, tls_zero = 0;
    int tls_index = -1;                  // the module TLS index, -1 without TLS
    std::vector<void(MSABI*)(void*, uint32_t, void*)> callbacks;
    int32_t(MSABI* dll_main)(void*, uint32_t, void*) = nullptr;
};
constexpr int kMaxTlsModules = 16;
std::mutex g_mod_lock;        // guards g_mods, g_tls_count and g_tebs
std::vector<Module*> g_mods;  // load order; [0] is the image
int g_tls_count = 0;

uint8_t* tls_block(const Module& m) {
    auto* b = static_cast<uint8_t*>(calloc(1, m.tls_raw + m.tls_zero + 16));
    std::memcpy(b, m.tls_start, m.tls_raw);
    return b;
}

template <typename T> T rd(const uint8_t* p) {
    T v;
    std::memcpy(&v, p, sizeof v);
    return v;
}

uint64_t pthread_gsbase() {  // what libpthread set gs to for this thread (Wine: mac_thread_gsbase)
    thread_identifier_info_data_t info;
    mach_msg_type_number_t count = THREAD_IDENTIFIER_INFO_COUNT;
    mach_port_t self = mach_thread_self();
    kern_return_t kr = thread_info(self, THREAD_IDENTIFIER_INFO, reinterpret_cast<thread_info_t>(&info), &count);
    mach_port_deallocate(mach_task_self(), self);
    if (kr != KERN_SUCCESS) fail("thread_info failed");
    return info.thread_handle;
}

}  // namespace

[[noreturn]] void fail(const std::string& why) {
    std::fprintf(stderr, "host: %s\n", why.c_str());
    std::fflush(nullptr);
    _exit(70);
}

void set_last_error(uint32_t e) { teb_at<uint32_t>(t_teb, kTebLastError) = e; }

uint32_t thread_id() {
    uint64_t tid = 0;
    pthread_threadid_np(nullptr, &tid);
    return static_cast<uint32_t>(tid);
}

std::string g_game_dir;  // absolute folder of the game image, where GAME:\ points
std::string g_exe_path;

// API sets are names for functions that live in one real DLL: api-ms-win-core-* in kernel32, api-ms-win-crt-* in
// the UCRT, whose functions the host registers under whichever api-ms-win-crt-* set first needed them.
namespace {
bool starts(const std::string& s, const char* p) { return s.compare(0, std::strlen(p), p) == 0; }
std::vector<std::string> api_set_hosts(const std::string& dll) {
    if (starts(dll, "api-ms-win-core-") || dll == "kernelbase.dll") return {"kernel32.dll"};
    if (!starts(dll, "api-ms-win-crt-") && dll != "ucrtbase.dll") return {};
    std::vector<std::string> crt;
    for (auto it = g_exports.lower_bound("api-ms-win-crt-"); it != g_exports.end() && starts(it->first, "api-ms-win-crt-");
         ++it)
        if (std::string d = it->first.substr(0, it->first.find('!')); crt.empty() || crt.back() != d) crt.push_back(d);
    return crt;
}
bool has_exact(const std::string& dll) {
    auto it = g_exports.lower_bound(dll + "!");
    return it != g_exports.end() && it->first.compare(0, dll.size() + 1, dll + "!") == 0;
}
}  // namespace

void* find_export(const std::string& key) {
    if (auto it = g_exports.find(key); it != g_exports.end()) return it->second;
    size_t bang = key.find('!');
    if (bang == std::string::npos) return nullptr;
    for (const auto& dll : api_set_hosts(key.substr(0, bang)))
        if (auto it = g_exports.find(dll + key.substr(bang)); it != g_exports.end()) return it->second;
    return nullptr;
}
bool has_dll(const std::string& dll) {
    if (has_exact(dll)) return true;
    for (const auto& d : api_set_hosts(dll))
        if (has_exact(d)) return true;
    return false;
}

std::string to_host_path(const char* p) {  // ponytail: no drive letters
    std::string s(p);
    if (strncasecmp(p, "GAME:", 5) == 0) s = g_game_dir + "/" + s.substr(s.size() > 5 && (p[5] == '\\' || p[5] == '/') ? 6 : 5);
    for (auto& c : s)
        if (c == '\\') c = '/';
    return s;
}

// Every TEB made (freed ones are cleared), so the fault handler can tell whether gs points at one without touching
// memory, and a DLL loaded later can give each live thread its TLS block.
// ponytail: fixed 1024 slots; threads past that get no late DLL TLS and fault-report with whatever gs they had.
uint8_t* g_tebs[1024];
std::atomic<int> g_teb_count{0};

uint8_t* make_teb() {
    auto* teb = static_cast<uint8_t*>(aligned_alloc(0x1000, kTebSize));
    if (!teb) fail("out of memory");
    std::memset(teb, 0, kTebSize);
    pthread_t self = pthread_self();
    auto* top = static_cast<uint8_t*>(pthread_get_stackaddr_np(self));
    teb_at<void*>(teb, kTebStackBase) = top;
    teb_at<void*>(teb, kTebStackLimit) = top - pthread_get_stacksize_np(self);
    teb_at<void*>(teb, kTebSelf) = teb;
    teb_at<uint64_t>(teb, kTebPid) = static_cast<uint64_t>(getpid());
    teb_at<uint64_t>(teb, kTebTid) = thread_id();
    teb_at<void*>(teb, kTebPeb) = g_peb;
    teb_at<uint64_t>(teb, kTebHostGs) = pthread_gsbase();
    auto** slots = static_cast<uint8_t**>(calloc(kMaxTlsModules, sizeof(void*)));  // MSVC reads gs:[0x58][_tls_index]
    teb_at<uint8_t**>(teb, kTebTlsPointer) = slots;
    {
        std::lock_guard<std::mutex> lock(g_mod_lock);
        for (auto* m : g_mods)
            if (m->tls_index >= 0) slots[m->tls_index] = tls_block(*m);
        if (int i = g_teb_count.fetch_add(1); i < 1024) g_tebs[i] = teb;
    }
    t_teb = teb;
    return teb;
}

// TLS callbacks and DllMain of every module, in load order (DLL_PROCESS_ATTACH, DLL_THREAD_ATTACH or _DETACH).
void notify_module(const Module& m, uint32_t reason) {
    for (auto cb : m.callbacks) call_guest(cb, static_cast<void*>(m.base), reason, static_cast<void*>(nullptr));
    if (m.dll_main) call_guest(m.dll_main, static_cast<void*>(m.base), reason, static_cast<void*>(nullptr));
}
void attach_thread(uint8_t*, uint32_t reason) {
    std::vector<Module*> mods;
    {
        std::lock_guard<std::mutex> lock(g_mod_lock);
        mods = g_mods;
    }
    for (auto* m : mods) notify_module(*m, reason);
}

void free_teb(uint8_t* teb) {
    {
        std::lock_guard<std::mutex> lock(g_mod_lock);
        for (int i = 0, n = std::min(g_teb_count.load(), 1024); i < n; ++i)
            if (g_tebs[i] == teb) g_tebs[i] = nullptr;
    }
    if (auto** slots = teb_at<uint8_t**>(teb, kTebTlsPointer)) {
        for (int i = 0; i < kMaxTlsModules; ++i) free(slots[i]);
        free(slots);
    }
    if (t_teb == teb) t_teb = nullptr;
    free(teb);
}

// ---- crashes -------------------------------------------------------------------------------------------------
// A fault prints where it happened, image-relative when it is in the game image, so llvm-symbolizer or
// llvm-objdump can name the function. Async-signal-unsafe printing is fine: the process is about to die.
namespace {
void on_fault(int sig, siginfo_t* info, void* uc) {
    auto& ss = static_cast<ucontext_t*>(uc)->uc_mcontext->__ss;
    uint64_t rip = ss.__rip, base = reinterpret_cast<uint64_t>(g_image);
    uint8_t* teb;  // a fault in guest code, or in host code called without HostGs: libc needs its own gs back
    __asm__ volatile("movq %%gs:0x30, %0" : "=r"(teb));
    bool guest = false;
    for (int i = 0, n = std::min(g_teb_count.load(), 1024); i < n; ++i)
        if (g_tebs[i] == teb) guest = true, _thread_set_tsd_base(teb_at<uint64_t>(teb, kTebHostGs));
    if (sig == SIGILL && ::fix_illegal(static_cast<ucontext_t*>(uc))) {  // illegal.cpp: patched, run on
        if (guest) _thread_set_tsd_base(reinterpret_cast<uint64_t>(teb));
        return;
    }
    std::fprintf(stderr, "host: signal %d at rip=0x%llx", sig, rip);
    if (g_image && rip >= base && rip < base + 0x10000000) std::fprintf(stderr, " (image+0x%llx)", rip - base);
    std::fprintf(stderr, " addr=%p rsp=0x%llx [rsp]=0x%llx\n", info->si_addr, ss.__rsp,
                 *reinterpret_cast<uint64_t*>(ss.__rsp));
    std::fprintf(stderr, "host:   rax=%llx rbx=%llx rcx=%llx rdx=%llx rsi=%llx rdi=%llx rbp=%llx\n", ss.__rax, ss.__rbx,
                 ss.__rcx, ss.__rdx, ss.__rsi, ss.__rdi, ss.__rbp);
    std::fprintf(stderr, "host:   r8=%llx r9=%llx r10=%llx r11=%llx r12=%llx r13=%llx r14=%llx r15=%llx\n", ss.__r8,
                 ss.__r9, ss.__r10, ss.__r11, ss.__r12, ss.__r13, ss.__r14, ss.__r15);
    // The first words each pointer register points at, when readable: the object a failed check was looking at.
    const char* names[] = {"rbx", "rcx", "rsi", "rdi"};
    uint64_t regs[] = {ss.__rbx, ss.__rcx, ss.__rsi, ss.__rdi};
    for (int r = 0; r < 4; ++r) {
        uint64_t w[4];
        mach_vm_size_t got = 0;
        if (regs[r] >= 0x10000 && mach_vm_read_overwrite(mach_task_self(), regs[r], sizeof w,
                                                         reinterpret_cast<mach_vm_address_t>(w), &got) == KERN_SUCCESS)
            std::fprintf(stderr, "host:   [%s] %016llx %016llx %016llx %016llx\n", names[r], w[0], w[1], w[2], w[3]);
    }
    Dl_info dl;
    for (uint64_t pc : {rip, *reinterpret_cast<uint64_t*>(ss.__rsp)})
        if (pc && dladdr(reinterpret_cast<void*>(pc), &dl) && dl.dli_sname)
            std::fprintf(stderr, "host:   0x%llx = %s+0x%llx\n", pc, dl.dli_sname,
                         pc - reinterpret_cast<uint64_t>(dl.dli_saddr));
    // Return addresses on the stack, for a backtrace without unwinding: any word that points into a mapped module.
    auto* sp = reinterpret_cast<uint64_t*>(ss.__rsp);
    for (int i = 0, shown = 0; i < 512 && shown < 12; ++i)
        if (uint64_t m = code_base(sp[i]); m && sp[i] != m)  // modules, and images registered with RtlAddFunctionTable
            std::fprintf(stderr, "host:   stack[%d] 0x%llx (module 0x%llx +0x%llx)\n", i, sp[i], m, sp[i] - m), ++shown;
    _exit(128 + sig);
}
}  // namespace

// ---- unimplemented imports --------------------------------------------------------------------------------
// Each missing import gets a stub `mov rcx, name; mov rax, trap; jmp rax`, so the image loads and only a call that
// actually happens stops the run, naming the function. LEONIDA_STRICT_IMPORTS=1 restores the load-time check.
namespace {
[[noreturn]] MSABI void trap(const char* name) {
    HostGs gs;
    fail(std::string("unimplemented import called: ") + name);
}
void* import_trap(const std::string& name) {
    static uint8_t* page = nullptr;
    static size_t used = 0;
    constexpr size_t kPage = 1 << 16, kStub = 24;
    if (!page || used + kStub > kPage) {
        page = static_cast<uint8_t*>(mmap(nullptr, kPage, PROT_READ | PROT_WRITE | PROT_EXEC, MAP_PRIVATE | MAP_ANON, -1, 0));
        if (page == MAP_FAILED) fail("mmap import traps");
        used = 0;
    }
    uint8_t* p = page + used;
    used += kStub;
    uint64_t n = reinterpret_cast<uint64_t>(strdup(name.c_str())), t = reinterpret_cast<uint64_t>(&trap);
    p[0] = 0x48, p[1] = 0xB9, std::memcpy(p + 2, &n, 8);   // mov rcx, imm64
    p[10] = 0x48, p[11] = 0xB8, std::memcpy(p + 12, &t, 8); // mov rax, imm64
    p[20] = 0xFF, p[21] = 0xE0;                             // jmp rax
    return p;
}
}  // namespace

// ---- threads -------------------------------------------------------------------------------------------------
namespace {
struct ThreadStart {
    Object* obj;
    ThreadProc proc;
    void* arg;
    pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;
    pthread_cond_t ready = PTHREAD_COND_INITIALIZER;
    bool started = false;
};
thread_local Object* t_self = nullptr;  // the guest thread's own handle, for ExitThread
void* thread_main(void* p) {
    auto* s = static_cast<ThreadStart*>(p);
    Object* obj = s->obj;
    ThreadProc proc = s->proc;
    void* arg = s->arg;
    obj->tid = thread_id();
    t_self = obj;
    pthread_mutex_lock(&s->lock);
    s->started = true;
    pthread_cond_signal(&s->ready);
    pthread_mutex_unlock(&s->lock);  // s belongs to the creator from here on
    uint8_t* teb = make_teb();
    attach_thread(teb, 2 /* DLL_THREAD_ATTACH */);
    obj->exit_code = call_guest(proc, arg);
    attach_thread(teb, 3 /* DLL_THREAD_DETACH */);
    free_teb(teb);
    obj->finished = true;
    return nullptr;
}
}  // namespace

// ExitThread from guest code: the thread's guest frames are abandoned, as on Windows.
[[noreturn]] void exit_thread(uint32_t code) {
    if (!t_self) {  // the main thread
        std::fflush(nullptr);
        _exit(static_cast<int>(code));
    }
    t_self->exit_code = code;
    uint8_t* teb = t_teb;
    attach_thread(teb, 3 /* DLL_THREAD_DETACH */);
    free_teb(teb);
    t_self->finished = true;
    pthread_exit(nullptr);
}
namespace {
}  // namespace

Object* spawn_thread(ThreadProc proc, void* arg) {
    auto* obj = new Object{Object::Thread};
    ThreadStart start{obj, proc, arg};
    if (pthread_create(&obj->thread, nullptr, thread_main, &start) != 0) {
        delete obj;
        return nullptr;
    }
    pthread_mutex_lock(&start.lock);  // wait for the id, as CreateThread returns it
    while (!start.started) pthread_cond_wait(&start.ready, &start.lock);
    pthread_mutex_unlock(&start.lock);
    return obj;
}

uint32_t wait_thread(Object* o) {
    if (!o->joined) {
        pthread_join(o->thread, nullptr);
        o->joined = true;
    }
    return o->exit_code;
}

Object* duplicate_object(void* h) {
    if (h == reinterpret_cast<void*>(-2)) {
        if (!t_self) return new Object{Object::Thread, -1, pthread_self(), thread_id(), 0, true};  // ponytail: the main
        h = t_self;  // thread's duplicate is never joined, so waiting on it returns at once
    }
    auto* o = static_cast<Object*>(h);
    __atomic_add_fetch(&o->refs, 1, __ATOMIC_RELAXED);
    return o;
}

void close_object(Object* o) {
    if (__atomic_sub_fetch(&o->refs, 1, __ATOMIC_ACQ_REL) > 0) return;
    if (o->kind == Object::File && o->fd > 2) close(o->fd);
    if (o->kind == Object::Thread && !o->joined) pthread_detach(o->thread);
    if (o->signal) free_signal(o->signal);
    if (o->fd > 2 || o->kind != Object::File) delete o;
}

// ---- PE loading ----------------------------------------------------------------------------------------------
namespace {

Module* load_pe(const char* path, bool dll) {
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
        if (rd<uint16_t>(coff + 18) & 0x1) fail("image wants its exact base (relocations stripped)");
        auto [rva, size] = dir(5);
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
            // By ordinal ("d3d12.dll!#102", as d3d12.lib imports) or by name.
            std::string key = *t >> 63 ? dll + "!#" + std::to_string(*t & 0xffff)
                                       : dll + "!" + reinterpret_cast<char*>(image + (*t & 0x7fffffff) + 2);
            auto it = g_exports.find(key);
            if (it == g_exports.end()) {
                missing.push_back(key);
                *iat = reinterpret_cast<uint64_t>(import_trap(key));
            } else {
                *iat = reinterpret_cast<uint64_t>(it->second);
            }
        }
    }
    if (!missing.empty() && std::getenv("LEONIDA_STRICT_IMPORTS")) {
        for (auto& m : missing) std::fprintf(stderr, "host: unresolved import %s\n", m.c_str());
        fail(std::to_string(missing.size()) + " unresolved imports");
    }

    auto* mod = new Module;
    mod->base = image;
    mod->entry = entry_rva ? image + entry_rva : nullptr;
    mod->size = image_size;
    mod->path = path;
    if (dll && entry_rva) mod->dll_main = reinterpret_cast<int32_t(MSABI*)(void*, uint32_t, void*)>(image + entry_rva);
    if (auto [rva, size] = dir(9); size) {
        const uint8_t* t = image + rva;
        uint64_t start = rd<uint64_t>(t), end = rd<uint64_t>(t + 8), index = rd<uint64_t>(t + 16),
                 callbacks = rd<uint64_t>(t + 24);
        mod->tls_start = reinterpret_cast<const uint8_t*>(start);
        mod->tls_raw = end - start;
        mod->tls_zero = rd<uint32_t>(t + 32);
        for (auto* cb = reinterpret_cast<uint64_t*>(callbacks); cb && *cb; ++cb)
            mod->callbacks.push_back(reinterpret_cast<void(MSABI*)(void*, uint32_t, void*)>(*cb));
        std::lock_guard<std::mutex> lock(g_mod_lock);
        if (g_tls_count == kMaxTlsModules) fail("too many modules with TLS");
        // Game code built as an EXE reads its TLS at index 0 without loading _tls_index (Windows always gives the
        // EXE index 0). Under gta6rt the game is a mapped image that takes the index of gta6rt_tls_host.dll, so
        // that DLL gets 0 and gta6rt.exe, which does load its _tls_index, moves to 1.
        static bool reserve_zero = false;
        std::string file = path;
        file.erase(0, file.find_last_of('/') + 1);
        if (!dll && strcasecmp(file.c_str(), "gta6rt.exe") == 0 && g_tls_count == 0) reserve_zero = true, g_tls_count = 1;
        if (reserve_zero && dll && strcasecmp(file.c_str(), "gta6rt_tls_host.dll") == 0)
            mod->tls_index = 0, reserve_zero = false;
        else
            mod->tls_index = g_tls_count++;
        if (index) *reinterpret_cast<uint32_t*>(index) = static_cast<uint32_t>(mod->tls_index);
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
    std::lock_guard<std::mutex> lock(g_mod_lock);
    if (mod->tls_index >= 0)  // threads that already exist get this module's TLS block too
        for (int i = 0, n = std::min(g_teb_count.load(), 1024); i < n; ++i)
            if (g_tebs[i]) teb_at<uint8_t**>(g_tebs[i], kTebTlsPointer)[mod->tls_index] = tls_block(*mod);
    g_mods.push_back(mod);
    return mod;
}

}  // namespace

// ---- DLLs loaded from disk ------------------------------------------------------------------------------------
void* load_module(const std::string& host_path) {
    char real[PATH_MAX];
    if (!realpath(host_path.c_str(), real)) return nullptr;
    {
        std::lock_guard<std::mutex> lock(g_mod_lock);
        for (auto* m : g_mods)
            if (m->path == real) return m->base;
    }
    Module* m = load_pe(real, true);
    notify_module(*m, 1 /* DLL_PROCESS_ATTACH */);
    return m->base;
}

const Module* module_at(const void* p) {  // the module whose image holds p
    auto a = reinterpret_cast<const uint8_t*>(p);
    std::lock_guard<std::mutex> lock(g_mod_lock);
    for (auto* m : g_mods)
        if (a >= m->base && a < m->base + m->size) return m;
    return nullptr;
}
bool is_module(void* h) {
    const Module* m = module_at(h);
    return m && m->base == h;
}
void* module_containing(const void* p) {
    const Module* m = module_at(p);
    return m ? m->base : nullptr;
}
void* find_module(const std::string& name) {  // by file name, case-insensitive
    std::lock_guard<std::mutex> lock(g_mod_lock);
    for (auto* m : g_mods) {
        std::string file = m->path.substr(m->path.find_last_of('/') + 1);
        if (strcasecmp(file.c_str(), name.c_str()) == 0) return m->base;
    }
    return nullptr;
}
void* module_export(void* h, const char* name) {  // by name, or by ordinal when name < 0x10000
    auto* base = static_cast<uint8_t*>(h);
    const uint8_t* opt = base + rd<uint32_t>(base + 0x3C) + 24;
    uint32_t rva = rd<uint32_t>(opt + 112), size = rd<uint32_t>(opt + 116);
    if (!size) return nullptr;
    const uint8_t* ed = base + rva;
    uint32_t ordinal_base = rd<uint32_t>(ed + 16), nfuncs = rd<uint32_t>(ed + 20), nnames = rd<uint32_t>(ed + 24);
    auto* funcs = reinterpret_cast<const uint32_t*>(base + rd<uint32_t>(ed + 28));
    auto* names = reinterpret_cast<const uint32_t*>(base + rd<uint32_t>(ed + 32));
    auto* ords = reinterpret_cast<const uint16_t*>(base + rd<uint32_t>(ed + 36));
    uint32_t index = UINT32_MAX;
    if (reinterpret_cast<uintptr_t>(name) < 0x10000) index = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(name)) - ordinal_base;
    else
        for (uint32_t i = 0; i < nnames; ++i)
            if (std::strcmp(reinterpret_cast<const char*>(base + names[i]), name) == 0) index = ords[i];
    if (index >= nfuncs || !funcs[index]) return nullptr;
    if (funcs[index] >= rva && funcs[index] < rva + size) return nullptr;  // ponytail: forwarders are not followed
    return base + funcs[index];
}

}  // namespace host

int main(int argc, char** argv) {
    using namespace host;
    if (argc < 2) {
        std::fprintf(stderr, "usage: macos_host <image.exe> [guest arguments...]\n");
        return 64;
    }
    // Rosetta runs AVX/AVX2/FMA/F16C/BMI but only reports them in CPUID when started with this variable; without it
    // the guest CRT's __isa_available check picks its SSE paths.
    if (!getenv("ROSETTA_ADVERTISE_AVX")) {
        char self[PATH_MAX];
        uint32_t size = sizeof self;
        if (_NSGetExecutablePath(self, &size) == 0) {
            setenv("ROSETTA_ADVERTISE_AVX", "1", 1);
            execv(self, argv);  // falls through on failure: the guest still runs, without AVX in CPUID
        }
    }
    for (auto* list : {&kernel32_exports, &crt_exports, &eh_exports, &vulkan_exports, &user32_exports, &kernel32_crt_exports, &kernel32_file_exports, &d3d12_serialize_exports, &seh_exports, &kernel32_loader_exports, &threadpool_exports, &crt_lib_exports})
        for (auto& [name, fn] : (*list)()) g_exports[name] = fn;

    char real[PATH_MAX];
    if (!realpath(argv[1], real)) fail(std::string("cannot open ") + argv[1]);
    g_game_dir.assign(real, std::strrchr(real, '/'));
    g_exe_path = real;
    const char* slash = std::strrchr(argv[1], '/');
    g_argc = argc - 1;
    g_argv = argv + 1;
    g_argv[0] = const_cast<char*>(slash ? slash + 1 : argv[1]);
    g_command_line = g_argv[0];
    for (int i = 1; i < g_argc; ++i) g_command_line += std::string(" ") + g_argv[i];

    struct sigaction sa = {};
    sa.sa_sigaction = on_fault;
    sa.sa_flags = SA_SIGINFO | SA_ONSTACK;
    for (int sig : {SIGSEGV, SIGBUS, SIGILL, SIGFPE, SIGTRAP}) sigaction(sig, &sa, nullptr);
    g_peb = static_cast<uint8_t*>(calloc(1, 0x1000));
    Module* exe = load_pe(real, false);
    g_image = exe->base;
    auto entry = reinterpret_cast<void(MSABI*)()>(exe->entry);
    std::memcpy(g_peb + 0x10, &g_image, sizeof g_image);  // PEB.ImageBaseAddress
    uint8_t* teb = make_teb();
    attach_thread(teb, 1 /* DLL_PROCESS_ATTACH */);
    call_guest(entry);
    std::fprintf(stderr, "host: entry point returned\n");
    return 0;
}
