// Prototype: run a Windows x64 PE natively on macOS (an x86-64 Mach-O under Rosetta 2), without Wine.
// Maps the image, applies DIR64 relocations, binds imports to host functions with the Windows x64 calling
// convention, gives every guest thread a TEB (with static TLS) reached through gs:, and swaps gs back to the
// pthread base while a host function runs. Proof of concept for #2947.
//
//   macos_host <image.exe> [guest arguments...]
#include "host.h"

#include <mach/mach.h>
#include <sys/mman.h>
#include <unistd.h>

#include <cctype>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <mutex>

namespace host {

thread_local uint8_t* t_teb = nullptr;
uint8_t* g_image = nullptr;
std::string g_command_line;
int g_argc = 0;
char** g_argv = nullptr;

namespace {

uint8_t* g_peb = nullptr;
std::map<std::string, void*> g_exports;

struct StaticTls {  // IMAGE_TLS_DIRECTORY64, resolved
    const uint8_t* start = nullptr;
    size_t raw_size = 0, zero_fill = 0;
    std::vector<void(MSABI*)(void*, uint32_t, void*)> callbacks;
} g_tls;

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

std::string to_host_path(const char* p) {  // ponytail: backslashes only; no drive letters or GAME: yet
    std::string s(p);
    for (auto& c : s)
        if (c == '\\') c = '/';
    return s;
}

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
    if (g_tls.start) {  // MSVC reads gs:[0x58][_tls_index]; the image has one module, so _tls_index is 0
        auto** slots = static_cast<uint8_t**>(calloc(1, sizeof(void*)));
        slots[0] = static_cast<uint8_t*>(calloc(1, g_tls.raw_size + g_tls.zero_fill + 16));
        std::memcpy(slots[0], g_tls.start, g_tls.raw_size);
        teb_at<uint8_t**>(teb, kTebTlsPointer) = slots;
    }
    t_teb = teb;
    return teb;
}

void attach_thread(uint8_t*, uint32_t reason) {
    for (auto cb : g_tls.callbacks) call_guest(cb, static_cast<void*>(g_image), reason, static_cast<void*>(nullptr));
}

void free_teb(uint8_t* teb) {
    if (auto** slots = teb_at<uint8_t**>(teb, kTebTlsPointer)) {
        free(slots[0]);
        free(slots);
    }
    if (t_teb == teb) t_teb = nullptr;
    free(teb);
}

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
void* thread_main(void* p) {
    auto* s = static_cast<ThreadStart*>(p);
    Object* obj = s->obj;
    ThreadProc proc = s->proc;
    void* arg = s->arg;
    obj->tid = thread_id();
    pthread_mutex_lock(&s->lock);
    s->started = true;
    pthread_cond_signal(&s->ready);
    pthread_mutex_unlock(&s->lock);  // s belongs to the creator from here on
    uint8_t* teb = make_teb();
    attach_thread(teb, 2 /* DLL_THREAD_ATTACH */);
    obj->exit_code = call_guest(proc, arg);
    attach_thread(teb, 3 /* DLL_THREAD_DETACH */);
    free_teb(teb);
    return nullptr;
}
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

void close_object(Object* o) {
    if (o->kind == Object::File && o->fd > 2) close(o->fd);
    if (o->kind == Object::Thread && !o->joined) pthread_detach(o->thread);
    if (o->fd > 2 || o->kind == Object::Thread) delete o;
}

// ---- PE loading ----------------------------------------------------------------------------------------------
namespace {

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
            if (*t >> 63) {
                missing.push_back(dll + "!#" + std::to_string(*t & 0xffff));
                continue;
            }
            std::string key = dll + "!" + reinterpret_cast<char*>(image + (*t & 0x7fffffff) + 2);
            auto it = g_exports.find(key);
            if (it == g_exports.end()) missing.push_back(key);
            else *iat = reinterpret_cast<uint64_t>(it->second);
        }
    }
    if (!missing.empty()) {
        for (auto& m : missing) std::fprintf(stderr, "host: unresolved import %s\n", m.c_str());
        fail(std::to_string(missing.size()) + " unresolved imports");
    }

    if (auto [rva, size] = dir(9); size) {
        const uint8_t* t = image + rva;
        uint64_t start = rd<uint64_t>(t), end = rd<uint64_t>(t + 8), index = rd<uint64_t>(t + 16),
                 callbacks = rd<uint64_t>(t + 24);
        g_tls.start = reinterpret_cast<const uint8_t*>(start);
        g_tls.raw_size = end - start;
        g_tls.zero_fill = rd<uint32_t>(t + 32);
        if (index) *reinterpret_cast<uint32_t*>(index) = 0;
        for (auto* cb = reinterpret_cast<uint64_t*>(callbacks); cb && *cb; ++cb)
            g_tls.callbacks.push_back(reinterpret_cast<void(MSABI*)(void*, uint32_t, void*)>(*cb));
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
    g_image = image;
    return image + entry_rva;
}

}  // namespace
}  // namespace host

int main(int argc, char** argv) {
    using namespace host;
    if (argc < 2) {
        std::fprintf(stderr, "usage: macos_host <image.exe> [guest arguments...]\n");
        return 64;
    }
    for (auto* list : {&kernel32_exports, &crt_exports, &eh_exports})
        for (auto& [name, fn] : (*list)()) g_exports[name] = fn;

    const char* slash = std::strrchr(argv[1], '/');
    g_argc = argc - 1;
    g_argv = argv + 1;
    g_argv[0] = const_cast<char*>(slash ? slash + 1 : argv[1]);
    g_command_line = g_argv[0];
    for (int i = 1; i < g_argc; ++i) g_command_line += std::string(" ") + g_argv[i];

    g_peb = static_cast<uint8_t*>(calloc(1, 0x1000));
    auto entry = reinterpret_cast<void(MSABI*)()>(load_pe(argv[1]));
    std::memcpy(g_peb + 0x10, &g_image, sizeof g_image);  // PEB.ImageBaseAddress
    uint8_t* teb = make_teb();
    attach_thread(teb, 1 /* DLL_PROCESS_ATTACH */);
    call_guest(entry);
    std::fprintf(stderr, "host: entry point returned\n");
    return 0;
}
