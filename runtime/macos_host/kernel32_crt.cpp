// The kernel32 the static MSVC CRT (/MT) and Leonida's D3D12 layer call: locks, one-time init, modules, text
// conversion, environment, console. Windows lock objects (SRWLOCK, CONDITION_VARIABLE, CRITICAL_SECTION) live in
// guest memory; each holds a pointer to a host object, created on first use.
#include "host.h"

#include <malloc/malloc.h>
#include <sched.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#include <atomic>
#include <condition_variable>
#include <cctype>
#include <cstdio>
#include <cstring>
#include <map>
#include <mutex>

extern char** environ;

namespace host {
namespace {

using HANDLE = void*;

std::string narrow(const char16_t* s, int64_t n = -1) {  // UTF-16 -> UTF-8
    std::string out;
    for (int64_t i = 0; n < 0 ? s[i] != 0 : i < n; ++i) {
        uint32_t c = s[i];
        if (c >= 0xD800 && c < 0xDC00 && (n < 0 || i + 1 < n) && s[i + 1] >= 0xDC00 && s[i + 1] < 0xE000)
            c = 0x10000 + ((c - 0xD800) << 10) + (s[++i] - 0xDC00);
        if (c < 0x80) out += static_cast<char>(c);
        else if (c < 0x800) out += {static_cast<char>(0xC0 | c >> 6), static_cast<char>(0x80 | (c & 63))};
        else if (c < 0x10000)
            out += {static_cast<char>(0xE0 | c >> 12), static_cast<char>(0x80 | (c >> 6 & 63)),
                    static_cast<char>(0x80 | (c & 63))};
        else
            out += {static_cast<char>(0xF0 | c >> 18), static_cast<char>(0x80 | (c >> 12 & 63)),
                    static_cast<char>(0x80 | (c >> 6 & 63)), static_cast<char>(0x80 | (c & 63))};
    }
    return out;
}
std::u16string widen(const char* s, int64_t n = -1) {  // UTF-8 -> UTF-16; a bad byte passes through as Latin-1
    std::u16string out;
    auto* u = reinterpret_cast<const uint8_t*>(s);
    int64_t len = n < 0 ? static_cast<int64_t>(std::strlen(s)) : n;
    for (int64_t i = 0; i < len; ++i) {
        uint32_t c = u[i];
        int extra = c >= 0xF0 ? 3 : c >= 0xE0 ? 2 : c >= 0xC0 ? 1 : 0;
        if (extra && i + extra < len) {
            uint32_t v = c & (0x3F >> extra);
            bool ok = true;
            for (int k = 1; ok && k <= extra; ++k) {
                ok = (u[i + k] & 0xC0) == 0x80;
                v = v << 6 | (u[i + k] & 63);
            }
            if (ok) c = v, i += extra;
        }
        if (c >= 0x10000)
            out += {static_cast<char16_t>(0xD800 + ((c - 0x10000) >> 10)),
                    static_cast<char16_t>(0xDC00 + ((c - 0x10000) & 0x3FF))};
        else
            out += static_cast<char16_t>(c);
    }
    return out;
}
int64_t wlen(const char16_t* s) {
    int64_t n = 0;
    while (s[n]) ++n;
    return n;
}
template <typename C> int32_t copy_out(const std::basic_string<C>& s, bool with_nul, C* dst, int32_t cap) {
    int32_t need = static_cast<int32_t>(s.size()) + (with_nul ? 1 : 0);
    if (cap == 0) return need;
    if (cap < need) {
        set_last_error(122 /* ERROR_INSUFFICIENT_BUFFER */);
        return 0;
    }
    std::memcpy(dst, s.data(), s.size() * sizeof(C));
    if (with_nul) dst[s.size()] = 0;
    return need;
}

// ---- modules: the image, plus one handle per DLL the host exports from ----------------------------------------
std::mutex g_modules_lock;
std::map<std::string, std::string> g_modules;  // lowercase name -> itself; the handle is the value's address
HANDLE load_library(std::string name) {
    for (auto& c : name) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    if (auto slash = name.find_last_of("\\/"); slash != std::string::npos) name.erase(0, slash + 1);
    if (name.find('.') == std::string::npos) name += ".dll";
    if (!has_dll(name)) {
        set_last_error(126 /* ERROR_MOD_NOT_FOUND */);
        return nullptr;
    }
    std::lock_guard<std::mutex> lock(g_modules_lock);
    return &g_modules.try_emplace(name, name).first->second;
}
HANDLE LoadLibraryA(const char* name) { return load_library(name); }
HANDLE LoadLibraryExA(const char* name, HANDLE, uint32_t) { return load_library(name); }
HANDLE LoadLibraryExW(const char16_t* name, HANDLE, uint32_t) { return load_library(narrow(name)); }
int FreeLibrary(HANDLE) { return 1; }
HANDLE GetModuleHandleW(const char16_t* name) {
    if (!name) return g_image;
    std::string n = narrow(name);
    std::string probe = n;  // a module counts as loaded when the host exports from it
    for (auto& c : probe) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    if (probe.find('.') == std::string::npos) probe += ".dll";
    return has_dll(probe) ? load_library(n) : (set_last_error(126), nullptr);
}
void* GetProcAddress(HANDLE module, const char* name) {
    void* fn = nullptr;
    if (module && module != g_image && reinterpret_cast<uintptr_t>(name) >= 0x10000)
        fn = find_export(*static_cast<std::string*>(module) + "!" + name);
    if (!fn) set_last_error(127 /* ERROR_PROC_NOT_FOUND */);
    return fn;
}
int GetModuleHandleExW(uint32_t flags, const char16_t* name, HANDLE* out) {
    if (!name || (flags & 4 /* FROM_ADDRESS */)) *out = g_image;  // ponytail: every address is in the image
    else *out = load_library(narrow(name));
    return *out != nullptr;
}
uint32_t GetModuleFileNameW(HANDLE, char16_t* buf, uint32_t n) {
    std::u16string s = widen(g_exe_path.c_str());
    uint32_t len = static_cast<uint32_t>(std::min<size_t>(s.size(), n ? n - 1 : 0));
    std::memcpy(buf, s.data(), len * 2);
    if (n) buf[len] = 0;
    return len;
}
uint32_t GetModuleFileNameA(HANDLE, char* buf, uint32_t n) {
    uint32_t len = static_cast<uint32_t>(std::min<size_t>(g_exe_path.size(), n ? n - 1 : 0));
    std::memcpy(buf, g_exe_path.data(), len);
    if (n) buf[len] = 0;
    return len;
}
HANDLE RtlPcToFileHeader(uint64_t pc, HANDLE* base) {
    const uint8_t* nt = g_image + *reinterpret_cast<const uint32_t*>(g_image + 0x3C);
    uint32_t size = *reinterpret_cast<const uint32_t*>(nt + 24 + 56);  // OptionalHeader.SizeOfImage
    bool in = pc >= reinterpret_cast<uint64_t>(g_image) && pc < reinterpret_cast<uint64_t>(g_image) + size;
    *base = in ? g_image : nullptr;
    return *base;
}

// ---- locks -----------------------------------------------------------------------------------------------------
template <typename T> T* lazy(void* slot) {  // the host object behind a zero-initialised Windows lock
    auto* p = static_cast<std::atomic<T*>*>(slot);
    T* cur = p->load(std::memory_order_acquire);
    if (cur) return cur;
    T* fresh = new T;
    if (p->compare_exchange_strong(cur, fresh)) return fresh;
    delete fresh;
    return cur;
}

struct Srw {  // reader/writer lock that a condition variable can release from either mode
    std::mutex m;
    std::condition_variable_any c;
    int readers = 0;
    bool writer = false;
};
void AcquireSRWLockExclusive(void* l) {
    Srw* s = lazy<Srw>(l);
    std::unique_lock<std::mutex> g(s->m);
    s->c.wait(g, [s] { return !s->writer && !s->readers; });
    s->writer = true;
}
void AcquireSRWLockShared(void* l) {
    Srw* s = lazy<Srw>(l);
    std::unique_lock<std::mutex> g(s->m);
    s->c.wait(g, [s] { return !s->writer; });
    ++s->readers;
}
uint8_t TryAcquireSRWLockExclusive(void* l) {
    Srw* s = lazy<Srw>(l);
    std::lock_guard<std::mutex> g(s->m);
    if (s->writer || s->readers) return 0;
    return s->writer = true;
}
uint8_t TryAcquireSRWLockShared(void* l) {
    Srw* s = lazy<Srw>(l);
    std::lock_guard<std::mutex> g(s->m);
    if (s->writer) return 0;
    ++s->readers;
    return 1;
}
void ReleaseSRWLockExclusive(void* l) {
    Srw* s = lazy<Srw>(l);
    {
        std::lock_guard<std::mutex> g(s->m);
        s->writer = false;
    }
    s->c.notify_all();
}
void ReleaseSRWLockShared(void* l) {
    Srw* s = lazy<Srw>(l);
    bool last;
    {
        std::lock_guard<std::mutex> g(s->m);
        last = --s->readers == 0;
    }
    if (last) s->c.notify_all();
}

struct Cv {  // a generation counter, so a wake between the release and the wait is not lost
    std::mutex m;
    std::condition_variable c;
    uint64_t gen = 0;
};
int SleepConditionVariableSRW(void* cv, void* l, uint32_t ms, uint32_t flags) {
    Cv* v = lazy<Cv>(cv);
    bool shared = flags & 1;  // CONDITION_VARIABLE_LOCKMODE_SHARED
    std::unique_lock<std::mutex> g(v->m);
    uint64_t gen = v->gen;
    shared ? ReleaseSRWLockShared(l) : ReleaseSRWLockExclusive(l);
    bool woke = true;
    if (ms == 0xFFFFFFFFu) v->c.wait(g, [&] { return v->gen != gen; });
    else woke = v->c.wait_for(g, std::chrono::milliseconds(ms), [&] { return v->gen != gen; });
    g.unlock();
    shared ? AcquireSRWLockShared(l) : AcquireSRWLockExclusive(l);
    if (!woke) set_last_error(1460 /* ERROR_TIMEOUT */);
    return woke;
}
void WakeConditionVariable(void* cv) {
    Cv* v = lazy<Cv>(cv);
    {
        std::lock_guard<std::mutex> g(v->m);
        ++v->gen;
    }
    v->c.notify_one();
}
void WakeAllConditionVariable(void* cv) {
    Cv* v = lazy<Cv>(cv);
    {
        std::lock_guard<std::mutex> g(v->m);
        ++v->gen;
    }
    v->c.notify_all();
}

// CRITICAL_SECTION is 40 bytes; its first field (DebugInfo) holds our recursive mutex.
int InitializeCriticalSectionAndSpinCount(void* cs, uint32_t) {
    std::memset(cs, 0, 40);
    lazy<std::recursive_mutex>(cs);
    return 1;
}
int InitializeCriticalSectionEx(void* cs, uint32_t spin, uint32_t) { return InitializeCriticalSectionAndSpinCount(cs, spin); }
void EnterCriticalSection(void* cs) { lazy<std::recursive_mutex>(cs)->lock(); }
void LeaveCriticalSection(void* cs) { lazy<std::recursive_mutex>(cs)->unlock(); }
void DeleteCriticalSection(void* cs) {
    delete *static_cast<std::recursive_mutex**>(cs);
    *static_cast<void**>(cs) = nullptr;
}

// INIT_ONCE: 0 = not yet, otherwise the context with bit 1 set. ponytail: one global lock for every first run.
std::recursive_mutex g_once_lock;
using InitFn = int32_t(MSABI*)(void*, void*, void**);
int InitOnceBeginInitialize(void* once, uint32_t, int32_t* pending, void** ctx) {
    auto* v = static_cast<std::atomic<uint64_t>*>(once);
    if (uint64_t x = v->load(std::memory_order_acquire)) {
        *pending = 0;
        if (ctx) *ctx = reinterpret_cast<void*>(x & ~3ull);
        return 1;
    }
    g_once_lock.lock();
    if (uint64_t x = v->load(std::memory_order_acquire)) {
        g_once_lock.unlock();
        *pending = 0;
        if (ctx) *ctx = reinterpret_cast<void*>(x & ~3ull);
        return 1;
    }
    *pending = 1;  // the lock stays held until InitOnceComplete
    return 1;
}
int InitOnceComplete(void* once, uint32_t flags, void* ctx) {
    if (!(flags & 4 /* INIT_ONCE_INIT_FAILED */))
        static_cast<std::atomic<uint64_t>*>(once)->store(reinterpret_cast<uint64_t>(ctx) | 2, std::memory_order_release);
    g_once_lock.unlock();
    return 1;
}
int InitOnceExecuteOnce(void* once, InitFn fn, void* param, void** ctx) {
    int32_t pending;
    InitOnceBeginInitialize(once, 0, &pending, ctx);
    if (!pending) return 1;
    void* c = nullptr;
    int32_t ok = call_guest(fn, once, param, &c);
    InitOnceComplete(once, ok ? 0 : 4, c);
    if (ctx) *ctx = c;
    return ok;
}

// SLIST_HEADER: ours keeps the first entry in its first 8 bytes. The guest only touches it through these calls.
std::mutex g_slist_lock;
void* InterlockedPushEntrySList(void* head, void* entry) {
    std::lock_guard<std::mutex> g(g_slist_lock);
    void* old = *static_cast<void**>(head);
    *static_cast<void**>(entry) = old;
    *static_cast<void**>(head) = entry;
    return old;
}
void* InterlockedFlushSList(void* head) {
    std::lock_guard<std::mutex> g(g_slist_lock);
    void* old = *static_cast<void**>(head);
    *static_cast<void**>(head) = nullptr;
    return old;
}

// ---- events and semaphores: one counter each. An event holds 0 or 1; an auto-reset event or a semaphore takes
// one from it per satisfied wait, a manual-reset event takes nothing.
}  // namespace
struct Signal {
    std::mutex m;
    std::condition_variable c;
    int64_t count, max;
    bool manual;
};
void free_signal(Signal* s) { delete s; }
namespace {

HANDLE make_signal(Object::Kind kind, int64_t count, int64_t max, bool manual) {
    return new Object{kind, -1, {}, 0, 0, false, new Signal{{}, {}, count, max, manual}};
}
HANDLE CreateEventW(void*, int32_t manual, int32_t initial, const char16_t* name) {
    if (name) fail("CreateEventW: named events are not supported");
    return make_signal(Object::Event, initial ? 1 : 0, 1, manual);
}
HANDLE CreateEventExW(void*, const char16_t* name, uint32_t flags, uint32_t) {
    if (name) fail("CreateEventExW: named events are not supported");
    return make_signal(Object::Event, (flags & 2) ? 1 : 0, 1, flags & 1);  // INITIAL_SET, MANUAL_RESET
}
HANDLE CreateSemaphoreW(void*, int32_t initial, int32_t max, const char16_t* name) {
    if (name) fail("CreateSemaphoreW: named semaphores are not supported");
    return make_signal(Object::Semaphore, initial, max, false);
}
HANDLE CreateSemaphoreExW(void* sa, int32_t initial, int32_t max, const char16_t* name, uint32_t, uint32_t) {
    return CreateSemaphoreW(sa, initial, max, name);
}
int SetEvent(HANDLE h) {
    Signal* s = static_cast<Object*>(h)->signal;
    {
        std::lock_guard<std::mutex> g(s->m);
        s->count = 1;
    }
    s->c.notify_all();
    return 1;
}
int ReleaseSemaphore(HANDLE h, int32_t n, int32_t* previous) {
    Signal* s = static_cast<Object*>(h)->signal;
    {
        std::lock_guard<std::mutex> g(s->m);
        if (previous) *previous = static_cast<int32_t>(s->count);
        if (n <= 0 || s->count + n > s->max) return set_last_error(298 /* ERROR_TOO_MANY_POSTS */), 0;
        s->count += n;
    }
    s->c.notify_all();
    return 1;
}
uint32_t WaitForSingleObjectEx(HANDLE h, uint32_t ms, int32_t) {
    auto* o = static_cast<Object*>(h);
    if (o->kind == Object::Thread) {  // ponytail: a thread is waited for without a timeout
        if (ms != 0xFFFFFFFFu) fail("WaitForSingleObject: thread waits need INFINITE");
        wait_thread(o);
        return 0;
    }
    if (!o->signal) return set_last_error(6), 0xFFFFFFFFu;
    Signal* s = o->signal;
    std::unique_lock<std::mutex> g(s->m);
    auto ready = [s] { return s->count > 0; };
    if (ms == 0xFFFFFFFFu) s->c.wait(g, ready);
    else if (!s->c.wait_for(g, std::chrono::milliseconds(ms), ready)) return 0x102;  // WAIT_TIMEOUT
    if (!s->manual) --s->count;
    return 0;  // WAIT_OBJECT_0
}
[[noreturn]] void ExitThread(uint32_t code) { exit_thread(code); }
[[noreturn]] void FreeLibraryAndExitThread(HANDLE, uint32_t code) { exit_thread(code); }
int GetExitCodeThread(HANDLE h, uint32_t* code) {  // ponytail: exact only once the thread has been waited for
    auto* o = static_cast<Object*>(h);
    *code = o->joined ? o->exit_code : 259 /* STILL_ACTIVE */;
    return 1;
}
uint32_t WaitForSingleObject(HANDLE h, uint32_t ms) { return WaitForSingleObjectEx(h, ms, 0); }

// ---- text and locale: UTF-8 is the ANSI code page --------------------------------------------------------------
uint32_t GetACP() { return 65001; }
uint32_t GetOEMCP() { return 65001; }
uint32_t GetConsoleCP() { return 65001; }
int IsValidCodePage(uint32_t) { return 1; }
int GetCPInfo(uint32_t, uint8_t* info) {  // CPINFO: MaxCharSize, DefaultChar[2], LeadByte[12]
    std::memset(info, 0, 18);
    info[0] = 4;
    info[4] = '?';
    return 1;
}
int32_t MultiByteToWideChar(uint32_t, uint32_t, const char* src, int32_t n, char16_t* dst, int32_t cap) {
    std::u16string s = widen(src, n < 0 ? static_cast<int64_t>(std::strlen(src)) : n);
    return copy_out(s, n < 0, dst, cap);
}
int32_t WideCharToMultiByte(uint32_t, uint32_t, const char16_t* src, int32_t n, char* dst, int32_t cap,
                            const char*, int32_t* used_default) {
    if (used_default) *used_default = 0;
    std::string s = narrow(src, n < 0 ? wlen(src) : n);
    return copy_out(s, n < 0, dst, cap);
}
int GetStringTypeW(uint32_t type, const char16_t* src, int32_t n, uint16_t* out) {
    if (type != 1 /* CT_CTYPE1 */) fail("GetStringTypeW: only CT_CTYPE1");
    if (n < 0) n = static_cast<int32_t>(wlen(src)) + 1;
    for (int32_t i = 0; i < n; ++i) {
        unsigned c = src[i];
        uint16_t f = 0x200;  // C1_DEFINED
        if (c < 128) {
            if (std::isupper(c)) f |= 0x1 | 0x100;
            if (std::islower(c)) f |= 0x2 | 0x100;
            if (std::isdigit(c)) f |= 0x4;
            if (std::isspace(c)) f |= 0x8;
            if (std::ispunct(c)) f |= 0x10;
            if (std::iscntrl(c)) f |= 0x20;
            if (c == ' ' || c == '\t') f |= 0x40;
            if (std::isxdigit(c)) f |= 0x80;
        } else {
            f |= 0x100;  // ponytail: non-ASCII counts as a letter
        }
        out[i] = f;
    }
    return 1;
}
int32_t map_string(uint32_t flags, const char16_t* src, int32_t n, char16_t* dst, int32_t cap) {
    if (n < 0) n = static_cast<int32_t>(wlen(src)) + 1;
    if (flags & 0x400) {  // LCMAP_SORTKEY: code units big-endian, then a 0 byte (ordinal order)
        int32_t need = 2 * n + 1;
        if (!cap) return need;
        if (cap < need) return set_last_error(122), 0;
        auto* b = reinterpret_cast<uint8_t*>(dst);
        for (int32_t i = 0; i < n; ++i) b[2 * i] = src[i] >> 8, b[2 * i + 1] = src[i] & 0xFF;
        b[2 * n] = 0;
        return need;
    }
    if (!cap) return n;
    if (cap < n) return set_last_error(122), 0;
    for (int32_t i = 0; i < n; ++i) {
        char16_t c = src[i];
        if (c < 128 && (flags & 0x100)) c = static_cast<char16_t>(std::tolower(c));
        if (c < 128 && (flags & 0x200)) c = static_cast<char16_t>(std::toupper(c));
        dst[i] = c;
    }
    return n;
}
int32_t LCMapStringEx(const char16_t*, uint32_t flags, const char16_t* src, int32_t n, char16_t* dst, int32_t cap,
                      void*, void*, int64_t) {
    return map_string(flags, src, n, dst, cap);
}
int32_t LCMapStringW(uint32_t, uint32_t flags, const char16_t* src, int32_t n, char16_t* dst, int32_t cap) {
    return map_string(flags, src, n, dst, cap);
}
int32_t compare(uint32_t flags, const char16_t* a, int32_t na, const char16_t* b, int32_t nb) {
    if (na < 0) na = static_cast<int32_t>(wlen(a));
    if (nb < 0) nb = static_cast<int32_t>(wlen(b));
    for (int32_t i = 0; i < na && i < nb; ++i) {
        char16_t x = a[i], y = b[i];
        if ((flags & 1) && x < 128 && y < 128) x = std::tolower(x), y = std::tolower(y);  // NORM_IGNORECASE
        if (x != y) return x < y ? 1 : 3;
    }
    return na == nb ? 2 : na < nb ? 1 : 3;  // CSTR_LESS_THAN / EQUAL / GREATER_THAN
}
int32_t CompareStringEx(const char16_t*, uint32_t flags, const char16_t* a, int32_t na, const char16_t* b, int32_t nb,
                        void*, void*, int64_t) {
    return compare(flags, a, na, b, nb);
}
int32_t CompareStringW(uint32_t, uint32_t flags, const char16_t* a, int32_t na, const char16_t* b, int32_t nb) {
    return compare(flags, a, na, b, nb);
}
uint32_t GetUserDefaultLCID() { return 0x409; }
int IsValidLocale(uint32_t, uint32_t) { return 1; }

// ---- process environment ---------------------------------------------------------------------------------------
const char16_t* GetCommandLineW() {
    static std::u16string line = widen(g_command_line.c_str());
    return line.c_str();
}
char16_t* GetEnvironmentStringsW() {  // "A=1\0B=2\0\0", freshly allocated like Windows does
    std::u16string block;
    for (char** e = environ; *e; ++e) block += widen(*e) + u'\0';
    block += u'\0';
    auto* out = static_cast<char16_t*>(malloc(block.size() * 2));
    std::memcpy(out, block.data(), block.size() * 2);
    return out;
}
int FreeEnvironmentStringsW(char16_t* p) {
    free(p);
    return 1;
}
uint32_t GetEnvironmentVariableA(const char* name, char* buf, uint32_t n) {
    const char* v = std::getenv(name);
    if (!v) return set_last_error(203 /* ERROR_ENVVAR_NOT_FOUND */), 0;
    uint32_t len = static_cast<uint32_t>(std::strlen(v));
    if (n <= len) return len + 1;
    std::memcpy(buf, v, len + 1);
    return len;
}
int SetEnvironmentVariableA(const char* name, const char* value) {
    return (value ? setenv(name, value, 1) : unsetenv(name)) == 0;
}
int SetEnvironmentVariableW(const char16_t* name, const char16_t* value) {
    std::string n = narrow(name);
    if (!value) return unsetenv(n.c_str()) == 0;
    return setenv(n.c_str(), narrow(value).c_str(), 1) == 0;
}
void GetStartupInfoW(uint8_t* info) {
    std::memset(info, 0, 104);
    *reinterpret_cast<uint32_t*>(info) = 104;
}
void SetLastError(uint32_t e) { set_last_error(e); }

// ---- console and standard handles ------------------------------------------------------------------------------
int fd_of(HANDLE h) { return h && h != reinterpret_cast<HANDLE>(-1) ? static_cast<Object*>(h)->fd : -1; }
uint32_t GetFileType(HANDLE h) {
    struct stat st;
    int fd = fd_of(h);
    if (fd < 0 || fstat(fd, &st) != 0) return 0;  // FILE_TYPE_UNKNOWN
    if (S_ISCHR(st.st_mode)) return 2;           // FILE_TYPE_CHAR
    if (S_ISFIFO(st.st_mode) || S_ISSOCK(st.st_mode)) return 3;  // FILE_TYPE_PIPE
    return 1;                                    // FILE_TYPE_DISK
}
int SetStdHandle(uint32_t, HANDLE) { return 1; }  // ponytail: redirection ignored
int GetConsoleMode(HANDLE h, uint32_t* mode) {
    if (!isatty(fd_of(h))) return set_last_error(6 /* ERROR_INVALID_HANDLE */), 0;
    *mode = 3;
    return 1;
}
int WriteConsoleW(HANDLE h, const char16_t* buf, uint32_t n, uint32_t* written, void*) {
    std::string s = narrow(buf, n);
    ssize_t r = write(fd_of(h), s.data(), s.size());
    if (written) *written = r < 0 ? 0 : n;
    return r >= 0;
}
void OutputDebugStringA(const char* s) { std::fputs(s, stderr); }
void OutputDebugStringW(const char16_t* s) { std::fputs(narrow(s).c_str(), stderr); }

// ---- memory, time, system --------------------------------------------------------------------------------------
void* HeapReAlloc(HANDLE, uint32_t flags, void* p, uint64_t size) {
    size_t old = malloc_size(p);
    void* q = realloc(p, size);
    if (q && (flags & 0x8) && size > old) std::memset(static_cast<char*>(q) + old, 0, size - old);
    return q;
}
uint64_t HeapSize(HANDLE, uint32_t, const void* p) { return malloc_size(p); }
int HeapQueryInformation(HANDLE, int32_t, void*, uint64_t, uint64_t*) { return 0; }
int VirtualProtect(void* addr, uint64_t size, uint32_t prot, uint32_t* old) {
    auto a = reinterpret_cast<uintptr_t>(addr) & ~uintptr_t(4095);
    int p = PROT_READ;
    if (prot & 0xCC) p |= PROT_WRITE;  // PAGE_READWRITE, WRITECOPY, EXECUTE_READWRITE, EXECUTE_WRITECOPY
    if (prot & 0xF0) p |= PROT_EXEC;
    if (prot == 0x01) p = PROT_NONE;   // PAGE_NOACCESS
    if (old) *old = 0x04;              // ponytail: the previous protection is not tracked
    return mprotect(reinterpret_cast<void*>(a), size + (reinterpret_cast<uintptr_t>(addr) - a), p) == 0;
}
int QueryPerformanceFrequency(int64_t* f) {
    *f = 1000000000;  // QueryPerformanceCounter counts nanoseconds
    return 1;
}
uint64_t GetTickCount64() { return clock_gettime_nsec_np(CLOCK_UPTIME_RAW) / 1000000; }
void system_info(uint8_t* si) {  // SYSTEM_INFO
    std::memset(si, 0, 48);
    *reinterpret_cast<uint16_t*>(si) = 9;  // PROCESSOR_ARCHITECTURE_AMD64
    *reinterpret_cast<uint32_t*>(si + 4) = 4096;
    *reinterpret_cast<uint64_t*>(si + 8) = 0x10000;
    *reinterpret_cast<uint64_t*>(si + 16) = 0x7FFFFFFEFFFFull;
    long n = sysconf(_SC_NPROCESSORS_ONLN);
    *reinterpret_cast<uint64_t*>(si + 24) = n >= 64 ? ~0ull : (1ull << n) - 1;
    *reinterpret_cast<uint32_t*>(si + 32) = static_cast<uint32_t>(n);
    *reinterpret_cast<uint32_t*>(si + 36) = 8664;
    *reinterpret_cast<uint32_t*>(si + 40) = 65536;
    *reinterpret_cast<uint16_t*>(si + 44) = 6;
}
void GetSystemInfo(uint8_t* si) { system_info(si); }
void GetNativeSystemInfo(uint8_t* si) { system_info(si); }
uint32_t GetCurrentProcessorNumber() { return 0; }
void FlushProcessWriteBuffers() { std::atomic_thread_fence(std::memory_order_seq_cst); }
HANDLE GetCurrentThread() { return reinterpret_cast<HANDLE>(-2); }
int SwitchToThread() { return sched_yield() == 0; }
// ponytail: the handler is recorded nowhere; a fault reaches the host's signal handler and ends the run.
void* AddVectoredExceptionHandler(uint32_t, void* handler) { return handler; }
void* EncodePointer(void* p) { return p; }

// advapi32: there are no Windows security tokens here, so no privilege is ever held.
int OpenProcessToken(HANDLE, uint32_t, HANDLE*) { return set_last_error(5 /* ERROR_ACCESS_DENIED */), 0; }
int OpenThreadToken(HANDLE, uint32_t, int32_t, HANDLE*) { return set_last_error(1008 /* ERROR_NO_TOKEN */), 0; }
void* DecodePointer(void* p) { return p; }

}  // namespace

#define K(name) EXPORT("kernel32.dll", name)
Exports kernel32_crt_exports() {
    return {
        K(LoadLibraryA), K(LoadLibraryExA), K(LoadLibraryExW), K(FreeLibrary), K(GetModuleHandleW), K(GetProcAddress),
        K(GetModuleHandleExW), K(GetModuleFileNameW), K(GetModuleFileNameA), K(RtlPcToFileHeader),
        K(AcquireSRWLockExclusive), K(AcquireSRWLockShared), K(TryAcquireSRWLockExclusive),
        K(TryAcquireSRWLockShared), K(ReleaseSRWLockExclusive), K(ReleaseSRWLockShared),
        K(SleepConditionVariableSRW), K(WakeConditionVariable), K(WakeAllConditionVariable),
        K(InitializeCriticalSectionAndSpinCount), K(InitializeCriticalSectionEx), K(EnterCriticalSection),
        K(LeaveCriticalSection), K(DeleteCriticalSection), K(InitOnceBeginInitialize), K(InitOnceComplete),
        K(InitOnceExecuteOnce), K(InterlockedPushEntrySList), K(InterlockedFlushSList), K(GetACP), K(GetOEMCP),
        K(GetConsoleCP), K(IsValidCodePage), K(GetCPInfo), K(MultiByteToWideChar), K(WideCharToMultiByte),
        K(GetStringTypeW), K(LCMapStringEx), K(LCMapStringW), K(CompareStringEx), K(CompareStringW),
        K(GetUserDefaultLCID), K(IsValidLocale), K(GetCommandLineW), K(GetEnvironmentStringsW),
        K(FreeEnvironmentStringsW), K(GetEnvironmentVariableA), K(SetEnvironmentVariableA),
        K(SetEnvironmentVariableW), K(GetStartupInfoW), K(SetLastError), K(GetFileType), K(SetStdHandle),
        K(GetConsoleMode), K(WriteConsoleW), K(OutputDebugStringA), K(OutputDebugStringW), K(HeapReAlloc),
        K(HeapSize), K(HeapQueryInformation), K(VirtualProtect), K(QueryPerformanceFrequency), K(GetTickCount64),
        K(GetSystemInfo), K(GetNativeSystemInfo), K(GetCurrentProcessorNumber), K(FlushProcessWriteBuffers),
        K(GetCurrentThread), K(SwitchToThread), K(AddVectoredExceptionHandler), K(EncodePointer), K(CreateEventW), K(CreateEventExW), K(CreateSemaphoreW), K(CreateSemaphoreExW),
        K(SetEvent), K(ReleaseSemaphore), K(WaitForSingleObjectEx), K(WaitForSingleObject), K(ExitThread), K(FreeLibraryAndExitThread),
        K(GetExitCodeThread), EXPORT("advapi32.dll", OpenProcessToken), EXPORT("advapi32.dll", OpenThreadToken), K(DecodePointer),
    };
}

}  // namespace host
