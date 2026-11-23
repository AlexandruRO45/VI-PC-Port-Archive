// The UCRT, vcruntime140, msvcp140 and xgameruntime pieces tiny_game imports, on the host libc. Guest FILE* are
// host FILE* (the UCRT's FILE is opaque to callers). Guest function pointers are entered through call_guest.
#include "host.h"

#include <unistd.h>

#include <cstdio>
#include <cstring>
#include <mutex>

extern char** environ;

namespace host {
namespace {

[[noreturn]] void unsupported(const char* what) { fail(std::string("unsupported: ") + what); }

using VoidFn = void(MSABI*)();
using IntFn = int(MSABI*)();

// ---- startup and exit ----
std::mutex g_atexit_lock;
std::vector<VoidFn> g_atexit;
int g_commode = 0;

int* p_argc() { return &g_argc; }
char*** p_argv() { return &g_argv; }
int* p_commode() { return &g_commode; }
int return0() { return 0; }
int return0_i(int32_t) { return 0; }
void noop_p(void*) {}
void noop() {}
char** initial_env() { return environ; }
int initialize_onexit_table(void*) { return 0; }
int register_onexit(void*, VoidFn fn) {
    std::lock_guard<std::mutex> lock(g_atexit_lock);
    g_atexit.push_back(fn);
    return 0;
}
int crt_atexit(VoidFn fn) { return register_onexit(nullptr, fn); }
void run_atexit() {
    for (;;) {
        VoidFn fn;
        {
            std::lock_guard<std::mutex> lock(g_atexit_lock);
            if (g_atexit.empty()) return;
            fn = g_atexit.back();
            g_atexit.pop_back();
        }
        call_guest(fn);
    }
}
void initterm(VoidFn* begin, VoidFn* end) {
    for (; begin < end; ++begin)
        if (*begin) call_guest(*begin);
}
int initterm_e(IntFn* begin, IntFn* end) {
    for (; begin < end; ++begin)
        if (*begin)
            if (int r = call_guest(*begin)) return r;
    return 0;
}
[[noreturn]] void crt_exit(int32_t code) {
    run_atexit();
    std::fflush(nullptr);
    _exit(code);
}
[[noreturn]] void crt_quick_exit(int32_t code) {
    std::fflush(nullptr);
    _exit(code);
}
void crt_cexit() {
    run_atexit();
    std::fflush(nullptr);
}
int32_t seh_filter_exe(uint32_t, void*) { return 0; }  // EXCEPTION_CONTINUE_SEARCH
[[noreturn]] void invoke_watson(const void*, const void*, const void*, uint32_t, uint64_t) {
    unsupported("_invoke_watson (invalid CRT parameter)");
}
[[noreturn]] void crt_terminate() { unsupported("terminate()"); }

uint64_t beginthreadex(void*, uint32_t, ThreadProc proc, void* arg, uint32_t, uint32_t* tid) {
    Object* o = spawn_thread(proc, arg);
    if (o && tid) *tid = o->tid;
    return reinterpret_cast<uint64_t>(o);
}

// ---- stdio ----
FILE* iob(uint32_t i) { return i == 0 ? stdin : i == 1 ? stdout : stderr; }
int crt_setvbuf(FILE* f, char* buf, int32_t mode, uint64_t size) {
    int host_mode = mode == 4 ? _IONBF : mode == 0x40 ? _IOLBF : _IOFBF;  // UCRT: _IONBF 4, _IOLBF 0x40
    return setvbuf(f, buf, host_mode, size);
}
FILE* crt_fopen(const char* path, const char* mode) {
    std::string m;
    for (; *mode; ++mode)
        if (*mode != 't') m.push_back(*mode);
    return std::fopen(to_host_path(path).c_str(), m.c_str());
}
int crt_fclose(FILE* f) { return std::fclose(f); }
int crt_fflush(FILE* f) { return std::fflush(f); }
uint64_t crt_fread(void* p, uint64_t size, uint64_t n, FILE* f) { return std::fread(p, size, n, f); }
uint64_t crt_fwrite(const void* p, uint64_t size, uint64_t n, FILE* f) { return std::fwrite(p, size, n, f); }
int crt_puts(const char* s) { return std::puts(s); }
int crt_remove(const char* path) { return std::remove(to_host_path(path).c_str()); }

// printf over an MS va_list: a char* walking 8-byte slots. Each conversion is re-issued to the host snprintf with
// host length modifiers, since Windows `l` is 32-bit and I64/I32/I are MS-only.
std::string ms_format(const char* fmt, char* va) {
    auto next = [&va]() { uint64_t v; std::memcpy(&v, va, 8); va += 8; return v; };
    std::string out;
    char buf[512];
    while (*fmt) {
        if (*fmt != '%') { out.push_back(*fmt++); continue; }
        if (fmt[1] == '%') { out.push_back('%'); fmt += 2; continue; }
        std::string spec = "%";
        ++fmt;
        while (std::strchr("-+ #0", *fmt)) spec.push_back(*fmt++);
        int stars[2], nstars = 0;
        auto field = [&]() {
            if (*fmt == '*') { stars[nstars++] = static_cast<int32_t>(next()); spec.push_back('*'); ++fmt; }
            else while (*fmt >= '0' && *fmt <= '9') spec.push_back(*fmt++);
        };
        field();
        if (*fmt == '.') { spec.push_back(*fmt++); field(); }
        int bits = 32;  // integer width
        bool wide = false;
        for (;;) {
            if (*fmt == 'h') { ++fmt; if (*fmt == 'h') ++fmt; }  // promoted to int anyway
            else if (*fmt == 'l') { ++fmt; if (*fmt == 'l') { ++fmt; bits = 64; } else wide = true; }
            else if (*fmt == 'L') ++fmt;
            else if (*fmt == 'z' || *fmt == 'j' || *fmt == 't') { ++fmt; bits = 64; }
            else if (!std::strncmp(fmt, "I64", 3)) { fmt += 3; bits = 64; }
            else if (!std::strncmp(fmt, "I32", 3)) fmt += 3;
            else if (*fmt == 'I') { ++fmt; bits = 64; }
            else if (*fmt == 'w') { ++fmt; wide = true; }
            else break;
        }
        char conv = *fmt ? *fmt++ : 0;
        auto emit = [&](auto value) {
            switch (nstars) {
                case 0: std::snprintf(buf, sizeof buf, spec.c_str(), value); break;
                case 1: std::snprintf(buf, sizeof buf, spec.c_str(), stars[0], value); break;
                default: std::snprintf(buf, sizeof buf, spec.c_str(), stars[0], stars[1], value); break;
            }
            out += buf;
        };
        switch (conv) {
            case 'd': case 'i': case 'u': case 'x': case 'X': case 'o': {
                uint64_t v = next();
                spec += "ll";
                spec.push_back(conv);
                bool is_signed = conv == 'd' || conv == 'i';
                if (bits == 32) emit(is_signed ? static_cast<long long>(static_cast<int32_t>(v))
                                               : static_cast<long long>(static_cast<uint32_t>(v)));
                else emit(static_cast<long long>(v));
                break;
            }
            case 'c': spec.push_back('c'); emit(static_cast<int>(next() & 0xFF)); break;
            case 'p': spec.push_back('p'); emit(reinterpret_cast<void*>(next())); break;
            case 'e': case 'E': case 'f': case 'F': case 'g': case 'G': case 'a': case 'A': {
                uint64_t v = next();
                double d;
                std::memcpy(&d, &v, 8);
                spec.push_back(conv);
                emit(d);
                break;
            }
            case 's': case 'S': {
                const void* p = reinterpret_cast<const void*>(next());
                std::string narrow;
                if (!p) narrow = "(null)";
                else if (wide || conv == 'S')  // ponytail: UTF-16 to ASCII only
                    for (auto* w = static_cast<const char16_t*>(p); *w; ++w) narrow.push_back(static_cast<char>(*w));
                else narrow = static_cast<const char*>(p);
                spec.push_back('s');
                emit(narrow.c_str());
                break;
            }
            default: unsupported(("printf conversion %" + std::string(1, conv)).c_str());
        }
    }
    return out;
}
int32_t stdio_common_vfprintf(uint64_t, FILE* f, const char* fmt, void*, char* va) {
    std::string s = ms_format(fmt, va);
    return std::fwrite(s.data(), 1, s.size(), f) == s.size() ? static_cast<int32_t>(s.size()) : -1;
}

// ---- heap and strings: the host libc, wrapped so the thunks see plain signatures ----
void* crt_malloc(uint64_t n) { return malloc(n); }
void crt_free(void* p) { free(p); }
int callnewh(uint64_t) { return 0; }  // no new handler: operator new reports the failure
int crt_strcmp(const char* a, const char* b) { return std::strcmp(a, b); }
uint64_t crt_strlen(const char* s) { return std::strlen(s); }
const char* crt_strstr(const char* a, const char* b) { return std::strstr(a, b); }
int crt_memcmp(const void* a, const void* b, uint64_t n) { return std::memcmp(a, b, n); }
void* crt_memcpy(void* d, const void* s, uint64_t n) { return std::memcpy(d, s, n); }
void* crt_memmove(void* d, const void* s, uint64_t n) { return std::memmove(d, s, n); }
void* crt_memset(void* d, int32_t c, uint64_t n) { return std::memset(d, c, n); }

// ---- vcruntime140: throw and the frame handlers are in eh.cpp ----
struct ExceptionData {
    const char* what;
    bool do_free;
};
void std_exception_copy(const ExceptionData* from, ExceptionData* to) {
    if (from->do_free && from->what) *to = {strdup(from->what), true};
    else *to = {from->what, false};
}
void std_exception_destroy(ExceptionData* d) {
    if (d->do_free) free(const_cast<char*>(d->what));
    *d = {nullptr, false};
}
[[noreturn]] void current_exception() { unsupported("__current_exception"); }

// ---- msvcp140: std::thread ----
struct Thrd {  // _Thrd_t; 16 bytes, so the MS ABI passes it by pointer
    void* handle;
    uint32_t id;
};
int32_t thrd_join(const Thrd* t, int32_t* result) {
    auto* o = static_cast<Object*>(t->handle);
    uint32_t code = wait_thread(o);
    close_object(o);
    if (result) *result = static_cast<int32_t>(code);
    return 0;  // _Thrd_success
}
uint32_t thrd_id() { return thread_id(); }
[[noreturn]] void throw_cpp_error(int32_t code) { unsupported(("std::_Throw_Cpp_error " + std::to_string(code)).c_str()); }
[[noreturn]] void xlength_error(const char* what) { unsupported(what); }

// ---- xgameruntime ----
int32_t xgame_init() { return 0; }  // S_OK
uint32_t xsystem_device_type() { return 7; }  // what tiny_game expects from Leonida's stub (XSystemDeviceType)

}  // namespace

#define RT "api-ms-win-crt-runtime-l1-1-0.dll"
#define IO "api-ms-win-crt-stdio-l1-1-0.dll"
#define HEAP "api-ms-win-crt-heap-l1-1-0.dll"
#define STR "api-ms-win-crt-string-l1-1-0.dll"
#define VC "vcruntime140.dll"
#define CPP "msvcp140.dll"
Exports crt_exports() {
    return {
        EXPORT_AS(RT, "__p___argc", p_argc), EXPORT_AS(RT, "__p___argv", p_argv),
        EXPORT_AS(RT, "_beginthreadex", beginthreadex), EXPORT_AS(RT, "_c_exit", noop),
        EXPORT_AS(RT, "_cexit", crt_cexit), EXPORT_AS(RT, "_configure_narrow_argv", return0_i),
        EXPORT_AS(RT, "_crt_atexit", crt_atexit), EXPORT_AS(RT, "_exit", crt_quick_exit),
        EXPORT_AS(RT, "_get_initial_narrow_environment", initial_env),
        EXPORT_AS(RT, "_initialize_narrow_environment", return0),
        EXPORT_AS(RT, "_initialize_onexit_table", initialize_onexit_table),
        EXPORT_AS(RT, "_initterm", initterm), EXPORT_AS(RT, "_initterm_e", initterm_e),
        EXPORT_AS(RT, "_invoke_watson", invoke_watson),
        EXPORT_AS(RT, "_register_onexit_function", register_onexit),
        EXPORT_AS(RT, "_register_thread_local_exe_atexit_callback", noop_p),  // ponytail: never called back
        EXPORT_AS(RT, "_seh_filter_exe", seh_filter_exe), EXPORT_AS(RT, "_set_app_type", return0_i),
        EXPORT_AS(RT, "exit", crt_exit), EXPORT_AS(RT, "terminate", crt_terminate),
        EXPORT_AS(IO, "__acrt_iob_func", iob), EXPORT_AS(IO, "__p__commode", p_commode),
        EXPORT_AS(IO, "__stdio_common_vfprintf", stdio_common_vfprintf), EXPORT_AS(IO, "_set_fmode", return0_i),
        EXPORT_AS(IO, "fclose", crt_fclose), EXPORT_AS(IO, "fflush", crt_fflush), EXPORT_AS(IO, "fopen", crt_fopen),
        EXPORT_AS(IO, "fread", crt_fread), EXPORT_AS(IO, "fwrite", crt_fwrite), EXPORT_AS(IO, "puts", crt_puts),
        EXPORT_AS(IO, "setvbuf", crt_setvbuf),
        EXPORT_AS(HEAP, "_callnewh", callnewh), EXPORT_AS(HEAP, "_set_new_mode", return0_i),
        EXPORT_AS(HEAP, "free", crt_free), EXPORT_AS(HEAP, "malloc", crt_malloc),
        EXPORT_AS(STR, "strcmp", crt_strcmp), EXPORT_AS(STR, "strlen", crt_strlen),
        EXPORT_AS("api-ms-win-crt-math-l1-1-0.dll", "__setusermatherr", noop_p),
        EXPORT_AS("api-ms-win-crt-locale-l1-1-0.dll", "_configthreadlocale", return0_i),
        EXPORT_AS("api-ms-win-crt-filesystem-l1-1-0.dll", "remove", crt_remove),
        EXPORT_AS(VC, "__current_exception", current_exception),
        EXPORT_AS(VC, "__current_exception_context", current_exception),
        EXPORT_AS(VC, "__std_exception_copy", std_exception_copy),
        EXPORT_AS(VC, "__std_exception_destroy", std_exception_destroy),
        EXPORT_AS(VC, "memcmp", crt_memcmp), EXPORT_AS(VC, "memcpy", crt_memcpy),
        EXPORT_AS(VC, "memmove", crt_memmove), EXPORT_AS(VC, "memset", crt_memset),
        EXPORT_AS(VC, "strstr", crt_strstr),
        EXPORT_AS(CPP, "?_Throw_Cpp_error@std@@YAXH@Z", throw_cpp_error),
        EXPORT_AS(CPP, "?_Xlength_error@std@@YAXPEBD@Z", xlength_error),
        EXPORT_AS(CPP, "?_Xout_of_range@std@@YAXPEBD@Z", xlength_error),
        EXPORT_AS(CPP, "_Cnd_do_broadcast_at_thread_exit", noop), EXPORT_AS(CPP, "_Thrd_id", thrd_id),
        EXPORT_AS(CPP, "_Thrd_join", thrd_join),
        EXPORT_AS("xgameruntime.dll", "XGameRuntimeInitialize", xgame_init),
        EXPORT_AS("xgameruntime.dll", "XGameRuntimeUninitialize", noop),
        EXPORT_AS("xgameruntime.dll", "XSystemGetDeviceType", xsystem_device_type),
    };
}

}  // namespace host
