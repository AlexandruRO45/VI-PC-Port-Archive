// The rest of the UCRT a retail game imports: math, strings, conversions, time, stdio and sorting, on the host libc.
// Windows differences handled here: long is 32-bit, wchar_t is UTF-16, struct tm has no fields past tm_isdst, and
// rand() is the MS linear congruential generator. ponytail: the C locale only, wide strings compared as code units.
#include "host.h"

#include <sys/timeb.h>

#include <cctype>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>

namespace host {
namespace {

// ---- errno and the invalid-parameter handler ----
thread_local int t_errno = 0;
int* crt_errno() { return &t_errno; }
int32_t get_errno(int* out) { return out ? (*out = t_errno, 0) : 22; }
int32_t set_errno(int v) { return t_errno = v, 0; }
void* g_invalid_handler = nullptr;
void* set_invalid_parameter_handler(void* h) { void* old = g_invalid_handler; g_invalid_handler = h; return old; }
[[noreturn]] void invalid_parameter() { fail("guest CRT reported an invalid parameter"); }

// ---- math: wrappers fix one overload of each libm function for the thunks ----
#define F1(name) float m_##name(float x) { return std::name(x); }
#define D1(name) double m_##name(double x) { return std::name(x); }
F1(acosf) F1(asinf) F1(atanf) F1(cbrtf) F1(ceilf) F1(cosf) F1(erff) F1(exp2f) F1(expf) F1(log10f) F1(log2f) F1(logf)
F1(sinf) F1(sinhf) F1(tanf)
D1(acos) D1(asin) D1(atan) D1(cos) D1(exp) D1(exp2) D1(log) D1(log2) D1(sin) D1(tan)
#undef F1
#undef D1
float m_atan2f(float y, float x) { return std::atan2f(y, x); }
float m_fmodf(float a, float b) { return std::fmodf(a, b); }
float m_powf(float a, float b) { return std::powf(a, b); }
float m_remainderf(float a, float b) { return std::remainderf(a, b); }
float m_nextafterf(float a, float b) { return std::nextafterf(a, b); }
float m_nexttowardf(float a, double b) { return std::nextafterf(a, static_cast<float>(b)); }  // MS long double = double
double m_atan2(double y, double x) { return std::atan2(y, x); }
double m_fmod(double a, double b) { return std::fmod(a, b); }
double m_pow(double a, double b) { return std::pow(a, b); }
double m_modf(double x, double* i) { return std::modf(x, i); }
float m_modff(float x, float* i) { return std::modff(x, i); }
int32_t m_lroundf(float x) { return static_cast<int32_t>(std::lroundf(x)); }
// _dtest/_fdtest: _FINITE -1, _INFCODE 1, _NANCODE 2, _DENORM -2, 0 for zero.
template <typename T> int16_t fp_class(T v) {
    switch (std::fpclassify(v)) {
        case FP_INFINITE: return 1;
        case FP_NAN: return 2;
        case FP_SUBNORMAL: return -2;
        case FP_ZERO: return 0;
        default: return -1;
    }
}
int16_t dtest(const double* v) { return fp_class(*v); }
int16_t fdtest(const float* v) { return fp_class(*v); }
int32_t fdsign(float v) { return std::signbit(v) ? 0x8000 : 0; }

// ---- character classes and narrow strings (ASCII outside the C locale's range) ----
int ascii(int c) { return c >= 0 && c < 128 ? c : 0; }
int c_isalnum(int c) { return std::isalnum(ascii(c)); }
int c_isalpha(int c) { return std::isalpha(ascii(c)); }
int c_isdigit(int c) { return std::isdigit(ascii(c)); }
int c_islower(int c) { return std::islower(ascii(c)); }
int c_isspace(int c) { return std::isspace(ascii(c)); }
int c_isxdigit(int c) { return std::isxdigit(ascii(c)); }
int c_tolower(int c) { return c >= 'A' && c <= 'Z' ? c + 32 : c; }
int c_toupper(int c) { return c >= 'a' && c <= 'z' ? c - 32 : c; }
int s_stricmp(const char* a, const char* b) { return strcasecmp(a, b); }
int s_strnicmp(const char* a, const char* b, uint64_t n) { return strncasecmp(a, b, n); }
char* s_strcat(char* d, const char* s) { return std::strcat(d, s); }
char* s_strcpy(char* d, const char* s) { return std::strcpy(d, s); }
char* s_strncat(char* d, const char* s, uint64_t n) { return std::strncat(d, s, n); }
char* s_strncpy(char* d, const char* s, uint64_t n) { return std::strncpy(d, s, n); }
uint64_t s_strcspn(const char* a, const char* b) { return std::strcspn(a, b); }
const char* s_strpbrk(const char* a, const char* b) { return std::strpbrk(a, b); }
thread_local char* t_strtok;
char* s_strtok(char* s, const char* d) { return strtok_r(s, d, &t_strtok); }
char* s_strtok_s(char* s, const char* d, char** ctx) { return strtok_r(s, d, ctx); }
int32_t s_strcat_s(char* d, uint64_t size, const char* s) {
    if (!d || !s || !size) return 22;
    uint64_t a = strnlen(d, size), b = std::strlen(s);
    if (a + b >= size) return d[0] = 0, 34;  // ERANGE
    std::memcpy(d + a, s, b + 1);
    return 0;
}
int32_t s_strncpy_s(char* d, uint64_t size, const char* s, uint64_t count) {  // count ~0 is _TRUNCATE
    if (!d || !size) return 22;
    if (!s) return d[0] = 0, 22;
    uint64_t n = strnlen(s, count == ~0ull ? size : count);
    if (n >= size) {
        if (count != ~0ull) return d[0] = 0, 34;
        n = size - 1;
        std::memcpy(d, s, n);
        d[n] = 0;
        return 80;  // STRUNCATE
    }
    std::memcpy(d, s, n);
    d[n] = 0;
    return 0;
}

// ---- wide strings: Windows wchar_t is 16-bit ----
uint64_t w_len(const char16_t* s) { const char16_t* p = s; while (*p) ++p; return p - s; }
int w_cmp(const char16_t* a, const char16_t* b) {
    for (; *a && *a == *b; ++a, ++b) {}
    return *a < *b ? -1 : *a > *b;
}
char16_t w_lower(char16_t c) { return c >= 'A' && c <= 'Z' ? c + 32 : c; }
int w_nicmp(const char16_t* a, const char16_t* b, uint64_t n) {
    for (; n; --n, ++a, ++b) {
        char16_t x = w_lower(*a), y = w_lower(*b);
        if (x != y || !x) return x < y ? -1 : x > y;
    }
    return 0;
}
int w_icoll(const char16_t* a, const char16_t* b) { return w_nicmp(a, b, ~0ull); }
char16_t* w_ncpy(char16_t* d, const char16_t* s, uint64_t n) {
    uint64_t i = 0;
    for (; i < n && s[i]; ++i) d[i] = s[i];
    for (; i < n; ++i) d[i] = 0;
    return d;
}
int32_t w_cat_s(char16_t* d, uint64_t size, const char16_t* s) {
    if (!d || !s || !size) return 22;
    uint64_t a = 0, b = w_len(s);
    while (a < size && d[a]) ++a;
    if (a + b >= size) return d[0] = 0, 34;
    std::memcpy(d + a, s, (b + 1) * 2);
    return 0;
}

// ---- conversions: Windows long is 32-bit ----
int32_t c_atoi(const char* s) { return std::atoi(s); }
int64_t c_atoll(const char* s) { return std::atoll(s); }
double c_atof(const char* s) { return std::atof(s); }
double c_strtod(const char* s, char** e) { return std::strtod(s, e); }
float c_strtof(const char* s, char** e) { return std::strtof(s, e); }
int32_t c_strtol(const char* s, char** e, int b) {
    long long v = std::strtoll(s, e, b);
    if (v > INT32_MAX) return t_errno = ERANGE, INT32_MAX;
    if (v < INT32_MIN) return t_errno = ERANGE, INT32_MIN;
    return static_cast<int32_t>(v);
}
uint32_t c_strtoul(const char* s, char** e, int b) {
    unsigned long long v = std::strtoull(s, e, b);
    bool negative = false;
    for (const char* p = s; *p; ++p) if (!std::isspace(static_cast<unsigned char>(*p))) { negative = *p == '-'; break; }
    if (!negative && v > UINT32_MAX) return t_errno = ERANGE, UINT32_MAX;
    return static_cast<uint32_t>(v);
}
int64_t c_strtoll(const char* s, char** e, int b) { return std::strtoll(s, e, b); }
uint64_t c_strtoull(const char* s, char** e, int b) { return std::strtoull(s, e, b); }
char* c_itoa(int32_t v, char* buf, int radix) {
    if (radix == 10) return std::snprintf(buf, 34, "%d", v), buf;
    uint32_t u = static_cast<uint32_t>(v);  // other radixes print the two's-complement bits, as the UCRT does
    char tmp[34];
    int n = 0;
    do tmp[n++] = "0123456789abcdefghijklmnopqrstuvwxyz"[u % radix]; while (u /= radix);
    for (int i = 0; i < n; ++i) buf[i] = tmp[n - 1 - i];
    buf[n] = 0;
    return buf;
}

// ---- time: 64-bit time_t; MS struct tm is the first nine ints of the host's ----
struct MsTm { int v[9]; };
void to_ms(const std::tm& t, MsTm* out) { std::memcpy(out->v, &t, sizeof out->v); }
std::tm from_ms(const MsTm* in) {
    std::tm t{};
    std::memcpy(&t, in->v, sizeof in->v);
    return t;
}
int64_t c_time64(int64_t* out) { int64_t t = std::time(nullptr); if (out) *out = t; return t; }
double c_difftime64(int64_t a, int64_t b) { return static_cast<double>(a - b); }
int32_t c_gmtime64_s(MsTm* out, const int64_t* t) {
    std::tm r;
    time_t v = *t;
    if (!out || !gmtime_r(&v, &r)) return 22;
    return to_ms(r, out), 0;
}
int32_t c_localtime64_s(MsTm* out, const int64_t* t) {
    std::tm r;
    time_t v = *t;
    if (!out || !localtime_r(&v, &r)) return 22;
    return to_ms(r, out), 0;
}
thread_local MsTm t_tm;
MsTm* c_gmtime64(const int64_t* t) { return c_gmtime64_s(&t_tm, t) ? nullptr : &t_tm; }
int64_t c_mktime64(MsTm* in) { std::tm t = from_ms(in); int64_t r = std::mktime(&t); to_ms(t, in); return r; }
int64_t c_mkgmtime64(MsTm* in) { std::tm t = from_ms(in); int64_t r = timegm(&t); to_ms(t, in); return r; }
struct MsTimeb { int64_t time; uint16_t millitm; int16_t timezone, dstflag; };
void c_ftime64(MsTimeb* out) {
    timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    std::tm local;
    localtime_r(&ts.tv_sec, &local);
    *out = {ts.tv_sec, static_cast<uint16_t>(ts.tv_nsec / 1000000), static_cast<int16_t>(-local.tm_gmtoff / 60),
            static_cast<int16_t>(local.tm_isdst > 0)};
}
uint64_t c_strftime(char* buf, uint64_t n, const char* fmt, const MsTm* in) {
    std::tm t = from_ms(in);
    return std::strftime(buf, n, fmt, &t);
}

// ---- stdio over host FILE*: fpos_t is 64-bit on both ----
int32_t f_seek(FILE* f, int32_t off, int whence) { return std::fseek(f, off, whence); }
int32_t f_seeki64(FILE* f, int64_t off, int whence) { return fseeko(f, off, whence); }
int32_t f_tell(FILE* f) { return static_cast<int32_t>(std::ftell(f)); }
int f_getc(FILE* f) { return std::fgetc(f); }
int f_putc(int c, FILE* f) { return std::fputc(c, f); }
int f_ungetc(int c, FILE* f) { return std::ungetc(c, f); }
void f_rewind(FILE* f) { std::rewind(f); }
int f_getpos(FILE* f, fpos_t* p) { return std::fgetpos(f, p); }
int f_setpos(FILE* f, const fpos_t* p) { return std::fsetpos(f, p); }
void f_lock(FILE* f) { flockfile(f); }
void f_unlock(FILE* f) { funlockfile(f); }
FILE* f_wopen(const char16_t* path, const char16_t* mode) {
    std::string m, n = narrow(mode);
    for (char c : n)
        if (c != 't') m.push_back(c);
    return std::fopen(to_host_path(narrow(path).c_str()).c_str(), m.c_str());
}
int32_t f_wstat64(const char16_t*, void*) { return t_errno = ENOENT, -1; }  // ponytail: no _stat64 layout yet

// ---- sorting with a guest comparator, and the MS rand ----
using Compare = int32_t(MSABI*)(const void*, const void*);
int sort_thunk(void* fn, const void* a, const void* b) { return call_guest(reinterpret_cast<Compare>(fn), a, b); }
void c_qsort(void* base, uint64_t n, uint64_t size, Compare cmp) {
    qsort_r(base, n, size, reinterpret_cast<void*>(cmp), sort_thunk);
}
const void* c_bsearch(const void* key, const void* base, uint64_t n, uint64_t size, Compare cmp) {
    auto* lo = static_cast<const char*>(base);
    while (n) {
        const char* mid = lo + (n / 2) * size;
        int32_t r = call_guest(cmp, key, static_cast<const void*>(mid));
        if (r == 0) return mid;
        if (r > 0) lo = mid + size, n -= n / 2 + 1;
        else n /= 2;
    }
    return nullptr;
}
thread_local uint32_t t_rand = 1;
void c_srand(uint32_t seed) { t_rand = seed; }
int32_t c_rand() { return t_rand = t_rand * 214013u + 2531011u, (t_rand >> 16) & 0x7FFF; }

// ---- localeconv: the C locale's decimal point and empty groupings ----
struct MsLconv { const char* text[10]; char chars[8]; const char16_t* wide[8]; };
MsLconv g_lconv = {{".", "", "", "", "", "", "", "", "", ""}, {127, 127, 127, 127, 127, 127, 127, 127},
                   {u".", u"", u"", u"", u"", u"", u"", u""}};
MsLconv* c_localeconv() { return &g_lconv; }

}  // namespace

#define A(dll, name, fn) EXPORT_AS("api-ms-win-crt-" dll "-l1-1-0.dll", name, fn)
#define M(name) A("math", #name, m_##name)
Exports crt_lib_exports() {
    return {
        A("runtime", "_errno", crt_errno), A("runtime", "_get_errno", get_errno), A("runtime", "_set_errno", set_errno),
        A("runtime", "_set_invalid_parameter_handler", set_invalid_parameter_handler),
        A("runtime", "_invalid_parameter_noinfo", invalid_parameter),
        A("runtime", "_invalid_parameter_noinfo_noreturn", invalid_parameter),
        M(acosf), M(asinf), M(atanf), M(cbrtf), M(ceilf), M(cosf), M(erff), M(exp2f), M(expf), M(log10f), M(log2f),
        M(logf), M(sinf), M(sinhf), M(tanf), M(acos), M(asin), M(atan), M(cos), M(exp), M(exp2), M(log), M(log2),
        M(sin), M(tan), M(atan2f), M(fmodf), M(powf), M(remainderf), M(nextafterf), M(nexttowardf), M(atan2), M(fmod),
        M(pow), M(modf), M(modff), M(lroundf),
        A("math", "_dtest", dtest), A("math", "_fdtest", fdtest), A("math", "_fdsign", fdsign),
        A("string", "isalnum", c_isalnum), A("string", "isalpha", c_isalpha), A("string", "isdigit", c_isdigit),
        A("string", "islower", c_islower), A("string", "isspace", c_isspace), A("string", "isxdigit", c_isxdigit),
        A("string", "tolower", c_tolower), A("string", "toupper", c_toupper),
        A("string", "_stricmp", s_stricmp), A("string", "_strnicmp", s_strnicmp), A("string", "strcat", s_strcat),
        A("string", "strcpy", s_strcpy), A("string", "strncat", s_strncat), A("string", "strncpy", s_strncpy),
        A("string", "strcspn", s_strcspn), A("string", "strpbrk", s_strpbrk), A("string", "strtok", s_strtok),
        A("string", "strtok_s", s_strtok_s), A("string", "strcat_s", s_strcat_s), A("string", "strncpy_s", s_strncpy_s),
        A("string", "wcslen", w_len), A("string", "wcscmp", w_cmp), A("string", "wcscoll", w_cmp),
        A("string", "_wcsnicmp", w_nicmp), A("string", "_wcsicoll", w_icoll), A("string", "wcsncpy", w_ncpy),
        A("string", "wcscat_s", w_cat_s),
        A("convert", "atoi", c_atoi), A("convert", "atoll", c_atoll), A("convert", "atof", c_atof),
        A("convert", "strtod", c_strtod), A("convert", "strtof", c_strtof), A("convert", "strtol", c_strtol),
        A("convert", "strtoul", c_strtoul), A("convert", "strtoll", c_strtoll), A("convert", "strtoull", c_strtoull),
        A("convert", "_strtoi64", c_strtoll), A("convert", "_strtoui64", c_strtoull), A("convert", "_itoa", c_itoa),
        A("time", "_time64", c_time64), A("time", "_difftime64", c_difftime64), A("time", "_gmtime64_s", c_gmtime64_s),
        A("time", "_localtime64_s", c_localtime64_s), A("time", "_gmtime64", c_gmtime64),
        A("time", "_mktime64", c_mktime64), A("time", "_mkgmtime64", c_mkgmtime64), A("time", "_ftime64", c_ftime64),
        A("time", "strftime", c_strftime),
        A("stdio", "fseek", f_seek), A("stdio", "_fseeki64", f_seeki64), A("stdio", "ftell", f_tell),
        A("stdio", "fgetc", f_getc), A("stdio", "fputc", f_putc), A("stdio", "ungetc", f_ungetc),
        A("stdio", "rewind", f_rewind), A("stdio", "fgetpos", f_getpos), A("stdio", "fsetpos", f_setpos),
        A("stdio", "_wfopen", f_wopen), A("filesystem", "_lock_file", f_lock), A("filesystem", "_unlock_file", f_unlock),
        A("filesystem", "_wstat64", f_wstat64),
        A("utility", "qsort", c_qsort), A("utility", "bsearch", c_bsearch), A("utility", "srand", c_srand),
        A("utility", "rand", c_rand), A("locale", "localeconv", c_localeconv),
    };
}
#undef A
#undef M

}  // namespace host
