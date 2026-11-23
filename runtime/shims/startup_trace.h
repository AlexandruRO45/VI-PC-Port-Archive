// Opt-in startup trace: gta6rt --trace FILE.
//
// One JSON object per line, flushed per event so the file survives a crash. Off by default; every entry point is a
// single relaxed load when it is off. Events:
//   {"seq":1,"t":0.000,"kind":"trace_open"}
//   {"seq":2,"t":0.001,"kind":"map","module":"a.dll","base":"0x...","size":"0x2000","relocated":false}
//   {"seq":3,"t":0.002,"kind":"tls_callback","module":"a.dll","at":"a.dll+0x1200","reason":1}
//   {"seq":4,"t":0.002,"kind":"dllmain","module":"a.dll","result":1}
//   {"seq":5,"t":0.010,"kind":"unimplemented_call","import":"x.dll!Foo","caller":"a.dll+0x3f10"}   (first call only)
//   {"seq":6,"t":0.011,"kind":"debug_string","text":"..."}
//   {"seq":7,"t":0.020,"kind":"exception","code":"0xc0000005","at":"a.dll+0x4a00","access":"read","address":"0x0",
//        "stack":["a.dll+0x4a00","a.dll+0x2210"]}
//   {"seq":8,"t":0.030,"kind":"exit","code":0}
//
// The exception handler is vectored, logs and returns EXCEPTION_CONTINUE_SEARCH: it never changes what the process
// does (except LEONIDA_SKIP_INT3, below).
// OutputDebugStringA/W raise DBG_PRINTEXCEPTION_C / DBG_PRINTEXCEPTION_WIDE_C when no debugger is attached, so the
// same handler records their text without a shim.
//
// The file can quote whatever the game prints: it stays on the player's machine (AGENTS.md, "No game data").
#pragma once

#include <windows.h>
#include <set>

#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>
#include <vector>
#include <cwchar>

#include "shims.h"

namespace rt::startup_trace {

namespace detail {
inline std::atomic<bool> on{false};
// Intentionally immortal: vectored exception handlers or late teardown paths can still run after normal statics would
// have been destroyed.
inline std::mutex& mutex() {
    static auto* m = new std::mutex;
    return *m;
}
inline FILE* file = nullptr;
inline unsigned long long seq = 0;
inline std::chrono::steady_clock::time_point start;
inline void* veh = nullptr;
inline thread_local bool in_handler = false;
// With nothing to handle it, OutputDebugStringW's DBG_PRINTEXCEPTION_WIDE_C falls back to OutputDebugStringA on its
// own ANSI copy (ExceptionInformation[3]), a second, lossy exception for the same text. Every debug string on the
// thread resets this, so only the exception right after the wide one, with that same pointer, is skipped.
inline thread_local ULONG_PTR wide_fallback = 0;

inline void append_escaped(std::string& out, const char* s, size_t n) {
    static const char hex[] = "0123456789abcdef";
    for (size_t i = 0; i < n; ++i) {
        const unsigned char c = static_cast<unsigned char>(s[i]);
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (c < 0x20) {
                    out += "\\u00";
                    out += hex[c >> 4];
                    out += hex[c & 15];
                } else {
                    out += static_cast<char>(c);  // UTF-8 passes through
                }
        }
    }
}
}  // namespace detail

inline bool enabled() noexcept { return detail::on.load(std::memory_order_acquire); }

// A JSON string literal, quotes included.
inline std::string quote(const std::string& s) {
    std::string out = "\"";
    detail::append_escaped(out, s.data(), s.size());
    out += '"';
    return out;
}

inline std::string hex(uint64_t v) {
    char b[24];
    std::snprintf(b, sizeof b, "0x%llx", static_cast<unsigned long long>(v));
    return b;
}

// "module+0xRVA" for a mapped image or a host module, else the bare address.
inline std::string where(uint64_t address) { return shims::module_offset(address); }

// `fields` is the rest of the object, already JSON: `"module":"a.dll","result":1`. Empty is allowed.
inline void event(const char* kind, const std::string& fields = {}) {
    if (!enabled()) return;
    const std::lock_guard lock(detail::mutex());
    if (!detail::file) return;
    const double t = std::chrono::duration<double>(std::chrono::steady_clock::now() - detail::start).count();
    std::fprintf(detail::file, "{\"seq\":%llu,\"t\":%.6f,\"kind\":\"%s\"%s%s}\n", ++detail::seq, t, kind,
                 fields.empty() ? "" : ",", fields.c_str());
    std::fflush(detail::file);
}

namespace detail {
constexpr DWORD kDbgPrint = 0x40010006;      // DBG_PRINTEXCEPTION_C
constexpr DWORD kDbgPrintWide = 0x4001000A;  // DBG_PRINTEXCEPTION_WIDE_C
constexpr DWORD kCppException = 0xE06D7363;  // MSVC throw: logged with its address, never its payload
constexpr int kMaxFrames = 16;

inline std::string stack_from(const CONTEXT& start_context) {
    CONTEXT c = start_context;
    std::string out = "[";
    for (int i = 0; i < kMaxFrames && c.Rip; ++i) {
        if (i) out += ',';
        out += quote(where(c.Rip));
        DWORD64 image_base = 0;
        PRUNTIME_FUNCTION fn = RtlLookupFunctionEntry(c.Rip, &image_base, nullptr);
        if (!fn) {  // a leaf function: the return address is at [rsp]
            if (!c.Rsp || IsBadReadPtr(reinterpret_cast<void*>(c.Rsp), 8)) break;
            c.Rip = *reinterpret_cast<DWORD64*>(c.Rsp);
            c.Rsp += 8;
            continue;
        }
        void* handler_data = nullptr;
        DWORD64 frame = 0;
        RtlVirtualUnwind(UNW_FLAG_NHANDLER, image_base, c.Rip, fn, &c, &handler_data, &frame, nullptr);
    }
    return out + "]";
}

inline std::string utf8_from_wide(const wchar_t* w, int len) {
    const int bytes = len > 0 ? WideCharToMultiByte(CP_UTF8, 0, w, len, nullptr, 0, nullptr, nullptr) : 0;
    std::string s(bytes > 0 ? bytes : 0, '\0');
    if (bytes > 0) WideCharToMultiByte(CP_UTF8, 0, w, len, s.data(), bytes, nullptr, nullptr);
    return s;
}

// OutputDebugStringA text is in the ANSI code page, not UTF-8: convert it, or a byte such as 0xE9 would land in the
// file as is and break the JSON lines for a strict UTF-8 reader.
inline std::string utf8_from_ansi(const char* a, int len) {
    const int chars = len > 0 ? MultiByteToWideChar(CP_ACP, 0, a, len, nullptr, 0) : 0;
    if (chars <= 0) return {};
    std::wstring w(static_cast<size_t>(chars), L'\0');
    MultiByteToWideChar(CP_ACP, 0, a, len, w.data(), chars);
    return utf8_from_wide(w.data(), chars);
}

inline std::string debug_text(const EXCEPTION_RECORD& r) {
    // Parameters: [0] length in characters including the terminator, [1] the string.
    if (r.NumberParameters < 2 || !r.ExceptionInformation[1]) return {};
    const size_t n = static_cast<size_t>(r.ExceptionInformation[0]);
    if (r.ExceptionCode == kDbgPrintWide) {
        const auto* w = reinterpret_cast<const wchar_t*>(r.ExceptionInformation[1]);
        return utf8_from_wide(w, static_cast<int>(n ? n - 1 : wcslen(w)));
    }
    const auto* a = reinterpret_cast<const char*>(r.ExceptionInformation[1]);
    return utf8_from_ansi(a, static_cast<int>(n ? n - 1 : strlen(a)));
}

inline void log_exception(const EXCEPTION_POINTERS& info) {
    const EXCEPTION_RECORD& r = *info.ExceptionRecord;
    try {
        if (r.ExceptionCode == kDbgPrint || r.ExceptionCode == kDbgPrintWide) {
            const ULONG_PTR fallback = wide_fallback;
            const bool wide = r.ExceptionCode == kDbgPrintWide;
            wide_fallback = wide && r.NumberParameters >= 4 ? r.ExceptionInformation[3] : 0;
            if (!wide && fallback && r.NumberParameters >= 2 && r.ExceptionInformation[1] == fallback) return;
            std::string text = debug_text(r);
            while (!text.empty() && (text.back() == '\n' || text.back() == '\r')) text.pop_back();
            event("debug_string", "\"text\":" + quote(text));
        } else {
            std::string f = "\"code\":" + quote(hex(r.ExceptionCode)) +
                            ",\"at\":" + quote(where(reinterpret_cast<uint64_t>(r.ExceptionAddress)));
            if (r.ExceptionCode == EXCEPTION_ACCESS_VIOLATION && r.NumberParameters >= 2) {
                const auto op = r.ExceptionInformation[0];
                f += ",\"access\":" + quote(op == 0 ? "read" : op == 1 ? "write" : "execute") +
                     ",\"address\":" + quote(hex(r.ExceptionInformation[1]));
            }
            if (r.ExceptionCode == kCppException) f += ",\"cpp\":true";
            if (info.ContextRecord) f += ",\"stack\":" + stack_from(*info.ContextRecord);
            event("exception", f);
        }
    } catch (...) {
        // Out of memory while formatting: drop the event rather than disturb the process.
    }
}

// LEONIDA_SKIP_INT3 (diagnostic, with --trace): an int3 in guest code is logged once per address and stepped over,
// so a boot can be followed past the title's soft asserts (an int3 followed by code that carries on). Off by default:
// a console title never reaches one in a correct run, so each skipped address is a finding, not a fix.
inline bool skip_int3(EXCEPTION_POINTERS* info) {
    static const bool skip = GetEnvironmentVariableW(L"LEONIDA_SKIP_INT3", nullptr, 0) != 0;
    if (!skip || info->ExceptionRecord->ExceptionCode != EXCEPTION_BREAKPOINT || !info->ContextRecord) return false;
    const auto* at = static_cast<const unsigned char*>(info->ExceptionRecord->ExceptionAddress);
    if (!at || *at != 0xCC || info->ContextRecord->Rip != reinterpret_cast<DWORD64>(at)) return false;
    static std::mutex m;
    static std::set<const void*> seen;
    bool first = false;
    {
        const std::lock_guard lock(m);
        first = seen.insert(at).second;
    }
    if (first) event("int3_skipped", "\"at\":" + quote(where(reinterpret_cast<uint64_t>(at))));
    info->ContextRecord->Rip += 1;
    return true;
}

// LEONIDA_PROBE=<rva>[,<rva>...] (diagnostic, with --trace): a breakpoint at each RVA of the main image (hex, as
// `where` prints them). Each hit logs a "probe" event with the general registers and the qword at rsp (the return
// address at a function's first instruction), then the original byte is put back, the instruction runs under the trap
// flag, and the breakpoint is re-armed after it. The first 16 hits of each probe are logged (LEONIDA_PROBE_MAX=N for N). Answers "which call
// returned what" on the player's machine without a debugger; nothing of the image is written to the trace.
struct Probe {
    unsigned char* at;
    unsigned char original;
    unsigned hits;
};
inline std::vector<Probe>& probes() {
    static std::vector<Probe> list;
    return list;
}
inline std::mutex& probe_mutex() {
    static std::mutex m;
    return m;
}
inline thread_local unsigned char* probe_rearm = nullptr;

inline void write_code_byte(unsigned char* at, unsigned char value) {
    DWORD old = 0;
    if (!VirtualProtect(at, 1, PAGE_EXECUTE_READWRITE, &old)) return;
    *at = value;
    VirtualProtect(at, 1, old, &old);
    FlushInstructionCache(GetCurrentProcess(), at, 1);
}

inline bool probe(EXCEPTION_POINTERS* info) {
    if (probes().empty() || !info->ContextRecord) return false;
    CONTEXT& c = *info->ContextRecord;
    const DWORD code = info->ExceptionRecord->ExceptionCode;
    if (code == EXCEPTION_SINGLE_STEP && probe_rearm) {
        write_code_byte(probe_rearm, 0xCC);
        probe_rearm = nullptr;
        return true;
    }
    if (code != EXCEPTION_BREAKPOINT) return false;
    auto* at = static_cast<unsigned char*>(info->ExceptionRecord->ExceptionAddress);
    Probe* hit = nullptr;
    unsigned count = 0;
    {
        const std::lock_guard lock(probe_mutex());
        for (auto& p : probes())
            if (p.at == at) {
                hit = &p;
                count = ++p.hits;
                break;
            }
    }
    if (!hit || c.Rip != reinterpret_cast<DWORD64>(at)) return false;
    static const unsigned max_hits = [] {
        wchar_t v[16] = {};
        const DWORD n = GetEnvironmentVariableW(L"LEONIDA_PROBE_MAX", v, 16);
        const unsigned long parsed = (n && n < 16) ? std::wcstoul(v, nullptr, 10) : 0;
        return parsed ? static_cast<unsigned>(parsed) : 16u;
    }();
    if (count <= max_hits) {
        char regs[640];
        const auto ret = c.Rsp ? *reinterpret_cast<const uint64_t*>(c.Rsp) : 0;
        std::snprintf(regs, sizeof regs,
                      "\"hit\":%u,\"rax\":\"%llx\",\"rbx\":\"%llx\",\"rcx\":\"%llx\",\"rdx\":\"%llx\",\"rsi\":\"%llx\","
                      "\"rdi\":\"%llx\",\"r8\":\"%llx\",\"r9\":\"%llx\",\"r12\":\"%llx\",\"r13\":\"%llx\",\"r14\":\"%llx\","
                      "\"r15\":\"%llx\",\"rsp\":\"%llx\",\"at_rsp\":",
                      count, c.Rax, c.Rbx, c.Rcx, c.Rdx, c.Rsi, c.Rdi, c.R8, c.R9, c.R12, c.R13, c.R14, c.R15, c.Rsp);
        const std::string where_at = quote(where(reinterpret_cast<uint64_t>(at)));
        event("probe", "\"at\":" + where_at + "," + regs + quote(where(ret)));
        std::fprintf(stderr, "[probe] %s hit %u: rax=%llx rcx=%llx rdx=%llx r8=%llx r9=%llx ret=%s\n",
                     where(reinterpret_cast<uint64_t>(at)).c_str(), count, c.Rax, c.Rcx, c.Rdx, c.R8, c.R9,
                     where(ret).c_str());
        // LEONIDA_PROBE_MEM=1: also the first 8 qwords at rdi and rsi (read with ReadProcessMemory, so a bad pointer
        // prints "-"). Local diagnostics only: these bytes can be game data, so never paste them anywhere public.
        static const bool mem = GetEnvironmentVariableW(L"LEONIDA_PROBE_MEM", nullptr, 0) != 0;
        if (mem) {
            for (const auto& [name, address] : {std::pair<const char*, DWORD64>{"rdi", c.Rdi}, {"rsi", c.Rsi}}) {
                uint64_t q[8] = {};
                SIZE_T got = 0;
                const bool ok = address && ReadProcessMemory(GetCurrentProcess(), reinterpret_cast<const void*>(address),
                                                             q, sizeof q, &got) && got == sizeof q;
                if (!ok) {
                    std::fprintf(stderr, "[probe]   [%s] -\n", name);
                    continue;
                }
                std::fprintf(stderr, "[probe]   [%s] %llx %llx %llx %llx %llx %llx %llx %llx\n", name, q[0], q[1], q[2],
                             q[3], q[4], q[5], q[6], q[7]);
            }
        }
    }
    write_code_byte(at, hit->original);
    c.EFlags |= 0x100;  // trap after the original instruction, then re-arm
    probe_rearm = at;
    return true;
}

inline LONG CALLBACK handler(EXCEPTION_POINTERS* info) {
    if (!enabled() || in_handler || !info || !info->ExceptionRecord) return EXCEPTION_CONTINUE_SEARCH;
    if (probe(info)) return EXCEPTION_CONTINUE_EXECUTION;
    if (skip_int3(info)) return EXCEPTION_CONTINUE_EXECUTION;
    in_handler = true;
#ifdef _MSC_VER
    // Under /EHsc, catch(...) does not catch SEH faults (e.g. a bad DBG_PRINTEXCEPTION_C pointer or corrupt unwind).
    // Keep the __try frame free of C++ destructors so in_handler is always reset and the exception still propagates.
    __try {
        log_exception(*info);
    } __except (EXCEPTION_EXECUTE_HANDLER) {
    }
#else
    struct Reset {
        ~Reset() { in_handler = false; }
    } reset;
    log_exception(*info);
#endif
    in_handler = false;
    return EXCEPTION_CONTINUE_SEARCH;
}
}  // namespace detail

inline void close();

// Arms LEONIDA_PROBE in the mapped main image (see detail::probe). Call before any image code runs. RVAs outside the
// image are ignored with a line on stderr.
inline void install_probes(uint8_t* base, uint32_t size) {
    wchar_t value[1024] = {};
    const DWORD length = GetEnvironmentVariableW(L"LEONIDA_PROBE", value, 1024);
    if (!length || length >= 1024 || !base) return;
    const std::lock_guard lock(detail::probe_mutex());
    for (wchar_t* cursor = value; *cursor;) {
        wchar_t* end = nullptr;
        const unsigned long long rva = std::wcstoull(cursor, &end, 16);
        if (end == cursor) break;
        if (rva && rva < size) {
            unsigned char* at = base + rva;
            detail::probes().push_back({at, *at, 0});
            detail::write_code_byte(at, 0xCC);
            std::fprintf(stderr, "[probe] armed at +0x%llx\n", rva);
        } else {
            std::fprintf(stderr, "[probe] +0x%llx is outside the image; ignored\n", rva);
        }
        cursor = end;
        while (*cursor == L',' || *cursor == L' ') ++cursor;
    }
}

// Creates the trace file and installs the exception handler. The file must not exist yet ("x" mode): the trace is
// opened before the loader reads anything, so overwriting would let `--trace` destroy an input, such as the image
// itself. Returns false if the file exists or cannot be created; `error`, when given, receives the errno.
inline bool open(const wchar_t* path, int* error = nullptr) {
    {
        const std::lock_guard lock(detail::mutex());
        if (detail::file) return true;
        const errno_t e = _wfopen_s(&detail::file, path, L"wxb");
        if (e != 0 || !detail::file) {
            detail::file = nullptr;
            if (error) *error = e ? e : EINVAL;
            return false;
        }
        detail::start = std::chrono::steady_clock::now();
        detail::seq = 0;
        static const int atexit_registered = std::atexit(&close);
        (void)atexit_registered;
    }
    detail::veh = AddVectoredExceptionHandler(1, &detail::handler);
    detail::on.store(true, std::memory_order_release);
    event("trace_open", "\"pid\":" + std::to_string(GetCurrentProcessId()));
    return true;
}

// Removes the handler and closes the file (tests and process exit via atexit).
inline void close() {
    detail::on.store(false, std::memory_order_release);
    if (detail::veh) RemoveVectoredExceptionHandler(detail::veh);
    detail::veh = nullptr;
    const std::lock_guard lock(detail::mutex());
    if (detail::file) std::fclose(detail::file);
    detail::file = nullptr;
}

}  // namespace rt::startup_trace
