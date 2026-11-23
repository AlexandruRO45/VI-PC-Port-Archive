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
// does. OutputDebugStringA/W raise DBG_PRINTEXCEPTION_C / DBG_PRINTEXCEPTION_WIDE_C when no debugger is attached, so
// the same handler records their text without a shim.
//
// The file can quote whatever the game prints: it stays on the player's machine (AGENTS.md, "No game data").
#pragma once

#include <windows.h>

#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>

#include "host_tls.h"
#include "shims.h"

// src/host_tls_pad.cpp: the runtime's only static TLS variable, the block the title's template is copied into.
unsigned char* guest_tls_pad();

// rt::t_in_handler, rt::t_wide_fallback and rt::t_single_stepping replace the thread_local variables this header
// used to hold: the runtime's static TLS block belongs to the mapped title (host_tls.h).

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
// With nothing to handle it, OutputDebugStringW's DBG_PRINTEXCEPTION_WIDE_C falls back to OutputDebugStringA on its
// own ANSI copy (ExceptionInformation[3]), a second, lossy exception for the same text. Every debug string on the
// thread resets this, so only the exception right after the wide one, with that same pointer, is skipped.

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
constexpr DWORD kGuardPage = 0x80000001;      // STATUS_GUARD_PAGE_VIOLATION
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

// Write watch on the page holding `address`. PAGE_GUARD faults on the next access and Windows clears the guard bit on
// the way in. The handler logs the writer, re-arms, and sets the trap flag for one instruction so the retry does not
// fault on that same guard again. Capped: the title's own install writes here too, so the interesting hit is a later one.
inline std::atomic<uint64_t> watch_address{0};
inline std::atomic<uint64_t> watched_page{0};
inline std::atomic<DWORD> watch_protect{0};
inline std::atomic<int> watch_hits{0};
inline constexpr int kWatchCap = 8;

inline bool watch_page(uint64_t address) {
    const uint64_t page = address & ~static_cast<uint64_t>(0xfff);
    MEMORY_BASIC_INFORMATION info{};
    if (VirtualQuery(reinterpret_cast<const void*>(page), &info, sizeof info) != sizeof info) return false;
    if (info.State != MEM_COMMIT) return false;
    DWORD old = 0;
    if (!VirtualProtect(reinterpret_cast<void*>(page), 0x1000, info.Protect | PAGE_GUARD, &old)) return false;
    watch_protect.store(old);
    watched_page.store(page);
    watch_address.store(address);
    watch_hits.store(0);
    return true;
}

inline void log_guard_seen(const EXCEPTION_POINTERS& info, bool ours) {
    try {
        event("guard_seen", std::string("\"ours\":") + (ours ? "true" : "false") + ",\"at\":" +
                                quote(where(reinterpret_cast<uint64_t>(info.ExceptionRecord->ExceptionAddress))));
    } catch (...) {
    }
}

inline bool readable_window(const void* p, size_t bytes) {
    MEMORY_BASIC_INFORMATION info{};
    if (VirtualQuery(p, &info, sizeof info) != sizeof info) return false;
    if (info.State != MEM_COMMIT || (info.Protect & (PAGE_NOACCESS | PAGE_GUARD))) return false;
    const uintptr_t offset = reinterpret_cast<uintptr_t>(p) - reinterpret_cast<uintptr_t>(info.BaseAddress);
    return offset < info.RegionSize && bytes <= static_cast<size_t>(info.RegionSize - offset);
}

inline void log_guard(const EXCEPTION_POINTERS& info, uint64_t page, int hit) {
    try {
        std::string f = "\"hit\":" + std::to_string(hit) + ",\"watch\":\"" + hex(watch_address.load()) +
                        "\",\"page\":\"" + hex(page) + "\",\"at\":" +
                        quote(where(reinterpret_cast<uint64_t>(info.ExceptionRecord->ExceptionAddress)));
        const uint64_t watch = watch_address.load();
        if (watch >= 0x10000 && readable_window(reinterpret_cast<const void*>(watch), 32)) {
            f += ",\"qwords\":[";
            for (int i = 0; i < 4; ++i) {
                if (i) f += ',';
                f += quote(hex(reinterpret_cast<const uint64_t*>(watch)[i]));
            }
            f += "]";
        }
        if (info.ContextRecord) {
            const CONTEXT& c = *info.ContextRecord;
            f += ",\"regs\":[" + quote(hex(c.Rcx)) + "," + quote(hex(c.Rdx)) + "," + quote(hex(c.R8)) + "," +
                 quote(hex(c.R9)) + "]";
            f += ",\"stack\":" + stack_from(c);
        }
        event("guard_page", f);
    } catch (...) {
    }
}

// Last-resort record: a guard violation that no handler consumed terminates the process with 0x80000001 and leaves no
// event behind. This writes one before the process dies.
inline LONG WINAPI top_level(EXCEPTION_POINTERS* info) {
    if (!enabled() || !info || !info->ExceptionRecord) return EXCEPTION_CONTINUE_SEARCH;
    try {
        std::string f = "\"code\":" + quote(hex(info->ExceptionRecord->ExceptionCode)) +
                        ",\"at\":" + quote(where(reinterpret_cast<uint64_t>(info->ExceptionRecord->ExceptionAddress))) +
                        ",\"watch\":\"" + hex(watch_address.load()) + "\"";
        if (info->ContextRecord) f += ",\"stack\":" + stack_from(*info->ContextRecord);
        event("unhandled", f);
    } catch (...) {
    }
    return EXCEPTION_CONTINUE_SEARCH;
}

inline void log_exception(const EXCEPTION_POINTERS& info) {
    const EXCEPTION_RECORD& r = *info.ExceptionRecord;
    try {
        if (r.ExceptionCode == kDbgPrint || r.ExceptionCode == kDbgPrintWide) {
            const ULONG_PTR fallback = t_wide_fallback::get();
            const bool wide = r.ExceptionCode == kDbgPrintWide;
            t_wide_fallback::set(wide && r.NumberParameters >= 4 ? r.ExceptionInformation[3] : 0);
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
            // The faulting registers separate "called through a bad pointer" from "loaded a bad field": for the
            // 1.0.2.1 allocator fault the interesting pair is rcx and the qword it points at.
            if (info.ContextRecord) {
                const CONTEXT& c = *info.ContextRecord;
                f += ",\"regs\":[" + quote(hex(c.Rcx)) + "," + quote(hex(c.Rdx)) + "," + quote(hex(c.R8)) + "," +
                     quote(hex(c.R9)) + "]";
                if (r.ExceptionCode == EXCEPTION_ACCESS_VIOLATION && c.Rcx >= 0x10000 &&
                    !IsBadReadPtr(reinterpret_cast<const void*>(c.Rcx), 8)) {
                    f += ",\"rcx0\":" + quote(hex(*reinterpret_cast<const uint64_t*>(c.Rcx)));
                }
            }
            // The 1.0.2.1 allocator fault reads its cached interface from the guest TLS block at +0x13a8. Log the TEB's
            // TLS vector, slot 0, and that qword so a garbage pointer there is distinguishable from a garbage table.
            if (info.ContextRecord && r.ExceptionCode == EXCEPTION_ACCESS_VIOLATION) {
                const auto* teb = NtCurrentTeb();
                uint64_t vector = 0;
                if (teb) vector = *reinterpret_cast<const uint64_t*>(reinterpret_cast<const uint8_t*>(teb) + 0x58);
                uint64_t slot0 = 0;
                if (vector >= 0x10000 && readable_window(reinterpret_cast<const void*>(vector), 8))
                    slot0 = *reinterpret_cast<const uint64_t*>(vector);
                f += ",\"tls\":[" + quote(hex(vector)) + "," + quote(hex(slot0)) + "]";
                if (slot0 >= 0x10000 && readable_window(reinterpret_cast<const void*>(slot0 + 0x13a0), 0x10)) {
                    const auto* p = reinterpret_cast<const uint64_t*>(slot0 + 0x13a0);
                    f += ",\"tls_13a8\":[" + quote(hex(p[0])) + "," + quote(hex(p[1])) + "]";
                }
            }
            if (info.ContextRecord) f += ",\"stack\":" + stack_from(*info.ContextRecord);
            event("exception", f);
        }
    } catch (...) {
        // Out of memory while formatting: drop the event rather than disturb the process.
    }
}

inline LONG CALLBACK handler(EXCEPTION_POINTERS* info) {
    if (!enabled() || t_in_handler::flag() || !info || !info->ExceptionRecord) return EXCEPTION_CONTINUE_SEARCH;
    if (info->ExceptionRecord->ExceptionCode == kGuardPage) {
        const uint64_t page = watched_page.load();
        log_guard_seen(*info, page != 0);
        if (page) {
            const int hit = watch_hits.fetch_add(1) + 1;
            const DWORD protect = watch_protect.load();
            VirtualProtect(reinterpret_cast<void*>(page), 0x1000, protect, nullptr);
            log_guard(*info, page, hit);
            if (hit < kWatchCap && info->ContextRecord) {
                VirtualProtect(reinterpret_cast<void*>(page), 0x1000, protect | PAGE_GUARD, nullptr);
                info->ContextRecord->EFlags |= 0x100;  // trap flag: one instruction, then the single-step handler
                t_single_stepping::set_flag(true);
            }
            return EXCEPTION_CONTINUE_EXECUTION;
        }
    }
    if (info->ExceptionRecord->ExceptionCode == EXCEPTION_SINGLE_STEP && t_single_stepping::flag()) {
        t_single_stepping::set_flag(false);
        if (info->ContextRecord) info->ContextRecord->EFlags &= ~0x100u;
        return EXCEPTION_CONTINUE_EXECUTION;
    }
    t_in_handler::set_flag(true);
#ifdef _MSC_VER
    // Under /EHsc, catch(...) does not catch SEH faults (e.g. a bad DBG_PRINTEXCEPTION_C pointer or corrupt unwind).
    // Keep the __try frame free of C++ destructors so in_handler is always reset and the exception still propagates.
    __try {
        log_exception(*info);
    } __except (EXCEPTION_EXECUTE_HANDLER) {
    }
#else
    struct Reset {
        ~Reset() { t_in_handler::set_flag(false); }
    } reset;
    log_exception(*info);
#endif
    t_in_handler::set_flag(false);
    return EXCEPTION_CONTINUE_SEARCH;
}
}  // namespace detail

inline void close();

// Watch one page for its next access and log the faulting instruction as a `guard_page` event. Used to name the code
// that writes a guest structure when static reasoning cannot. The watch is one-shot and the access is allowed through.
inline bool watch_page(uint64_t address) { return detail::watch_page(address); }

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
    SetUnhandledExceptionFilter(&detail::top_level);
    if (const char* tls_info = std::getenv("LEONIDA_TLS_INFO")) {
        if (*tls_info && *tls_info != '0') {
            auto* base = reinterpret_cast<uint8_t*>(GetModuleHandleW(nullptr));
            auto* dos = reinterpret_cast<PIMAGE_DOS_HEADER>(base);
            auto* nt = reinterpret_cast<PIMAGE_NT_HEADERS64>(base + dos->e_lfanew);
            const auto& dir = nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_TLS];
            if (dir.VirtualAddress) {
                auto* tls = reinterpret_cast<PIMAGE_TLS_DIRECTORY64>(base + dir.VirtualAddress);
                const DWORD index = *reinterpret_cast<const DWORD*>(tls->AddressOfIndex);
                void** vector = reinterpret_cast<void**>(__readgsqword(0x58));
                const auto* pad = reinterpret_cast<const uint8_t*>(guest_tls_pad());
                const uint64_t pad_offset = reinterpret_cast<uint64_t>(pad) - reinterpret_cast<uint64_t>(vector[0]);
                std::fprintf(stderr,
                             "[rt] host tls index=%lu block=%p vector[0]=%p vector[index]=%p template=%llu "
                             "pad_offset=%llu pad_in_block=%d\n",
                             static_cast<unsigned long>(index), vector[0], vector[0], vector[index],
                             static_cast<unsigned long long>(tls->EndAddressOfRawData - tls->StartAddressOfRawData),
                             static_cast<unsigned long long>(pad_offset),
                             static_cast<int>(reinterpret_cast<const uint8_t*>(vector[0]) + pad_offset == pad));
            }
        }
    }
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
