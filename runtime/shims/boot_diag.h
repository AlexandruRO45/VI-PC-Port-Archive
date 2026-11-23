// Boot diagnostics: say which import, which thread and which caller when the first boot fails or hangs.
//
// What this answers, in the order a person asks it:
//   - which thread: its id, the name the game gave it (the old 0x406D1388 exception, or SetThreadDescription) and the
//     address it started at, as module+RVA;
//   - which import: for a stack frame that returns into a mapped image, the call instruction before the return
//     address is decoded (call rel32, call [rip+disp], through a jmp [rip+disp] thunk) and the address it reached is
//     named: a shim from the registry ("kernel32.dll!Foo") or a host export ("kernelbase.dll!WaitForSingleObject");
//   - who called it: the first frame inside a mapped image, as module+RVA.
// Nothing here changes what the process does. Output goes to stderr as "[diag] ..." lines (one block per report, never
// interleaved) and, when --trace is on, to the startup trace as an event with the same text.
//
// Everything is measured from the live process: no symbol files, no game data. Frames come from the unwind data
// the loader registers for each image (RtlAddFunctionTable), so they are exact where a function has an entry; a stack
// scan of the raw stack (every slot holding an address inside a mapped image that follows a call instruction) lists
// what the unwind could not reach. See docs/boot.md, "Diagnosing a hang or a crash".
//
// The idea of a periodic all-thread stack dump, of reading thread names from the old exception, and of a stack scan
// comes from xvalkyrie (src/runtime/modules/watchdog.cpp, src/runtime/common/execute.cpp, commit 53af7df77c10,
// GPL-3.0). The code is written for this runtime's loader and tests.
#pragma once

#include <windows.h>
#include <tlhelp32.h>

#include <algorithm>
#include <atomic>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "shims.h"
#include "startup_trace.h"

namespace rt::boot_diag {

inline constexpr int kMaxFrames = 24;
inline constexpr size_t kRawStackBytes = 16 * 1024;
inline constexpr size_t kMaxScanHits = 12;
inline constexpr DWORD kThreadNameException = 0x406D1388;  // MS_VC_EXCEPTION, the old way to name a thread
inline constexpr double kRunningCpuPercent = 30.0;         // at least this much of the sample window: running
inline constexpr double kWaitingCpuPercent = 5.0;          // below this: waiting

inline std::string format(const char* fmt, ...) {
    char stack_buf[512];
    va_list args;
    va_start(args, fmt);
    va_list copy;
    va_copy(copy, args);
    const int n = std::vsnprintf(stack_buf, sizeof stack_buf, fmt, args);
    va_end(args);
    std::string out;
    if (n < 0) {
        out = "?";
    } else if (static_cast<size_t>(n) < sizeof stack_buf) {
        out.assign(stack_buf, static_cast<size_t>(n));
    } else {
        out.resize(static_cast<size_t>(n) + 1);
        std::vsnprintf(out.data(), out.size(), fmt, copy);
        out.resize(static_cast<size_t>(n));
    }
    va_end(copy);
    return out;
}

// ---- output ---------------------------------------------------------------------------------------------------------

// A test replaces the sink to read what a report says; the default writes "[diag] line" to stderr.
using Sink = void (*)(void* ctx, const std::string& line);

namespace detail {
inline std::atomic<Sink> sink{nullptr};
inline std::atomic<void*> sink_ctx{nullptr};
inline std::mutex& out_mutex() {  // immortal: reports can come from late exception paths
    static auto* m = new std::mutex;
    return *m;
}
}  // namespace detail

inline void set_sink(Sink sink, void* ctx) {
    detail::sink_ctx.store(ctx);
    detail::sink.store(sink);
}

// One report: its lines stay together, and the startup trace gets one event (kind) with the whole text.
inline void report(const char* kind, const std::vector<std::string>& lines) {
    std::string whole;
    {
        const std::lock_guard lock(detail::out_mutex());
        for (const auto& l : lines) {
            if (Sink s = detail::sink.load()) {
                s(detail::sink_ctx.load(), l);
            } else {
                std::fprintf(stderr, "[diag] %s\n", l.c_str());
            }
            whole += l;
            whole += '\n';
        }
        if (!detail::sink.load()) std::fflush(stderr);
    }
    if (startup_trace::enabled()) startup_trace::event(kind, "\"text\":" + startup_trace::quote(whole));
}

// module+RVA; an address in the first 64 KiB (a null pointer, a null pointer plus a field offset) is shown bare.
inline std::string where(uint64_t address) {
    return address < 0x10000 ? format("0x%llx", static_cast<unsigned long long>(address)) : shims::module_offset(address);
}

inline bool in_game_image(uint64_t address) {
    uint64_t base = 0;
    return address && shims::mapped_image_base(address, base);
}

// ---- memory and thread names ----------------------------------------------------------------------------------------

// Reads what is readable of [p, p+n) without faulting; page by page, so a range that ends in an unmapped page still
// returns its first part. Returns the byte count.
inline size_t read_memory(const void* p, void* out, size_t n) {
    size_t total = 0;
    auto* dst = static_cast<uint8_t*>(out);
    auto src = reinterpret_cast<uintptr_t>(p);
    while (total < n) {
        const size_t to_page = 4096 - (src & 4095);
        const size_t want = std::min(n - total, to_page);
        SIZE_T got = 0;
        if (!ReadProcessMemory(GetCurrentProcess(), reinterpret_cast<const void*>(src), dst + total, want, &got) ||
            got != want)
            return total + got;
        total += want;
        src += want;
    }
    return total;
}

inline bool read_u64(uint64_t address, uint64_t& value) { return read_memory(reinterpret_cast<void*>(address), &value, 8) == 8; }

namespace detail {
inline std::mutex& names_mutex() {
    static auto* m = new std::mutex;
    return *m;
}
inline std::unordered_map<DWORD, std::string>& names() {
    static auto* m = new std::unordered_map<DWORD, std::string>;
    return *m;
}
}  // namespace detail

inline void set_thread_name(DWORD tid, const std::string& name) {
    const std::lock_guard lock(detail::names_mutex());
    detail::names()[tid] = name;
}

// The name the game gave the thread: the exception's, else SetThreadDescription's, else "".
inline std::string thread_name(DWORD tid) {
    {
        const std::lock_guard lock(detail::names_mutex());
        const auto it = detail::names().find(tid);
        if (it != detail::names().end()) return it->second;
    }
    using GetDesc = HRESULT(WINAPI*)(HANDLE, PWSTR*);
    static const auto get_description =
        reinterpret_cast<GetDesc>(reinterpret_cast<void*>(GetProcAddress(GetModuleHandleW(L"kernelbase.dll"), "GetThreadDescription")));
    if (!get_description) return {};
    HANDLE h = OpenThread(THREAD_QUERY_LIMITED_INFORMATION, FALSE, tid);
    if (!h) return {};
    std::string out;
    PWSTR w = nullptr;
    if (SUCCEEDED(get_description(h, &w)) && w) {
        out = startup_trace::detail::utf8_from_wide(w, static_cast<int>(wcslen(w)));
        LocalFree(w);
    }
    CloseHandle(h);
    return out;
}

// The exception record of the old thread-name exception: [0] 0x1000 in the low dword, [1] the ANSI name, [2] the thread
// id in the low dword (-1: the calling thread) and flags in the high dword. Returns true if it was one.
inline bool note_thread_name_exception(const EXCEPTION_RECORD& r) {
    if (r.ExceptionCode != kThreadNameException || r.NumberParameters < 3) return false;
    if (static_cast<DWORD>(r.ExceptionInformation[0]) != 0x1000) return false;
    char name[65] = {};
    const size_t got = read_memory(reinterpret_cast<const void*>(r.ExceptionInformation[1]), name, sizeof name - 1);
    std::string clean;
    for (size_t i = 0; i < got && name[i]; ++i) clean += (name[i] >= 0x20 && name[i] < 0x7f) ? name[i] : '?';
    DWORD tid = static_cast<DWORD>(r.ExceptionInformation[2]);
    if (tid == static_cast<DWORD>(-1)) tid = GetCurrentThreadId();
    if (!clean.empty()) set_thread_name(tid, clean);
    return true;
}

// ---- who a call reached ---------------------------------------------------------------------------------------------

// The file name (lower case) and export name of a host function, by its exact address; "" if it is not an export.
// When several exports share the address the first by name is given.
inline std::string host_export_name(uint64_t address) {
    HMODULE mod = nullptr;
    if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                            reinterpret_cast<LPCWSTR>(address), &mod) || !mod)
        return {};
    const auto* base = reinterpret_cast<const uint8_t*>(mod);
    const auto* dos = reinterpret_cast<const IMAGE_DOS_HEADER*>(base);
    if (dos->e_magic != IMAGE_DOS_SIGNATURE) return {};
    const auto* nt = reinterpret_cast<const IMAGE_NT_HEADERS64*>(base + dos->e_lfanew);
    if (nt->Signature != IMAGE_NT_SIGNATURE) return {};
    const auto& dir = nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_EXPORT];
    if (!dir.VirtualAddress) return {};
    const auto* exp = reinterpret_cast<const IMAGE_EXPORT_DIRECTORY*>(base + dir.VirtualAddress);
    const auto* funcs = reinterpret_cast<const DWORD*>(base + exp->AddressOfFunctions);
    const auto* names = reinterpret_cast<const DWORD*>(base + exp->AddressOfNames);
    const auto* ords = reinterpret_cast<const WORD*>(base + exp->AddressOfNameOrdinals);
    const uint64_t rva = address - reinterpret_cast<uint64_t>(base);
    std::string best;
    for (DWORD i = 0; i < exp->NumberOfNames; ++i) {
        if (ords[i] >= exp->NumberOfFunctions || funcs[ords[i]] != rva) continue;
        const char* n = reinterpret_cast<const char*>(base + names[i]);
        if (best.empty() || std::strcmp(n, best.c_str()) < 0) best = n;
    }
    if (best.empty()) return {};
    char file[MAX_PATH] = {};
    const DWORD len = GetModuleFileNameA(mod, file, MAX_PATH);
    std::string f(file, len < MAX_PATH ? len : 0);
    const size_t slash = f.find_last_of("\\/");
    if (slash != std::string::npos) f.erase(0, slash + 1);
    for (char& c : f) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return f + "!" + best;
}

// "dll!name" for an address a call can reach: a shim of the registry first, then a host export. Follows a jmp [rip+disp]
// thunk (the form the linker puts between a call site and an import).
inline std::string import_name(uint64_t target) {
    for (int hop = 0; hop < 3 && target; ++hop) {
        if (std::string s = shims::shim_import_name(reinterpret_cast<const void*>(target)); !s.empty()) return s;
        if (std::string s = host_export_name(target); !s.empty()) return s;
        uint8_t code[6] = {};
        if (read_memory(reinterpret_cast<const void*>(target), code, sizeof code) != sizeof code) return {};
        if (code[0] != 0xFF || code[1] != 0x25) return {};  // not jmp [rip+disp32]
        int32_t disp;
        std::memcpy(&disp, code + 2, 4);
        if (!read_u64(target + 6 + static_cast<int64_t>(disp), target)) return {};
    }
    return {};
}

// What the call instruction that ends at return address 'ret' reached: "dll!name", or "" when the instruction is not a
// call of a form decoded here (call rel32, call [rip+disp32]), or the target is not a known import.
inline std::string callee_of_call_ending_at(uint64_t ret) {
    uint8_t before[6] = {};
    if (read_memory(reinterpret_cast<const void*>(ret - 6), before, 6) != 6) return {};
    if (before[0] == 0xFF && before[1] == 0x15) {  // call [rip+disp32]
        int32_t disp;
        std::memcpy(&disp, before + 2, 4);
        uint64_t target = 0;
        if (!read_u64(ret + static_cast<int64_t>(disp), target)) return {};
        return import_name(target);
    }
    if (before[1] == 0xE8) {  // call rel32
        int32_t rel;
        std::memcpy(&rel, before + 2, 4);
        return import_name(ret + static_cast<int64_t>(rel));
    }
    return {};
}

// Whether the bytes before 'ret' end a call instruction of a common form: rel32, [rip+disp32], register, [reg],
// [reg+disp8], [reg+disp32], with or without a REX prefix and SIB byte. A stack slot that passes is very probably a
// return address; one that does not is data.
inline bool follows_call(uint64_t ret) {
    uint8_t b[8] = {};
    if (read_memory(reinterpret_cast<const void*>(ret - 8), b, 8) != 8) return false;
    if (b[3] == 0xE8) return true;                                   // call rel32
    if (b[2] == 0xFF && b[3] == 0x15) return true;                   // call [rip+disp32]
    if (b[6] == 0xFF && (b[7] & 0xF8) == 0xD0) return true;          // call reg
    if (b[5] == 0xFF && (b[6] & 0xF8) == 0xD0 && (b[4] & 0xF0) == 0x40) return true;  // call reg with REX
    if (b[6] == 0xFF && (b[7] & 0xF8) == 0x10 && (b[7] & 7) != 4 && (b[7] & 7) != 5) return true;  // call [reg]
    if (b[5] == 0xFF && (b[6] & 0xF8) == 0x50 && (b[6] & 7) != 4) return true;                     // call [reg+d8]
    if (b[4] == 0xFF && b[5] == 0x54) return true;                                                  // call [reg+idx+d8]
    if (b[2] == 0xFF && (b[3] & 0xF8) == 0x90 && (b[3] & 7) != 4) return true;                     // call [reg+d32]
    if (b[5] == 0xFF && b[6] == 0x14) return true;                                                  // call [sib]
    return false;
}

// ---- stacks ---------------------------------------------------------------------------------------------------------

// A thread's registers and stack as copied while it was stopped. Copying is all that happens while a thread is suspended:
// no allocation, no lock, so a thread stopped inside the heap cannot block the watchdog.
struct StackSample {
    DWORD tid = 0;
    bool ok = false;
    uint64_t rip = 0, rsp = 0;
    int frame_count = 0;
    uint64_t frames[kMaxFrames] = {};
    size_t raw_len = 0;
    uint8_t raw[kRawStackBytes] = {};
};

// Walks the unwind data from 'context'; the first frame is the instruction pointer itself, the rest are return
// addresses. A frame without an entry is treated as a leaf (return address at [rsp]). POD only: it runs while a thread
// is suspended, and __try cannot sit in a function that has destructors to run.
inline int unwind_frames(CONTEXT context, uint64_t* out, int max) {
    int n = 0;
#ifdef _MSC_VER
    __try {
#endif
        while (n < max && (context.Rip || n == 0)) {  // a call through a null pointer has rip 0: its caller is at [rsp]
            out[n++] = context.Rip;
            DWORD64 image_base = 0;
            PRUNTIME_FUNCTION fn = RtlLookupFunctionEntry(context.Rip, &image_base, nullptr);
            const DWORD64 before = context.Rsp;
            if (!fn) {
                uint64_t ret = 0;
                if (!context.Rsp || !read_u64(context.Rsp, ret)) break;
                context.Rip = ret;
                context.Rsp += 8;
            } else {
                void* handler_data = nullptr;
                DWORD64 frame = 0;
                RtlVirtualUnwind(UNW_FLAG_NHANDLER, image_base, context.Rip, fn, &context, &handler_data, &frame, nullptr);
            }
            if (context.Rsp <= before) break;  // no progress: stop rather than loop
        }
#ifdef _MSC_VER
    } __except (EXCEPTION_EXECUTE_HANDLER) {
    }
#endif
    return n;
}

// Fills 's' from a context: frames, plus a copy of the stack above rsp for the scan.
inline void sample_context(const CONTEXT& context, StackSample& s) {
    s.rip = context.Rip;
    s.rsp = context.Rsp;
    s.frame_count = unwind_frames(context, s.frames, kMaxFrames);
    s.raw_len = context.Rsp ? read_memory(reinterpret_cast<const void*>(context.Rsp), s.raw, kRawStackBytes) : 0;
    s.ok = true;
}

struct ScanHit {
    uint64_t slot_offset;  // from rsp
    uint64_t address;
};

// Every 8-byte slot of the raw stack that holds an address inside a mapped image right after a call instruction.
// 'in_image' and 'after_call' are parameters so the scan itself is testable on a buffer.
template <typename InImage, typename AfterCall>
inline std::vector<ScanHit> scan_stack(const uint8_t* raw, size_t len, InImage in_image, AfterCall after_call,
                                       size_t max_hits = kMaxScanHits) {
    std::vector<ScanHit> hits;
    for (size_t off = 0; off + 8 <= len && hits.size() < max_hits; off += 8) {
        uint64_t v;
        std::memcpy(&v, raw + off, 8);
        if (v > 0x10000 && in_image(v) && after_call(v)) hits.push_back({off, v});
    }
    return hits;
}

inline std::vector<ScanHit> scan_sample(const StackSample& s) {
    return scan_stack(s.raw, s.raw_len, [](uint64_t v) { return in_game_image(v); },
                      [](uint64_t v) { return follows_call(v); });
}

// ---- reading a stack as text ----------------------------------------------------------------------------------------

// The frames as one line: "a < b < c" (innermost first).
inline std::string frames_text(const uint64_t* frames, int count, int limit = 12) {
    std::string out;
    for (int i = 0; i < count && i < limit; ++i) {
        if (i) out += " < ";
        out += where(frames[i]);
    }
    if (count > limit) out += " < ...";
    return out;
}

// The first frame inside a mapped image: where the game was, and (if it is a return address with a decodable call)
// the import it called. index 0 is the instruction pointer itself, which is not a return address.
struct GuestFrame {
    bool found = false;
    int index = -1;
    uint64_t address = 0;
    std::string callee;  // "dll!name" the call reached, "" if not decoded
};

inline GuestFrame first_guest_frame(const uint64_t* frames, int count) {
    GuestFrame g;
    for (int i = 0; i < count; ++i) {
        if (!in_game_image(frames[i])) continue;
        g.found = true;
        g.index = i;
        g.address = frames[i];
        if (i > 0) g.callee = callee_of_call_ending_at(frames[i]);
        return g;
    }
    return g;
}

// The thread's start address as the OS reports it, as module+RVA; "" if unknown.
inline std::string thread_start_text(HANDLE thread) {
    using QueryThread = LONG(NTAPI*)(HANDLE, ULONG, PVOID, ULONG, PULONG);
    static const auto query =
        reinterpret_cast<QueryThread>(reinterpret_cast<void*>(GetProcAddress(GetModuleHandleW(L"ntdll.dll"), "NtQueryInformationThread")));
    uint64_t start = 0;
    if (!query || query(thread, 9 /* ThreadQuerySetWin32StartAddress */, &start, sizeof start, nullptr) < 0 || !start) return {};
    return where(start);
}

// "thread 4412 "RenderThread"" (the quotes only when it has a name).
inline std::string thread_label(DWORD tid) {
    const std::string name = thread_name(tid);
    return name.empty() ? format("thread %lu", tid) : format("thread %lu \"%s\"", tid, name.c_str());
}

// The lines that say who the stack belongs to and what it was doing; shared by the thread dump and the fault report.
inline void describe_stack(const StackSample& s, std::vector<std::string>& lines) {
    const GuestFrame g = first_guest_frame(s.frames, s.frame_count);
    if (!g.found) {
        lines.push_back("  no frame inside a mapped image (the thread is in host code only)");
    } else if (g.index == 0) {
        lines.push_back("  running image code at " + where(g.address));
    } else {
        std::string line = "  image code " + where(g.address) + " called ";
        line += g.callee.empty() ? "an address that is no known import (indirect call)" : g.callee;
        lines.push_back(line);
    }
    lines.push_back("  frames: " + frames_text(s.frames, s.frame_count));
    const auto hits = scan_sample(s);
    std::string scan;
    for (const auto& h : hits) {
        bool dup = false;
        for (int i = 0; i < s.frame_count; ++i) dup = dup || s.frames[i] == h.address;
        if (dup) continue;
        scan += (scan.empty() ? "" : ", ") + format("[rsp+0x%llx] ", static_cast<unsigned long long>(h.slot_offset)) +
                where(h.address);
    }
    if (!scan.empty()) lines.push_back("  stack scan, not in the frames above: " + scan);
}

}  // namespace rt::boot_diag
