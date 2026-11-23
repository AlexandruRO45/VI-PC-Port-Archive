// Exception triage: say what faulted, on which thread, and who called (LEONIDA_FIRST_CHANCE, LEONIDA_BREAK,
// LEONIDA_FAULTS; docs/boot.md).
//
//   - Fault reports. Every access violation, illegal instruction, divide by zero and the like is described by one block:
//     the kind of fault (a call through a null pointer, a null object plus a field offset, a jump to non-code, a fault
//     inside a host function the game called), the thread and its name, the registers, the frames, the stack scan. The
//     first LEONIDA_FAULTS (default 8) are shown, as first-chance (the game may handle them). The top-level filter shows
//     the one nobody handled, and dumps every thread.
//   - First-chance log. LEONIDA_FIRST_CHANCE=N logs the first N other exceptions, C++ throws with the decorated name of
//     the first catchable type read from the MSVC throw info (and a readable form of it).
//   - Breakpoints. LEONIDA_BREAK=rva,module+rva,... patches an int3 at image offsets and logs the registers at each hit
//     (40 hits per breakpoint), then runs on: the original byte is put back for one step and the int3 is set again.
//     Meant for a first look at what a function receives, not for a debugger's job. A second thread passing the address
//     during that single step is not seen.
//   - Thread names: the 0x406D1388 exception is read so every report names the thread.
//
// The handler only reads and reports; it never changes the exception except to step over its own breakpoints. The
// reading of the throw info and the idea of breakpoints at image offsets with a hit cap come from xvalkyrie
// (src/runtime/common/execute.cpp at commit 53af7df77c10, GPL-3.0); this is a new implementation.
#pragma once

#include <cerrno>
#include <functional>
#include <memory>

#include "boot_diag.h"
#include "boot_watchdog.h"

namespace rt::boot_diag {

inline constexpr DWORD kCppException = 0xE06D7363;
inline constexpr int kMaxBreakHits = 40;
inline constexpr DWORD kStackOverflow = 0xC00000FD;

// ".?AVruntime_error@std@@" -> "std::runtime_error". Only plain class and struct names (the nested name parts up to the
// closing "@@", an anonymous namespace as "(anonymous namespace)") are made readable; templates and everything else come
// back unchanged.
inline std::string readable_type_name(const std::string& decorated) {
    if (decorated.size() < 6 || decorated.compare(0, 2, ".?") != 0 || (decorated[2] != 'A') ||
        (decorated[3] != 'V' && decorated[3] != 'U') || decorated.compare(decorated.size() - 2, 2, "@@") != 0)
        return decorated;
    std::vector<std::string> parts;
    std::string current;
    for (size_t i = 4; i + 2 <= decorated.size(); ++i) {
        const char c = decorated[i];
        if (c == '@') {
            if (current.empty()) {
                if (i + 1 == decorated.size() - 1) break;  // the closing "@@"
                return decorated;
            }
            parts.push_back(current);
            current.clear();
        } else if (c == '?' && current.empty() && decorated.compare(i, 3, "?A0") == 0) {  // an anonymous namespace
            const size_t at = decorated.find('@', i);
            if (at == std::string::npos) return decorated;
            parts.push_back("(anonymous namespace)");
            i = at;
        } else if (c == '?' || c == '$') {
            return decorated;
        } else {
            current += c;
        }
    }
    if (!current.empty() || parts.empty()) return decorated;
    std::string out;
    for (size_t i = parts.size(); i-- > 0;) out += parts[i] + (i ? "::" : "");
    return out;
}

// The decorated name of the first catchable type of a C++ exception record (x64: the ThrowInfo members are RVAs from
// ExceptionInformation[3]); "" if the record does not look like an MSVC throw or anything cannot be read.
inline std::string cpp_exception_type(const EXCEPTION_RECORD& r) {
    if (r.ExceptionCode != kCppException || r.NumberParameters < 4) return {};
    const uint64_t magic = r.ExceptionInformation[0];
    if (magic < 0x19930520 || magic > 0x19930522) return {};
    const uint64_t throw_info = r.ExceptionInformation[2], image = r.ExceptionInformation[3];
    if (!throw_info || !image) return {};
    uint32_t info[4];  // attributes, unwind, forward compat, catchable type array
    if (read_memory(reinterpret_cast<const void*>(throw_info), info, sizeof info) != sizeof info || !info[3]) return {};
    int32_t array_header[2];  // count, first catchable type RVA
    if (read_memory(reinterpret_cast<const void*>(image + info[3]), array_header, sizeof array_header) != sizeof array_header ||
        array_header[0] < 1)
        return {};
    uint32_t type_rva[2];  // properties, type descriptor RVA
    if (read_memory(reinterpret_cast<const void*>(image + static_cast<uint32_t>(array_header[1])), type_rva, sizeof type_rva) != sizeof type_rva ||
        !type_rva[1])
        return {};
    char name[128] = {};
    const size_t got = read_memory(reinterpret_cast<const void*>(image + type_rva[1] + 16), name, sizeof name - 1);
    std::string out;
    for (size_t i = 0; i < got && name[i]; ++i) out += (name[i] >= 0x20 && name[i] < 0x7f) ? name[i] : '?';
    return out;
}

inline bool is_fault_code(DWORD code) {
    switch (code) {
        case EXCEPTION_ACCESS_VIOLATION:
        case EXCEPTION_IN_PAGE_ERROR:
        case EXCEPTION_ILLEGAL_INSTRUCTION:
        case EXCEPTION_PRIV_INSTRUCTION:
        case EXCEPTION_INT_DIVIDE_BY_ZERO:
        case EXCEPTION_INT_OVERFLOW:
        case EXCEPTION_ARRAY_BOUNDS_EXCEEDED:
        case EXCEPTION_DATATYPE_MISALIGNMENT:
        case EXCEPTION_STACK_OVERFLOW:
        case 0xC0000409:  // STATUS_STACK_BUFFER_OVERRUN (fail-fast)
        case 0xC0000374:  // STATUS_HEAP_CORRUPTION
            return true;
        default:
            return false;
    }
}

inline const char* fault_name(DWORD code) {
    switch (code) {
        case EXCEPTION_ACCESS_VIOLATION: return "access violation";
        case EXCEPTION_IN_PAGE_ERROR: return "in-page error";
        case EXCEPTION_ILLEGAL_INSTRUCTION: return "illegal instruction";
        case EXCEPTION_PRIV_INSTRUCTION: return "privileged instruction";
        case EXCEPTION_INT_DIVIDE_BY_ZERO: return "integer divide by zero";
        case EXCEPTION_INT_OVERFLOW: return "integer overflow";
        case EXCEPTION_ARRAY_BOUNDS_EXCEEDED: return "array bounds exceeded";
        case EXCEPTION_DATATYPE_MISALIGNMENT: return "misaligned data";
        case EXCEPTION_STACK_OVERFLOW: return "stack overflow";
        case 0xC0000409: return "fail-fast (stack buffer overrun or __fastfail)";
        case 0xC0000374: return "heap corruption";
        default: return "exception";
    }
}

// The kind of fault in a sentence a person can act on, from the record and the first frames. Pure: takes what it reads.
inline std::string classify_fault(const EXCEPTION_RECORD& r, const uint64_t* frames, int frame_count) {
    const uint64_t rip = reinterpret_cast<uint64_t>(r.ExceptionAddress);
    if (r.ExceptionCode == EXCEPTION_ACCESS_VIOLATION && r.NumberParameters >= 2) {
        const uint64_t op = r.ExceptionInformation[0], address = r.ExceptionInformation[1];
        const std::string caller = frame_count > 1 ? where(frames[1]) : std::string("an unknown caller");
        if (op == 8 || (op != 0 && op != 1)) {
            if (address == rip && address < 0x10000)
                return format("call through a null pointer (target 0x%llx), called from %s", static_cast<unsigned long long>(address), caller.c_str());
            return format("jump to 0x%llx, which is not executable code, from %s", static_cast<unsigned long long>(address), caller.c_str());
        }
        const char* what = op == 0 ? "read" : "write";
        if (address < 0x10000)
            return format("%s of a null object: address 0x%llx is a null pointer plus the field offset 0x%llx", what,
                          static_cast<unsigned long long>(address), static_cast<unsigned long long>(address));
        return format("%s of unmapped or protected memory at 0x%llx", what, static_cast<unsigned long long>(address));
    }
    if (r.ExceptionCode == EXCEPTION_STACK_OVERFLOW) return "the thread ran out of stack: endless recursion or a very large frame";
    return {};
}

// The block describing an exception: used for first-chance faults, the top-level filter and the tests.
inline std::vector<std::string> describe_fault(const EXCEPTION_POINTERS& info, const char* label) {
    const EXCEPTION_RECORD& r = *info.ExceptionRecord;
    const DWORD tid = GetCurrentThreadId();
    auto s = std::make_unique<StackSample>();
    s->tid = tid;
    if (info.ContextRecord) sample_context(*info.ContextRecord, *s);
    std::vector<std::string> lines;
    lines.push_back(format("%s %s (0x%08lx) at %s on %s", label, fault_name(r.ExceptionCode), r.ExceptionCode,
                           where(reinterpret_cast<uint64_t>(r.ExceptionAddress)).c_str(), thread_label(tid).c_str()));
    if (const std::string kind = classify_fault(r, s->frames, s->frame_count); !kind.empty()) lines.push_back("  " + kind);
    if (const std::string start = thread_start_text(GetCurrentThread()); !start.empty()) lines.push_back("  thread started at " + start);
    if (s->ok) {
        const auto& c = *info.ContextRecord;
        lines.push_back(format("  rax=%llx rcx=%llx rdx=%llx r8=%llx r9=%llx rbx=%llx rsi=%llx rdi=%llx rbp=%llx rsp=%llx",
                               c.Rax, c.Rcx, c.Rdx, c.R8, c.R9, c.Rbx, c.Rsi, c.Rdi, c.Rbp, c.Rsp));
        const GuestFrame g = first_guest_frame(s->frames, s->frame_count);
        if (g.found && g.index > 0)
            lines.push_back("  the fault is in host code; image code " + where(g.address) + " called " +
                            (g.callee.empty() ? std::string("an address that is no known import") : g.callee));
        else if (!g.found)
            lines.push_back("  no frame inside a mapped image");
        lines.push_back("  frames: " + frames_text(s->frames, s->frame_count));
        const auto hits = scan_sample(*s);
        std::string scan;
        for (const auto& h : hits) {
            bool dup = false;
            for (int i = 0; i < s->frame_count; ++i) dup = dup || s->frames[i] == h.address;
            if (!dup) scan += (scan.empty() ? "" : ", ") + format("[rsp+0x%llx] ", static_cast<unsigned long long>(h.slot_offset)) + where(h.address);
        }
        if (!scan.empty()) lines.push_back("  stack scan, not in the frames above: " + scan);
    }
    return lines;
}

// ---- breakpoints ----------------------------------------------------------------------------------------------------

namespace detail {
struct Breakpoint {
    uint8_t* address = nullptr;
    uint8_t original = 0;
    std::string label;
    std::atomic<int> hits{0};
    std::atomic<bool> disabled{false};
};
inline std::mutex& break_mutex() {
    static auto* m = new std::mutex;
    return *m;
}
inline std::vector<std::unique_ptr<Breakpoint>>& breakpoints() {
    static auto* v = new std::vector<std::unique_ptr<Breakpoint>>;
    return *v;
}
inline thread_local Breakpoint* rearm = nullptr;

inline bool write_code_byte(uint8_t* address, uint8_t value) {
    DWORD old = 0;
    if (!VirtualProtect(address, 1, PAGE_EXECUTE_READWRITE, &old)) return false;
    *address = value;
    VirtualProtect(address, 1, old, &old);
    FlushInstructionCache(GetCurrentProcess(), address, 1);
    return true;
}

inline Breakpoint* find_break(const void* address) {
    const std::lock_guard lock(break_mutex());
    for (auto& b : breakpoints())
        if (b->address == address) return b.get();
    return nullptr;
}
}  // namespace detail

// Sets an int3 at 'address' (inside a mapped image) and names it 'label' in the hit lines.
inline bool add_breakpoint(uint8_t* address, const std::string& label) {
    if (detail::find_break(address)) return true;
    auto b = std::make_unique<detail::Breakpoint>();
    b->address = address;
    b->label = label;
    if (read_memory(address, &b->original, 1) != 1 || b->original == 0xCC || !detail::write_code_byte(address, 0xCC)) return false;
    const std::lock_guard lock(detail::break_mutex());
    detail::breakpoints().push_back(std::move(b));
    return true;
}

inline int breakpoint_hits(const void* address) {
    detail::Breakpoint* b = detail::find_break(address);
    return b ? b->hits.load() : -1;
}

// Resolves "module" to a mapped image base and size; an empty name means the executable. False if unknown.
using ImageResolver = std::function<bool(const std::string& module, uint8_t*& base, uint32_t& size)>;

// "1c4a0,client.dll+3f20": adds each; returns what could not be set, as readable lines.
inline std::vector<std::string> set_breakpoints_from_text(const std::string& text, const ImageResolver& resolve) {
    std::vector<std::string> problems;
    size_t pos = 0;
    while (pos <= text.size()) {
        size_t end = text.find(',', pos);
        if (end == std::string::npos) end = text.size();
        std::string item = text.substr(pos, end - pos);
        pos = end + 1;
        while (!item.empty() && item.front() == ' ') item.erase(0, 1);
        while (!item.empty() && item.back() == ' ') item.pop_back();
        if (item.empty()) continue;
        std::string module, rva_text = item;
        if (const size_t plus = item.rfind('+'); plus != std::string::npos) {
            module = item.substr(0, plus);
            rva_text = item.substr(plus + 1);
        }
        char* stop = nullptr;
        const unsigned long long rva = std::strtoull(rva_text.c_str(), &stop, 16);
        uint8_t* base = nullptr;
        uint32_t size = 0;
        if (rva_text.empty() || *stop != 0)
            problems.push_back("breakpoint '" + item + "' not set: expected [module+]hex RVA");
        else if (!resolve(module, base, size))
            problems.push_back("breakpoint '" + item + "' not set: no mapped image " + (module.empty() ? "for the executable" : "named " + module));
        else if (rva >= size)
            problems.push_back("breakpoint '" + item + "' not set: RVA is outside the image (size " + format("0x%x", size) + ")");
        else if (!add_breakpoint(base + rva, item))
            problems.push_back("breakpoint '" + item + "' not set: the byte cannot be patched or is already an int3");
    }
    return problems;
}

// ---- the handler ----------------------------------------------------------------------------------------------------

struct FaultConfig {
    int fault_reports = 8;       // first-chance faults shown (LEONIDA_FAULTS); 0 turns them off
    int first_chance_limit = 0;  // other exceptions logged (LEONIDA_FIRST_CHANCE); 0: none
};

namespace detail {
inline FaultConfig fault_config;
inline std::atomic<int> faults_shown{0}, first_chance_shown{0};
inline std::atomic<bool> faults_silenced_noted{false};
inline void* veh = nullptr;
inline LPTOP_LEVEL_EXCEPTION_FILTER previous_filter = nullptr;
inline thread_local bool in_handler = false;

inline bool is_debug_print(DWORD code) { return code == 0x40010006 || code == 0x4001000A; }

inline void log_breakpoint_hit(Breakpoint& b, const CONTEXT& c) {
    const int n = ++b.hits;
    if (n > kMaxBreakHits) return;
    std::vector<std::string> lines;
    uint64_t ret = 0;
    const std::string caller = c.Rsp && read_u64(c.Rsp, ret) ? where(ret) : std::string("unknown");
    lines.push_back(format("breakpoint %s hit %d of %d on %s, return address %s", b.label.c_str(), n, kMaxBreakHits,
                           thread_label(GetCurrentThreadId()).c_str(), caller.c_str()));
    lines.push_back(format("  rax=%llx rcx=%llx rdx=%llx r8=%llx r9=%llx rbx=%llx rsi=%llx rdi=%llx rsp=%llx", c.Rax, c.Rcx,
                           c.Rdx, c.R8, c.R9, c.Rbx, c.Rsi, c.Rdi, c.Rsp));
    if (n == kMaxBreakHits) lines.push_back("  (the limit: later hits are counted, not logged)");
    report("break_hit", lines);
}

inline void log_first_chance(const EXCEPTION_POINTERS& info) {
    const EXCEPTION_RECORD& r = *info.ExceptionRecord;
    std::string text = format("first-chance exception 0x%08lx at %s on %s", r.ExceptionCode,
                              where(reinterpret_cast<uint64_t>(r.ExceptionAddress)).c_str(), thread_label(GetCurrentThreadId()).c_str());
    if (r.ExceptionCode == kCppException) {
        const std::string type = cpp_exception_type(r);
        text += type.empty() ? ", a C++ throw (type not readable)" : ", C++ throw of " + readable_type_name(type) + " (" + type + ")";
    }
    std::vector<std::string> lines{text};
    if (info.ContextRecord) {
        uint64_t frames[8];
        const int n = unwind_frames(*info.ContextRecord, frames, 8);
        lines.push_back("  frames: " + frames_text(frames, n, 8));
    }
    report("first_chance", lines);
}

inline LONG handle(EXCEPTION_POINTERS* info) {
    const EXCEPTION_RECORD& r = *info->ExceptionRecord;
    if (r.ExceptionCode == kThreadNameException) {
        note_thread_name_exception(r);
        return EXCEPTION_CONTINUE_SEARCH;
    }
    if (is_debug_print(r.ExceptionCode)) return EXCEPTION_CONTINUE_SEARCH;
    if (r.ExceptionCode == EXCEPTION_BREAKPOINT) {
        if (Breakpoint* b = find_break(r.ExceptionAddress)) {
            // Put the byte back, run the one original instruction (trap flag), then set the int3 again.
            if (b->disabled.load() || !info->ContextRecord) return EXCEPTION_CONTINUE_SEARCH;
            if (!write_code_byte(b->address, b->original)) {
                if (!b->disabled.exchange(true))
                    report("break_error", {"breakpoint " + b->label + " disabled: could not restore the original byte"});
                return EXCEPTION_CONTINUE_SEARCH;
            }
            info->ContextRecord->Rip = reinterpret_cast<uint64_t>(b->address);
            info->ContextRecord->EFlags |= 0x100;
            rearm = b;
            log_breakpoint_hit(*b, *info->ContextRecord);
            return EXCEPTION_CONTINUE_EXECUTION;
        }
        return EXCEPTION_CONTINUE_SEARCH;
    }
    if (r.ExceptionCode == EXCEPTION_SINGLE_STEP && rearm) {
        Breakpoint* b = rearm;
        rearm = nullptr;
        info->ContextRecord->EFlags &= ~0x100u;
        if (!b->disabled.load()) write_code_byte(b->address, 0xCC);
        return EXCEPTION_CONTINUE_EXECUTION;
    }
    if (r.ExceptionCode == kStackOverflow) {
        // Almost no stack is left: one fixed line, nothing that allocates.
        std::fprintf(stderr, "[diag] stack overflow at 0x%llx on thread %lu\n", static_cast<unsigned long long>(reinterpret_cast<uint64_t>(r.ExceptionAddress)), GetCurrentThreadId());
        return EXCEPTION_CONTINUE_SEARCH;
    }
    if (is_fault_code(r.ExceptionCode)) {
        if (fault_config.fault_reports <= 0) return EXCEPTION_CONTINUE_SEARCH;
        const int n = ++faults_shown;
        if (n <= fault_config.fault_reports) {
            report("fault", describe_fault(*info, "first-chance fault:"));
        } else if (!faults_silenced_noted.exchange(true)) {
            report("fault", {format("more than %d first-chance faults: not shown (LEONIDA_FAULTS=N shows more)", fault_config.fault_reports)});
        }
        return EXCEPTION_CONTINUE_SEARCH;
    }
    if (fault_config.first_chance_limit > 0 && ++first_chance_shown <= fault_config.first_chance_limit) log_first_chance(*info);
    return EXCEPTION_CONTINUE_SEARCH;
}

inline LONG CALLBACK veh_entry(EXCEPTION_POINTERS* info) {
    if (in_handler || !info || !info->ExceptionRecord) return EXCEPTION_CONTINUE_SEARCH;
    const DWORD last_error = GetLastError();
    const int saved_errno = errno;
    in_handler = true;
    LONG result = EXCEPTION_CONTINUE_SEARCH;
#ifdef _MSC_VER
    __try {
        result = handle(info);
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        result = EXCEPTION_CONTINUE_SEARCH;
    }
#else
    result = handle(info);
#endif
    in_handler = false;
    errno = saved_errno;
    SetLastError(last_error);
    return result;
}

inline LONG WINAPI top_level_filter(EXCEPTION_POINTERS* info) {
    if (info && info->ExceptionRecord && !in_handler && info->ExceptionRecord->ExceptionCode != kStackOverflow) {
        in_handler = true;
        report("fault", describe_fault(*info, "UNHANDLED"));
        dump_all_threads("an exception nobody handled; the process ends after this", nullptr);
        in_handler = false;
    }
    return EXCEPTION_CONTINUE_SEARCH;  // the process ends as it would without this filter
}
}  // namespace detail

inline void install_fault_handlers(const FaultConfig& config) {
    if (detail::veh) return;
    detail::fault_config = config;
    detail::faults_shown = 0;
    detail::first_chance_shown = 0;
    detail::faults_silenced_noted = false;
    detail::veh = AddVectoredExceptionHandler(1, &detail::veh_entry);
    detail::previous_filter = SetUnhandledExceptionFilter(&detail::top_level_filter);
}

inline void remove_fault_handlers() {
    if (!detail::veh) return;
    RemoveVectoredExceptionHandler(detail::veh);
    detail::veh = nullptr;
    SetUnhandledExceptionFilter(detail::previous_filter);
    detail::previous_filter = nullptr;
}

}  // namespace rt::boot_diag
