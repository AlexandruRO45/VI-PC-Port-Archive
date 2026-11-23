#include "cpu_ud_emulate.h"

#include <intrin.h>

#include <cstring>

namespace rt::cpu_ud_emulate {
namespace {

constexpr size_t kCacheLine = 64;

// Whether the whole of [p, p + n) is committed memory this process may write: every page of it writable, none a
// guard page. (A guard page would fault, and a write the emulator does itself must not consume the guard.)
bool writable(const void* p, size_t n) {
    const auto begin = reinterpret_cast<uintptr_t>(p);
    const uintptr_t end = begin + n;
    for (uintptr_t at = begin; at < end;) {
        MEMORY_BASIC_INFORMATION mbi;
        if (VirtualQuery(reinterpret_cast<const void*>(at), &mbi, sizeof mbi) != sizeof mbi) return false;
        if (mbi.State != MEM_COMMIT || (mbi.Protect & (PAGE_GUARD | PAGE_NOACCESS))) return false;
        if (!(mbi.Protect & (PAGE_READWRITE | PAGE_WRITECOPY | PAGE_EXECUTE_READWRITE | PAGE_EXECUTE_WRITECOPY)))
            return false;
        at = reinterpret_cast<uintptr_t>(mbi.BaseAddress) + mbi.RegionSize;
    }
    return true;
}

// The line the last write_fault() on this thread reported.
thread_local uintptr_t t_fault_address = 0;

// Zeroes the line under SEH: another thread may unmap or reprotect it after writable() looked.
bool zero_line(uint8_t* line) {
    __try {
        std::memset(line, 0, kCacheLine);
        return true;
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        return false;
    }
}

// Whether `a` is a canonical x64 address (bits 63:47 all equal; Windows does not use 5-level paging).
bool canonical(uint64_t a) {
    const uint64_t top = a >> 47;
    return top == 0 || top == 0x1FFFF;
}

// CLZERO ignores REX, operand size (66), REP/REPNE (F3/F2) and the segment overrides. LOCK (F0) causes #UD.
bool ignored_prefix(uint8_t b) {
    if ((b & 0xF0) == 0x40) return true;  // REX
    switch (b) {
        case 0x66: case 0xF2: case 0xF3:
        case 0x2E: case 0x36: case 0x3E: case 0x26: case 0x64: case 0x65:
            return true;
        default:
            return false;
    }
}

// CLZERO, 0F 01 FC, zeroes the 64-byte cache line that contains the address in RAX (the low 6 bits are ignored).
// All REX bytes count toward the length, including before legacy prefixes. The address-size prefix 67 selects EAX
// (zero-extended) instead. If the line cannot be written (not writable, not committed, a non-canonical address, or one
// whose line wraps past the top of memory) the real instruction raises a page fault: the emulator reports it with
// write_fault(), and the dispatcher turns it into the access violation the game would see on real hardware. A guard
// page is reported the same way, without consuming the guard.
size_t emulate_clzero(CONTEXT& ctx, const uint8_t* code, size_t available) {
    size_t at = 0;
    bool addr32 = false;
    // A valid instruction is at most 15 bytes: up to 12 prefix bytes (REX included) and the 3 opcode bytes.
    while (at < available && (ignored_prefix(code[at]) || code[at] == 0x67)) {
        addr32 |= code[at] == 0x67;
        if (++at > kMaxInstructionLength - 3) return 0;
    }
    if (available < at + 3 || code[at] != 0x0F || code[at + 1] != 0x01 || code[at + 2] != 0xFC) return 0;
    const uint64_t address = addr32 ? (ctx.Rax & 0xFFFFFFFFull) : ctx.Rax;
    const uint64_t line_address = address & ~static_cast<uint64_t>(kCacheLine - 1);
    auto* line = reinterpret_cast<uint8_t*>(static_cast<uintptr_t>(line_address));
    // A line whose end wraps past the top of the address space cannot be a mapped line.
    if (!canonical(line_address) || line_address > UINT64_MAX - kCacheLine) return write_fault(line_address);
    if (!writable(line, kCacheLine) || !zero_line(line)) return write_fault(line_address);
    return at + 3;
}

// INSERTQ (SSE4a), F2 [REX] 0F 79 /r (xmm1, xmm2: length and index come from the high quadword of xmm2) and
// F2 [REX] 0F 78 /r ib ib (xmm1, xmm2, length, index). Both take ModRM.reg as xmm1 (destination, also a source) and
// ModRM.rm as xmm2, register forms only. The low `length` bits of the low quadword of xmm2 replace the bits of the low
// quadword of xmm1 from bit `index` up; the high quadword of xmm1 is zeroed. Length and index are 6-bit fields
// (bits 5:0 of the immediate, or bits 69:64 and 77:72 of xmm2; the bits around them are ignored); a length of 0
// means 64. For fields extending past bit 63, the unsigned shifts discard the excess bits, as measured on Zen 2.
size_t emulate_insertq(CONTEXT& ctx, const uint8_t* code, size_t available) {
    size_t at = 0;
    if (available < 1 || code[at++] != 0xF2) return 0;
    uint8_t rex = 0;
    if (at < available && (code[at] & 0xF0) == 0x40) rex = code[at++];
    if (available < at + 3 || code[at] != 0x0F) return 0;
    const uint8_t opcode = code[at + 1];
    const uint8_t modrm = code[at + 2];
    at += 3;
    if ((opcode != 0x78 && opcode != 0x79) || (modrm >> 6) != 3) return 0;
    const unsigned dst_index = ((modrm >> 3) & 7) | ((rex & 4) << 1);
    const unsigned src_index = (modrm & 7) | ((rex & 1) << 3);
    M128A& dst = ctx.FltSave.XmmRegisters[dst_index];
    const M128A& src = ctx.FltSave.XmmRegisters[src_index];

    unsigned length, index;
    if (opcode == 0x79) {
        length = static_cast<unsigned>(src.High) & 0x3F;
        index = static_cast<unsigned>(src.High >> 8) & 0x3F;
    } else {
        if (available < at + 2) return 0;
        length = code[at] & 0x3F;
        index = code[at + 1] & 0x3F;
        at += 2;
    }
    if (length == 0) length = 64;
    const ULONGLONG mask = length == 64 ? ~0ULL : (1ULL << length) - 1;
    dst.Low = (dst.Low & ~(mask << index)) | ((src.Low & mask) << index);
    dst.High = 0;
    return at;
}

// The length of the ModRM byte at code[at], with its SIB byte and displacement, or 0 when `available` cuts it off.
// mod 00 with rm 101 is RIP-relative (disp32); rm 100 adds a SIB byte, and mod 00 with SIB base 101 has a disp32 and
// no base. REX bits do not change the length.
size_t modrm_length(const uint8_t* code, size_t available, size_t at) {
    if (at >= available) return 0;
    const unsigned mod = code[at] >> 6, rm = code[at] & 7;
    size_t length = 1;
    unsigned disp = mod == 1 ? 1 : mod == 2 ? 4 : 0;
    if (rm == 4) {
        if (at + 1 >= available) return 0;
        ++length;
        if (mod == 0 && (code[at + 1] & 7) == 5) disp = 4;
    } else if (mod == 0 && rm == 5) {
        disp = 4;
    }
    length += disp;
    return at + length <= available ? length : 0;
}

// CLWB, 66 [REX] 0F AE /6 with a memory operand: write the cache line back without evicting it. That changes no
// architectural state, so the emulation is a no-op that skips the instruction; the work is its length (prefix, ModRM,
// SIB, displacement). 0F AE /6 without the 66 is XSAVEOPT and 66 0F AE /7 is CLFLUSHOPT: neither is taken.
size_t emulate_clwb(CONTEXT&, const uint8_t* code, size_t available) {
    size_t at = 0;
    if (available < 4 || code[at++] != 0x66) return 0;
    if ((code[at] & 0xF0) == 0x40) ++at;  // REX
    if (at + 3 > available || code[at] != 0x0F || code[at + 1] != 0xAE) return 0;
    at += 2;
    const unsigned modrm = code[at];
    if ((modrm >> 6) == 3 || ((modrm >> 3) & 7) != 6) return 0;
    const size_t tail = modrm_length(code, available, at);
    return tail ? at + tail : 0;
}

// MONITORX (0F 01 FA) and MWAITX (0F 01 FB), AMD's user-mode MONITOR/MWAIT. MONITORX only arms an address monitor and
// has no visible effect, so it does nothing. MWAITX waits until the monitored line is written, an interrupt arrives
// or its timer runs out; it is allowed to return early, and callers re-check their condition in a loop, so it yields
// the thread and returns at once. Neither touches a register or memory.
size_t emulate_monitorx_mwaitx(CONTEXT&, const uint8_t* code, size_t available) {
    if (available < 3 || code[0] != 0x0F || code[1] != 0x01 || (code[2] != 0xFA && code[2] != 0xFB)) return 0;
    if (code[2] == 0xFB) SwitchToThread();
    return 3;
}

const Emulator kTable[] = {
    {"clzero", &emulate_clzero},
    {"insertq", &emulate_insertq},
    {"clwb", &emulate_clwb},
    {"monitorx_mwaitx", &emulate_monitorx_mwaitx},
};

// Copies up to kMaxInstructionLength bytes of the code at `ip`, fewer when it runs into unreadable memory, and
// returns how many.
size_t fetch_code(const void* ip, uint8_t (&out)[kMaxInstructionLength]) {
    for (size_t n = kMaxInstructionLength; n > 0; --n) {
        SIZE_T got = 0;
        if (ReadProcessMemory(GetCurrentProcess(), ip, out, n, &got) && got == n) return n;
    }
    return 0;
}

PVOID g_handler = nullptr;

LONG CALLBACK handler(EXCEPTION_POINTERS* info) { return emulate(info); }

}  // namespace

size_t write_fault(uintptr_t address) {
    t_fault_address = address;
    return kFault;
}

LONG emulate(EXCEPTION_POINTERS* info) {
    if (!info || !info->ExceptionRecord || !info->ContextRecord) return EXCEPTION_CONTINUE_SEARCH;
    if (info->ExceptionRecord->ExceptionCode != STATUS_ILLEGAL_INSTRUCTION) return EXCEPTION_CONTINUE_SEARCH;
    CONTEXT& ctx = *info->ContextRecord;
    uint8_t code[kMaxInstructionLength];
    const size_t available = fetch_code(reinterpret_cast<const void*>(ctx.Rip), code);
    if (!available) return EXCEPTION_CONTINUE_SEARCH;
    for (const Emulator& e : kTable) {
        // Each emulator works on a copy, so a refusal after a partial change cannot leak into the saved context.
        CONTEXT work = ctx;
        const size_t length = e.fn(work, code, available);
        if (!length) continue;
        if (length == kFault) {
            // The CPU would have raised a page fault, not a #UD: say so to the handlers after this one.
            EXCEPTION_RECORD& record = *info->ExceptionRecord;
            record.ExceptionCode = STATUS_ACCESS_VIOLATION;
            record.NumberParameters = 2;
            record.ExceptionInformation[0] = 1;  // a write
            record.ExceptionInformation[1] = t_fault_address;
            return EXCEPTION_CONTINUE_SEARCH;
        }
        ctx = work;
        ctx.Rip += length;
        return EXCEPTION_CONTINUE_EXECUTION;
    }
    return EXCEPTION_CONTINUE_SEARCH;
}

bool host_has_clzero() {
    int regs[4];
    __cpuid(regs, static_cast<int>(0x80000000));
    if (static_cast<unsigned>(regs[0]) < 0x80000008u) return false;
    __cpuid(regs, static_cast<int>(0x80000008));
    return (regs[1] & 1) != 0;
}

bool host_has_sse4a() {
    int regs[4];
    __cpuid(regs, static_cast<int>(0x80000000));
    if (static_cast<unsigned>(regs[0]) < 0x80000001u) return false;
    __cpuid(regs, static_cast<int>(0x80000001));
    return (regs[2] & (1 << 6)) != 0;
}

bool host_has_clwb() {
    int regs[4];
    __cpuid(regs, 0);
    if (regs[0] < 7) return false;
    __cpuidex(regs, 7, 0);
    return (regs[1] & (1 << 24)) != 0;
}

bool host_has_monitorx() {
    int regs[4];
    __cpuid(regs, static_cast<int>(0x80000000));
    if (static_cast<unsigned>(regs[0]) < 0x80000001u) return false;
    __cpuid(regs, static_cast<int>(0x80000001));
    return (regs[2] & (1 << 29)) != 0;
}

bool install(bool force) {
    if (g_handler) return true;
    if (!force && host_has_clzero() && host_has_monitorx() && host_has_clwb() && host_has_sse4a()) return false;
    g_handler = AddVectoredExceptionHandler(1, &handler);
    return g_handler != nullptr;
}

void uninstall() {
    if (g_handler) RemoveVectoredExceptionHandler(g_handler);
    g_handler = nullptr;
}

}  // namespace rt::cpu_ud_emulate
