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

// CLZERO, 0F 01 FC: zero the 64-byte cache line that contains the address in RAX (the low 6 bits are ignored). If
// that line is not writable the real instruction raises a page fault; here the emulator declines, so the access
// violation the game would see on real hardware is not turned into a silent success.
size_t emulate_clzero(CONTEXT& ctx, const uint8_t* code, size_t available) {
    if (available < 3 || code[0] != 0x0F || code[1] != 0x01 || code[2] != 0xFC) return 0;
    auto* line = reinterpret_cast<uint8_t*>(ctx.Rax & ~static_cast<DWORD64>(kCacheLine - 1));
    if (!writable(line, kCacheLine)) return 0;
    std::memset(line, 0, kCacheLine);
    return 3;
}

// INSERTQ (SSE4a), F2 [REX] 0F 79 /r (xmm1, xmm2: length and index come from the high quadword of xmm2) and
// F2 [REX] 0F 78 /r ib ib (xmm1, xmm2, length, index). Both take ModRM.reg as xmm1 (destination, also a source) and
// ModRM.rm as xmm2, register forms only. The low `length` bits of the low quadword of xmm2 replace the bits of the low
// quadword of xmm1 from bit `index` up; the high quadword of xmm1 is unchanged. Length and index are 6-bit fields
// (bits 5:0 of the immediate, or bits 69:64 and 77:72 of xmm2; the bits around them are ignored); a length of 0
// means 64. A field that reaches past bit 63 (index + length > 64, which includes length 64 at a non-zero index) is
// undefined in the APM (vol. 4), so the emulator declines it rather than invent a result.
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
    if (index + length > 64) return 0;
    const ULONGLONG mask = length == 64 ? ~0ULL : (1ULL << length) - 1;
    dst.Low = (dst.Low & ~(mask << index)) | ((src.Low & mask) << index);
    return at;
}

const Emulator kTable[] = {
    {"clzero", &emulate_clzero},
    {"insertq", &emulate_insertq},
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

bool install(bool force) {
    if (g_handler) return true;
    if (!force && host_has_clzero() && host_has_sse4a()) return false;
    g_handler = AddVectoredExceptionHandler(1, &handler);
    return g_handler != nullptr;
}

void uninstall() {
    if (g_handler) RemoveVectoredExceptionHandler(g_handler);
    g_handler = nullptr;
}

}  // namespace rt::cpu_ud_emulate
