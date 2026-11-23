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
// Zeroes the line under SEH: another thread may unmap or reprotect it after writable() looked.
bool zero_line(uint8_t* line) {
    __try {
        std::memset(line, 0, kCacheLine);
        return true;
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        return false;
    }
}

size_t emulate_clzero(CONTEXT& ctx, const uint8_t* code, size_t available) {
    if (available < 3 || code[0] != 0x0F || code[1] != 0x01 || code[2] != 0xFC) return 0;
    auto* line = reinterpret_cast<uint8_t*>(ctx.Rax & ~static_cast<DWORD64>(kCacheLine - 1));
    // A line whose end wraps past the top of the address space (RAX near 2^64) cannot be a mapped line.
    if (reinterpret_cast<uintptr_t>(line) > UINTPTR_MAX - kCacheLine) return 0;
    if (!writable(line, kCacheLine) || !zero_line(line)) return 0;
    return 3;
}

const Emulator kTable[] = {
    {"clzero", &emulate_clzero},
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

bool install(bool force) {
    if (g_handler) return true;
    if (!force && host_has_clzero()) return false;
    g_handler = AddVectoredExceptionHandler(1, &handler);
    return g_handler != nullptr;
}

void uninstall() {
    if (g_handler) RemoveVectoredExceptionHandler(g_handler);
    g_handler = nullptr;
}

}  // namespace rt::cpu_ud_emulate
