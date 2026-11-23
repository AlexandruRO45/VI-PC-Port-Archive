// Tests for the #UD emulation (src/cpu_ud_emulate.*): CLZERO, 0F 01 FC, with its prefixes and its faults.
//
// The emulation is driven directly with a synthetic CONTEXT and EXCEPTION_RECORD, so it is tested on any host, one
// that has the instruction included. A real run through the vectored handler needs a CPU that lacks the instruction;
// on one that has it, the handler is checked with an instruction every CPU rejects (UD2) and the emulation is
// compared with what the CPU itself does.
#include <windows.h>

#include <cstdio>
#include <cstring>
#include <initializer_list>

#include "../src/cpu_ud_emulate.h"

namespace {

namespace ud = rt::cpu_ud_emulate;

int g_checks = 0;
int g_failed = 0;

void check(bool ok, const char* what) {
    ++g_checks;
    if (ok) {
        std::printf("ok    %s\n", what);
    } else {
        ++g_failed;
        std::printf("FAIL  %s\n", what);
    }
}

constexpr size_t kPage = 4096;

// A 64-byte-aligned buffer of three lines of 0xAB, so the middle line can be zeroed and its neighbours checked.
struct alignas(64) Lines {
    uint8_t b[192];
    Lines() { std::memset(b, 0xAB, sizeof b); }
    bool zero(size_t from, size_t to) const {
        for (size_t i = from; i < to; ++i)
            if (b[i] != 0) return false;
        return true;
    }
    bool all_ab(size_t from, size_t to) const {
        for (size_t i = from; i < to; ++i)
            if (b[i] != 0xAB) return false;
        return true;
    }
};

struct Fault {
    EXCEPTION_RECORD record{};
    CONTEXT context{};
    EXCEPTION_POINTERS pointers{};
    Fault(const uint8_t* code, DWORD64 rax, DWORD code_number = STATUS_ILLEGAL_INSTRUCTION) {
        record.ExceptionCode = code_number;
        record.ExceptionAddress = const_cast<uint8_t*>(code);
        context.Rip = reinterpret_cast<DWORD64>(code);
        context.Rax = rax;
        pointers.ExceptionRecord = &record;
        pointers.ContextRecord = &context;
    }
};

const uint8_t kClzero[] = {0x0F, 0x01, 0xFC, 0x90};

void test_zeroes_one_line() {
    Lines mem;
    Fault f(kClzero, reinterpret_cast<DWORD64>(mem.b + 64));
    const LONG r = ud::emulate(&f.pointers);
    check(r == EXCEPTION_CONTINUE_EXECUTION, "CLZERO: the handler resumes execution");
    check(mem.zero(64, 128), "CLZERO: the 64 bytes at RAX are zero");
    check(mem.all_ab(0, 64) && mem.all_ab(128, 192), "CLZERO: the bytes on either side are untouched");
    check(f.context.Rip == reinterpret_cast<DWORD64>(kClzero) + 3, "CLZERO: RIP moves past the 3-byte instruction");
    check(f.context.Rax == reinterpret_cast<DWORD64>(mem.b + 64), "CLZERO: RAX is unchanged");
}

void test_unaligned_address() {
    Lines mem;
    // Any address inside the line selects the line: the low 6 bits of RAX are ignored.
    Fault f(kClzero, reinterpret_cast<DWORD64>(mem.b + 64 + 37));
    check(ud::emulate(&f.pointers) == EXCEPTION_CONTINUE_EXECUTION, "CLZERO at an unaligned address resumes");
    check(mem.zero(64, 128) && mem.all_ab(0, 64) && mem.all_ab(128, 192),
          "CLZERO at an unaligned address zeroes the line that contains it, and only that line");
    Lines last;
    Fault g(kClzero, reinterpret_cast<DWORD64>(last.b + 63));
    ud::emulate(&g.pointers);
    check(last.zero(0, 64) && last.all_ab(64, 192), "CLZERO at the last byte of a line zeroes that line");
}

void test_other_instructions_pass() {
    const uint8_t clzero_neighbour[] = {0x0F, 0x01, 0xFD, 0x90};  // MONITORX; no emulator yet
    const uint8_t nop[] = {0x90, 0x90, 0x90, 0x90};
    const uint8_t truncated[] = {0x0F, 0x01};  // too short to be CLZERO
    struct Case {
        const uint8_t* code;
        const char* what;
    } cases[] = {
        {clzero_neighbour, "an opcode next to CLZERO (0F 01 FD)"},
        {nop, "a NOP"},
    };
    for (const Case& c : cases) {
        Lines mem;
        Fault f(c.code, reinterpret_cast<DWORD64>(mem.b + 64));
        const CONTEXT before = f.context;
        const LONG r = ud::emulate(&f.pointers);
        char label[160];
        std::snprintf(label, sizeof label, "%s: passed on, nothing modified", c.what);
        check(r == EXCEPTION_CONTINUE_SEARCH && mem.all_ab(0, 192) && std::memcmp(&before, &f.context, sizeof before) == 0,
              label);
    }
    {
        // The two bytes of CLZERO's prefix at the very end of a readable page: the third byte is unreadable.
        auto* page = static_cast<uint8_t*>(VirtualAlloc(nullptr, 2 * kPage, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
        DWORD old;
        VirtualProtect(page + kPage, kPage, PAGE_NOACCESS, &old);
        uint8_t* at = page + kPage - sizeof truncated;
        std::memcpy(at, truncated, sizeof truncated);
        Lines mem;
        Fault f(at, reinterpret_cast<DWORD64>(mem.b + 64));
        check(ud::emulate(&f.pointers) == EXCEPTION_CONTINUE_SEARCH && mem.all_ab(0, 192),
              "an instruction cut off by unreadable memory: passed on, nothing modified");
        VirtualFree(page, 0, MEM_RELEASE);
    }
    Lines mem;
    Fault access_violation(kClzero, reinterpret_cast<DWORD64>(mem.b + 64), STATUS_ACCESS_VIOLATION);
    check(ud::emulate(&access_violation.pointers) == EXCEPTION_CONTINUE_SEARCH && mem.all_ab(0, 192),
          "CLZERO bytes under another exception code (access violation): passed on");
    check(ud::emulate(nullptr) == EXCEPTION_CONTINUE_SEARCH, "no exception information: passed on");
}

// True when `f` now carries the page fault a CPU raises for a CLZERO write at `address`, and the CONTEXT is as it was.
bool is_write_fault(const Fault& f, const CONTEXT& before, uintptr_t address) {
    return f.record.ExceptionCode == STATUS_ACCESS_VIOLATION && f.record.NumberParameters == 2 &&
           f.record.ExceptionInformation[0] == 1 && f.record.ExceptionInformation[1] == address &&
           std::memcmp(&before, &f.context, sizeof before) == 0;
}

// The record is untouched: still the #UD it came in as.
bool still_ud(const Fault& f) {
    return f.record.ExceptionCode == STATUS_ILLEGAL_INSTRUCTION && f.record.NumberParameters == 0;
}

void test_unwritable_target() {
    // The handler must not fault inside itself. A target it cannot write is a page fault on the real CPU, so the
    // record is rewritten into that access violation (write, address of the line) for the handlers after it.
    auto* page = static_cast<uint8_t*>(VirtualAlloc(nullptr, kPage, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
    std::memset(page, 0xAB, kPage);
    DWORD old;
    VirtualProtect(page, kPage, PAGE_READONLY, &old);
    Fault ro(kClzero, reinterpret_cast<DWORD64>(page + 64 + 9));
    CONTEXT before = ro.context;
    check(ud::emulate(&ro.pointers) == EXCEPTION_CONTINUE_SEARCH && is_write_fault(ro, before, uintptr_t(page + 64)) &&
              page[64] == 0xAB,
          "CLZERO on a read-only page: access violation (write, line address), context and memory untouched");
    VirtualProtect(page, kPage, PAGE_NOACCESS, &old);
    Fault na(kClzero, reinterpret_cast<DWORD64>(page));
    before = na.context;
    check(ud::emulate(&na.pointers) == EXCEPTION_CONTINUE_SEARCH && is_write_fault(na, before, uintptr_t(page)),
          "CLZERO on a no-access page: access violation (write, line address)");
    VirtualProtect(page, kPage, PAGE_READWRITE | PAGE_GUARD, &old);
    Fault guard(kClzero, reinterpret_cast<DWORD64>(page));
    before = guard.context;
    check(ud::emulate(&guard.pointers) == EXCEPTION_CONTINUE_SEARCH && is_write_fault(guard, before, uintptr_t(page)),
          "CLZERO on a guard page: access violation (write, line address)");
    VirtualFree(page, 0, MEM_RELEASE);
    Fault unmapped(kClzero, 0x10);
    before = unmapped.context;
    check(ud::emulate(&unmapped.pointers) == EXCEPTION_CONTINUE_SEARCH && is_write_fault(unmapped, before, 0),
          "CLZERO at an unmapped address: access violation at the line");
    Fault top(kClzero, ~static_cast<DWORD64>(0));
    before = top.context;
    check(ud::emulate(&top.pointers) == EXCEPTION_CONTINUE_SEARCH &&
              is_write_fault(top, before, ~static_cast<uintptr_t>(0) & ~uintptr_t(63)),
          "CLZERO whose line wraps past the top of the address space: access violation");
    Fault noncanonical(kClzero, 0x1234567800000040ull);
    before = noncanonical.context;
    check(ud::emulate(&noncanonical.pointers) == EXCEPTION_CONTINUE_SEARCH &&
              is_write_fault(noncanonical, before, 0x1234567800000040ull),
          "CLZERO at a non-canonical address: access violation");
}

// Runs the instruction `bytes` (followed by a NOP) with RAX at the middle line of `mem` and reports whether it was
// emulated as an instruction `length` bytes long that zeroed exactly that line.
bool emulated_as(const uint8_t* bytes, size_t length, DWORD64 rax_high = 0) {
    uint8_t code[16] = {};
    std::memcpy(code, bytes, length);
    code[length] = 0x90;
    Lines mem;
    Fault f(code, reinterpret_cast<DWORD64>(mem.b + 64) | rax_high);
    const DWORD64 rax = f.context.Rax;
    return ud::emulate(&f.pointers) == EXCEPTION_CONTINUE_EXECUTION && mem.zero(64, 128) && mem.all_ab(0, 64) &&
           mem.all_ab(128, 192) && f.context.Rip == reinterpret_cast<DWORD64>(code) + length && f.context.Rax == rax;
}

// Runs `bytes` and reports whether it was left alone: passed on as the original #UD, nothing modified.
bool declined(const uint8_t* bytes, size_t length) {
    uint8_t code[16] = {};
    std::memcpy(code, bytes, length);
    Lines mem;
    Fault f(code, reinterpret_cast<DWORD64>(mem.b + 64));
    const CONTEXT before = f.context;
    return ud::emulate(&f.pointers) == EXCEPTION_CONTINUE_SEARCH && mem.all_ab(0, 192) &&
           std::memcmp(&before, &f.context, sizeof before) == 0 && still_ud(f);
}

void test_prefixes() {
    // Legacy prefixes CLZERO ignores, each alone: 66, F2, F3 and the six segment overrides.
    const uint8_t legacy[] = {0x66, 0xF2, 0xF3, 0x2E, 0x36, 0x3E, 0x26, 0x64, 0x65};
    for (uint8_t p : legacy) {
        const uint8_t bytes[] = {p, 0x0F, 0x01, 0xFC};
        char label[80];
        std::snprintf(label, sizeof label, "CLZERO with prefix %02X: emulated, 4 bytes long", p);
        check(emulated_as(bytes, sizeof bytes), label);
    }
    for (uint8_t rex : {0x40, 0x48, 0x4F}) {
        const uint8_t bytes[] = {rex, 0x0F, 0x01, 0xFC};
        char label[80];
        std::snprintf(label, sizeof label, "CLZERO with REX %02X: emulated, 4 bytes long", rex);
        check(emulated_as(bytes, sizeof bytes), label);
    }
    const uint8_t stacked[] = {0x66, 0xF3, 0x2E, 0x48, 0x0F, 0x01, 0xFC};
    check(emulated_as(stacked, sizeof stacked), "CLZERO with 66 F3 2E and REX.W: emulated, 7 bytes long");
    // AMD hardware counts every REX byte toward the instruction length.
    const uint8_t rex_then_legacy[] = {0x48, 0x66, 0x0F, 0x01, 0xFC};
    check(emulated_as(rex_then_legacy, sizeof rex_then_legacy), "REX before a legacy prefix: emulated, 5 bytes long");
    const uint8_t two_rex[] = {0x48, 0x48, 0x0F, 0x01, 0xFC};
    check(emulated_as(two_rex, sizeof two_rex), "two REX bytes: emulated, 5 bytes long");
    const uint8_t three_rex[] = {0x40, 0x48, 0x4F, 0x0F, 0x01, 0xFC};
    check(emulated_as(three_rex, sizeof three_rex), "three REX bytes: emulated, 6 bytes long");
    // LOCK on CLZERO is a real #UD, alone or among other prefixes.
    const uint8_t lock[] = {0xF0, 0x0F, 0x01, 0xFC};
    check(declined(lock, sizeof lock), "LOCK CLZERO: not emulated, nothing modified");
    const uint8_t lock_among[] = {0x66, 0xF0, 0x48, 0x0F, 0x01, 0xFC};
    check(declined(lock_among, sizeof lock_among), "66 LOCK REX CLZERO: not emulated, nothing modified");
    // More prefix bytes than fit in a 15-byte instruction.
    uint8_t too_long[16];
    std::memset(too_long, 0x66, sizeof too_long);
    std::memcpy(too_long + 13, "\x0F\x01\xFC", 3);
    check(declined(too_long, sizeof too_long), "a 16-byte CLZERO (13 prefixes): not emulated, nothing modified");
    uint8_t longest[15];
    std::memset(longest, 0x66, 12);
    std::memcpy(longest + 12, "\x0F\x01\xFC", 3);
    check(emulated_as(longest, sizeof longest), "CLZERO with 12 prefix bytes (15 in all): emulated, 15 bytes long");
}

void test_address_size_prefix() {
    // Without 67 the whole of RAX is the address: garbage in the upper half makes it non-canonical, a fault.
    const uint8_t plain[] = {0x0F, 0x01, 0xFC, 0x90};
    {
        Lines mem;
        Fault f(plain, reinterpret_cast<DWORD64>(mem.b + 64) | 0x1234567800000000ull);
        const CONTEXT before = f.context;
        check(ud::emulate(&f.pointers) == EXCEPTION_CONTINUE_SEARCH && mem.all_ab(0, 192) &&
                  std::memcmp(&before, &f.context, sizeof before) == 0 &&
                  f.record.ExceptionCode == STATUS_ACCESS_VIOLATION,
              "without 67 the upper half of RAX counts: a non-canonical address faults");
    }
    // With 67 the address is EAX, zero-extended. Needs memory below 4 GiB: ask for it at a few fixed places.
    uint8_t* low = nullptr;
    for (uintptr_t hint : {0x20000000u, 0x30000000u, 0x50000000u, 0x70000000u}) {
        low = static_cast<uint8_t*>(VirtualAlloc(reinterpret_cast<void*>(hint), kPage, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
        if (low) break;
    }
    if (!low) {
        std::printf("SKIP  CLZERO with 67: no memory below 4 GiB to aim at\n");
        return;
    }
    std::memset(low, 0xAB, kPage);
    const uint8_t with67[] = {0x67, 0x0F, 0x01, 0xFC, 0x90};
    const uint8_t with67_rex[] = {0x67, 0x48, 0x0F, 0x01, 0xFC, 0x90};
    for (const uint8_t* code : {with67, with67_rex}) {
        std::memset(low, 0xAB, kPage);
        Fault f(code, 0x1234567800000000ull | (reinterpret_cast<uintptr_t>(low) + 64 + 11));
        const DWORD64 rax = f.context.Rax;
        const size_t length = code == with67 ? 4 : 5;
        check(ud::emulate(&f.pointers) == EXCEPTION_CONTINUE_EXECUTION && f.context.Rax == rax &&
                  f.context.Rip == reinterpret_cast<DWORD64>(code) + length,
              "CLZERO with 67: emulated, RAX unchanged, RIP past the instruction");
        bool only_line = true;
        for (size_t i = 0; i < kPage; ++i)
            only_line &= low[i] == ((i >= 64 && i < 128) ? 0 : 0xAB);
        check(only_line, "CLZERO with 67: the line at EAX is zeroed, the upper half of RAX ignored");
    }
    VirtualFree(low, 0, MEM_RELEASE);
}

// --- through the installed vectored handler ---

// Runs after ours (it is added last): counts the #UD exceptions that reach it and skips the 2-byte UD2.
volatile LONG g_reached = 0;
LONG CALLBACK probe(EXCEPTION_POINTERS* info) {
    if (info->ExceptionRecord->ExceptionCode != STATUS_ILLEGAL_INSTRUCTION) return EXCEPTION_CONTINUE_SEARCH;
    InterlockedIncrement(&g_reached);
    info->ContextRecord->Rip += 2;
    return EXCEPTION_CONTINUE_EXECUTION;
}

using VoidFn = void (*)();
using LineFn = void (*)(void*);

LineFn make_code(const uint8_t* bytes, size_t n) {
    void* mem = VirtualAlloc(nullptr, kPage, MEM_RESERVE | MEM_COMMIT, PAGE_EXECUTE_READWRITE);
    std::memcpy(mem, bytes, n);
    return reinterpret_cast<LineFn>(mem);
}

void test_installed_handler() {
    check(ud::install(true), "install(force) puts the handler in place");
    PVOID later = AddVectoredExceptionHandler(0, probe);

    // An instruction every CPU rejects reaches the handlers behind ours.
    const uint8_t ud2_ret[] = {0x0F, 0x0B, 0xC3};
    LineFn ud2 = make_code(ud2_ret, sizeof ud2_ret);
    ud2(nullptr);
    check(g_reached == 1, "UD2 goes through the installed handler to the next one");

    // mov rax, rcx; clzero; ret
    const uint8_t clzero_ret[] = {0x48, 0x89, 0xC8, 0x0F, 0x01, 0xFC, 0xC3};
    LineFn clzero = make_code(clzero_ret, sizeof clzero_ret);
    Lines mem;
    if (ud::host_has_clzero()) {
        // The CPU runs it itself: the handler must not be involved, and the effect is the one the emulation has.
        clzero(mem.b + 64 + 5);
        check(g_reached == 1 && mem.zero(64, 128) && mem.all_ab(0, 64) && mem.all_ab(128, 192),
              "CLZERO runs natively here and zeroes the same line the emulation does");
        std::printf("SKIP  CLZERO through the vectored handler: this CPU has CLZERO\n");
    } else {
        clzero(mem.b + 64 + 5);
        check(g_reached == 1 && mem.zero(64, 128) && mem.all_ab(0, 64) && mem.all_ab(128, 192),
              "CLZERO on a CPU without it: emulated by the installed handler, the next handler never sees it");
    }
    RemoveVectoredExceptionHandler(later);
    ud::uninstall();
}

}  // namespace

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    test_zeroes_one_line();
    test_unaligned_address();
    test_other_instructions_pass();
    test_unwritable_target();
    test_prefixes();
    test_address_size_prefix();
    test_installed_handler();
    std::printf("%d checks, %d failed\n", g_checks, g_failed);
    return g_failed ? 1 : 0;
}
