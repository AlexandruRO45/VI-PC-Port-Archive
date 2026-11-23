// Tests for the #UD emulation (src/cpu_ud_emulate.*): CLZERO, 0F 01 FC.
//
// The emulation is driven directly with a synthetic CONTEXT and EXCEPTION_RECORD, so it is tested on any host, one
// that has the instruction included. A real run through the vectored handler needs a CPU that lacks the instruction;
// on one that has it, the handler is checked with an instruction every CPU rejects (UD2) and the emulation is
// compared with what the CPU itself does.
#include <windows.h>

#include <cstdio>
#include <cstring>

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

void test_unwritable_target() {
    // The handler must not fault inside itself: a target it cannot write is left for the next handler, which then
    // sees the original #UD, as the real instruction's page fault would reach it.
    auto* page = static_cast<uint8_t*>(VirtualAlloc(nullptr, kPage, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
    std::memset(page, 0xAB, kPage);
    DWORD old;
    VirtualProtect(page, kPage, PAGE_READONLY, &old);
    Fault ro(kClzero, reinterpret_cast<DWORD64>(page));
    const CONTEXT before = ro.context;
    check(ud::emulate(&ro.pointers) == EXCEPTION_CONTINUE_SEARCH && std::memcmp(&before, &ro.context, sizeof before) == 0 &&
              page[0] == 0xAB,
          "CLZERO on a read-only page: passed on, nothing modified");
    VirtualProtect(page, kPage, PAGE_NOACCESS, &old);
    Fault na(kClzero, reinterpret_cast<DWORD64>(page));
    check(ud::emulate(&na.pointers) == EXCEPTION_CONTINUE_SEARCH, "CLZERO on a no-access page: passed on");
    Fault unmapped(kClzero, 0x10);
    check(ud::emulate(&unmapped.pointers) == EXCEPTION_CONTINUE_SEARCH, "CLZERO at an unmapped address: passed on");
    Fault top(kClzero, ~static_cast<DWORD64>(0));
    check(ud::emulate(&top.pointers) == EXCEPTION_CONTINUE_SEARCH, "CLZERO at the top of the address space: passed on");
    VirtualFree(page, 0, MEM_RELEASE);
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
    test_installed_handler();
    std::printf("%d checks, %d failed\n", g_checks, g_failed);
    return g_failed ? 1 : 0;
}
