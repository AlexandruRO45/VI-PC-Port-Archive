// Tests for the CLWB emulation in src/cpu_ud_emulate.*: 66 [REX] 0F AE /6 with a memory operand.
//
// CLWB changes nothing a program can see, so what matters is the instruction's LENGTH: the handler resumes after it, and
// a wrong length lands in the middle of the next instruction. Every expected length below is written by hand from the
// Intel SDM encoding rules (prefix 66, optional REX, 0F AE, ModRM, optional SIB, disp8/disp32; mod 00 with rm 101 is
// RIP-relative disp32; mod 00 with SIB base 101 has a disp32), not computed by the code under test.
//
// The emulation is driven directly with a synthetic CONTEXT and EXCEPTION_RECORD, so it runs on any host, one that
// has CLWB included.
#include <windows.h>

#include <cstdio>
#include <cstring>
#include <initializer_list>
#include <vector>

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

// Runs the dispatcher on `bytes` (followed by NOP padding, so a too-long length would show) and returns how far RIP
// moved, or -1 when the exception was passed on. Registers and flags must come back unchanged either way.
long run(const std::vector<uint8_t>& bytes, bool* state_kept = nullptr, int length_case = -1) {
    std::vector<uint8_t> code = bytes;
    code.insert(code.end(), 16, 0x90);
    EXCEPTION_RECORD record{};
    CONTEXT context{};
    EXCEPTION_POINTERS pointers{};
    record.ExceptionCode = STATUS_ILLEGAL_INSTRUCTION;
    record.ExceptionAddress = code.data();
    context.Rip = reinterpret_cast<DWORD64>(code.data());
    context.Rax = 0x1111;
    context.Rcx = 0x2222;
    context.Rsp = 0x3333;
    context.EFlags = 0x246;
    void* page = VirtualAlloc(nullptr, kPage, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
    const DWORD64 target = reinterpret_cast<DWORD64>(page) + 128;
    if (length_case >= 0) {
        context.Rax = target; context.Rcx = target; context.Rsp = target;
        context.Rbp = target; context.R12 = target; context.R13 = target; context.Rdi = target;
        if (length_case == 2) context.Rax -= 8;
        if (length_case == 3) context.Rax -= 0x12345678;
        if (length_case == 6) context.Rsp -= 0x20;
        if (length_case == 8) context.Rdx = (target - 0x1000) / 4;
        if (length_case == 11) context.R12 -= 0x10;
        if (length_case == 12) context.R8 = target - 0x1000;
    }
    const CONTEXT before = context;
    pointers.ExceptionRecord = &record;
    pointers.ContextRecord = &context;
    const LONG r = ud::emulate(&pointers);
    if (state_kept)
        *state_kept = context.Rax == before.Rax && context.Rcx == before.Rcx && context.Rsp == before.Rsp && context.EFlags == before.EFlags;
    VirtualFree(page, 0, MEM_RELEASE);
    if (r == EXCEPTION_CONTINUE_SEARCH) return context.Rip == reinterpret_cast<DWORD64>(code.data()) ? -1 : -2;
    return static_cast<long>(context.Rip - reinterpret_cast<DWORD64>(code.data()));
}

struct Encoding {
    const char* what;
    std::vector<uint8_t> bytes;
    long length;  // by hand
};

void test_fault_addresses() {
    struct Case { const char* name; std::vector<uint8_t> bytes; DWORD64 rax, rcx, rdx, r8, r9, r12, r13; uintptr_t expected; };
    const Case cases[] = {
        {"base", {0x66,0x0f,0xae,0x31}, 0,0x10,0,0,0,0,0,0x10},
        {"ignored segment override", {0x2e,0x66,0x0f,0xae,0x31}, 0,0x10,0,0,0,0,0,0x10},
        {"disp8 negative", {0x66,0x0f,0xae,0x71,0xf0}, 0,0x40,0,0,0,0,0,0x30},
        {"disp32", {0x66,0x0f,0xae,0xb0,0x34,0x12,0,0}, 0x20,0,0,0,0,0,0,0x1254},
        {"SIB scale", {0x66,0x0f,0xae,0x74,0x90,0x08}, 0x20,0,3,0,0,0,0,0x34},
        {"SIB no base", {0x66,0x0f,0xae,0x34,0x95,0x00,0x10,0,0}, 0,0,3,0,0,0,0,0x100c},
        {"REX B and X", {0x66,0x43,0x0f,0xae,0x74,0xc8,0xf0}, 0,0,0,0x40,3,0,0,0x48},
        {"r12 base", {0x66,0x41,0x0f,0xae,0x34,0x24}, 0,0,0,0,0,0x50,0,0x50},
        {"r13 base", {0x66,0x41,0x0f,0xae,0x75,0}, 0,0,0,0,0,0,0x60,0x60},
    };
    for (const auto& c : cases) {
        CONTEXT ctx{}; EXCEPTION_RECORD rec{};
        ctx.Rip = reinterpret_cast<DWORD64>(c.bytes.data());
        ctx.Rax=c.rax; ctx.Rcx=c.rcx; ctx.Rdx=c.rdx; ctx.R8=c.r8; ctx.R9=c.r9; ctx.R12=c.r12; ctx.R13=c.r13;
        rec.ExceptionCode = STATUS_ILLEGAL_INSTRUCTION; rec.ExceptionAddress = const_cast<uint8_t*>(c.bytes.data());
        EXCEPTION_POINTERS info{&rec, &ctx};
        const auto result = ud::emulate(&info);
        check(result == EXCEPTION_CONTINUE_SEARCH && rec.ExceptionCode == STATUS_ACCESS_VIOLATION &&
              rec.NumberParameters == 2 && rec.ExceptionInformation[0] == 0 &&
              rec.ExceptionInformation[1] == c.expected && ctx.Rip == reinterpret_cast<DWORD64>(c.bytes.data()), c.name);
    }
    uint8_t rip_code[] = {0x66,0x0f,0xae,0x35,0,0,0,0x80}; // RIP - 0x80000000
    CONTEXT ctx{}; EXCEPTION_RECORD rec{};
    ctx.Rip = reinterpret_cast<DWORD64>(rip_code);
    rec.ExceptionCode = STATUS_ILLEGAL_INSTRUCTION;
    EXCEPTION_POINTERS info{&rec, &ctx};
    auto result = ud::emulate(&info);
    check(result == EXCEPTION_CONTINUE_SEARCH && rec.ExceptionCode == STATUS_ACCESS_VIOLATION &&
          rec.ExceptionInformation[0] == 0 && rec.ExceptionInformation[1] == ctx.Rip + 8 - 0x80000000ULL,
          "RIP-relative negative displacement");
    uint8_t code[] = {0x66,0x0f,0xae,0x31};
    void* page = VirtualAlloc(nullptr, kPage, MEM_RESERVE | MEM_COMMIT, PAGE_NOACCESS);
    ctx = {}; ctx.Rip = reinterpret_cast<DWORD64>(code); ctx.Rcx = reinterpret_cast<DWORD64>(page);
    rec = {}; rec.ExceptionCode = STATUS_ILLEGAL_INSTRUCTION;
    result = ud::emulate(&info);
    check(result == EXCEPTION_CONTINUE_SEARCH && rec.ExceptionCode == STATUS_ACCESS_VIOLATION &&
          rec.ExceptionInformation[0] == 0 && rec.ExceptionInformation[1] == reinterpret_cast<DWORD64>(page),
          "PAGE_NOACCESS faults as a read");
    DWORD old;
    VirtualProtect(page, kPage, PAGE_READONLY, &old);
    ctx.Rip = reinterpret_cast<DWORD64>(code); rec.ExceptionCode = STATUS_ILLEGAL_INSTRUCTION;
    result = ud::emulate(&info);
    check(result == EXCEPTION_CONTINUE_EXECUTION && ctx.Rip == reinterpret_cast<DWORD64>(code) + sizeof code,
          "PAGE_READONLY is accepted");
    VirtualProtect(page, kPage, PAGE_READONLY | PAGE_GUARD, &old);
    ctx.Rip = reinterpret_cast<DWORD64>(code); rec.ExceptionCode = STATUS_ILLEGAL_INSTRUCTION;
    result = ud::emulate(&info);
    check(result == EXCEPTION_CONTINUE_SEARCH && rec.ExceptionCode == STATUS_ACCESS_VIOLATION &&
          rec.ExceptionInformation[1] == reinterpret_cast<DWORD64>(page), "PAGE_GUARD faults");
    VirtualFree(page, 0, MEM_RELEASE);
}

void test_segment_overrides() {
    for (uint8_t prefix : {uint8_t{0x64}, uint8_t{0x65}}) {
        bool kept = false;
        const long got = run({prefix, 0x66, 0x0f, 0xae, 0x31}, &kept);
        check(got == 5 && kept, prefix == 0x64 ? "FS override skips address check" : "GS override skips address check");
    }
}

void test_lengths() {
    const Encoding cases[] = {
        {"clwb [rcx]", {0x66, 0x0F, 0xAE, 0x31}, 4},
        {"clwb [rax+rdx*4]", {0x66, 0x0F, 0xAE, 0x34, 0x90}, 5},
        {"clwb [rax+rdx*4+8]", {0x66, 0x0F, 0xAE, 0x74, 0x90, 0x08}, 6},
        {"clwb [rax+0x12345678]", {0x66, 0x0F, 0xAE, 0xB0, 0x78, 0x56, 0x34, 0x12}, 8},
        {"clwb [rip+0x100]", {0x66, 0x0F, 0xAE, 0x35, 0x00, 0x01, 0x00, 0x00}, 8},
        {"clwb [rsp]", {0x66, 0x0F, 0xAE, 0x34, 0x24}, 5},
        {"clwb [rsp+0x20]", {0x66, 0x0F, 0xAE, 0x74, 0x24, 0x20}, 6},
        {"clwb [rbp+0] (disp8 stands in for no base)", {0x66, 0x0F, 0xAE, 0x75, 0x00}, 5},
        {"clwb [rdx*4+0x1000] (SIB without base, disp32)", {0x66, 0x0F, 0xAE, 0x34, 0x95, 0x00, 0x10, 0x00, 0x00}, 9},
        {"clwb [r13+0] (REX.B)", {0x66, 0x41, 0x0F, 0xAE, 0x75, 0x00}, 6},
        {"clwb [r12] (REX.B, SIB)", {0x66, 0x41, 0x0F, 0xAE, 0x34, 0x24}, 6},
        {"clwb [r12+0x10]", {0x66, 0x41, 0x0F, 0xAE, 0x74, 0x24, 0x10}, 7},
        {"clwb [r8+r9*8+0x1000]", {0x66, 0x43, 0x0F, 0xAE, 0xB4, 0xC8, 0x00, 0x10, 0x00, 0x00}, 10},
        {"clwb [rcx] with REX.W", {0x66, 0x48, 0x0F, 0xAE, 0x31}, 5},
        {"clwb [rip+0x100] with REX.W", {0x66, 0x48, 0x0F, 0xAE, 0x35, 0x00, 0x01, 0x00, 0x00}, 9},
        {"clwb [r15] with bare REX (40)", {0x66, 0x40, 0x0F, 0xAE, 0x37}, 5},
    };
    int case_index = 0;
    for (const Encoding& c : cases) {
        bool kept = false;
        const long got = run(c.bytes, &kept, case_index++);
        char label[200];
        std::snprintf(label, sizeof label, "%s: RIP moves %ld bytes (got %ld), registers untouched", c.what, c.length, got);
        check(got == c.length && kept, label);
    }
}

void test_other_instructions_pass() {
    const Encoding cases[] = {
        {"clflush [rcx] (0F AE /7)", {0x0F, 0xAE, 0x39}, 0},
        {"clflushopt [rcx] (66 0F AE /7)", {0x66, 0x0F, 0xAE, 0x39}, 0},
        {"clflushopt [rcx] with REX", {0x66, 0x41, 0x0F, 0xAE, 0x39}, 0},
        {"xsaveopt [rcx] (0F AE /6 without 66)", {0x0F, 0xAE, 0x31}, 0},
        {"xsaveopt [rcx] with REX.W (no 66)", {0x48, 0x0F, 0xAE, 0x31}, 0},
        {"66 0F AE /6 with a register operand (mod 11)", {0x66, 0x0F, 0xAE, 0xF1}, 0},
        {"66 0F AE /5 [rcx]", {0x66, 0x0F, 0xAE, 0x29}, 0},
        {"66 0F AE /0 [rcx]", {0x66, 0x0F, 0xAE, 0x01}, 0},
        {"66 0F AF [rcx] (next opcode)", {0x66, 0x0F, 0xAF, 0x31}, 0},
        {"66 REX 0F 01 FC (CLZERO with prefixes is not CLWB)", {0x66, 0x40, 0x0F, 0x01, 0xFC}, 0},
        {"a NOP", {0x90}, 0},
        {"66 alone", {0x66}, 0},
    };
    for (const Encoding& c : cases) {
        bool kept = false;
        const long got = run(c.bytes, &kept);
        char label[200];
        std::snprintf(label, sizeof label, "%s: passed on, nothing modified", c.what);
        check(got == -1 && kept, label);
    }
}

// A readable page followed by a page that is reserved but not committed, so nothing can be read past the first.
uint8_t* page_before_a_hole() {
    auto* page = static_cast<uint8_t*>(VirtualAlloc(nullptr, 2 * kPage, MEM_RESERVE, PAGE_NOACCESS));
    VirtualAlloc(page, kPage, MEM_COMMIT, PAGE_READWRITE);
    uint8_t probe;
    SIZE_T got = 0;
    if (ReadProcessMemory(GetCurrentProcess(), page + kPage, &probe, 1, &got) && got == 1) {
        // Wine lets a process read its own uncommitted memory through ReadProcessMemory, so the dispatcher cannot see
        // where the readable code ends there. Windows can.
        std::printf("SKIP  the memory after the page is readable on this system (Wine): cut-off checks\n");
        VirtualFree(page, 0, MEM_RELEASE);
        return nullptr;
    }
    return page;
}

void test_cut_off_by_unreadable_memory() {
    // The instruction is complete only with its displacement, which sits on an unreadable page: not emulated.
    uint8_t* page = page_before_a_hole();
    if (!page) return;
    const std::vector<std::vector<uint8_t>> cut = {
        {0x66, 0x0F, 0xAE, 0x74, 0x90},        // [rax+rdx*4+disp8], disp8 missing
        {0x66, 0x0F, 0xAE, 0xB0, 0x78, 0x56},  // [rax+disp32], 2 of 4 bytes
        {0x66, 0x0F, 0xAE, 0x34},              // SIB byte missing
        {0x66, 0x0F, 0xAE},                    // ModRM missing
    };
    for (const auto& bytes : cut) {
        uint8_t* at = page + kPage - bytes.size();
        std::memcpy(at, bytes.data(), bytes.size());
        EXCEPTION_RECORD record{};
        CONTEXT context{};
        EXCEPTION_POINTERS pointers{};
        record.ExceptionCode = STATUS_ILLEGAL_INSTRUCTION;
        record.ExceptionAddress = at;
        context.Rip = reinterpret_cast<DWORD64>(at);
        pointers.ExceptionRecord = &record;
        pointers.ContextRecord = &context;
        char label[120];
        std::snprintf(label, sizeof label, "CLWB cut off after %zu bytes by unreadable memory: passed on, RIP unchanged",
                      bytes.size());
        check(ud::emulate(&pointers) == EXCEPTION_CONTINUE_SEARCH && context.Rip == reinterpret_cast<DWORD64>(at), label);
    }
    VirtualFree(page, 0, MEM_RELEASE);
}

void test_complete_at_page_end() {
    // The same instruction ending exactly at the last readable byte is taken.
    uint8_t* page = page_before_a_hole();
    if (!page) return;
    const uint8_t bytes[] = {0x66, 0x0F, 0xAE, 0x74, 0x90, 0x08};
    uint8_t* at = page + kPage - sizeof bytes;
    std::memcpy(at, bytes, sizeof bytes);
    EXCEPTION_RECORD record{};
    CONTEXT context{};
    EXCEPTION_POINTERS pointers{};
    record.ExceptionCode = STATUS_ILLEGAL_INSTRUCTION;
    record.ExceptionAddress = at;
    context.Rip = reinterpret_cast<DWORD64>(at);
    context.Rax = reinterpret_cast<DWORD64>(page);
    context.Rdx = 1;
    pointers.ExceptionRecord = &record;
    pointers.ContextRecord = &context;
    check(ud::emulate(&pointers) == EXCEPTION_CONTINUE_EXECUTION && context.Rip == reinterpret_cast<DWORD64>(at) + 6,
          "CLWB ending at the last readable byte: emulated, RIP past it");
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

using LineFn = void (*)(void*);

void test_installed_handler() {
    check(ud::install(true), "install(force) puts the handler in place");
    PVOID later = AddVectoredExceptionHandler(0, probe);
    // clwb [rcx]; clwb [rcx+8]; ret: where a real #UD would fall, the second instruction must still run.
    const uint8_t code[] = {0x66, 0x0F, 0xAE, 0x31, 0x66, 0x0F, 0xAE, 0x71, 0x08, 0xC3};
    void* mem = VirtualAlloc(nullptr, kPage, MEM_RESERVE | MEM_COMMIT, PAGE_EXECUTE_READWRITE);
    std::memcpy(mem, code, sizeof code);
    alignas(64) uint8_t line[128] = {};
    reinterpret_cast<LineFn>(mem)(line);
    if (ud::host_has_clwb())
        std::printf("SKIP  CLWB through the vectored handler: this CPU has CLWB, it ran natively\n");
    check(g_reached == 0, "two CLWB in a row run to the RET: none reaches the next handler as an unknown #UD");
    VirtualFree(mem, 0, MEM_RELEASE);
    RemoveVectoredExceptionHandler(later);
    ud::uninstall();
}

}  // namespace

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    test_fault_addresses();
    test_segment_overrides();
    test_lengths();
    test_other_instructions_pass();
    test_cut_off_by_unreadable_memory();
    test_complete_at_page_end();
    test_installed_handler();
    std::printf("%d checks, %d failed\n", g_checks, g_failed);
    return g_failed ? 1 : 0;
}
