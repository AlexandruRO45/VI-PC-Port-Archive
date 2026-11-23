// Tests for the INSERTQ (SSE4a) emulator in src/cpu_ud_emulate.*.
//
// The emulation is driven directly with a synthetic CONTEXT and EXCEPTION_RECORD, so it is tested on any host, one
// that has SSE4a included. Expected values are written out by hand from the AMD APM (vol. 4) semantics. On a host
// with SSE4a the low quadword is also compared with what the CPU itself does for every length and index.
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

constexpr ULONGLONG kSrcLow = 0x0123456789ABCDEFull;
constexpr ULONGLONG kDstHigh = 0xDEADBEEFCAFEF00Dull;
constexpr ULONGLONG kOnes = ~0ull;

struct Fault {
    EXCEPTION_RECORD record{};
    CONTEXT context{};
    EXCEPTION_POINTERS pointers{};
    std::vector<uint8_t> bytes;
    explicit Fault(std::initializer_list<uint8_t> code, DWORD code_number = STATUS_ILLEGAL_INSTRUCTION) : bytes(code) {
        // Padding after the instruction so the dispatcher can always read its full 15-byte window.
        bytes.resize(bytes.size() + ud::kMaxInstructionLength, 0x90);
        record.ExceptionCode = code_number;
        record.ExceptionAddress = bytes.data();
        context.Rip = reinterpret_cast<DWORD64>(bytes.data());
        pointers.ExceptionRecord = &record;
        pointers.ContextRecord = &context;
    }
    M128A& xmm(unsigned n) { return context.FltSave.XmmRegisters[n]; }
};

// One INSERTQ case: dst = xmm1's low quadword before, src = xmm2's low quadword, then the field.
// `length` is the 6-bit encoded field (0 means 64).
struct Case {
    ULONGLONG dst;
    unsigned length;
    unsigned index;
    ULONGLONG expected;
};

// Hand-written. The source is 0x0123456789ABCDEF throughout.
const Case kCases[] = {
    {0, 8, 0, 0x00000000000000EFull},
    {0, 8, 8, 0x000000000000EF00ull},
    {0, 1, 0, 0x0000000000000001ull},
    {0, 1, 63, 0x8000000000000000ull},
    {0, 3, 5, 0x00000000000000E0ull},
    {0, 7, 57, 0xDE00000000000000ull},
    {0, 32, 0, 0x0000000089ABCDEFull},
    {0, 32, 16, 0x000089ABCDEF0000ull},
    {0, 32, 32, 0x89ABCDEF00000000ull},
    {0, 33, 31, 0xC4D5E6F780000000ull},
    {0, 63, 0, 0x0123456789ABCDEFull},
    {0, 63, 1, 0x02468ACF13579BDEull},
    {0, 0, 0, 0x0123456789ABCDEFull},  // length 0 is 64: the whole quadword
    {kOnes, 8, 0, 0xFFFFFFFFFFFFFFEFull},
    {kOnes, 1, 0, 0xFFFFFFFFFFFFFFFFull},  // a 1 into a 1
    {kOnes, 32, 16, 0xFFFF89ABCDEFFFFFull},
    {kOnes, 16, 48, 0xCDEFFFFFFFFFFFFFull},
    {kOnes, 63, 0, 0x8123456789ABCDEFull},  // bit 63 is outside the field and stays set
    {0x5555555555555555ull, 0, 0, 0x0123456789ABCDEFull},
};

// F2 0F 79 /r: xmm1 = ModRM.reg, xmm2 = ModRM.rm. The length and index travel in the high quadword of xmm2, with
// other bits set around the two fields, which the instruction ignores.
constexpr ULONGLONG kFieldJunk = 0xFFFF0000000000C0ull | (0xC0ull << 8);
ULONGLONG field_quad(unsigned length, unsigned index) { return length | (static_cast<ULONGLONG>(index) << 8) | kFieldJunk; }

void run_register_form(const Case& c, const char* label) {
    Fault f({0xF2, 0x0F, 0x79, 0xC1});  // insertq xmm0, xmm1
    f.xmm(0).Low = static_cast<LONGLONG>(c.dst);
    f.xmm(0).High = static_cast<LONGLONG>(kDstHigh);
    f.xmm(1).Low = static_cast<LONGLONG>(kSrcLow);
    f.xmm(1).High = static_cast<LONGLONG>(field_quad(c.length, c.index));
    const M128A src_before = f.xmm(1);
    const LONG r = ud::emulate(&f.pointers);
    char text[160];
    std::snprintf(text, sizeof text, "register form %s: dst low, dst high, source and RIP", label);
    check(r == EXCEPTION_CONTINUE_EXECUTION && static_cast<ULONGLONG>(f.xmm(0).Low) == c.expected &&
              static_cast<ULONGLONG>(f.xmm(0).High) == kDstHigh && std::memcmp(&f.xmm(1), &src_before, sizeof src_before) == 0 &&
              f.context.Rip == reinterpret_cast<DWORD64>(f.bytes.data()) + 4,
          text);
}

void run_immediate_form(const Case& c, const char* label) {
    Fault f({0xF2, 0x0F, 0x78, 0xC1, static_cast<uint8_t>(c.length | 0xC0), static_cast<uint8_t>(c.index | 0xC0)});
    f.xmm(0).Low = static_cast<LONGLONG>(c.dst);
    f.xmm(0).High = static_cast<LONGLONG>(kDstHigh);
    f.xmm(1).Low = static_cast<LONGLONG>(kSrcLow);
    f.xmm(1).High = 0x1122334455667788ll;  // not consulted by this form
    const M128A src_before = f.xmm(1);
    const LONG r = ud::emulate(&f.pointers);
    char text[160];
    std::snprintf(text, sizeof text, "immediate form %s: dst low, dst high, source and RIP", label);
    check(r == EXCEPTION_CONTINUE_EXECUTION && static_cast<ULONGLONG>(f.xmm(0).Low) == c.expected &&
              static_cast<ULONGLONG>(f.xmm(0).High) == kDstHigh && std::memcmp(&f.xmm(1), &src_before, sizeof src_before) == 0 &&
              f.context.Rip == reinterpret_cast<DWORD64>(f.bytes.data()) + 6,
          text);
}

void test_known_values() {
    for (const Case& c : kCases) {
        char label[64];
        std::snprintf(label, sizeof label, "len %u idx %u", c.length, c.index);
        run_register_form(c, label);
        run_immediate_form(c, label);
    }
}

void test_rex_registers() {
    // REX.R extends the destination (ModRM.reg), REX.B the source (ModRM.rm).
    struct Form {
        std::initializer_list<uint8_t> code;
        unsigned dst, src;
        bool immediate;
        const char* what;
    };
    const Form forms[] = {
        {{0xF2, 0x44, 0x0F, 0x79, 0xC1}, 8, 1, false, "REX.R: xmm8 <- xmm1"},
        {{0xF2, 0x41, 0x0F, 0x79, 0xC1}, 0, 9, false, "REX.B: xmm0 <- xmm9"},
        {{0xF2, 0x45, 0x0F, 0x79, 0xF7}, 14, 15, false, "REX.RB: xmm14 <- xmm15"},
        {{0xF2, 0x4D, 0x0F, 0x79, 0xFE}, 15, 14, false, "REX.WRB: xmm15 <- xmm14"},
        {{0xF2, 0x44, 0x0F, 0x78, 0xC1, 0x20, 0x10}, 8, 1, true, "immediate, REX.R: xmm8 <- xmm1"},
        {{0xF2, 0x41, 0x0F, 0x78, 0xC1, 0x20, 0x10}, 0, 9, true, "immediate, REX.B: xmm0 <- xmm9"},
        {{0xF2, 0x45, 0x0F, 0x78, 0xFE, 0x20, 0x10}, 15, 14, true, "immediate, REX.RB: xmm15 <- xmm14"},
    };
    for (const Form& form : forms) {
        Fault f(form.code);
        for (unsigned n = 0; n < 16; ++n) {  // every register a distinct pattern, so a wrong one shows
            f.xmm(n).Low = static_cast<LONGLONG>(0x1111111111111111ull * (n + 1) & 0x00FFFFFFFFFFFFFFull);
            f.xmm(n).High = static_cast<LONGLONG>(0x2222222222222222ull * (n + 1));
        }
        f.xmm(form.src).Low = static_cast<LONGLONG>(kSrcLow);
        if (!form.immediate) f.xmm(form.src).High = static_cast<LONGLONG>(field_quad(32, 16));
        f.xmm(form.dst).Low = 0;
        f.xmm(form.dst).High = static_cast<LONGLONG>(kDstHigh);
        const CONTEXT before = f.context;
        const LONG r = ud::emulate(&f.pointers);
        bool others_unchanged = true;
        for (unsigned n = 0; n < 16; ++n)
            if (n != form.dst && std::memcmp(&f.xmm(n), &before.FltSave.XmmRegisters[n], sizeof(M128A)) != 0)
                others_unchanged = false;
        char text[96];
        std::snprintf(text, sizeof text, "%s: field lands in the right register, the other 15 are untouched", form.what);
        check(r == EXCEPTION_CONTINUE_EXECUTION && static_cast<ULONGLONG>(f.xmm(form.dst).Low) == 0x000089ABCDEF0000ull &&
                  static_cast<ULONGLONG>(f.xmm(form.dst).High) == kDstHigh && others_unchanged,
              text);
    }
}

void test_undefined_fields_decline() {
    struct Undef {
        unsigned length, index;
        const char* what;
    } cases[] = {
        {0, 8, "length 64 at index 8"},
        {40, 30, "length 40 at index 30 (70 bits)"},
        {2, 63, "length 2 at index 63 (65 bits)"},
        {63, 2, "length 63 at index 2 (65 bits)"},
        {32, 33, "length 32 at index 33 (65 bits)"},
    };
    for (const Undef& u : cases) {
        for (int imm = 0; imm < 2; ++imm) {
            Fault f(imm ? std::initializer_list<uint8_t>{0xF2, 0x0F, 0x78, 0xC1, static_cast<uint8_t>(u.length), static_cast<uint8_t>(u.index)}
                        : std::initializer_list<uint8_t>{0xF2, 0x0F, 0x79, 0xC1});
            f.xmm(0).Low = 0x77;
            f.xmm(1).Low = static_cast<LONGLONG>(kSrcLow);
            f.xmm(1).High = static_cast<LONGLONG>(field_quad(u.length, u.index));
            const CONTEXT before = f.context;
            char text[128];
            std::snprintf(text, sizeof text, "%s form, %s: declined, nothing modified", imm ? "immediate" : "register", u.what);
            check(ud::emulate(&f.pointers) == EXCEPTION_CONTINUE_SEARCH && std::memcmp(&before, &f.context, sizeof before) == 0, text);
        }
    }
}

void test_other_instructions_pass() {
    struct Other {
        std::initializer_list<uint8_t> code;
        const char* what;
    } cases[] = {
        {{0x66, 0x0F, 0x78, 0xC1, 0x08, 0x04}, "EXTRQ xmm, imm8, imm8 (66 0F 78 /0)"},
        {{0x66, 0x0F, 0x79, 0xC1}, "EXTRQ xmm, xmm (66 0F 79)"},
        {{0x0F, 0x79, 0xC1}, "0F 79 without the F2 prefix"},
        {{0xF3, 0x0F, 0x79, 0xC1}, "F3 0F 79"},
        {{0xF2, 0x0F, 0x7A, 0xC1}, "F2 0F 7A, the neighbouring opcode"},
        {{0xF2, 0x0F, 0x79, 0x00}, "F2 0F 79 with a memory operand (mod != 3)"},
        {{0xF2, 0x0F, 0x78, 0x40, 0x08, 0x04, 0x00}, "F2 0F 78 with a memory operand (mod != 3)"},
        {{0x66, 0xF2, 0x0F, 0x79, 0xC1}, "66 F2 0F 79: an extra operand-size prefix"},
        {{0xF2, 0x90}, "F2 then a NOP"},
    };
    for (const Other& o : cases) {
        Fault f(o.code);
        f.xmm(0).Low = 0x77;
        f.xmm(1).Low = static_cast<LONGLONG>(kSrcLow);
        f.xmm(1).High = static_cast<LONGLONG>(field_quad(8, 8));
        const CONTEXT before = f.context;
        char text[128];
        std::snprintf(text, sizeof text, "%s: passed on, nothing modified", o.what);
        check(ud::emulate(&f.pointers) == EXCEPTION_CONTINUE_SEARCH && std::memcmp(&before, &f.context, sizeof before) == 0, text);
    }
    {
        // The immediate form with its second immediate byte on an unreadable page.
        auto* page = static_cast<uint8_t*>(VirtualAlloc(nullptr, 8192, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
        DWORD old;
        VirtualProtect(page + 4096, 4096, PAGE_NOACCESS, &old);
        const uint8_t cut[] = {0xF2, 0x0F, 0x78, 0xC1, 0x08};
        uint8_t* at = page + 4096 - sizeof cut;
        std::memcpy(at, cut, sizeof cut);
        Fault f({0x90});
        f.context.Rip = reinterpret_cast<DWORD64>(at);
        f.xmm(1).Low = static_cast<LONGLONG>(kSrcLow);
        // Wine reads a no-access page of its own process without complaint; there this check cannot be made.
        SIZE_T got = 0;
        uint8_t probe[sizeof cut + 1];
        if (ReadProcessMemory(GetCurrentProcess(), at, probe, sizeof probe, &got) && got == sizeof probe) {
            std::printf("SKIP  immediate form cut off by unreadable memory: this host reads no-access pages\n");
        } else {
            const CONTEXT before = f.context;
            const LONG r = ud::emulate(&f.pointers);
            check(r == EXCEPTION_CONTINUE_SEARCH && std::memcmp(&before, &f.context, sizeof before) == 0,
              "immediate form cut off before its second immediate by unreadable memory: passed on, nothing modified");
        }
        VirtualFree(page, 0, MEM_RELEASE);
    }
    Fault av({0xF2, 0x0F, 0x79, 0xC1}, STATUS_ACCESS_VIOLATION);
    check(ud::emulate(&av.pointers) == EXCEPTION_CONTINUE_SEARCH, "INSERTQ bytes under another exception code: passed on");
}

// The same instruction run by the CPU. Only where the CPU has SSE4a.
using InsertqFn = void (*)(M128A* dst_then_result, const M128A* src);

void test_against_the_cpu() {
    if (!ud::host_has_sse4a()) {
        std::printf("SKIP  comparison with the CPU: this CPU lacks SSE4a\n");
        return;
    }
    // movups xmm0,[rcx]; movups xmm1,[rdx]; insertq xmm0,xmm1; movups [rcx],xmm0; ret
    const uint8_t native_reg[] = {0x0F, 0x10, 0x01, 0x0F, 0x10, 0x0A, 0xF2, 0x0F, 0x79, 0xC1, 0x0F, 0x11, 0x01, 0xC3};
    void* mem = VirtualAlloc(nullptr, 4096, MEM_RESERVE | MEM_COMMIT, PAGE_EXECUTE_READWRITE);
    std::memcpy(mem, native_reg, sizeof native_reg);
    const auto native = reinterpret_cast<InsertqFn>(mem);

    const ULONGLONG dsts[] = {0, kOnes, 0x5555555555555555ull, 0xA5C3F00F12345678ull};
    int compared = 0, equal = 0, cpu_zeroed_high = 0;
    for (ULONGLONG dst : dsts) {
        for (unsigned length = 0; length < 64; ++length) {
            for (unsigned index = 0; index < 64; ++index) {
                if (index + (length ? length : 64) > 64) continue;  // undefined: the emulator declines these
                alignas(16) M128A cpu_dst, cpu_src;
                cpu_dst.Low = static_cast<LONGLONG>(dst);
                cpu_dst.High = static_cast<LONGLONG>(kDstHigh);
                cpu_src.Low = static_cast<LONGLONG>(kSrcLow);
                cpu_src.High = static_cast<LONGLONG>(field_quad(length, index));
                native(&cpu_dst, &cpu_src);

                Fault f({0xF2, 0x0F, 0x79, 0xC1});
                f.xmm(0).Low = static_cast<LONGLONG>(dst);
                f.xmm(0).High = static_cast<LONGLONG>(kDstHigh);
                f.xmm(1) = cpu_src;
                const bool ok = ud::emulate(&f.pointers) == EXCEPTION_CONTINUE_EXECUTION && f.xmm(0).Low == cpu_dst.Low &&
                                static_cast<ULONGLONG>(f.xmm(0).High) == kDstHigh;
                ++compared;
                if (ok) ++equal;
                if (cpu_dst.High == 0) ++cpu_zeroed_high;
            }
        }
    }
    char text[96];
    std::snprintf(text, sizeof text, "emulated low quadword equals the CPU's for every defined length and index (%d of %d)", equal, compared);
    check(compared > 0 && equal == compared, text);
    // The APM calls the high quadword of the result undefined; the emulator leaves it as it was.
    std::printf("      (the CPU zeroed the high quadword in %d of %d runs; not compared)\n", cpu_zeroed_high, compared);
    VirtualFree(mem, 0, MEM_RELEASE);
}

// Through the installed vectored handler: only meaningful on a CPU that lacks SSE4a.
void test_installed_handler() {
    if (ud::host_has_sse4a()) {
        std::printf("SKIP  INSERTQ through the vectored handler: this CPU has SSE4a\n");
        return;
    }
    ud::install(true);
    // movups xmm0,[rcx]; movups xmm1,[rdx]; insertq xmm0,xmm1; movups [rcx],xmm0; ret
    const uint8_t code[] = {0x0F, 0x10, 0x01, 0x0F, 0x10, 0x0A, 0xF2, 0x0F, 0x79, 0xC1, 0x0F, 0x11, 0x01, 0xC3};
    void* mem = VirtualAlloc(nullptr, 4096, MEM_RESERVE | MEM_COMMIT, PAGE_EXECUTE_READWRITE);
    std::memcpy(mem, code, sizeof code);
    alignas(16) M128A dst{0, static_cast<LONGLONG>(kDstHigh)}, src{static_cast<LONGLONG>(kSrcLow), static_cast<LONGLONG>(field_quad(32, 16))};
    reinterpret_cast<InsertqFn>(mem)(&dst, &src);
    check(static_cast<ULONGLONG>(dst.Low) == 0x000089ABCDEF0000ull && static_cast<ULONGLONG>(dst.High) == kDstHigh,
          "INSERTQ on a CPU without SSE4a: emulated by the installed handler");
    VirtualFree(mem, 0, MEM_RELEASE);
    ud::uninstall();
}

}  // namespace

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    test_known_values();
    test_rex_registers();
    test_undefined_fields_decline();
    test_other_instructions_pass();
    test_against_the_cpu();
    test_installed_handler();
    std::printf("%d checks, %d failed\n", g_checks, g_failed);
    return g_failed ? 1 : 0;
}
