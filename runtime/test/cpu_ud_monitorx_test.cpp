// Tests for the #UD emulation of MONITORX (0F 01 FA) and MWAITX (0F 01 FB) in src/cpu_ud_emulate.*.
//
// Driven directly with a synthetic CONTEXT and EXCEPTION_RECORD, so it runs on any host, one that has the
// instructions included.
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

struct Fault {
    EXCEPTION_RECORD record{};
    CONTEXT context{};
    EXCEPTION_POINTERS pointers{};
    explicit Fault(const uint8_t* code) {
        record.ExceptionCode = STATUS_ILLEGAL_INSTRUCTION;
        record.ExceptionAddress = const_cast<uint8_t*>(code);
        context.Rip = reinterpret_cast<DWORD64>(code);
        context.Rax = 0x1111;
        context.Rcx = 0x2222;
        context.Rbx = 0x3333;
        pointers.ExceptionRecord = &record;
        pointers.ContextRecord = &context;
    }
};

const uint8_t kMonitorx[] = {0x0F, 0x01, 0xFA, 0x90};
const uint8_t kMwaitx[] = {0x0F, 0x01, 0xFB, 0x90};

void test_resumes_after(const uint8_t* code, const char* name) {
    Fault f(code);
    const CONTEXT before = f.context;
    char label[96];
    std::snprintf(label, sizeof label, "%s: the handler resumes execution", name);
    check(ud::emulate(&f.pointers) == EXCEPTION_CONTINUE_EXECUTION, label);
    std::snprintf(label, sizeof label, "%s: RIP moves past the 3-byte instruction", name);
    check(f.context.Rip == reinterpret_cast<DWORD64>(code) + 3, label);
    CONTEXT expected = before;
    expected.Rip += 3;
    std::snprintf(label, sizeof label, "%s: no other register changes", name);
    check(std::memcmp(&expected, &f.context, sizeof expected) == 0, label);
}

void test_neighbours_pass() {
    // 0F 01 F9 is RDTSCP, not MONITORX/MWAITX.
    const uint8_t rdtscp[] = {0x0F, 0x01, 0xF9, 0x90};
    const uint8_t truncated[] = {0x0F, 0x01};
    for (const uint8_t* code : {rdtscp, truncated}) {
        Fault f(code);
        const CONTEXT before = f.context;
        check(ud::emulate(&f.pointers) == EXCEPTION_CONTINUE_SEARCH &&
                  std::memcmp(&before, &f.context, sizeof before) == 0,
              code == rdtscp ? "an opcode next to MWAITX (0F 01 F9): passed on, nothing modified"
                             : "a 2-byte prefix of the instruction: passed on, nothing modified");
    }
    Fault other(kMonitorx);
    other.record.ExceptionCode = STATUS_ACCESS_VIOLATION;
    check(ud::emulate(&other.pointers) == EXCEPTION_CONTINUE_SEARCH,
          "MONITORX bytes under another exception code: passed on");
}

void test_install_condition() {
    // The handler is needed when the CPU lacks any emulated instruction; a CPU with all of them keeps its own.
    const bool needed = !(ud::host_has_clzero() && ud::host_has_monitorx());
    const bool installed = ud::install(false);
    check(installed == needed, "install() without force installs exactly when the host lacks CLZERO or MONITORX");
    ud::uninstall();
}

}  // namespace

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    test_resumes_after(kMonitorx, "MONITORX");
    test_resumes_after(kMwaitx, "MWAITX");
    test_neighbours_pass();
    test_install_condition();
    std::printf("%d checks, %d failed\n", g_checks, g_failed);
    return g_failed ? 1 : 0;
}
