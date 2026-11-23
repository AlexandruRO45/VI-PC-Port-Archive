#include <windows.h>
#include <cstdio>
#include <cwchar>

#include "../src/cpu_affinity.h"

namespace {
int failures = 0;

void check(bool condition, const char* message) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", message);
        ++failures;
    }
}

void test_parse_affinity() {
    check(leonida::parse_affinity_mask(L"0x5555") == 0x5555, "parse prefixed mask");
    check(leonida::parse_affinity_mask(L"5555") == 0x5555, "parse unprefixed mask");
    for (const wchar_t* invalid : {L"", L"0", L"zz", L"0x", L"-1", L"10000000000000000"}) {
        check(leonida::parse_affinity_mask(invalid) == 0, "reject malformed, zero or overlarge mask");
    }
}

void test_apply_affinity() {
    if (GetActiveProcessorGroupCount() != 1) {
        check(leonida::configure_process_affinity(L"physical") == 0, "skip physical mode across processor groups");
        return;
    }

    const HANDLE proc = GetCurrentProcess();
    DWORD_PTR original = 0, system = 0;
    if (!GetProcessAffinityMask(proc, &original, &system)) {
        check(false, "read process affinity");
        return;
    }

    check(SetEnvironmentVariableW(L"LEONIDA_CPU_AFFINITY", L"physical") != FALSE, "set affinity environment value");
    check(leonida::configure_process_affinity(L"zz") == 0, "invalid CLI value overrides the environment");
    DWORD_PTR actual = 0;
    check(GetProcessAffinityMask(proc, &actual, &system) && actual == original, "invalid input preserves affinity");

    const DWORD_PTR selected = leonida::configure_process_affinity();
    check(selected != 0 && (selected & original) == selected, "select physical cores from the allowed CPUs");
    check(GetProcessAffinityMask(proc, &actual, &system) && actual == selected, "apply physical affinity from the environment");
    check(SetProcessAffinityMask(proc, original) != FALSE, "restore original affinity");

    // With only an odd-numbered logical CPU allowed, physical mode must still select it.
    const DWORD_PTR odd = original & 0xAAAAAAAAAAAAAAAAULL;
    if (odd) {
        const DWORD_PTR chosen = odd & (0 - odd);
        if (SetProcessAffinityMask(proc, chosen)) {
            check(leonida::configure_process_affinity(L"physical") == chosen, "preserve the only eligible logical CPU");
            check(SetProcessAffinityMask(proc, original) != FALSE, "restore affinity after odd-only check");
        } else {
            check(false, "apply odd-only process affinity");
        }
    }

    const DWORD_PTR first = original & (0 - original);
    wchar_t explicit_mask[32];
    swprintf_s(explicit_mask, L"0x%llX", static_cast<unsigned long long>(first));
    check(leonida::configure_process_affinity(explicit_mask) == first, "apply explicit mask over environment default");
    check(GetProcessAffinityMask(proc, &actual, &system) && actual == first, "apply exact explicit mask");
    check(SetProcessAffinityMask(proc, original) != FALSE, "restore affinity after explicit-mask check");
    check(SetEnvironmentVariableW(L"LEONIDA_CPU_AFFINITY", nullptr) != FALSE, "clear affinity environment value");
}

}  // namespace

int main() {
    test_parse_affinity();
    test_apply_affinity();
    if (failures) {
        std::printf("%d failure(s)\n", failures);
        return 1;
    }
    std::printf("PASS cpu_affinity_test\n");
    return 0;
}
