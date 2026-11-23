#include <windows.h>
#include <cstdio>
#include <cstdlib>
#include "../src/cpu_affinity.h"

namespace {
int failures = 0;

void test_parse_affinity() {
    DWORD_PTR m1 = leonida::parse_affinity_mask(L"0x5555");
    if (m1 != 0x5555) {
        std::fprintf(stderr, "FAIL: expected 0x5555, got 0x%llX\n", static_cast<unsigned long long>(m1));
        ++failures;
    }

    DWORD_PTR m2 = leonida::parse_affinity_mask(L"5555");
    if (m2 != 0x5555) {
        std::fprintf(stderr, "FAIL: expected 0x5555 without 0x prefix, got 0x%llX\n", static_cast<unsigned long long>(m2));
        ++failures;
    }

    DWORD_PTR m3 = leonida::parse_affinity_mask_narrow("0xFFFF");
    if (m3 != 0xFFFF) {
        std::fprintf(stderr, "FAIL: narrow parse expected 0xFFFF, got 0x%llX\n", static_cast<unsigned long long>(m3));
        ++failures;
    }

    DWORD_PTR m_phys = leonida::parse_affinity_mask(L"physical");
    if (m_phys == 0) {
        std::fprintf(stderr, "FAIL: physical affinity mask returned 0\n");
        ++failures;
    } else if (m_phys & 0xAAAAAAAAAAAAAAAAULL) {
        std::fprintf(stderr, "FAIL: physical affinity mask has odd bits set: 0x%llX\n",
                     static_cast<unsigned long long>(m_phys));
        ++failures;
    }
}

void test_apply_affinity() {
    DWORD_PTR orig_proc = 0, sys_mask = 0;
    if (!GetProcessAffinityMask(GetCurrentProcess(), &orig_proc, &sys_mask)) {
        std::fprintf(stderr, "FAIL: GetProcessAffinityMask failed\n");
        ++failures;
        return;
    }

    DWORD_PTR target = orig_proc & 0x5555555555555555ULL;
    if (target != 0) {
        _putenv_s("LEONIDA_CPU_AFFINITY", "physical");
        DWORD_PTR applied = leonida::configure_process_affinity();
        if (applied != target) {
            std::fprintf(stderr, "FAIL: expected applied mask 0x%llX, got 0x%llX\n",
                         static_cast<unsigned long long>(target), static_cast<unsigned long long>(applied));
            ++failures;
        }

        DWORD_PTR cur_proc = 0;
        GetProcessAffinityMask(GetCurrentProcess(), &cur_proc, &sys_mask);
        if (cur_proc != target) {
            std::fprintf(stderr, "FAIL: actual process mask 0x%llX != target 0x%llX\n",
                         static_cast<unsigned long long>(cur_proc), static_cast<unsigned long long>(target));
            ++failures;
        }

        // Restore original mask
        SetProcessAffinityMask(GetCurrentProcess(), orig_proc);
    }
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
