#include <windows.h>
#include <cstdio>
#include <cstdlib>
#include "../src/timer_resolution.h"

namespace {
int failures = 0;

void test_timer_resolution() {
    leonida::ScopedTimerResolution timer(1);
    if (!timer.active()) {
        std::fprintf(stderr, "FAIL: timer resolution 1 ms failed to activate\n");
        ++failures;
    }
    if (timer.period_ms() != 1) {
        std::fprintf(stderr, "FAIL: unexpected period_ms\n");
        ++failures;
    }
}

void test_priority_configuration() {
    _putenv_s("LEONIDA_PROCESS_PRIORITY", "normal");
    DWORD p1 = leonida::configure_process_priority();
    if (p1 != NORMAL_PRIORITY_CLASS) {
        std::fprintf(stderr, "FAIL: expected NORMAL_PRIORITY_CLASS\n");
        ++failures;
    }

    _putenv_s("LEONIDA_PROCESS_PRIORITY", "above_normal");
    DWORD p2 = leonida::configure_process_priority();
    if (p2 != ABOVE_NORMAL_PRIORITY_CLASS) {
        std::fprintf(stderr, "FAIL: expected ABOVE_NORMAL_PRIORITY_CLASS\n");
        ++failures;
    }

    // Reset to normal
    _putenv_s("LEONIDA_PROCESS_PRIORITY", "normal");
    leonida::configure_process_priority();
}

}  // namespace

int main() {
    test_timer_resolution();
    test_priority_configuration();
    if (failures) {
        std::printf("%d failure(s)\n", failures);
        return 1;
    }
    std::printf("PASS timer_resolution_test\n");
    return 0;
}
