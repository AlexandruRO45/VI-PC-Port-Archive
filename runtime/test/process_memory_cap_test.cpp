// process_memory_cap.h: the job-object cap vk_pipeline_check puts on itself on Windows. No Vulkan device needed.
// On Windows the test caps its own process at 256 MiB and checks that a larger commit fails while a small one still
// succeeds; elsewhere it checks that there is no default cap and that a requested one is refused.
#include "process_memory_cap.h"

#include <cstdio>
#include <string>

static int failures = 0;

#define CHECK(cond)                                                    \
    do {                                                               \
        if (!(cond)) {                                                 \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            ++failures;                                                \
        }                                                              \
    } while (0)

int main() {
    std::string error;
    // 0 means no cap, and must not touch the process.
    CHECK(leonida::apply_process_memory_cap(0, error));
    CHECK(error.empty());

#ifdef _WIN32
    CHECK(leonida::default_process_memory_cap() > 0);

    constexpr SIZE_T kMiB = 1024 * 1024;
    // Before the cap, a 512 MiB commit is an ordinary allocation (backed by the page file, never touched).
    void* before = VirtualAlloc(nullptr, 512 * kMiB, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
    CHECK(before != nullptr);
    if (before) VirtualFree(before, 0, MEM_RELEASE);

    CHECK(leonida::apply_process_memory_cap(256 * kMiB, error));
    if (!error.empty()) std::printf("apply_process_memory_cap: %s\n", error.c_str());

    // Past the cap the commit fails instead of growing the system's commit charge.
    void* over = VirtualAlloc(nullptr, 512 * kMiB, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
    CHECK(over == nullptr);
    if (over) VirtualFree(over, 0, MEM_RELEASE);

    // Under it, the process works as before.
    void* under = VirtualAlloc(nullptr, 32 * kMiB, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
    CHECK(under != nullptr);
    if (under) VirtualFree(under, 0, MEM_RELEASE);

    // A second, nested cap (an --isolate child inside the parent's job) is accepted too.
    CHECK(leonida::apply_process_memory_cap(128 * kMiB, error));
    void* nested = VirtualAlloc(nullptr, 192 * kMiB, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
    CHECK(nested == nullptr);
    if (nested) VirtualFree(nested, 0, MEM_RELEASE);
#else
    // No default cap, and a requested one is refused rather than reported as installed.
    CHECK(leonida::default_process_memory_cap() == 0);
    CHECK(!leonida::apply_process_memory_cap(256ull * 1024 * 1024, error));
    CHECK(error.find("only implemented on Windows") != std::string::npos);
#endif

    if (failures) {
        std::printf("%d check(s) failed\n", failures);
        return 1;
    }
    std::printf("process_memory_cap: all checks passed\n");
    return 0;
}
