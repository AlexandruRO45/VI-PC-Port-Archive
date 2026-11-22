// CPU-only checks: the Windows integration case creates a child to exercise shared job accounting.
#include "pipeline_commit_budget.h"

#include <cstdio>
#include <string>
#include <string_view>

namespace {
using namespace leonida::pipeline_budget;

int failures = 0;

void check(bool condition, const char* test) {
    std::printf("%s %s\n", condition ? "PASS" : "FAIL", test);
    if (!condition) ++failures;
}

void check_policy() {
    for (const auto invalid : {"", "0", "-1", "+1", "1MiB", "4096 ", "1.5", "AUTO", "18446744073709551616"})
        check(!parse(invalid), invalid[0] ? invalid : "reject empty cap");
    check(!parse(std::to_string(std::numeric_limits<std::size_t>::max() / mib + 1)),
          "reject MiB count that overflows its byte conversion");
    const auto largest = parse(std::to_string(std::numeric_limits<std::size_t>::max() / mib));
    check(largest && largest->bytes == std::numeric_limits<std::size_t>::max() / mib * mib,
          "largest byte count representable in whole MiB");
    const auto explicit_limit = parse("1024");
    check(explicit_limit && explicit_limit->bytes == 1024 * mib && !explicit_limit->automatic,
          "parse an explicit cap");
    const auto automatic = parse("auto");
    check(automatic && automatic->automatic && !automatic->bytes, "parse an automatic cap");
    check(automatic_budget(4000 * mib, 8000ull * mib) == 3000 * mib, "auto is bounded by available commit");
    check(automatic_budget(8000ull * mib, 4000 * mib) == 3000 * mib, "auto is bounded by available physical memory");
    check(automatic_budget(342 * mib, 900 * mib) == 256 * mib, "auto rounds down to exact whole MiB");
    check(automatic_budget(341 * mib, 900 * mib) == 0, "auto rejects a cap below the minimum");
    check(automatic_budget(0, 900 * mib) == 0, "auto refuses unknown or exhausted commit");
    check(automatic_budget(0xffffffffffffffffull, 0xffffffffffffffffull) > 0,
          "automatic calculation accepts the largest possible OS counter");
}

#ifdef _WIN32
constexpr std::size_t limit_mib = 192;
constexpr std::size_t large_mib = 112;
constexpr std::size_t small_mib = 16;

void* allocate(std::size_t size_mib, DWORD flags = MEM_RESERVE | MEM_COMMIT) {
    return VirtualAlloc(nullptr, size_mib * mib, flags, PAGE_READWRITE);
}

bool allocation_denied(std::size_t size_mib) {
    void* p = allocate(size_mib);
    if (!p) return true;
    VirtualFree(p, 0, MEM_RELEASE);
    return false;
}

int child() {
    BOOL in_job = FALSE;
    check(IsProcessInJob(GetCurrentProcess(), nullptr, &in_job) && in_job, "child inherited the job");
    void* small = allocate(small_mib);
    check(small != nullptr, "child can commit below remaining budget");
    if (small) VirtualFree(small, 0, MEM_RELEASE);
    check(allocation_denied(large_mib), "child and parent share the same commit cap");
    return failures != 0;
}

void check_child() {
    std::wstring path(1024, L'\0');
    while (true) {
        const DWORD n = GetModuleFileNameW(nullptr, path.data(), static_cast<DWORD>(path.size()));
        if (!n) {
            check(false, "find test executable");
            return;
        }
        if (n < path.size()) {
            path.resize(n);
            break;
        }
        path.resize(path.size() * 2);
    }
    std::wstring command = L"\"" + path + L"\" --child";
    STARTUPINFOW startup{};
    startup.cb = sizeof(startup);
    PROCESS_INFORMATION process{};
    if (!CreateProcessW(path.c_str(), command.data(), nullptr, nullptr, FALSE, 0, nullptr, nullptr, &startup,
                        &process)) {
        check(false, "launch a child inside the job");
        return;
    }
    CloseHandle(process.hThread);
    DWORD status = 1;
    const DWORD wait = WaitForSingleObject(process.hProcess, 15000);
    if (wait != WAIT_OBJECT_0) {
        TerminateProcess(process.hProcess, 1);
        WaitForSingleObject(process.hProcess, 5000);
    }
    const bool got_code = GetExitCodeProcess(process.hProcess, &status);
    CloseHandle(process.hProcess);
    check(wait == WAIT_OBJECT_0 && got_code && status == 0, "child exited and verified the shared limit");
}

void check_windows_job() {
    CommitJob job;
    if (const auto error = job.apply({limit_mib * mib, false})) {
        std::fprintf(stderr, "%s failed with Windows error %lu\n",
                     error->operation, static_cast<unsigned long>(error->code));
        check(false, "install the Windows job commit cap");
        return;
    }
    check(job.limit() == limit_mib * mib, "report installed budget");
    check(job.apply({limit_mib * mib, false}).has_value(), "refuse to replace a live job");

    void* p = allocate(large_mib);
    check(p != nullptr, "parent can commit below the cap");
    if (p) {
        // Query the counter while the successful allocation is live, before the reservation and denial probes.
        const auto peak = job.peak_job_memory_used();
        check(peak && *peak >= large_mib * mib, "Windows peak counter includes the successful allocation");
        check_child();
        VirtualFree(p, 0, MEM_RELEASE);
    }

    // Windows distinguishes address reservation from memory commit, important to GPU drivers that reserve
    // a large range up front. A single commit above the cap must fail even when the VA reservation fits.
    void* reservation = allocate(2 * limit_mib, MEM_RESERVE);
    check(reservation != nullptr, "uncommitted address reservation can exceed the cap");
    if (reservation) VirtualFree(reservation, 0, MEM_RELEASE);
    check(allocation_denied(2 * limit_mib), "a commit above the cap is denied");
}
#endif
}  // namespace

int main(int argc, char** argv) {
#ifdef _WIN32
    if (argc == 2 && std::string_view(argv[1]) == "--child") return child();
#else
    (void)argc;
    (void)argv;
#endif
    check_policy();
#ifdef _WIN32
    check_windows_job();
#endif
    return failures != 0;
}
