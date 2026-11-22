// Checks the title CPU affinity model, LEONIDA_TITLE_CORES parsing, shim registry bindings, and host behavior.
#include <windows.h>

#include <algorithm>
#include <bit>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "cpu_affinity.h"
#include "shims.h"

namespace {

using SetThreadAffinityMaskFn = DWORD_PTR(WINAPI*)(HANDLE, DWORD_PTR);
using GetProcessAffinityMaskFn = BOOL(WINAPI*)(HANDLE, PDWORD_PTR, PDWORD_PTR);
using SetProcessAffinityMaskFn = BOOL(WINAPI*)(HANDLE, DWORD_PTR);
using SetThreadIdealProcessorFn = DWORD(WINAPI*)(HANDLE, DWORD);
using SetThreadIdealProcessorExFn = BOOL(WINAPI*)(HANDLE, PPROCESSOR_NUMBER, PPROCESSOR_NUMBER);
using SetThreadSelectedCpuSetsFn = BOOL(WINAPI*)(HANDLE, const ULONG*, ULONG);
using SetProcessDefaultCpuSetsFn = BOOL(WINAPI*)(HANDLE, const ULONG*, ULONG);

int g_checks = 0;
int g_failures = 0;

void check(bool condition, const char* what) {
    ++g_checks;
    std::printf("%s %s\n", condition ? "PASS" : "FAIL", what);
    if (!condition) ++g_failures;
}

template <typename Fn>
Fn resolve(const char* dll, const char* name) {
    return reinterpret_cast<Fn>(rt::shims::find(dll, name));
}

DWORD WINAPI noop_thread_proc(LPVOID) { return 0; }

// Discovers host physical core 0's logical-processor mask in Group 0 (same rule as the shim).
uint64_t query_host_core0_mask(uint64_t host_proc_mask) {
    std::vector<uint64_t> cores;
    DWORD bytes = 0;
    ::GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &bytes);
    if (bytes > 0 && ::GetLastError() == ERROR_INSUFFICIENT_BUFFER) {
        std::vector<uint8_t> buf(bytes);
        auto* ptr = reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buf.data());
        if (::GetLogicalProcessorInformationEx(RelationProcessorCore, ptr, &bytes)) {
            size_t offset = 0;
            while (offset + sizeof(DWORD) * 2 <= bytes) {
                const auto* info =
                    reinterpret_cast<const SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(buf.data() + offset);
                if (info->Size == 0 || offset + info->Size > bytes) break;
                if (info->Relationship == RelationProcessorCore && info->Processor.GroupCount > 0) {
                    const GROUP_AFFINITY& gm = info->Processor.GroupMask[0];
                    if (gm.Group == 0) {
                        const uint64_t m = static_cast<uint64_t>(gm.Mask) & host_proc_mask;
                        if (m != 0) cores.push_back(m);
                    }
                }
                offset += info->Size;
            }
            std::sort(cores.begin(), cores.end(), [](uint64_t a, uint64_t b) {
                return std::countr_zero(a) < std::countr_zero(b);
            });
        }
    }
    if (!cores.empty()) return cores[0];
    return host_proc_mask != 0 ? (1ull << std::countr_zero(host_proc_mask)) : 1ull;
}

}  // namespace

int main() {
    // 1. Registry lookup across kernel32, kernelbase and the processthreads api sets.
    const char* const dlls[] = {
        "kernel32.dll",
        "kernelbase.dll",
        "api-ms-win-core-processthreads-l1-1-0.dll",
        "api-ms-win-core-processthreads-l1-1-1.dll",
        "api-ms-win-core-processthreads-l1-1-2.dll",
        "api-ms-win-core-processthreads-l1-1-3.dll",
    };
    const char* const names[] = {
        "SetThreadAffinityMask",
        "GetProcessAffinityMask",
        "SetProcessAffinityMask",
        "SetThreadIdealProcessor",
        "SetThreadIdealProcessorEx",
        "SetThreadSelectedCpuSets",
        "SetProcessDefaultCpuSets",
    };
    bool all_resolved = true;
    for (const char* dll : dlls) {
        for (const char* name : names) {
            if (!rt::shims::find(dll, name)) all_resolved = false;
        }
    }
    check(all_resolved, "all cpu_affinity shims resolve through rt::shims::find across kernel32/kernelbase/api-sets");

    // 2. Synthetic 8-core SMT host {0x3,0xC,0x30,0xC0,0x300,0xC00,0x3000,0xC000} and N=7.
    {
        const rt::cpu::Topology t =
            rt::cpu::make_topology({0x3, 0xC, 0x30, 0xC0, 0x300, 0xC00, 0x3000, 0xC000}, 7);
        check(t.cores == 7 && t.full == 0x7F, "8-core SMT N=7: full mask is 0x7F");
        check(t.to_host(0x1) == 0x3ull, "8-core SMT N=7: to_host(0x1) == 0x3");
        check(t.to_host(0x7F) == 0x3FFFull, "8-core SMT N=7: to_host(0x7F) == 0x3FFF");
        check(!t.to_host(0x80).has_value(), "8-core SMT N=7: to_host(0x80) is invalid");
        check(!t.to_host(0).has_value(), "8-core SMT N=7: to_host(0) is invalid");
        check(t.to_console(0xC) == 0x2ull, "8-core SMT N=7: to_console(0xC) == 0x2");
        check(t.to_console(0xC000) == 0ull, "8-core SMT N=7: to_console(0xC000) == 0 (core 7 unmapped)");
        check(t.host_processor(3) == 6u, "8-core SMT N=7: host_processor(3) == 6");
        check(t.to_console_index(6) == 3u, "8-core SMT N=7: to_console_index(6) == 3");
        check(t.to_console_index(14) == 0u, "8-core SMT N=7: to_console_index(14) wraps via host % N");
    }

    // 3. Smaller host (2 cores {0x3,0xC}, N=7): map wraps.
    {
        const rt::cpu::Topology t = rt::cpu::make_topology({0x3, 0xC}, 7);
        check(t.to_host(0x7F) == 0xFull, "2-core SMT N=7: to_host(0x7F) == 0xF");
        check(t.to_console(0x3) == 0b1010101ull, "2-core SMT N=7: to_console(0x3) == 0b1010101");
        check(t.host_processor(2) == 0u && t.host_processor(3) == 2u, "2-core SMT N=7: host_processor wraps");
    }

    // 4. N=1 and N=64 edge cases for `full`.
    {
        const rt::cpu::Topology t1 = rt::cpu::make_topology({0x3, 0xC}, 1);
        check(t1.full == 0x1ull, "N=1: full == 0x1");
        check(t1.to_host(0x1) == 0x3ull && !t1.to_host(0x2).has_value(), "N=1: mask bounds checked");

        const rt::cpu::Topology t64 = rt::cpu::make_topology({0x3, 0xC}, 64);
        check(t64.full == ~0ull, "N=64: full == ~0ull");
        check(t64.to_host(1ull << 63) == 0xCull && t64.to_host(~0ull) == 0xFull, "N=64: high bit translates");
    }

    // 5. LEONIDA_TITLE_CORES parsing.
    check(rt::cpu::parse_title_cores("7") == 7, "parse_title_cores(\"7\") == 7");
    check(rt::cpu::parse_title_cores("12") == 12, "parse_title_cores(\"12\") == 12");
    check(rt::cpu::parse_title_cores("0") == 7, "parse_title_cores(\"0\") == 7");
    check(rt::cpu::parse_title_cores("65") == 7, "parse_title_cores(\"65\") == 7");
    check(rt::cpu::parse_title_cores("abc") == 7, "parse_title_cores(\"abc\") == 7");
    check(rt::cpu::parse_title_cores(nullptr) == 7, "parse_title_cores(nullptr) == 7");

    // 6. Real machine checks through the shims.
    rt::cpu::set_topology_for_test(nullptr);
    const auto set_thread_aff = resolve<SetThreadAffinityMaskFn>("kernel32.dll", "SetThreadAffinityMask");
    const auto get_proc_aff = resolve<GetProcessAffinityMaskFn>("kernelbase.dll", "GetProcessAffinityMask");
    const auto set_proc_aff =
        resolve<SetProcessAffinityMaskFn>("api-ms-win-core-processthreads-l1-1-0.dll", "SetProcessAffinityMask");
    const auto set_ideal = resolve<SetThreadIdealProcessorFn>("kernel32.dll", "SetThreadIdealProcessor");
    const auto set_ideal_ex =
        resolve<SetThreadIdealProcessorExFn>("api-ms-win-core-processthreads-l1-1-0.dll", "SetThreadIdealProcessorEx");
    const auto set_thread_cpusets = resolve<SetThreadSelectedCpuSetsFn>("kernel32.dll", "SetThreadSelectedCpuSets");
    const auto set_proc_cpusets = resolve<SetProcessDefaultCpuSetsFn>("kernel32.dll", "SetProcessDefaultCpuSets");

    DWORD_PTR proc_mask = 0;
    DWORD_PTR sys_mask = 0;
    check(get_proc_aff(::GetCurrentProcess(), &proc_mask, &sys_mask) != FALSE,
          "real GetProcessAffinityMask succeeds");
    check(proc_mask != 0 && sys_mask != 0 && (proc_mask & ~sys_mask) == 0,
          "real GetProcessAffinityMask returns non-zero process subset of full system mask");
    const int n_cores = std::popcount(static_cast<uint64_t>(sys_mask));
    check(n_cores >= 1 && n_cores <= 64, "console core count deduced from system mask is in 1..64");

    // Null out-params validation.
    ::SetLastError(0);
    check(get_proc_aff(::GetCurrentProcess(), nullptr, &sys_mask) == FALSE &&
              ::GetLastError() == ERROR_INVALID_PARAMETER,
          "GetProcessAffinityMask rejects null process_mask");
    ::SetLastError(0);
    check(set_proc_aff(::GetCurrentProcess(), 0) == FALSE && ::GetLastError() == ERROR_INVALID_PARAMETER,
          "SetProcessAffinityMask rejects zero mask");

    DWORD_PTR host_proc_mask = 0;
    DWORD_PTR host_sys_mask = 0;
    const BOOL host_ok = ::GetProcessAffinityMask(::GetCurrentProcess(), &host_proc_mask, &host_sys_mask);
    const uint64_t expected_core0 = query_host_core0_mask(static_cast<uint64_t>(host_proc_mask));

    HANDLE thread = ::CreateThread(nullptr, 0, noop_thread_proc, nullptr, CREATE_SUSPENDED, nullptr);
    check(thread != nullptr, "CreateThread(CREATE_SUSPENDED) creates test thread");
    if (thread) {
        if (host_ok && std::popcount(static_cast<uint64_t>(host_proc_mask)) <= 1) {
            std::printf("SKIP SetThreadAffinityMask(0x1) core-0 confinement check (single logical processor)\n");
            const DWORD_PTR prev = set_thread_aff(thread, 0x1);
            check(prev != 0, "SetThreadAffinityMask(thread, 0x1) returns non-zero on single-CPU host");
        } else {
            const DWORD_PTR prev = set_thread_aff(thread, 0x1);
            GROUP_AFFINITY ga{};
            const BOOL got_ga = ::GetThreadGroupAffinity(thread, &ga);
            check(prev != 0 && got_ga != FALSE && ga.Group == 0 && static_cast<uint64_t>(ga.Mask) == expected_core0,
                  "SetThreadAffinityMask(thread, 0x1) confines thread to host core 0's logical processors");
        }

        if (n_cores < 64) {
            ::SetLastError(0);
            check(set_thread_aff(thread, static_cast<DWORD_PTR>(1ull << 63)) == 0 &&
                      ::GetLastError() == ERROR_INVALID_PARAMETER,
                  "SetThreadAffinityMask(thread, 1<<63) fails with ERROR_INVALID_PARAMETER");
        } else {
            ::SetLastError(0);
            check(set_thread_aff(thread, 0) == 0 && ::GetLastError() == ERROR_INVALID_PARAMETER,
                  "SetThreadAffinityMask(thread, 0) fails with ERROR_INVALID_PARAMETER");
        }

        const DWORD queried_ideal = set_ideal(thread, MAXIMUM_PROCESSORS);
        check(queried_ideal < static_cast<DWORD>(n_cores),
              "SetThreadIdealProcessor(thread, MAXIMUM_PROCESSORS) returns < N");

        ::SetLastError(0);
        check(set_ideal(thread, static_cast<DWORD>(n_cores)) == static_cast<DWORD>(-1) &&
                  ::GetLastError() == ERROR_INVALID_PARAMETER,
              "SetThreadIdealProcessor(thread, N) fails with ERROR_INVALID_PARAMETER");

        PROCESSOR_NUMBER ideal_in{};
        ideal_in.Group = 0;
        ideal_in.Number = 0;
        PROCESSOR_NUMBER prev_ex{};
        check(set_ideal_ex(thread, &ideal_in, &prev_ex) != FALSE && prev_ex.Group == 0 &&
                  prev_ex.Number < static_cast<BYTE>(n_cores),
              "SetThreadIdealProcessorEx sets core 0 and translates previous to console index");

        ::SetLastError(0);
        PROCESSOR_NUMBER bad_group{1, 0, 0};
        check(set_ideal_ex(thread, &bad_group, nullptr) == FALSE && ::GetLastError() == ERROR_INVALID_PARAMETER,
              "SetThreadIdealProcessorEx rejects non-zero Group");

        ::SetLastError(0);
        const BOOL cs_ok = set_thread_cpusets(thread, nullptr, 0);
        check(cs_ok != FALSE || ::GetLastError() == ERROR_CALL_NOT_IMPLEMENTED,
              "SetThreadSelectedCpuSets forwards or reports ERROR_CALL_NOT_IMPLEMENTED");
        ::SetLastError(0);
        const BOOL pcs_ok = set_proc_cpusets(::GetCurrentProcess(), nullptr, 0);
        check(pcs_ok != FALSE || ::GetLastError() == ERROR_CALL_NOT_IMPLEMENTED,
              "SetProcessDefaultCpuSets forwards or reports ERROR_CALL_NOT_IMPLEMENTED");

        ::ResumeThread(thread);
        ::WaitForSingleObject(thread, INFINITE);
        ::CloseHandle(thread);
    }

    std::printf("%d check(s), %d failure(s)\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
