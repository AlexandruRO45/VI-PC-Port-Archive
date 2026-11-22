// cpu_affinity: title CPU affinity and ideal-processor translation for mapped images.
//
// An Xbox Series title runs on 7 of the console's 8 CPU cores (SMT on: 2 logical processors per physical core).
// ASSUMPTION (not measured; the game executable is not available here to confirm it): the title's affinity masks
// address one bit per console core (bits 0..N-1, default N = 7, overridden by LEONIDA_TITLE_CORES in 1..64). On a PC
// with SMT, Windows pairs adjacent logical processors (2k, 2k+1) onto physical core k, so passing a console mask
// such as 0x7F through unchanged would pin the title onto 3.5 physical cores and fail whenever bits exceed the host's
// processor count.
//
// The rules:
//   - host physical cores in Group 0 are discovered once via GetLogicalProcessorInformationEx(RelationProcessorCore),
//     ANDed with the host process affinity mask, empty entries dropped, and sorted by lowest set bit (falling back to
//     one entry per set bit of the host process mask if the query fails or yields nothing);
//   - console core i maps to host_cores[i] (or host_cores[i % host_cores.size()] when the host has fewer cores);
//   - SetThreadAffinityMask, GetProcessAffinityMask, SetProcessAffinityMask, SetThreadIdealProcessor and
//     SetThreadIdealProcessorEx translate masks and Group 0 core indices between console and host space;
//   - SetThreadSelectedCpuSets and SetProcessDefaultCpuSets are forwarded unchanged: CPU set IDs are opaque values
//     obtained from GetSystemCpuSetInformation (which goes to the host), so they are already host IDs. Resolved via
//     GetProcAddress on kernel32.dll on first use so the runtime still loads on older Windows.
// Only images mapped by the loader see these: they are bound through the registry, the runtime's own code calls the
// host directly.
#include <windows.h>

#include <algorithm>
#include <atomic>
#include <bit>
#include <cstdint>
#include <cstdio>
#include <optional>
#include <vector>

#include "cpu_affinity.h"
#include "shims.h"

namespace rt::cpu {
namespace {

constexpr int kDefaultTitleCores = 7;

int parse_title_cores_checked(const char* text, bool* valid) noexcept {
    if (valid) *valid = true;
    if (!text || !*text) {
        if (valid) *valid = false;
        return kDefaultTitleCores;
    }
    int value = 0;
    for (const char* p = text; *p; ++p) {
        if (*p < '0' || *p > '9') {
            if (valid) *valid = false;
            return kDefaultTitleCores;
        }
        value = value * 10 + (*p - '0');
        if (value > 64) {
            if (valid) *valid = false;
            return kDefaultTitleCores;
        }
    }
    if (value < 1 || value > 64) {
        if (valid) *valid = false;
        return kDefaultTitleCores;
    }
    return value;
}

int read_title_cores_from_env() noexcept {
    char buf[64] = {};
    ::SetLastError(ERROR_SUCCESS);
    const DWORD n = ::GetEnvironmentVariableA("LEONIDA_TITLE_CORES", buf, sizeof(buf));
    if (n == 0 && ::GetLastError() == ERROR_ENVVAR_NOT_FOUND) return kDefaultTitleCores;
    if (n >= sizeof(buf)) {
        std::fprintf(stderr, "[rt] cpu_affinity: LEONIDA_TITLE_CORES=<too long> ignored, using 7\n");
        return kDefaultTitleCores;
    }
    bool valid = true;
    const int cores = parse_title_cores_checked(buf, &valid);
    if (!valid) {
        std::fprintf(stderr, "[rt] cpu_affinity: LEONIDA_TITLE_CORES=%s ignored, using 7\n", buf);
    }
    return cores;
}

std::vector<uint64_t> query_host_cores() {
    DWORD_PTR proc_mask = 0;
    DWORD_PTR sys_mask = 0;
    if (!::GetProcessAffinityMask(::GetCurrentProcess(), &proc_mask, &sys_mask) || proc_mask == 0) {
        proc_mask = 1;
    }
    const uint64_t host_proc_mask = static_cast<uint64_t>(proc_mask);

    std::vector<uint64_t> host_cores;
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
                        const uint64_t masked = static_cast<uint64_t>(gm.Mask) & host_proc_mask;
                        if (masked != 0) host_cores.push_back(masked);
                    }
                }
                offset += info->Size;
            }
            std::sort(host_cores.begin(), host_cores.end(), [](uint64_t a, uint64_t b) {
                return std::countr_zero(a) < std::countr_zero(b);
            });
        }
    }

    if (host_cores.empty()) {
        for (uint64_t m = host_proc_mask; m != 0; m &= (m - 1)) {
            host_cores.push_back(1ull << std::countr_zero(m));
        }
    }
    return host_cores;
}

Topology build_host_topology() noexcept {
    const DWORD saved_err = ::GetLastError();
    const int cores = read_title_cores_from_env();
    Topology t = make_topology(query_host_cores(), cores);
    ::SetLastError(saved_err);
    return t;
}

// Intentionally immortal: mapped-DLL teardown can run after normal static destructors.
const Topology& default_topology() noexcept {
    static const Topology* const t = new Topology(build_host_topology());
    return *t;
}

std::atomic<const Topology*> g_test_topology{nullptr};

const Topology& active_topology() noexcept {
    if (const Topology* t = g_test_topology.load(std::memory_order_acquire)) return *t;
    return default_topology();
}

// ---- Win32 shims ------------------------------------------------------------------------------------------------

DWORD_PTR WINAPI set_thread_affinity_mask(HANDLE thread, DWORD_PTR mask) noexcept {
    const Topology& topo = active_topology();
    const std::optional<uint64_t> host_mask = topo.to_host(static_cast<uint64_t>(mask));
    if (!host_mask) {
        ::SetLastError(ERROR_INVALID_PARAMETER);
        return 0;
    }
    const DWORD_PTR prev = ::SetThreadAffinityMask(thread, static_cast<DWORD_PTR>(*host_mask));
    if (prev == 0) return 0;
    const uint64_t console_prev = topo.to_console(static_cast<uint64_t>(prev));
    return static_cast<DWORD_PTR>(console_prev != 0 ? console_prev : topo.full);
}

BOOL WINAPI get_process_affinity_mask(HANDLE process, PDWORD_PTR process_mask, PDWORD_PTR system_mask) noexcept {
    if (!process_mask || !system_mask) {
        ::SetLastError(ERROR_INVALID_PARAMETER);
        return FALSE;
    }
    DWORD_PTR host_proc = 0;
    DWORD_PTR host_sys = 0;
    if (!::GetProcessAffinityMask(process, &host_proc, &host_sys)) return FALSE;
    const Topology& topo = active_topology();
    const uint64_t console_proc = topo.to_console(static_cast<uint64_t>(host_proc));
    *process_mask = static_cast<DWORD_PTR>(console_proc != 0 ? console_proc : topo.full);
    *system_mask = static_cast<DWORD_PTR>(topo.full);
    return TRUE;
}

BOOL WINAPI set_process_affinity_mask(HANDLE process, DWORD_PTR mask) noexcept {
    const Topology& topo = active_topology();
    const std::optional<uint64_t> host_mask = topo.to_host(static_cast<uint64_t>(mask));
    if (!host_mask) {
        ::SetLastError(ERROR_INVALID_PARAMETER);
        return FALSE;
    }
    return ::SetProcessAffinityMask(process, static_cast<DWORD_PTR>(*host_mask));
}

DWORD WINAPI set_thread_ideal_processor(HANDLE thread, DWORD ideal) noexcept {
    const Topology& topo = active_topology();
    if (ideal == MAXIMUM_PROCESSORS) {
        const DWORD prev = ::SetThreadIdealProcessor(thread, MAXIMUM_PROCESSORS);
        if (prev == static_cast<DWORD>(-1)) return static_cast<DWORD>(-1);
        return topo.to_console_index(prev);
    }
    if (ideal >= static_cast<DWORD>(topo.cores)) {
        ::SetLastError(ERROR_INVALID_PARAMETER);
        return static_cast<DWORD>(-1);
    }
    const DWORD prev = ::SetThreadIdealProcessor(thread, topo.host_processor(ideal));
    if (prev == static_cast<DWORD>(-1)) return static_cast<DWORD>(-1);
    return topo.to_console_index(prev);
}

BOOL WINAPI set_thread_ideal_processor_ex(HANDLE thread, PPROCESSOR_NUMBER ideal, PPROCESSOR_NUMBER previous) noexcept {
    if (!ideal) {
        ::SetLastError(ERROR_INVALID_PARAMETER);
        return FALSE;
    }
    const Topology& topo = active_topology();
    if (ideal->Group != 0 || ideal->Number >= static_cast<DWORD>(topo.cores)) {
        ::SetLastError(ERROR_INVALID_PARAMETER);
        return FALSE;
    }
    PROCESSOR_NUMBER host_ideal{};
    host_ideal.Group = 0;
    host_ideal.Number = static_cast<BYTE>(topo.host_processor(ideal->Number));
    host_ideal.Reserved = 0;

    PROCESSOR_NUMBER host_prev{};
    if (!::SetThreadIdealProcessorEx(thread, &host_ideal, previous ? &host_prev : nullptr)) return FALSE;
    if (previous) {
        previous->Group = 0;
        previous->Number = static_cast<BYTE>(
            host_prev.Group == 0 ? topo.to_console_index(host_prev.Number)
                                 : (static_cast<uint32_t>(host_prev.Number) % static_cast<uint32_t>(topo.cores)));
        previous->Reserved = 0;
    }
    return TRUE;
}

using SetThreadSelectedCpuSetsFn = BOOL(WINAPI*)(HANDLE, const ULONG*, ULONG);
using SetProcessDefaultCpuSetsFn = BOOL(WINAPI*)(HANDLE, const ULONG*, ULONG);

template <typename Fn>
Fn resolve_kernel32(const char* name) noexcept {
    const HMODULE k32 = ::GetModuleHandleW(L"kernel32.dll");
    return k32 ? reinterpret_cast<Fn>(::GetProcAddress(k32, name)) : nullptr;
}

BOOL WINAPI set_thread_selected_cpu_sets(HANDLE thread, const ULONG* cpu_set_ids, ULONG count) noexcept {
    static const auto fn = resolve_kernel32<SetThreadSelectedCpuSetsFn>("SetThreadSelectedCpuSets");
    if (!fn) {
        ::SetLastError(ERROR_CALL_NOT_IMPLEMENTED);
        return FALSE;
    }
    return fn(thread, cpu_set_ids, count);
}

BOOL WINAPI set_process_default_cpu_sets(HANDLE process, const ULONG* cpu_set_ids, ULONG count) noexcept {
    static const auto fn = resolve_kernel32<SetProcessDefaultCpuSetsFn>("SetProcessDefaultCpuSets");
    if (!fn) {
        ::SetLastError(ERROR_CALL_NOT_IMPLEMENTED);
        return FALSE;
    }
    return fn(process, cpu_set_ids, count);
}

}  // namespace

std::optional<uint64_t> Topology::to_host(uint64_t console_mask) const noexcept {
    if (cores <= 0 || console_mask == 0 || (console_mask & ~full) != 0) return std::nullopt;
    uint64_t host = 0;
    for (int i = 0; i < cores; ++i) {
        if ((console_mask & (1ull << i)) != 0) host |= map[static_cast<size_t>(i)];
    }
    return host;
}

uint64_t Topology::to_console(uint64_t host_mask) const noexcept {
    uint64_t console = 0;
    for (int i = 0; i < cores; ++i) {
        if ((map[static_cast<size_t>(i)] & host_mask) != 0) console |= (1ull << i);
    }
    return console;
}

uint32_t Topology::host_processor(uint32_t console_index) const noexcept {
    if (cores <= 0 || map.empty()) return 0;
    const uint64_t m = map[static_cast<size_t>(console_index % static_cast<uint32_t>(cores))];
    return m != 0 ? static_cast<uint32_t>(std::countr_zero(m)) : 0;
}

uint32_t Topology::to_console_index(uint32_t host_number) const noexcept {
    if (cores <= 0) return 0;
    if (host_number < 64) {
        const uint64_t bit = 1ull << host_number;
        for (int i = 0; i < cores; ++i) {
            if ((map[static_cast<size_t>(i)] & bit) != 0) return static_cast<uint32_t>(i);
        }
    }
    return host_number % static_cast<uint32_t>(cores);
}

int parse_title_cores(const char* text) noexcept { return parse_title_cores_checked(text, nullptr); }

Topology make_topology(const std::vector<uint64_t>& host_cores, int cores) {
    Topology t;
    t.cores = (cores >= 1 && cores <= 64) ? cores : kDefaultTitleCores;
    t.full = (t.cores == 64) ? ~0ull : ((1ull << t.cores) - 1);
    t.map.resize(static_cast<size_t>(t.cores));
    const size_t count = host_cores.size();
    for (int i = 0; i < t.cores; ++i) {
        t.map[static_cast<size_t>(i)] = (count > 0) ? host_cores[static_cast<size_t>(i) % count] : 1ull;
    }
    return t;
}

void set_topology_for_test(const Topology* t) noexcept { g_test_topology.store(t, std::memory_order_release); }

}  // namespace rt::cpu

namespace {
using namespace rt::cpu;

#define CPU_AFFINITY_API(dll)                                                                                   \
    {dll, "SetThreadAffinityMask", reinterpret_cast<void*>(&set_thread_affinity_mask)},                         \
        {dll, "GetProcessAffinityMask", reinterpret_cast<void*>(&get_process_affinity_mask)},                   \
        {dll, "SetProcessAffinityMask", reinterpret_cast<void*>(&set_process_affinity_mask)},                   \
        {dll, "SetThreadIdealProcessor", reinterpret_cast<void*>(&set_thread_ideal_processor)},                 \
        {dll, "SetThreadIdealProcessorEx", reinterpret_cast<void*>(&set_thread_ideal_processor_ex)},            \
        {dll, "SetThreadSelectedCpuSets", reinterpret_cast<void*>(&set_thread_selected_cpu_sets)},              \
        {dll, "SetProcessDefaultCpuSets", reinterpret_cast<void*>(&set_process_default_cpu_sets)}
}  // namespace

SHIM_TABLE(cpu_affinity,
    CPU_AFFINITY_API("kernel32.dll"),
    CPU_AFFINITY_API("kernelbase.dll"),
    CPU_AFFINITY_API("api-ms-win-core-processthreads-l1-1-0.dll"),
    CPU_AFFINITY_API("api-ms-win-core-processthreads-l1-1-1.dll"),
    CPU_AFFINITY_API("api-ms-win-core-processthreads-l1-1-2.dll"),
    CPU_AFFINITY_API("api-ms-win-core-processthreads-l1-1-3.dll"),
);
