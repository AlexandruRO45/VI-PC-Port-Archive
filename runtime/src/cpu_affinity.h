#pragma once

#ifdef _WIN32
#include <windows.h>

#include <cerrno>
#include <cstdio>
#include <cwchar>
#include <cwctype>
#include <limits>
#include <vector>

namespace leonida {

inline DWORD_PTR physical_core_affinity() {
    // SetProcessAffinityMask represents only one processor group.
    if (GetActiveProcessorGroupCount() != 1) return 0;

    DWORD_PTR allowed = 0, system = 0;
    if (!GetProcessAffinityMask(GetCurrentProcess(), &allowed, &system)) return 0;

    DWORD bytes = 0;
    if (GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &bytes) ||
        GetLastError() != ERROR_INSUFFICIENT_BUFFER) {
        return 0;
    }
    std::vector<unsigned char> topology(bytes);
    if (!GetLogicalProcessorInformationEx(RelationProcessorCore,
            reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(topology.data()), &bytes)) {
        return 0;
    }

    DWORD_PTR chosen = 0;
    for (DWORD offset = 0; offset < bytes;) {
        const auto* entry =
            reinterpret_cast<const SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(topology.data() + offset);
        // A RelationProcessorCore record has exactly one group mask.
        const DWORD_PTR eligible = entry->Processor.GroupMask[0].Mask & allowed;
        chosen |= eligible & (0 - eligible);  // lowest allowed logical CPU in this core
        offset += entry->Size;
    }
    return chosen;
}

inline DWORD_PTR parse_affinity_mask(const wchar_t* val) {
    if (!std::iswxdigit(*val)) return 0;
    wchar_t* end = nullptr;
    errno = 0;
    const auto mask = std::wcstoull(val, &end, 16);
    if (*end || errno == ERANGE || mask > (std::numeric_limits<DWORD_PTR>::max)()) return 0;
    return static_cast<DWORD_PTR>(mask);
}

inline DWORD_PTR configure_process_affinity(const wchar_t* cli_affinity = nullptr) {
    constexpr DWORD env_size = 64;
    wchar_t env[env_size];
    const wchar_t* value = cli_affinity;
    if (!value) {
        const DWORD len = GetEnvironmentVariableW(L"LEONIDA_CPU_AFFINITY", env, env_size);
        if (len == 0) return 0;
        if (len >= env_size) {
            std::fprintf(stderr, "[rt] warning: LEONIDA_CPU_AFFINITY is too long\n");
            return 0;
        }
        value = env;
    }

    const bool physical = std::wcscmp(value, L"physical") == 0;
    const DWORD_PTR mask = physical ? physical_core_affinity() : parse_affinity_mask(value);
    if (!mask) {
        std::fprintf(stderr, physical
            ? "[rt] warning: could not select physical CPU cores (requires one processor group)\n"
            : "[rt] warning: invalid CPU affinity (expected a nonzero hexadecimal mask or physical)\n");
        return 0;
    }
    if (!SetProcessAffinityMask(GetCurrentProcess(), mask)) {
        std::fprintf(stderr, "[rt] warning: failed to set process affinity to 0x%llX (error %lu)\n",
                     static_cast<unsigned long long>(mask), GetLastError());
        return 0;
    }

    std::fprintf(stderr, "[rt] process affinity set to 0x%llX\n", static_cast<unsigned long long>(mask));
    return mask;
}

}  // namespace leonida
#endif
