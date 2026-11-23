#pragma once

#ifdef _WIN32
#include <windows.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

namespace leonida {

inline DWORD_PTR parse_affinity_mask(const std::wstring& val) {
    if (val == L"physical") {
        DWORD_PTR proc_mask = 0, sys_mask = 0;
        if (GetProcessAffinityMask(GetCurrentProcess(), &proc_mask, &sys_mask)) {
            // Even logical threads: 0x5555555555555555ULL (1 thread per physical core on SMT)
            return proc_mask & 0x5555555555555555ULL;
        }
        return 0;
    }
    wchar_t* end = nullptr;
    const unsigned long long m = std::wcstoull(val.c_str(), &end, 16);
    return (end && !*end) ? static_cast<DWORD_PTR>(m) : 0;
}

inline DWORD_PTR parse_affinity_mask_narrow(const char* val) {
    if (!val || !*val) return 0;
    if (std::strcmp(val, "physical") == 0) {
        DWORD_PTR proc_mask = 0, sys_mask = 0;
        if (GetProcessAffinityMask(GetCurrentProcess(), &proc_mask, &sys_mask)) {
            return proc_mask & 0x5555555555555555ULL;
        }
        return 0;
    }
    char* end = nullptr;
    const unsigned long long m = std::strtoull(val, &end, 16);
    return (end && !*end) ? static_cast<DWORD_PTR>(m) : 0;
}

inline bool apply_process_affinity(DWORD_PTR mask) {
    if (!mask) return false;
    return SetProcessAffinityMask(GetCurrentProcess(), mask) != FALSE;
}

inline DWORD_PTR configure_process_affinity(const std::wstring& cli_affinity = L"") {
    DWORD_PTR mask = 0;
    if (!cli_affinity.empty()) {
        mask = parse_affinity_mask(cli_affinity);
    } else if (const char* env = std::getenv("LEONIDA_CPU_AFFINITY"); env && *env) {
        mask = parse_affinity_mask_narrow(env);
    }
    if (mask) {
        if (apply_process_affinity(mask)) {
            std::fprintf(stderr, "[rt] process affinity set to 0x%llX\n",
                         static_cast<unsigned long long>(mask));
        } else {
            std::fprintf(stderr, "[rt] warning: failed to set process affinity to 0x%llX (error %lu)\n",
                         static_cast<unsigned long long>(mask), GetLastError());
        }
    }
    return mask;
}

}  // namespace leonida
#endif
