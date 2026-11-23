#include <windows.h>

#include <cstdio>
#include <cstring>
#include <vector>

#include "shims.h"

int main() {
    constexpr const char* name = "GetSystemCpuSetInformation";
    auto query = reinterpret_cast<decltype(&GetSystemCpuSetInformation)>(rt::shims::find("kernel32.dll", name));
    const bool shimmed = query != nullptr;
    if (!query) query = &GetSystemCpuSetInformation;  // the loader falls back to the host export

    ULONG required = 0;
    const BOOL probed = query(nullptr, 0, &required, nullptr, 0);
    const DWORD probe_error = GetLastError();
    if (probed || probe_error != ERROR_INSUFFICIENT_BUFFER || required < sizeof(SYSTEM_CPU_SET_INFORMATION)) {
        std::fprintf(stderr, "size query: result=%d error=%lu bytes=%lu\n", probed, probe_error, required);
        return 1;
    }

    std::vector<unsigned char> buffer(required);
    auto* data = reinterpret_cast<SYSTEM_CPU_SET_INFORMATION*>(buffer.data());
    ULONG written = 0;
    if (query(data, required - 1, &written, nullptr, 0) ||
        GetLastError() != ERROR_INSUFFICIENT_BUFFER || written != required) {
        std::fprintf(stderr, "short buffer: error=%lu bytes=%lu expected=%lu\n", GetLastError(), written, required);
        return 1;
    }
    if (!query(data, required, &written, nullptr, 0) || written != required) {
        std::fprintf(stderr, "exact buffer: error=%lu bytes=%lu expected=%lu\n", GetLastError(), written, required);
        return 1;
    }

    unsigned entries = 0, invalid = 0;
    const WORD groups = GetMaximumProcessorGroupCount();
    for (ULONG offset = 0; offset < written;) {
        SYSTEM_CPU_SET_INFORMATION entry{};
        if (written - offset < sizeof(entry)) return 1;
        std::memcpy(&entry, buffer.data() + offset, sizeof(entry));
        if (entry.Size < sizeof(entry) || entry.Size > written - offset || entry.Type != CpuSetInformation) return 1;
        ++entries;
        if (entry.CpuSet.Group >= groups ||
            entry.CpuSet.LogicalProcessorIndex >= GetMaximumProcessorCount(entry.CpuSet.Group)) ++invalid;
        offset += entry.Size;
    }
    std::printf("%s: probe error=%lu bytes=%lu entries=%u out-of-range=%u groups=%u max=%lu\n",
                shimmed ? "shim" : "host", probe_error, written, entries, invalid, static_cast<unsigned>(groups),
                GetMaximumProcessorCount(ALL_PROCESSOR_GROUPS));
    return invalid != 0;
}
