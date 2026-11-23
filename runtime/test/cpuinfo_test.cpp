#include <windows.h>

#include <cstdio>
#include <cstring>
#include <vector>

#include "cpuinfo.h"
#include "shims.h"

namespace {

DWORD WINAPI synthetic_processor_count(WORD) {
    return 2;  // Only the filter enforces the group bound.
}

bool test_filter() {
    SYSTEM_CPU_SET_INFORMATION records[3]{};
    for (auto& record : records) {
        record.Size = sizeof(record);
        record.Type = CpuSetInformation;
    }
    records[0].CpuSet.LogicalProcessorIndex = 2;  // first index outside group 0
    records[1].CpuSet.Group = 1;                 // first group outside the topology
    records[2].CpuSet.LogicalProcessorIndex = 1;
    records[2].CpuSet.Id = 42;
    unsigned char expected[sizeof(records[2])];
    std::memcpy(expected, &records[2], sizeof(expected));

    ULONG written = 0;
    const DWORD error = rt::cpuinfo::filter_cpu_sets(reinterpret_cast<unsigned char*>(records),
                                                     sizeof(records), 1, &synthetic_processor_count, written);
    if (error != ERROR_SUCCESS || written != sizeof(expected) ||
        std::memcmp(&records[0], expected, sizeof(expected)) != 0) {
        std::fprintf(stderr, "synthetic filter: error=%lu bytes=%lu expected=%zu\n", error, written,
                     sizeof(expected));
        return false;
    }
    std::puts("synthetic filter: kept 1 of 3 records");
    return true;
}

}  // namespace

int main() {
    if (!test_filter()) return 1;

    constexpr const char* name = "GetSystemCpuSetInformation";
    auto query = reinterpret_cast<decltype(&GetSystemCpuSetInformation)>(rt::shims::find("kernel32.dll", name));
    if (!query) {
        std::fprintf(stderr, "missing shim registration: kernel32.dll!%s\n", name);
        return 1;
    }

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
    std::printf("shim: probe error=%lu bytes=%lu entries=%u out-of-range=%u groups=%u max=%lu\n",
                probe_error, written, entries, invalid, static_cast<unsigned>(groups),
                GetMaximumProcessorCount(ALL_PROCESSOR_GROUPS));
    return invalid != 0;
}
