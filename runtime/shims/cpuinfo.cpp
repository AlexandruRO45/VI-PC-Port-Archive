#include <windows.h>

#include <cstddef>
#include <cstring>
#include <memory>
#include <new>

#include "cpuinfo.h"
#include "shims.h"

namespace {

DWORD read_cpu_sets(std::unique_ptr<unsigned char[]>& data, ULONG& bytes, HANDLE process, ULONG flags) {
    ULONG capacity = 0;
    for (;;) {
        if (::GetSystemCpuSetInformation(reinterpret_cast<SYSTEM_CPU_SET_INFORMATION*>(data.get()), capacity,
                                         &bytes, process, flags)) return ERROR_SUCCESS;
        const DWORD error = GetLastError();
        if (error != ERROR_INSUFFICIENT_BUFFER || bytes <= capacity) return error;
        capacity = bytes;
        data.reset(new (std::nothrow) unsigned char[capacity]);
        if (!data) return ERROR_NOT_ENOUGH_MEMORY;
    }
}

}  // namespace

DWORD rt::cpuinfo::filter_cpu_sets(unsigned char* data, ULONG bytes, WORD groups,
                                  decltype(&GetMaximumProcessorCount) processor_count, ULONG& written) {
    ULONG kept = 0;
    for (ULONG offset = 0; offset < bytes;) {
        constexpr size_t header_size = offsetof(SYSTEM_CPU_SET_INFORMATION, CpuSet);
        SYSTEM_CPU_SET_INFORMATION entry{};
        if (bytes - offset < header_size) return ERROR_INVALID_DATA;
        std::memcpy(&entry, data + offset, header_size);
        if (entry.Size < header_size || entry.Size > bytes - offset) return ERROR_INVALID_DATA;
        bool usable = true;
        if (entry.Type == CpuSetInformation) {
            if (entry.Size < sizeof(entry)) return ERROR_INVALID_DATA;
            std::memcpy(&entry, data + offset, sizeof(entry));
            usable = entry.CpuSet.Group < groups &&
                     entry.CpuSet.LogicalProcessorIndex < processor_count(entry.CpuSet.Group);
        }
        if (usable) {
            std::memmove(data + kept, data + offset, entry.Size);
            kept += entry.Size;
        }
        offset += entry.Size;
    }

    written = kept;
    return ERROR_SUCCESS;
}

namespace {

DWORD copy_cpu_sets(SYSTEM_CPU_SET_INFORMATION* output, ULONG capacity, ULONG& written,
                    HANDLE process, ULONG flags) {
    std::unique_ptr<unsigned char[]> data;
    ULONG bytes = 0;
    DWORD error = read_cpu_sets(data, bytes, process, flags);
    if (error != ERROR_SUCCESS) return error;

    // Wine 9 can enumerate more CPU sets than its processor groups can address.
    // Keep the host IDs and metadata for usable sets and size the result after filtering.
    ULONG kept = 0;
    error = rt::cpuinfo::filter_cpu_sets(data.get(), bytes, GetMaximumProcessorGroupCount(),
                                        &GetMaximumProcessorCount, kept);
    if (error != ERROR_SUCCESS) return error;
    written = kept;
    if (!output || capacity < kept) return ERROR_INSUFFICIENT_BUFFER;
    if (kept) std::memcpy(output, data.get(), kept);
    return ERROR_SUCCESS;
}

BOOL WINAPI game_get_system_cpu_set_information(SYSTEM_CPU_SET_INFORMATION* output, ULONG capacity,
                                                ULONG* written, HANDLE process, ULONG flags) {
    const DWORD previous_error = GetLastError();
    DWORD error = ERROR_INVALID_PARAMETER;
    if (written) {
        *written = 0;
        error = copy_cpu_sets(output, capacity, *written, process, flags);
    }
    SetLastError(error == ERROR_SUCCESS ? previous_error : error);
    return error == ERROR_SUCCESS;
}

}  // namespace

#define CPU_SETS(dll) {dll, "GetSystemCpuSetInformation", reinterpret_cast<void*>(&game_get_system_cpu_set_information)}
SHIM_TABLE(cpuinfo,
    CPU_SETS("kernel32.dll"),
    CPU_SETS("kernelbase.dll"),
    CPU_SETS("api-ms-win-core-processthreads-l1-1-3.dll"),
    CPU_SETS("api-ms-win-core-processthreads-l1-1-4.dll"),
    CPU_SETS("api-ms-win-core-processthreads-l1-1-5.dll"),
    CPU_SETS("api-ms-win-core-processthreads-l1-1-6.dll"),
    CPU_SETS("api-ms-win-core-processthreads-l1-1-7.dll"),
    CPU_SETS("api-ms-win-core-processthreads-l1-1-8.dll"),
);
