#pragma once

#include <windows.h>

namespace rt::cpuinfo {

// Compact the CPU-set records in place using the supplied processor group limits.
DWORD filter_cpu_sets(unsigned char* data, ULONG bytes, WORD groups,
                      decltype(&GetMaximumProcessorCount) processor_count, ULONG& written);

}  // namespace rt::cpuinfo
