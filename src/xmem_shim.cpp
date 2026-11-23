// Desktop xmem.dll. The default allocation exports share the unhooked allocator.
// Call shapes come from the observed call sites, not from a published header:
//   XMemAlloc(size, attributes) and XMemFree(allocation, attributes): Bink and AMD callers.
//   XMemVirtualQuery(process, address, buffer): the four BVH callers pass no length, supply a
//     zeroed 64-byte buffer and test the 32-bit result.
//   XMemGetWorkingSetStatistics(selector, output): the seven AMD callers pass 0x20000001 and
//     read the 64-bit value at +8 (zero on failure).
// Graphics allocations go through era_title_gpu, which registers them with era_query and, once a
// D3D12 device is attached to the registry, gives them desktop GPU addresses. The exports are
// unchanged; nothing here loads D3D12. DXGI and gdi32 are delay-loaded.
#include "era_heap.h"
#include "era_query.h"
#include "era_title_gpu.h"
#include "xmem_gpu_usage.h"

#include <cstdint>
#include <cstring>
#include <unordered_set>

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

namespace {
// The BVH callers' query buffer. era_query writes at most its first 0x31 bytes.
constexpr std::size_t query_buffer_bytes = 64;

bool is_graphics(std::uint64_t attributes) noexcept
{
    const auto type = era_heap::decode(attributes).memory_type;
    return type >= 1 && type <= 6;
}

// The selector the AMD callers pass, and the offset of the value they read.
constexpr UINT32 gpu_usage_selector = 0x20000001;
constexpr std::size_t gpu_usage_offset = 8;

// Physical page tracking: 64 KiB granularity with 4-bit shift as confirmed by title bextr usage.
constexpr std::size_t physical_page_size = 64 * 1024;
constexpr std::uint64_t pfn_shift = 4;

SRWLOCK physical_lock = SRWLOCK_INIT;
std::uint64_t next_pfn = 1;
std::unordered_set<std::uint64_t> tracked_pfns;
}

extern "C" void* XMemAlloc(SIZE_T bytes, ULONGLONG attributes)
{
    void* allocation = era_heap::allocate(bytes, attributes);
    // Desktop policy: graphics memory types count as GPU-intended for XMemVirtualQuery.
    if (allocation && is_graphics(attributes) && !era_title_gpu::track(allocation, bytes)) {
        era_heap::release_with_attributes(allocation, attributes);
        return nullptr;
    }
    return allocation;
}

extern "C" void XMemFree(void* allocation, ULONGLONG attributes)
{
    if (allocation && is_graphics(attributes)) {
        era_title_gpu::untrack(allocation);
    }
    era_heap::release_with_attributes(allocation, attributes);
}

extern "C" BOOL XMemVirtualQuery(HANDLE process, LPCVOID address, void* buffer)
{
    return era_query::query(process, address, buffer, query_buffer_bytes) != 0;
}

// Desktop policy (docs/experiments.md): writes only +8, with this process's GPU memory in bytes;
// on failure the output is untouched and the callers use zero.
extern "C" BOOL XMemGetWorkingSetStatistics(UINT32 selector, void* statistics)
{
    if (!statistics) {
        SetLastError(ERROR_INVALID_PARAMETER);
        return FALSE;
    }
    const auto bytes = selector == gpu_usage_selector ? xmem_gpu_usage::process_bytes() : std::nullopt;
    if (!bytes) {
        SetLastError(ERROR_NOT_SUPPORTED);
        return FALSE;
    }
    std::memcpy(static_cast<unsigned char*>(statistics) + gpu_usage_offset, &*bytes, sizeof(*bytes));
    return TRUE;
}

extern "C" BOOL XMemAllocatePhysicalPages(UINT32 attributes, ULONG_PTR* page_count, ULONG_PTR* physical_pages)
{
    if (!page_count || !physical_pages) {
        SetLastError(ERROR_INVALID_PARAMETER);
        return FALSE;
    }
    const ULONG_PTR requested = *page_count;
    if (requested == 0) {
        return TRUE;
    }
    constexpr ULONG_PTR max_physical_pages = 16ull * 1024 * 1024 * 1024 / physical_page_size;
    if (requested > max_physical_pages) {
        SetLastError(ERROR_OUTOFMEMORY);
        *page_count = 0;
        return FALSE;
    }

    AcquireSRWLockExclusive(&physical_lock);
    for (ULONG_PTR i = 0; i < requested; ++i) {
        const std::uint64_t pfn = next_pfn++;
        const std::uint64_t token = (pfn << pfn_shift) | (attributes & 0xF);
        physical_pages[i] = token;
        tracked_pfns.insert(token);
    }
    ReleaseSRWLockExclusive(&physical_lock);

    *page_count = requested;
    return TRUE;
}

extern "C" BOOL XMemFreePhysicalPages(ULONG_PTR page_count, const ULONG_PTR* physical_pages)
{
    if (!physical_pages || page_count == 0) {
        SetLastError(ERROR_INVALID_PARAMETER);
        return FALSE;
    }

    bool all_owned = true;
    AcquireSRWLockExclusive(&physical_lock);
    for (ULONG_PTR i = 0; i < page_count; ++i) {
        auto it = tracked_pfns.find(physical_pages[i]);
        if (it != tracked_pfns.end()) {
            tracked_pfns.erase(it);
        } else {
            all_owned = false;
        }
    }
    ReleaseSRWLockExclusive(&physical_lock);

    if (!all_owned) {
        SetLastError(ERROR_INVALID_PARAMETER);
        return FALSE;
    }
    return TRUE;
}

extern "C" BOOL XMemMapPhysicalPages(void* virtual_address, SIZE_T page_count, const ULONG_PTR* physical_pages)
{
    if (!virtual_address || page_count == 0) {
        SetLastError(ERROR_INVALID_PARAMETER);
        return FALSE;
    }
    const SIZE_T bytes = page_count * physical_page_size;
    if (physical_pages) {
        void* mapped = VirtualAlloc(virtual_address, bytes, MEM_COMMIT, PAGE_READWRITE);
        if (!mapped) {
            return FALSE;
        }
        return TRUE;
    } else {
        if (!VirtualFree(virtual_address, bytes, MEM_DECOMMIT)) {
            return FALSE;
        }
        return TRUE;
    }
}

extern "C" BOOL XMemMapPhysicalPagesScatter(void** virtual_addresses, SIZE_T page_count, const ULONG_PTR* physical_pages)
{
    if (!virtual_addresses || page_count == 0) {
        SetLastError(ERROR_INVALID_PARAMETER);
        return FALSE;
    }
    bool all_ok = true;
    for (SIZE_T i = 0; i < page_count; ++i) {
        void* va = virtual_addresses[i];
        if (!va) {
            all_ok = false;
            continue;
        }
        if (physical_pages) {
            if (!VirtualAlloc(va, physical_page_size, MEM_COMMIT, PAGE_READWRITE)) {
                all_ok = false;
            }
        } else {
            if (!VirtualFree(va, physical_page_size, MEM_DECOMMIT)) {
                all_ok = false;
            }
        }
    }
    return all_ok ? TRUE : FALSE;
}

extern "C" BOOL XMemTransferMemory(void* destination, const void* source, SIZE_T bytes, ULONGLONG attributes)
{
    (void)attributes;
    if (!destination || !source || bytes == 0) {
        SetLastError(ERROR_INVALID_PARAMETER);
        return FALSE;
    }
    std::memcpy(destination, source, bytes);
    return TRUE;
}

extern "C" void* XMemVirtualAlloc(void* address, SIZE_T bytes, DWORD allocation_type, DWORD protect, ULONGLONG attributes)
{
    (void)attributes;
    return VirtualAlloc(address, bytes, allocation_type, protect);
}

extern "C" BOOL XMemIsToolsMemAvailable()
{
    return FALSE;
}
