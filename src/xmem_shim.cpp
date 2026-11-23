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

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <vector>

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

// Physical pages. The names, call shapes and failure outcomes follow the public
// reimplementation evidence at omonkeyz/xbox-emulator f89e410, crates/compat/xmem/src/virt.rs:
// 64 KiB pages, a nonzero PFN array, a required 32-page multiple for the large-page flag,
// a zero count accepted, and chunked release with unknown or duplicate PFNs rejected.
// The backing is a desktop policy: each PFN owns one zero-filled 64 KiB VirtualAlloc block.
// Mapping, scatter mapping and transfer are separate contracts and remain unimplemented.
constexpr std::uint32_t xmem_large_pages = 0x20000000;
constexpr std::uint32_t xmem_64k_pages = 0x20400000;
constexpr std::size_t physical_page_bytes = 64 * 1024;
constexpr std::size_t physical_free_chunk = 0x1000;
constexpr std::size_t physical_page_limit = 1 << 20;

std::mutex physical_pages_mutex;
std::uint64_t next_physical_pfn = 0x1000;
std::unordered_map<std::uint64_t, void*> physical_pages;
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

// Desktop policy backed by one VirtualAlloc block per PFN. The PFN is an opaque, nonzero
// identifier until a mapping contract exists; the count is an in/out 64 KiB page count.
extern "C" BOOL XMemAllocatePhysicalPages(ULONGLONG flags, SIZE_T* page_count, ULONGLONG* pages)
{
    if (!page_count) {
        SetLastError(ERROR_INVALID_PARAMETER);
        return FALSE;
    }
    const SIZE_T count = *page_count;
    if ((static_cast<std::uint32_t>(flags) & xmem_64k_pages) == xmem_large_pages && (count & 0x1f) != 0) {
        *page_count = 0;
        SetLastError(ERROR_INVALID_PARAMETER);
        return FALSE;
    }
    if (count == 0) return TRUE;
    if (!pages) {
        *page_count = 0;
        SetLastError(ERROR_INVALID_PARAMETER);
        return FALSE;
    }
    if (count > physical_page_limit) {
        *page_count = 0;
        SetLastError(ERROR_NOT_ENOUGH_MEMORY);
        return FALSE;
    }

    std::lock_guard lock(physical_pages_mutex);
    std::vector<std::uint64_t> allocated;
    allocated.reserve(count);
    for (SIZE_T i = 0; i < count; ++i) {
        void* block = VirtualAlloc(nullptr, physical_page_bytes, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
        if (!block) {
            for (const auto pfn : allocated) {
                VirtualFree(physical_pages.at(pfn), 0, MEM_RELEASE);
                physical_pages.erase(pfn);
            }
            *page_count = 0;
            SetLastError(ERROR_NOT_ENOUGH_MEMORY);
            return FALSE;
        }
        const auto pfn = next_physical_pfn++;
        physical_pages.emplace(pfn, block);
        allocated.push_back(pfn);
    }
    for (SIZE_T i = 0; i < count; ++i) pages[i] = allocated[i];
    return TRUE;
}

// Chunked release from the public evidence: a rejected chunk leaves earlier chunks freed and
// the rest untouched. A zero count is a no-op even with a null array.
extern "C" BOOL XMemFreePhysicalPages(SIZE_T page_count, ULONGLONG* pages)
{
    if (page_count == 0) return TRUE;
    if (!pages) {
        SetLastError(ERROR_NOACCESS);
        return FALSE;
    }

    std::lock_guard lock(physical_pages_mutex);
    SIZE_T offset = 0;
    while (offset < page_count) {
        const SIZE_T chunk = std::min(page_count - offset, physical_free_chunk);
        std::unordered_set<std::uint64_t> seen;
        seen.reserve(chunk);
        for (SIZE_T i = 0; i < chunk; ++i) {
            const auto pfn = pages[offset + i];
            if (!physical_pages.contains(pfn) || !seen.insert(pfn).second) {
                SetLastError(ERROR_INVALID_PARAMETER);
                return FALSE;
            }
        }
        for (SIZE_T i = 0; i < chunk; ++i) {
            const auto pfn = pages[offset + i];
            VirtualFree(physical_pages.at(pfn), 0, MEM_RELEASE);
            physical_pages.erase(pfn);
        }
        offset += chunk;
    }
    return TRUE;
}
