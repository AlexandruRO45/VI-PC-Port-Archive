// xmem.dll: the GDK's tracked allocator. Mapped onto the CRT's aligned heap and VirtualQueryEx.
//
// The measured ABI (docs from the clean-room spec) shows four imports:
//   XMemAlloc(size, attributes)
//   XMemFree(ptr, attributes)
//   XMemVirtualQuery(process, address, out64)
//   XMemGetWorkingSetStatistics(kind, out)   (kind 0x20000001 at all seven FSR call sites)
//
// The attribute word carries the alignment the caller asks for (xmem_alignment below). No block is aligned to less
// than 64 bytes (covers cache lines and SIMD), and a size of zero is rounded up to one byte so the result is a stable,
// non-null pointer that XMemFree can release. The word's other fields (object type, allocator id, memory type) do not
// change the answer.
#include <windows.h>
#include <malloc.h>
#include <psapi.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>

#include "shims.h"

namespace {

constexpr size_t kAlign = 64;

// The alignment an attribute word asks for. Two fields of the word matter here:
//   bits 24-28  log2 of the alignment in bytes
//   bits 14-15  page size: 0 = 4 KiB, 1 = 64 KiB; a block on 64 KiB pages starts on a page boundary
// The package's two words agree with this: amd_fidelityfx_upscaler_xbox.dll passes 0xa8004000 (alignment field 8, page
// field 1) for a block whose size it has rounded up to 256 bytes itself, and Bink2Scarlett.dll passes 0x5e50000
// (alignment field 5, 32 bytes) and aligns the block to 64 bytes on its own. The layout is the one ivall/xcl and
// omonkeyz/xbox-emulator decode (the two share lineage, so they count as one reading), and brcly/TuxBox measured
// 0x0480009e -> 16 bytes and 0x30804000 -> 64 KiB on retail titles (facts only; docs/runtime_imports.md has the
// files).
//
// Two things are deliberately not acted on, and each is logged once. Page codes 2 and 3 are 2 MiB pages in
// xbox-emulator alone, and XMemVirtualAlloc below refuses the 2 MiB mode for the same lack of a measurement: such a
// block is aligned by its alignment field only. An alignment field above 21 (2 MiB, the largest page size any layer
// names; the largest measured field is 16) would make the heap commit the alignment for every block, so the block
// keeps the 64-byte floor, the answer every caller got before the word was decoded.
constexpr unsigned kMaxAlignmentLog2 = 21;

void warn_once(std::atomic<bool>& warned, const char* what, uint64_t attributes) {
    if (warned.exchange(true, std::memory_order_relaxed)) return;
    std::fprintf(stderr, "[rt] xmem: XMemAlloc(attributes 0x%llx): %s\n", static_cast<unsigned long long>(attributes),
                 what);
}

size_t xmem_alignment(uint64_t attributes) {
    const unsigned log2 = static_cast<unsigned>((attributes >> 24) & 0x1f);
    const unsigned page = static_cast<unsigned>((attributes >> 14) & 3);
    size_t alignment = kAlign;
    if (log2 > kMaxAlignmentLog2) {
        static std::atomic<bool> warned{false};
        warn_once(warned, "an alignment above 2 MiB is not honoured; the block is 64-byte aligned", attributes);
    } else if ((size_t{1} << log2) > alignment) {
        alignment = size_t{1} << log2;
    }
    if (page == 1) {
        if (alignment < 0x10000) alignment = 0x10000;
    } else if (page != 0) {
        static std::atomic<bool> warned{false};
        warn_once(warned, "page-size codes 2 and 3 are not measured; the block follows its alignment field only",
                  attributes);
    }
    return alignment;
}

void* __stdcall XMemAlloc(size_t size, uint64_t attributes) {
    return _aligned_malloc(size ? size : 1, xmem_alignment(attributes));
}

void __stdcall XMemFree(void* p, uint64_t /*attributes*/) { _aligned_free(p); }

// Measured call sites pass a zeroed 64-byte structure. The MEMORY_BASIC_INFORMATION-compatible prefix is filled from
// VirtualQueryEx. The 32-bit word after it, at +48, says what kind of memory the region is: all four call sites in
// XboxBvhBuilder.dll (1.0.2.1; calls at RVAs 0x19c69, 0x19ccb, 0x19d9d, 0x19df9) report the address as unusable
// unless the call returns non-zero, bits 0-1 of that word are not both clear and State has MEM_COMMIT (the first
// also wants PAGE_READONLY or PAGE_READWRITE in Protect). ivall/xcl and omonkeyz/xbox-emulator (shared lineage) both
// answer 1 there for ordinary CPU memory, so committed memory gets 1 and the rest of the structure stays zero.
constexpr size_t kVirtualQueryOutSize = 64;
constexpr size_t kVirtualQueryKindOffset = sizeof(MEMORY_BASIC_INFORMATION);
constexpr uint32_t kVirtualQueryCpuMemory = 1;
static_assert(sizeof(MEMORY_BASIC_INFORMATION) <= kVirtualQueryOutSize,
              "the 64-byte XMemVirtualQuery output must hold MEMORY_BASIC_INFORMATION");
static_assert(kVirtualQueryKindOffset == 48 && kVirtualQueryKindOffset + sizeof(uint32_t) <= kVirtualQueryOutSize,
              "the memory-kind word follows MEMORY_BASIC_INFORMATION at +48");

// Returns the number of bytes written (the VirtualQuery convention) or 0 on failure. The exact return type and the
// meaning of every output field are not fully known; callers in the package only test for success.
SIZE_T __stdcall XMemVirtualQuery(HANDLE process, const void* address, void* out) {
    if (!out) return 0;
    std::memset(out, 0, kVirtualQueryOutSize);
    if (!address) return 0;

    static std::atomic<bool> warned{false};
    if (!warned.exchange(true, std::memory_order_relaxed)) {
        std::fprintf(stderr,
                     "[rt] xmem: XMemVirtualQuery: fields after the memory-kind word at +48 are unknown; "
                     "the 64-byte output is zero-filled there\n");
    }

    // A null handle means the current process; GetCurrentProcess() is the (HANDLE)-1 pseudo handle.
    HANDLE target = process ? process : GetCurrentProcess();
    MEMORY_BASIC_INFORMATION info{};
    SIZE_T written = VirtualQueryEx(target, address, &info, sizeof(info));
    if (!written) return 0;

    std::memcpy(out, &info, written < kVirtualQueryOutSize ? written : kVirtualQueryOutSize);
    if (info.State == MEM_COMMIT) {
        std::memcpy(static_cast<uint8_t*>(out) + kVirtualQueryKindOffset, &kVirtualQueryCpuMemory,
                    sizeof(kVirtualQueryCpuMemory));
    }
    return written;
}

// BOOL XMemGetWorkingSetStatistics(UINT32 kind, XMEM_WORKING_SET_STATISTICS* stats). Measured at all seven call sites
// in amd_fidelityfx_upscaler_xbox.dll (1.0.2.1; calls at RVAs 0x26b3, 0x2ea1, 0x39a9, 0x7452, 0x74d2, 0x7568,
// 0x75b1): ECX is the constant 0x20000001 (its meaning is not known) and RDX points at a stack buffer (0x80 bytes
// where the frame shows it). The caller reads a 64-bit size at +8 and treats a zero return as "no data". Taking the
// pointer from the first argument wrote to address 0x20000001.
// Only the first 16 bytes are touched: zero at +0 (unknown) and the process working set at +8. Do not claim more of
// the struct than is measured.
BOOL __stdcall XMemGetWorkingSetStatistics(uint32_t /*kind*/, void* stats) {
    if (!stats) {
        static std::atomic<bool> warned{false};
        if (!warned.exchange(true, std::memory_order_relaxed)) {
            std::fprintf(stderr, "[rt] xmem: XMemGetWorkingSetStatistics(nullptr) -> FALSE\n");
        }
        return FALSE;
    }

    PROCESS_MEMORY_COUNTERS pmc{};
    if (!GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) return FALSE;

    uint8_t* bytes = static_cast<uint8_t*>(stats);
    std::memset(bytes, 0, 16);
    uint64_t working_set = pmc.WorkingSetSize;
    std::memcpy(bytes + 8, &working_set, sizeof(working_set));
    return TRUE;
}

// XMem allocation-type page bits select the guest page granularity and must not be forwarded as Win32 flags.
constexpr DWORD kXmemLargePages = 0x20000000;
constexpr DWORD kXmem64kPages = 0x20400000;
constexpr DWORD kXmemPageModeMask = kXmem64kPages;
constexpr SIZE_T kXmemPageSize64k = 0x10000;
constexpr DWORD kXmemHostAllocationTypeMask =
    MEM_COMMIT | MEM_RESERVE | MEM_RESET | MEM_TOP_DOWN | MEM_WRITE_WATCH | 0x01000000;

bool round_up_size(SIZE_T value, SIZE_T alignment, SIZE_T& rounded) {
    const SIZE_T max = (std::numeric_limits<SIZE_T>::max)();
    if (alignment == 0 || (alignment & (alignment - 1)) != 0 || value > max - (alignment - 1)) return false;
    rounded = (value + alignment - 1) & ~(alignment - 1);
    return true;
}

void* WINAPI XMemVirtualAlloc(void* address, SIZE_T size, DWORD allocation_type, ULONGLONG attributes, DWORD protect) {
    // Host-backed support covers ordinary and 64 KiB guest reservations. The 2 MiB guest mode needs an aligned
    // reservation strategy that this shim does not provide, so fail explicitly instead of returning a misaligned
    // region. Nonzero attribute flags also fail until their host mapping is known.
    if (size == 0) {
        SetLastError(ERROR_INVALID_PARAMETER);
        return nullptr;
    }
    if (attributes != 0) {
        SetLastError(ERROR_NOT_SUPPORTED);
        return nullptr;
    }

    const DWORD page_mode = allocation_type & kXmemPageModeMask;
    if (page_mode == kXmemLargePages) {
        SetLastError(ERROR_NOT_SUPPORTED);
        return nullptr;
    }
    if (page_mode != 0 && page_mode != kXmem64kPages) {
        SetLastError(ERROR_INVALID_PARAMETER);
        return nullptr;
    }

    const DWORD unsupported = allocation_type & ~(kXmemHostAllocationTypeMask | kXmemPageModeMask);
    if (unsupported != 0) {
        SetLastError(ERROR_INVALID_PARAMETER);
        return nullptr;
    }

    DWORD host_type = allocation_type & kXmemHostAllocationTypeMask;
    if (host_type == 0) host_type = MEM_RESERVE | MEM_COMMIT;

    SYSTEM_INFO system_info{};
    GetSystemInfo(&system_info);
    const SIZE_T page_granularity = page_mode == kXmem64kPages ? kXmemPageSize64k : system_info.dwPageSize;
    const SIZE_T base_granularity =
        page_mode == kXmem64kPages
            ? kXmemPageSize64k
            : ((host_type & MEM_RESERVE) ? system_info.dwAllocationGranularity : system_info.dwPageSize);
    if (page_granularity == 0 || (page_granularity & (page_granularity - 1)) != 0 || base_granularity == 0 ||
        (base_granularity & (base_granularity - 1)) != 0) {
        SetLastError(ERROR_INVALID_PARAMETER);
        return nullptr;
    }

    if (address != nullptr) {
        const uintptr_t requested = reinterpret_cast<uintptr_t>(address);
        const uintptr_t max = (std::numeric_limits<uintptr_t>::max)();
        if (size > max - requested) {
            SetLastError(ERROR_INVALID_PARAMETER);
            return nullptr;
        }
        const uintptr_t requested_end = requested + size;
        const uintptr_t base = requested & ~(static_cast<uintptr_t>(base_granularity) - 1);
        if (requested_end > max - (static_cast<uintptr_t>(page_granularity) - 1)) {
            SetLastError(ERROR_INVALID_PARAMETER);
            return nullptr;
        }
        const uintptr_t end = (requested_end + static_cast<uintptr_t>(page_granularity) - 1) &
                              ~(static_cast<uintptr_t>(page_granularity) - 1);
        address = reinterpret_cast<void*>(base);
        size = static_cast<SIZE_T>(end - base);
    } else if (page_mode == kXmem64kPages && !round_up_size(size, page_granularity, size)) {
        SetLastError(ERROR_INVALID_PARAMETER);
        return nullptr;
    }

    return VirtualAlloc(address, size, host_type, protect);
}

UINT WINAPI XMemGetWriteWatch(DWORD flags, void* base_address, SIZE_T region_size, void** addresses, ULONG_PTR* count,
                              ULONG* granularity) {
    return GetWriteWatch(flags, base_address, region_size, addresses, count, granularity);
}

UINT WINAPI XMemResetWriteWatch(void* base_address, SIZE_T region_size) {
    return ResetWriteWatch(base_address, region_size);
}

}  // namespace

SHIM_TABLE(xmem,
    {"xmem.dll", "XMemAlloc", reinterpret_cast<void*>(&XMemAlloc)},
    {"xmem.dll", "XMemFree", reinterpret_cast<void*>(&XMemFree)},
    {"xmem.dll", "XMemVirtualQuery", reinterpret_cast<void*>(&XMemVirtualQuery)},
    {"xmem.dll", "XMemGetWorkingSetStatistics", reinterpret_cast<void*>(&XMemGetWorkingSetStatistics)},
    {"xmem.dll", "XMemVirtualAlloc", reinterpret_cast<void*>(&XMemVirtualAlloc)},
    {"xmem.dll", "XMemGetWriteWatch", reinterpret_cast<void*>(&XMemGetWriteWatch)},
    {"xmem.dll", "XMemResetWriteWatch", reinterpret_cast<void*>(&XMemResetWriteWatch)},
);
