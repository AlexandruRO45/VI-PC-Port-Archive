// xmem.dll: the GDK's tracked allocator. Mapped onto the CRT's aligned heap and VirtualQueryEx.
//
// The measured ABI (docs from the clean-room spec) shows four imports:
//   XMemAlloc(size, attributes)
//   XMemFree(ptr, attributes)
//   XMemVirtualQuery(process, address, out64)
//   XMemGetWorkingSetStatistics(kind, out)   (kind 0x20000001 at all seven FSR call sites)
//
// ASSUMPTION: the attribute word is not decoded by the measured call sites (one passes 0x5e50000). Until a call
// site shows the alignment/type fields, every block is 64-byte aligned (covers cache lines and SIMD) and a size of
// zero is rounded up to one byte so the result is a stable, non-null pointer that XMemFree can release.
#include <windows.h>
#include <intrin.h>
#include <malloc.h>
#include <psapi.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "shims.h"
#include "startup_trace.h"

namespace {

constexpr size_t kAlign = 64;

// gta6rt --trace FILE: the first XMemAlloc and the first XMemFree for each distinct raw attribute word go to the
// startup trace as an "xmem_attributes" event (api, the word, the size for an allocation, the caller as module+RVA). The
// bits of the attribute word are not known yet; this records them from a real run without logging every allocation,
// which would fill the file. Off: one relaxed load per call.
void note_attributes(bool is_alloc, uint64_t attributes, size_t size, const void* caller) {
    if (!rt::startup_trace::enabled()) return;
    static std::mutex mutex;
    static std::set<std::pair<bool, uint64_t>> seen;
    {
        const std::lock_guard lock(mutex);
        if (!seen.insert({is_alloc, attributes}).second) return;
    }
    std::string fields = std::string("\"api\":\"") + (is_alloc ? "XMemAlloc" : "XMemFree") +
                         "\",\"attributes\":" + rt::startup_trace::quote(rt::startup_trace::hex(attributes));
    if (is_alloc) fields += ",\"size\":" + std::to_string(size);
    fields += ",\"caller\":" +
              rt::startup_trace::quote(rt::startup_trace::where(reinterpret_cast<uint64_t>(caller)));
    rt::startup_trace::event("xmem_attributes", fields);
}

void* __stdcall XMemAlloc(size_t size, uint64_t attributes) {
    note_attributes(true, attributes, size, _ReturnAddress());
    return _aligned_malloc(size ? size : 1, kAlign);
}

void __stdcall XMemFree(void* p, uint64_t attributes) {
    note_attributes(false, attributes, 0, _ReturnAddress());
    _aligned_free(p);
}

// Measured call sites pass a zeroed 64-byte structure. Only the MEMORY_BASIC_INFORMATION-compatible prefix is
// known: callers read at least the base and size fields. Fill that prefix from VirtualQueryEx and zero the rest.
constexpr size_t kVirtualQueryOutSize = 64;
static_assert(sizeof(MEMORY_BASIC_INFORMATION) <= kVirtualQueryOutSize,
              "the 64-byte XMemVirtualQuery output must hold MEMORY_BASIC_INFORMATION");

// Returns the number of bytes written (the VirtualQuery convention) or 0 on failure. The exact return type and the
// meaning of every output field are not fully known; callers in the package only test for success.
SIZE_T __stdcall XMemVirtualQuery(HANDLE process, const void* address, void* out) {
    if (!out) return 0;
    std::memset(out, 0, kVirtualQueryOutSize);
    if (!address) return 0;

    static std::atomic<bool> warned{false};
    if (!warned.exchange(true, std::memory_order_relaxed)) {
        std::fprintf(stderr,
                     "[rt] xmem: XMemVirtualQuery: fields after MEMORY_BASIC_INFORMATION are unknown; "
                     "the 64-byte output is zero-filled there\n");
    }

    // A null handle means the current process; GetCurrentProcess() is the (HANDLE)-1 pseudo handle.
    HANDLE target = process ? process : GetCurrentProcess();
    MEMORY_BASIC_INFORMATION info{};
    SIZE_T written = VirtualQueryEx(target, address, &info, sizeof(info));
    if (!written) return 0;

    std::memcpy(out, &info, written < kVirtualQueryOutSize ? written : kVirtualQueryOutSize);
    return written;
}

// BOOL XMemGetWorkingSetStatistics(UINT32 kind, XMEM_WORKING_SET_STATISTICS* stats). Measured at all seven call sites
// in amd_fidelityfx_upscaler_xbox.dll (1.0.2.1; calls at RVAs 0x26b3, 0x2ea1, 0x39a9, 0x7452, 0x74d2, 0x7568,
// 0x75b1): ECX is the constant 0x20000001 (its meaning is not known) and RDX points at a stack buffer (0x80 bytes
// where the frame shows it). The caller reads a 64-bit size at +8 and treats a zero return as "no data". Taking the
// pointer from the first argument wrote to address 0x20000001.
// The measured FSR sites establish argument shape, kind and the 64-bit working-set read at +8; +0's title-total memory
// limit interpretation is inferred from other Series implementations (xcl, xbox-emulator) and title engine telemetry
// sizing memory pools (RT-220). Note that if GlobalMemoryStatusEx reports 0 physical memory, XMem falls back to the
// 12 GiB title limit, whereas XSystemGetAppMemoryUsageLimit remains 0.
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

    MEMORYSTATUSEX ms{sizeof(ms)};
    uint64_t total_memory = 12ULL * 1024 * 1024 * 1024;
    if (GlobalMemoryStatusEx(&ms) && ms.ullTotalPhys != 0) {
        total_memory = ms.ullTotalPhys;
    }

    uint8_t* bytes = static_cast<uint8_t*>(stats);
    std::memset(bytes, 0, 16);
    std::memcpy(bytes + 0, &total_memory, sizeof(total_memory));
    uint64_t working_set = pmc.WorkingSetSize;
    std::memcpy(bytes + 8, &working_set, sizeof(working_set));
    return TRUE;
}

// The console's allocation type selects the page size with bits the host reads differently, so they are never
// forwarded as Win32 flags:
//   MEM_LARGE_PAGES (0x20000000) alone                        2 MiB pages
//   MEM_LARGE_PAGES | MEM_PHYSICAL (0x20400000)               64 KiB pages
//   MEM_4MB_PAGES (0x80000000), with or without MEM_LARGE_PAGES  4 MiB pages
// Values: the GDK's names as read in OpenScarlett (include/openscarlett/kernel/kernelx_abi.h, MIT) and xbox-emulator
// (crates/compat/xmem/src/virt.rs, GPL-3.0-or-later), which also picked this approach: on the host, MEM_LARGE_PAGES
// needs SeLockMemoryPrivilege and MEM_4MB_PAGES is rejected outright, so the shim serves every page size from
// ordinary pages and keeps what a title can see of the page size: the base is aligned to it and the region is a
// whole number of pages. The host still pages 4 KiB at a time underneath, which a title cannot observe except through
// TLB timing. Any other combination of these bits fails with ERROR_INVALID_PARAMETER.
constexpr DWORD kXmemLargePages = 0x20000000;
constexpr DWORD kXmemPhysical = 0x00400000;
constexpr DWORD kXmem64kPages = kXmemLargePages | kXmemPhysical;
constexpr DWORD kXmem4MbPages = 0x80000000;
constexpr DWORD kXmemPageModeMask = kXmem64kPages | kXmem4MbPages;
constexpr SIZE_T kXmemPageSize64k = 0x10000;
constexpr SIZE_T kXmemPageSize2M = 0x200000;
constexpr SIZE_T kXmemPageSize4M = 0x400000;
constexpr DWORD kVirtualAllocOperations = MEM_COMMIT | MEM_RESERVE | MEM_RESET | MEM_RESET_UNDO;
constexpr DWORD kXmemHostAllocationTypeMask = kVirtualAllocOperations | MEM_TOP_DOWN | MEM_WRITE_WATCH;
// Console-only allocation annotations (#3390). Preserve the caller's ordinary Win32 protection bits.
constexpr DWORD kConsoleAllocationTags = 0x10000000 | 0x40000000;  // MEM_GRAPHICS | MEM_TITLE
constexpr DWORD kConsoleProtectionTag = 0x04000000;
// 0x4000 next to MEM_RESERVE in a 64 KiB reservation (measured: allocation type 0x20406000 at the retail
// executable's RVA 0x1e86622, a range that XMemMapPhysicalPages fills right after). On Win32 it is MEM_DECOMMIT,
// which makes no sense with MEM_RESERVE. ASSUMPTION: a console hint for a range that receives physical pages; the
// physical-page mapper below converts any reserve-only range on first use, so the bit is dropped.
constexpr DWORD kConsoleReserveHint = 0x00004000;
// PAGE_GRAPHICS_NOACCESS (0x800) through PAGE_GRAPHICS_NOCACHE (0x40000): the GPU half of a console protection word.
constexpr DWORD kGraphicsProtectionMask = 0x0007F800;
constexpr DWORD kCpuProtectionMask = 0xFF;  // PAGE_NOACCESS .. PAGE_EXECUTE_WRITECOPY

// The guest page size the allocation type asks for, or 0 for the host's own pages. False for a combination that names
// no page size.
bool guest_page_size(DWORD allocation_type, SIZE_T& page) {
    switch (allocation_type & kXmemPageModeMask) {
        case 0: page = 0; return true;
        case kXmem64kPages: page = kXmemPageSize64k; return true;
        case kXmemLargePages: page = kXmemPageSize2M; return true;
        case kXmem4MbPages:
        case kXmem4MbPages | kXmemLargePages: page = kXmemPageSize4M; return true;
        default: return false;
    }
}

bool round_up_size(SIZE_T value, SIZE_T alignment, SIZE_T& rounded) {
    const SIZE_T max = (std::numeric_limits<SIZE_T>::max)();
    if (alignment == 0 || (alignment & (alignment - 1)) != 0 || value > max - (alignment - 1)) return false;
    rounded = (value + alignment - 1) & ~(alignment - 1);
    return true;
}

// A console protection word as the host reads it: the console tag and the GPU half (PAGE_GRAPHICS_*) are dropped,
// the CPU half and its modifiers (PAGE_GUARD, PAGE_NOCACHE, PAGE_WRITECOMBINE) are kept. Measured words: 0x801
// (graphics no-access, CPU no-access), 0x2004, 0x2404, 0x22004. A word with a GPU half and no CPU half gets
// PAGE_NOACCESS, or PAGE_READWRITE when the GPU may write: the D3D12 layer reads GPU memory from the CPU side.
DWORD host_protection(DWORD protect) {
    const DWORD graphics = protect & kGraphicsProtectionMask;
    DWORD host = protect & ~(kGraphicsProtectionMask | kConsoleProtectionTag);
    if (graphics != 0 && (host & kCpuProtectionMask) == 0) {
        const DWORD gpu_writes = PAGE_GRAPHICS_READWRITE | PAGE_GRAPHICS_EXECUTE_READWRITE;
        host |= (graphics & gpu_writes) ? PAGE_READWRITE : PAGE_NOACCESS;
    }
    return host;
}

// VirtualAlloc2, MapViewOfFile3 and UnmapViewOfFile2 (Windows 10 1803 and later; Wine has them too) are looked up
// rather than imported so the runtime still loads where they are missing. LEONIDA_XMEM_NO_VIRTUALALLOC2 (any value)
// makes the shim behave as if VirtualAlloc2 were missing, so the test suite runs the fallback below as well
// (xmem_probe).
using VirtualAlloc2Fn = PVOID(WINAPI*)(HANDLE, PVOID, SIZE_T, ULONG, ULONG, MEM_EXTENDED_PARAMETER*, ULONG);
using MapViewOfFile3Fn = PVOID(WINAPI*)(HANDLE, HANDLE, PVOID, ULONG64, SIZE_T, ULONG, ULONG,
                                        MEM_EXTENDED_PARAMETER*, ULONG);
using UnmapViewOfFile2Fn = BOOL(WINAPI*)(HANDLE, PVOID, ULONG);

FARPROC kernelbase_proc(const char* name) {
    HMODULE kernelbase = GetModuleHandleW(L"kernelbase.dll");
    return kernelbase ? GetProcAddress(kernelbase, name) : nullptr;
}

VirtualAlloc2Fn virtual_alloc2() {
    static const VirtualAlloc2Fn fn = []() -> VirtualAlloc2Fn {
        if (GetEnvironmentVariableW(L"LEONIDA_XMEM_NO_VIRTUALALLOC2", nullptr, 0) != 0) return nullptr;
        return reinterpret_cast<VirtualAlloc2Fn>(reinterpret_cast<void*>(kernelbase_proc("VirtualAlloc2")));
    }();
    return fn;
}

// The physical-page mapper needs VirtualAlloc2 even when LEONIDA_XMEM_NO_VIRTUALALLOC2 hides it from the aligned
// reservation path: placeholders have no fallback.
VirtualAlloc2Fn placeholder_alloc2() {
    static const VirtualAlloc2Fn fn =
        reinterpret_cast<VirtualAlloc2Fn>(reinterpret_cast<void*>(kernelbase_proc("VirtualAlloc2")));
    return fn;
}

MapViewOfFile3Fn map_view_of_file3() {
    static const MapViewOfFile3Fn fn =
        reinterpret_cast<MapViewOfFile3Fn>(reinterpret_cast<void*>(kernelbase_proc("MapViewOfFile3")));
    return fn;
}

UnmapViewOfFile2Fn unmap_view_of_file2() {
    static const UnmapViewOfFile2Fn fn =
        reinterpret_cast<UnmapViewOfFile2Fn>(reinterpret_cast<void*>(kernelbase_proc("UnmapViewOfFile2")));
    return fn;
}

// A new region of `size` bytes (a multiple of `alignment`) whose base is a multiple of `alignment`, which is larger than
// the host's 64 KiB allocation granularity. Without VirtualAlloc2, or when it refuses the alignment parameter, reserve
// `size + alignment` to find a free range, release it and reserve the aligned part of it; another thread can take
// that range in between, so try again a bounded number of times (the method of xbox-emulator's reserve_aligned in
// crates/compat/xmem/src/sys.rs). Each region is a single host allocation, so VirtualFree(base, 0, MEM_RELEASE) and
// VirtualQuery see exactly what was returned.
void* reserve_aligned(SIZE_T size, SIZE_T alignment, DWORD host_type, DWORD protect) {
    if (const VirtualAlloc2Fn alloc2 = virtual_alloc2()) {
        MEM_ADDRESS_REQUIREMENTS requirements{};
        requirements.Alignment = alignment;
        MEM_EXTENDED_PARAMETER parameter{};
        parameter.Type = MemExtendedParameterAddressRequirements;
        parameter.Pointer = &requirements;
        if (void* p = alloc2(GetCurrentProcess(), nullptr, size, host_type | MEM_RESERVE, protect, &parameter, 1)) {
            return p;
        }
        // Only a refused parameter can mean an older host without alignment support; anything else is final.
        if (GetLastError() != ERROR_INVALID_PARAMETER) return nullptr;
    }

    if (size > (std::numeric_limits<SIZE_T>::max)() - alignment) {
        SetLastError(ERROR_NOT_ENOUGH_MEMORY);
        return nullptr;
    }
    const SIZE_T probe_size = size + alignment;
    for (int attempt = 0; attempt < 64; ++attempt) {
        void* const probe = VirtualAlloc(nullptr, probe_size, MEM_RESERVE | (host_type & MEM_TOP_DOWN), PAGE_NOACCESS);
        if (!probe) return nullptr;
        const uintptr_t start = reinterpret_cast<uintptr_t>(probe);
        uintptr_t aligned = (start + alignment - 1) & ~(static_cast<uintptr_t>(alignment) - 1);
        // Top-down: take the highest aligned base inside the probe, as the host would have.
        if (host_type & MEM_TOP_DOWN) aligned = (start + alignment) & ~(static_cast<uintptr_t>(alignment) - 1);
        VirtualFree(probe, 0, MEM_RELEASE);
        if (void* p = VirtualAlloc(reinterpret_cast<void*>(aligned), size, host_type | MEM_RESERVE, protect)) return p;
        const DWORD error = GetLastError();
        if (error != ERROR_INVALID_ADDRESS) return nullptr;  // anything but losing the race is final
    }
    SetLastError(ERROR_NOT_ENOUGH_MEMORY);
    return nullptr;
}

// XMemVirtualAlloc's and a title's VirtualAlloc's common part. `default_operation`: no MEM_RESERVE/MEM_COMMIT/MEM_RESET
// bit means reserve and commit (XMemVirtualAlloc); a title's VirtualAlloc requires an operation, as on Win32.
// `protect` is already in host form (host_protection).
void* console_virtual_alloc(void* address, SIZE_T size, DWORD allocation_type, DWORD protect, bool default_operation) {
    if (size == 0) {
        SetLastError(ERROR_INVALID_PARAMETER);
        return nullptr;
    }
    allocation_type &= ~kConsoleAllocationTags;
    if ((allocation_type & kConsoleReserveHint) && (allocation_type & MEM_RESERVE) &&
        (allocation_type & kXmemPageModeMask)) {
        allocation_type &= ~kConsoleReserveHint;
    }
    SIZE_T guest_page = 0;
    if (!guest_page_size(allocation_type, guest_page)) {
        SetLastError(ERROR_INVALID_PARAMETER);
        return nullptr;
    }

    const DWORD unsupported = allocation_type & ~(kXmemHostAllocationTypeMask | kXmemPageModeMask);
    if (unsupported != 0) {
        SetLastError(ERROR_INVALID_PARAMETER);
        return nullptr;
    }

    DWORD host_type = allocation_type & kXmemHostAllocationTypeMask;
    if ((host_type & kVirtualAllocOperations) == 0) {
        if (!default_operation) {
            SetLastError(ERROR_INVALID_PARAMETER);
            return nullptr;
        }
        host_type |= MEM_RESERVE | MEM_COMMIT;
    }

    SYSTEM_INFO system_info{};
    GetSystemInfo(&system_info);
    const SIZE_T page_granularity = guest_page ? guest_page : system_info.dwPageSize;
    const SIZE_T base_granularity =
        guest_page ? guest_page
                   : ((host_type & MEM_RESERVE) ? system_info.dwAllocationGranularity : system_info.dwPageSize);
    if (page_granularity == 0 || (page_granularity & (page_granularity - 1)) != 0 || base_granularity == 0 ||
        (base_granularity & (base_granularity - 1)) != 0) {
        SetLastError(ERROR_INVALID_PARAMETER);
        return nullptr;
    }

    if (address != nullptr) {
        // Round the range out to whole guest pages: a commit inside a 2 MiB reservation commits that whole page. The
        // page size comes from this call's bits; nothing remembers the page size a region was reserved with.
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
    } else if (guest_page != 0 && !round_up_size(size, page_granularity, size)) {
        SetLastError(ERROR_INVALID_PARAMETER);
        return nullptr;
    }

    // A new region with pages coarser than the host's 64 KiB allocation granularity needs an aligned reservation.
    // (MEM_COMMIT without MEM_RESERVE and no address also makes a new region on Win32.) MEM_RESET with no address is
    // left to the host to refuse.
    if (address == nullptr && guest_page > system_info.dwAllocationGranularity &&
        (host_type & (MEM_RESERVE | MEM_COMMIT))) {
        return reserve_aligned(size, guest_page, host_type, protect);
    }
    return VirtualAlloc(address, size, host_type, protect);
}

// gta6rt --trace FILE / stderr: the first XMemVirtualAlloc for each distinct (attributes, type, protect) word, so a
// run records what the shim was asked for without logging every call.
void note_virtual_alloc(ULONGLONG attributes, DWORD allocation_type, DWORD protect, SIZE_T size, const void* caller) {
    static std::mutex mutex;
    static std::set<std::tuple<ULONGLONG, DWORD, DWORD>> seen;
    {
        const std::lock_guard lock(mutex);
        if (!seen.insert({attributes, allocation_type, protect}).second) return;
    }
    std::fprintf(stderr, "[rt] xmem: XMemVirtualAlloc attributes 0x%llx type 0x%lx protect 0x%lx (%zu bytes)\n",
                 static_cast<unsigned long long>(attributes), static_cast<unsigned long>(allocation_type),
                 static_cast<unsigned long>(protect), static_cast<size_t>(size));
    if (!rt::startup_trace::enabled()) return;
    rt::startup_trace::event(
        "xmem_attributes",
        std::string("\"api\":\"XMemVirtualAlloc\",\"attributes\":") +
            rt::startup_trace::quote(rt::startup_trace::hex(attributes)) +
            ",\"type\":" + rt::startup_trace::quote(rt::startup_trace::hex(allocation_type)) +
            ",\"protect\":" + rt::startup_trace::quote(rt::startup_trace::hex(protect)) +
            ",\"size\":" + std::to_string(static_cast<uint64_t>(size)) + ",\"caller\":" +
            rt::startup_trace::quote(rt::startup_trace::where(reinterpret_cast<uint64_t>(caller))));
}

void* WINAPI XMemVirtualAlloc(void* address, SIZE_T size, DWORD allocation_type, ULONGLONG attributes, DWORD protect) {
    // Host-backed support covers ordinary, 64 KiB, 2 MiB and 4 MiB guest pages (#4531). The attribute word (memory
    // type, GPU access; measured 0x3, 0x21, 0x23, 0x27, 0x181, 0x183, 0x283) has no host meaning: the host has one
    // memory pool, so it is recorded once per value and not otherwise used. #6709 accepted only its two reserve-only
    // pairs; the retail executable's static initializers also commit 2 MiB pools with the other words (#6824).
    if (size == 0) {
        SetLastError(ERROR_INVALID_PARAMETER);
        return nullptr;
    }
    note_virtual_alloc(attributes, allocation_type, protect, size, _ReturnAddress());
    return console_virtual_alloc(address, size, allocation_type, host_protection(protect), true);
}

UINT WINAPI XMemGetWriteWatch(DWORD flags, void* base_address, SIZE_T region_size, void** addresses, ULONG_PTR* count,
                              ULONG* granularity) {
    return GetWriteWatch(flags, base_address, region_size, addresses, count, granularity);
}

UINT WINAPI XMemResetWriteWatch(void* base_address, SIZE_T region_size) {
    return ResetWriteWatch(base_address, region_size);
}

// BOOL XMemTransferMemory(XMEM_TRANSFER_TYPE type, SIZE_T size). Its reference page and the XMEM_TRANSFER_TYPE enum
// page (gaming/gdk/docs/reference/system/xmem/{functions/xmemtransfermemory,enums/xmem_transfer_type}) are NDA
// topics on Microsoft Learn, so the shape comes from public code built against the GDK's xmem.h: Microsoft's
// XBOX-MonoGame-NetRumble sample (console/NetRumble.Bootstrap/Main.cpp) and FNA-XNA/NativeAOT-Xbox (Bootstrap/Main.cpp)
// both call `if (!XMemTransferMemory(XMemTransferTitleToSystem, additionalKernelPoolMemory))` with a `const size_t`,
// commented "Transfer additional memory to kernel pool" and "Must be multiples of 2MB sizes". So: two arguments, the
// enum (32 bits) in ECX and the size in RDX, and a zero return is failure. The public XLaunchNewGame page adds that
// "Memory transferred to the system using XMemTransferMemory (NDA topic) is persisted", and the game-save debugging
// page suggests it for E_OUTOFMEMORY "to transfer some memory to the system partition".
//
// On the console it moves budget between the title's partition and the system's. A PC has one pool and no budget
// split, so there is nothing to move: every direction and size returns TRUE and nothing is allocated or reserved. Not
// known (the pages are NDA): the numeric XMEM_TRANSFER_TYPE values, and whether the console refuses or rounds a size
// that is not a multiple of 2 MiB. Refusing would stop a title at startup (both samples return -1), while succeeding
// changes nothing here. TRUE is 1 in EAX, so it also reads as true if the real return type is a one-byte bool.
//
// gta6rt --trace FILE: the first call for each distinct (type, size) pair goes to the startup trace as an
// "xmem_transfer" event with the caller as module+RVA, so a run with the executable records the values it passes.
BOOL WINAPI XMemTransferMemory(uint32_t type, SIZE_T size) {
    static std::atomic<bool> warned{false};
    if (!warned.exchange(true, std::memory_order_relaxed)) {
        std::fprintf(stderr,
                     "[rt] xmem: XMemTransferMemory(type %u, %zu bytes): a PC has no title/system memory partition; "
                     "returning TRUE without moving anything (logged once)\n",
                     type, static_cast<size_t>(size));
    }
    if (rt::startup_trace::enabled()) {
        static std::mutex mutex;
        static std::set<std::pair<uint32_t, uint64_t>> seen;
        bool first = false;
        {
            const std::lock_guard lock(mutex);
            first = seen.insert({type, static_cast<uint64_t>(size)}).second;
        }
        if (first) {
            rt::startup_trace::event(
                "xmem_transfer",
                "\"api\":\"XMemTransferMemory\",\"type\":" + std::to_string(type) +
                    ",\"size\":" + std::to_string(static_cast<uint64_t>(size)) + ",\"caller\":" +
                    rt::startup_trace::quote(
                        rt::startup_trace::where(reinterpret_cast<uint64_t>(_ReturnAddress()))));
        }
    }
    return TRUE;
}

// ---- Physical pages ---------------------------------------------------------------------------------------------
//
// The retail executable (1.0.2.1) imports four physical-page exports. Their shapes, read from the call sites (static
// look; nothing executed), are those of Win32's AWE functions with 64 KiB pages and no process handle:
//
//   BOOL  XMemAllocatePhysicalPages(ULONG64 attributes, ULONG_PTR* count, ULONG_PTR* pages)
//         ECX 0x180, 0x280 or 0x380 (the same family of words XMemVirtualAlloc gets); RDX points at the page count
//         (a size >> 16); R8 at an array of `count` ULONG_PTRs. Every caller requires EAX == 1 and the count unchanged,
//         and breaks (int3) otherwise (RVAs 0x1e6c22d, 0x1e866e4, 0x2216493, 0x5eb27a7).
//   PVOID XMemMapPhysicalPages(PVOID address, ULONG_PTR count, const ULONG_PTR* pages)
//         maps `count` pages at `address`, a range reserved by XMemVirtualAlloc just before (attributes 0x23 / 0x27,
//         protection 0x2004); a null `pages` unmaps (RVA 0x24730e9 and the two after it). The callers test RAX for
//         non-null (RVAs 0x1e6c295, 0x1e86738, ...).
//   BOOL  XMemMapPhysicalPagesScatter(PVOID* addresses, ULONG_PTR count, const ULONG_PTR* pages)
//         one address per page; a null `pages` unmaps them all (RVA 0xb18dde). Callers test EAX for nonzero.
//   BOOL  XMemFreePhysicalPages(ULONG_PTR count, const ULONG_PTR* pages)
//         count by value, unlike Win32's FreeUserPhysicalPages (RVAs 0x1e6c32d, 0x1e868e2, 0x2473133). The return
//         value is not tested.
//
// Host model: a physical page is a 64 KiB slice of a pagefile-backed section (one section per allocation call, so
// the pages of one call are contiguous in it), and its page number is an index into a table of those slices. Mapping
// a page puts a view of its slice at the title's address. Win32 can only map a view into a placeholder, and the
// title reserves the range with an ordinary XMemVirtualAlloc, so the first map into a reserve-only region releases it
// and reserves the same range again as one placeholder; each page then splits a 64 KiB placeholder off and replaces
// it with a view. A page can be mapped at several addresses at once, as on the console. Unmapping restores the
// placeholder.
//
// ASSUMPTIONS: the attribute word (cacheability, GPU access) has no host meaning and is ignored; page numbers are
// opaque to the title (they start at kFirstPageNumber and increase; GPU physical addresses are not modelled); pages are
// zeroed when allocated, as AllocateUserPhysicalPages does; a mapping is always PAGE_READWRITE for the CPU. Not done:
// releasing a region that holds views (VirtualFree on it fails while views remain) and committing ordinary pages into
// a region that was converted to placeholders.
namespace physical {

constexpr SIZE_T kPage = kXmemPageSize64k;
constexpr ULONG_PTR kFirstPageNumber = 0x100;  // so that 0 is never a valid page

struct Frame {
    HANDLE section = nullptr;
    ULONG64 offset = 0;
    uint32_t mappings = 0;  // views of this page that are currently mapped
    bool allocated = false;
};

struct State {
    std::mutex mutex;
    std::vector<Frame> frames;                   // page number - kFirstPageNumber
    std::vector<ULONG_PTR> free_pages;           // freed pages with no mapping, reused (zeroed) first
    std::map<uintptr_t, uintptr_t> regions;      // placeholder regions: base -> end
    std::unordered_map<uintptr_t, ULONG_PTR> views;  // mapped address -> page number
};

State& state() {
    static State s;
    return s;
}

Frame* frame(State& s, ULONG_PTR page) {
    if (page < kFirstPageNumber) return nullptr;
    const ULONG_PTR index = page - kFirstPageNumber;
    if (index >= s.frames.size()) return nullptr;
    Frame& f = s.frames[index];
    return f.allocated ? &f : nullptr;
}

bool zero_frame(const Frame& f) {
    void* view = MapViewOfFile(f.section, FILE_MAP_WRITE, static_cast<DWORD>(f.offset >> 32),
                               static_cast<DWORD>(f.offset), kPage);
    if (!view) return false;
    std::memset(view, 0, kPage);
    UnmapViewOfFile(view);
    return true;
}

// The placeholder region holding [va, va + kPage), converting the ordinary reservation there on first use.
bool ensure_placeholder(State& s, uintptr_t va) {
    auto it = s.regions.upper_bound(va);
    if (it != s.regions.begin()) {
        --it;
        if (va >= it->first && va + kPage <= it->second) return true;
    }
    const VirtualAlloc2Fn alloc2 = placeholder_alloc2();
    if (!alloc2 || !map_view_of_file3() || !unmap_view_of_file2()) {
        SetLastError(ERROR_NOT_SUPPORTED);
        return false;
    }
    MEMORY_BASIC_INFORMATION info{};
    if (!VirtualQuery(reinterpret_cast<void*>(va), &info, sizeof(info)) || info.State != MEM_RESERVE) {
        // Free address space, or committed pages that a release would destroy.
        SetLastError(ERROR_INVALID_ADDRESS);
        return false;
    }
    // Walk the whole host allocation: every part must be reserve-only.
    const uintptr_t base = reinterpret_cast<uintptr_t>(info.AllocationBase);
    uintptr_t end = base;
    for (;;) {
        MEMORY_BASIC_INFORMATION part{};
        if (!VirtualQuery(reinterpret_cast<void*>(end), &part, sizeof(part)) || part.AllocationBase != info.AllocationBase) {
            break;
        }
        if (part.State != MEM_RESERVE) {
            SetLastError(ERROR_INVALID_ADDRESS);
            return false;
        }
        end = reinterpret_cast<uintptr_t>(part.BaseAddress) + part.RegionSize;
    }
    if (!VirtualFree(reinterpret_cast<void*>(base), 0, MEM_RELEASE)) return false;
    void* const p = alloc2(GetCurrentProcess(), reinterpret_cast<void*>(base), end - base,
                           MEM_RESERVE | MEM_RESERVE_PLACEHOLDER, PAGE_NOACCESS, nullptr, 0);
    if (p != reinterpret_cast<void*>(base)) {
        // Another thread took the range in the gap; nothing can be done about it here.
        std::fprintf(stderr, "[rt] xmem: could not re-reserve 0x%llx..0x%llx as a placeholder (error %lu)\n",
                     static_cast<unsigned long long>(base), static_cast<unsigned long long>(end), GetLastError());
        if (p) VirtualFree(p, 0, MEM_RELEASE);
        SetLastError(ERROR_INVALID_ADDRESS);
        return false;
    }
    s.regions[base] = end;
    return true;
}

bool unmap_page(State& s, uintptr_t va) {
    auto it = s.views.find(va);
    if (it == s.views.end()) return true;
    if (!unmap_view_of_file2()(GetCurrentProcess(), reinterpret_cast<void*>(va), MEM_PRESERVE_PLACEHOLDER)) {
        return false;
    }
    const ULONG_PTR index = it->second - kFirstPageNumber;
    if (index < s.frames.size()) {
        Frame& f = s.frames[index];
        if (f.mappings) --f.mappings;
        // A page freed while it was mapped becomes reusable when its last view goes.
        if (!f.allocated && f.mappings == 0 && f.section) s.free_pages.push_back(it->second);
    }
    s.views.erase(it);
    return true;
}

// Maps `page` at `va` (or unmaps `va` when `page` is 0). The caller holds the lock.
bool map_page(State& s, uintptr_t va, ULONG_PTR page) {
    if (va == 0 || (va & (kPage - 1)) != 0) {
        SetLastError(ERROR_INVALID_PARAMETER);
        return false;
    }
    Frame* f = nullptr;
    if (page != 0) {
        f = frame(s, page);
        if (!f) {
            SetLastError(ERROR_INVALID_PARAMETER);
            return false;
        }
    }
    if (!ensure_placeholder(s, va)) return false;
    if (!unmap_page(s, va)) return false;
    if (!f) return true;

    // Split this page off its placeholder. It fails when the placeholder is exactly this page, which is fine.
    VirtualFree(reinterpret_cast<void*>(va), kPage, MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER);
    void* const view = map_view_of_file3()(f->section, GetCurrentProcess(), reinterpret_cast<void*>(va), f->offset,
                                           kPage, MEM_REPLACE_PLACEHOLDER, PAGE_READWRITE, nullptr, 0);
    if (view != reinterpret_cast<void*>(va)) {
        if (view) UnmapViewOfFile(view);
        return false;
    }
    ++f->mappings;
    s.views[va] = page;
    return true;
}

void note_physical(const char* api, ULONG64 attributes, ULONG_PTR count, const void* caller) {
    static std::mutex mutex;
    static std::set<std::pair<std::string, ULONG64>> seen;
    {
        const std::lock_guard lock(mutex);
        if (!seen.insert({api, attributes}).second) return;
    }
    std::fprintf(stderr, "[rt] xmem: %s attributes 0x%llx, %llu pages (logged once per value)\n", api,
                 static_cast<unsigned long long>(attributes), static_cast<unsigned long long>(count));
    if (!rt::startup_trace::enabled()) return;
    rt::startup_trace::event(
        "xmem_physical",
        std::string("\"api\":\"") + api + "\",\"attributes\":" +
            rt::startup_trace::quote(rt::startup_trace::hex(attributes)) +
            ",\"count\":" + std::to_string(static_cast<uint64_t>(count)) + ",\"caller\":" +
            rt::startup_trace::quote(rt::startup_trace::where(reinterpret_cast<uint64_t>(caller))));
}

}  // namespace physical

BOOL WINAPI XMemAllocatePhysicalPages(ULONG64 attributes, ULONG_PTR* count, ULONG_PTR* pages) {
    if (!count || (*count != 0 && !pages)) {
        SetLastError(ERROR_INVALID_PARAMETER);
        return FALSE;
    }
    const ULONG_PTR wanted = *count;
    physical::note_physical("XMemAllocatePhysicalPages", attributes, wanted, _ReturnAddress());
    if (wanted == 0) return TRUE;

    physical::State& s = physical::state();
    const std::lock_guard lock(s.mutex);
    ULONG_PTR got = 0;
    // Freed pages first, zeroed as fresh ones are.
    while (got < wanted && !s.free_pages.empty()) {
        const ULONG_PTR page = s.free_pages.back();
        s.free_pages.pop_back();
        physical::Frame& f = s.frames[page - physical::kFirstPageNumber];
        if (f.allocated || f.mappings) continue;
        if (!physical::zero_frame(f)) continue;
        f.allocated = true;
        pages[got++] = page;
    }
    const ULONG_PTR rest = wanted - got;
    if (rest != 0) {
        if (rest > (std::numeric_limits<ULONG64>::max)() / physical::kPage) {
            *count = got;
            SetLastError(ERROR_NOT_ENOUGH_MEMORY);
            return got != 0;
        }
        const ULONG64 bytes = static_cast<ULONG64>(rest) * physical::kPage;
        HANDLE section = CreateFileMappingW(INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE | SEC_COMMIT,
                                            static_cast<DWORD>(bytes >> 32), static_cast<DWORD>(bytes), nullptr);
        if (!section) {
            std::fprintf(stderr, "[rt] xmem: XMemAllocatePhysicalPages: no section for %llu pages (error %lu)\n",
                         static_cast<unsigned long long>(rest), GetLastError());
            *count = got;
            SetLastError(ERROR_NOT_ENOUGH_MEMORY);
            return got != 0;
        }
        // The section stays open for the life of the process: its pages are reused after XMemFreePhysicalPages.
        for (ULONG_PTR i = 0; i < rest; ++i) {
            physical::Frame f;
            f.section = section;
            f.offset = static_cast<ULONG64>(i) * physical::kPage;
            f.allocated = true;
            s.frames.push_back(f);
            pages[got++] = physical::kFirstPageNumber + s.frames.size() - 1;
        }
    }
    *count = got;
    return TRUE;
}

BOOL WINAPI XMemFreePhysicalPages(ULONG_PTR count, const ULONG_PTR* pages) {
    if (count != 0 && !pages) {
        SetLastError(ERROR_INVALID_PARAMETER);
        return FALSE;
    }
    physical::State& s = physical::state();
    const std::lock_guard lock(s.mutex);
    BOOL ok = TRUE;
    for (ULONG_PTR i = 0; i < count; ++i) {
        physical::Frame* f = physical::frame(s, pages[i]);
        if (!f) {
            ok = FALSE;
            continue;
        }
        f->allocated = false;
        if (f->mappings == 0) s.free_pages.push_back(pages[i]);
    }
    if (!ok) SetLastError(ERROR_INVALID_PARAMETER);
    return ok;
}

PVOID WINAPI XMemMapPhysicalPages(PVOID address, ULONG_PTR count, const ULONG_PTR* pages) {
    const uintptr_t base = reinterpret_cast<uintptr_t>(address);
    if (!address || count > ((std::numeric_limits<uintptr_t>::max)() - base) / physical::kPage) {
        SetLastError(ERROR_INVALID_PARAMETER);
        return nullptr;
    }
    physical::State& s = physical::state();
    const std::lock_guard lock(s.mutex);
    for (ULONG_PTR i = 0; i < count; ++i) {
        if (!physical::map_page(s, base + i * physical::kPage, pages ? pages[i] : 0)) {
            const DWORD error = GetLastError();
            std::fprintf(stderr, "[rt] xmem: XMemMapPhysicalPages(0x%llx, %llu): page %llu failed (error %lu)\n",
                         static_cast<unsigned long long>(base), static_cast<unsigned long long>(count),
                         static_cast<unsigned long long>(i), error);
            SetLastError(error);
            return nullptr;
        }
    }
    return address;
}

BOOL WINAPI XMemMapPhysicalPagesScatter(PVOID* addresses, ULONG_PTR count, const ULONG_PTR* pages) {
    if (count != 0 && !addresses) {
        SetLastError(ERROR_INVALID_PARAMETER);
        return FALSE;
    }
    physical::State& s = physical::state();
    const std::lock_guard lock(s.mutex);
    for (ULONG_PTR i = 0; i < count; ++i) {
        if (!physical::map_page(s, reinterpret_cast<uintptr_t>(addresses[i]), pages ? pages[i] : 0)) {
            const DWORD error = GetLastError();
            std::fprintf(stderr, "[rt] xmem: XMemMapPhysicalPagesScatter: page %llu at 0x%llx failed (error %lu)\n",
                         static_cast<unsigned long long>(i),
                         static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(addresses[i])), error);
            SetLastError(error);
            return FALSE;
        }
    }
    return TRUE;
}

// kernel32!VirtualAlloc as a title calls it. The console's allocation type has two bits of its own, MEM_GRAPHICS
// (0x10000000, memory the GPU reads) and MEM_TITLE (0x40000000, the title's own heap). On a PC the host rejects both
// with ERROR_INVALID_PARAMETER (measured on Windows 11 26200: VirtualAlloc(NULL, 64 KiB, MEM_COMMIT | MEM_RESERVE |
// 0x10000000, PAGE_READWRITE) returns NULL; so does 0x40000000), so an engine allocating its heaps this way gets
// NULL at boot. Both are hints about where the memory lives; the host has no such split, so they are dropped and
// the rest is passed through, as are the console protection tag (0x04000000) and the PAGE_GRAPHICS_* half of the
// protection. A page-size request (MEM_LARGE_PAGES or MEM_4MB_PAGES, see above) is the console's, not the host's
// MEM_LARGE_PAGES, which needs SeLockMemoryPrivilege: it goes through XMemVirtualAlloc's page handling, keeping
// Win32's rule that an operation is required. MEM_PHYSICAL without MEM_LARGE_PAGES is passed through. Not done:
// whether graphics memory has to come from an arena the D3D12 layer can see (#313).
LPVOID WINAPI TitleVirtualAlloc(LPVOID address, SIZE_T size, DWORD allocation_type, DWORD protect) {
    allocation_type &= ~kConsoleAllocationTags;
    protect = host_protection(protect);
    if (allocation_type & (kXmemLargePages | kXmem4MbPages)) {
        return console_virtual_alloc(address, size, allocation_type, protect, false);
    }
    return VirtualAlloc(address, size, allocation_type, protect);
}

}  // namespace

#define TITLE_VIRTUAL_ALLOC(dll) {dll, "VirtualAlloc", reinterpret_cast<void*>(&TitleVirtualAlloc)}

// The classic libraries and the memory api sets a title's imports name.
SHIM_TABLE(title_memory,
    TITLE_VIRTUAL_ALLOC("kernel32.dll"),
    TITLE_VIRTUAL_ALLOC("kernelbase.dll"),
    TITLE_VIRTUAL_ALLOC("api-ms-win-core-memory-l1-1-0.dll"),
    TITLE_VIRTUAL_ALLOC("api-ms-win-core-memory-l1-1-1.dll"),
    TITLE_VIRTUAL_ALLOC("api-ms-win-core-memory-l1-1-2.dll"),
    TITLE_VIRTUAL_ALLOC("api-ms-win-core-memory-l1-1-3.dll"),
    TITLE_VIRTUAL_ALLOC("api-ms-win-core-memory-l1-1-4.dll"),
    TITLE_VIRTUAL_ALLOC("api-ms-win-core-memory-l1-1-5.dll"),
    TITLE_VIRTUAL_ALLOC("api-ms-win-core-memory-l1-1-6.dll"),
);

SHIM_TABLE(xmem,
    {"xmem.dll", "XMemAlloc", reinterpret_cast<void*>(&XMemAlloc)},
    {"xmem.dll", "XMemFree", reinterpret_cast<void*>(&XMemFree)},
    {"xmem.dll", "XMemVirtualQuery", reinterpret_cast<void*>(&XMemVirtualQuery)},
    {"xmem.dll", "XMemGetWorkingSetStatistics", reinterpret_cast<void*>(&XMemGetWorkingSetStatistics)},
    {"xmem.dll", "XMemVirtualAlloc", reinterpret_cast<void*>(&XMemVirtualAlloc)},
    {"xmem.dll", "XMemGetWriteWatch", reinterpret_cast<void*>(&XMemGetWriteWatch)},
    {"xmem.dll", "XMemResetWriteWatch", reinterpret_cast<void*>(&XMemResetWriteWatch)},
    {"xmem.dll", "XMemTransferMemory", reinterpret_cast<void*>(&XMemTransferMemory)},
    {"xmem.dll", "XMemAllocatePhysicalPages", reinterpret_cast<void*>(&XMemAllocatePhysicalPages)},
    {"xmem.dll", "XMemFreePhysicalPages", reinterpret_cast<void*>(&XMemFreePhysicalPages)},
    {"xmem.dll", "XMemMapPhysicalPages", reinterpret_cast<void*>(&XMemMapPhysicalPages)},
    {"xmem.dll", "XMemMapPhysicalPagesScatter", reinterpret_cast<void*>(&XMemMapPhysicalPagesScatter)},
);
