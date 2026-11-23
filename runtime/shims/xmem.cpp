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
#include <mutex>
#include <set>
#include <string>
#include <utility>

#include "shims.h"
#include "startup_trace.h"

namespace {

constexpr size_t kAlign = 64;

// Defined with the block registry at the end of this file (before the first use the compiler sees).
bool free_xmem_block(void* p);
void note_xmem_virtual_block(void* p);
void* tracked_alloc(size_t size, size_t alignment);

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
    return tracked_alloc(size, kAlign);
}

void __stdcall XMemFree(void* p, uint64_t attributes) {
    note_attributes(false, attributes, 0, _ReturnAddress());
    // Only a block this shim handed out is released. A console pointer (a title's own arena, a mapped physical
    // page) is not on the host heap, and passing it to the CRT allocator is what turned into STATUS_HEAP_CORRUPTION
    // (0xC0000374) on the retail executable.
    if (!free_xmem_block(p)) {
        static std::atomic<bool> warned{false};
        if (p && !warned.exchange(true, std::memory_order_relaxed)) {
            std::fprintf(stderr, "[rt] xmem: XMemFree(%p) was not allocated by this shim; ignoring (logged once)\n", p);
        }
    }
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

// The host accepts eight page protections and nothing else in the low byte. The console's word carries extra bits of
// its own: the retail title (1.0.2.1) asks for 0x801 on a 24 GiB reservation and traps on int3 (game+0x1e1d04a) when
// VirtualAlloc rejects it with ERROR_INVALID_PARAMETER. A reservation protects nothing until it is committed, so an
// unknown word is mapped to PAGE_READWRITE and recorded once (`xmem_virtual_alloc_protect`).
bool valid_page_protection(DWORD value) {
    switch (value) {
        case PAGE_NOACCESS:
        case PAGE_READONLY:
        case PAGE_READWRITE:
        case PAGE_WRITECOPY:
        case PAGE_EXECUTE:
        case PAGE_EXECUTE_READ:
        case PAGE_EXECUTE_READWRITE:
        case PAGE_EXECUTE_WRITECOPY:
            return true;
        default:
            return false;
    }
}

// Diagnostic for the 1.0.2.1 boot fault. The title keeps its allocator interface table at game+0xaad45b0 and
// installs table[0] from table[1] on the success path of its memory-manager init. When table[1] is -1 the first
// malloc calls through it and faults reading 0xffffffffffffffff (game+0x104075). This copies the table around every
// XMemVirtualAlloc when LEONIDA_XMEM_HEAP_PROBE=1 so a run shows which call leaves the sentinel behind.
constexpr uint64_t kGuestHeapTableRva = 0xaad45b0;

// Whether `bytes` at `p` can be read. The probe walks guest pointers that may never have been initialised.
bool readable_bytes(const void* p, size_t bytes) {
    if (!p) return false;
    MEMORY_BASIC_INFORMATION info{};
    if (VirtualQuery(p, &info, sizeof(info)) != sizeof(info)) return false;
    if (info.State != MEM_COMMIT) return false;
    if (info.Protect & (PAGE_NOACCESS | PAGE_GUARD)) return false;
    const uintptr_t offset = reinterpret_cast<uintptr_t>(p) - reinterpret_cast<uintptr_t>(info.BaseAddress);
    if (offset >= info.RegionSize) return false;
    return bytes <= static_cast<size_t>(info.RegionSize - offset);
}

void probe_heap_table(const void* caller, const void* result) {
    static const bool on = [] {
        const char* value = std::getenv("LEONIDA_XMEM_HEAP_PROBE");
        return value && *value && *value != '0';
    }();
    if (!on || !rt::startup_trace::enabled()) return;
    uint64_t base = 0;
    if (!rt::shims::mapped_image_base(reinterpret_cast<uint64_t>(caller), base)) return;
    const auto* table = reinterpret_cast<const uint64_t*>(base + kGuestHeapTableRva);
    std::string fields = "\"result\":\"" + rt::startup_trace::hex(reinterpret_cast<uint64_t>(result)) + "\"";
    fields += ",\"table\":[";
    for (int i = 0; i < 4; ++i) {
        if (i) fields += ',';
        fields += "\"" + rt::startup_trace::hex(table[i]) + "\"";
    }
    fields += "],\"obj\":[";
    // The interface is installed at table[1] first and copied to table[0] when the memory manager finishes, so the
    // probe follows whichever entry is live. Its first qword must be a vtable in the image.
    const uint64_t live = table[0] ? table[0] : table[1];
    for (int i = 0; i < 2; ++i) {
        uint64_t value = 0;
        if (i == 0) {
            if (live >= 0x10000 && readable_bytes(reinterpret_cast<const void*>(live), 8)) {
                value = *reinterpret_cast<const uint64_t*>(live);
            }
        } else {
            uint64_t vtable = 0;
            if (live >= 0x10000 && readable_bytes(reinterpret_cast<const void*>(live), 8)) {
                vtable = *reinterpret_cast<const uint64_t*>(live);
            }
            if (vtable >= 0x10000 && readable_bytes(reinterpret_cast<const void*>(vtable), 8)) {
                value = *reinterpret_cast<const uint64_t*>(vtable);
            }
        }
        if (i) fields += ',';
        fields += "\"" + rt::startup_trace::hex(value) + "\"";
    }
    fields += "]";
    rt::startup_trace::event("xmem_heap_table", fields);
    // Arm once the interface exists: from here any write into its page that is not the title's own install path is the
    // clobber. The table lives at the start of a page (0xaad45b0), so the whole install lands on one watched page.
    static bool armed = false;
    if (!armed && table[1]) {
        armed = true;
        rt::startup_trace::watch_page(base + kGuestHeapTableRva);
    }
}

DWORD host_protection(DWORD protect, const void* caller) {
    const DWORD base = protect & 0xFF;
    if (valid_page_protection(base)) return base;
    if (rt::startup_trace::enabled()) {
        static std::mutex mutex;
        static std::set<DWORD> seen;
        bool first = false;
        {
            const std::lock_guard lock(mutex);
            first = seen.insert(protect).second;
        }
        if (first) {
            rt::startup_trace::event(
                "xmem_virtual_alloc_protect",
                "\"protect\":" + rt::startup_trace::hex(protect) + ",\"caller\":" +
                    rt::startup_trace::quote(
                        rt::startup_trace::where(reinterpret_cast<uint64_t>(caller))));
        }
    }
    return PAGE_READWRITE;
}

// The attribute word of XMemVirtualAlloc. A retail title (1.0.2.1) passes 4 and then 0x801 and traps on int3
// (game+0x1e1d01b) when the call returns null, so the word is recorded once per distinct value and otherwise ignored:
// it describes where the block lives on the console, and the host has one pool.
void note_virtual_alloc_attributes(ULONGLONG attributes, SIZE_T size, const void* caller) {
    if (!rt::startup_trace::enabled()) return;
    static std::mutex mutex;
    static std::set<ULONGLONG> seen;
    {
        const std::lock_guard lock(mutex);
        if (!seen.insert(attributes).second) return;
    }
    rt::startup_trace::event("xmem_virtual_alloc_attributes",
                             "\"attributes\":" + rt::startup_trace::hex(attributes) +
                                 ",\"size\":" + std::to_string(static_cast<uint64_t>(size)) + ",\"caller\":" +
                                 rt::startup_trace::quote(
                                     rt::startup_trace::where(reinterpret_cast<uint64_t>(caller))));
}

void* WINAPI XMemVirtualAlloc(void* address, SIZE_T size, DWORD allocation_type, ULONGLONG attributes, DWORD protect) {
    // Host-backed support covers ordinary and 64 KiB guest reservations. The 2 MiB guest mode needs an aligned
    // reservation strategy that this shim does not provide, so fail explicitly instead of returning a misaligned
    // region.
    if (size == 0) {
        SetLastError(ERROR_INVALID_PARAMETER);
        return nullptr;
    }
    if (attributes != 0) note_virtual_alloc_attributes(attributes, size, _ReturnAddress());

    const DWORD page_mode = allocation_type & kXmemPageModeMask;
    if (page_mode == kXmemLargePages) {
        // Console 2 MiB pages. MEM_LARGE_PAGES on the host needs SeLockMemoryPrivilege, which a normal token does not
        // carry; the retail title (1.0.2.1) asks for 0x20003000 with a size that is already a multiple of 2 MiB and
        // traps on int3 (game+0x1e1d3a7) when the call returns null. Ordinary committed pages are the fallback.
        if (rt::startup_trace::enabled()) {
            static std::mutex mutex;
            static bool noted = false;
            bool first = false;
            {
                const std::lock_guard lock(mutex);
                if (!noted) {
                    noted = true;
                    first = true;
                }
            }
            if (first) {
                rt::startup_trace::event(
                    "xmem_large_pages_dropped",
                    "\"allocation_type\":" + rt::startup_trace::hex(allocation_type) + ",\"caller\":" +
                        rt::startup_trace::quote(
                            rt::startup_trace::where(reinterpret_cast<uint64_t>(_ReturnAddress()))));
            }
        }
    } else if (page_mode != 0 && page_mode != kXmem64kPages) {
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

    void* result = VirtualAlloc(address, size, host_type, host_protection(protect, _ReturnAddress()));
    if (result) note_xmem_virtual_block(result);
    probe_heap_table(_ReturnAddress(), result);
    return result;
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

// kernel32!VirtualAlloc as a title calls it. The console's allocation type has two bits of its own, MEM_GRAPHICS
// (0x10000000, memory the GPU reads) and MEM_TITLE (0x40000000, the title's own heap). On a PC the host rejects both
// with ERROR_INVALID_PARAMETER (measured on Windows 11 26200: VirtualAlloc(NULL, 64 KiB, MEM_COMMIT | MEM_RESERVE |
// 0x10000000, PAGE_READWRITE) returns NULL; so does 0x40000000), so an engine allocating its heaps this way gets
// NULL at boot. Both are hints about where the memory lives; the host has no such split, so they are dropped and
// the rest is passed through. Not done: the console protection bits, and whether graphics memory has to come from
// an arena the D3D12 layer can see (#313).
constexpr DWORD kMemGraphics = 0x10000000;
constexpr DWORD kMemTitle = 0x40000000;

LPVOID WINAPI TitleVirtualAlloc(LPVOID address, SIZE_T size, DWORD allocation_type, DWORD protect) {
    return VirtualAlloc(address, size, allocation_type & ~(kMemGraphics | kMemTitle), protect);
}

// ---------------------------------------------------------------------------------------------------------------
// The physical-page family. HYPOTHESIS on the ABI: the call shapes below follow the public GDK declarations, which
// are (size, alignment, attributes, out-physical-addresses) for the allocate call and (address) for the free call.
// None of it is measured on this title yet; the first call of each writes its four argument registers and caller to
// the startup trace (`xmem_physical`), so a run replaces the guess with a measurement. What matters for boot either
// way: a title that asks for pages and receives 0 writes into what it believes it owns, which is the failure the
// retail executable showed (0xC0000005 at 0x38, 0xC0000374 heap corruption).
//
// Every block handed out is tracked, and only a tracked block is released, so a title that frees a console-owned
// pointer cannot take the host heap with it.
std::mutex& alloc_mutex() {
    static std::mutex value;
    return value;
}

std::set<void*>& alloc_blocks() {
    static std::set<void*> value;
    return value;
}

// Blocks that came back from XMemVirtualAlloc. They belong to VirtualFree, not to the CRT heap, and mixing the two
// is what corrupts the heap when a title frees a reservation.
std::set<void*>& virtual_blocks() {
    static std::set<void*> value;
    return value;
}

void note_xmem_virtual_block(void* p) {
    if (!p) return;
    const std::lock_guard lock(alloc_mutex());
    virtual_blocks().insert(p);
}

bool free_xmem_block(void* p) {
    if (!p) return false;
    // Diagnostic switch: the title can free a block through a path this shim does not see (the host's free, or a
    // second shim), and the double release then shows up as STATUS_HEAP_CORRUPTION in the host heap. With
    // LEONIDA_XMEM_NOFREE=1 the registries still drop the block but nothing is returned to the allocator, so a run
    // separates "bad free" from "bad write".
    static const bool no_free = [] {
        const char* value = std::getenv("LEONIDA_XMEM_NOFREE");
        return value && *value && *value != '0';
    }();
    bool is_virtual = false;
    {
        const std::lock_guard lock(alloc_mutex());
        if (virtual_blocks().erase(p) != 0) {
            is_virtual = true;
        }
        else if (alloc_blocks().erase(p) == 0) {
            return false;
        }
    }
    if (no_free) return true;
    if (is_virtual) {
        VirtualFree(p, 0, MEM_RELEASE);
    }
    else {
        _aligned_free(p);
    }
    return true;
}

size_t clamp_alignment(size_t alignment) {
    if (alignment < kAlign) return kAlign;
    if (alignment > 65536) return 65536;
    // _aligned_malloc requires a power of two.
    return (alignment & (alignment - 1)) == 0 ? alignment : kAlign;
}

void* tracked_alloc(size_t size, size_t alignment) {
    void* p = _aligned_malloc(size ? size : 1, clamp_alignment(alignment));
    if (p) {
        const std::lock_guard lock(alloc_mutex());
        alloc_blocks().insert(p);
    }
    return p;
}

// Whether zeroing `bytes` at `p` is safe. Used before writing an out-parameter a title may have passed as null or
// as a buffer in memory it has not committed yet.
bool writable_range(const void* p, size_t bytes) {
    if (!p) return false;
    MEMORY_BASIC_INFORMATION info{};
    if (VirtualQuery(p, &info, sizeof(info)) != sizeof(info)) return false;
    if (info.State != MEM_COMMIT) return false;
    const DWORD base = info.Protect & 0xFF;
    if (base == PAGE_NOACCESS || base == PAGE_READONLY || base == PAGE_EXECUTE_READ ||
        base == PAGE_EXECUTE || base == PAGE_EXECUTE_READ || base == PAGE_EXECUTE_WRITECOPY) {
        return false;
    }
    const uintptr_t offset = reinterpret_cast<uintptr_t>(p) - reinterpret_cast<uintptr_t>(info.BaseAddress);
    if (offset >= info.RegionSize) return false;
    return bytes <= static_cast<size_t>(info.RegionSize - offset);
}

void note_physical(const char* api, uint64_t a, uint64_t b, uint64_t c, uint64_t d, const void* caller) {
    if (!rt::startup_trace::enabled()) return;
    static std::mutex mutex;
    static std::set<std::string> seen;
    {
        const std::lock_guard lock(mutex);
        if (!seen.insert(api).second) return;
    }
    std::string fields = std::string("\"api\":\"") + api + "\"";
    fields += ",\"args\":[\"" + rt::startup_trace::hex(a) + "\",\"" + rt::startup_trace::hex(b) + "\",\"" +
              rt::startup_trace::hex(c) + "\",\"" + rt::startup_trace::hex(d) + "\"]";
    fields += ",\"caller\":" +
              rt::startup_trace::quote(rt::startup_trace::where(reinterpret_cast<uint64_t>(caller)));
    rt::startup_trace::event("xmem_physical", fields);
}

// PVOID XMemAllocDefault(SIZE_T size, uint64_t attributes) - the default allocator a title installs when it wants
// the console's own behaviour. Same shape as XMemAlloc, so it takes the same path.
void* __stdcall XMemAllocDefault(size_t size, uint64_t attributes) {
    note_attributes(true, attributes, size, _ReturnAddress());
    return tracked_alloc(size, kAlign);
}

void __stdcall XMemFreeDefault(void* p, uint64_t attributes) {
    note_attributes(false, attributes, 0, _ReturnAddress());
    if (!free_xmem_block(p) && p) {
        static std::atomic<bool> warned{false};
        if (!warned.exchange(true, std::memory_order_relaxed)) {
            std::fprintf(stderr, "[rt] xmem: XMemFreeDefault(%p) was not allocated by this shim; ignoring\n", p);
        }
    }
}

// PVOID XMemAllocatePhysicalPages(SIZE_T size, SIZE_T alignment, ULONG attributes, uint64_t* physicalAddresses)
// The host has no physical-address split: committed, 64-byte aligned memory is the closest honest answer. The
// optional output array is zeroed (page 0 of a fresh mapping) when the title handed us a writable buffer.
void* __stdcall XMemAllocatePhysicalPages(size_t size, size_t alignment, uint32_t attributes, void* physical_out) {
    note_physical("XMemAllocatePhysicalPages", size, alignment, attributes,
                  reinterpret_cast<uint64_t>(physical_out), _ReturnAddress());
    if (writable_range(physical_out, 8)) std::memset(physical_out, 0, 8);
    return tracked_alloc(size, alignment);
}

void __stdcall XMemFreePhysicalPages(void* p) {
    note_physical("XMemFreePhysicalPages", reinterpret_cast<uint64_t>(p), 0, 0, 0, _ReturnAddress());
    free_xmem_block(p);
}

// PVOID XMemMapPhysicalPages(PVOID address, SIZE_T size, uint64_t* addresses, SIZE_T pages, ULONG flags)
// Mapping pages that are already ours is the identity: the title owns the address, it just wants it wired up. When
// the address is null, this allocates, which is what a null `address` asks for.
void* __stdcall XMemMapPhysicalPages(void* address, size_t size, void* addresses, size_t pages, uint32_t flags) {
    note_physical("XMemMapPhysicalPages", reinterpret_cast<uint64_t>(address), size,
                  reinterpret_cast<uint64_t>(addresses), pages, _ReturnAddress());
    (void)flags;
    if (address) return address;
    return tracked_alloc(size ? size : pages * 4096, kAlign);
}

// The scatter form takes a per-range address list. The argument list is not measured; the first pointer decides
// identity-or-allocate, and the rest is logged for the trace.
void* __stdcall XMemMapPhysicalPagesScatter(void* address, uint64_t b, uint64_t c, uint64_t d, uint64_t e) {
    note_physical("XMemMapPhysicalPagesScatter", reinterpret_cast<uint64_t>(address), b, c, d, _ReturnAddress());
    (void)e;
    if (address) return address;
    return tracked_alloc(b ? static_cast<size_t>(b) : 4096, kAlign);
}

// BOOL XMemIsToolsMemAvailable() - the tools memory region is a development-kit feature. A retail title asks so it
// can skip its own diagnostics; FALSE is the correct answer on a PC.
BOOL __stdcall XMemIsToolsMemAvailable() {
    note_physical("XMemIsToolsMemAvailable", 0, 0, 0, 0, _ReturnAddress());
    return FALSE;
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
    {"xmem.dll", "XMemAllocDefault", reinterpret_cast<void*>(&XMemAllocDefault)},
    {"xmem.dll", "XMemFreeDefault", reinterpret_cast<void*>(&XMemFreeDefault)},
    {"xmem.dll", "XMemAllocatePhysicalPages", reinterpret_cast<void*>(&XMemAllocatePhysicalPages)},
    {"xmem.dll", "XMemFreePhysicalPages", reinterpret_cast<void*>(&XMemFreePhysicalPages)},
    {"xmem.dll", "XMemMapPhysicalPages", reinterpret_cast<void*>(&XMemMapPhysicalPages)},
    {"xmem.dll", "XMemMapPhysicalPagesScatter", reinterpret_cast<void*>(&XMemMapPhysicalPagesScatter)},
    {"xmem.dll", "XMemIsToolsMemAvailable", reinterpret_cast<void*>(&XMemIsToolsMemAvailable)},
);
