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
//
// The remaining exports (the default routines and their hooks, XMemMakeTag, XMemSetOption and the heap queries) follow
// the public GDK XMem reference. The physical-page family has no public description and is left to the loader's
// unimplemented-import stub, so a run reports it if the title calls it.
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

void* __stdcall XMemAllocDefault(size_t size, uint64_t /*attributes*/) {
    return _aligned_malloc(size ? size : 1, kAlign);
}

void __stdcall XMemFreeDefault(void* p, uint64_t /*attributes*/) { _aligned_free(p); }

// XMemSetAllocationHooks replaces the routines XMemAlloc and XMemFree call. Titles install them once at startup, before
// allocating, so the pair is two atomics rather than a lock on every allocation. The free hook is published first: a
// block that comes from a new alloc hook always meets its matching free hook.
using AllocRoutine = void*(__stdcall*)(size_t, uint64_t);
using FreeRoutine = void(__stdcall*)(void*, uint64_t);
std::atomic<AllocRoutine> g_alloc_hook{nullptr};
std::atomic<FreeRoutine> g_free_hook{nullptr};

void* __stdcall XMemAlloc(size_t size, uint64_t attributes) {
    note_attributes(true, attributes, size, _ReturnAddress());
    const AllocRoutine hook = g_alloc_hook.load(std::memory_order_acquire);
    return hook ? hook(size, attributes) : XMemAllocDefault(size, attributes);
}

void __stdcall XMemFree(void* p, uint64_t attributes) {
    note_attributes(false, attributes, 0, _ReturnAddress());
    const FreeRoutine hook = g_free_hook.load(std::memory_order_acquire);
    if (hook)
        hook(p, attributes);
    else
        XMemFreeDefault(p, attributes);
}

// Both routines or neither: the GDK documents hooking only one of them as an error, and a lone hook would hand out
// blocks the other side cannot release. Two nulls restore the defaults.
void __stdcall XMemSetAllocationHooks(AllocRoutine alloc, FreeRoutine free_routine) {
    if ((alloc == nullptr) != (free_routine == nullptr)) {
        std::fprintf(stderr, "[rt] xmem: XMemSetAllocationHooks with only one routine ignored\n");
        return;
    }
    g_free_hook.store(free_routine, std::memory_order_release);
    g_alloc_hook.store(alloc, std::memory_order_release);
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

// XMemMakeTag packs a four-character code into the top 24 bits of an XMem flags word, six bits per character from the
// documented alphabet (a-z, A-Z, 0-9 and '$'). ASSUMPTION: the console's order inside that alphabet is not public. The
// tag only travels back into XMem calls, which ignore it here, so any stable packing is enough.
uint64_t tag_code(uint32_t c) {
    if (c >= 'A' && c <= 'Z') return c - 'A';
    if (c >= 'a' && c <= 'z') return 26 + (c - 'a');
    if (c >= '0' && c <= '9') return 52 + (c - '0');
    return c == '$' ? 62 : 63;
}

ULONGLONG __stdcall XMemMakeTag(uint32_t tag) {
    uint64_t code = 0;
    for (int i = 0; i < 4; ++i) code = (code << 6) | tag_code((tag >> (8 * i)) & 0xff);
    return code << 40;
}

// The XMEM_OPTION settings all configure the console's heap manager (tag tree, page-size demotion, use-after-free
// traps, file-mapping rules). There is no such manager behind XMemAllocDefault, so the call is recorded and ignored.
// The argument is not read: the structure's layout is not public.
void __stdcall XMemSetOption(uint64_t option) {
    static std::atomic<bool> warned{false};
    if (!warned.exchange(true, std::memory_order_relaxed))
        std::fprintf(stderr, "[rt] xmem: XMemSetOption(0x%llx) ignored: no console heap manager on the host\n",
                     static_cast<unsigned long long>(option));
}

// The hooks fire when the console's memory manager services HeapAlloc, HeapReAlloc and HeapFree. Those calls go to the
// host heap here, so the registration succeeds and the hooks never run.
void __stdcall XMemSetWin32HeapTrackingHooks(void* alloc, void* realloc, void* free_routine) {
    static std::atomic<bool> warned{false};
    if (!warned.exchange(true, std::memory_order_relaxed))
        std::fprintf(stderr, "[rt] xmem: XMemSetWin32HeapTrackingHooks(%p, %p, %p) accepted; the host heap never calls "
                             "them\n", alloc, realloc, free_routine);
}

// The tools partition exists only on a development kit in profiling mode.
BOOL __stdcall XMemIsToolsMemAvailable() { return FALSE; }

// NULL is the documented answer when the title partition has no heap. Returning the host process heap instead would
// let a title free title-heap blocks into a heap it does not own.
HANDLE __stdcall XMemGetTitleHeap() { return nullptr; }

// Grows a console heap to its hysteresis size ahead of time. The defaults commit on demand: nothing to grow.
BOOL __stdcall XMemPreallocateFreeSpace(DWORD /*memory_type*/, BOOL /*large_heap*/) { return TRUE; }

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
    {"xmem.dll", "XMemAllocDefault", reinterpret_cast<void*>(&XMemAllocDefault)},
    {"xmem.dll", "XMemFreeDefault", reinterpret_cast<void*>(&XMemFreeDefault)},
    {"xmem.dll", "XMemSetAllocationHooks", reinterpret_cast<void*>(&XMemSetAllocationHooks)},
    {"xmem.dll", "XMemMakeTag", reinterpret_cast<void*>(&XMemMakeTag)},
    {"xmem.dll", "XMemSetOption", reinterpret_cast<void*>(&XMemSetOption)},
    {"xmem.dll", "XMemSetWin32HeapTrackingHooks", reinterpret_cast<void*>(&XMemSetWin32HeapTrackingHooks)},
    {"xmem.dll", "XMemIsToolsMemAvailable", reinterpret_cast<void*>(&XMemIsToolsMemAvailable)},
    {"xmem.dll", "XMemGetTitleHeap", reinterpret_cast<void*>(&XMemGetTitleHeap)},
    {"xmem.dll", "XMemPreallocateFreeSpace", reinterpret_cast<void*>(&XMemPreallocateFreeSpace)},
    {"xmem.dll", "XMemTransferMemory", reinterpret_cast<void*>(&XMemTransferMemory)},
);
