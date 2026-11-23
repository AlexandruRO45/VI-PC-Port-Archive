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
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>
#include <set>
#include <string>
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

// XMem allocation-type page bits select the guest page granularity and must not be forwarded as Win32 flags.
//
// Measured on the executable's static initializers (#6824, and the same site here): the call is
// XMemVirtualAlloc(address, size, allocation_type, protect) — four arguments, no attribute word. After two 64 KiB
// reservations (0x21) it commits 2 MiB-page pools with 0x20400000 and 0x20402000, the high word 0x2040 selecting the
// mode and 0x2000 being console MEM_VERIFY, which the host does not have and which is dropped. Rejecting the mode
// stops the boot at an int3 right after the call, so it is folded onto the host: the region is rounded to 2 MiB and
// the host commits on first touch.
constexpr DWORD kXmemLargePages = 0x20000000;
constexpr DWORD kXmem2mPages = 0x20400000;
constexpr DWORD kXmem64kPages = 0x20400000;
constexpr DWORD kXmemPageModeMask = 0x20400000 | 0x00200000;
constexpr DWORD kXmemMemVerify = 0x00200000;
constexpr SIZE_T kXmemPageSize2m = 0x200000;
constexpr SIZE_T kXmemPageSize64k = 0x10000;
constexpr DWORD kXmemHostAllocationTypeMask =
    MEM_COMMIT | MEM_RESERVE | MEM_RESET | MEM_TOP_DOWN | MEM_WRITE_WATCH | 0x01000000 | 0x00400000;

bool round_up_size(SIZE_T value, SIZE_T alignment, SIZE_T& rounded) {
    const SIZE_T max = (std::numeric_limits<SIZE_T>::max)();
    if (alignment == 0 || (alignment & (alignment - 1)) != 0 || value > max - (alignment - 1)) return false;
    rounded = (value + alignment - 1) & ~(alignment - 1);
    return true;
}

void* WINAPI XMemVirtualAlloc(void* address, SIZE_T size, DWORD allocation_type, DWORD attributes, DWORD protect) {
    if (std::getenv("LEONIDA_XMEM_TRACE"))
        std::fprintf(stderr, "[rt] xmem: entry addr=%p size=%zu type=0x%lx attr=0x%lx prot=0x%lx\n", address,
                     static_cast<size_t>(size), static_cast<unsigned long>(allocation_type),
                     static_cast<unsigned long>(attributes), static_cast<unsigned long>(protect));
    // Host-backed support covers ordinary, 64 KiB and 2 MiB guest modes. The 2 MiB mode is rounded rather than
    // rejected: rejecting it stops the boot (see above).
    if (size == 0) {
        SetLastError(ERROR_INVALID_PARAMETER);
        return nullptr;
    }

    // The console's page-mode and verify bits are pure console: any of them that is set selects the 2 MiB rounding.
    // (0x20000000 large pages and 0x00400000 are both part of the measured word; the host has neither.)
    const bool two_mib = (allocation_type & (0x20000000 | 0x00400000)) != 0;

    if (attributes != 0 && std::getenv("LEONIDA_XMEM_TRACE")) {
        std::fprintf(stderr, "[rt] xmem: XMemVirtualAlloc attributes=0x%lx (undecoded)\n",
                     static_cast<unsigned long>(attributes));
    }

    const DWORD unsupported = allocation_type & ~(kXmemHostAllocationTypeMask | 0x20000000 | 0x00400000 | 0x00200000 |
                                                  MEM_4MB_PAGES | 0x00010000 | 0x00003000 | 0x0000C000);
    if (unsupported != 0) {
        if (std::getenv("LEONIDA_XMEM_TRACE"))
            std::fprintf(stderr, "[rt] xmem: XMemVirtualAlloc type=0x%lx unsupported bits=0x%lx\n",
                         static_cast<unsigned long>(allocation_type), static_cast<unsigned long>(unsupported));
        SetLastError(ERROR_INVALID_PARAMETER);
        return nullptr;
    }

    // Host bits taken by name only: the console also sets low flags the host does not have (0x10000, 0x3000), so the
    // rest of the word is not forwarded.
    DWORD host_type = 0;
    if (allocation_type & MEM_RESERVE) host_type |= MEM_RESERVE;
    if (allocation_type & MEM_COMMIT) host_type |= MEM_COMMIT;
    if (allocation_type & MEM_RESET) host_type |= MEM_RESET;
    if (allocation_type & MEM_TOP_DOWN) host_type |= MEM_TOP_DOWN;
    if (allocation_type & MEM_WRITE_WATCH) host_type |= MEM_WRITE_WATCH;
    if (allocation_type & 0x01000000u) host_type |= 0x01000000u;
    if (host_type == 0) host_type = MEM_RESERVE | MEM_COMMIT;

    // PAGE_GRAPHICS_* (0x800..0x40000) is a console modifier on top of the access bits: 0x801 is graphics +
    // PAGE_NOACCESS, which after stripping alone would stay NOACCESS and fail. Take the access from the low bits and
    // map a stripped NOACCESS to read-write. Measured words: 0x4, 0x801, 0x2404, 0x22004 (#6824 and this run).
    DWORD access = protect & 0x7u;
    if (access == 0 || access == 1u) access = PAGE_READWRITE;
    if (protect & 0x20u) access |= PAGE_WRITECOMBINE;
    DWORD host_protect = access;

    SYSTEM_INFO system_info{};
    GetSystemInfo(&system_info);
    const SIZE_T page_granularity = two_mib ? kXmemPageSize2m : system_info.dwPageSize;
    const SIZE_T base_granularity =
        two_mib ? kXmemPageSize2m
                : ((host_type & MEM_RESERVE) ? 0x10000 /* allocation granularity: never below, keeps reservations cheap */
                                             : system_info.dwPageSize);
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
    } else if (two_mib && !round_up_size(size, page_granularity, size)) {
        SetLastError(ERROR_INVALID_PARAMETER);
        return nullptr;
    }

    void* host = VirtualAlloc(address, size, host_type, host_protect);
    if (!host) {
        const DWORD last = GetLastError();
        if (std::getenv("LEONIDA_XMEM_TRACE"))
            std::fprintf(stderr,
                         "[rt] xmem: VirtualAlloc(addr=%p size=%zu type=0x%lx -> host_type=0x%lx prot=0x%lx->0x%lx) "
                         "failed %lu\n",
                         address, static_cast<size_t>(size), static_cast<unsigned long>(allocation_type),
                         static_cast<unsigned long>(host_type), static_cast<unsigned long>(protect),
                         static_cast<unsigned long>(host_protect), static_cast<unsigned long>(last));
        SetLastError(last);
        std::fprintf(stderr,
                     "[rt] xmem: XMemVirtualAlloc(%p, %zu, type=0x%lx, prot=0x%lx) FAILED last=%lu\n", address,
                     static_cast<size_t>(size), static_cast<unsigned long>(allocation_type),
                     static_cast<unsigned long>(protect), static_cast<unsigned long>(GetLastError()));
        return nullptr;
    }

    // Diagnostic for the boot's pool commits (#6824): one line per call, via LEONIDA_XMEM_TRACE.
    if (std::getenv("LEONIDA_XMEM_TRACE")) {
        std::fprintf(stderr, "[rt] xmem: XMemVirtualAlloc(%p, %zu, type=0x%lx, prot=0x%lx) -> %p\n", address,
                     static_cast<size_t>(size), static_cast<unsigned long>(allocation_type),
                     static_cast<unsigned long>(protect), host);
    }
    return host;
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

// ---- The default-allocator pair and the physical-page set ----
//
// The executable imports these seven and nothing else of xmem that was not already served (gta6rt --dry-run).
//
// The physical-page calls are not free to fake: the measured call site asks for 0x2c60 pages of 64 KiB with flags
// 0x280 and the title breaks (`int3`) unless the call returns 1 *and* the full count (issue #6824). So a real page
// store answers them. A PC has already one virtual space, which is why the mapping calls only have to record what
// each page holds: pages are handed out zeroed on allocation, the count is returned in full, and a map remembers
// which page sits where so the next allocate can hand out a different one and a free stays all-or-nothing.
//
// `XMemAllocatePhysicalPages` returns the count through its out pointer; the flags word stays undecoded exactly as it
// is at the FSR call sites.
//
// ASSUMPTION: the argument shapes are the #6824 call site's (`count`, flags, out) and (address, `pages`, count,
// flags); the free form takes (`pages`, count, flags). Nothing is committed per page beyond bookkeeping - the host
// commits on first touch, which a 64 KiB granularity does not change.

// One page is 64 KiB, the granularity the measured call site uses (0x2c60 pages, #6824).
constexpr size_t kPhysicalPageSize = 0x10000;

struct PhysicalPage {
    uint32_t id;   // 0 = not allocated by this store
};

std::mutex g_page_mutex;
uint32_t g_next_page = 1;
std::unordered_map<uint32_t, uintptr_t> g_page_host;   // page id -> the backing bytes
std::set<uint32_t> g_page_live;
std::set<uint32_t> g_page_mapped;
std::vector<uint32_t> g_page_last;   // the pages the last allocation handed out

// The flags word every allocation is logged with, once per value, so a run shows what the title asks for.
void note_page_flags(uint32_t flags) {
    if (!rt::startup_trace::enabled()) return;
    static std::mutex mutex;
    static std::set<uint32_t> seen;
    {
        const std::lock_guard lock(mutex);
        if (!seen.insert(flags).second) return;
    }
    rt::startup_trace::event("xmem_page_flags",
                             "\"api\":\"XMemAllocatePhysicalPages\",\"flags\":" +
                                 rt::startup_trace::quote(rt::startup_trace::hex(flags)));
}

void* __stdcall XMemAllocDefault(size_t size, uint64_t attributes) { return XMemAlloc(size, attributes); }

void __stdcall XMemFreeDefault(void* p, uint64_t attributes) { XMemFree(p, attributes); }

// XMEM_WORKING_SET_API kind 0x20000001 is the only one measured (the FSR sites). Nothing here answers a query, so
// "tools memory" is always present: FALSE would make a title fall back to a path that does not exist on a PC.
BOOL __stdcall XMemIsToolsMemAvailable(uint32_t /*kind*/) { return TRUE; }

// BOOL XMemAllocatePhysicalPages(uint32_t count, uint32_t flags, uint32_t* out_count): hands out `count` pages and
// writes `count` back. TRUE and the full count, or FALSE with nothing handed out - the measured site checks both.
BOOL __stdcall XMemAllocatePhysicalPages(uint32_t count, uint32_t flags, uint32_t* out_count) {
    note_page_flags(flags);
    if (count == 0) {
        if (out_count) *out_count = 0;
        return TRUE;
    }
    if (!out_count) return FALSE;

    std::vector<uint32_t> ids;
    ids.reserve(count);
    {
        const std::lock_guard lock(g_page_mutex);
        for (uint32_t i = 0; i < count; ++i) {
            void* host = std::calloc(1, kPhysicalPageSize);
            if (!host) {  // all or nothing: give back what this call took
                for (uint32_t id : ids) {
                    std::free(reinterpret_cast<void*>(g_page_host[id]));
                    g_page_host.erase(id);
                    g_page_live.erase(id);
                }
                return FALSE;
            }
            const uint32_t id = g_next_page++;
            g_page_host[id] = reinterpret_cast<uintptr_t>(host);
            g_page_live.insert(id);
            ids.push_back(id);
        }
    }
    g_page_last.assign(ids.begin(), ids.end());
    *out_count = count;
    return TRUE;
}

// BOOL XMemFreePhysicalPages(const uint32_t* pages, uint32_t count, uint32_t flags): all or nothing, refused while a
// page is mapped at a virtual address.
BOOL __stdcall XMemFreePhysicalPages(const uint32_t* pages, uint32_t count, uint32_t flags) {
    note_page_flags(flags);
    if (!pages || count == 0) return FALSE;
    const std::lock_guard lock(g_page_mutex);
    for (uint32_t i = 0; i < count; ++i) {
        if (!g_page_live.count(pages[i]) || g_page_mapped.count(pages[i])) return FALSE;
    }
    for (uint32_t i = 0; i < count; ++i) {
        std::free(reinterpret_cast<void*>(g_page_host[pages[i]]));
        g_page_host.erase(pages[i]);
        g_page_live.erase(pages[i]);
    }
    return TRUE;
}

// BOOL XMemMapPhysicalPages(void* address, const uint32_t* pages, uint32_t count, uint32_t flags): records which
// page each slot shows. The store is the one the last XMemAllocatePhysicalPages filled, the way the measured call
// site consumes it.
BOOL __stdcall XMemMapPhysicalPages(void* address, const uint32_t* pages, uint32_t count, uint32_t flags) {
    note_page_flags(flags);
    if (!address || count == 0) return FALSE;
    const std::lock_guard lock(g_page_mutex);
    for (uint32_t i = 0; i < count; ++i) {
        const uint32_t id = pages ? pages[i] : (i < g_page_last.size() ? g_page_last[i] : 0);
        if (id && (!g_page_live.count(id) || g_page_mapped.count(id))) return FALSE;
    }
    for (uint32_t i = 0; i < count; ++i) {
        const uint32_t id = pages ? pages[i] : (i < g_page_last.size() ? g_page_last[i] : 0);
        if (id) g_page_mapped.insert(id);
    }
    return TRUE;
}

// BOOL XMemMapPhysicalPagesScatter(const uint32_t* const* addresses, const uint32_t* pages, uint32_t count,
// uint32_t flags): the scattered form, one address per slot.
BOOL __stdcall XMemMapPhysicalPagesScatter(const void* const* addresses, const uint32_t* pages, uint32_t count,
                                           uint32_t flags) {
    note_page_flags(flags);
    if (!addresses || count == 0) return FALSE;
    const std::lock_guard lock(g_page_mutex);
    for (uint32_t i = 0; i < count; ++i) {
        const uint32_t id = pages ? pages[i] : 0;
        if (id && (!g_page_live.count(id) || g_page_mapped.count(id))) return FALSE;
    }
    for (uint32_t i = 0; i < count; ++i) {
        const uint32_t id = pages ? pages[i] : 0;
        if (id) g_page_mapped.insert(id);
    }
    return TRUE;
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
    {"xmem.dll", "XMemIsToolsMemAvailable", reinterpret_cast<void*>(&XMemIsToolsMemAvailable)},
    {"xmem.dll", "XMemAllocatePhysicalPages", reinterpret_cast<void*>(&XMemAllocatePhysicalPages)},
    {"xmem.dll", "XMemFreePhysicalPages", reinterpret_cast<void*>(&XMemFreePhysicalPages)},
    {"xmem.dll", "XMemMapPhysicalPages", reinterpret_cast<void*>(&XMemMapPhysicalPages)},
    {"xmem.dll", "XMemMapPhysicalPagesScatter", reinterpret_cast<void*>(&XMemMapPhysicalPagesScatter)},
);
