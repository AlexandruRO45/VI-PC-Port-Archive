// Checks the registered xmem.dll shim against the measured Xbox ABI.
//
// Every function is resolved through rt::shims::find, so the test exercises the same shim table and calling
// convention the runtime installs for xmem.dll imports instead of a private copy of the implementation. The checks
// cover what the measured call sites use: allocations aligned as the attribute word asks and never to less than 64
// bytes, XMemFree, working-set statistics with the size at +8, and XMemVirtualQuery's
// MEMORY_BASIC_INFORMATION prefix and memory-kind word at +48 inside an otherwise zeroed 64-byte output.
#include <windows.h>
#include <psapi.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <initializer_list>

#include "shims.h"

namespace {

using XMemAllocFn = void*(__stdcall*)(size_t, uint64_t);
using XMemFreeFn = void(__stdcall*)(void*, uint64_t);
using XMemVirtualQueryFn = SIZE_T(__stdcall*)(HANDLE, const void*, void*);
// BOOL XMemGetWorkingSetStatistics(UINT32 kind, stats*): the FSR DLL passes kind 0x20000001 at every call site.
using XMemGetWorkingSetStatisticsFn = BOOL(__stdcall*)(uint32_t, void*);
using XMemVirtualAllocFn = void*(WINAPI*)(void*, SIZE_T, DWORD, ULONGLONG, DWORD);
using XMemGetWriteWatchFn = UINT(WINAPI*)(DWORD, void*, SIZE_T, void**, ULONG_PTR*, ULONG*);
using XMemResetWriteWatchFn = UINT(WINAPI*)(void*, SIZE_T);
constexpr uint32_t kMeasuredKind = 0x20000001;

int g_checks = 0;
int g_failures = 0;

void check(bool condition, const char* what) {
    ++g_checks;
    std::printf("%s %s\n", condition ? "PASS" : "FAIL", what);
    if (!condition) ++g_failures;
}

bool aligned_64(const void* p) { return p && (reinterpret_cast<uintptr_t>(p) & 63u) == 0; }

void test_alloc(XMemAllocFn alloc, XMemFreeFn free_mem) {
    void* const zero = alloc(0, 0);
    check(zero != nullptr, "XMemAlloc(0, 0) returns non-null");
    check(aligned_64(zero), "XMemAlloc(0, 0) returns a 64-byte aligned block");

    bool zero_readable = false;
    if (zero) {
        auto* bytes = static_cast<uint8_t*>(zero);
        bytes[0] = 0xa5;
        zero_readable = bytes[0] == 0xa5;
    }
    check(zero_readable, "XMemAlloc(0, 0) block is writable and readable");

    void* const big = alloc(4096, 0x5e50000);
    check(big != nullptr, "XMemAlloc(4096, 0x5e50000) returns non-null");
    check(aligned_64(big), "XMemAlloc(4096, 0x5e50000) returns a 64-byte aligned block");

    bool pattern_ok = big != nullptr;
    if (big) {
        auto* bytes = static_cast<uint8_t*>(big);
        for (size_t i = 0; i < 4096; ++i) bytes[i] = static_cast<uint8_t>((i * 37u + 11u) & 0xffu);
        for (size_t i = 0; i < 4096; ++i) {
            if (bytes[i] != static_cast<uint8_t>((i * 37u + 11u) & 0xffu)) {
                pattern_ok = false;
                break;
            }
        }
    }
    check(pattern_ok, "XMemAlloc(4096, 0x5e50000) block holds a write/read pattern");

    free_mem(zero, 0);
    free_mem(big, 0x5e50000);
    check(true, "XMemFree accepts both blocks (no crash)");
}

// The attribute word's alignment: log2 in bits 24-28, and a page-size code in bits 14-15 whose value 1 (64 KiB) the
// block's start follows too. 0xa8004000 is the word amd_fidelityfx_upscaler_xbox.dll passes, 0x5e50000
// Bink2Scarlett.dll's; 0x0480009e and 0x30804000 are words other layers measured on retail titles. The rest are made
// up to cover one field at a time; bits outside the two fields are set in several to show they change nothing. What
// is not measured keeps the answer from before the word was decoded: page codes 2 and 3 add nothing to the alignment
// field, and a field above 21 (2 MiB) gets the 64-byte floor instead of a block that commits gigabytes.
void test_alloc_alignment(XMemAllocFn alloc, XMemFreeFn free_mem) {
    struct Case {
        uint64_t attributes;
        size_t size;
        size_t alignment;
        const char* what;
    };
    const Case cases[] = {
        {0x0480009e, 12, 64, "XMemAlloc(12, 0x0480009e): a 16-byte request keeps the 64-byte floor"},
        {0x05e50000, 4096, 64, "XMemAlloc(4096, 0x5e50000): a 32-byte request keeps the 64-byte floor"},
        {uint64_t{7} << 24, 100, 128, "XMemAlloc(100, alignment field 7) is 128-byte aligned"},
        {uint64_t{8} << 24, 1000, 256, "XMemAlloc(1000, alignment field 8) is 256-byte aligned"},
        {(uint64_t{12} << 24) | 0x00ff3fff, 1, 4096,
         "XMemAlloc(1, alignment field 12, object type and allocator id all ones) is 4 KiB aligned"},
        {(uint64_t{12} << 24) | (uint64_t{0xf} << 29), 5000, 4096,
         "XMemAlloc(5000, alignment field 12, memory type all ones) is 4 KiB aligned"},
        {uint64_t{16} << 24, 3, 0x10000, "XMemAlloc(3, alignment field 16) is 64 KiB aligned"},
        {0xa8004000, 256, 0x10000, "XMemAlloc(256, 0xa8004000): 64 KiB pages win over the 256-byte field"},
        {0x30804000, 0x10000, 0x10000, "XMemAlloc(64 KiB, 0x30804000) is 64 KiB aligned"},
        {uint64_t{1} << 14, 100, 0x10000, "XMemAlloc(100, page field 1) is 64 KiB aligned"},
        {(uint64_t{20} << 24) | (uint64_t{1} << 14), 100, 0x100000,
         "XMemAlloc(100, alignment field 20, page field 1): the 1 MiB field wins over 64 KiB pages"},
        {uint64_t{21} << 24, 100, 0x200000, "XMemAlloc(100, alignment field 21) is 2 MiB aligned, the largest honoured"},
        {0, 100, 64, "XMemAlloc(100, alignment field 0) keeps the 64-byte floor"},
        {(uint64_t{2} << 14) | (uint64_t{8} << 24), 100, 256,
         "XMemAlloc(100, page field 2, alignment field 8) follows the alignment field (256 bytes)"},
        {(uint64_t{3} << 14) | (uint64_t{8} << 24), 100, 256,
         "XMemAlloc(100, page field 3, alignment field 8) follows the alignment field (256 bytes)"},
        {uint64_t{22} << 24, 100, 64, "XMemAlloc(100, alignment field 22) returns a 64-byte aligned block"},
        {uint64_t{31} << 24, 100, 64, "XMemAlloc(100, alignment field 31) returns a 64-byte aligned block"},
        {(uint64_t{8} << 24) | (uint64_t{1} << 33), 100, 256,
         "XMemAlloc(100, alignment field 8, bit 33 set): bits above the memory type are not alignment"},
    };
    for (const Case& c : cases) {
        void* const p = alloc(c.size, c.attributes);
        bool ok = p != nullptr && (reinterpret_cast<uintptr_t>(p) & (c.alignment - 1)) == 0;
        if (ok) {
            auto* bytes = static_cast<uint8_t*>(p);
            bytes[0] = 0x5a;
            bytes[c.size - 1] = 0xa5;
            ok = bytes[c.size - 1] == 0xa5 && (c.size == 1 || bytes[0] == 0x5a);
        }
        check(ok, c.what);
        free_mem(p, c.attributes);
    }

    // An alignment field alone does not put every block on a coarser boundary: of several live 256-byte-aligned
    // blocks at least one is off the 64 KiB grid, so the cases above test the field they name.
    void* live[8] = {};
    bool finer = false;
    for (void*& p : live) {
        p = alloc(64, uint64_t{8} << 24);
        if (p && (reinterpret_cast<uintptr_t>(p) & 0xffff) != 0) finer = true;
    }
    for (void* p : live) free_mem(p, uint64_t{8} << 24);
    check(finer, "256-byte-aligned blocks are not all 64 KiB aligned (the alignment is the one asked for)");

    // The unmeasured cases must not cost what they would if they were honoured: eight live blocks with page code 2
    // or with alignment field 31 fit in far less than one 2 MiB page each, so at least two of them share one.
    for (const uint64_t attributes : {uint64_t{2} << 14, uint64_t{31} << 24}) {
        bool shared = false;
        for (void*& p : live) p = alloc(64, attributes);
        for (void* a : live) {
            for (void* b : live) {
                const uintptr_t x = reinterpret_cast<uintptr_t>(a), y = reinterpret_cast<uintptr_t>(b);
                if (a && b && a != b && (x >> 21) == (y >> 21)) shared = true;
            }
        }
        for (void* p : live) free_mem(p, attributes);
        check(shared, attributes >> 24 ? "blocks with alignment field 31 are not spread 2 MiB or more apart"
                                       : "blocks with page field 2 are not spread 2 MiB apart");
    }

    free_mem(nullptr, uint64_t{8} << 24);
    check(true, "XMemFree(nullptr) is a no-op");
}

void test_working_set(XMemGetWorkingSetStatisticsFn get_stats) {
    alignas(64) uint8_t stats[64];
    std::memset(stats, 0xa5, sizeof(stats));
    // The pointer is the second argument: the first (0x20000001 here, as the FSR DLL passes it) must never be written
    // through, which is what the one-argument form did.
    const BOOL ok = get_stats(kMeasuredKind, stats);
    check(ok != FALSE, "XMemGetWorkingSetStatistics(kind, 64-byte out) returns TRUE");

    uint64_t working_set = 0;
    std::memcpy(&working_set, stats + 8, sizeof(working_set));
    check(ok != FALSE && working_set != 0, "XMemGetWorkingSetStatistics writes a non-zero size at +8");

    bool tail_untouched = true;
    for (size_t i = 16; i < sizeof(stats); ++i) {
        if (stats[i] != 0xa5) {
            tail_untouched = false;
            break;
        }
    }
    check(ok != FALSE && tail_untouched, "XMemGetWorkingSetStatistics leaves bytes 16..63 untouched");

    check(get_stats(kMeasuredKind, nullptr) == FALSE,
          "XMemGetWorkingSetStatistics(kind, nullptr) returns FALSE without crashing");
}

void test_virtual_query(XMemVirtualQueryFn query) {
    uint8_t local = 0;
    const void* const local_address = &local;

    alignas(64) uint8_t out[64];
    std::memset(out, 0xa5, sizeof(out));
    const SIZE_T written = query(GetCurrentProcess(), local_address, out);
    check(written != 0, "XMemVirtualQuery(current process, stack local) returns non-zero");

    MEMORY_BASIC_INFORMATION info{};
    std::memcpy(&info, out, sizeof(info));
    const uintptr_t base = reinterpret_cast<uintptr_t>(info.BaseAddress);
    const uintptr_t address = reinterpret_cast<uintptr_t>(local_address);
    const bool contains_local =
        base != 0 && info.RegionSize != 0 && address >= base && address - base < info.RegionSize;
    check(written != 0 && contains_local,
          "XMemVirtualQuery fills MEMORY_BASIC_INFORMATION with a region containing the local");

    // The word at +48 is the memory kind: 1 for committed CPU memory. Everything after it stays zero.
    uint32_t kind = 0;
    std::memcpy(&kind, out + sizeof(info), sizeof(kind));
    check(written != 0 && kind == 1, "XMemVirtualQuery writes memory kind 1 at +48 for committed memory");

    bool tail_zero = true;
    for (size_t i = sizeof(info) + sizeof(kind); i < sizeof(out); ++i) {
        if (out[i] != 0) {
            tail_zero = false;
            break;
        }
    }
    check(written != 0 && tail_zero, "XMemVirtualQuery leaves bytes 52..63 of the output zero");

    // What XboxBvhBuilder.dll requires of an input pointer at each of its four call sites, on the bytes it reads:
    // a non-zero return, bits 0-1 of the byte at +0x30, MEM_COMMIT in the State at +0x20, and (at the first site)
    // PAGE_READONLY or PAGE_READWRITE in the Protect byte at +0x24. A zero at +0x30 failed it for every pointer.
    uint32_t state = 0;
    std::memcpy(&state, out + 0x20, sizeof(state));
    const bool bvh_accepts = written != 0 && (out[0x30] & 0x3) != 0 && (state & 0x1000) != 0 && (out[0x24] & 0x6) != 0;
    check(bvh_accepts, "a readable, writable local passes the checks XboxBvhBuilder.dll makes on the output");

    // Reserved, uncommitted memory is not memory the caller can read: the kind stays zero, so the BVH builder's
    // check rejects it on that word as well as on the state.
    void* const reserved = VirtualAlloc(nullptr, 0x10000, MEM_RESERVE, PAGE_NOACCESS);
    check(reserved != nullptr, "VirtualAlloc(MEM_RESERVE) for the uncommitted case");
    if (reserved) {
        alignas(64) uint8_t reserved_out[64];
        std::memset(reserved_out, 0xa5, sizeof(reserved_out));
        const SIZE_T reserved_written = query(GetCurrentProcess(), reserved, reserved_out);
        MEMORY_BASIC_INFORMATION reserved_info{};
        std::memcpy(&reserved_info, reserved_out, sizeof(reserved_info));
        bool reserved_tail_zero = true;
        for (size_t i = sizeof(reserved_info); i < sizeof(reserved_out); ++i) {
            if (reserved_out[i] != 0) reserved_tail_zero = false;
        }
        check(reserved_written != 0 && reserved_info.State == MEM_RESERVE && reserved_tail_zero,
              "XMemVirtualQuery on reserved memory reports MEM_RESERVE and leaves the kind at +48 zero");
        VirtualFree(reserved, 0, MEM_RELEASE);
    }

    alignas(64) uint8_t bad_out[64];
    std::memset(bad_out, 0xa5, sizeof(bad_out));
    // Address 1 is queryable (it falls in the free region starting at 0), so use a non-canonical address that
    // VirtualQueryEx rejects.
    const void* const unmappable = reinterpret_cast<const void*>(~static_cast<uintptr_t>(0));
    const SIZE_T bad = query(GetCurrentProcess(), unmappable, bad_out);
    check(bad == 0, "XMemVirtualQuery on an invalid address returns 0 without crashing");
}

void test_virtual_alloc(XMemVirtualAllocFn virtual_alloc, XMemGetWriteWatchFn get_write_watch,
                       XMemResetWriteWatchFn reset_write_watch) {
    SYSTEM_INFO system_info{};
    GetSystemInfo(&system_info);
    const SIZE_T page_size = system_info.dwPageSize;
    check(page_size != 0, "GetSystemInfo reports a non-zero page size");
    if (page_size == 0) return;

    const SIZE_T region_size = page_size * 3;
    void* const base = virtual_alloc(nullptr, region_size, MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH, 0,
                                     PAGE_READWRITE);
    check(base != nullptr, "XMemVirtualAlloc creates a synthetic write-watch region");
    if (!base) return;

    void* watched[4]{};
    ULONG_PTR count = 4;
    ULONG granularity = 0;
    UINT status = get_write_watch(0, base, region_size, watched, &count, &granularity);
    check(status == NO_ERROR && count == 0, "new write-watch region starts with no dirty pages");
    check(status == NO_ERROR && granularity == page_size, "write-watch reports the host page granularity");

    auto* const bytes = static_cast<volatile uint8_t*>(base);
    bytes[0] = 0xa1;
    bytes[page_size * 2] = 0xb2;

    count = 4;
    status = get_write_watch(0, base, region_size, watched, &count, &granularity);
    bool saw_first_page = false;
    bool saw_third_page = false;
    bool all_addresses_are_page_aligned = count <= 4;
    for (ULONG_PTR i = 0; i < count && i < 4; ++i) {
        const uintptr_t address = reinterpret_cast<uintptr_t>(watched[i]);
        const uintptr_t start = reinterpret_cast<uintptr_t>(base);
        all_addresses_are_page_aligned &= address >= start && (address - start) % page_size == 0;
        saw_first_page |= address == start;
        saw_third_page |= address == start + page_size * 2;
    }
    check(status == NO_ERROR && count == 2 && saw_first_page && saw_third_page,
          "write-watch reports each written page by its relative address");
    check(all_addresses_are_page_aligned, "write-watch addresses are page aligned and within the region");

    count = 4;
    status = get_write_watch(0, base, region_size, watched, &count, &granularity);
    check(status == NO_ERROR && count == 2, "query-only write-watch leaves dirty pages recorded");

    status = reset_write_watch(base, region_size);
    check(status == NO_ERROR, "XMemResetWriteWatch clears the synthetic region");
    count = 4;
    status = get_write_watch(0, base, region_size, watched, &count, &granularity);
    check(status == NO_ERROR && count == 0, "reset write-watch query returns no dirty pages");

    bytes[page_size] = 0xc3;
    count = 4;
    status = get_write_watch(0, base, region_size, watched, &count, &granularity);
    check(status == NO_ERROR && count == 1 &&
              reinterpret_cast<uintptr_t>(watched[0]) == reinterpret_cast<uintptr_t>(base) + page_size,
          "write-watch records writes again after reset");

    check(VirtualFree(base, 0, MEM_RELEASE), "release the synthetic write-watch region");

    SetLastError(ERROR_SUCCESS);
    void* const large_page = virtual_alloc(nullptr, page_size, MEM_RESERVE | MEM_COMMIT | 0x20000000, 0,
                                           PAGE_READWRITE);
    check(large_page == nullptr && GetLastError() == ERROR_NOT_SUPPORTED,
          "unsupported large-page allocation is rejected before host allocation");

    SetLastError(ERROR_SUCCESS);
    void* const attributed = virtual_alloc(nullptr, page_size, MEM_RESERVE | MEM_COMMIT, 1, PAGE_READWRITE);
    check(attributed == nullptr && GetLastError() == ERROR_NOT_SUPPORTED,
          "nonzero XMem allocation attributes are rejected before host allocation");
}

void print_result() {
    if (g_failures == 0) {
        std::printf("xmem_test: OK (%d checks)\n", g_checks);
    } else {
        std::printf("xmem_test: FAILED (%d checks, %d failure%s)\n", g_checks, g_failures,
                    g_failures == 1 ? "" : "s");
    }
}

}  // namespace

int main() {
    std::printf("xmem_test: registered xmem.dll shim ABI\n");

    const auto alloc = reinterpret_cast<XMemAllocFn>(rt::shims::find("xmem.dll", "XMemAlloc"));
    const auto free_mem = reinterpret_cast<XMemFreeFn>(rt::shims::find("xmem.dll", "XMemFree"));
    const auto query = reinterpret_cast<XMemVirtualQueryFn>(rt::shims::find("xmem.dll", "XMemVirtualQuery"));
    const auto get_stats =
        reinterpret_cast<XMemGetWorkingSetStatisticsFn>(rt::shims::find("xmem.dll", "XMemGetWorkingSetStatistics"));
    const auto virtual_alloc =
        reinterpret_cast<XMemVirtualAllocFn>(rt::shims::find("xmem.dll", "XMemVirtualAlloc"));
    const auto get_write_watch =
        reinterpret_cast<XMemGetWriteWatchFn>(rt::shims::find("xmem.dll", "XMemGetWriteWatch"));
    const auto reset_write_watch =
        reinterpret_cast<XMemResetWriteWatchFn>(rt::shims::find("xmem.dll", "XMemResetWriteWatch"));

    check(alloc != nullptr, "registry exposes xmem.dll!XMemAlloc");
    check(free_mem != nullptr, "registry exposes xmem.dll!XMemFree");
    check(query != nullptr, "registry exposes xmem.dll!XMemVirtualQuery");
    check(get_stats != nullptr, "registry exposes xmem.dll!XMemGetWorkingSetStatistics");
    check(virtual_alloc != nullptr, "registry exposes xmem.dll!XMemVirtualAlloc");
    check(get_write_watch != nullptr, "registry exposes xmem.dll!XMemGetWriteWatch");
    check(reset_write_watch != nullptr, "registry exposes xmem.dll!XMemResetWriteWatch");

    if (!alloc || !free_mem || !query || !get_stats) {
        print_result();
        return 1;
    }

    test_alloc(alloc, free_mem);
    test_alloc_alignment(alloc, free_mem);
    test_working_set(get_stats);
    test_virtual_query(query);
    if (virtual_alloc && get_write_watch && reset_write_watch) {
        test_virtual_alloc(virtual_alloc, get_write_watch, reset_write_watch);
    }

    print_result();
    return g_failures ? 1 : 0;
}
