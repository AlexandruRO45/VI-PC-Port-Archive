// Exercise the mapped executable's real xmem.dll imports during a synthetic boot.
#include <windows.h>

#include <cstdint>
#include <cstdio>

extern "C" {
__declspec(dllimport) void* WINAPI XMemVirtualAlloc(void*, SIZE_T, DWORD, uint64_t, DWORD);
__declspec(dllimport) BOOL WINAPI XMemAllocatePhysicalPages(uint64_t, uint64_t*, uint64_t*);
__declspec(dllimport) void* WINAPI XMemMapPhysicalPages(void*, uint64_t, const uint64_t*);
__declspec(dllimport) BOOL WINAPI XMemMapPhysicalPagesScatter(void* const*, uint32_t, const uint64_t*);
__declspec(dllimport) BOOL WINAPI XMemFreePhysicalPages(uint64_t, const uint64_t*);
}

namespace {

constexpr SIZE_T kPageSize = 64 * 1024;
constexpr DWORD kXmem64kPages = 0x20400000;
constexpr uint32_t kPageOneTag = 0x13579bdf;
constexpr uint32_t kPageTwoTag = 0x2468ace0;

int fail(const char* stage) {
    std::fprintf(stderr, "tiny_xmem_physical: %s failed (Win32 error %lu)\n", stage, GetLastError());
    return 1;
}

}  // namespace

int main() {
    auto* const reservation = static_cast<uint8_t*>(
        XMemVirtualAlloc(nullptr, 4 * kPageSize, MEM_RESERVE | MEM_RESERVE_PLACEHOLDER, 0, PAGE_NOACCESS));
    if (!reservation) return fail("reserve placeholder");

    uint8_t* const window = reservation + kPageSize;
    if (XMemVirtualAlloc(window, 2 * kPageSize, MEM_RESERVE | MEM_REPLACE_PLACEHOLDER | kXmem64kPages, 0,
                         PAGE_READWRITE) != window) {
        return fail("carve a physical-page window");
    }

    uint64_t pages[2]{};
    uint64_t count = 2;
    if (!XMemAllocatePhysicalPages(0, &count, pages) || count != 2 || !pages[0] || !pages[1] || pages[0] == pages[1]) {
        return fail("allocate two physical pages");
    }
    if (XMemMapPhysicalPages(window, count, pages) != window) return fail("map physical pages");

    auto* const first = reinterpret_cast<volatile uint32_t*>(window);
    auto* const second = reinterpret_cast<volatile uint32_t*>(window + kPageSize);
    *first = kPageOneTag;
    *second = kPageTwoTag;
    if (XMemMapPhysicalPages(window, count, nullptr) != window) return fail("unmap physical pages");

    void* const reversed[2] = {window + kPageSize, window};
    if (!XMemMapPhysicalPagesScatter(reversed, 2, pages)) return fail("scatter physical pages in reverse order");
    if (*first != kPageTwoTag || *second != kPageOneTag) {
        std::fprintf(stderr, "tiny_xmem_physical: remapped pages lost their data\n");
        return 1;
    }

    if (!XMemMapPhysicalPagesScatter(reversed, 2, nullptr)) return fail("unmap scattered pages");
    if (!XMemFreePhysicalPages(count, pages)) return fail("free physical pages");
    // The loader process releases the placeholder fragments on exit.
    std::puts("tiny_xmem_physical: physical-page boot sequence passed");
    return 0;
}
