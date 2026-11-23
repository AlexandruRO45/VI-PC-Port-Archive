// Wall time of the xmem physical-page calls on synthetic memory: one large map, a split, a sub-range unmap, a full
// remap and a reverse-order scatter. Usage: xmem_phys_bench [pages] (default 65536; ctest runs 256 as a smoke test).
// Exits 1 when a call fails or a page does not hold its data.
#include <windows.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "shims.h"

namespace {

using XMemVirtualAllocFn = void*(WINAPI*)(void*, SIZE_T, DWORD, ULONGLONG, DWORD);
using XMemAllocatePhysicalPagesFn = BOOL(WINAPI*)(uint64_t, uint64_t*, uint64_t*);
using XMemFreePhysicalPagesFn = BOOL(WINAPI*)(uint64_t, const uint64_t*);
using XMemMapPhysicalPagesFn = void*(WINAPI*)(void*, uint64_t, const uint64_t*);
using XMemMapPhysicalPagesScatterFn = BOOL(WINAPI*)(void* const*, uint32_t, const uint64_t*);
constexpr DWORD kXmem64kPages = 0x20400000;
constexpr uint64_t kPage = 0x10000;

double now_ms() {
    static LARGE_INTEGER frequency = [] {
        LARGE_INTEGER f;
        QueryPerformanceFrequency(&f);
        return f;
    }();
    LARGE_INTEGER counter;
    QueryPerformanceCounter(&counter);
    return static_cast<double>(counter.QuadPart) * 1000.0 / static_cast<double>(frequency.QuadPart);
}

bool g_ok = true;

template <typename F>
void step(const char* what, F&& call) {
    const double start = now_ms();
    const bool ok = call();
    std::printf("%-40s %10.1f ms%s\n", what, now_ms() - start, ok ? "" : "  FAILED");
    g_ok &= ok;
}

}  // namespace

int main(int argc, char** argv) {
    const uint64_t pages = argc > 1 ? std::strtoull(argv[1], nullptr, 0) : 65536;
    const auto virtual_alloc = reinterpret_cast<XMemVirtualAllocFn>(rt::shims::find("xmem.dll", "XMemVirtualAlloc"));
    const auto allocate =
        reinterpret_cast<XMemAllocatePhysicalPagesFn>(rt::shims::find("xmem.dll", "XMemAllocatePhysicalPages"));
    const auto free_pages =
        reinterpret_cast<XMemFreePhysicalPagesFn>(rt::shims::find("xmem.dll", "XMemFreePhysicalPages"));
    const auto map = reinterpret_cast<XMemMapPhysicalPagesFn>(rt::shims::find("xmem.dll", "XMemMapPhysicalPages"));
    const auto scatter =
        reinterpret_cast<XMemMapPhysicalPagesScatterFn>(rt::shims::find("xmem.dll", "XMemMapPhysicalPagesScatter"));
    if (pages < 64 || !virtual_alloc || !allocate || !free_pages || !map || !scatter) {
        std::printf("xmem_phys_bench: needs at least 64 pages and the registered xmem.dll exports\n");
        return 1;
    }
    auto* const window =
        static_cast<uint8_t*>(virtual_alloc(nullptr, pages * kPage, MEM_RESERVE | kXmem64kPages, 0x21, PAGE_READWRITE));
    std::vector<uint64_t> numbers(pages);
    uint64_t count = pages;
    if (!window || !allocate(0x280, &count, numbers.data()) || count != pages) {
        std::printf("xmem_phys_bench: could not reserve %llu pages\n", static_cast<unsigned long long>(pages));
        return 1;
    }
    std::printf("xmem_phys_bench: %llu pages (%llu MiB)\n", static_cast<unsigned long long>(pages),
                static_cast<unsigned long long>(pages * kPage >> 20));
    auto at = [&](uint64_t i) { return window + i * kPage; };
    auto holds = [&](uint64_t i) { return *reinterpret_cast<volatile uint64_t*>(at(i)) == i + 1; };

    step("map all pages in one call", [&] { return map(window, pages, numbers.data()) == window; });
    for (uint64_t i = 0; i < pages; ++i) *reinterpret_cast<volatile uint64_t*>(at(i)) = i + 1;
    const uint64_t middle = pages / 2;
    step("remap one page in the middle", [&] { return map(at(middle), 1, &numbers[middle]) == at(middle); });
    step("unmap 16 pages in the middle", [&] { return map(at(middle + 8), 16, nullptr) == at(middle + 8); });
    step("map all pages again", [&] { return map(window, pages, numbers.data()) == window; });
    bool kept = true;
    for (uint64_t i = 0; i < pages; ++i) kept &= holds(i);
    step("check every page's data", [&] { return kept; });
    step("unmap all pages", [&] { return map(window, pages, nullptr) == window; });

    const uint64_t scattered = pages < 4096 ? pages : 4096;
    std::vector<void*> addresses(scattered);
    std::vector<uint64_t> reversed(scattered);
    for (uint64_t i = 0; i < scattered; ++i) {
        addresses[i] = at(i);
        reversed[i] = numbers[scattered - 1 - i];
    }
    step("scatter 4096 pages in reverse order",
         [&] { return scatter(addresses.data(), static_cast<uint32_t>(scattered), reversed.data()) == TRUE; });
    step("check the scattered pages", [&] {
        auto value = [&](uint64_t i) { return *reinterpret_cast<volatile uint64_t*>(at(i)); };
        return value(0) == scattered && value(scattered - 1) == 1;
    });
    step("unmap the scattered pages", [&] { return map(window, scattered, nullptr) == window; });

    g_ok &= free_pages(pages, numbers.data()) == TRUE;
    VirtualFree(window, 0, MEM_RELEASE);
    std::printf("xmem_phys_bench: %s\n", g_ok ? "OK" : "FAILED");
    return g_ok ? 0 : 1;
}
