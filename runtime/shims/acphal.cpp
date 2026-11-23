// AcpHal.dll: AMD Audio Co-Processor Hardware Abstraction Layer. The console's custom audio chip (ACP)
// handles mixing and DSP effects in hardware; a PC has no equivalent. The stubs succeed silently so the
// game's audio initialisation completes. No audio is produced through this path.
//
// The APU memory functions (ApuAlloc, ApuFree, ApuCreateHeap) manage audio-processor-visible memory on the
// console. On PC, ApuAlloc maps to VirtualAlloc (the "APU address space" is the same as the CPU's) and
// ApuMapVirtualAddress/ApuMapApuAddress return the input unchanged. ApuCreateHeap returns a dummy nonzero
// handle. AcpHalCreate returns a dummy handle; shape-context calls succeed without allocating.
//
// ABI hypothesis: names from the game's import table and docs/d3d12_vulkan.md row on sibling modules.
// Parameter types are not measured; each function takes up to four register arguments.
#include <windows.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>

#include "shims.h"

namespace {

constexpr size_t kApuAlign = 4096;

uint64_t __stdcall AcpHalCreate(uint64_t a, uint64_t b, uint64_t c, uint64_t d) {
    static std::atomic<bool> warned{false};
    if (!warned.exchange(true, std::memory_order_relaxed))
        std::fprintf(stderr, "[rt] acphal: AcpHalCreate: stub returning dummy handle (no ACP on PC)\n");
    return 1;
}

uint64_t __stdcall AcpHalAllocateShapeContexts(uint64_t a, uint64_t b, uint64_t c, uint64_t d) {
    static std::atomic<bool> warned{false};
    if (!warned.exchange(true, std::memory_order_relaxed))
        std::fprintf(stderr, "[rt] acphal: AcpHalAllocateShapeContexts: stub\n");
    return 0;
}

uint64_t __stdcall AcpHalReleaseShapeContexts(uint64_t a, uint64_t b, uint64_t c, uint64_t d) {
    return 0;
}

// ApuAlloc(size, alignment, ...): allocate APU-visible memory. On PC, standard page-aligned allocation.
void* __stdcall ApuAlloc(size_t size, size_t alignment, uint64_t c, uint64_t d) {
    if (alignment < kApuAlign) alignment = kApuAlign;
    void* p = VirtualAlloc(nullptr, size ? size : 1, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
    static std::atomic<bool> warned{false};
    if (!warned.exchange(true, std::memory_order_relaxed))
        std::fprintf(stderr, "[rt] acphal: ApuAlloc(%zu, %zu): mapped to VirtualAlloc -> %p\n",
                     size, alignment, p);
    return p;
}

uint64_t __stdcall ApuCreateHeap(uint64_t a, uint64_t b, uint64_t c, uint64_t d) {
    static std::atomic<bool> warned{false};
    if (!warned.exchange(true, std::memory_order_relaxed))
        std::fprintf(stderr, "[rt] acphal: ApuCreateHeap: stub returning dummy handle\n");
    return 1;
}

void __stdcall ApuFree(void* p, uint64_t b, uint64_t c, uint64_t d) {
    if (p) VirtualFree(p, 0, MEM_RELEASE);
}

BOOL __stdcall ApuIsVirtualAddressValid(void* addr, uint64_t b, uint64_t c, uint64_t d) {
    return addr != nullptr ? TRUE : FALSE;
}

// ApuMapApuAddress: given an APU address, return the CPU-visible address. On PC they are the same.
void* __stdcall ApuMapApuAddress(void* apuAddr, uint64_t /*b*/, uint64_t /*c*/, uint64_t /*d*/) {
    return apuAddr;
}

// ApuMapVirtualAddress: given a CPU address, return the APU address. On PC they are the same.
void* __stdcall ApuMapVirtualAddress(void* cpuAddr, uint64_t /*b*/, uint64_t /*c*/, uint64_t /*d*/) {
    return cpuAddr;
}

}  // namespace

SHIM_TABLE(acphal,
    {"acphal.dll", "AcpHalCreate", reinterpret_cast<void*>(&AcpHalCreate)},
    {"acphal.dll", "AcpHalAllocateShapeContexts", reinterpret_cast<void*>(&AcpHalAllocateShapeContexts)},
    {"acphal.dll", "AcpHalReleaseShapeContexts", reinterpret_cast<void*>(&AcpHalReleaseShapeContexts)},
    {"acphal.dll", "ApuAlloc", reinterpret_cast<void*>(&ApuAlloc)},
    {"acphal.dll", "ApuCreateHeap", reinterpret_cast<void*>(&ApuCreateHeap)},
    {"acphal.dll", "ApuFree", reinterpret_cast<void*>(&ApuFree)},
    {"acphal.dll", "ApuIsVirtualAddressValid", reinterpret_cast<void*>(&ApuIsVirtualAddressValid)},
    {"acphal.dll", "ApuMapApuAddress", reinterpret_cast<void*>(&ApuMapApuAddress)},
    {"acphal.dll", "ApuMapVirtualAddress", reinterpret_cast<void*>(&ApuMapVirtualAddress)},
);
