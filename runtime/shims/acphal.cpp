// AcpHal.dll: the APU's memory side (Apu* allocations and mappings) and its shape contexts.
//
// The executable imports nine functions (gta6rt --dry-run; the whole set):
//   ApuAlloc, ApuFree, ApuCreateHeap, ApuMapApuAddress, ApuMapVirtualAddress, ApuIsVirtualAddressValid,
//   AcpHalCreate, AcpHalAllocateShapeContexts, AcpHalReleaseShapeContexts
//
// On the console the APU owns a second address space the DSP reads from. The title allocates APU-side memory and maps
// its own buffers into it. A PC has no APU, so the memory is ordinary heap and the "APU address" of a block is its
// host pointer widened: the mapping calls answer with the host address, which is what a later submit would hand the
// DSP.
//
// ASSUMPTION: no argument list or return type is measured here (docs/d3d12_vulkan.md §7 lists the exports only).
// Allocations return non-null; the mapping calls return the host address; BOOL-shaped ones answer TRUE. The valid-
// address check walks the blocks this shim handed out, so a stray pointer is refused rather than accepted. Shapes
// (the DSP programs a shape context holds) are not interpreted: a context is a stable non-null token.
#include <windows.h>

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <unordered_set>

#include "shims.h"

namespace {

std::mutex g_mutex;
std::unordered_set<void*> g_blocks;
std::atomic<bool> g_warned{false};

void warn_once(const char* what) {
    if (!g_warned.exchange(true, std::memory_order_relaxed)) {
        std::fprintf(stderr,
                     "[rt] AcpHal: the APU has no host counterpart; %s and the rest are ordinary heap and host "
                     "addresses (logged once)\n",
                     what);
    }
}

// void* ApuAlloc(size_t size, flags) -> APU-side block.
void* __stdcall ApuAlloc(size_t size, uint32_t /*flags*/) {
    warn_once("ApuAlloc");
    void* p = std::malloc(size ? size : 1);
    if (!p) return nullptr;
    const std::lock_guard lock(g_mutex);
    g_blocks.insert(p);
    return p;
}

// void ApuFree(void* p, flags) or an HRESULT; both read the same to a caller that ignores the result.
HRESULT __stdcall ApuFree(void* p, uint32_t /*flags*/) {
    if (!p) return S_OK;
    {
        const std::lock_guard lock(g_mutex);
        if (g_blocks.erase(p) == 0) return S_OK;  // not ours, or freed twice
    }
    std::free(p);
    return S_OK;
}

// void* ApuCreateHeap(size_t size, flags) -> a heap the APU allocations come from. One host heap stands in.
void* __stdcall ApuCreateHeap(size_t size, uint32_t /*flags*/) {
    warn_once("ApuCreateHeap");
    void* p = std::malloc(size ? size : 1);
    return p;
}

// ApuMapApuAddress(apu_address) / ApuMapVirtualAddress(virtual_address, flags): both yield the address the DSP
// reads. With one address space that is the host pointer itself.
void* __stdcall ApuMapApuAddress(void* apu_address) { return apu_address; }

void* __stdcall ApuMapVirtualAddress(void* virtual_address, uint32_t /*flags*/) { return virtual_address; }

BOOL __stdcall ApuIsVirtualAddressValid(const void* address) {
    if (!address) return FALSE;
    const std::lock_guard lock(g_mutex);
    return g_blocks.count(const_cast<void*>(address)) != 0 ? TRUE : FALSE;
}

// AcpHalCreate(params, out) / AcpHalAllocateShapeContexts(hal, count, out) / AcpHalReleaseShapeContexts(...): the
// HAL object and the shape contexts are tokens; only null-failure matters to the caller.
HRESULT __stdcall AcpHalCreate(const void* /*params*/, void** out) {
    warn_once("AcpHalCreate");
    static char hal;
    if (out) *out = &hal;
    return S_OK;
}

HRESULT __stdcall AcpHalAllocateShapeContexts(void* /*hal*/, uint32_t /*count*/, void** out) {
    static char context;
    if (out) *out = &context;
    return S_OK;
}

HRESULT __stdcall AcpHalReleaseShapeContexts(void* /*hal*/, void* /*contexts*/) { return S_OK; }

}  // namespace

SHIM_TABLE(AcpHal,
    {"AcpHal.dll", "ApuAlloc", reinterpret_cast<void*>(&ApuAlloc)},
    {"AcpHal.dll", "ApuFree", reinterpret_cast<void*>(&ApuFree)},
    {"AcpHal.dll", "ApuCreateHeap", reinterpret_cast<void*>(&ApuCreateHeap)},
    {"AcpHal.dll", "ApuMapApuAddress", reinterpret_cast<void*>(&ApuMapApuAddress)},
    {"AcpHal.dll", "ApuMapVirtualAddress", reinterpret_cast<void*>(&ApuMapVirtualAddress)},
    {"AcpHal.dll", "ApuIsVirtualAddressValid", reinterpret_cast<void*>(&ApuIsVirtualAddressValid)},
    {"AcpHal.dll", "AcpHalCreate", reinterpret_cast<void*>(&AcpHalCreate)},
    {"AcpHal.dll", "AcpHalAllocateShapeContexts", reinterpret_cast<void*>(&AcpHalAllocateShapeContexts)},
    {"AcpHal.dll", "AcpHalReleaseShapeContexts", reinterpret_cast<void*>(&AcpHalReleaseShapeContexts)},
);
