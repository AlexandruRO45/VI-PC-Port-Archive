// AcpHal.dll: the console's audio hardware layer. The Apu* functions are a heap of memory the audio processor reads,
// with 32-bit "APU addresses" for it; AcpHalCreate returns the object commands go through and the shape contexts are
// the per-voice records the hardware works on. A PC has no audio processor: the heap is ordinary memory, an APU
// address is an offset into it and contexts are zeroed memory. Hardware creation fails with E_NOTIMPL until an
// audio backend exists; reporting success and dropping commands would hide missing emulation.
//
// Measured from the game executable (one thunk per import; call-site counts in brackets), statically read:
//   HRESULT ApuCreateHeap(UINT32 a, UINT32 b)                 [1] a = MiB << 20, b = a byte count; failure is fatal
//   HRESULT ApuAlloc(void** va, UINT32* apu, UINT32 size, UINT32 align, UINT32 flags)
//                                                             [1] apu NULL, align 0x800, flags forwarded by wrapper;
//                                                                 the wrapper does not test the result
//   ApuFree(void* va)                                         [1] tail call at shutdown, result ignored
//   bool ApuIsVirtualAddressValid(void* va, UINT32 align)     [4] align 0x40, 0x80 or 0x800; result in AL
//   UINT32 ApuMapVirtualAddress(void* va)                     [4] stored into a 32-bit context field; once with NULL
//   void* ApuMapApuAddress(UINT32 apu)                        [2] result tested for NULL, then written through
//   HRESULT AcpHalAllocateShapeContexts(contexts*)            [1] failure is fatal, see kShapeTypes
//   HRESULT AcpHalCreate(IAcpHal** out)                       [1] failure is fatal; object ABI incomplete
//   AcpHalReleaseShapeContexts()                              [1] no argument register set up; result ignored
// The argument meanings (cached/non-cached sizes and flags) remain hypotheses.
#include <windows.h>

#include <malloc.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <map>
#include <mutex>
#include <new>

#include "shims.h"

namespace {

constexpr HRESULT kEFail = static_cast<HRESULT>(0x80004005L);
constexpr HRESULT kEOutOfMemory = static_cast<HRESULT>(0x8007000EL);
constexpr HRESULT kEInvalidArg = static_cast<HRESULT>(0x80070057L);
constexpr HRESULT kEPointer = static_cast<HRESULT>(0x80004003L);

// Hypothesis: any nonzero base works, the title only passes APU addresses back to this DLL and to the hardware.
// Nonzero so that address 0 stays "none" (ApuMapVirtualAddress(NULL) is 0).
constexpr uint32_t kApuBase = 0x40000000;

std::mutex g_mutex;
uint8_t* g_heap = nullptr;
uint32_t g_heap_size = 0;
std::map<uint32_t, uint32_t> g_blocks;  // offset -> size of each live ApuAlloc block

// One region for both sizes: a PC has no cached/non-cached split. The single call site's second size is the first
// ApuAlloc's size plus 16 KiB, so flags 2 is probably the non-cached heap.
HRESULT ApuCreateHeap(uint32_t cached, uint32_t non_cached) {
    const uint64_t total = uint64_t(cached) + non_cached;
    if (total == 0 || total > 0xFFFFFFFFull - kApuBase) return kEInvalidArg;
    std::lock_guard lock(g_mutex);
    if (g_heap) return kEFail;  // shortcut: one heap per process, the only call site creates it once at boot
    g_heap = static_cast<uint8_t*>(VirtualAlloc(nullptr, size_t(total), MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
    if (!g_heap) return kEOutOfMemory;
    g_heap_size = static_cast<uint32_t>(total);
    return S_OK;
}

// First fit between live blocks. Only observed/default flags are accepted; cachedness is not modeled.
HRESULT ApuAlloc(void** va, uint32_t* apu, uint32_t size, uint32_t align, uint32_t flags) {
    if (!va) return kEPointer;
    *va = nullptr;
    if (apu) *apu = 0;
    if (flags != 0 && flags != 2) return E_NOTIMPL;
    if (align == 0) align = 1;
    if (size == 0 || align > 0x10000 || (align & (align - 1)) != 0) return kEInvalidArg;
    std::lock_guard lock(g_mutex);
    if (!g_heap) return kEFail;
    uint64_t at = 0;
    for (auto it = g_blocks.begin();; ++it) {
        at = (at + align - 1) & ~uint64_t(align - 1);
        const uint64_t limit = it == g_blocks.end() ? g_heap_size : it->first;
        if (at + size <= limit) break;
        if (it == g_blocks.end()) return kEOutOfMemory;
        at = uint64_t(it->first) + it->second;
    }
    try {
        g_blocks.emplace(static_cast<uint32_t>(at), size);
    } catch (const std::bad_alloc&) {
        return kEOutOfMemory;
    }
    std::memset(g_heap + at, 0, size);
    *va = g_heap + at;
    if (apu) *apu = kApuBase + static_cast<uint32_t>(at);
    return S_OK;
}

HRESULT ApuFree(void* va) {
    std::lock_guard lock(g_mutex);
    const uintptr_t address = reinterpret_cast<uintptr_t>(va);
    const uintptr_t base = reinterpret_cast<uintptr_t>(g_heap);
    if (!g_heap || address < base || address - base >= g_heap_size) return kEInvalidArg;
    return g_blocks.erase(static_cast<uint32_t>(address - base)) ? S_OK : kEInvalidArg;
}

// TRUE when the address is inside a live block and its APU address is a multiple of `align`.
BOOL ApuIsVirtualAddressValid(const void* va, uint32_t align) {
    if (align != 0 && (align & (align - 1)) != 0) return FALSE;
    std::lock_guard lock(g_mutex);
    const uintptr_t address = reinterpret_cast<uintptr_t>(va);
    const uintptr_t base = reinterpret_cast<uintptr_t>(g_heap);
    if (!g_heap || address < base || address - base >= g_heap_size) return FALSE;
    const uint32_t offset = static_cast<uint32_t>(address - base);
    auto it = g_blocks.upper_bound(offset);
    if (it == g_blocks.begin()) return FALSE;
    --it;
    if (offset - it->first >= it->second) return FALSE;
    return align == 0 || ((kApuBase + offset) & (align - 1)) == 0;
}

uint32_t ApuMapVirtualAddress(const void* va) {
    std::lock_guard lock(g_mutex);
    const uintptr_t address = reinterpret_cast<uintptr_t>(va);
    const uintptr_t base = reinterpret_cast<uintptr_t>(g_heap);
    if (!g_heap || address < base || address - base >= g_heap_size) return 0;
    const uint32_t offset = static_cast<uint32_t>(address - base);
    auto it = g_blocks.upper_bound(offset);
    if (it == g_blocks.begin()) return 0;
    --it;
    return offset - it->first < it->second ? kApuBase + offset : 0;
}

void* ApuMapApuAddress(uint32_t apu) {
    std::lock_guard lock(g_mutex);
    if (!g_heap || apu < kApuBase || apu - kApuBase >= g_heap_size) return nullptr;
    const uint32_t offset = apu - kApuBase;
    auto it = g_blocks.upper_bound(offset);
    if (it == g_blocks.begin()) return nullptr;
    --it;
    return offset - it->first < it->second ? g_heap + offset : nullptr;
}

// The one caller zeroes 6 dwords, sets the fifth to 350, and afterwards reads a pointer at +0x38 and clears
// count * 64 bytes there; nothing else in the image touches the struct. Hypothesis: 6 counts, then 6 pointers, one
// per context type, 64 bytes per context (measured for the fifth type only; the others are zero at the call site).
constexpr int kShapeTypes = 6;
constexpr size_t kShapeContextSize = 64;
struct ShapeContexts {
    uint32_t count[kShapeTypes];
    void* contexts[kShapeTypes];
};
static_assert(offsetof(ShapeContexts, contexts) + 4 * sizeof(void*) == 0x38, "fifth pointer is read at +0x38");
void* g_shape[kShapeTypes] = {};

HRESULT AcpHalAllocateShapeContexts(ShapeContexts* s) {
    if (!s) return kEPointer;
    std::lock_guard lock(g_mutex);
    // Only the fifth context type's size is measured. Other nonzero counts need an ABI measurement.
    for (int i = 0; i < kShapeTypes; ++i)
        if (i != 4 && s->count[i]) return E_NOTIMPL;
    void* replacement = s->count[4] ? _aligned_malloc(size_t(s->count[4]) * kShapeContextSize, 64) : nullptr;
    if (s->count[4] && !replacement) return kEOutOfMemory;
    if (replacement) std::memset(replacement, 0, size_t(s->count[4]) * kShapeContextSize);
    _aligned_free(g_shape[4]);
    g_shape[4] = replacement;
    s->contexts[4] = replacement;
    return S_OK;
}

HRESULT AcpHalReleaseShapeContexts() {
    std::lock_guard lock(g_mutex);
    for (void*& p : g_shape) {
        _aligned_free(p);
        p = nullptr;
    }
    return S_OK;
}

// The object ABI is only partly measured. Do not manufacture a successful hardware object.
HRESULT AcpHalCreate(void** out) {
    if (!out) return kEPointer;
    *out = nullptr;
    return E_NOTIMPL;
}

}  // namespace

SHIM_TABLE(acphal,
    {"acphal.dll", "AcpHalAllocateShapeContexts", reinterpret_cast<void*>(&AcpHalAllocateShapeContexts)},
    {"acphal.dll", "AcpHalCreate", reinterpret_cast<void*>(&AcpHalCreate)},
    {"acphal.dll", "AcpHalReleaseShapeContexts", reinterpret_cast<void*>(&AcpHalReleaseShapeContexts)},
    {"acphal.dll", "ApuAlloc", reinterpret_cast<void*>(&ApuAlloc)},
    {"acphal.dll", "ApuCreateHeap", reinterpret_cast<void*>(&ApuCreateHeap)},
    {"acphal.dll", "ApuFree", reinterpret_cast<void*>(&ApuFree)},
    {"acphal.dll", "ApuIsVirtualAddressValid", reinterpret_cast<void*>(&ApuIsVirtualAddressValid)},
    {"acphal.dll", "ApuMapApuAddress", reinterpret_cast<void*>(&ApuMapApuAddress)},
    {"acphal.dll", "ApuMapVirtualAddress", reinterpret_cast<void*>(&ApuMapVirtualAddress)},
);
