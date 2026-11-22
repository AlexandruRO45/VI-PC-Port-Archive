#include "era_heap.h"

#include <cstddef>
#include <cstdint>

using Word = std::uint64_t;
using Allocate = void* (*)(Word, Word);
using Release = void (*)(void*, Word);

namespace {
// This desktop binding accepts the ordinary Bink fallback request observed in
// this experiment. Other Scarlett heap policies need their own bindings.
constexpr Word attributes = 0x05e50000;
thread_local Allocate observe_allocate = nullptr;
thread_local Release observe_release = nullptr;
}

// The allocation policy comes from the shared ERA heap implementation, so the Bink shim and the
// heap experiments decode the attribute word and align, zero and release with the same code.
extern "C" __declspec(dllexport) void* windows_allocate(Word bytes, Word flags)
{
    // A nonempty payload in this profile always needs 64 bytes of headroom.
    if (bytes <= 64 || flags != attributes) return nullptr;
    return era_heap::allocate(static_cast<std::size_t>(bytes), flags);
}

extern "C" __declspec(dllexport) bool windows_release(void* base, Word flags)
{
    if (!base || flags != attributes) return false;
    era_heap::release_with_attributes(base, flags);
    return true;
}

extern "C" __declspec(dllexport) void bind_heap_observers(Allocate alloc, Release release)
{
    observe_allocate = alloc;
    observe_release = release;
}

extern "C" __declspec(dllexport) void* XMemAlloc(Word bytes, Word flags)
{
    return observe_allocate ? observe_allocate(bytes, flags) : windows_allocate(bytes, flags);
}

extern "C" __declspec(dllexport) void XMemFree(void* base, Word flags)
{
    if (observe_release) observe_release(base, flags);
    else windows_release(base, flags);
}
