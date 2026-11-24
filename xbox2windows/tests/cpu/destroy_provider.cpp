#include <cstdint>

// Source-owned provider. The host keeps this observation alive across wrapper cleanup.
struct Observation {
    void** slot;
    const void* allocator;
    std::uint32_t calls;
    std::uint32_t status;
};

struct Allocator;

extern "C" __declspec(dllexport) std::uint32_t ffxDestroyContext(void** slot, const Allocator* allocator) {
    auto* observation = static_cast<Observation*>(*slot);
    observation->slot = slot;
    observation->allocator = allocator;
    ++observation->calls;
    *slot = nullptr;
    return observation->status;
}
