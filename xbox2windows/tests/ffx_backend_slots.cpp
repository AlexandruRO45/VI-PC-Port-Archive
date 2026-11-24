#include "ffx_device_init.h"
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <stdexcept>

namespace {
// The first 15 words are the effect, four descriptor cursors, four (base,
// size) pairs and a half-open range. Bit zero of the following byte is in-use.
struct Slot {
    uint32_t effect, cursors[4], ranges[8], begin, end;
    uint8_t flags, tail[35];
};
static_assert(sizeof(Slot) == 96 && offsetof(Slot, flags) == 0x3c);

template<typename T> T read(const uint8_t* p)
{
    T value;
    std::memcpy(&value, p, sizeof value);
    return value;
}
template<typename T> void put(uint8_t* p, T value) { std::memcpy(p, &value, sizeof value); }

// Independent native Windows control for the bounded callback.
uint32_t native_slots(uint8_t* scratch, FfxSlotOutput& provider, const uint32_t* sizes)
{
    const auto count = read<uint32_t>(scratch);
    const auto capacity = read<uint32_t>(scratch + 4);
    if (count == 0) put(scratch + 0x50, capacity * 0x19000u);
    put(scratch, count + 1);
    auto* table = read<uint8_t*>(scratch + 0x1078);
    for (uint32_t i = 0; i < capacity; ++i) {
        auto* address = table + sizeof(Slot) * i;
        Slot slot = read<Slot>(address);
        if (slot.flags & 1) continue;
        provider.index = i;
        slot.effect = 0x14;
        slot.flags = (slot.flags | 1) & 0xfd;
        slot.cursors[0] = slot.cursors[2] = (i << 9) | 1;
        slot.cursors[1] = slot.cursors[3] = (i << 9) | 511;
        slot.begin = slot.end = 0;
        if (sizes) {
            slot.begin = read<uint32_t>(scratch + 0x50);
            const uint32_t length = sizes[0] + sizes[1] + sizes[2] + sizes[3];
            for (uint32_t j = 0; j < capacity; ++j) {
                if (j == i) continue;
                const auto other = read<Slot>(table + j * sizeof(Slot));
                if ((other.flags & 1) && slot.begin < other.end && slot.begin + length > other.begin) {
                    slot.begin = other.end;
                    j = 0; // The original loop increments after assigning zero.
                }
            }
            slot.end = slot.begin + length;
        }
        uint32_t base = slot.begin;
        for (unsigned n = 0; n < 4; ++n) {
            slot.ranges[2*n] = base;
            slot.ranges[2*n+1] = sizes ? sizes[n] : 0;
            base += slot.ranges[2*n+1];
        }
        put(address, slot);
        break;
    }
    return 0;
}

constexpr uint32_t sentinel = 0xfeedface;
constexpr uint32_t provider_sizes[] = {0, 0, 0, 1};

void compare_case(const char* name, std::vector<uint8_t>& scratch, const uint32_t* sizes,
                  uint32_t want_index, uint32_t want_begin = 0, uint32_t want_end = 0)
{
    const auto count = read<uint32_t>(scratch.data());
    FfxSlotOutput output{{}, sentinel, 0xdeadbeef};
    output.prefix.fill(0xa5);
    auto expected_output = output;
    expected_output.index = want_index;
    const auto before = scratch;
#ifdef FFX_PACKAGE_DEVICE_SLICE
    auto reference_output = output;
    const auto native_rc = native_slots(scratch.data(), reference_output, sizes);
    const auto reference = scratch;
    std::copy(before.begin(), before.end(), scratch.begin()); // Preserve embedded pointers.
#endif
    const auto rc = run_backend_slots(scratch.data(), output, sizes);
#ifdef FFX_PACKAGE_DEVICE_SLICE
    if (rc != native_rc || std::memcmp(&output, &reference_output, sizeof output) || scratch != reference)
        throw std::runtime_error("Translated callback differs from native control");
#endif
    if (rc != 0 || std::memcmp(&output, &expected_output, sizeof output) ||
        read<uint32_t>(scratch.data()) != count + 1)
        throw std::runtime_error(name);

    auto* table = read<uint8_t*>(scratch.data() + 0x1078);
    const size_t offset = reinterpret_cast<uintptr_t>(table) - reinterpret_cast<uintptr_t>(scratch.data());
    const auto capacity = read<uint32_t>(scratch.data() + 4);
    if (offset > scratch.size() || capacity > (scratch.size() - offset) / sizeof(Slot))
        throw std::runtime_error("Slot table outside scratch");
    for (uint32_t i = 0; i < capacity; ++i) {
        const auto* p = table + i * sizeof(Slot);
        const auto* old = before.data() + offset + i * sizeof(Slot);
        if (i != want_index) {
            if (std::memcmp(p, old, sizeof(Slot))) throw std::runtime_error("Unselected slot changed");
            continue;
        }
        const Slot slot = read<Slot>(p);
        if (slot.effect != 0x14 || slot.flags != ((old[0x3c] | 1) & 0xfd) ||
            slot.begin != want_begin || slot.end != want_end ||
            slot.cursors[0] != 512*i + 1 || slot.cursors[1] != 512*i + 511 ||
            slot.cursors[2] != slot.cursors[0] || slot.cursors[3] != slot.cursors[1])
            throw std::runtime_error("Selected slot fields differ");
        uint32_t base = want_begin;
        for (unsigned n = 0; n < 4; ++n) {
            const uint32_t length = sizes ? sizes[n] : 0;
            if (slot.ranges[2*n] != base || slot.ranges[2*n+1] != length)
                throw std::runtime_error("Selected slot subrange differs");
            base += length;
        }
        if (std::memcmp(slot.tail, old + offsetof(Slot, tail), sizeof slot.tail))
            throw std::runtime_error("Selected slot tail changed");
    }
    if (want_index == sentinel && !std::equal(before.begin()+4, before.end(), scratch.begin()+4))
        throw std::runtime_error("Exhaustion changed scratch outside the counter");
    std::printf("slot_case=%s rc=%u index=0x%x count=%u->%u\n", name, rc, output.index, count, count+1);
}
}

#ifndef FFX_PACKAGE_DEVICE_SLICE
uint32_t run_backend_slots(uint8_t* scratch, FfxSlotOutput& provider, const uint32_t* sizes)
{
    return native_slots(scratch, provider, sizes);
}
#endif

void check_backend_slots(FfxDeviceProbe& probe)
{
    auto& scratch = probe.scratch;
    if (read<uint8_t*>(scratch.data() + 0x1078) != scratch.data() + scratch.size() - sizeof(Slot))
        throw std::runtime_error("Unexpected single-context slot table after heap creation");
    compare_case("first", scratch, provider_sizes, 0, 0x19000, 0x19001);
    compare_case("exhausted", scratch, provider_sizes, sentinel);

    // Controlled capacity-two input to exercise scanning past a live record.
    std::vector<uint8_t> multi(0x1100 + 2 * sizeof(Slot), 0);
    uint8_t* table = multi.data() + 0x1100;
    put(multi.data(), 1u);
    put(multi.data()+4, 2u);
    put(multi.data()+0x50, 0x32000u);
    put(multi.data()+0x1078, table);
    Slot occupied{};
    occupied.flags = 1;
    occupied.begin = 0x32000;
    occupied.end = 0x32008;
    put(table, occupied);
    std::fill_n(table + sizeof(Slot), sizeof(Slot), uint8_t{0xa6});
    const uint32_t sizes[] = {1, 2, 3, 4};
    compare_case("second", multi, sizes, 1, 0x32008, 0x32012);
    compare_case("both-occupied", multi, sizes, sentinel);
}
