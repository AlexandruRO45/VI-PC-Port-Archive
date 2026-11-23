// The acceleration structures inside D3D12 acceleration-structure buffers, for the KHR path (LEONIDA_RT_AS=khr).
//
// D3D12 lets one RAYTRACING_ACCELERATION_STRUCTURE resource hold several structures: a build's
// DestAccelerationStructureData and an instance's AccelerationStructure only have to be 256-byte aligned and inside
// such a resource (the DXR spec's compaction note describes packing structures into one buffer). Vulkan names a
// structure by a VkAccelerationStructureKHR over a buffer range, so a structure at GetGPUVirtualAddress() + N needs
// its own GENERIC object at offset N. vkd3d-proton makes one at each build address on first use
// (vkd3d_va_map_place_acceleration_structure); this does the same.
//
// The class only keeps the books: each resource registers its range (start address, size and where it sits, for
// the caller's create function), and find() creates a structure at an aligned address inside a range on first use.
// Removing the range hands back what was made inside it, for the caller to destroy once the GPU is done with it.
// No Vulkan types are needed here, so runtime/test/acceleration_structure_ranges_test.cpp runs it without a device.
#pragma once

#include <cstdint>
#include <map>
#include <utility>
#include <vector>

namespace leonida {

template<typename Handle, typename Location>
class AccelerationStructureRanges {
public:
    // D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BYTE_ALIGNMENT; a GENERIC structure's offset must be a multiple of
    // 256 as well (VkAccelerationStructureCreateInfoKHR).
    static constexpr std::uint64_t alignment = 256;

    // A range registered again at the same start replaces the old one; what was made in the old one is handed back.
    void add(std::uint64_t va, std::uint64_t size, const Location& where, std::vector<Handle>& retired) {
        remove(va, retired);
        if (!va || !size) return;
        ranges_.emplace(va, Range{size, where, {}});
    }

    // Forgets the range that starts at `va` and appends the structures made inside it to `retired`.
    void remove(std::uint64_t va, std::vector<Handle>& retired) {
        const auto it = ranges_.find(va);
        if (it == ranges_.end()) return;
        for (const auto& made : it->second.made) retired.push_back(made.second);
        ranges_.erase(it);
    }

    // The structure at `va`: one made earlier, or `create(where, offset, room)` for an aligned address inside a
    // range, where `offset` is from the range's start and `room` the bytes from there to its end. A null Handle{}
    // when `va` is in no range, is not 256-byte aligned or `create` fails (nothing is kept then).
    template<typename Create>
    Handle find(std::uint64_t va, Create&& create) {
        if (va % alignment) return Handle{};
        auto it = ranges_.upper_bound(va);
        if (it == ranges_.begin()) return Handle{};
        --it;
        Range& range = it->second;
        const std::uint64_t offset = va - it->first;
        if (offset >= range.size) return Handle{};
        if (const auto made = range.made.find(offset); made != range.made.end()) return made->second;
        const Handle handle = create(static_cast<const Location&>(range.where), offset, range.size - offset);
        if (handle) range.made.emplace(offset, handle);
        return handle;
    }

    std::size_t ranges() const { return ranges_.size(); }

private:
    struct Range {
        std::uint64_t size;
        Location where;
        std::map<std::uint64_t, Handle> made;  // by offset from the range's start
    };
    std::map<std::uint64_t, Range> ranges_;  // by start address; the resources' ranges do not overlap
};

}  // namespace leonida
