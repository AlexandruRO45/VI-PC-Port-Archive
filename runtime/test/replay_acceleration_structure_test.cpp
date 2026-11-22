// The opt-in KHR acceleration-structure replay preserves D3D12's TLAS instance layout. In particular,
// D3D12_ELEMENTS_LAYOUT_ARRAY_OF_POINTERS must become VkAccelerationStructureGeometryInstancesDataKHR::
// arrayOfPointers instead of treating the GPU pointer array as inline instance descriptions (D3D-425).
// CPU mocks only: no Vulkan loader, GPU, shaders or game data.
#include "../src/vulkan_replay.h"

#include <array>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <vector>

namespace {

bool called = false;
VkBool32 captured_array_of_pointers = VK_FALSE;
VkDeviceAddress captured_instances = 0;
VkDeviceAddress captured_scratch = 0;
std::uint32_t captured_instance_count = 0;
VkDeviceSize captured_update_offset = 0;
std::vector<std::uint64_t> captured_pointers;

VKAPI_ATTR void VKAPI_CALL build_acceleration_structures(
    VkCommandBuffer, std::uint32_t info_count, const VkAccelerationStructureBuildGeometryInfoKHR* infos,
    const VkAccelerationStructureBuildRangeInfoKHR* const* ranges) {
    called = true;
    if (info_count != 1 || !infos || infos[0].geometryCount != 1 || !infos[0].pGeometries || !ranges ||
        !ranges[0])
        return;
    const auto& instances = infos[0].pGeometries[0].geometry.instances;
    captured_array_of_pointers = instances.arrayOfPointers;
    captured_instances = instances.data.deviceAddress;
    captured_scratch = infos[0].scratchData.deviceAddress;
    captured_instance_count = ranges[0][0].primitiveCount;
}

VKAPI_ATTR void VKAPI_CALL update_buffer(VkCommandBuffer, VkBuffer, VkDeviceSize offset, VkDeviceSize size,
                                         const void* data) {
    captured_update_offset = offset;
    const auto* pointers = static_cast<const std::uint64_t*>(data);
    captured_pointers.assign(pointers, pointers + size / sizeof(std::uint64_t));
}

VKAPI_ATTR void VKAPI_CALL barrier(VkCommandBuffer, VkPipelineStageFlags, VkPipelineStageFlags, VkDependencyFlags,
                                   std::uint32_t, const VkMemoryBarrier*, std::uint32_t,
                                   const VkBufferMemoryBarrier*, std::uint32_t, const VkImageMemoryBarrier*) {}

int failures = 0;
void expect(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++failures;
    }
}

void run(bool pointer_array) {
    constexpr std::uint64_t heap_base = 0x100000;
    constexpr std::uint64_t instances_va = 0x2400;
    constexpr std::uint64_t scratch_va = 0x8000;
    constexpr std::uint32_t instance_count = 3;
    constexpr std::array<std::uint64_t, instance_count> instance_pointers{0x5000, 0x5100, 0x5200};
    std::array<std::uint8_t, 0x9000> heap{};
    std::memcpy(heap.data() + instances_va, instance_pointers.data(), sizeof(instance_pointers));

    leonida::BuildAccelerationStructure build{};
    build.is_tlas = true;
    build.instance_array_of_pointers = pointer_array;
    build.dst_va = 0x4000;
    build.scratch_va = scratch_va;
    build.instances_va = instances_va;
    build.instance_count = instance_count;

    leonida::CommandList list;
    expect(list.build_acceleration_structure(build) && list.close(), "record a TLAS build");

    leonida::ReplayFunctions vk{};
    vk.heap_base = heap_base;
    vk.acceleration_structure_scratch_alignment = 256;
    vk.pipeline_barrier = barrier;
    vk.update_buffer = update_buffer;
    vk.buffer_at = [&](std::uint64_t va, std::uint64_t& size, VkBuffer& buffer, VkDeviceSize& offset) {
        if (va > heap.size() || size > heap.size() - va) return false;
        buffer = reinterpret_cast<VkBuffer>(std::uintptr_t{2});
        offset = va;
        return true;
    };
    vk.host_at = [&](std::uint64_t va, std::uint64_t& size, const std::uint8_t*& host) {
        if (va > heap.size() || size > heap.size() - va) return false;
        host = heap.data() + va;
        return true;
    };
    vk.build_acceleration_structures = build_acceleration_structures;
    vk.resolve_acceleration_structure = [](std::uint64_t) {
        return reinterpret_cast<VkAccelerationStructureKHR>(std::uintptr_t{1});
    };

    called = false;
    captured_array_of_pointers = VK_FALSE;
    captured_instances = 0;
    captured_scratch = 0;
    captured_instance_count = 0;
    captured_update_offset = 0;
    captured_pointers.clear();
    const auto result = leonida::replay(
        list, VK_NULL_HANDLE, [](leonida::ResourceId) -> const leonida::ResourceBinding* { return nullptr; }, vk);
    expect(result == leonida::ReplayResult::success, "replay the TLAS build");
    expect(called, "call vkCmdBuildAccelerationStructuresKHR");
    expect(captured_array_of_pointers == (pointer_array ? VK_TRUE : VK_FALSE),
           pointer_array ? "preserve ARRAY_OF_POINTERS" : "preserve ARRAY layout");
    expect(captured_instances == heap_base + (pointer_array ? scratch_va : instances_va),
           pointer_array ? "build from the rebased pointer table" : "translate the inline instance-array VA");
    expect(captured_scratch == heap_base + scratch_va + (pointer_array ? 256 : 0),
           pointer_array ? "keep the driver scratch after the rebased table" : "keep the original scratch VA");
    expect(captured_instance_count == instance_count, "preserve the instance count");
    expect(captured_pointers.size() == (pointer_array ? instance_count : 0),
           pointer_array ? "write the rebased pointer table" : "do not stage an inline array");
    if (pointer_array) {
        expect(captured_update_offset == scratch_va, "write the pointer table at the start of scratch");
        for (std::uint32_t n = 0; n < instance_count && n < captured_pointers.size(); ++n)
            expect(captured_pointers[n] == heap_base + instance_pointers[n], "add heap_base to an instance pointer");
    }
}

} // namespace

int main() {
    run(false);
    run(true);
    if (failures) return 1;
    std::puts("replay_acceleration_structure_test: TLAS ARRAY and ARRAY_OF_POINTERS layouts preserved");
    return 0;
}
