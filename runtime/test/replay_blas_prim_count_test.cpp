// KHR BLAS triangle primitive count: an indexed geometry (UINT16 / UINT32) with index_count < 3 has 0 triangles
// and must not fall back to vertex_count / 3 while keeping indexType indexed. Pure CPU mock, no GPU or game data.
#include "../src/vulkan_replay.h"
#include <cstdint>
#include <cstdio>
#include <vector>

namespace {
struct RecordedGeometry {
    VkIndexType index_type = VK_INDEX_TYPE_NONE_KHR;
    std::uint32_t prim_count = 0;
};

std::vector<RecordedGeometry> recorded;

VKAPI_ATTR void VKAPI_CALL record_build(VkCommandBuffer, std::uint32_t info_count,
                                        const VkAccelerationStructureBuildGeometryInfoKHR* infos,
                                        const VkAccelerationStructureBuildRangeInfoKHR* const* ranges) {
    for (std::uint32_t i = 0; i < info_count; ++i) {
        for (std::uint32_t g = 0; g < infos[i].geometryCount; ++g) {
            const auto& geom = infos[i].pGeometries[g];
            recorded.push_back({geom.geometry.triangles.indexType, ranges[i][g].primitiveCount});
        }
    }
}

int failures = 0;
void expect(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++failures;
    }
}

leonida::RtTriangleGeometry make_tri(std::uint32_t vertex_count, std::uint32_t index_type,
                                     std::uint32_t index_count) {
    leonida::RtTriangleGeometry t{};
    t.vertex_va = 0x1000;
    t.vertex_count = vertex_count;
    t.vertex_stride = 12;
    t.vertex_format = static_cast<std::uint32_t>(VK_FORMAT_R32G32B32_SFLOAT);
    t.index_va = (index_type != 0 && index_count > 0) ? 0x2000 : 0;
    t.index_count = index_count;
    t.index_type = index_type;
    return t;
}
} // namespace

int main() {
    leonida::BuildAccelerationStructure build{};
    build.is_tlas = false;
    build.dst_va = 0x10000;
    build.scratch_va = 0x20000;
    build.triangles = {
        make_tri(30, 1, 0),  // UINT16 indexed, 0 indices -> 0 primitives (must not fall back to 30/3 = 10)
        make_tri(15, 2, 2),  // UINT32 indexed, 2 indices -> 0 primitives (must not fall back to 15/3 = 5)
        make_tri(30, 1, 9),  // UINT16 indexed, 9 indices -> 3 primitives
        make_tri(30, 0, 0),  // non-indexed, 30 vertices -> 10 primitives
        make_tri(30, 3, 9),  // unknown index_type maps to VK_INDEX_TYPE_NONE_KHR -> non-indexed 30/3 = 10
    };

    leonida::CommandList list;
    expect(list.build_acceleration_structure(build) && list.close(), "record BLAS build");

    leonida::ReplayFunctions vk{};
    vk.build_acceleration_structures = record_build;
    vk.resolve_acceleration_structure = [](std::uint64_t) {
        return reinterpret_cast<VkAccelerationStructureKHR>(std::uintptr_t{1});
    };

    const auto res = leonida::replay(list, VK_NULL_HANDLE, [](leonida::ResourceId) { return nullptr; }, vk);
    expect(res == leonida::ReplayResult::success, "replay succeeds");
    expect(recorded.size() == 5, "all five geometries recorded");
    if (recorded.size() == 5) {
        expect(recorded[0].index_type == VK_INDEX_TYPE_UINT16 && recorded[0].prim_count == 0,
               "UINT16 with 0 indices emits 0 primitives");
        expect(recorded[1].index_type == VK_INDEX_TYPE_UINT32 && recorded[1].prim_count == 0,
               "UINT32 with 2 indices emits 0 primitives");
        expect(recorded[2].index_type == VK_INDEX_TYPE_UINT16 && recorded[2].prim_count == 3,
               "UINT16 with 9 indices emits 3 primitives");
        expect(recorded[3].index_type == VK_INDEX_TYPE_NONE_KHR && recorded[3].prim_count == 10,
               "non-indexed with 30 vertices emits 10 primitives");
        expect(recorded[4].index_type == VK_INDEX_TYPE_NONE_KHR && recorded[4].prim_count == 10,
               "unknown index_type falls back to VK_INDEX_TYPE_NONE_KHR and vertex_count / 3");
    }

    if (failures) return 1;
    std::printf("replay_blas_prim_count_test: KHR BLAS primitive counts match indexType\n");
    return 0;
}
