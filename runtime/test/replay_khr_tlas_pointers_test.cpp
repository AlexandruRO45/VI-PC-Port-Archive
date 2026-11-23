// Regression tests for prepare()/emit() continuing past a KHR ARRAY_OF_POINTERS TLAS build:
// commands after it in the same list must be prepared and emitted, and post-loop query checks
// and segment planning must still run. No Vulkan loader, GPU or game data.
#include "../src/vulkan_replay.h"

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace {

using namespace leonida;

template<typename T>
T available() {
    return reinterpret_cast<T>(+[]() {});
}

template<typename T>
T handle(std::uintptr_t n) {
    return reinterpret_cast<T>(n);
}

int failures = 0;
void expect(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++failures;
    }
}

std::vector<std::string> emitted;
bool saw_array_of_pointers = false;
VkDeviceAddress saw_instances_va = 0;

VKAPI_ATTR void VKAPI_CALL mock_build_as(VkCommandBuffer, std::uint32_t info_count,
                                         const VkAccelerationStructureBuildGeometryInfoKHR* infos,
                                         const VkAccelerationStructureBuildRangeInfoKHR* const*) {
    emitted.push_back("build_as");
    if (info_count && infos && infos[0].geometryCount && infos[0].pGeometries &&
        infos[0].pGeometries[0].geometryType == VK_GEOMETRY_TYPE_INSTANCES_KHR) {
        saw_array_of_pointers = infos[0].pGeometries[0].geometry.instances.arrayOfPointers == VK_TRUE;
        saw_instances_va = infos[0].pGeometries[0].geometry.instances.data.deviceAddress;
    }
}

VKAPI_ATTR void VKAPI_CALL mock_copy_buffer(VkCommandBuffer, VkBuffer, VkBuffer, std::uint32_t,
                                            const VkBufferCopy*) {
    emitted.push_back("copy_buffer");
}

}  // namespace

int main() {
    ResourceBinding src_buf{};
    src_buf.buffer = handle<VkBuffer>(1);
    src_buf.size = 4096;

    ResourceBinding rb_buf{};
    rb_buf.buffer = handle<VkBuffer>(2);
    rb_buf.size = 4096;
    rb_buf.host_readable = true;

    const auto resolve = [&](ResourceId id) -> const ResourceBinding* {
        switch (id) {
        case 1: return &src_buf;
        case 2: return &rb_buf;
        default: return nullptr;
        }
    };

    QueryPool occ_pool{};
    occ_pool.pool = handle<VkQueryPool>(0x300);
    occ_pool.heap_type = d3d12_query_heap::occlusion;
    occ_pool.count = 8;

    ReplayResult khr_status = ReplayResult::success;
    bool allow_sum_occlusion = true;
    bool allow_resolve_as = true;

    ReplayFunctions vk{};
    vk.pipeline_barrier = available<PFN_vkCmdPipelineBarrier>();
    vk.copy_buffer = mock_copy_buffer;
    vk.copy_buffer_to_image = available<PFN_vkCmdCopyBufferToImage>();
    vk.copy_image_to_buffer = available<PFN_vkCmdCopyImageToBuffer>();
    vk.copy_image = available<PFN_vkCmdCopyImage>();
    vk.fill_buffer = available<PFN_vkCmdFillBuffer>();
    vk.build_acceleration_structures = mock_build_as;
    vk.resolve_acceleration_structure = [&](std::uint64_t va) -> VkAccelerationStructureKHR {
        return allow_resolve_as && va == 0x1000 ? handle<VkAccelerationStructureKHR>(0x100) : VK_NULL_HANDLE;
    };
    vk.host_at = [](std::uint64_t, std::uint64_t, const std::uint8_t*&) { return HostReadStatus::ok; };
    vk.buffer_at = [&](std::uint64_t, std::uint64_t&, VkBuffer& out_buf, VkDeviceSize& out_off) {
        out_buf = src_buf.buffer;
        out_off = 0;
        return true;
    };
    vk.prepare_khr_instances = [&](const BuildAccelerationStructure&, VkDeviceAddress& out_va) {
        out_va = 0xdeadbeef00ULL;
        return khr_status;
    };
    vk.reset_query_pool = available<PFN_vkCmdResetQueryPool>();
    vk.begin_query = available<PFN_vkCmdBeginQuery>();
    vk.end_query = available<PFN_vkCmdEndQuery>();
    vk.write_timestamp = available<PFN_vkCmdWriteTimestamp>();
    vk.copy_query_pool_results = available<PFN_vkCmdCopyQueryPoolResults>();
    vk.query_pool = [&](QueryHeapId id) -> QueryPool* { return id == 77 ? &occ_pool : nullptr; };
    vk.sum_occlusion = [&](VkCommandBuffer, VkQueryPool, std::uint32_t, std::uint32_t, VkBuffer, VkDeviceSize) {};

    const BuildAccelerationStructure k_tlas_ptr{
        true,   // is_tlas
        false,  // update
        0,      // build_flags
        0x1000, // dst_va
        0,      // src_va
        0x2000, // scratch_va
        0x3000, // instances_va
        1,      // instance_count
        true,   // instance_pointers
        {}      // triangles
    };

    PreparedList out;

    // 1. A KHR ARRAY_OF_POINTERS TLAS build followed by a readback CopyBuffer must prepare and emit both commands.
    {
        CommandList list;
        expect(list.build_acceleration_structure(k_tlas_ptr), "record TLAS pointer build");
        expect(list.copy_buffer({2, 0, 1, 0, 64}) && list.close(), "record post-TLAS copy");
        expect(prepare(list, resolve, vk, out) == ReplayResult::success, "prepare TLAS + post-TLAS copy");
        expect(out.commands.size() == 2 && out.commands[1].buffer_copies.size() == 1,
               "post-TLAS CopyBuffer is prepared");
        expect(out.wrote_readback && out.readback_src_stages == VK_PIPELINE_STAGE_2_COPY_BIT,
               "post-TLAS readback CopyBuffer sets wrote_readback");

        emitted.clear();
        saw_array_of_pointers = false;
        saw_instances_va = 0;
        emit(out, handle<VkCommandBuffer>(0x400), vk);
        expect(emitted.size() == 2 && emitted[0] == "build_as" && emitted[1] == "copy_buffer",
               "emit runs build_as then copy_buffer");
        expect(saw_array_of_pointers && saw_instances_va == 0xdeadbeef00ULL,
               "emit preserves staged KHR instance pointer address");
    }

    // 2. Errors on commands after the KHR ARRAY_OF_POINTERS TLAS build must still be caught.
    {
        BuildAccelerationStructure bad_second = k_tlas_ptr;
        bad_second.dst_va = 0x5000;  // unresolvable by vk.resolve_acceleration_structure
        CommandList list;
        expect(list.build_acceleration_structure(k_tlas_ptr), "record valid first TLAS");
        expect(list.build_acceleration_structure(bad_second) && list.close(), "record bad second TLAS");
        expect(prepare(list, resolve, vk, out) == ReplayResult::unresolved_resource,
               "subsequent unresolvable TLAS returns unresolved_resource");
    }

    // 3. Post-loop occlusion segment planning and sum_occlusion validation must still run.
    {
        CommandList list;
        expect(list.begin_query({77, d3d12_query::occlusion, 0}), "record BeginQuery 0");
        expect(list.begin_query({77, d3d12_query::occlusion, 1}), "record BeginQuery 1");
        expect(list.build_acceleration_structure(k_tlas_ptr), "record TLAS inside overlapping occlusion");
        expect(list.end_query({77, d3d12_query::occlusion, 0}), "record EndQuery 0");
        expect(list.end_query({77, d3d12_query::occlusion, 1}) && list.close(), "record EndQuery 1");

        expect(prepare(list, resolve, vk, out) == ReplayResult::success,
               "prepare overlapping occlusion around KHR TLAS");
        expect(out.queries.size() == 4, "all four query commands prepared past KHR TLAS");
        expect(out.segment_count == 3, "post-loop occlusion segment planner runs");

        vk.sum_occlusion = nullptr;
        expect(prepare(list, resolve, vk, out) == ReplayResult::unsupported,
               "post-loop sum_occlusion requirement enforced past KHR TLAS");
        vk.sum_occlusion = [&](VkCommandBuffer, VkQueryPool, std::uint32_t, std::uint32_t, VkBuffer, VkDeviceSize) {};
    }

    // 4. BeginQuery + EndQuery + ResolveQueries following a KHR ARRAY_OF_POINTERS TLAS build are all prepared.
    {
        CommandList list;
        expect(list.build_acceleration_structure(k_tlas_ptr), "record TLAS before query sequence");
        expect(list.begin_query({77, d3d12_query::occlusion, 2}), "record BeginQuery 2");
        expect(list.end_query({77, d3d12_query::occlusion, 2}), "record EndQuery 2");
        expect(list.resolve_queries({77, d3d12_query::occlusion, 2, 1, 2, 0}) && list.close(),
               "record ResolveQueries 2");
        expect(prepare(list, resolve, vk, out) == ReplayResult::success,
               "prepare TLAS followed by occlusion query lifecycle");
        expect(out.queries.size() == 3 && out.wrote_readback,
               "post-TLAS query lifecycle prepared and readback marked");
    }

    // 5. A failing prepare_khr_instances still aborts prepare() with its exact error.
    {
        khr_status = ReplayResult::invalid_copy;
        CommandList list;
        expect(list.build_acceleration_structure(k_tlas_ptr), "record failing TLAS pointer build");
        expect(list.copy_buffer({2, 0, 1, 0, 64}) && list.close(), "record trailing copy after failing TLAS");
        expect(prepare(list, resolve, vk, out) == ReplayResult::invalid_copy,
               "prepare_khr_instances error propagates");
        expect(!out.wrote_readback && out.commands[1].buffer_copies.empty(),
               "failing prepare_khr_instances stops before trailing copy");
        khr_status = ReplayResult::success;
    }

    // Sanity check unused lambda suppression / clean build without warnings.
    (void)allow_sum_occlusion;

    if (failures) return 1;
    std::printf("replay_khr_tlas_pointers: prepare/emit and query checks continue past KHR ARRAY_OF_POINTERS TLAS\n");
    return 0;
}
