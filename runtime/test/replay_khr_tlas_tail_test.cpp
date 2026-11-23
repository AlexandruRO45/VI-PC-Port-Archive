// A successful KHR TLAS ARRAY_OF_POINTERS build in prepare() must continue preparing and validating
// every subsequent command in the list (as well as running post-loop query planning), while still
// propagating errors returned by prepare_khr_instances.
// No Vulkan loader, GPU or game data.
#include "../src/vulkan_replay.h"

#include <cstdint>
#include <cstdio>
#include <vector>

namespace {

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

struct RecordedCalls {
    std::vector<VkDeviceAddress> tlas_instance_addrs;
    std::vector<VkBool32> tlas_array_of_pointers;
    std::vector<VkBuffer> copy_srcs;
    std::vector<VkBuffer> copy_dsts;
    std::vector<VkBufferCopy> copy_regions;
    std::vector<VkPipeline> bound_compute_pipelines;
    std::uint32_t dispatch_calls = 0;
};

RecordedCalls* g_recorded = nullptr;

VKAPI_ATTR void VKAPI_CALL record_build_as(VkCommandBuffer, std::uint32_t info_count,
                                           const VkAccelerationStructureBuildGeometryInfoKHR* infos,
                                           const VkAccelerationStructureBuildRangeInfoKHR* const*) {
    if (!g_recorded) return;
    for (std::uint32_t i = 0; i < info_count; ++i) {
        if (infos[i].geometryCount && infos[i].pGeometries &&
            infos[i].pGeometries[0].geometryType == VK_GEOMETRY_TYPE_INSTANCES_KHR) {
            const auto& inst = infos[i].pGeometries[0].geometry.instances;
            g_recorded->tlas_instance_addrs.push_back(inst.data.deviceAddress);
            g_recorded->tlas_array_of_pointers.push_back(inst.arrayOfPointers);
        }
    }
}

VKAPI_ATTR void VKAPI_CALL record_copy_buffer(VkCommandBuffer, VkBuffer src, VkBuffer dst,
                                              std::uint32_t count, const VkBufferCopy* regions) {
    if (!g_recorded) return;
    for (std::uint32_t i = 0; i < count; ++i) {
        g_recorded->copy_srcs.push_back(src);
        g_recorded->copy_dsts.push_back(dst);
        g_recorded->copy_regions.push_back(regions[i]);
    }
}

VKAPI_ATTR void VKAPI_CALL record_bind_pipeline(VkCommandBuffer, VkPipelineBindPoint point, VkPipeline pipeline) {
    if (!g_recorded) return;
    if (point == VK_PIPELINE_BIND_POINT_COMPUTE) g_recorded->bound_compute_pipelines.push_back(pipeline);
}

VKAPI_ATTR void VKAPI_CALL record_push_constants(VkCommandBuffer, VkPipelineLayout, VkShaderStageFlags,
                                                 std::uint32_t, std::uint32_t, const void*) {}

VKAPI_ATTR void VKAPI_CALL record_dispatch(VkCommandBuffer, std::uint32_t, std::uint32_t, std::uint32_t) {
    if (g_recorded) ++g_recorded->dispatch_calls;
}

}  // namespace

int main() {
    using namespace leonida;

    ResourceBinding src_buf{};
    src_buf.buffer = handle<VkBuffer>(0x101);
    src_buf.buffer_offset = 64;
    src_buf.size = 1024;

    ResourceBinding readback_buf{};
    readback_buf.buffer = handle<VkBuffer>(0x102);
    readback_buf.buffer_offset = 128;
    readback_buf.size = 1024;
    readback_buf.host_readable = true;

    const auto resolve = [&](ResourceId id) -> const ResourceBinding* {
        if (id == 1) return &src_buf;
        if (id == 2) return &readback_buf;
        return nullptr;
    };

    ComputePipeline comp_pipe{};
    comp_pipe.pipeline = handle<VkPipeline>(0x201);
    comp_pipe.layout = handle<VkPipelineLayout>(0x202);
    comp_pipe.code_base = 0x3000;

    QueryPool occlusion_heap{};
    occlusion_heap.heap_type = d3d12_query_heap::occlusion;
    occlusion_heap.pool = handle<VkQueryPool>(0x401);
    occlusion_heap.count = 8;

    ReplayResult stage_result = ReplayResult::success;
    VkDeviceAddress next_staged_addr = 0x50000;

    ReplayFunctions vk{};
    vk.pipeline_barrier = available<PFN_vkCmdPipelineBarrier>();
    vk.copy_buffer = record_copy_buffer;
    vk.copy_buffer_to_image = available<PFN_vkCmdCopyBufferToImage>();
    vk.copy_image_to_buffer = available<PFN_vkCmdCopyImageToBuffer>();
    vk.copy_image = available<PFN_vkCmdCopyImage>();
    vk.bind_pipeline = record_bind_pipeline;
    vk.push_constants = record_push_constants;
    vk.dispatch = record_dispatch;
    vk.compute_pipeline = [&](std::uintptr_t key) -> const ComputePipeline* {
        return key == 1 ? &comp_pipe : nullptr;
    };
    vk.build_acceleration_structures = record_build_as;
    vk.resolve_acceleration_structure = [](std::uint64_t va) -> VkAccelerationStructureKHR {
        if (va == 0x1000) return handle<VkAccelerationStructureKHR>(0x301);
        if (va == 0x2000) return handle<VkAccelerationStructureKHR>(0x302);
        return VK_NULL_HANDLE;
    };
    vk.host_at = [](std::uint64_t, std::uint64_t, const std::uint8_t*& out_ptr) -> HostReadStatus {
        static const std::uint8_t dummy[64]{};
        out_ptr = dummy;
        return HostReadStatus::ok;
    };
    vk.buffer_at = [](std::uint64_t, std::uint64_t&, VkBuffer& buf, VkDeviceSize& off) {
        buf = handle<VkBuffer>(0x101);
        off = 0;
        return true;
    };
    vk.prepare_khr_instances = [&](const BuildAccelerationStructure&, VkDeviceAddress& out_addr) {
        if (stage_result != ReplayResult::success) return stage_result;
        out_addr = next_staged_addr;
        next_staged_addr += 0x1000;
        return ReplayResult::success;
    };
    vk.reset_query_pool = available<PFN_vkCmdResetQueryPool>();
    vk.begin_query = available<PFN_vkCmdBeginQuery>();
    vk.end_query = available<PFN_vkCmdEndQuery>();
    vk.write_timestamp = available<PFN_vkCmdWriteTimestamp>();
    vk.copy_query_pool_results = available<PFN_vkCmdCopyQueryPoolResults>();
    vk.fill_buffer = available<PFN_vkCmdFillBuffer>();
    vk.query_pool = [&](QueryHeapId id) -> QueryPool* { return id == 1 ? &occlusion_heap : nullptr; };
    vk.sum_occlusion = [](VkCommandBuffer, VkQueryPool, std::uint32_t, std::uint32_t, VkBuffer, VkDeviceSize) {};

    const auto make_tlas = [](std::uint64_t dst_va) {
        BuildAccelerationStructure b{};
        b.is_tlas = true;
        b.dst_va = dst_va;
        b.scratch_va = 0x4000;
        b.instances_va = 0x8000;
        b.instance_count = 2;
        b.instance_pointers = true;
        return b;
    };

    PreparedList out;

    // 1. KHR ARRAY_OF_POINTERS TLAS build followed by CopyBuffer (into readback) and Dispatch:
    //    every trailing command must be prepared and emitted, and readback tracking must reflect the copy.
    {
        CommandList list;
        expect(list.build_acceleration_structure(make_tlas(0x1000)), "record tlas");
        expect(list.copy_buffer({2, 16, 1, 32, 256}), "record trailing copy_buffer");
        Dispatch disp{};
        disp.pipeline = 1;
        disp.x = 1;
        disp.y = 1;
        disp.z = 1;
        expect(list.dispatch(disp), "record trailing dispatch");
        expect(list.close(), "close list 1");

        next_staged_addr = 0x50000;
        stage_result = ReplayResult::success;
        expect(prepare(list, resolve, vk, out) == ReplayResult::success, "prepare list 1");
        expect(out.commands.size() == 3, "list 1 prepared command count");
        expect(out.commands[0].khr_instances == 0x50000, "list 1 tlas staged addr");
        expect(out.commands[1].binding.buffer == readback_buf.buffer &&
                   out.commands[1].source.buffer == src_buf.buffer && out.commands[1].buffer_copies.size() == 1,
               "list 1 trailing copy prepared");
        expect(out.wrote_readback && (out.readback_src_stages & VK_PIPELINE_STAGE_2_COPY_BIT),
               "list 1 trailing readback tracked");
        expect(out.commands[2].pipeline.pipeline == comp_pipe.pipeline, "list 1 trailing dispatch prepared");

        RecordedCalls rec;
        g_recorded = &rec;
        emit(out, handle<VkCommandBuffer>(0x900), vk);
        g_recorded = nullptr;

        expect(rec.tlas_instance_addrs.size() == 1 && rec.tlas_instance_addrs[0] == 0x50000 &&
                   rec.tlas_array_of_pointers[0] == VK_TRUE,
               "list 1 emitted tlas build");
        expect(rec.copy_srcs.size() == 1 && rec.copy_srcs[0] == src_buf.buffer &&
                   rec.copy_dsts[0] == readback_buf.buffer && rec.copy_regions[0].srcOffset == 96 &&
                   rec.copy_regions[0].dstOffset == 144 && rec.copy_regions[0].size == 256,
               "list 1 emitted trailing copy");
        expect(rec.bound_compute_pipelines.size() == 1 && rec.bound_compute_pipelines[0] == comp_pipe.pipeline &&
                   rec.dispatch_calls == 1,
               "list 1 emitted trailing dispatch");
    }

    // 2. Two consecutive KHR ARRAY_OF_POINTERS TLAS builds must both be prepared and emitted.
    {
        CommandList list;
        expect(list.build_acceleration_structure(make_tlas(0x1000)), "record first tlas");
        expect(list.build_acceleration_structure(make_tlas(0x2000)), "record second tlas");
        expect(list.close(), "close list 2");

        next_staged_addr = 0x60000;
        stage_result = ReplayResult::success;
        expect(prepare(list, resolve, vk, out) == ReplayResult::success, "prepare list 2");
        expect(out.commands[0].khr_instances == 0x60000 && out.commands[1].khr_instances == 0x61000,
               "list 2 both tlas builds staged");

        RecordedCalls rec;
        g_recorded = &rec;
        emit(out, handle<VkCommandBuffer>(0x900), vk);
        g_recorded = nullptr;
        expect(rec.tlas_instance_addrs.size() == 2 && rec.tlas_instance_addrs[0] == 0x60000 &&
                   rec.tlas_instance_addrs[1] == 0x61000,
               "list 2 emitted both tlas builds");
    }

    // 3. A trailing invalid command after the TLAS build must still be rejected by prepare().
    {
        CommandList list;
        expect(list.build_acceleration_structure(make_tlas(0x1000)), "record tlas before bad copy");
        expect(list.copy_buffer({99, 0, 1, 0, 64}), "record unresolved trailing copy");
        expect(list.close(), "close list 3");

        stage_result = ReplayResult::success;
        expect(prepare(list, resolve, vk, out) == ReplayResult::unresolved_resource,
               "list 3 rejects unresolved command after tlas");
    }

    // 4. Overlapping occlusion queries spanning the TLAS build must still reach post-loop planning.
    {
        CommandList list;
        expect(list.begin_query({1, d3d12_query::occlusion, 0}), "begin outer occlusion");
        expect(list.begin_query({1, d3d12_query::occlusion, 1}), "begin inner occlusion");
        expect(list.build_acceleration_structure(make_tlas(0x1000)), "record tlas inside queries");
        expect(list.end_query({1, d3d12_query::occlusion, 1}), "end inner occlusion");
        expect(list.end_query({1, d3d12_query::occlusion, 0}), "end outer occlusion");
        expect(list.close(), "close list 4");

        stage_result = ReplayResult::success;
        expect(prepare(list, resolve, vk, out) == ReplayResult::success, "prepare list 4");
        expect(out.queries.size() == 4, "list 4 collected all 4 query commands");
        expect(out.segment_count == 3, "list 4 ran plan_occlusion_segments after loop");
    }

    // 5. An error from prepare_khr_instances must still fail prepare() immediately.
    {
        CommandList list;
        expect(list.build_acceleration_structure(make_tlas(0x1000)), "record failing tlas");
        expect(list.copy_buffer({2, 0, 1, 0, 64}), "record trailing copy after failing tlas");
        expect(list.close(), "close list 5");

        stage_result = ReplayResult::invalid_copy;
        expect(prepare(list, resolve, vk, out) == ReplayResult::invalid_copy,
               "list 5 propagates prepare_khr_instances error");
    }

    if (failures) {
        std::fprintf(stderr, "%d failure(s)\n", failures);
        return 1;
    }
    std::puts("replay KHR TLAS ARRAY_OF_POINTERS tail preparation OK");
    return 0;
}
