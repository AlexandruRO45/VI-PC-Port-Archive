// D3D-162: XDXR compute passes (BuildAccelerationStructure, CopyRaytracingAccelerationStructure, and
// EmitRaytracingAccelerationStructurePostbuildInfo) dispatch internal compute shaders and must be excluded from
// active D3D12_QUERY_TYPE_PIPELINE_STATISTICS queries by splitting around them, both alone and when queries overlap.
// CPU mocks only, no GPU or game data.
#include "../src/vulkan_replay.h"

#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

namespace {

using namespace leonida;

struct Trace {
    bool stats_active = false;
    unsigned xdxr_in_active_stats = 0;
    std::vector<std::string> events;
};

Trace* g_trace = nullptr;

template<class T>
T handle(std::uintptr_t n) {
    return reinterpret_cast<T>(n);
}

std::string query_event(const char* name, VkQueryPool pool, std::uint32_t index) {
    return std::string(name) + " " + std::to_string(reinterpret_cast<std::uintptr_t>(pool)) + ":" +
           std::to_string(index);
}

VKAPI_ATTR void VKAPI_CALL mock_reset_query(VkCommandBuffer, VkQueryPool, std::uint32_t, std::uint32_t) {}
VKAPI_ATTR void VKAPI_CALL mock_begin_query(VkCommandBuffer, VkQueryPool pool, std::uint32_t index,
                                            VkQueryControlFlags) {
    g_trace->stats_active = true;
    g_trace->events.push_back(query_event("begin_query", pool, index));
}
VKAPI_ATTR void VKAPI_CALL mock_end_query(VkCommandBuffer, VkQueryPool pool, std::uint32_t index) {
    g_trace->stats_active = false;
    g_trace->events.push_back(query_event("end_query", pool, index));
}
VKAPI_ATTR void VKAPI_CALL mock_write_timestamp(VkCommandBuffer, VkPipelineStageFlagBits, VkQueryPool, std::uint32_t) {}
VKAPI_ATTR void VKAPI_CALL mock_copy_query(VkCommandBuffer, VkQueryPool, std::uint32_t, std::uint32_t, VkBuffer,
                                           VkDeviceSize, VkDeviceSize, VkQueryResultFlags) {}
VKAPI_ATTR void VKAPI_CALL mock_fill_buffer(VkCommandBuffer, VkBuffer, VkDeviceSize, VkDeviceSize, std::uint32_t) {}
VKAPI_ATTR void VKAPI_CALL mock_destroy_pool(VkDevice, VkQueryPool, const VkAllocationCallbacks*) {}
VKAPI_ATTR void VKAPI_CALL mock_khr_build(VkCommandBuffer, std::uint32_t,
                                          const VkAccelerationStructureBuildGeometryInfoKHR*,
                                          const VkAccelerationStructureBuildRangeInfoKHR* const*) {
    g_trace->events.push_back("khr_build");
}

int failures = 0;
void expect(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++failures;
    }
}

BuildAccelerationStructure sample_blas_build() {
    BuildAccelerationStructure build{};
    build.dst_va = 0x1000;
    build.scratch_va = 0x2000;
    RtTriangleGeometry tri{};
    tri.vertex_va = 0x3000;
    tri.vertex_count = 3;
    tri.vertex_stride = 12;
    tri.vertex_format = VK_FORMAT_R32G32B32_SFLOAT;
    build.triangles.push_back(tri);
    return build;
}

}  // namespace

int main() {
    ResourceBinding resolve_dst{};
    resolve_dst.buffer = handle<VkBuffer>(1);
    resolve_dst.buffer_offset = 0;
    resolve_dst.size = 512;

    QueryPool statistics_heap{handle<VkQueryPool>(7), d3d12_query_heap::pipeline_statistics, 4, false,
                              std::vector<bool>(4, false), {}};

    const auto resolve = [&](ResourceId id) -> const ResourceBinding* {
        return id == 1 ? &resolve_dst : nullptr;
    };

    ReplayFunctions vk{};
    vk.reset_query_pool = mock_reset_query;
    vk.begin_query = mock_begin_query;
    vk.end_query = mock_end_query;
    vk.write_timestamp = mock_write_timestamp;
    vk.copy_query_pool_results = mock_copy_query;
    vk.fill_buffer = mock_fill_buffer;
    vk.query_pool = [&](QueryHeapId id) -> QueryPool* { return id == 6 ? &statistics_heap : nullptr; };
    vk.sum_statistics = [](VkCommandBuffer, VkQueryPool pool, VkQueryPool, const std::uint8_t*, std::uint32_t first,
                           std::uint32_t count, VkBuffer, VkDeviceSize) {
        g_trace->events.push_back(query_event("sum_statistics", pool, first) + "+" + std::to_string(count));
    };
    vk.buffer_at = [](std::uint64_t, std::uint64_t& size, VkBuffer& buf, VkDeviceSize& off) {
        buf = handle<VkBuffer>(2);
        off = 0;
        size = 4096;
        return true;
    };
    vk.xdxr_check = [](const BuildAccelerationStructure&) { return true; };
    vk.xdxr_record = [](VkCommandBuffer, const BuildAccelerationStructure&) {
        if (g_trace->stats_active) ++g_trace->xdxr_in_active_stats;
        g_trace->events.push_back("xdxr_record");
    };
    vk.xdxr_copy = [](VkCommandBuffer, std::uint64_t, std::uint64_t, std::uint64_t) {
        if (g_trace->stats_active) ++g_trace->xdxr_in_active_stats;
        g_trace->events.push_back("xdxr_copy");
    };
    vk.xdxr_postbuild = [](VkCommandBuffer, std::uint64_t src_va, std::uint64_t) {
        if (g_trace->stats_active) ++g_trace->xdxr_in_active_stats;
        g_trace->events.push_back("xdxr_postbuild:" + std::to_string(src_va));
    };

    // 1. Each XDXR command inside an active pipeline-statistics query is refused when statistics_helpers == false.
    {
        for (int kind = 0; kind < 3; ++kind) {
            CommandList list;
            expect(list.begin_query({6, d3d12_query::pipeline_statistics, 0}), "single guard begin");
            if (kind == 0)
                expect(list.build_acceleration_structure(sample_blas_build()), "single guard build");
            else if (kind == 1)
                expect(list.copy_acceleration_structure({0x4000, 0x1000, 0}), "single guard copy");
            else
                expect(list.acceleration_structure_postbuild({0x5000, 0, {0x1000}}), "single guard postbuild");
            expect(list.end_query({6, d3d12_query::pipeline_statistics, 0}), "single guard end");
            expect(list.close(), "close single guard list");
            PreparedList prepared;
            expect(prepare(list, resolve, vk, prepared, false) == ReplayResult::unsupported,
                   "statistics_helpers=false refuses XDXR command inside pipeline statistics");
        }
    }

    // 2. Single pipeline-statistics query spanning build + copy + multi-source postbuild splits into 4 physical segments,
    //    rejects planning without a statistics segment pool, and keeps every XDXR pass outside active queries.
    {
        statistics_heap.ended.assign(4, false);
        statistics_heap.occlusion.clear();
        Trace trace;
        g_trace = &trace;
        CommandList list;
        expect(list.begin_query({6, d3d12_query::pipeline_statistics, 0}), "single begin");
        expect(list.build_acceleration_structure(sample_blas_build()), "single build");
        expect(list.copy_acceleration_structure({0x4000, 0x1000, 0}), "single copy");
        expect(list.acceleration_structure_postbuild({0x5000, 0, {0x1000, 0x4000}}), "single postbuild");
        expect(list.end_query({6, d3d12_query::pipeline_statistics, 0}), "single end");
        expect(list.resolve_queries({6, d3d12_query::pipeline_statistics, 0, 1, 1, 0}), "single resolve");
        expect(list.close(), "close single list");

        PreparedList prepared;
        expect(prepare(list, resolve, vk, prepared, true) == ReplayResult::success, "prepare single list");
        expect(prepared.statistics_segment_count == 4, "single list: 4 statistics segments");

        QueryReplay no_pool;
        expect(no_pool.plan(prepared, {}, 0, true) == ReplayResult::unsupported,
               "no statistics pool: refuses XDXR helper markers");

        auto pool = std::make_shared<OcclusionSegmentPool>();
        pool->pool = handle<VkQueryPool>(9);
        pool->destroy = mock_destroy_pool;
        QueryReplay queries;
        expect(queries.plan(prepared, {}, 0, true, pool, 4) == ReplayResult::success, "plan single list");
        expect(queries.needs_statistics_sum, "single list requires statistics sum");

        emit(prepared, VK_NULL_HANDLE, vk);
        queries.commit();

        const std::vector<std::string> want{
            "begin_query 9:4",
            "end_query 9:4",
            "xdxr_record",
            "begin_query 9:5",
            "end_query 9:5",
            "xdxr_copy",
            "begin_query 9:6",
            "end_query 9:6",
            "xdxr_postbuild:4096",
            "xdxr_postbuild:16384",
            "begin_query 9:7",
            "end_query 9:7",
            "sum_statistics 9:4+4",
        };
        expect(trace.events == want, "single list: exact segment and XDXR ordering");
        expect(trace.xdxr_in_active_stats == 0, "single list: zero XDXR calls inside active statistics query");
    }

    // 3. Overlapping pipeline-statistics queries spanning XDXR build, copy, and postbuild split into 6 physical segments
    //    and keep every XDXR callback outside active statistics queries.
    {
        statistics_heap.ended.assign(4, false);
        statistics_heap.occlusion.clear();
        Trace trace;
        g_trace = &trace;
        CommandList list;
        expect(list.begin_query({6, d3d12_query::pipeline_statistics, 0}), "overlap begin 0");
        expect(list.build_acceleration_structure(sample_blas_build()), "overlap build");
        expect(list.begin_query({6, d3d12_query::pipeline_statistics, 1}), "overlap begin 1");
        expect(list.copy_acceleration_structure({0x4000, 0x1000, 0}), "overlap copy");
        expect(list.end_query({6, d3d12_query::pipeline_statistics, 0}), "overlap end 0");
        expect(list.acceleration_structure_postbuild({0x5000, 0, {0x1000, 0x4000}}), "overlap postbuild");
        expect(list.end_query({6, d3d12_query::pipeline_statistics, 1}), "overlap end 1");
        expect(list.resolve_queries({6, d3d12_query::pipeline_statistics, 0, 2, 1, 0}), "overlap resolve");
        expect(list.close(), "close overlap list");

        PreparedList prepared;
        expect(prepare(list, resolve, vk, prepared, true) == ReplayResult::success, "prepare overlap list");
        expect(prepared.statistics_overlaps, "overlap list sets statistics_overlaps");
        expect(prepared.statistics_segment_count == 6, "overlap list: 6 statistics segments");

        auto pool = std::make_shared<OcclusionSegmentPool>();
        pool->pool = handle<VkQueryPool>(9);
        pool->destroy = mock_destroy_pool;
        QueryReplay queries;
        expect(queries.plan(prepared, {}, 0, true, pool, 10) == ReplayResult::success, "plan overlap list");
        expect(queries.needs_statistics_sum, "overlap list requires statistics sum");

        emit(prepared, VK_NULL_HANDLE, vk);
        queries.commit();

        const std::vector<std::string> want{
            "begin_query 9:10",
            "end_query 9:10",
            "xdxr_record",
            "begin_query 9:11",
            "end_query 9:11",
            "begin_query 9:12",
            "end_query 9:12",
            "xdxr_copy",
            "begin_query 9:13",
            "end_query 9:13",
            "begin_query 9:14",
            "end_query 9:14",
            "xdxr_postbuild:4096",
            "xdxr_postbuild:16384",
            "begin_query 9:15",
            "end_query 9:15",
            "sum_statistics 9:10+4",
            "sum_statistics 9:12+4",
        };
        expect(trace.events == want, "overlap list: exact segment and XDXR ordering");
        expect(trace.xdxr_in_active_stats == 0, "overlap list: zero XDXR calls inside active statistics query");
    }

    // 4. Outside any statistics query, XDXR commands succeed even when statistics_helpers == false and allocate no segments.
    {
        Trace trace;
        g_trace = &trace;
        CommandList list;
        expect(list.build_acceleration_structure(sample_blas_build()), "untracked build");
        expect(list.copy_acceleration_structure({0x4000, 0x1000, 0}), "untracked copy");
        expect(list.acceleration_structure_postbuild({0x5000, 0, {0x1000}}), "untracked postbuild");
        expect(list.close(), "close untracked list");

        PreparedList prepared;
        expect(prepare(list, resolve, vk, prepared, false) == ReplayResult::success,
               "XDXR outside statistics succeeds with statistics_helpers=false");
        expect(prepared.statistics_segment_count == 0, "XDXR outside statistics creates no segments");
        emit(prepared, VK_NULL_HANDLE, vk);
        const std::vector<std::string> want{"xdxr_record", "xdxr_copy", "xdxr_postbuild:4096"};
        expect(trace.events == want, "XDXR outside statistics emits cleanly");
    }

    // 5. KHR acceleration-structure builds (and skipped postbuild) inside pipeline statistics do not require
    //    statistics_helpers and allocate no segments.
    {
        Trace trace;
        g_trace = &trace;
        ReplayFunctions khr_vk = vk;
        khr_vk.xdxr_check = nullptr;
        khr_vk.xdxr_record = nullptr;
        khr_vk.xdxr_copy = nullptr;
        khr_vk.xdxr_postbuild = nullptr;
        khr_vk.build_acceleration_structures = mock_khr_build;
        khr_vk.resolve_acceleration_structure = [](std::uint64_t) { return handle<VkAccelerationStructureKHR>(42); };

        CommandList list;
        expect(list.begin_query({6, d3d12_query::pipeline_statistics, 0}), "khr begin");
        expect(list.build_acceleration_structure(sample_blas_build()), "khr build");
        expect(list.acceleration_structure_postbuild({0x5000, 0, {0x1000}}), "khr postbuild");
        expect(list.end_query({6, d3d12_query::pipeline_statistics, 0}), "khr end");
        expect(list.close(), "close khr list");

        PreparedList prepared;
        expect(prepare(list, resolve, khr_vk, prepared, false) == ReplayResult::success,
               "KHR build inside statistics succeeds with statistics_helpers=false");
        expect(prepared.statistics_segment_count == 0, "KHR path creates no statistics segments");
        emit(prepared, VK_NULL_HANDLE, khr_vk);
        const std::vector<std::string> want{"begin_query 7:0", "khr_build", "end_query 7:0"};
        expect(trace.events == want, "KHR path leaves statistics query intact");
    }

    // 6. Half-wired XDXR callbacks are refused explicitly rather than falling back silently.
    {
        CommandList list;
        expect(list.build_acceleration_structure(sample_blas_build()), "half-wired build");
        expect(list.close(), "close half-wired list");

        ReplayFunctions no_record = vk;
        no_record.xdxr_record = nullptr;
        PreparedList p1;
        expect(prepare(list, resolve, no_record, p1, true) == ReplayResult::unsupported,
               "xdxr_check without xdxr_record is unsupported");

        ReplayFunctions no_check = vk;
        no_check.xdxr_check = nullptr;
        PreparedList p2;
        expect(prepare(list, resolve, no_check, p2, true) == ReplayResult::unsupported,
               "xdxr_record without xdxr_check is unsupported");
    }

    if (failures) {
        std::fprintf(stderr, "replay_xdxr_statistics_test: %d failure(s)\n", failures);
        return 1;
    }
    std::puts("replay_xdxr_statistics_test: all cases passed");
    return 0;
}
