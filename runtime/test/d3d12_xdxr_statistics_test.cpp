// D3D-162: XDXR compute passes (BuildRaytracingAccelerationStructure, CopyRaytracingAccelerationStructure, and
// EmitRaytracingAccelerationStructurePostbuildInfo) must not pollute D3D12_QUERY_TYPE_PIPELINE_STATISTICS
// counters (specifically CSInvocations) when recorded inside a single or overlapping pipeline-statistics query.
// Runs on the standalone Vulkan D3D12 device with XDXR enabled. Skips (exit 77) without Vulkan, with
// LEONIDA_RT_AS=khr, without bufferDeviceAddress, or without pipelineStatisticsQuery. Synthetic geometry only, no
// game files.
#include "../shims/d3d12_com.h"

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace {

template<class T> struct Ref {
    T* p = nullptr;
    ~Ref() { if (p) p->Release(); }
    T* operator->() const { return p; }
    T** put() { return &p; }
};

int fail(const char* what, HRESULT hr = S_OK) {
    std::fprintf(stderr, "FAIL: %s (hr 0x%08lx)\n", what, static_cast<unsigned long>(hr));
    return 1;
}

HRESULT make_buffer(ID3D12Device* device, D3D12_HEAP_TYPE heap, UINT64 size, D3D12_RESOURCE_STATES state,
                    D3D12_RESOURCE_FLAGS flags, ID3D12Resource** out) {
    D3D12_HEAP_PROPERTIES props{};
    props.Type = heap;
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width = size ? size : 1;
    desc.Height = 1;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    desc.Flags = flags;
    return device->CreateCommittedResource(&props, D3D12_HEAP_FLAG_NONE, &desc, state, nullptr, IID_PPV_ARGS(out));
}

void uav_barrier(ID3D12GraphicsCommandList4* list) {
    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
    barrier.UAV.pResource = nullptr;
    list->ResourceBarrier(1, &barrier);
}

void transition_barrier(ID3D12GraphicsCommandList4* list, ID3D12Resource* resource,
                        D3D12_RESOURCE_STATES before, D3D12_RESOURCE_STATES after) {
    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition.pResource = resource;
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    barrier.Transition.StateBefore = before;
    barrier.Transition.StateAfter = after;
    list->ResourceBarrier(1, &barrier);
}

bool submit_and_wait(ID3D12GraphicsCommandList4* list, ID3D12CommandQueue* queue, ID3D12Fence* fence, UINT64 value) {
    if (FAILED(list->Close())) return false;
    ID3D12CommandList* lists[] = {list};
    queue->ExecuteCommandLists(1, lists);
    if (FAILED(queue->Signal(fence, value))) return false;
    HANDLE event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    const bool ok = event && SUCCEEDED(fence->SetEventOnCompletion(value, event)) &&
                    WaitForSingleObject(event, 10000) == WAIT_OBJECT_0;
    if (event) CloseHandle(event);
    return ok;
}

bool check_zero_statistics(ID3D12Resource* counters_rb, UINT count, const char* stage_name) {
    void* mapped = nullptr;
    const D3D12_RANGE read_range{0, SIZE_T(count) * sizeof(D3D12_QUERY_DATA_PIPELINE_STATISTICS)};
    const D3D12_RANGE none{};
    if (FAILED(counters_rb->Map(0, &read_range, &mapped)) || !mapped) {
        std::fprintf(stderr, "FAIL: Map statistics readback for %s\n", stage_name);
        return false;
    }
    bool ok = true;
    for (UINT q = 0; q < count; ++q) {
        D3D12_QUERY_DATA_PIPELINE_STATISTICS stats{};
        std::memcpy(&stats, static_cast<const char*>(mapped) + SIZE_T(q) * sizeof(stats), sizeof(stats));
        std::array<UINT64, 11> words{};
        std::memcpy(words.data(), &stats, sizeof(stats));
        for (std::size_t w = 0; w < words.size(); ++w) {
            if (words[w] != 0) {
                std::fprintf(stderr,
                             "FAIL (%s query %u): counter word %zu = %llu (CSInvocations = %llu), expected 0\n",
                             stage_name, q, w, static_cast<unsigned long long>(words[w]),
                             static_cast<unsigned long long>(stats.CSInvocations));
                ok = false;
            }
        }
    }
    counters_rb->Unmap(0, &none);
    return ok;
}

}  // namespace

int main() {
    char rt_as[8] = {};
    std::size_t rt_as_length = 0;
    if (getenv_s(&rt_as_length, rt_as, "LEONIDA_RT_AS") == 0 && rt_as_length > 1 && std::strcmp(rt_as, "khr") == 0) {
        std::puts("SKIP: LEONIDA_RT_AS=khr");
        return 77;
    }

    Ref<ID3D12Device5> device;
    if (FAILED(rt::d3d12::create_vulkan_device_standalone(IID_PPV_ARGS(device.put())))) {
        std::puts("SKIP: no Vulkan D3D12 device");
        return 77;
    }

    // Check pipeline statistics support.
    Ref<ID3D12QueryHeap> stats_heap;
    const D3D12_QUERY_HEAP_DESC heap_desc{D3D12_QUERY_HEAP_TYPE_PIPELINE_STATISTICS, 2, 0};
    if (FAILED(device->CreateQueryHeap(&heap_desc, IID_PPV_ARGS(stats_heap.put())))) {
        std::puts("SKIP: device does not support pipeline statistics queries");
        return 77;
    }

    // Synthetic triangle soup (16 triangles).
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
    constexpr UINT tri_count = 16;
    std::vector<float> vertices(tri_count * 9);
    for (float& v : vertices) v = dist(rng);
    const UINT64 vertex_bytes = vertices.size() * sizeof(float);

    Ref<ID3D12Resource> upload;
    HRESULT hr = make_buffer(device.p, D3D12_HEAP_TYPE_UPLOAD, 1024, D3D12_RESOURCE_STATE_GENERIC_READ,
                             D3D12_RESOURCE_FLAG_NONE, upload.put());
    if (FAILED(hr)) return fail("upload buffer", hr);

    char* mapped = nullptr;
    if (FAILED(upload->Map(0, nullptr, reinterpret_cast<void**>(&mapped)))) return fail("Map upload");
    std::memcpy(mapped, vertices.data(), SIZE_T(vertex_bytes));
    const UINT64 vertex_va = upload->GetGPUVirtualAddress();
    const UINT64 instances_offset = 768;
    const UINT64 instances_va = vertex_va + instances_offset;

    D3D12_RAYTRACING_GEOMETRY_DESC geo{};
    geo.Type = D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES;
    geo.Flags = D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE;
    geo.Triangles.VertexFormat = DXGI_FORMAT_R32G32B32_FLOAT;
    geo.Triangles.VertexCount = tri_count * 3;
    geo.Triangles.VertexBuffer = {vertex_va, 12};

    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS blas_inputs{};
    blas_inputs.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL;
    blas_inputs.Flags = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_UPDATE;
    blas_inputs.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
    blas_inputs.NumDescs = 1;
    blas_inputs.pGeometryDescs = &geo;

    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO blas_info{};
    device->GetRaytracingAccelerationStructurePrebuildInfo(&blas_inputs, &blas_info);
    if (!blas_info.ResultDataMaxSizeInBytes) {
        std::puts("SKIP: device does not build XDXR acceleration structures (no bufferDeviceAddress)");
        return 77;
    }

    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS tlas_inputs{};
    tlas_inputs.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL;
    tlas_inputs.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
    tlas_inputs.NumDescs = 1;

    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO tlas_info{};
    device->GetRaytracingAccelerationStructurePrebuildInfo(&tlas_inputs, &tlas_info);
    if (!tlas_info.ResultDataMaxSizeInBytes) return fail("TLAS prebuild info empty");
    tlas_inputs.InstanceDescs = instances_va;

    const D3D12_RESOURCE_FLAGS uav = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    const D3D12_RESOURCE_STATES as_state = D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE;
    const UINT64 scratch_size = (std::max)(blas_info.ScratchDataSizeInBytes, tlas_info.ScratchDataSizeInBytes);

    Ref<ID3D12Resource> blas, cloned_blas, tlas, scratch, postbuild_buf, postbuild_rb, counters_rb;
    if (FAILED(hr = make_buffer(device.p, D3D12_HEAP_TYPE_DEFAULT, blas_info.ResultDataMaxSizeInBytes, as_state, uav,
                                blas.put())) ||
        FAILED(hr = make_buffer(device.p, D3D12_HEAP_TYPE_DEFAULT, blas_info.ResultDataMaxSizeInBytes, as_state, uav,
                                cloned_blas.put())) ||
        FAILED(hr = make_buffer(device.p, D3D12_HEAP_TYPE_DEFAULT, tlas_info.ResultDataMaxSizeInBytes, as_state, uav,
                                tlas.put())) ||
        FAILED(hr = make_buffer(device.p, D3D12_HEAP_TYPE_DEFAULT, scratch_size, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                                uav, scratch.put())) ||
        FAILED(hr = make_buffer(device.p, D3D12_HEAP_TYPE_DEFAULT, 32, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, uav,
                                postbuild_buf.put())) ||
        FAILED(hr = make_buffer(device.p, D3D12_HEAP_TYPE_READBACK, 32, D3D12_RESOURCE_STATE_COPY_DEST,
                                D3D12_RESOURCE_FLAG_NONE, postbuild_rb.put())) ||
        FAILED(hr = make_buffer(device.p, D3D12_HEAP_TYPE_READBACK,
                                2 * sizeof(D3D12_QUERY_DATA_PIPELINE_STATISTICS),
                                D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_FLAG_NONE, counters_rb.put())))
        return fail("buffers", hr);

    D3D12_RAYTRACING_INSTANCE_DESC inst{};
    inst.Transform[0][0] = inst.Transform[1][1] = inst.Transform[2][2] = 1.0f;
    inst.InstanceMask = 0xff;
    inst.AccelerationStructure = blas->GetGPUVirtualAddress();
    std::memcpy(mapped + instances_offset, &inst, sizeof(inst));
    upload->Unmap(0, nullptr);

    D3D12_COMMAND_QUEUE_DESC qd{};
    qd.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    Ref<ID3D12CommandQueue> queue;
    Ref<ID3D12CommandAllocator> allocator;
    Ref<ID3D12GraphicsCommandList4> list;
    Ref<ID3D12Fence> fence;
    if (FAILED(device->CreateCommandQueue(&qd, IID_PPV_ARGS(queue.put()))) ||
        FAILED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(allocator.put()))) ||
        FAILED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.p, nullptr,
                                         IID_PPV_ARGS(list.put()))) ||
        FAILED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(fence.put()))))
        return fail("queue/allocator/list/fence");

    UINT64 fence_val = 0;

    // 1. Single pipeline-statistics query around BLAS build (with inline COMPACTED_SIZE postbuild), TLAS build, and
    //    BLAS PERFORM_UPDATE refit.
    {
        D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_DESC inline_pb{
            postbuild_buf->GetGPUVirtualAddress(), D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_COMPACTED_SIZE};
        list->BeginQuery(stats_heap.p, D3D12_QUERY_TYPE_PIPELINE_STATISTICS, 0);

        D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC build{};
        build.Inputs = blas_inputs;
        build.DestAccelerationStructureData = blas->GetGPUVirtualAddress();
        build.ScratchAccelerationStructureData = scratch->GetGPUVirtualAddress();
        list->BuildRaytracingAccelerationStructure(&build, 1, &inline_pb);
        uav_barrier(list.p);

        build.Inputs = tlas_inputs;
        build.DestAccelerationStructureData = tlas->GetGPUVirtualAddress();
        list->BuildRaytracingAccelerationStructure(&build, 0, nullptr);
        uav_barrier(list.p);

        build.Inputs = blas_inputs;
        build.Inputs.Flags |= D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PERFORM_UPDATE;
        build.DestAccelerationStructureData = blas->GetGPUVirtualAddress();
        build.SourceAccelerationStructureData = blas->GetGPUVirtualAddress();
        list->BuildRaytracingAccelerationStructure(&build, 0, nullptr);
        uav_barrier(list.p);

        list->EndQuery(stats_heap.p, D3D12_QUERY_TYPE_PIPELINE_STATISTICS, 0);
        list->ResolveQueryData(stats_heap.p, D3D12_QUERY_TYPE_PIPELINE_STATISTICS, 0, 1, counters_rb.p, 0);

        transition_barrier(list.p, postbuild_buf.p, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                           D3D12_RESOURCE_STATE_COPY_SOURCE);
        list->CopyBufferRegion(postbuild_rb.p, 0, postbuild_buf.p, 0, 8);
        transition_barrier(list.p, postbuild_buf.p, D3D12_RESOURCE_STATE_COPY_SOURCE,
                           D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

        if (!submit_and_wait(list.p, queue.p, fence.p, ++fence_val)) return fail("builds submit");
        if (FAILED(device->GetDeviceRemovedReason())) return fail("device removed (builds)", device->GetDeviceRemovedReason());
        if (!check_zero_statistics(counters_rb.p, 1, "builds (BLAS + TLAS + update)")) return 1;
    }

    // 2. Single pipeline-statistics query around EmitRaytracingAccelerationStructurePostbuildInfo (2 sources).
    UINT64 compacted_bytes = 0;
    {
        if (FAILED(allocator->Reset()) || FAILED(list->Reset(allocator.p, nullptr))) return fail("reset (postbuild)");
        list->BeginQuery(stats_heap.p, D3D12_QUERY_TYPE_PIPELINE_STATISTICS, 0);
        D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_DESC pb_desc{
            postbuild_buf->GetGPUVirtualAddress(), D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_COMPACTED_SIZE};
        const D3D12_GPU_VIRTUAL_ADDRESS sources[2] = {blas->GetGPUVirtualAddress(), tlas->GetGPUVirtualAddress()};
        list->EmitRaytracingAccelerationStructurePostbuildInfo(&pb_desc, 2, sources);
        uav_barrier(list.p);
        list->EndQuery(stats_heap.p, D3D12_QUERY_TYPE_PIPELINE_STATISTICS, 0);
        list->ResolveQueryData(stats_heap.p, D3D12_QUERY_TYPE_PIPELINE_STATISTICS, 0, 1, counters_rb.p, 0);

        transition_barrier(list.p, postbuild_buf.p, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                           D3D12_RESOURCE_STATE_COPY_SOURCE);
        list->CopyBufferRegion(postbuild_rb.p, 0, postbuild_buf.p, 0, 16);
        transition_barrier(list.p, postbuild_buf.p, D3D12_RESOURCE_STATE_COPY_SOURCE,
                           D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

        if (!submit_and_wait(list.p, queue.p, fence.p, ++fence_val)) return fail("postbuild submit");
        if (FAILED(device->GetDeviceRemovedReason())) return fail("device removed (postbuild)", device->GetDeviceRemovedReason());
        if (!check_zero_statistics(counters_rb.p, 1, "EmitPostbuildInfo")) return 1;

        void* p = nullptr;
        const D3D12_RANGE range{0, 16};
        const D3D12_RANGE none{};
        if (FAILED(postbuild_rb->Map(0, &range, &p)) || !p) return fail("Map postbuild");
        UINT64 sizes[2]{};
        std::memcpy(sizes, p, sizeof(sizes));
        postbuild_rb->Unmap(0, &none);
        if (!sizes[0] || sizes[0] > blas_info.ResultDataMaxSizeInBytes || !sizes[1] ||
            sizes[1] > tlas_info.ResultDataMaxSizeInBytes) {
            return fail("unexpected postbuild sizes");
        }
        compacted_bytes = sizes[0];
    }

    // 3. Single pipeline-statistics query around CopyRaytracingAccelerationStructure (CLONE + COMPACT).
    Ref<ID3D12Resource> compacted_blas;
    if (FAILED(hr = make_buffer(device.p, D3D12_HEAP_TYPE_DEFAULT, compacted_bytes, as_state, uav,
                                compacted_blas.put())))
        return fail("compacted AS buffer", hr);
    {
        if (FAILED(allocator->Reset()) || FAILED(list->Reset(allocator.p, nullptr))) return fail("reset (copy)");
        list->BeginQuery(stats_heap.p, D3D12_QUERY_TYPE_PIPELINE_STATISTICS, 0);
        list->CopyRaytracingAccelerationStructure(cloned_blas->GetGPUVirtualAddress(), blas->GetGPUVirtualAddress(),
                                                  D3D12_RAYTRACING_ACCELERATION_STRUCTURE_COPY_MODE_CLONE);
        list->CopyRaytracingAccelerationStructure(compacted_blas->GetGPUVirtualAddress(), blas->GetGPUVirtualAddress(),
                                                  D3D12_RAYTRACING_ACCELERATION_STRUCTURE_COPY_MODE_COMPACT);
        uav_barrier(list.p);
        list->EndQuery(stats_heap.p, D3D12_QUERY_TYPE_PIPELINE_STATISTICS, 0);
        list->ResolveQueryData(stats_heap.p, D3D12_QUERY_TYPE_PIPELINE_STATISTICS, 0, 1, counters_rb.p, 0);

        // Verify the compacted copy produced a valid non-zero structure via postbuild CURRENT_SIZE.
        D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_DESC cur_desc{
            postbuild_buf->GetGPUVirtualAddress() + 16,
            D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_CURRENT_SIZE};
        const D3D12_GPU_VIRTUAL_ADDRESS copied_va = compacted_blas->GetGPUVirtualAddress();
        list->EmitRaytracingAccelerationStructurePostbuildInfo(&cur_desc, 1, &copied_va);
        uav_barrier(list.p);
        transition_barrier(list.p, postbuild_buf.p, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                           D3D12_RESOURCE_STATE_COPY_SOURCE);
        list->CopyBufferRegion(postbuild_rb.p, 16, postbuild_buf.p, 16, 8);
        transition_barrier(list.p, postbuild_buf.p, D3D12_RESOURCE_STATE_COPY_SOURCE,
                           D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

        if (!submit_and_wait(list.p, queue.p, fence.p, ++fence_val)) return fail("copy submit");
        if (FAILED(device->GetDeviceRemovedReason())) return fail("device removed (copy)", device->GetDeviceRemovedReason());
        if (!check_zero_statistics(counters_rb.p, 1, "CopyRaytracingAccelerationStructure")) return 1;

        void* p = nullptr;
        const D3D12_RANGE range{16, 24};
        const D3D12_RANGE none{};
        if (FAILED(postbuild_rb->Map(0, &range, &p)) || !p) return fail("Map compacted current size");
        UINT64 current_size = 0;
        std::memcpy(&current_size, static_cast<const char*>(p) + 16, 8);
        postbuild_rb->Unmap(0, &none);
        if (current_size != compacted_bytes) return fail("compacted AS CURRENT_SIZE mismatch");
    }

    // 4. Overlapping pipeline-statistics queries spanning build, EmitPostbuildInfo, and CopyAS.
    {
        if (FAILED(allocator->Reset()) || FAILED(list->Reset(allocator.p, nullptr))) return fail("reset (overlap)");
        D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC build{};
        build.Inputs = blas_inputs;
        build.DestAccelerationStructureData = blas->GetGPUVirtualAddress();
        build.ScratchAccelerationStructureData = scratch->GetGPUVirtualAddress();

        list->BeginQuery(stats_heap.p, D3D12_QUERY_TYPE_PIPELINE_STATISTICS, 0);
        list->BuildRaytracingAccelerationStructure(&build, 0, nullptr);
        uav_barrier(list.p);

        list->BeginQuery(stats_heap.p, D3D12_QUERY_TYPE_PIPELINE_STATISTICS, 1);
        D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_DESC pb_desc{
            postbuild_buf->GetGPUVirtualAddress(), D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_CURRENT_SIZE};
        const D3D12_GPU_VIRTUAL_ADDRESS src_va = blas->GetGPUVirtualAddress();
        list->EmitRaytracingAccelerationStructurePostbuildInfo(&pb_desc, 1, &src_va);
        uav_barrier(list.p);

        list->EndQuery(stats_heap.p, D3D12_QUERY_TYPE_PIPELINE_STATISTICS, 0);
        list->CopyRaytracingAccelerationStructure(cloned_blas->GetGPUVirtualAddress(), blas->GetGPUVirtualAddress(),
                                                  D3D12_RAYTRACING_ACCELERATION_STRUCTURE_COPY_MODE_CLONE);
        uav_barrier(list.p);

        list->EndQuery(stats_heap.p, D3D12_QUERY_TYPE_PIPELINE_STATISTICS, 1);
        list->ResolveQueryData(stats_heap.p, D3D12_QUERY_TYPE_PIPELINE_STATISTICS, 0, 2, counters_rb.p, 0);

        if (!submit_and_wait(list.p, queue.p, fence.p, ++fence_val)) return fail("overlap submit");
        if (FAILED(device->GetDeviceRemovedReason())) return fail("device removed (overlap)", device->GetDeviceRemovedReason());
        if (!check_zero_statistics(counters_rb.p, 2, "overlapping XDXR statistics")) return 1;
    }

    std::puts("d3d12_xdxr_statistics_test: 4 GPU checks passed (CSInvocations == 0 across all XDXR passes)");
    return 0;
}
