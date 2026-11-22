// BuildRaytracingAccelerationStructure on the standalone Vulkan device writes XDXR BVH4 records (docs/d3d12_vulkan.md
// 4.8, src/xdxr_build.h), the layout the game's translated traversal shaders read. A BLAS of two geometries (an
// unindexed triangle soup, and an R16-indexed grid with a 16-byte stride and a 3x4 transform) and a TLAS of three
// instances are built through the D3D12 API, sized by GetRaytracingAccelerationStructurePrebuildInfo, copied to
// readback buffers and written to <out dir>; the BLAS is then updated in place (PERFORM_UPDATE) after its soup moves,
// and written again as d3d12_blas_update; then its postbuild COMPACTED_SIZE and a TLAS rebuild's CURRENT_SIZE are
// checked against the prebuild sizes, and the BLAS is compacted (CopyRaytracingAccelerationStructure COMPACT) into a
// buffer of exactly that size, written as d3d12_blas_compacted as <case>.bin + <case>.json at their console GPU virtual addresses.
// Then the input forms: the TLAS from D3D12_ELEMENTS_LAYOUT_ARRAY_OF_POINTERS (descs out of order and in two buffers)
// must equal the ARRAY build byte for byte, with the same prebuild sizes, and a null pointer, one past its buffer's
// end and one into no resource refuse the list; a scene in all six triangle vertex formats must equal its
// R32G32B32_FLOAT build (d3d12_fmt_*, with the source vertices for the checker to decode), and 16-bit vertex strides
// below the element size, odd strides and odd addresses are refused.
// `python -I tools/xdxr_bvh.py check-build <out dir>` validates them and traces rays against brute force (ctest:
// d3d12_xdxr_rt + d3d12_xdxr_rt_check). Exit 77 without a Vulkan device, or with LEONIDA_RT_AS=khr. No game files.
#include "../shims/d3d12_com.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
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

HRESULT buffer(ID3D12Device* device, D3D12_HEAP_TYPE heap, UINT64 size, D3D12_RESOURCE_STATES state,
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

std::string num(double x) {
    char s[64];
    std::snprintf(s, sizeof s, "%.9g", x);
    return s;
}

bool write_file(const std::string& path, const void* data, std::size_t bytes) {
    std::ofstream f(path, std::ios::binary);
    f.write(static_cast<const char*>(data), std::streamsize(bytes));
    return bool(f);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: d3d12_xdxr_rt_test <out dir>\n");
        return 2;
    }
    const std::string dir = argv[1];
    std::error_code ec;
    for (const auto& entry : std::filesystem::directory_iterator(dir, ec)) {
        const auto ext = entry.path().extension();
        if (ext == ".bin" || ext == ".json") std::filesystem::remove(entry.path(), ec);
    }

    // LEONIDA_RT_AS=khr builds #743's VkAccelerationStructureKHR instead, whose bytes are the driver's own.
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

    // ---- geometry: a triangle soup, and an R16-indexed grid (stride 16) with a transform ----
    std::mt19937 rng(11);
    std::uniform_real_distribution<float> spread(-4.0f, 4.0f), jitter(-0.6f, 0.6f);
    const UINT soup_tris = 300;
    std::vector<float> soup;
    for (UINT t = 0; t < soup_tris; ++t) {
        const float c[3] = {spread(rng), spread(rng), spread(rng)};
        for (int k = 0; k < 9; ++k) soup.push_back(c[k % 3] + jitter(rng));
    }
    const UINT n = 8;
    std::vector<float> grid;
    for (UINT j = 0; j <= n; ++j)
        for (UINT i = 0; i <= n; ++i) {
            grid.push_back(float(i) - n * 0.5f);
            grid.push_back(float(j) - n * 0.5f);
            grid.push_back(0.25f * float((i * 7 + j * 3) % 5));
            grid.push_back(9.0f);                                    // padding: the stride is 16
        }
    std::vector<std::uint16_t> indices;
    for (UINT j = 0; j < n; ++j)
        for (UINT i = 0; i < n; ++i) {
            const std::uint16_t a = std::uint16_t(j * (n + 1) + i);
            const std::uint16_t q[6] = {a, std::uint16_t(a + 1), std::uint16_t(a + n + 2), a, std::uint16_t(a + n + 2),
                                        std::uint16_t(a + n + 1)};
            indices.insert(indices.end(), q, q + 6);
        }
    const float transform[12] = {0, 2, 0, 1, 0, 0, 1, -2, 2, 0, 0, 3};   // exact in float

    const UINT64 soup_bytes = soup.size() * 4, grid_bytes = grid.size() * 4, index_bytes = indices.size() * 2;
    const UINT64 upload_size = 256 + soup_bytes + grid_bytes + index_bytes + 64 + 3 * 64;
    Ref<ID3D12Resource> upload;
    HRESULT hr = buffer(device.p, D3D12_HEAP_TYPE_UPLOAD, upload_size, D3D12_RESOURCE_STATE_GENERIC_READ,
                        D3D12_RESOURCE_FLAG_NONE, upload.put());
    if (FAILED(hr)) return fail("upload buffer", hr);
    char* mapped = nullptr;
    if (FAILED(upload->Map(0, nullptr, reinterpret_cast<void**>(&mapped)))) return fail("Map upload");
    UINT64 at = 0;
    auto place = [&](const void* data, UINT64 bytes, UINT64 align) {
        at = (at + align - 1) / align * align;
        std::memcpy(mapped + at, data, std::size_t(bytes));
        const UINT64 va = upload->GetGPUVirtualAddress() + at;
        at += bytes;
        return va;
    };
    const UINT64 soup_va = place(soup.data(), soup_bytes, 16);
    const UINT64 grid_va = place(grid.data(), grid_bytes, 16);
    const UINT64 index_va = place(indices.data(), index_bytes, 16);
    const UINT64 transform_va = place(transform, sizeof transform, 16);

    D3D12_RAYTRACING_GEOMETRY_DESC geometries[2]{};
    geometries[0].Type = D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES;
    geometries[0].Flags = D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE;
    geometries[0].Triangles.VertexFormat = DXGI_FORMAT_R32G32B32_FLOAT;
    geometries[0].Triangles.VertexCount = soup_tris * 3;
    geometries[0].Triangles.VertexBuffer = {soup_va, 12};
    geometries[1].Type = D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES;
    geometries[1].Flags = D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE;
    geometries[1].Triangles.VertexFormat = DXGI_FORMAT_R32G32B32_FLOAT;
    geometries[1].Triangles.VertexCount = (n + 1) * (n + 1);
    geometries[1].Triangles.VertexBuffer = {grid_va, 16};
    geometries[1].Triangles.IndexFormat = DXGI_FORMAT_R16_UINT;
    geometries[1].Triangles.IndexCount = UINT(indices.size());
    geometries[1].Triangles.IndexBuffer = index_va;
    geometries[1].Triangles.Transform3x4 = transform_va;

    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS blas_inputs{};
    blas_inputs.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL;
    blas_inputs.Flags = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_UPDATE;   // updated below
    blas_inputs.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
    blas_inputs.NumDescs = 2;
    blas_inputs.pGeometryDescs = geometries;
    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO blas_info{};
    device->GetRaytracingAccelerationStructurePrebuildInfo(&blas_inputs, &blas_info);
    if (!blas_info.ResultDataMaxSizeInBytes) {
        std::puts("SKIP: the device does not build XDXR acceleration structures (no bufferDeviceAddress)");
        return 77;
    }

    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS tlas_inputs{};
    tlas_inputs.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL;
    tlas_inputs.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
    tlas_inputs.NumDescs = 3;
    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO tlas_info{};
    device->GetRaytracingAccelerationStructurePrebuildInfo(&tlas_inputs, &tlas_info);
    if (!tlas_info.ResultDataMaxSizeInBytes) return fail("TLAS prebuild info is empty");

    const D3D12_RESOURCE_FLAGS uav = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    const D3D12_RESOURCE_STATES as_state = D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE;
    Ref<ID3D12Resource> blas, tlas, scratch, blas_rb, tlas_rb;
    const UINT64 scratch_size = blas_info.ScratchDataSizeInBytes > tlas_info.ScratchDataSizeInBytes
                                    ? blas_info.ScratchDataSizeInBytes : tlas_info.ScratchDataSizeInBytes;
    if (FAILED(hr = buffer(device.p, D3D12_HEAP_TYPE_DEFAULT, blas_info.ResultDataMaxSizeInBytes, as_state, uav, blas.put())) ||
        FAILED(hr = buffer(device.p, D3D12_HEAP_TYPE_DEFAULT, tlas_info.ResultDataMaxSizeInBytes, as_state, uav, tlas.put())) ||
        FAILED(hr = buffer(device.p, D3D12_HEAP_TYPE_DEFAULT, scratch_size, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, uav,
                           scratch.put())) ||
        FAILED(hr = buffer(device.p, D3D12_HEAP_TYPE_READBACK, blas_info.ResultDataMaxSizeInBytes,
                           D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_FLAG_NONE, blas_rb.put())) ||
        FAILED(hr = buffer(device.p, D3D12_HEAP_TYPE_READBACK, tlas_info.ResultDataMaxSizeInBytes,
                           D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_FLAG_NONE, tlas_rb.put())))
        return fail("acceleration-structure buffers", hr);
    if (blas->GetGPUVirtualAddress() % 256 || tlas->GetGPUVirtualAddress() % 256)
        return fail("acceleration structures must be 256-byte aligned");

    // ---- instances: identity, rotated + translated, and masked out ----
    const float c = std::cos(0.7f), s = std::sin(0.7f);
    const float xforms[3][12] = {{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0},
                                 {c, -s, 0, 9, s, c, 0, -1, 0, 0, 1, 0.5f},
                                 {2, 0, 0, -9, 0, 0.5f, 0, 2, 0, 0, 1.5f, 1}};
    const UINT masks[3] = {0xff, 0x01, 0x00};
    D3D12_RAYTRACING_INSTANCE_DESC instances[3]{};
    for (int i = 0; i < 3; ++i) {
        std::memcpy(instances[i].Transform, xforms[i], sizeof xforms[i]);
        instances[i].InstanceID = 20 + i;
        instances[i].InstanceMask = masks[i];
        instances[i].AccelerationStructure = blas->GetGPUVirtualAddress();
    }
    const UINT64 instances_va = place(instances, sizeof instances, 16);
    upload->Unmap(0, nullptr);
    tlas_inputs.InstanceDescs = instances_va;

    // ---- record: BLAS, barrier, TLAS, barrier, copies to readback ----
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
        return fail("queue, allocator, list or fence");
    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC build{};
    build.Inputs = blas_inputs;
    build.DestAccelerationStructureData = blas->GetGPUVirtualAddress();
    build.ScratchAccelerationStructureData = scratch->GetGPUVirtualAddress();
    list->BuildRaytracingAccelerationStructure(&build, 0, nullptr);
    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
    barrier.UAV.pResource = nullptr;                         // every UAV: the BLAS and the scratch
    list->ResourceBarrier(1, &barrier);
    build.Inputs = tlas_inputs;
    build.DestAccelerationStructureData = tlas->GetGPUVirtualAddress();
    list->BuildRaytracingAccelerationStructure(&build, 0, nullptr);
    list->ResourceBarrier(1, &barrier);
    // The layer keeps buffers in one memory, so a copy out of an acceleration structure reads its bytes as they are
    // (D3D12 itself only allows CopyRaytracingAccelerationStructure; this is a test of the layer's records).
    list->CopyBufferRegion(blas_rb.p, 0, blas.p, 0, blas_info.ResultDataMaxSizeInBytes);
    list->CopyBufferRegion(tlas_rb.p, 0, tlas.p, 0, tlas_info.ResultDataMaxSizeInBytes);
    if (FAILED(hr = list->Close())) return fail("Close", hr);
    ID3D12CommandList* lists[] = {list.p};
    queue->ExecuteCommandLists(1, lists);
    if (FAILED(queue->Signal(fence.p, 1))) return fail("Signal");
    HANDLE event = CreateEventW(nullptr, FALSE, FALSE, nullptr);   // reused for the update below
    const bool done = event && SUCCEEDED(fence->SetEventOnCompletion(1, event)) &&
                      WaitForSingleObject(event, 10000) == WAIT_OBJECT_0;
    if (event) CloseHandle(event);
    if (!done) return fail("the builds did not complete");
    if (FAILED(device->GetDeviceRemovedReason())) return fail("device removed", device->GetDeviceRemovedReason());

    // ---- dump ----
    auto dump = [&](ID3D12Resource* rb, UINT64 size, const std::string& name, const std::string& json) {
        void* p = nullptr;
        if (FAILED(rb->Map(0, nullptr, &p))) return false;
        std::uint32_t units = 0;
        std::memcpy(&units, static_cast<char*>(p) + 4, 4);    // header totalSizeIn64BNodes
        const UINT64 used = (UINT64(units) + 1) * 64;   // the field does not count the 64-byte header
        const bool ok = used && used <= size && write_file(dir + "/" + name + ".bin", p, std::size_t(used)) &&
                        write_file(dir + "/" + name + ".json", json.data(), json.size());
        rb->Unmap(0, nullptr);
        return ok;
    };
    // The BLAS's triangles as the reference sees them, for the soup's current positions.
    auto triangles_json = [&](const std::vector<float>& soup_now) {
        std::string tris = "[";
        bool first = true;
        auto add_tri = [&](const float v[3][3], int geometry, UINT primitive) {
            tris += first ? "\n  [" : ",\n  [";
            first = false;
            for (int k = 0; k < 3; ++k)
                tris += (k ? ", [" : "[") + num(v[k][0]) + ", " + num(v[k][1]) + ", " + num(v[k][2]) + "]";
            tris += ", " + std::to_string(geometry) + ", " + std::to_string(primitive) + "]";
        };
        for (UINT t = 0; t < soup_tris; ++t) {
            float v[3][3];
            for (int k = 0; k < 9; ++k) v[k / 3][k % 3] = soup_now[9 * t + k];
            add_tri(v, 0, t);
        }
        for (UINT t = 0; t < indices.size() / 3; ++t) {
            float v[3][3];
            for (int k = 0; k < 3; ++k) {
                const float* p = &grid[4 * indices[3 * t + k]];
                for (int r = 0; r < 3; ++r)
                    v[k][r] = transform[4 * r] * p[0] + transform[4 * r + 1] * p[1] + transform[4 * r + 2] * p[2] +
                              transform[4 * r + 3];
            }
            add_tri(v, 1, t);
        }
        return tris + "]";
    };
    const std::string tris = triangles_json(soup);
    if (!dump(blas_rb.p, blas_info.ResultDataMaxSizeInBytes, "d3d12_blas",
              "{\"kind\": \"blas\", \"address\": " + std::to_string(blas->GetGPUVirtualAddress()) +
                  ", \"triangles\": " + tris + "}\n"))
        return fail("BLAS readback");
    std::string inst = "[";
    for (int i = 0; i < 3; ++i) {
        inst += i ? ", {\"transform\": [" : "{\"transform\": [";
        for (int k = 0; k < 12; ++k) inst += (k ? ", " : "") + num(xforms[i][k]);
        inst += "], \"mask\": " + std::to_string(masks[i]) + ", \"blas\": 0}";
    }
    inst += "]";
    if (!dump(tlas_rb.p, tlas_info.ResultDataMaxSizeInBytes, "d3d12_tlas",
              "{\"kind\": \"tlas\", \"address\": " + std::to_string(tlas->GetGPUVirtualAddress()) +
                  ", \"bias\": 0, \"blas\": [\"d3d12_blas\"], \"instances\": " + inst + "}\n"))
        return fail("TLAS readback");

    // ---- PERFORM_UPDATE in place: the soup moves (every vertex), the grid stays; the BLAS is refitted ----
    std::vector<float> moved = soup;
    std::uniform_real_distribution<float> nudge(-1.0f, 1.0f);
    for (std::size_t v = 0; v < moved.size(); v += 3) {
        moved[v] = moved[v] * 1.3f + nudge(rng);
        moved[v + 1] = moved[v + 1] * 0.7f + nudge(rng);
        moved[v + 2] += nudge(rng);
    }
    if (FAILED(upload->Map(0, nullptr, reinterpret_cast<void**>(&mapped)))) return fail("Map upload for the update");
    std::memcpy(mapped + (soup_va - upload->GetGPUVirtualAddress()), moved.data(), std::size_t(soup_bytes));
    upload->Unmap(0, nullptr);
    if (FAILED(allocator->Reset()) || FAILED(list->Reset(allocator.p, nullptr))) return fail("Reset for the update");
    build.Inputs = blas_inputs;
    build.Inputs.Flags |= D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PERFORM_UPDATE;
    build.DestAccelerationStructureData = blas->GetGPUVirtualAddress();
    build.SourceAccelerationStructureData = blas->GetGPUVirtualAddress();
    list->BuildRaytracingAccelerationStructure(&build, 0, nullptr);
    list->ResourceBarrier(1, &barrier);
    list->CopyBufferRegion(blas_rb.p, 0, blas.p, 0, blas_info.ResultDataMaxSizeInBytes);
    if (FAILED(hr = list->Close())) return fail("Close (update)", hr);
    queue->ExecuteCommandLists(1, lists);
    if (FAILED(queue->Signal(fence.p, 2))) return fail("Signal (update)");
    event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    const bool updated = event && SUCCEEDED(fence->SetEventOnCompletion(2, event)) &&
                         WaitForSingleObject(event, 10000) == WAIT_OBJECT_0;
    if (event) CloseHandle(event);
    if (!updated) return fail("the update did not complete");
    if (!dump(blas_rb.p, blas_info.ResultDataMaxSizeInBytes, "d3d12_blas_update",
              "{\"kind\": \"blas\", \"address\": " + std::to_string(blas->GetGPUVirtualAddress()) +
                  ", \"triangles\": " + triangles_json(moved) + "}\n"))
        return fail("BLAS readback after the update");

    // ---- postbuild info and compaction: EmitRaytracingAccelerationStructurePostbuildInfo for both structures, a
    // TLAS rebuild with a CURRENT_SIZE postbuild desc, then COMPACT the BLAS into a buffer of exactly its size ----
    auto submit = [&](UINT64 value) {
        if (FAILED(list->Close())) return false;
        queue->ExecuteCommandLists(1, lists);
        if (FAILED(queue->Signal(fence.p, value))) return false;
        HANDLE e = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        const bool ok = e && SUCCEEDED(fence->SetEventOnCompletion(value, e)) &&
                        WaitForSingleObject(e, 10000) == WAIT_OBJECT_0;
        if (e) CloseHandle(e);
        return ok;
    };
    Ref<ID3D12Resource> info, info_rb;
    if (FAILED(hr = buffer(device.p, D3D12_HEAP_TYPE_DEFAULT, 24, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, uav, info.put())) ||
        FAILED(hr = buffer(device.p, D3D12_HEAP_TYPE_READBACK, 24, D3D12_RESOURCE_STATE_COPY_DEST,
                           D3D12_RESOURCE_FLAG_NONE, info_rb.put())))
        return fail("postbuild buffers", hr);
    if (FAILED(allocator->Reset()) || FAILED(list->Reset(allocator.p, nullptr))) return fail("Reset for postbuild");
    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_DESC compacted_desc{
        info->GetGPUVirtualAddress(), D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_COMPACTED_SIZE};
    const D3D12_GPU_VIRTUAL_ADDRESS both[2] = {blas->GetGPUVirtualAddress(), tlas->GetGPUVirtualAddress()};
    list->EmitRaytracingAccelerationStructurePostbuildInfo(&compacted_desc, 2, both);
    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_DESC current_desc{
        info->GetGPUVirtualAddress() + 16, D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_CURRENT_SIZE};
    build = {};
    build.Inputs = tlas_inputs;
    build.DestAccelerationStructureData = tlas->GetGPUVirtualAddress();
    build.ScratchAccelerationStructureData = scratch->GetGPUVirtualAddress();
    list->BuildRaytracingAccelerationStructure(&build, 1, &current_desc);
    list->ResourceBarrier(1, &barrier);
    list->CopyBufferRegion(info_rb.p, 0, info.p, 0, 24);
    if (!submit(3)) return fail("postbuild submission");
    UINT64 sizes[3] = {};
    {
        void* p = nullptr;
        if (FAILED(info_rb->Map(0, nullptr, &p))) return fail("Map postbuild");
        std::memcpy(sizes, p, sizeof sizes);
        info_rb->Unmap(0, nullptr);
    }
    // the builds are exactly their prebuild size, so the compacted size is that size
    if (sizes[0] != blas_info.ResultDataMaxSizeInBytes || sizes[1] != tlas_info.ResultDataMaxSizeInBytes ||
        sizes[2] != tlas_info.ResultDataMaxSizeInBytes) {
        std::fprintf(stderr, "postbuild sizes %llu %llu %llu, prebuild %llu %llu\n",
                     static_cast<unsigned long long>(sizes[0]), static_cast<unsigned long long>(sizes[1]),
                     static_cast<unsigned long long>(sizes[2]),
                     static_cast<unsigned long long>(blas_info.ResultDataMaxSizeInBytes),
                     static_cast<unsigned long long>(tlas_info.ResultDataMaxSizeInBytes));
        return fail("postbuild sizes");
    }
    Ref<ID3D12Resource> compacted, compacted_rb;
    if (FAILED(hr = buffer(device.p, D3D12_HEAP_TYPE_DEFAULT, sizes[0], as_state, uav, compacted.put())) ||
        FAILED(hr = buffer(device.p, D3D12_HEAP_TYPE_READBACK, sizes[0], D3D12_RESOURCE_STATE_COPY_DEST,
                           D3D12_RESOURCE_FLAG_NONE, compacted_rb.put())))
        return fail("compacted buffers", hr);
    if (FAILED(allocator->Reset()) || FAILED(list->Reset(allocator.p, nullptr))) return fail("Reset for compaction");
    list->CopyRaytracingAccelerationStructure(compacted->GetGPUVirtualAddress(), blas->GetGPUVirtualAddress(),
                                              D3D12_RAYTRACING_ACCELERATION_STRUCTURE_COPY_MODE_COMPACT);
    list->ResourceBarrier(1, &barrier);
    list->CopyBufferRegion(compacted_rb.p, 0, compacted.p, 0, sizes[0]);
    if (!submit(4)) return fail("compaction submission");
    if (!dump(compacted_rb.p, sizes[0], "d3d12_blas_compacted",
              "{\"kind\": \"blas\", \"address\": " + std::to_string(compacted->GetGPUVirtualAddress()) +
                  ", \"triangles\": " + triangles_json(moved) + "}\n"))
        return fail("compacted BLAS readback");
    // ---- the cases below record each refusal and comparison and go on, so one run shows every result ----
    int failures = 0;
    auto expect = [&](bool ok, const std::string& what) {
        if (!ok) {
            std::fprintf(stderr, "FAIL: %s\n", what.c_str());
            ++failures;
        }
        return ok;
    };
    UINT64 fence_value = 4;
    // Closes, executes and waits; returns what ExecuteCommandLists did (a refused list submits nothing).
    auto execute = [&]() {
        if (FAILED(list->Close())) return leonida::ReplayResult::unsupported;
        queue->ExecuteCommandLists(1, lists);
        const leonida::ReplayResult result = rt::d3d12::last_execute_result(queue.p);
        ++fence_value;
        HANDLE e = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        const bool waited = e && SUCCEEDED(queue->Signal(fence.p, fence_value)) &&
                            SUCCEEDED(fence->SetEventOnCompletion(fence_value, e)) &&
                            WaitForSingleObject(e, 10000) == WAIT_OBJECT_0;
        if (e) CloseHandle(e);
        return waited ? result : leonida::ReplayResult::unsupported;
    };
    auto reset = [&]() { return SUCCEEDED(allocator->Reset()) && SUCCEEDED(list->Reset(allocator.p, nullptr)); };
    auto read_back = [&](ID3D12Resource* rb, UINT64 size) {
        std::vector<std::uint8_t> bytes(static_cast<std::size_t>(size));
        void* p = nullptr;
        if (SUCCEEDED(rb->Map(0, nullptr, &p))) {
            std::memcpy(bytes.data(), p, bytes.size());
            rb->Unmap(0, nullptr);
        }
        return bytes;
    };
    auto fill = [&](ID3D12Resource* rb, UINT64 size, int value) {
        void* p = nullptr;
        if (FAILED(rb->Map(0, nullptr, &p))) return false;
        std::memset(p, value, std::size_t(size));
        rb->Unmap(0, nullptr);
        return true;
    };
    // The bytes a structure uses, from its header (totalSizeIn64BNodes does not count the header).
    auto used_bytes = [](const std::vector<std::uint8_t>& bytes) {
        std::uint32_t units = 0;
        if (bytes.size() >= 8) std::memcpy(&units, bytes.data() + 4, 4);
        const std::size_t used = (std::size_t(units) + 1) * 64;
        return used <= bytes.size() ? std::vector<std::uint8_t>(bytes.begin(), bytes.begin() + used)
                                    : std::vector<std::uint8_t>{};
    };

    // ---- TLAS from D3D12_ELEMENTS_LAYOUT_ARRAY_OF_POINTERS ----
    // The three instances above, through an array of GPU addresses: descs 0 and 2 lie swapped in a second upload
    // buffer, and desc 1 in a default-heap buffer (an arena range) that a copy earlier in the same list fills. The
    // ARRAY build of the same descs, recorded in the same list, must give the same bytes. Every pointer array is
    // padded with zeros to 64 bytes per instance, so a builder that read the pointers as descs reads only bytes
    // that exist (and finds no BLAS address in them).
    {
        Ref<ID3D12Resource> pointer_upload, desc_buffer, tlas_p, tlas_a_rb, tlas_p_rb;
        const UINT64 pointer_upload_size = 1024;
        if (FAILED(hr = buffer(device.p, D3D12_HEAP_TYPE_UPLOAD, pointer_upload_size, D3D12_RESOURCE_STATE_GENERIC_READ,
                               D3D12_RESOURCE_FLAG_NONE, pointer_upload.put())) ||
            FAILED(hr = buffer(device.p, D3D12_HEAP_TYPE_DEFAULT, 64, D3D12_RESOURCE_STATE_COPY_DEST,
                               D3D12_RESOURCE_FLAG_NONE, desc_buffer.put())) ||
            FAILED(hr = buffer(device.p, D3D12_HEAP_TYPE_DEFAULT, tlas_info.ResultDataMaxSizeInBytes, as_state, uav,
                               tlas_p.put())) ||
            FAILED(hr = buffer(device.p, D3D12_HEAP_TYPE_READBACK, tlas_info.ResultDataMaxSizeInBytes,
                               D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_FLAG_NONE, tlas_a_rb.put())) ||
            FAILED(hr = buffer(device.p, D3D12_HEAP_TYPE_READBACK, tlas_info.ResultDataMaxSizeInBytes,
                               D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_FLAG_NONE, tlas_p_rb.put())))
            return fail("pointer-layout buffers", hr);
        const UINT64 pointer_va = pointer_upload->GetGPUVirtualAddress();
        // 0: desc 2, 64: desc 0, 128: the pointer array, 320: desc 1 (copied to desc_buffer), 384: an array with a
        // null pointer, 576: one whose last pointer runs past the end of its buffer, 768: one into no resource.
        const UINT64 good = 128, with_null = 384, past_end = 576, nowhere = 768;
        const D3D12_GPU_VIRTUAL_ADDRESS pointers[3] = {pointer_va + 64, desc_buffer->GetGPUVirtualAddress(),
                                                       pointer_va + 0};
        const D3D12_GPU_VIRTUAL_ADDRESS null_pointers[3] = {pointer_va + 64, 0, pointer_va + 0};
        const D3D12_GPU_VIRTUAL_ADDRESS cut_pointers[3] = {pointer_va + 64, desc_buffer->GetGPUVirtualAddress(),
                                                           pointer_va + pointer_upload_size - 32};
        // 0x40 is in the arena's reserved zero page, which no resource holds: harmless even if it were read
        const D3D12_GPU_VIRTUAL_ADDRESS lost_pointers[3] = {pointer_va + 64, 0x40, pointer_va + 0};
        {
            char* p = nullptr;
            if (FAILED(pointer_upload->Map(0, nullptr, reinterpret_cast<void**>(&p)))) return fail("Map pointer upload");
            std::memcpy(p + 0, &instances[2], 64);
            std::memcpy(p + 64, &instances[0], 64);
            std::memcpy(p + good, pointers, sizeof pointers);
            std::memcpy(p + 320, &instances[1], 64);
            std::memcpy(p + with_null, null_pointers, sizeof null_pointers);
            std::memcpy(p + past_end, cut_pointers, sizeof cut_pointers);
            std::memcpy(p + nowhere, lost_pointers, sizeof lost_pointers);
            pointer_upload->Unmap(0, nullptr);
        }
        D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS pointer_inputs = tlas_inputs;
        pointer_inputs.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY_OF_POINTERS;
        pointer_inputs.InstanceDescs = pointer_va + good;
        D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO pointer_info{};
        device->GetRaytracingAccelerationStructurePrebuildInfo(&pointer_inputs, &pointer_info);
        expect(pointer_info.ResultDataMaxSizeInBytes == tlas_info.ResultDataMaxSizeInBytes &&
                   pointer_info.ScratchDataSizeInBytes == tlas_info.ScratchDataSizeInBytes &&
                   pointer_info.UpdateScratchDataSizeInBytes == tlas_info.UpdateScratchDataSizeInBytes,
               "TLAS prebuild sizes differ between ARRAY and ARRAY_OF_POINTERS");

        if (!reset()) return fail("Reset for the pointer TLAS");
        list->CopyBufferRegion(desc_buffer.p, 0, pointer_upload.p, 320, 64);
        D3D12_RESOURCE_BARRIER to_read{};
        to_read.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        to_read.Transition.pResource = desc_buffer.p;
        to_read.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        to_read.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
        to_read.Transition.StateAfter = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
        list->ResourceBarrier(1, &to_read);
        build = {};
        build.Inputs = tlas_inputs;                                    // ARRAY: the descs in instance order
        build.DestAccelerationStructureData = tlas->GetGPUVirtualAddress();
        build.ScratchAccelerationStructureData = scratch->GetGPUVirtualAddress();
        list->BuildRaytracingAccelerationStructure(&build, 0, nullptr);
        list->ResourceBarrier(1, &barrier);
        build.Inputs = pointer_inputs;
        build.DestAccelerationStructureData = tlas_p->GetGPUVirtualAddress();
        list->BuildRaytracingAccelerationStructure(&build, 0, nullptr);
        list->ResourceBarrier(1, &barrier);
        list->CopyBufferRegion(tlas_a_rb.p, 0, tlas.p, 0, tlas_info.ResultDataMaxSizeInBytes);
        list->CopyBufferRegion(tlas_p_rb.p, 0, tlas_p.p, 0, tlas_info.ResultDataMaxSizeInBytes);
        const leonida::ReplayResult built = execute();
        if (expect(built == leonida::ReplayResult::success,
                   std::string("pointer TLAS list: ") + std::to_string(int(built)))) {
            const auto array_bytes = used_bytes(read_back(tlas_a_rb.p, tlas_info.ResultDataMaxSizeInBytes));
            const auto pointer_bytes = used_bytes(read_back(tlas_p_rb.p, tlas_info.ResultDataMaxSizeInBytes));
            expect(!array_bytes.empty() && array_bytes == pointer_bytes,
                   "the ARRAY_OF_POINTERS TLAS differs from the ARRAY TLAS of the same instances");
        }

        // A null pointer, a pointer whose 64 bytes run past its buffer, and one into no resource: the list is
        // refused before anything runs, so the readback after the build keeps the bytes the CPU wrote.
        const struct { UINT64 at; const char* what; } refused[] = {
            {with_null, "a null instance pointer"}, {past_end, "an instance pointer past its buffer's end"},
            {nowhere, "an instance pointer into no resource"}};
        for (const auto& r : refused) {
            if (!fill(tlas_p_rb.p, tlas_info.ResultDataMaxSizeInBytes, 0xa5) || !reset())
                return fail("Reset for a refused pointer TLAS");
            build.Inputs = pointer_inputs;
            build.Inputs.InstanceDescs = pointer_va + r.at;
            build.DestAccelerationStructureData = tlas_p->GetGPUVirtualAddress();
            list->BuildRaytracingAccelerationStructure(&build, 0, nullptr);
            list->ResourceBarrier(1, &barrier);
            list->CopyBufferRegion(tlas_p_rb.p, 0, tlas_p.p, 0, tlas_info.ResultDataMaxSizeInBytes);
            const leonida::ReplayResult result = execute();
            expect(result == leonida::ReplayResult::unresolved_resource,
                   std::string(r.what) + ": ExecuteCommandLists gave " + std::to_string(int(result)) +
                       ", not unresolved_resource");
            const auto after = read_back(tlas_p_rb.p, tlas_info.ResultDataMaxSizeInBytes);
            expect(std::all_of(after.begin(), after.end(), [](std::uint8_t b) { return b == 0xa5; }),
                   std::string(r.what) + ": the list ran");
        }
        std::printf("pointer TLAS: ARRAY_OF_POINTERS compared with ARRAY, three bad pointer arrays\n");
    }

    // ---- vertex formats ----
    // One scene in every vertex format D3D12 allows for triangles, with coordinates in {-1, 0, 1}: exact in float,
    // half and SNORM16 (+-32767; -32768 also reads as -1.0), so every format must give the R32G32B32_FLOAT build's
    // bytes. "plane" is a 2x2 grid of squares at z = 0 with R16 indices, built in all six formats (the two-component
    // ones read z = 0); "cube" is the surface of the cube [-1, 1]^3 with R32 indices, built in the three formats with
    // z. Some 16-bit buffers start 2 bytes past a 4-byte boundary with a stride of 2 mod 4, so the components fall in
    // both halves of the words; the A component and the bytes between vertices hold junk the build must ignore.
    {
        using V3 = std::array<float, 3>;
        std::vector<V3> plane_v, cube_v;
        std::vector<std::uint32_t> plane_i, cube_i;
        for (int j = 0; j < 3; ++j)
            for (int i = 0; i < 3; ++i) plane_v.push_back({float(i - 1), float(j - 1), 0.0f});
        for (std::uint32_t j = 0; j < 2; ++j)
            for (std::uint32_t i = 0; i < 2; ++i) {
                const std::uint32_t a = 3 * j + i;
                const std::uint32_t q[6] = {a, a + 1, a + 4, a, a + 4, a + 3};
                plane_i.insert(plane_i.end(), q, q + 6);
            }
        for (int z = -1; z <= 1; ++z)
            for (int y = -1; y <= 1; ++y)
                for (int x = -1; x <= 1; ++x) cube_v.push_back({float(x), float(y), float(z)});
        auto lattice = [](int x, int y, int z) { return std::uint32_t((x + 1) + 3 * (y + 1) + 9 * (z + 1)); };
        for (int axis = 0; axis < 3; ++axis)
            for (int side = -1; side <= 1; side += 2)
                for (int cv = -1; cv <= 0; ++cv)
                    for (int cu = -1; cu <= 0; ++cu) {
                        auto corner = [&](int du, int dv) {
                            int c[3];
                            c[axis] = side;
                            c[(axis + 1) % 3] = cu + du;
                            c[(axis + 2) % 3] = cv + dv;
                            return lattice(c[0], c[1], c[2]);
                        };
                        const std::uint32_t q[6] = {corner(0, 0), corner(1, 0), corner(1, 1), corner(0, 0), corner(1, 1), corner(0, 1)};
                        cube_i.insert(cube_i.end(), q, q + 6);
                    }

        struct Format { DXGI_FORMAT format; const char* name; UINT element; UINT components; bool half, snorm; };
        const Format r32x3{DXGI_FORMAT_R32G32B32_FLOAT, "R32G32B32_FLOAT", 12, 3, false, false};
        const Format r32x2{DXGI_FORMAT_R32G32_FLOAT, "R32G32_FLOAT", 8, 2, false, false};
        const Format f16x4{DXGI_FORMAT_R16G16B16A16_FLOAT, "R16G16B16A16_FLOAT", 8, 4, true, false};
        const Format f16x2{DXGI_FORMAT_R16G16_FLOAT, "R16G16_FLOAT", 4, 2, true, false};
        const Format s16x4{DXGI_FORMAT_R16G16B16A16_SNORM, "R16G16B16A16_SNORM", 8, 4, false, true};
        const Format s16x2{DXGI_FORMAT_R16G16_SNORM, "R16G16_SNORM", 4, 2, false, true};
        // minus_one: how SNORM writes -1.0 (0: -32767, 1: -32768, 2: alternating by vertex, -32768 at even ones)
        auto encode = [](const std::vector<V3>& verts, const Format& f, UINT stride, int minus_one) {
            std::vector<std::uint8_t> out(verts.size() * stride, 0xcd);
            for (std::size_t v = 0; v < verts.size(); ++v) {
                std::uint8_t* at = out.data() + v * stride;
                for (UINT c = 0; c < f.components; ++c) {
                    const float x = c < 3 ? verts[v][c] : 0.0f;
                    if (!f.half && !f.snorm) {
                        std::memcpy(at + 4 * c, &x, 4);
                    } else if (f.half) {
                        const std::uint16_t h = c == 3 ? 0x7e00 : x > 0 ? 0x3c00 : x < 0 ? 0xbc00 : 0x0000;  // A: NaN
                        std::memcpy(at + 2 * c, &h, 2);
                    } else {
                        const bool low = minus_one == 1 || (minus_one == 2 && v % 2 == 0);
                        const std::int16_t s = c == 3 ? std::int16_t(-12345)
                                                      : x > 0 ? 32767 : x < 0 ? (low ? -32768 : -32767) : 0;
                        std::memcpy(at + 2 * c, &s, 2);
                    }
                }
            }
            return out;
        };
        std::string hex_digits = "0123456789abcdef";
        auto hex = [&](const std::vector<std::uint8_t>& bytes) {
            std::string s;
            for (std::uint8_t b : bytes) {
                s += hex_digits[b >> 4];
                s += hex_digits[b & 15];
            }
            return s;
        };

        // the BLAS sizes of the larger scene; prebuild does not depend on the format
        auto inputs_for = [](D3D12_RAYTRACING_GEOMETRY_DESC* g) {
            D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS in{};
            in.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL;
            in.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
            in.NumDescs = 1;
            in.pGeometryDescs = g;
            return in;
        };
        D3D12_RAYTRACING_GEOMETRY_DESC sizing{};
        sizing.Type = D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES;
        sizing.Triangles.VertexFormat = DXGI_FORMAT_R32G32B32_FLOAT;
        sizing.Triangles.IndexFormat = DXGI_FORMAT_R32_UINT;
        sizing.Triangles.IndexCount = UINT(cube_i.size());
        sizing.Triangles.VertexCount = UINT(cube_v.size());
        auto sizing_inputs = inputs_for(&sizing);
        D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO fmt_info{};
        device->GetRaytracingAccelerationStructurePrebuildInfo(&sizing_inputs, &fmt_info);
        Ref<ID3D12Resource> fmt_upload, fmt_blas, fmt_rb;
        const UINT64 fmt_upload_size = 64 * 1024;
        if (FAILED(hr = buffer(device.p, D3D12_HEAP_TYPE_UPLOAD, fmt_upload_size, D3D12_RESOURCE_STATE_GENERIC_READ,
                               D3D12_RESOURCE_FLAG_NONE, fmt_upload.put())) ||
            FAILED(hr = buffer(device.p, D3D12_HEAP_TYPE_DEFAULT, fmt_info.ResultDataMaxSizeInBytes, as_state, uav,
                               fmt_blas.put())) ||
            FAILED(hr = buffer(device.p, D3D12_HEAP_TYPE_READBACK, fmt_info.ResultDataMaxSizeInBytes,
                               D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_FLAG_NONE, fmt_rb.put())))
            return fail("vertex-format buffers", hr);
        if (fmt_info.ScratchDataSizeInBytes > scratch_size) return fail("vertex-format scratch");
        char* fmt_mapped = nullptr;
        if (FAILED(fmt_upload->Map(0, nullptr, reinterpret_cast<void**>(&fmt_mapped)))) return fail("Map format upload");
        UINT64 fmt_at = 0;
        // `skew` bytes past a 4-byte boundary
        auto put = [&](const void* data, UINT64 bytes, UINT64 skew) {
            fmt_at = (fmt_at + 15) / 16 * 16 + skew;
            std::memcpy(fmt_mapped + fmt_at, data, std::size_t(bytes));
            const UINT64 va = fmt_upload->GetGPUVirtualAddress() + fmt_at;
            fmt_at += bytes;
            return va;
        };
        std::vector<std::uint16_t> plane_i16(plane_i.begin(), plane_i.end());
        const UINT64 plane_index_va = put(plane_i16.data(), plane_i16.size() * 2, 0);
        const UINT64 cube_index_va = put(cube_i.data(), cube_i.size() * 4, 0);

        // One build in its own list; the structure's bytes, or nothing when the list was refused.
        auto build_one = [&](const D3D12_RAYTRACING_GEOMETRY_DESC& g, leonida::ReplayResult& result) {
            D3D12_RAYTRACING_GEOMETRY_DESC geometry = g;
            auto in = inputs_for(&geometry);
            D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO info{};
            device->GetRaytracingAccelerationStructurePrebuildInfo(&in, &info);
            std::vector<std::uint8_t> bytes;
            if (!fill(fmt_rb.p, fmt_info.ResultDataMaxSizeInBytes, 0) || !reset()) return bytes;
            D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC b{};
            b.Inputs = in;
            b.DestAccelerationStructureData = fmt_blas->GetGPUVirtualAddress();
            b.ScratchAccelerationStructureData = scratch->GetGPUVirtualAddress();
            list->BuildRaytracingAccelerationStructure(&b, 0, nullptr);
            list->ResourceBarrier(1, &barrier);
            list->CopyBufferRegion(fmt_rb.p, 0, fmt_blas.p, 0, fmt_info.ResultDataMaxSizeInBytes);
            result = execute();
            if (result == leonida::ReplayResult::success)
                bytes = used_bytes(read_back(fmt_rb.p, fmt_info.ResultDataMaxSizeInBytes));
            return bytes;
        };
        struct Case { const Format* format; UINT stride; UINT64 skew; int minus_one; };
        struct Scene {
            const char* name;
            const std::vector<V3>* verts;
            const std::vector<std::uint32_t>* indices;
            bool u16;
            UINT64 index_va;
            std::vector<Case> cases;    // the first is the R32G32B32_FLOAT reference
        };
        const Scene scenes[] = {
            {"plane", &plane_v, &plane_i, true, plane_index_va,
             {{&r32x3, 12, 0, 0}, {&r32x2, 8, 0, 0}, {&f16x4, 10, 2, 0}, {&f16x2, 6, 2, 0}, {&s16x4, 8, 0, 1},
              {&s16x2, 6, 2, 2}}},
            {"cube", &cube_v, &cube_i, false, cube_index_va,
             {{&r32x3, 12, 0, 0}, {&f16x4, 8, 0, 0}, {&s16x4, 12, 2, 2}}},
        };
        for (const Scene& scene : scenes) {
            std::vector<std::uint8_t> reference;
            D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO reference_info{};
            for (const Case& fc : scene.cases) {
                const auto vertex_bytes = encode(*scene.verts, *fc.format, fc.stride, fc.minus_one);
                D3D12_RAYTRACING_GEOMETRY_DESC g{};
                g.Type = D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES;
                g.Flags = D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE;
                g.Triangles.VertexFormat = fc.format->format;
                g.Triangles.VertexCount = UINT(scene.verts->size());
                g.Triangles.VertexBuffer = {put(vertex_bytes.data(), vertex_bytes.size(), fc.skew), fc.stride};
                g.Triangles.IndexFormat = scene.u16 ? DXGI_FORMAT_R16_UINT : DXGI_FORMAT_R32_UINT;
                g.Triangles.IndexCount = UINT(scene.indices->size());
                g.Triangles.IndexBuffer = scene.index_va;
                auto in = inputs_for(&g);
                D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO case_info{};
                device->GetRaytracingAccelerationStructurePrebuildInfo(&in, &case_info);
                const std::string what = std::string(scene.name) + " in " + fc.format->name;
                if (&fc == &scene.cases[0]) reference_info = case_info;
                expect(case_info.ResultDataMaxSizeInBytes == reference_info.ResultDataMaxSizeInBytes &&
                           case_info.ScratchDataSizeInBytes == reference_info.ScratchDataSizeInBytes,
                       what + ": prebuild sizes differ from R32G32B32_FLOAT");
                leonida::ReplayResult result = leonida::ReplayResult::unsupported;
                const auto bytes = build_one(g, result);
                if (!expect(result == leonida::ReplayResult::success && !bytes.empty(),
                            what + ": the build was refused (" + std::to_string(int(result)) + ")"))
                    continue;
                if (&fc == &scene.cases[0]) reference = bytes;
                expect(bytes == reference, what + ": the records differ from the R32G32B32_FLOAT build");
                // the case, with the source vertices for tools/xdxr_bvh.py to decode itself
                std::string case_tris = "[";
                for (std::size_t t = 0; t < scene.indices->size() / 3; ++t) {
                    case_tris += t ? ",\n  [" : "\n  [";
                    for (int k = 0; k < 3; ++k) {
                        const V3& p = (*scene.verts)[(*scene.indices)[3 * t + k]];
                        case_tris += (k ? ", [" : "[") + num(p[0]) + ", " + num(p[1]) + ", " + num(p[2]) + "]";
                    }
                    case_tris += ", 0, " + std::to_string(t) + "]";
                }
                case_tris += "]";
                std::string idx = "[";
                for (std::size_t i = 0; i < scene.indices->size(); ++i)
                    idx += (i ? ", " : "") + std::to_string((*scene.indices)[i]);
                idx += "]";
                const std::string name = std::string("d3d12_fmt_") + scene.name + "_" + fc.format->name;
                const std::string json = "{\"kind\": \"blas\", \"address\": " +
                                         std::to_string(fmt_blas->GetGPUVirtualAddress()) + ", \"triangles\": " + case_tris +
                                         ", \"sources\": [{\"geometry\": 0, \"format\": \"" + fc.format->name +
                                         "\", \"stride\": " + std::to_string(fc.stride) + ", \"vertices\": \"" +
                                         hex(vertex_bytes) + "\", \"indices\": " + idx + "}]}\n";
                expect(write_file(dir + "/" + name + ".bin", bytes.data(), bytes.size()) &&
                           write_file(dir + "/" + name + ".json", json.data(), json.size()),
                       name + ": write");
            }
        }

        // Values that are not exact in every format: 40 triangles of random halves (denormals among them) and of
        // random SNORM16 values (-32768, -32767 and 32767 among them), each against the R32G32B32_FLOAT build of the
        // floats they stand for: the half's value, and SNORM's float(c / 32767.0) clamped to -1 (correctly rounded;
        // see tools/xdxr_bvh.py snorm16). Unindexed, stride 8, the A component random.
        {
            std::mt19937 vrng(5);
            const UINT soup_vertices = 40 * 3;
            std::vector<std::uint16_t> halves(4 * soup_vertices), snorms(4 * soup_vertices);
            for (std::size_t i = 0; i < halves.size(); ++i) {
                const std::uint32_t r = vrng();
                halves[i] = std::uint16_t((r & 0x8000u) | (((r >> 16) % 17u) << 10) | (r & 0x3ffu));   // |x| < 4
                snorms[i] = std::uint16_t(vrng());
            }
            snorms[0] = 0x8000;
            snorms[1] = 0x8001;
            snorms[2] = 0x7fff;
            auto half_value = [](std::uint16_t h) {
                const std::uint32_t sign = std::uint32_t(h & 0x8000u) << 16, e = (h >> 10) & 0x1fu, m = h & 0x3ffu;
                float f = 0;
                if (e) {
                    const std::uint32_t bits = sign | ((e + 112u) << 23) | (m << 13);
                    std::memcpy(&f, &bits, 4);
                    return f;
                }
                f = std::ldexp(float(m), -24);
                return sign ? -f : f;
            };
            auto snorm_value = [](std::uint16_t s) {
                return std::max(float(double(std::int16_t(s)) / 32767.0), -1.0f);
            };
            const struct { const Format* format; const std::vector<std::uint16_t>* bits; } soups[] = {
                {&f16x4, &halves}, {&s16x4, &snorms}};
            for (const auto& random_soup : soups) {
                std::vector<float> floats;
                std::vector<std::uint8_t> raw(random_soup.bits->size() * 2);
                std::memcpy(raw.data(), random_soup.bits->data(), raw.size());
                for (UINT v = 0; v < soup_vertices; ++v)
                    for (UINT k = 0; k < 3; ++k) {
                        const std::uint16_t b = (*random_soup.bits)[4 * v + k];
                        floats.push_back(random_soup.format->half ? half_value(b) : snorm_value(b));
                    }
                D3D12_RAYTRACING_GEOMETRY_DESC g{};
                g.Type = D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES;
                g.Flags = D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE;
                g.Triangles.VertexFormat = DXGI_FORMAT_R32G32B32_FLOAT;
                g.Triangles.VertexCount = soup_vertices;
                g.Triangles.VertexBuffer = {put(floats.data(), floats.size() * 4, 0), 12};
                const std::string what = std::string("random ") + random_soup.format->name;
                leonida::ReplayResult result = leonida::ReplayResult::unsupported;
                const auto reference = build_one(g, result);
                if (!expect(result == leonida::ReplayResult::success && !reference.empty(),
                            what + ": the R32G32B32_FLOAT build was refused"))
                    continue;
                g.Triangles.VertexFormat = random_soup.format->format;
                g.Triangles.VertexBuffer = {put(raw.data(), raw.size(), 0), 8};
                const auto bytes = build_one(g, result);
                if (!expect(result == leonida::ReplayResult::success && !bytes.empty(),
                            what + ": the build was refused (" + std::to_string(int(result)) + ")"))
                    continue;
                expect(bytes == reference, what + ": the records differ from the R32G32B32_FLOAT build");
                std::string random_tris = "[";
                for (UINT t = 0; t < soup_vertices / 3; ++t) {
                    random_tris += t ? ",\n  [" : "\n  [";
                    for (UINT k = 0; k < 3; ++k) {
                        const float* p = &floats[3 * (3 * t + k)];
                        random_tris += (k ? ", [" : "[") + num(p[0]) + ", " + num(p[1]) + ", " + num(p[2]) + "]";
                    }
                    random_tris += ", 0, " + std::to_string(t) + "]";
                }
                random_tris += "]";
                const std::string name = std::string("d3d12_fmt_random_") + random_soup.format->name;
                const std::string json = "{\"kind\": \"blas\", \"address\": " +
                                         std::to_string(fmt_blas->GetGPUVirtualAddress()) + ", \"triangles\": " +
                                         random_tris + ", \"sources\": [{\"geometry\": 0, \"format\": \"" +
                                         random_soup.format->name + "\", \"stride\": 8, \"vertices\": \"" + hex(raw) +
                                         "\", \"indices\": null}]}\n";
                expect(write_file(dir + "/" + name + ".bin", bytes.data(), bytes.size()) &&
                           write_file(dir + "/" + name + ".json", json.data(), json.size()),
                       name + ": write");
            }
        }

        // Bad strides and addresses for the 16-bit formats: below the element size, odd, or an odd address.
        const struct { const Format* format; UINT stride; UINT64 skew; const char* what; } bad[] = {
            {&f16x4, 6, 0, "R16G16B16A16_FLOAT with stride 6"}, {&f16x2, 2, 0, "R16G16_FLOAT with stride 2"},
            {&s16x2, 5, 0, "R16G16_SNORM with stride 5"}, {&s16x4, 8, 1, "R16G16B16A16_SNORM at an odd address"}};
        for (const auto& b : bad) {
            const auto vertex_bytes = encode(plane_v, *b.format, b.stride < b.format->element ? b.format->element : b.stride, 0);
            D3D12_RAYTRACING_GEOMETRY_DESC g{};
            g.Type = D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES;
            g.Triangles.VertexFormat = b.format->format;
            g.Triangles.VertexCount = UINT(plane_v.size());
            g.Triangles.VertexBuffer = {put(vertex_bytes.data(), vertex_bytes.size(), b.skew), b.stride};
            g.Triangles.IndexFormat = DXGI_FORMAT_R16_UINT;
            g.Triangles.IndexCount = UINT(plane_i.size());
            g.Triangles.IndexBuffer = plane_index_va;
            leonida::ReplayResult result = leonida::ReplayResult::success;
            build_one(g, result);
            expect(result == leonida::ReplayResult::unsupported,
                   std::string(b.what) + ": gave " + std::to_string(int(result)) + ", not unsupported");
        }
        fmt_upload->Unmap(0, nullptr);
        std::printf("vertex formats: plane in 6 formats, cube in 3, 4 refused strides or addresses\n");
    }

    std::printf("d3d12_xdxr_rt: BLAS %llu bytes (prebuild %llu), TLAS (prebuild %llu), written to %s\n",
                static_cast<unsigned long long>(blas_info.ResultDataMaxSizeInBytes),
                static_cast<unsigned long long>(blas_info.ResultDataMaxSizeInBytes),
                static_cast<unsigned long long>(tlas_info.ResultDataMaxSizeInBytes), dir.c_str());
    return failures ? 1 : 0;
}
