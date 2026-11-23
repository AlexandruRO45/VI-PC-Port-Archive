// Lanterns stage 11 (lanterns_rt_gpu.h).
#include "lanterns_rt_gpu.h"

#include "lanterns_d3d12.h"

#include <cstring>

using namespace d3d12test;

namespace lanterns {
namespace {

constexpr std::uint64_t kPixelBytes = std::uint64_t(kWidth) * kHeight * 4;
constexpr unsigned kRtTile = 8;

}  // namespace

bool load_rt_shader(const std::filesystem::path& dir, std::vector<std::uint8_t>& out, const char* name) {
    out = read_file(dir / name);
    CHECK(out.size() > 64);
    return true;
}

bool RtShadows::create(Gpu& gpu, const std::vector<std::uint8_t>& shader, const Scene& scene, unsigned masked, bool opaque) {
    opaque_ = opaque;
    gpu_ = &gpu;
    device_ = gpu.device();
    queue_ = gpu.queue();
    CHECK(device_ && queue_ && gpu.gbuffer_normal());
    if (FAILED(device_->QueryInterface(IID_PPV_ARGS(&device5_.p))) || !device5_.p) {
        std::fprintf(stderr, "lanterns: the device has no ID3D12Device5 (ray tracing calls)\n");
        return false;
    }
    // [0] a table (t0 albedo, t1 normal, t2 view depth, t3 the TLAS as a raw buffer, u0 pixels, u1 steps) -> s[0:1];
    // [1] 16 constants -> s2..s17.
    const D3D12_DESCRIPTOR_RANGE ranges[2] = {{D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 4, 0, 0, 0},
                                              {D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 2, 0, 0, 4}};
    D3D12_ROOT_PARAMETER params[2] = {};
    params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    params[0].DescriptorTable = {2, ranges};
    params[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    params[1].Constants = {0, 0, 16};
    D3D12_VERSIONED_ROOT_SIGNATURE_DESC desc{};
    desc.Version = D3D_ROOT_SIGNATURE_VERSION_1_0;
    desc.Desc_1_0 = {2, params, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_NONE};
    const auto blob = serialize_root_signature(desc);
    CHECK(!blob.empty());
    CHECK(SUCCEEDED(device_->CreateRootSignature(0, blob.data(), blob.size(), IID_PPV_ARGS(&root_.p))));
    D3D12_COMPUTE_PIPELINE_STATE_DESC cs{};
    cs.pRootSignature = root_.p;
    cs.CS = {shader.data(), shader.size()};
    CHECK(SUCCEEDED(device_->CreateComputePipelineState(&cs, IID_PPV_ARGS(&pso_.p))));
    CHECK(SUCCEEDED(device_->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator_.p))));
    if (FAILED(device_->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator_.p, nullptr,
                                          IID_PPV_ARGS(&list_.p))) || !list_.p) {
        std::fprintf(stderr, "lanterns: the device gives no ID3D12GraphicsCommandList4\n");
        return false;
    }
    CHECK(list_->Close() == S_OK);
    CHECK(SUCCEEDED(device_->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence_.p))));

    const D3D12_HEAP_PROPERTIES default_heap{D3D12_HEAP_TYPE_DEFAULT, {}, {}, 0, 0};
    const D3D12_HEAP_PROPERTIES readback_heap{D3D12_HEAP_TYPE_READBACK, {}, {}, 0, 0};
    const auto make = [&](const D3D12_HEAP_PROPERTIES& heap, std::uint64_t bytes, D3D12_RESOURCE_STATES state,
                          Ref<ID3D12Resource>& out, D3D12_RESOURCE_FLAGS flags = D3D12_RESOURCE_FLAG_NONE) {
        const auto d = buffer_desc(bytes, flags);
        return SUCCEEDED(device_->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &d, state, nullptr,
                                                          IID_PPV_ARGS(&out.p)));
    };
    const auto uav = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    CHECK(make(default_heap, kPixelBytes, D3D12_RESOURCE_STATE_COMMON, output_, uav));
    CHECK(make(default_heap, kPixelBytes, D3D12_RESOURCE_STATE_COMMON, steps_, uav));
    CHECK(make(readback_heap, 2 * kPixelBytes, D3D12_RESOURCE_STATE_COPY_DEST, readback_));
    if (!build(scene, masked)) return false;
    // The faked stalls, two "TLAS"es of one BOX32 root node each (unit 1, type 5: pointer 13), 256 bytes apart:
    //   at 0, the stall: the root lists itself as its only child (the other three slots invalid, with empty boxes), so
    //     each step pops one entry and pushes one; only the shader's step cap ends it, and the stack never grows;
    //   at 256, the overflow: the root lists itself four times, so each step pops one and pushes four. The 32-entry
    //     stack is full after ten steps and the eleventh must stop the lane as overflowed instead of writing past it.
    // They are read through a V# and as a BVH, so they live in a DEFAULT buffer, filled once from an UPLOAD one (a V#
    // over an UPLOAD heap is outside what the layer maps for the guest; see docs/d3d12_vulkan.md).
    {
        const D3D12_HEAP_PROPERTIES upload_heap{D3D12_HEAP_TYPE_UPLOAD, {}, {}, 0, 0};
        CHECK(make(default_heap, 512, D3D12_RESOURCE_STATE_COMMON, stall_));
        CHECK(stall_->GetGPUVirtualAddress() % 256 == 0);
        Ref<ID3D12Resource> staging;   // kept alive until the copy's fence has passed
        CHECK(make(upload_heap, 512, D3D12_RESOURCE_STATE_GENERIC_READ, staging));
        std::uint32_t words[128] = {};
        for (unsigned fixture = 0; fixture < 2; ++fixture) {
            std::uint32_t* w = words + 64 * fixture;
            w[0] = 0x00000006u;   // a top-level XDXR BVH4, version 6
            w[1] = 3;             // 64-byte units after the header
            w[12] = 13;           // rootNodePtr
            const unsigned children = fixture ? 4 : 1;
            for (unsigned k = 0; k < 4; ++k) {
                w[16 + k] = k < children ? 13u : 0xffffffffu;
                for (int c = 0; c < 6; ++c) {
                    const bool hit = k < children;
                    const float v = c < 3 ? (hit ? -1e30f : 1e30f) : (hit ? 1e30f : -1e30f);
                    std::memcpy(&w[20 + 6 * k + c], &v, 4);
                }
            }
        }
        void* mapped = nullptr;
        CHECK(SUCCEEDED(staging->Map(0, nullptr, &mapped)));
        std::memcpy(mapped, words, sizeof(words));
        staging->Unmap(0, nullptr);
        CHECK(SUCCEEDED(allocator_->Reset()));
        CHECK(SUCCEEDED(list_->Reset(allocator_.p, nullptr)));
        const auto to_dest = transition(stall_.p, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST);
        list_->ResourceBarrier(1, &to_dest);
        list_->CopyBufferRegion(stall_.p, 0, staging.p, 0, sizeof(words));
        const auto to_read = transition(stall_.p, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        list_->ResourceBarrier(1, &to_read);
        CHECK(list_->Close() == S_OK);
        ID3D12CommandList* lists[] = {list_.p};
        queue_->ExecuteCommandLists(1, lists);
        CHECK(SUCCEEDED(queue_->Signal(fence_.p, ++fence_value_)));
        CHECK(wait_fence(fence_, fence_value_));
    }
    // Poison: copied over the output and the step counts before every pass, so a lane that writes nothing shows.
    {
        const D3D12_HEAP_PROPERTIES upload_heap{D3D12_HEAP_TYPE_UPLOAD, {}, {}, 0, 0};
        const auto d = buffer_desc(kPixelBytes);
        CHECK(SUCCEEDED(device_->CreateCommittedResource(&upload_heap, D3D12_HEAP_FLAG_NONE, &d,
                                                         D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
                                                         IID_PPV_ARGS(&poison_.p))));
        void* mapped = nullptr;
        CHECK(SUCCEEDED(poison_->Map(0, nullptr, &mapped)));
        auto* p = static_cast<std::uint32_t*>(mapped);
        for (std::size_t i = 0; i < kPixelBytes / 4; ++i) p[i] = kRtPoison;
        poison_->Unmap(0, nullptr);
    }

    const D3D12_DESCRIPTOR_HEAP_DESC heap_desc{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 6,
                                               D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, 0};
    CHECK(SUCCEEDED(device_->CreateDescriptorHeap(&heap_desc, IID_PPV_ARGS(&heap_.p))));
    heap_cpu_ = heap_cpu(heap_.p);
    heap_gpu_ = heap_gpu(heap_.p);
    CHECK(device_->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV) == 32);
    const auto texture_view = [&](ID3D12Resource* texture, DXGI_FORMAT format, unsigned slot) {
        D3D12_SHADER_RESOURCE_VIEW_DESC srv{};
        srv.Format = format;
        srv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
        srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        srv.Texture2D.MipLevels = 1;
        device_->CreateShaderResourceView(texture, &srv, D3D12_CPU_DESCRIPTOR_HANDLE{SIZE_T(heap_cpu_ + slot * 32)});
    };
    texture_view(gpu.target(), DXGI_FORMAT_R8G8B8A8_UNORM, 0);
    texture_view(gpu.gbuffer_normal(), kGbufferNormalFormat, 1);
    texture_view(gpu.gbuffer_depth(), kGbufferDepthFormat, 2);
    const std::array<std::uint32_t, 4> outs[2] = {make_raw(output_->GetGPUVirtualAddress(), std::uint32_t(kPixelBytes)),
                                                  make_raw(steps_->GetGPUVirtualAddress(), std::uint32_t(kPixelBytes))};
    for (unsigned s = 0; s < 2; ++s) std::memcpy(reinterpret_cast<void*>(heap_cpu_ + (4 + s) * 32), outs[s].data(), 16);
    return true;
}

// The two BLASes (one per traced mesh), then the three TLASes, in one list: BLAS, BLAS, UAV barrier, then each TLAS
// with a UAV barrier after it (the scratch is shared).
bool RtShadows::build(const Scene& scene, unsigned masked) {
    const auto transforms = rt_transforms(scene);
    CHECK(masked < transforms.size() && kRtSwapA < transforms.size() && kRtSwapB < transforms.size());
    std::vector<std::uint16_t> indices[kRtMeshes];
    std::uint64_t vb_off[kRtMeshes], ib_off[kRtMeshes], at = 0;
    for (unsigned m = 0; m < kRtMeshes; ++m) {
        indices[m] = rt_indices(scene.meshes[m]);
        vb_off[m] = at;
        ib_off[m] = align(at + scene.meshes[m].vertices.size() * sizeof(Vertex), 256);
        at = align(ib_off[m] + indices[m].size() * 2, 256);
    }
    const std::uint64_t inst_off = at, inst_bytes = transforms.size() * sizeof(D3D12_RAYTRACING_INSTANCE_DESC);
    const std::uint64_t up_bytes = inst_off + kRtTlases * align(inst_bytes, 256);
    const D3D12_HEAP_PROPERTIES upload_heap{D3D12_HEAP_TYPE_UPLOAD, {}, {}, 0, 0};
    const D3D12_HEAP_PROPERTIES default_heap{D3D12_HEAP_TYPE_DEFAULT, {}, {}, 0, 0};
    {
        const auto d = buffer_desc(up_bytes);
        CHECK(SUCCEEDED(device_->CreateCommittedResource(&upload_heap, D3D12_HEAP_FLAG_NONE, &d,
                                                         D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
                                                         IID_PPV_ARGS(&geometry_.p))));
    }
    D3D12_RAYTRACING_GEOMETRY_DESC geometry[kRtMeshes]{};
    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS blas_inputs[kRtMeshes]{};
    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO blas_info[kRtMeshes]{}, tlas_info{};
    std::uint64_t scratch = 0;
    for (unsigned m = 0; m < kRtMeshes; ++m) {
        geometry[m].Type = D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES;
        geometry[m].Flags = opaque_ ? D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE : D3D12_RAYTRACING_GEOMETRY_FLAG_NONE;
        geometry[m].Triangles.VertexFormat = DXGI_FORMAT_R32G32B32_FLOAT;
        geometry[m].Triangles.VertexCount = UINT(scene.meshes[m].vertices.size());
        geometry[m].Triangles.VertexBuffer = {geometry_->GetGPUVirtualAddress() + vb_off[m], sizeof(Vertex)};
        geometry[m].Triangles.IndexFormat = DXGI_FORMAT_R16_UINT;
        geometry[m].Triangles.IndexCount = UINT(indices[m].size());
        geometry[m].Triangles.IndexBuffer = geometry_->GetGPUVirtualAddress() + ib_off[m];
        blas_inputs[m].Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL;
        blas_inputs[m].Flags = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE;
        blas_inputs[m].DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
        blas_inputs[m].NumDescs = 1;
        blas_inputs[m].pGeometryDescs = &geometry[m];
        device5_->GetRaytracingAccelerationStructurePrebuildInfo(&blas_inputs[m], &blas_info[m]);
        if (!blas_info[m].ResultDataMaxSizeInBytes) {
            std::fprintf(stderr, "lanterns: GetRaytracingAccelerationStructurePrebuildInfo returned no BLAS size\n");
            return false;
        }
        scratch = std::max(scratch, blas_info[m].ScratchDataSizeInBytes);
    }
    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS tlas_inputs{};
    tlas_inputs.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL;
    tlas_inputs.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
    tlas_inputs.NumDescs = UINT(transforms.size());
    device5_->GetRaytracingAccelerationStructurePrebuildInfo(&tlas_inputs, &tlas_info);
    if (!tlas_info.ResultDataMaxSizeInBytes) {
        std::fprintf(stderr, "lanterns: GetRaytracingAccelerationStructurePrebuildInfo returned no TLAS size\n");
        return false;
    }
    sizes_ = {blas_info[0].ResultDataMaxSizeInBytes + blas_info[1].ResultDataMaxSizeInBytes,
              tlas_info.ResultDataMaxSizeInBytes, std::max(scratch, tlas_info.ScratchDataSizeInBytes)};
    const auto as_state = D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE;
    const auto make_as = [&](std::uint64_t bytes, D3D12_RESOURCE_STATES state, Ref<ID3D12Resource>& out) {
        const auto d = buffer_desc(bytes, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        return SUCCEEDED(device_->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &d, state, nullptr,
                                                          IID_PPV_ARGS(&out.p))) &&
               out->GetGPUVirtualAddress() % 256 == 0;
    };
    for (unsigned m = 0; m < kRtMeshes; ++m) CHECK(make_as(blas_info[m].ResultDataMaxSizeInBytes, as_state, blas_[m]));
    for (unsigned set = 0; set < kRtTlases; ++set) CHECK(make_as(sizes_.tlas, as_state, tlas_[set]));
    CHECK(make_as(sizes_.scratch, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, scratch_));
    CHECK(blas_[0]->GetGPUVirtualAddress() != blas_[1]->GetGPUVirtualAddress());

    std::uint8_t* up = nullptr;
    CHECK(SUCCEEDED(geometry_->Map(0, nullptr, reinterpret_cast<void**>(&up))));
    for (unsigned m = 0; m < kRtMeshes; ++m) {
        std::memcpy(up + vb_off[m], scene.meshes[m].vertices.data(), scene.meshes[m].vertices.size() * sizeof(Vertex));
        std::memcpy(up + ib_off[m], indices[m].data(), indices[m].size() * 2);
    }
    // Set 0: as drawn; set 1: instance 'masked' has mask 0; set 2: kRtSwapA and kRtSwapB name each other's BLAS.
    for (unsigned set = 0; set < kRtTlases; ++set) {
        const auto meshes = rt_instance_meshes(scene, set == 2);
        for (std::size_t i = 0; i < transforms.size(); ++i) {
            D3D12_RAYTRACING_INSTANCE_DESC inst{};
            std::memcpy(inst.Transform, transforms[i].data(), sizeof(inst.Transform));
            inst.InstanceID = UINT(i);
            inst.InstanceMask = set == 1 && i == masked ? 0x00 : 0xff;
            inst.AccelerationStructure = blas_[meshes[i]]->GetGPUVirtualAddress();
            std::memcpy(up + inst_off + set * align(inst_bytes, 256) + i * sizeof(inst), &inst, sizeof(inst));
        }
    }
    geometry_->Unmap(0, nullptr);

    CHECK(SUCCEEDED(allocator_->Reset()));
    CHECK(SUCCEEDED(list_->Reset(allocator_.p, nullptr)));
    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC build{};
    build.ScratchAccelerationStructureData = scratch_->GetGPUVirtualAddress();
    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;   // every UAV: the structures and the scratch
    for (unsigned m = 0; m < kRtMeshes; ++m) {
        build.Inputs = blas_inputs[m];
        build.DestAccelerationStructureData = blas_[m]->GetGPUVirtualAddress();
        list_->BuildRaytracingAccelerationStructure(&build, 0, nullptr);
        list_->ResourceBarrier(1, &barrier);   // the scratch is shared
    }
    for (unsigned set = 0; set < kRtTlases; ++set) {
        tlas_inputs.InstanceDescs = geometry_->GetGPUVirtualAddress() + inst_off + set * align(inst_bytes, 256);
        build.Inputs = tlas_inputs;
        build.DestAccelerationStructureData = tlas_[set]->GetGPUVirtualAddress();
        list_->BuildRaytracingAccelerationStructure(&build, 0, nullptr);
        list_->ResourceBarrier(1, &barrier);
    }
    CHECK(list_->Close() == S_OK);
    ID3D12CommandList* lists[] = {list_.p};
    queue_->ExecuteCommandLists(1, lists);
    CHECK(SUCCEEDED(queue_->Signal(fence_.p, ++fence_value_)));
    CHECK(wait_fence(fence_, fence_value_));
    return true;
}

bool RtShadows::run(const Scene& scene, RtRun which, std::vector<std::uint32_t>& pixels,
                    std::vector<std::uint32_t>& steps) {
    const bool fake = which == RtRun::stall || which == RtRun::overflow;
    const D3D12_GPU_VIRTUAL_ADDRESS tlas = fake ? stall_->GetGPUVirtualAddress() + (which == RtRun::overflow ? 256 : 0)
                                         : tlas_[which == RtRun::masked ? 1 : which == RtRun::swapped ? 2 : 0]->GetGPUVirtualAddress();
    // Slot 3: the TLAS as a raw buffer (the instance records and the header's root pointer). The previous run is done.
    const auto raw = make_raw(tlas, fake ? 256u : std::uint32_t(sizes_.tlas));
    std::memcpy(reinterpret_cast<void*>(heap_cpu_ + 3 * 32), raw.data(), 16);
    std::array<float, 16> constants = deferred_constants(scene, 0);
    // s15 is unused since the shader takes each instance's BLAS from its record; 0 makes a shader that still read it fail.
    const std::uint32_t words[4] = {std::uint32_t(tlas >> 8), 0u, 0xffu, kRtStepCap};
    std::memcpy(&constants[12], words, sizeof(words));

    CHECK(SUCCEEDED(allocator_->Reset()));
    CHECK(SUCCEEDED(list_->Reset(allocator_.p, nullptr)));
    const auto copy_src = D3D12_RESOURCE_STATE_COPY_SOURCE, read = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
    const auto uav = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    const auto out_from = first_run_ ? D3D12_RESOURCE_STATE_COMMON : copy_src;
    first_run_ = false;
    const auto dest = D3D12_RESOURCE_STATE_COPY_DEST;
    const D3D12_RESOURCE_BARRIER poison[2] = {transition(output_.p, out_from, dest), transition(steps_.p, out_from, dest)};
    list_->ResourceBarrier(2, poison);
    list_->CopyBufferRegion(output_.p, 0, poison_.p, 0, kPixelBytes);
    list_->CopyBufferRegion(steps_.p, 0, poison_.p, 0, kPixelBytes);
    const D3D12_RESOURCE_BARRIER start[3] = {transition(gpu_->target(), copy_src, read),
                                             transition(output_.p, dest, uav), transition(steps_.p, dest, uav)};
    list_->ResourceBarrier(3, start);
    ID3D12DescriptorHeap* heaps[] = {heap_.p};
    list_->SetDescriptorHeaps(1, heaps);
    list_->SetComputeRootSignature(root_.p);
    list_->SetComputeRootDescriptorTable(0, D3D12_GPU_DESCRIPTOR_HANDLE{heap_gpu_});
    list_->SetComputeRoot32BitConstants(1, 16, constants.data(), 0);
    list_->SetPipelineState(pso_.p);
    list_->Dispatch(kWidth / kRtTile, kHeight / kRtTile, 1);
    const D3D12_RESOURCE_BARRIER end[3] = {transition(gpu_->target(), read, copy_src),
                                           transition(output_.p, uav, copy_src), transition(steps_.p, uav, copy_src)};
    list_->ResourceBarrier(3, end);
    list_->CopyBufferRegion(readback_.p, 0, output_.p, 0, kPixelBytes);
    list_->CopyBufferRegion(readback_.p, kPixelBytes, steps_.p, 0, kPixelBytes);
    CHECK(list_->Close() == S_OK);
    ID3D12CommandList* lists[] = {list_.p};
    queue_->ExecuteCommandLists(1, lists);
    CHECK(SUCCEEDED(queue_->Signal(fence_.p, ++fence_value_)));
    CHECK(wait_fence(fence_, fence_value_));
    void* mapped = nullptr;
    CHECK(SUCCEEDED(readback_->Map(0, nullptr, &mapped)));
    pixels.resize(std::size_t(kWidth) * kHeight);
    steps.resize(pixels.size());
    std::memcpy(pixels.data(), mapped, kPixelBytes);
    std::memcpy(steps.data(), static_cast<const std::uint8_t*>(mapped) + kPixelBytes, kPixelBytes);
    readback_->Unmap(0, nullptr);
    return true;
}

}  // namespace lanterns
