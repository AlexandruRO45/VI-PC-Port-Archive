// Lanterns stage 24: local X and local Y thread IDs (lanterns_xy_gpu.h).
#include "lanterns_xy_gpu.h"

#include "lanterns_d3d12.h"

#include <cstring>

using namespace d3d12test;

namespace lanterns {
namespace {

constexpr std::uint64_t kOutputBytes = std::uint64_t(kXyWidth) * kXyHeight * 8;

}  // namespace

bool load_xy_shader(const std::filesystem::path& dir, std::vector<std::uint8_t>& out) {
    out = read_file(dir / "cs_lanterns_xy.dxbc");
    CHECK(out.size() > 64);
    return true;
}

bool XyKernel::create(Gpu& gpu, const std::vector<std::uint8_t>& shader) {
    device_ = gpu.device();
    queue_ = gpu.queue();
    CHECK(device_ && queue_);
    const D3D12_DESCRIPTOR_RANGE range{D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 0, 0, 0};
    D3D12_ROOT_PARAMETER param{};
    param.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    param.DescriptorTable = {1, &range};
    D3D12_VERSIONED_ROOT_SIGNATURE_DESC desc{};
    desc.Version = D3D_ROOT_SIGNATURE_VERSION_1_0;
    desc.Desc_1_0 = {1, &param, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_NONE};
    const auto blob = serialize_root_signature(desc);
    CHECK(!blob.empty());
    CHECK(SUCCEEDED(device_->CreateRootSignature(0, blob.data(), blob.size(), IID_PPV_ARGS(&root_.p))));
    D3D12_COMPUTE_PIPELINE_STATE_DESC cs{};
    cs.pRootSignature = root_.p;
    cs.CS = {shader.data(), shader.size()};
    CHECK(SUCCEEDED(device_->CreateComputePipelineState(&cs, IID_PPV_ARGS(&pso_.p))));

    const D3D12_HEAP_PROPERTIES default_heap{D3D12_HEAP_TYPE_DEFAULT, {}, {}, 0, 0};
    const D3D12_HEAP_PROPERTIES upload_heap{D3D12_HEAP_TYPE_UPLOAD, {}, {}, 0, 0};
    const D3D12_HEAP_PROPERTIES readback_heap{D3D12_HEAP_TYPE_READBACK, {}, {}, 0, 0};
    const auto uav = buffer_desc(kOutputBytes, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS), plain = buffer_desc(kOutputBytes);
    CHECK(SUCCEEDED(device_->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &uav, D3D12_RESOURCE_STATE_COMMON,
                                                     nullptr, IID_PPV_ARGS(&output_.p))));
    CHECK(SUCCEEDED(device_->CreateCommittedResource(&upload_heap, D3D12_HEAP_FLAG_NONE, &plain,
                                                     D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&poison_.p))));
    CHECK(SUCCEEDED(device_->CreateCommittedResource(&readback_heap, D3D12_HEAP_FLAG_NONE, &plain,
                                                     D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&readback_.p))));
    void* mapped = nullptr;
    CHECK(SUCCEEDED(poison_->Map(0, nullptr, &mapped)));
    for (std::uint64_t i = 0; i < kOutputBytes / 4; ++i) static_cast<std::uint32_t*>(mapped)[i] = kXyPoison;
    poison_->Unmap(0, nullptr);
    const D3D12_DESCRIPTOR_HEAP_DESC heap_desc{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 1,
                                               D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, 0};
    CHECK(SUCCEEDED(device_->CreateDescriptorHeap(&heap_desc, IID_PPV_ARGS(&heap_.p))));
    heap_gpu_ = heap_gpu(heap_.p);
    const auto v = make_raw(output_->GetGPUVirtualAddress(), std::uint32_t(kOutputBytes));
    std::memcpy(reinterpret_cast<void*>(heap_cpu(heap_.p)), v.data(), 16);
    CHECK(SUCCEEDED(device_->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator_.p))));
    CHECK(SUCCEEDED(device_->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator_.p, nullptr,
                                               IID_PPV_ARGS(&list_.p))));
    CHECK(list_->Close() == S_OK);
    CHECK(SUCCEEDED(device_->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence_.p))));
    return true;
}

bool XyKernel::run(std::vector<std::uint32_t>& out) {
    CHECK(SUCCEEDED(allocator_->Reset()));
    CHECK(SUCCEEDED(list_->Reset(allocator_.p, nullptr)));
    const D3D12_RESOURCE_BARRIER to_dest = transition(output_.p, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST);
    list_->ResourceBarrier(1, &to_dest);
    list_->CopyBufferRegion(output_.p, 0, poison_.p, 0, kOutputBytes);
    const D3D12_RESOURCE_BARRIER to_uav =
        transition(output_.p, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    list_->ResourceBarrier(1, &to_uav);
    ID3D12DescriptorHeap* heaps[] = {heap_.p};
    list_->SetDescriptorHeaps(1, heaps);
    list_->SetComputeRootSignature(root_.p);
    list_->SetComputeRootDescriptorTable(0, D3D12_GPU_DESCRIPTOR_HANDLE{heap_gpu_});
    list_->SetPipelineState(pso_.p);
    list_->Dispatch(kXyGroupsX, kXyGroupsY, 1);
    const D3D12_RESOURCE_BARRIER to_copy =
        transition(output_.p, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
    list_->ResourceBarrier(1, &to_copy);
    list_->CopyBufferRegion(readback_.p, 0, output_.p, 0, kOutputBytes);
    const D3D12_RESOURCE_BARRIER back = transition(output_.p, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON);
    list_->ResourceBarrier(1, &back);
    CHECK(list_->Close() == S_OK);
    ID3D12CommandList* lists[] = {list_.p};
    queue_->ExecuteCommandLists(1, lists);
    CHECK(SUCCEEDED(queue_->Signal(fence_.p, ++fence_value_)));
    CHECK(wait_fence(fence_, fence_value_));
    void* mapped = nullptr;
    CHECK(SUCCEEDED(readback_->Map(0, nullptr, &mapped)));
    out.resize(kOutputBytes / 4);
    std::memcpy(out.data(), mapped, kOutputBytes);
    readback_->Unmap(0, nullptr);
    return true;
}

}  // namespace lanterns
