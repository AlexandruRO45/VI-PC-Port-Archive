// Lanterns stage 9: the tiled lighting pass (lanterns_deferred_gpu.h, cs_lanterns_tiled.asm).
#include "lanterns_deferred_gpu.h"

#include "lanterns_d3d12.h"

#include <cstring>

using namespace d3d12test;

namespace lanterns {
namespace {

constexpr std::uint64_t kPixelBytes = std::uint64_t(kWidth) * kHeight * 4, kTiles = kTilesX * kTilesY;
constexpr std::uint64_t kLightBytes = kLights * sizeof(PointLight), kCountBytes = kTiles * 4;
constexpr std::uint64_t kListBytes = kTiles * kLights * 4;   // the largest output: the poison source is this big

}  // namespace

bool load_tiled_shader(const std::filesystem::path& dir, std::vector<std::uint8_t>& out) {
    out = read_file(dir / "cs_lanterns_tiled.dxbc");
    CHECK(out.size() > 64);
    return true;
}

bool Deferred::create(Gpu& gpu, const std::vector<std::uint8_t>& shader) {
    gpu_ = &gpu;
    device_ = gpu.device();
    queue_ = gpu.queue();
    CHECK(device_ && queue_ && gpu.gbuffer_normal() && gpu.gbuffer_depth());
    // [0] a table (t0 albedo, t1 normal, t2 view depth, t3 lights, u0 pixels, u1 tile counts, u2 tile lists) -> s[0:1];
    // [1] 16 constants -> s2..s17.
    const D3D12_DESCRIPTOR_RANGE ranges[2] = {{D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 4, 0, 0, 0},
                                              {D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 3, 0, 0, 4}};
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

    const D3D12_HEAP_PROPERTIES default_heap{D3D12_HEAP_TYPE_DEFAULT, {}, {}, 0, 0};
    const D3D12_HEAP_PROPERTIES upload_heap{D3D12_HEAP_TYPE_UPLOAD, {}, {}, 0, 0};
    const D3D12_HEAP_PROPERTIES readback_heap{D3D12_HEAP_TYPE_READBACK, {}, {}, 0, 0};
    const auto make = [&](const D3D12_HEAP_PROPERTIES& heap, std::uint64_t bytes, D3D12_RESOURCE_STATES state,
                          Ref<ID3D12Resource>& out, D3D12_RESOURCE_FLAGS flags = D3D12_RESOURCE_FLAG_NONE) {
        const auto d = buffer_desc(bytes, flags);
        return SUCCEEDED(device_->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &d, state, nullptr,
                                                          IID_PPV_ARGS(&out.p)));
    };
    const auto uav = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    CHECK(make(upload_heap, kLightBytes, D3D12_RESOURCE_STATE_GENERIC_READ, lights_));   // read in place
    CHECK(make(default_heap, kPixelBytes, D3D12_RESOURCE_STATE_COMMON, output_, uav));
    CHECK(make(default_heap, kCountBytes, D3D12_RESOURCE_STATE_COMMON, counts_, uav));
    CHECK(make(default_heap, kListBytes, D3D12_RESOURCE_STATE_COMMON, lists_, uav));
    CHECK(make(upload_heap, kListBytes, D3D12_RESOURCE_STATE_GENERIC_READ, poison_));
    CHECK(make(readback_heap, kPixelBytes + kCountBytes + kListBytes, D3D12_RESOURCE_STATE_COPY_DEST, readback_));
    CHECK(make(readback_heap, 2 * kPixelBytes, D3D12_RESOURCE_STATE_COPY_DEST, gbuffer_readback_));
    {
        void* mapped = nullptr;
        CHECK(SUCCEEDED(poison_->Map(0, nullptr, &mapped)));
        auto* words = static_cast<std::uint32_t*>(mapped);
        for (std::uint64_t i = 0; i < kListBytes / 4; ++i) words[i] = kDeferredPoison;
        poison_->Unmap(0, nullptr);
    }

    const D3D12_DESCRIPTOR_HEAP_DESC heap_desc{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 7,
                                               D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, 0};
    CHECK(SUCCEEDED(device_->CreateDescriptorHeap(&heap_desc, IID_PPV_ARGS(&heap_.p))));
    const std::uint64_t cpu = heap_cpu(heap_.p);
    heap_gpu_ = heap_gpu(heap_.p);
    CHECK(device_->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV) == 32);
    const auto texture_view = [&](ID3D12Resource* texture, DXGI_FORMAT format, unsigned slot) {
        D3D12_SHADER_RESOURCE_VIEW_DESC srv{};
        srv.Format = format;
        srv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
        srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        srv.Texture2D.MipLevels = 1;
        device_->CreateShaderResourceView(texture, &srv, D3D12_CPU_DESCRIPTOR_HANDLE{SIZE_T(cpu + slot * 32)});
    };
    texture_view(gpu.target(), DXGI_FORMAT_R8G8B8A8_UNORM, 0);
    texture_view(gpu.gbuffer_normal(), kGbufferNormalFormat, 1);
    texture_view(gpu.gbuffer_depth(), kGbufferDepthFormat, 2);
    const std::array<std::uint32_t, 4> sharps[4] = {make_vsharp(lights_->GetGPUVirtualAddress(), sizeof(PointLight), kLights),
                                                    make_raw(output_->GetGPUVirtualAddress(), std::uint32_t(kPixelBytes)),
                                                    make_raw(counts_->GetGPUVirtualAddress(), std::uint32_t(kCountBytes)),
                                                    make_raw(lists_->GetGPUVirtualAddress(), std::uint32_t(kListBytes))};
    for (unsigned s = 0; s < 4; ++s) std::memcpy(reinterpret_cast<void*>(cpu + (3 + s) * 32), sharps[s].data(), 16);

    CHECK(SUCCEEDED(device_->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator_.p))));
    CHECK(SUCCEEDED(device_->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator_.p, nullptr,
                                               IID_PPV_ARGS(&list_.p))));
    CHECK(list_->Close() == S_OK);
    CHECK(SUCCEEDED(device_->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence_.p))));
    return true;
}

bool Deferred::run(const std::vector<PointLight>& lights, const std::array<float, 16>& constants,
                   std::vector<std::uint32_t>& pixels, std::vector<std::uint32_t>& tile_counts,
                   std::vector<std::uint32_t>& tile_lists) {
    CHECK(lights.size() == kLights);
    void* mapped = nullptr;
    CHECK(SUCCEEDED(lights_->Map(0, nullptr, &mapped)));
    std::memcpy(mapped, lights.data(), kLightBytes);
    lights_->Unmap(0, nullptr);

    CHECK(SUCCEEDED(allocator_->Reset()));
    CHECK(SUCCEEDED(list_->Reset(allocator_.p, nullptr)));
    const auto copy_src = D3D12_RESOURCE_STATE_COPY_SOURCE, read = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
    const auto uav = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    const auto out_from = first_run_ ? D3D12_RESOURCE_STATE_COMMON : copy_src;
    first_run_ = false;
    // Poison first: a run that skips a store leaves 0xdeadbeef, not the previous run's value.
    const auto dest = D3D12_RESOURCE_STATE_COPY_DEST;
    const D3D12_RESOURCE_BARRIER poison[3] = {transition(output_.p, out_from, dest), transition(counts_.p, out_from, dest),
                                              transition(lists_.p, out_from, dest)};
    list_->ResourceBarrier(3, poison);
    list_->CopyBufferRegion(output_.p, 0, poison_.p, 0, kPixelBytes);
    list_->CopyBufferRegion(counts_.p, 0, poison_.p, 0, kCountBytes);
    list_->CopyBufferRegion(lists_.p, 0, poison_.p, 0, kListBytes);
    // The albedo is the Gpu's target, which a frame leaves in COPY_SOURCE; it goes back there afterwards.
    const D3D12_RESOURCE_BARRIER start[4] = {transition(gpu_->target(), copy_src, read), transition(output_.p, dest, uav),
                                             transition(counts_.p, dest, uav), transition(lists_.p, dest, uav)};
    list_->ResourceBarrier(4, start);
    ID3D12DescriptorHeap* heaps[] = {heap_.p};
    list_->SetDescriptorHeaps(1, heaps);
    list_->SetComputeRootSignature(root_.p);
    list_->SetComputeRootDescriptorTable(0, D3D12_GPU_DESCRIPTOR_HANDLE{heap_gpu_});
    list_->SetComputeRoot32BitConstants(1, 16, constants.data(), 0);
    list_->SetPipelineState(pso_.p);
    list_->Dispatch(kTilesX, kTilesY, 1);
    const D3D12_RESOURCE_BARRIER end[4] = {transition(gpu_->target(), read, copy_src), transition(output_.p, uav, copy_src),
                                           transition(counts_.p, uav, copy_src), transition(lists_.p, uav, copy_src)};
    list_->ResourceBarrier(4, end);
    list_->CopyBufferRegion(readback_.p, 0, output_.p, 0, kPixelBytes);
    list_->CopyBufferRegion(readback_.p, kPixelBytes, counts_.p, 0, kCountBytes);
    list_->CopyBufferRegion(readback_.p, kPixelBytes + kCountBytes, lists_.p, 0, kListBytes);
    CHECK(list_->Close() == S_OK);
    ID3D12CommandList* lists[] = {list_.p};
    queue_->ExecuteCommandLists(1, lists);
    CHECK(SUCCEEDED(queue_->Signal(fence_.p, ++fence_value_)));
    CHECK(wait_fence(fence_, fence_value_));

    CHECK(SUCCEEDED(readback_->Map(0, nullptr, &mapped)));
    const auto* bytes = static_cast<const std::uint8_t*>(mapped);
    pixels.resize(std::size_t(kWidth) * kHeight);
    tile_counts.resize(kTiles);
    std::memcpy(pixels.data(), bytes, kPixelBytes);
    std::memcpy(tile_counts.data(), bytes + kPixelBytes, kCountBytes);
    tile_lists.resize(kTiles * kLights);
    std::memcpy(tile_lists.data(), bytes + kPixelBytes + kCountBytes, kListBytes);
    readback_->Unmap(0, nullptr);
    return true;
}

bool Deferred::read_gbuffer(std::vector<std::uint32_t>& normal, std::vector<float>& depth) {
    CHECK(SUCCEEDED(allocator_->Reset()));
    CHECK(SUCCEEDED(list_->Reset(allocator_.p, nullptr)));
    const auto read = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, copy_src = D3D12_RESOURCE_STATE_COPY_SOURCE;
    ID3D12Resource* targets[2] = {gpu_->gbuffer_normal(), gpu_->gbuffer_depth()};
    const DXGI_FORMAT formats[2] = {kGbufferNormalFormat, kGbufferDepthFormat};
    const D3D12_RESOURCE_BARRIER to_copy[2] = {transition(targets[0], read, copy_src), transition(targets[1], read, copy_src)};
    list_->ResourceBarrier(2, to_copy);
    for (unsigned k = 0; k < 2; ++k) {
        D3D12_TEXTURE_COPY_LOCATION dst{gbuffer_readback_.p, D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, {}};
        dst.PlacedFootprint = {k * kPixelBytes, {formats[k], kWidth, kHeight, 1, kWidth * 4}};
        D3D12_TEXTURE_COPY_LOCATION src{targets[k], D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, {}};
        src.SubresourceIndex = 0;
        list_->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
    }
    const D3D12_RESOURCE_BARRIER back[2] = {transition(targets[0], copy_src, read), transition(targets[1], copy_src, read)};
    list_->ResourceBarrier(2, back);
    CHECK(list_->Close() == S_OK);
    ID3D12CommandList* lists[] = {list_.p};
    queue_->ExecuteCommandLists(1, lists);
    CHECK(SUCCEEDED(queue_->Signal(fence_.p, ++fence_value_)));
    CHECK(wait_fence(fence_, fence_value_));
    void* mapped = nullptr;
    CHECK(SUCCEEDED(gbuffer_readback_->Map(0, nullptr, &mapped)));
    normal.resize(std::size_t(kWidth) * kHeight);
    depth.resize(normal.size());
    std::memcpy(normal.data(), mapped, kPixelBytes);
    std::memcpy(depth.data(), static_cast<const std::uint8_t*>(mapped) + kPixelBytes, kPixelBytes);
    gbuffer_readback_->Unmap(0, nullptr);
    return true;
}

}  // namespace lanterns
