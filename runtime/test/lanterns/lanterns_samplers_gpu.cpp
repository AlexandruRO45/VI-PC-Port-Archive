// Lanterns stage 20: samplers a shader builds from immediates (lanterns_samplers_gpu.h).
#include "lanterns_samplers_gpu.h"

#include "lanterns_d3d12.h"

#include <cstring>

using namespace d3d12test;

namespace lanterns {
namespace {

constexpr std::uint64_t kOutputBytes = kSamplerProbes * 16;
constexpr UINT kRowPitch = D3D12_TEXTURE_DATA_PITCH_ALIGNMENT;   // one 256-byte row per texel row of the colour texture

D3D12_STATIC_SAMPLER_DESC static_sampler(D3D12_FILTER filter, UINT reg, D3D12_COMPARISON_FUNC compare) {
    D3D12_STATIC_SAMPLER_DESC s{};
    s.Filter = filter;
    s.AddressU = s.AddressV = s.AddressW = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
    s.MaxAnisotropy = 1;
    s.ComparisonFunc = compare;
    s.BorderColor = D3D12_STATIC_BORDER_COLOR_TRANSPARENT_BLACK;
    s.MinLOD = s.MaxLOD = 0.0f;
    s.ShaderRegister = reg;
    s.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    return s;
}

D3D12_RESOURCE_DESC texture_desc(DXGI_FORMAT format, D3D12_RESOURCE_FLAGS flags) {
    D3D12_RESOURCE_DESC d{};
    d.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    d.Width = d.Height = kSamplerSize;
    d.DepthOrArraySize = 1;
    d.MipLevels = 1;
    d.Format = format;
    d.SampleDesc.Count = 1;
    d.Flags = flags;
    return d;
}

}  // namespace

bool load_samplers_shader(const std::filesystem::path& dir, std::vector<std::uint8_t>& out) {
    out = read_file(dir / "cs_lanterns_samplers.dxbc");
    CHECK(out.size() > 64);
    return true;
}

bool SamplerKernel::create(Gpu& gpu, const std::vector<std::uint8_t>& shader) {
    device_ = gpu.device();
    queue_ = gpu.queue();
    CHECK(device_ && queue_);
    // One table at s[0:1]: t0 the colour texture, t1 the depth texture, u0 the output. Three static samplers, which
    // the shader does not read from anywhere: it holds the same samplers as literals.
    const D3D12_DESCRIPTOR_RANGE ranges[2] = {{D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 2, 0, 0, 0},
                                              {D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 0, 0, 2}};
    D3D12_ROOT_PARAMETER param{};
    param.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    param.DescriptorTable = {2, ranges};
    const D3D12_STATIC_SAMPLER_DESC samplers[3] = {
        static_sampler(D3D12_FILTER_MIN_MAG_MIP_POINT, 0, D3D12_COMPARISON_FUNC_NEVER),
        static_sampler(D3D12_FILTER_MIN_MAG_LINEAR_MIP_POINT, 1, D3D12_COMPARISON_FUNC_NEVER),
        static_sampler(D3D12_FILTER_COMPARISON_MIN_MAG_MIP_POINT, 2, D3D12_COMPARISON_FUNC_LESS_EQUAL)};
    D3D12_VERSIONED_ROOT_SIGNATURE_DESC desc{};
    desc.Version = D3D_ROOT_SIGNATURE_VERSION_1_0;
    desc.Desc_1_0 = {1, &param, 3, samplers, D3D12_ROOT_SIGNATURE_FLAG_NONE};
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
    const auto make = [&](const D3D12_HEAP_PROPERTIES& heap, const D3D12_RESOURCE_DESC& d, D3D12_RESOURCE_STATES state,
                          const D3D12_CLEAR_VALUE* clear, Ref<ID3D12Resource>& out) {
        return SUCCEEDED(device_->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &d, state, clear,
                                                          IID_PPV_ARGS(&out.p)));
    };
    CHECK(make(default_heap, texture_desc(DXGI_FORMAT_R32_FLOAT, D3D12_RESOURCE_FLAG_NONE),
               D3D12_RESOURCE_STATE_COPY_DEST, nullptr, colour_));
    D3D12_CLEAR_VALUE far_depth{};
    far_depth.Format = DXGI_FORMAT_D32_FLOAT;
    far_depth.DepthStencil = {1.0f, 0};
    CHECK(make(default_heap, texture_desc(DXGI_FORMAT_R32_TYPELESS, D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL),
               D3D12_RESOURCE_STATE_DEPTH_WRITE, &far_depth, depth_));
    CHECK(make(upload_heap, buffer_desc(std::uint64_t(kRowPitch) * kSamplerSize), D3D12_RESOURCE_STATE_GENERIC_READ,
               nullptr, staging_));
    CHECK(make(default_heap, buffer_desc(kOutputBytes, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS),
               D3D12_RESOURCE_STATE_COMMON, nullptr, output_));
    CHECK(make(upload_heap, buffer_desc(kOutputBytes), D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, poison_));
    CHECK(make(readback_heap, buffer_desc(kOutputBytes), D3D12_RESOURCE_STATE_COPY_DEST, nullptr, readback_));
    std::uint8_t* mapped = nullptr;
    CHECK(SUCCEEDED(staging_->Map(0, nullptr, reinterpret_cast<void**>(&mapped))));
    for (unsigned y = 0; y < kSamplerSize; ++y)
        for (unsigned x = 0; x < kSamplerSize; ++x) {
            const float v = sampler_texel(int(x), int(y));
            std::memcpy(mapped + y * kRowPitch + x * 4, &v, 4);
        }
    staging_->Unmap(0, nullptr);
    CHECK(SUCCEEDED(poison_->Map(0, nullptr, reinterpret_cast<void**>(&mapped))));
    for (unsigned i = 0; i < kOutputBytes / 4; ++i) std::memcpy(mapped + i * 4, &kSamplerPoison, 4);
    poison_->Unmap(0, nullptr);

    const D3D12_DESCRIPTOR_HEAP_DESC heap_desc{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 3,
                                               D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, 0};
    CHECK(SUCCEEDED(device_->CreateDescriptorHeap(&heap_desc, IID_PPV_ARGS(&heap_.p))));
    const std::uint64_t cpu = heap_cpu(heap_.p);
    heap_gpu_ = heap_gpu(heap_.p);
    CHECK(device_->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV) == 32);
    D3D12_SHADER_RESOURCE_VIEW_DESC srv{};
    srv.Format = DXGI_FORMAT_R32_FLOAT;
    srv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    srv.Texture2D.MipLevels = 1;
    device_->CreateShaderResourceView(colour_.p, &srv, D3D12_CPU_DESCRIPTOR_HANDLE{SIZE_T(cpu)});
    device_->CreateShaderResourceView(depth_.p, &srv, D3D12_CPU_DESCRIPTOR_HANDLE{SIZE_T(cpu + 32)});
    const auto v = make_raw(output_->GetGPUVirtualAddress(), std::uint32_t(kOutputBytes));
    std::memcpy(reinterpret_cast<void*>(cpu + 64), v.data(), 16);
    const D3D12_DESCRIPTOR_HEAP_DESC dsv_desc{D3D12_DESCRIPTOR_HEAP_TYPE_DSV, 1, D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0};
    CHECK(SUCCEEDED(device_->CreateDescriptorHeap(&dsv_desc, IID_PPV_ARGS(&dsv_heap_.p))));
    D3D12_DEPTH_STENCIL_VIEW_DESC dsv{};
    dsv.Format = DXGI_FORMAT_D32_FLOAT;
    dsv.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2D;
    const D3D12_CPU_DESCRIPTOR_HANDLE dsv_handle{SIZE_T(heap_cpu(dsv_heap_.p))};
    device_->CreateDepthStencilView(depth_.p, &dsv, dsv_handle);

    CHECK(SUCCEEDED(device_->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator_.p))));
    CHECK(SUCCEEDED(device_->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator_.p, nullptr,
                                               IID_PPV_ARGS(&list_.p))));
    CHECK(SUCCEEDED(device_->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence_.p))));
    // The colour texture from the staging rows; the depth texture one quadrant per rectangle clear.
    D3D12_TEXTURE_COPY_LOCATION dst{colour_.p, D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, {}};
    dst.SubresourceIndex = 0;
    D3D12_TEXTURE_COPY_LOCATION src{staging_.p, D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, {}};
    src.PlacedFootprint = {0, {DXGI_FORMAT_R32_FLOAT, kSamplerSize, kSamplerSize, 1, kRowPitch}};
    list_->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
    for (unsigned q = 0; q < 4; ++q) {
        const LONG x = LONG(q & 1) * 2, y = LONG(q >> 1) * 2;
        const D3D12_RECT rect{x, y, x + 2, y + 2};
        list_->ClearDepthStencilView(dsv_handle, D3D12_CLEAR_FLAG_DEPTH, kSamplerDepth[q], 0, 1, &rect);
    }
    const D3D12_RESOURCE_BARRIER ready[] = {
        transition(colour_.p, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE),
        transition(depth_.p, D3D12_RESOURCE_STATE_DEPTH_WRITE, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE)};
    list_->ResourceBarrier(2, ready);
    return submit_and_wait();
}

bool SamplerKernel::submit_and_wait() {
    CHECK(list_->Close() == S_OK);
    ID3D12CommandList* lists[] = {list_.p};
    queue_->ExecuteCommandLists(1, lists);
    CHECK(SUCCEEDED(queue_->Signal(fence_.p, ++fence_value_)));
    CHECK(wait_fence(fence_, fence_value_));
    return true;
}

bool SamplerKernel::run(std::array<std::array<std::uint32_t, 4>, kSamplerProbes>& out) {
    CHECK(SUCCEEDED(allocator_->Reset()));
    CHECK(SUCCEEDED(list_->Reset(allocator_.p, nullptr)));
    const D3D12_RESOURCE_BARRIER to_dest =
        transition(output_.p, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST);
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
    list_->Dispatch(1, 1, 1);
    const D3D12_RESOURCE_BARRIER to_copy =
        transition(output_.p, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
    list_->ResourceBarrier(1, &to_copy);
    list_->CopyBufferRegion(readback_.p, 0, output_.p, 0, kOutputBytes);
    const D3D12_RESOURCE_BARRIER back =
        transition(output_.p, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON);
    list_->ResourceBarrier(1, &back);
    if (!submit_and_wait()) return false;
    void* mapped = nullptr;
    CHECK(SUCCEEDED(readback_->Map(0, nullptr, &mapped)));
    std::memcpy(out.data(), mapped, kOutputBytes);
    readback_->Unmap(0, nullptr);
    return true;
}

}  // namespace lanterns
