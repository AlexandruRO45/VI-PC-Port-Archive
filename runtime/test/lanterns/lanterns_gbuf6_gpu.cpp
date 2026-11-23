// Lanterns stage 22: a G-buffer of five or six targets through packed half exports, and its decode
// (lanterns_gbuf6_gpu.h).
#include "lanterns_gbuf6_gpu.h"

#include "lanterns_d3d12.h"

#include <cstring>

using namespace d3d12test;

namespace lanterns {
namespace {

// The ID target and the albedo target are named here so that a check can swap them for the formats it must refuse:
// a binary16 target in place of RGBA8 (the truncating decode then loses the IDs), UNORM in place of sRGB.
constexpr DXGI_FORMAT kIdFormat = DXGI_FORMAT_R8G8B8A8_UNORM;
constexpr DXGI_FORMAT kAlbedoFormat = DXGI_FORMAT_R8G8B8A8_UNORM_SRGB;
constexpr std::array<DXGI_FORMAT, kGbufTargets> kFormats = {
    kIdFormat, DXGI_FORMAT_R10G10B10A2_UNORM, DXGI_FORMAT_R16G16_FLOAT, kAlbedoFormat,
    DXGI_FORMAT_R16G16B16A16_FLOAT, DXGI_FORMAT_R8G8B8A8_UNORM};
constexpr std::uint32_t kPixels = kWidth * kHeight;
constexpr std::uint64_t kDecodedBytes = std::uint64_t(kPixels) * 32;
constexpr unsigned kOutputSlot = kGbufTargets;   // descriptor slot of the decode output's raw V#

unsigned texel_bytes(DXGI_FORMAT f) { return f == DXGI_FORMAT_R16G16B16A16_FLOAT ? 8u : 4u; }
std::uint64_t pitch(DXGI_FORMAT f) { return align(std::uint64_t(kWidth) * texel_bytes(f), D3D12_TEXTURE_DATA_PITCH_ALIGNMENT); }

}  // namespace

bool load_gbuf6_shaders(const std::filesystem::path& dir, Gbuf6Shaders& out) {
    out.vs_full = read_file(dir / "vs_lanterns_full.dxbc");
    out.ps_gbuf6 = read_file(dir / "ps_lanterns_gbuf6.dxbc");
    out.cs_decode = read_file(dir / "cs_lanterns_gbuf_decode.dxbc");
    for (const auto* blob : {&out.vs_full, &out.ps_gbuf6, &out.cs_decode}) CHECK(blob->size() > 64);
    return true;
}

bool Gbuf6::create(Gpu& gpu, const Gbuf6Shaders& shaders) {
    device_ = gpu.device();
    queue_ = gpu.queue();
    CHECK(device_ && queue_);
    // Graphics: no root parameters (the pixel shader reads only its position). Compute: one table, t0..t5 and u0.
    D3D12_VERSIONED_ROOT_SIGNATURE_DESC desc{};
    desc.Version = D3D_ROOT_SIGNATURE_VERSION_1_0;
    desc.Desc_1_0 = {0, nullptr, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_NONE};
    auto blob = serialize_root_signature(desc);
    CHECK(!blob.empty());
    CHECK(SUCCEEDED(device_->CreateRootSignature(0, blob.data(), blob.size(), IID_PPV_ARGS(&graphics_root_.p))));
    const D3D12_DESCRIPTOR_RANGE ranges[2] = {{D3D12_DESCRIPTOR_RANGE_TYPE_SRV, kGbufTargets, 0, 0, 0},
                                              {D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 0, 0, kOutputSlot}};
    D3D12_ROOT_PARAMETER param{};
    param.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    param.DescriptorTable = {2, ranges};
    desc.Desc_1_0 = {1, &param, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_NONE};
    blob = serialize_root_signature(desc);
    CHECK(!blob.empty());
    CHECK(SUCCEEDED(device_->CreateRootSignature(0, blob.data(), blob.size(), IID_PPV_ARGS(&compute_root_.p))));

    D3D12_GRAPHICS_PIPELINE_STATE_DESC pso{};
    pso.pRootSignature = graphics_root_.p;
    pso.VS = {shaders.vs_full.data(), shaders.vs_full.size()};
    pso.PS = {shaders.ps_gbuf6.data(), shaders.ps_gbuf6.size()};
    for (auto& rt : pso.BlendState.RenderTarget) rt.RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
    pso.BlendState.IndependentBlendEnable = TRUE;
    pso.SampleMask = 0xffffffff;
    pso.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
    pso.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
    pso.RasterizerState.DepthClipEnable = TRUE;
    pso.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    pso.SampleDesc.Count = 1;
    for (unsigned t = 0; t < kGbufTargets; ++t) pso.RTVFormats[t] = kFormats[t];
    pso.NumRenderTargets = kGbufTargets;
    CHECK(SUCCEEDED(device_->CreateGraphicsPipelineState(&pso, IID_PPV_ARGS(&pso6_.p))));
    pso.NumRenderTargets = kGbufTargets - 1;
    pso.RTVFormats[kGbufTargets - 1] = DXGI_FORMAT_UNKNOWN;
    CHECK(SUCCEEDED(device_->CreateGraphicsPipelineState(&pso, IID_PPV_ARGS(&pso5_.p))));
    D3D12_COMPUTE_PIPELINE_STATE_DESC cs{};
    cs.pRootSignature = compute_root_.p;
    cs.CS = {shaders.cs_decode.data(), shaders.cs_decode.size()};
    CHECK(SUCCEEDED(device_->CreateComputePipelineState(&cs, IID_PPV_ARGS(&decode_.p))));

    const D3D12_HEAP_PROPERTIES default_heap{D3D12_HEAP_TYPE_DEFAULT, {}, {}, 0, 0};
    const D3D12_HEAP_PROPERTIES readback_heap{D3D12_HEAP_TYPE_READBACK, {}, {}, 0, 0};
    std::uint64_t readback_bytes = 0;
    for (unsigned t = 0; t < kGbufTargets; ++t) {
        D3D12_RESOURCE_DESC tex{};
        tex.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        tex.Width = kWidth;
        tex.Height = kHeight;
        tex.DepthOrArraySize = 1;
        tex.MipLevels = 1;
        tex.Format = kFormats[t];
        tex.SampleDesc.Count = 1;
        tex.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
        CHECK(SUCCEEDED(device_->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &tex,
                                                         D3D12_RESOURCE_STATE_COMMON, nullptr,
                                                         IID_PPV_ARGS(&targets_[t].p))));
        readback_offset_[t] = readback_bytes;
        readback_bytes += pitch(kFormats[t]) * kHeight;
    }
    decoded_offset_ = readback_bytes;
    readback_bytes += kDecodedBytes;
    const auto uav = buffer_desc(kDecodedBytes, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    CHECK(SUCCEEDED(device_->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &uav, D3D12_RESOURCE_STATE_COMMON,
                                                     nullptr, IID_PPV_ARGS(&output_.p))));
    const auto rb = buffer_desc(readback_bytes);
    CHECK(SUCCEEDED(device_->CreateCommittedResource(&readback_heap, D3D12_HEAP_FLAG_NONE, &rb,
                                                     D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&readback_.p))));

    const D3D12_DESCRIPTOR_HEAP_DESC heap_desc{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, kGbufTargets + 1,
                                               D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, 0};
    CHECK(SUCCEEDED(device_->CreateDescriptorHeap(&heap_desc, IID_PPV_ARGS(&heap_.p))));
    const std::uint64_t cpu = heap_cpu(heap_.p);
    heap_gpu_ = heap_gpu(heap_.p);
    CHECK(device_->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV) == 32);
    const D3D12_DESCRIPTOR_HEAP_DESC rtv_desc{D3D12_DESCRIPTOR_HEAP_TYPE_RTV, kGbufTargets, D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0};
    CHECK(SUCCEEDED(device_->CreateDescriptorHeap(&rtv_desc, IID_PPV_ARGS(&rtv_heap_.p))));
    rtv_cpu_ = heap_cpu(rtv_heap_.p);
    rtv_step_ = device_->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
    for (unsigned t = 0; t < kGbufTargets; ++t) {
        D3D12_SHADER_RESOURCE_VIEW_DESC srv{};
        srv.Format = kFormats[t];
        srv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
        srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        srv.Texture2D.MipLevels = 1;
        device_->CreateShaderResourceView(targets_[t].p, &srv, D3D12_CPU_DESCRIPTOR_HANDLE{SIZE_T(cpu + t * 32)});
        device_->CreateRenderTargetView(targets_[t].p, nullptr, D3D12_CPU_DESCRIPTOR_HANDLE{SIZE_T(rtv_cpu_ + t * rtv_step_)});
    }
    const auto v = make_raw(output_->GetGPUVirtualAddress(), std::uint32_t(kDecodedBytes));
    std::memcpy(reinterpret_cast<void*>(cpu + kOutputSlot * 32), v.data(), 16);
    CHECK(SUCCEEDED(device_->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator_.p))));
    CHECK(SUCCEEDED(device_->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator_.p, nullptr,
                                               IID_PPV_ARGS(&list_.p))));
    CHECK(list_->Close() == S_OK);
    CHECK(SUCCEEDED(device_->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence_.p))));
    return true;
}

bool Gbuf6::run(bool six, Gbuf6Result& out) {
    CHECK(SUCCEEDED(allocator_->Reset()));
    CHECK(SUCCEEDED(list_->Reset(allocator_.p, nullptr)));
    std::array<D3D12_RESOURCE_BARRIER, kGbufTargets + 1> to_rt{}, to_read{}, back{};
    constexpr auto kRead = D3D12_RESOURCE_STATES(D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_COPY_SOURCE);
    for (unsigned t = 0; t < kGbufTargets; ++t) {
        to_rt[t] = transition(targets_[t].p, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_RENDER_TARGET);
        to_read[t] = transition(targets_[t].p, D3D12_RESOURCE_STATE_RENDER_TARGET, kRead);
        back[t] = transition(targets_[t].p, kRead, D3D12_RESOURCE_STATE_COMMON);
    }
    to_rt[kGbufTargets] = transition(output_.p, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    to_read[kGbufTargets] = transition(output_.p, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
    back[kGbufTargets] = transition(output_.p, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON);
    list_->ResourceBarrier(UINT(to_rt.size()), to_rt.data());
    const float zero[4] = {0, 0, 0, 0};
    std::array<D3D12_CPU_DESCRIPTOR_HANDLE, kGbufTargets> rtvs{};
    for (unsigned t = 0; t < kGbufTargets; ++t) {
        rtvs[t] = {SIZE_T(rtv_cpu_ + t * rtv_step_)};
        list_->ClearRenderTargetView(rtvs[t], zero, 0, nullptr);
    }
    list_->SetGraphicsRootSignature(graphics_root_.p);
    list_->SetPipelineState(six ? pso6_.p : pso5_.p);
    const D3D12_VIEWPORT viewport{0, 0, float(kWidth), float(kHeight), 0, 1};
    const D3D12_RECT scissor{0, 0, LONG(kWidth), LONG(kHeight)};
    list_->RSSetViewports(1, &viewport);
    list_->RSSetScissorRects(1, &scissor);
    list_->OMSetRenderTargets(six ? kGbufTargets : kGbufTargets - 1, rtvs.data(), FALSE, nullptr);
    list_->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    list_->DrawInstanced(3, 1, 0, 0);
    list_->ResourceBarrier(kGbufTargets, to_read.data());
    ID3D12DescriptorHeap* heaps[] = {heap_.p};
    list_->SetDescriptorHeaps(1, heaps);
    list_->SetComputeRootSignature(compute_root_.p);
    list_->SetComputeRootDescriptorTable(0, D3D12_GPU_DESCRIPTOR_HANDLE{heap_gpu_});
    list_->SetPipelineState(decode_.p);
    list_->Dispatch(kWidth / 32, kHeight, 1);
    list_->ResourceBarrier(1, &to_read[kGbufTargets]);
    for (unsigned t = 0; t < kGbufTargets; ++t) {
        D3D12_TEXTURE_COPY_LOCATION dst{readback_.p, D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, {}};
        dst.PlacedFootprint = {readback_offset_[t], {kFormats[t], kWidth, kHeight, 1, UINT(pitch(kFormats[t]))}};
        D3D12_TEXTURE_COPY_LOCATION src{targets_[t].p, D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, {}};
        src.SubresourceIndex = 0;
        list_->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
    }
    list_->CopyBufferRegion(readback_.p, decoded_offset_, output_.p, 0, kDecodedBytes);
    list_->ResourceBarrier(UINT(back.size()), back.data());
    CHECK(list_->Close() == S_OK);
    ID3D12CommandList* lists[] = {list_.p};
    queue_->ExecuteCommandLists(1, lists);
    CHECK(SUCCEEDED(queue_->Signal(fence_.p, ++fence_value_)));
    CHECK(wait_fence(fence_, fence_value_));
    std::uint8_t* mapped = nullptr;
    CHECK(SUCCEEDED(readback_->Map(0, nullptr, reinterpret_cast<void**>(&mapped))));
    for (unsigned t = 0; t < kGbufTargets; ++t) {
        const unsigned bytes = texel_bytes(kFormats[t]);
        out.texel_bytes[t] = bytes;
        out.targets[t].resize(std::size_t(kPixels) * bytes);
        for (unsigned y = 0; y < kHeight; ++y)
            std::memcpy(out.targets[t].data() + std::size_t(y) * kWidth * bytes,
                        mapped + readback_offset_[t] + y * pitch(kFormats[t]), std::size_t(kWidth) * bytes);
    }
    out.decoded.resize(std::size_t(kPixels) * 8);
    std::memcpy(out.decoded.data(), mapped + decoded_offset_, kDecodedBytes);
    readback_->Unmap(0, nullptr);
    return true;
}

}  // namespace lanterns
