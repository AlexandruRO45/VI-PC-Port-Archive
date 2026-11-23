// Lanterns stage 5: the post-process on the GPU (lanterns_post_gpu.h). The texture SRVs go through
// CreateShaderResourceView, as test/d3d12_m2_textured_test.cpp (M2) makes its T#s; the buffers get raw V#s written
// through the CPU handle, as the rest of Lanterns does.
#include "lanterns_post_gpu.h"

#include "lanterns_d3d12.h"

#include <cstring>

using namespace d3d12test;

namespace lanterns {
namespace {

constexpr std::uint32_t kPixels = kWidth * kHeight;
constexpr std::uint64_t kBufferBytes = std::uint64_t(kPixels) * 16;   // one float4 per pixel
constexpr std::uint64_t kImageBytes = std::uint64_t(kPitch) * kHeight;
// Descriptor slots (32 bytes apart): three tables of two.
constexpr unsigned kRowsTable = 0;       // [scene T#, rows V#]       cs_lanterns_bloom_h
constexpr unsigned kColumnsTable = 2;    // [rows V#, bloom V#]       cs_lanterns_bloom_v
constexpr unsigned kCompositeTable = 4;  // [scene T#, bloom V#]      ps_lanterns_composite
constexpr unsigned kControlTable = 6;    // [scene T#, zero V#]       ps_lanterns_composite, skip_blur
constexpr unsigned kSlots = 8;

}  // namespace

bool load_post_shaders(const std::filesystem::path& dir, PostShaders& out) {
    out.bloom_h = read_file(dir / "cs_lanterns_bloom_h.dxbc");
    out.bloom_h_no_barrier = read_file(dir / "cs_lanterns_bloom_h_nobarrier.dxbc");
    out.bloom_v = read_file(dir / "cs_lanterns_bloom_v.dxbc");
    out.vs_full = read_file(dir / "vs_lanterns_full.dxbc");
    out.ps_composite = read_file(dir / "ps_lanterns_composite.dxbc");
    for (const auto* blob : {&out.bloom_h, &out.bloom_h_no_barrier, &out.bloom_v, &out.vs_full, &out.ps_composite})
        CHECK(blob->size() > 64);
    return true;
}

bool Post::create(Gpu& gpu, const PostShaders& shaders) {
    device_ = gpu.device();
    queue_ = gpu.queue();
    scene_ = gpu.target();
    CHECK(device_ && queue_ && scene_ && scene_->GetDesc().Format == DXGI_FORMAT_R16G16B16A16_FLOAT);
    CHECK(create_pipelines(shaders));
    CHECK(create_resources());
    CHECK(fill_descriptors());
    CHECK(SUCCEEDED(device_->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator_.p))));
    CHECK(SUCCEEDED(device_->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator_.p, nullptr,
                                               IID_PPV_ARGS(&list_.p))));
    CHECK(list_->Close() == S_OK);
    CHECK(SUCCEEDED(device_->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence_.p))));
    return true;
}

// One table of two descriptors, [0] t0 and [1] u0 for compute, t0 and t1 for graphics -> s[0:1] in every shader here.
static bool make_root(ID3D12Device* device, bool compute, Ref<ID3D12RootSignature>& out) {
    D3D12_DESCRIPTOR_RANGE ranges[2] = {
        {D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0, 0, 0},
        {compute ? D3D12_DESCRIPTOR_RANGE_TYPE_UAV : D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, compute ? 0u : 1u, 0, 1}};
    D3D12_ROOT_PARAMETER param{};
    param.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    param.DescriptorTable = {2, ranges};
    D3D12_VERSIONED_ROOT_SIGNATURE_DESC desc{};
    desc.Version = D3D_ROOT_SIGNATURE_VERSION_1_0;
    desc.Desc_1_0 = {1, &param, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_NONE};
    const auto blob = serialize_root_signature(desc);
    CHECK(!blob.empty());
    CHECK(SUCCEEDED(device->CreateRootSignature(0, blob.data(), blob.size(), IID_PPV_ARGS(&out.p))));
    return true;
}

bool Post::create_pipelines(const PostShaders& shaders) {
    CHECK(make_root(device_, true, compute_root_));
    CHECK(make_root(device_, false, graphics_root_));
    D3D12_COMPUTE_PIPELINE_STATE_DESC cs{};
    cs.pRootSignature = compute_root_.p;
    const std::pair<const std::vector<std::uint8_t>*, Ref<ID3D12PipelineState>*> computes[] = {
        {&shaders.bloom_h, &bloom_h_}, {&shaders.bloom_h_no_barrier, &bloom_h_no_barrier_}, {&shaders.bloom_v, &bloom_v_}};
    for (const auto& [code, pso] : computes) {
        cs.CS = {code->data(), code->size()};
        CHECK(SUCCEEDED(device_->CreateComputePipelineState(&cs, IID_PPV_ARGS(&pso->p))));
    }
    D3D12_GRAPHICS_PIPELINE_STATE_DESC desc{};
    desc.pRootSignature = graphics_root_.p;
    desc.VS = {shaders.vs_full.data(), shaders.vs_full.size()};
    desc.PS = {shaders.ps_composite.data(), shaders.ps_composite.size()};
    desc.BlendState.RenderTarget[0].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
    desc.SampleMask = 0xffffffff;
    desc.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
    desc.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
    desc.RasterizerState.DepthClipEnable = TRUE;
    desc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    desc.NumRenderTargets = 1;
    desc.RTVFormats[0] = DXGI_FORMAT_R32G32B32A32_FLOAT;
    desc.SampleDesc.Count = 1;
    CHECK(SUCCEEDED(device_->CreateGraphicsPipelineState(&desc, IID_PPV_ARGS(&composite_.p))));
    return true;
}

bool Post::create_resources() {
    const D3D12_HEAP_PROPERTIES default_heap{D3D12_HEAP_TYPE_DEFAULT, {}, {}, 0, 0};
    const D3D12_HEAP_PROPERTIES upload_heap{D3D12_HEAP_TYPE_UPLOAD, {}, {}, 0, 0};
    const D3D12_HEAP_PROPERTIES readback_heap{D3D12_HEAP_TYPE_READBACK, {}, {}, 0, 0};
    const auto make = [&](const D3D12_HEAP_PROPERTIES& heap, const D3D12_RESOURCE_DESC& desc, D3D12_RESOURCE_STATES state,
                          Ref<ID3D12Resource>& out) {
        return SUCCEEDED(device_->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, state, nullptr,
                                                          IID_PPV_ARGS(&out.p)));
    };
    const auto uav = buffer_desc(kBufferBytes, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    CHECK(make(default_heap, uav, D3D12_RESOURCE_STATE_COMMON, rows_));
    CHECK(make(default_heap, uav, D3D12_RESOURCE_STATE_COMMON, bloom_));
    CHECK(make(upload_heap, buffer_desc(kBufferBytes), D3D12_RESOURCE_STATE_GENERIC_READ, zero_));
    CHECK(make(readback_heap, buffer_desc(kImageBytes), D3D12_RESOURCE_STATE_COPY_DEST, readback_));
    void* mapped = nullptr;
    CHECK(SUCCEEDED(zero_->Map(0, nullptr, &mapped)));
    std::memset(mapped, 0, kBufferBytes);
    zero_->Unmap(0, nullptr);
    D3D12_RESOURCE_DESC tex{};
    tex.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    tex.Width = kWidth;
    tex.Height = kHeight;
    tex.DepthOrArraySize = 1;
    tex.MipLevels = 1;
    tex.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
    tex.SampleDesc.Count = 1;
    tex.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
    CHECK(make(default_heap, tex, D3D12_RESOURCE_STATE_COMMON, final_));
    return true;
}

bool Post::fill_descriptors() {
    const D3D12_DESCRIPTOR_HEAP_DESC heap_desc{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, kSlots,
                                               D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, 0};
    CHECK(SUCCEEDED(device_->CreateDescriptorHeap(&heap_desc, IID_PPV_ARGS(&heap_.p))));
    const D3D12_DESCRIPTOR_HEAP_DESC rtv_desc{D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 1, D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0};
    CHECK(SUCCEEDED(device_->CreateDescriptorHeap(&rtv_desc, IID_PPV_ARGS(&rtv_heap_.p))));
    const std::uint64_t cpu = heap_cpu(heap_.p);
    heap_gpu_ = heap_gpu(heap_.p);
    step_ = device_->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    // The shaders load the second descriptor of a table 0x20 bytes after the first.
    CHECK(cpu && heap_gpu_ >= 0x10000 && step_ == 32);
    D3D12_SHADER_RESOURCE_VIEW_DESC srv{};
    srv.Format = DXGI_FORMAT_R16G16B16A16_FLOAT;
    srv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    srv.Texture2D.MipLevels = 1;
    for (unsigned slot : {kRowsTable, kCompositeTable, kControlTable})
        device_->CreateShaderResourceView(scene_, &srv, D3D12_CPU_DESCRIPTOR_HANDLE{SIZE_T(cpu + slot * step_)});
    const auto put = [&](unsigned slot, ID3D12Resource* buffer) {
        const auto v = make_vsharp(buffer->GetGPUVirtualAddress(), 16, kPixels);
        std::memcpy(reinterpret_cast<void*>(cpu + slot * step_), v.data(), 16);
    };
    put(kRowsTable + 1, rows_.p);
    put(kColumnsTable, rows_.p);
    put(kColumnsTable + 1, bloom_.p);
    put(kCompositeTable + 1, bloom_.p);
    put(kControlTable + 1, zero_.p);
    rtv_ = {SIZE_T(heap_cpu(rtv_heap_.p))};
    device_->CreateRenderTargetView(final_.p, nullptr, rtv_);
    return true;
}

void Post::record_blur(PostMode mode) {
    list_->SetComputeRootSignature(compute_root_.p);
    list_->SetPipelineState(mode == PostMode::no_barrier ? bloom_h_no_barrier_.p : bloom_h_.p);
    list_->SetComputeRootDescriptorTable(0, D3D12_GPU_DESCRIPTOR_HANDLE{heap_gpu_ + kRowsTable * step_});
    list_->Dispatch(kWidth / 128, kHeight, 1);
    D3D12_RESOURCE_BARRIER uav{};
    uav.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
    uav.UAV.pResource = rows_.p;
    list_->ResourceBarrier(1, &uav);   // the column pass reads what the row pass wrote
    list_->SetPipelineState(bloom_v_.p);
    list_->SetComputeRootDescriptorTable(0, D3D12_GPU_DESCRIPTOR_HANDLE{heap_gpu_ + kColumnsTable * step_});
    list_->Dispatch(kWidth, kHeight / 128, 1);
}

bool Post::run(PostMode mode, std::vector<Color>& pixels) {
    CHECK(SUCCEEDED(allocator_->Reset()));
    CHECK(SUCCEEDED(list_->Reset(allocator_.p, nullptr)));
    const auto read_states = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
    const D3D12_RESOURCE_BARRIER before[] = {
        transition(scene_, D3D12_RESOURCE_STATE_COPY_SOURCE, read_states),
        transition(rows_.p, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS),
        transition(bloom_.p, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS),
        transition(final_.p, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_RENDER_TARGET)};
    list_->ResourceBarrier(4, before);
    ID3D12DescriptorHeap* heaps[] = {heap_.p};
    list_->SetDescriptorHeaps(1, heaps);
    if (mode != PostMode::skip_blur) record_blur(mode);
    const D3D12_RESOURCE_BARRIER to_read = transition(bloom_.p, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                                                      D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
    list_->ResourceBarrier(1, &to_read);
    list_->SetGraphicsRootSignature(graphics_root_.p);
    list_->SetPipelineState(composite_.p);
    const unsigned table = mode == PostMode::skip_blur ? kControlTable : kCompositeTable;
    list_->SetGraphicsRootDescriptorTable(0, D3D12_GPU_DESCRIPTOR_HANDLE{heap_gpu_ + table * step_});
    const D3D12_VIEWPORT viewport{0, 0, float(kWidth), float(kHeight), 0, 1};
    const D3D12_RECT scissor{0, 0, LONG(kWidth), LONG(kHeight)};
    list_->RSSetViewports(1, &viewport);
    list_->RSSetScissorRects(1, &scissor);
    list_->OMSetRenderTargets(1, &rtv_, FALSE, nullptr);
    list_->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    list_->DrawInstanced(3, 1, 0, 0);
    const D3D12_RESOURCE_BARRIER to_copy = transition(final_.p, D3D12_RESOURCE_STATE_RENDER_TARGET,
                                                      D3D12_RESOURCE_STATE_COPY_SOURCE);
    list_->ResourceBarrier(1, &to_copy);
    D3D12_TEXTURE_COPY_LOCATION dst{};
    dst.pResource = readback_.p;
    dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    dst.PlacedFootprint = {0, {DXGI_FORMAT_R32G32B32A32_FLOAT, kWidth, kHeight, 1, kPitch}};
    D3D12_TEXTURE_COPY_LOCATION src{};
    src.pResource = final_.p;
    src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    list_->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
    const D3D12_RESOURCE_BARRIER after[] = {
        transition(scene_, read_states, D3D12_RESOURCE_STATE_COPY_SOURCE),
        transition(rows_.p, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON),
        transition(bloom_.p, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COMMON),
        transition(final_.p, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON)};
    list_->ResourceBarrier(4, after);
    CHECK(list_->Close() == S_OK);
    ID3D12CommandList* lists[] = {list_.p};
    queue_->ExecuteCommandLists(1, lists);
    CHECK(SUCCEEDED(queue_->Signal(fence_.p, ++fence_value_)));
    CHECK(wait_fence(fence_, fence_value_));
    void* mapped = nullptr;
    CHECK(SUCCEEDED(readback_->Map(0, nullptr, &mapped)));
    const auto* bytes = static_cast<const std::uint8_t*>(mapped);
    pixels.resize(kPixels);
    for (unsigned y = 0; y < kHeight; ++y)
        std::memcpy(pixels.data() + std::size_t(y) * kWidth, bytes + y * kPitch, kWidth * sizeof(Color));
    readback_->Unmap(0, nullptr);
    return true;
}

}  // namespace lanterns
