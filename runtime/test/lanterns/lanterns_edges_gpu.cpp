// Lanterns stage 1: the shared edges on the GPU (lanterns_edges_gpu.h).
#include "lanterns_edges_gpu.h"

#include "lanterns_d3d12.h"

#include <cstring>

using namespace d3d12test;

namespace lanterns {

bool load_edge_shaders(const std::filesystem::path& dir, EdgeShaders& out) {
    out.vs = read_file(dir / "vs_lanterns_edges.dxbc");
    out.ps = read_file(dir / "ps_lanterns_edges.dxbc");
    CHECK(out.vs.size() > 64 && out.ps.size() > 64);
    return true;
}

bool EdgeDraw::create(Gpu& gpu, const EdgeShaders& shaders) {
    device_ = gpu.device();
    queue_ = gpu.queue();
    CHECK(device_ && queue_);
    // [0] eight constants -> s0..s7; [1] a table (t0 the vertices, t1 the alpha) -> s[8:9].
    const D3D12_DESCRIPTOR_RANGE range{D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 2, 0, 0, 0};
    D3D12_ROOT_PARAMETER params[2] = {};
    params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    params[0].Constants = {0, 0, 8};
    params[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    params[1].DescriptorTable = {1, &range};
    D3D12_VERSIONED_ROOT_SIGNATURE_DESC desc{};
    desc.Version = D3D_ROOT_SIGNATURE_VERSION_1_0;
    desc.Desc_1_0 = {2, params, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_NONE};
    const auto blob = serialize_root_signature(desc);
    CHECK(!blob.empty());
    CHECK(SUCCEEDED(device_->CreateRootSignature(0, blob.data(), blob.size(), IID_PPV_ARGS(&root_.p))));
    D3D12_GRAPHICS_PIPELINE_STATE_DESC pso{};
    pso.pRootSignature = root_.p;
    pso.VS = {shaders.vs.data(), shaders.vs.size()};
    pso.PS = {shaders.ps.data(), shaders.ps.size()};
    auto& blend = pso.BlendState.RenderTarget[0];
    blend.BlendEnable = TRUE;
    blend.SrcBlend = blend.DestBlend = D3D12_BLEND_ONE;
    blend.SrcBlendAlpha = blend.DestBlendAlpha = D3D12_BLEND_ONE;
    blend.BlendOp = blend.BlendOpAlpha = D3D12_BLEND_OP_ADD;
    blend.RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
    pso.SampleMask = 0xffffffff;
    pso.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
    pso.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
    pso.RasterizerState.DepthClipEnable = TRUE;
    pso.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    pso.NumRenderTargets = 1;
    pso.RTVFormats[0] = DXGI_FORMAT_R32G32B32A32_FLOAT;
    pso.SampleDesc.Count = 1;
    CHECK(SUCCEEDED(device_->CreateGraphicsPipelineState(&pso, IID_PPV_ARGS(&pso_.p))));

    const D3D12_HEAP_PROPERTIES default_heap{D3D12_HEAP_TYPE_DEFAULT, {}, {}, 0, 0};
    const D3D12_HEAP_PROPERTIES upload_heap{D3D12_HEAP_TYPE_UPLOAD, {}, {}, 0, 0};
    const D3D12_HEAP_PROPERTIES readback_heap{D3D12_HEAP_TYPE_READBACK, {}, {}, 0, 0};
    const auto vertices = edge_vertices();
    const std::uint64_t vertex_bytes = vertices.size() * sizeof(EdgeVertex);
    const auto make = [&](const D3D12_HEAP_PROPERTIES& heap, const D3D12_RESOURCE_DESC& d, D3D12_RESOURCE_STATES state,
                          Ref<ID3D12Resource>& out) {
        return SUCCEEDED(device_->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &d, state, nullptr,
                                                          IID_PPV_ARGS(&out.p)));
    };
    // The vertices and the PS's alpha word live in DEFAULT buffers, filled from one UPLOAD staging buffer by
    // CopyBufferRegion, as the main scene's buffers are (lanterns_gpu.cpp). Reading them straight from UPLOAD heaps
    // drew nothing on an Intel UHD 770 under native Windows.
    const std::uint64_t alpha_offset = align(vertex_bytes, 256);
    CHECK(make(default_heap, buffer_desc(vertex_bytes), D3D12_RESOURCE_STATE_COPY_DEST, vertices_));
    CHECK(make(default_heap, buffer_desc(256), D3D12_RESOURCE_STATE_COPY_DEST, alpha_));
    Ref<ID3D12Resource> staging;   // kept alive until the copy's fence has passed
    CHECK(make(upload_heap, buffer_desc(alpha_offset + 256), D3D12_RESOURCE_STATE_GENERIC_READ, staging));
    CHECK(make(readback_heap, buffer_desc(std::uint64_t(kPitch) * kHeight), D3D12_RESOURCE_STATE_COPY_DEST, readback_));
    std::uint8_t* mapped = nullptr;
    CHECK(SUCCEEDED(staging->Map(0, nullptr, reinterpret_cast<void**>(&mapped))));
    std::memcpy(mapped, vertices.data(), vertex_bytes);
    const float alpha[4] = {1.0f, 0.0f, 0.0f, 0.0f};
    std::memcpy(mapped + alpha_offset, alpha, sizeof(alpha));
    staging->Unmap(0, nullptr);
    D3D12_RESOURCE_DESC tex{};
    tex.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    tex.Width = kWidth;
    tex.Height = kHeight;
    tex.DepthOrArraySize = 1;
    tex.MipLevels = 1;
    tex.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
    tex.SampleDesc.Count = 1;
    tex.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
    CHECK(make(default_heap, tex, D3D12_RESOURCE_STATE_COMMON, target_));

    const D3D12_DESCRIPTOR_HEAP_DESC heap_desc{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 2,
                                               D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, 0};
    CHECK(SUCCEEDED(device_->CreateDescriptorHeap(&heap_desc, IID_PPV_ARGS(&heap_.p))));
    const std::uint64_t cpu = heap_cpu(heap_.p);
    heap_gpu_ = heap_gpu(heap_.p);
    CHECK(device_->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV) == 32);
    const auto v0 = make_vsharp(vertices_->GetGPUVirtualAddress(), sizeof(EdgeVertex), std::uint32_t(vertices.size()));
    const auto v1 = make_raw(alpha_->GetGPUVirtualAddress(), 16);
    std::memcpy(reinterpret_cast<void*>(cpu), v0.data(), 16);
    std::memcpy(reinterpret_cast<void*>(cpu + 32), v1.data(), 16);
    const D3D12_DESCRIPTOR_HEAP_DESC rtv_desc{D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 1, D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0};
    CHECK(SUCCEEDED(device_->CreateDescriptorHeap(&rtv_desc, IID_PPV_ARGS(&rtv_heap_.p))));
    device_->CreateRenderTargetView(target_.p, nullptr, D3D12_CPU_DESCRIPTOR_HANDLE{SIZE_T(heap_cpu(rtv_heap_.p))});

    CHECK(SUCCEEDED(device_->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator_.p))));
    CHECK(SUCCEEDED(device_->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator_.p, nullptr,
                                               IID_PPV_ARGS(&list_.p))));
    CHECK(SUCCEEDED(device_->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence_.p))));
    list_->CopyBufferRegion(vertices_.p, 0, staging.p, 0, vertex_bytes);
    list_->CopyBufferRegion(alpha_.p, 0, staging.p, alpha_offset, sizeof(alpha));
    const D3D12_RESOURCE_BARRIER filled[] = {
        transition(vertices_.p, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE),
        transition(alpha_.p, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE)};
    list_->ResourceBarrier(2, filled);
    CHECK(list_->Close() == S_OK);
    ID3D12CommandList* lists[] = {list_.p};
    queue_->ExecuteCommandLists(1, lists);
    CHECK(SUCCEEDED(queue_->Signal(fence_.p, ++fence_value_)));
    CHECK(wait_fence(fence_, fence_value_));
    return true;
}

bool EdgeDraw::run(std::vector<Color>& pixels) {
    CHECK(SUCCEEDED(allocator_->Reset()));
    CHECK(SUCCEEDED(list_->Reset(allocator_.p, nullptr)));
    const auto to_rt = transition(target_.p, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_RENDER_TARGET);
    list_->ResourceBarrier(1, &to_rt);
    const D3D12_CPU_DESCRIPTOR_HANDLE rtv{SIZE_T(heap_cpu(rtv_heap_.p))};
    const float zero[4] = {0, 0, 0, 0};
    list_->ClearRenderTargetView(rtv, zero, 0, nullptr);
    ID3D12DescriptorHeap* heaps[] = {heap_.p};
    list_->SetDescriptorHeaps(1, heaps);
    list_->SetGraphicsRootSignature(root_.p);
    // s0..s3: clip x = x / 192 - 1, clip y = 1 - y / 128; s4..s7: words nothing reads.
    const float constants[8] = {1.0f / 192.0f, -1.0f, -1.0f / 128.0f, 1.0f, 0, 0, 0, 0};
    std::uint32_t words[8];
    std::memcpy(words, constants, sizeof(words));
    for (unsigned i = 4; i < 8; ++i) words[i] = 0x4c4e5400u | i;
    list_->SetGraphicsRoot32BitConstants(0, 8, words, 0);
    list_->SetGraphicsRootDescriptorTable(1, D3D12_GPU_DESCRIPTOR_HANDLE{heap_gpu_});
    list_->SetPipelineState(pso_.p);
    const D3D12_VIEWPORT viewport{0, 0, float(kWidth), float(kHeight), 0, 1};
    const D3D12_RECT scissor{0, 0, LONG(kWidth), LONG(kHeight)};
    list_->RSSetViewports(1, &viewport);
    list_->RSSetScissorRects(1, &scissor);
    list_->OMSetRenderTargets(1, &rtv, FALSE, nullptr);
    list_->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    list_->DrawInstanced(3 * kEdgeTriangles, 1, 0, 0);
    const auto to_copy = transition(target_.p, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_COPY_SOURCE);
    list_->ResourceBarrier(1, &to_copy);
    D3D12_TEXTURE_COPY_LOCATION dst{readback_.p, D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, {}};
    dst.PlacedFootprint = {0, {DXGI_FORMAT_R32G32B32A32_FLOAT, kWidth, kHeight, 1, kPitch}};
    D3D12_TEXTURE_COPY_LOCATION src{target_.p, D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, {}};
    src.SubresourceIndex = 0;
    list_->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
    const auto back = transition(target_.p, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON);
    list_->ResourceBarrier(1, &back);
    CHECK(list_->Close() == S_OK);
    ID3D12CommandList* lists[] = {list_.p};
    queue_->ExecuteCommandLists(1, lists);
    CHECK(SUCCEEDED(queue_->Signal(fence_.p, ++fence_value_)));
    CHECK(wait_fence(fence_, fence_value_));
    void* mapped = nullptr;
    CHECK(SUCCEEDED(readback_->Map(0, nullptr, &mapped)));
    pixels.resize(std::size_t(kWidth) * kHeight);
    for (unsigned y = 0; y < kHeight; ++y)
        std::memcpy(pixels.data() + std::size_t(y) * kWidth, static_cast<const std::uint8_t*>(mapped) + y * kPitch,
                    kWidth * sizeof(Color));
    readback_->Unmap(0, nullptr);
    return true;
}

}  // namespace lanterns