// Lanterns stage 8: the shadow map's resources and its depth-only pass, as members of Gpu (lanterns_gpu.h). The map is
// an R32_TYPELESS texture with a D32_FLOAT depth view and an R32_FLOAT shader view; the sampler compares LESS_EQUAL with
// point filtering. Depth bias comes from the depth pass's PSO.
#include "lanterns_d3d12.h"
#include "lanterns_gpu.h"
#include "lanterns_shadow.h"

#include <cstring>

using namespace d3d12test;

namespace lanterns {

bool Gpu::create_shadow() {
    D3D12_RESOURCE_DESC tex{};
    tex.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    tex.Width = kShadowSize;
    tex.Height = kShadowSize;
    tex.DepthOrArraySize = 1;
    tex.MipLevels = 1;
    tex.Format = DXGI_FORMAT_R32_TYPELESS;
    tex.SampleDesc.Count = 1;
    tex.Flags = D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;
    D3D12_CLEAR_VALUE clear{};
    clear.Format = DXGI_FORMAT_D32_FLOAT;
    clear.DepthStencil = {1.0f, 0};
    const D3D12_HEAP_PROPERTIES default_heap{D3D12_HEAP_TYPE_DEFAULT, {}, {}, 0, 0};
    CHECK(SUCCEEDED(device_->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &tex,
                                                     D3D12_RESOURCE_STATE_DEPTH_WRITE, &clear,
                                                     IID_PPV_ARGS(&shadow_map_.p))));
    shadow_dsv_ = {dsv_.ptr + device_->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_DSV)};
    D3D12_DEPTH_STENCIL_VIEW_DESC dsv{};
    dsv.Format = DXGI_FORMAT_D32_FLOAT;
    dsv.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2D;
    device_->CreateDepthStencilView(shadow_map_.p, &dsv, shadow_dsv_);
    D3D12_SHADER_RESOURCE_VIEW_DESC srv{};
    srv.Format = DXGI_FORMAT_R32_FLOAT;
    srv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    srv.Texture2D.MipLevels = 1;
    device_->CreateShaderResourceView(shadow_map_.p, &srv,
                                      D3D12_CPU_DESCRIPTOR_HANDLE{SIZE_T(srv_cpu_ + srv_texture_slot_ * srv_step_)});
    const D3D12_DESCRIPTOR_HEAP_DESC sampler_desc{D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER, 1,
                                                  D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, 0};
    CHECK(SUCCEEDED(device_->CreateDescriptorHeap(&sampler_desc, IID_PPV_ARGS(&sampler_heap_.p))));
    D3D12_SAMPLER_DESC sampler{};
    sampler.Filter = D3D12_FILTER_COMPARISON_MIN_MAG_MIP_POINT;
    sampler.AddressU = sampler.AddressV = sampler.AddressW = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
    sampler.MaxAnisotropy = 1;
    sampler.ComparisonFunc = D3D12_COMPARISON_FUNC_LESS_EQUAL;
    sampler.MaxLOD = D3D12_FLOAT32_MAX;
    device_->CreateSampler(&sampler, D3D12_CPU_DESCRIPTOR_HANDLE{SIZE_T(heap_cpu(sampler_heap_.p))});
    textures_bound_ = true;

    // The depth-only pipeline: vs_lanterns, no pixel shader, no render target, the bias.
    D3D12_GRAPHICS_PIPELINE_STATE_DESC desc{};
    desc.pRootSignature = root_.p;
    desc.VS = {vs_.data(), vs_.size()};
    desc.SampleMask = 0xffffffff;
    desc.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
    desc.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
    desc.RasterizerState.DepthClipEnable = TRUE;
    desc.RasterizerState.DepthBias = kShadowDepthBias;
    desc.RasterizerState.SlopeScaledDepthBias = kShadowSlopeBias;
    desc.DepthStencilState.DepthEnable = TRUE;
    desc.DepthStencilState.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ALL;
    desc.DepthStencilState.DepthFunc = D3D12_COMPARISON_FUNC_LESS;
    desc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    desc.NumRenderTargets = 0;
    desc.DSVFormat = DXGI_FORMAT_D32_FLOAT;
    desc.SampleDesc.Count = 1;
    CHECK(SUCCEEDED(device_->CreateGraphicsPipelineState(&desc, IID_PPV_ARGS(&shadow_pso_.p))));
    return true;
}

bool Gpu::set_light(const std::array<float, 16>& clip, const std::array<float, 12>& rows) {
    CHECK(shadow_);
    void* mapped = nullptr;
    CHECK(SUCCEEDED(upload_->Map(0, nullptr, &mapped)));
    auto* up = static_cast<std::uint8_t*>(mapped);
    std::memcpy(up + upload_cbv_ + 64, rows.data(), sizeof(rows));
    std::memcpy(up + upload_light_, clip.data(), sizeof(clip));
    upload_->Unmap(0, nullptr);
    return true;
}

// Before the main pass: the map cleared to 1, the solids drawn into it from the light (unless the control), and the
// map left in PIXEL_SHADER_RESOURCE for the main pass's comparison samples.
void Gpu::record_shadow_pass(const Scene& scene) {
    const auto read = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, write = D3D12_RESOURCE_STATE_DEPTH_WRITE;
    if (shadow_written_) {
        const D3D12_RESOURCE_BARRIER to_write = transition(shadow_map_.p, read, write);
        list_->ResourceBarrier(1, &to_write);
    }
    shadow_written_ = true;
    list_->ClearDepthStencilView(shadow_dsv_, D3D12_CLEAR_FLAG_DEPTH, 1.0f, 0, 0, nullptr);
    if (shadow_casters_) {
        const D3D12_VIEWPORT viewport{0, 0, float(kShadowSize), float(kShadowSize), 0, 1};
        const D3D12_RECT scissor{0, 0, LONG(kShadowSize), LONG(kShadowSize)};
        list_->RSSetViewports(1, &viewport);
        list_->RSSetScissorRects(1, &scissor);
        list_->OMSetRenderTargets(0, nullptr, FALSE, &shadow_dsv_);
        list_->SetGraphicsRootSignature(root_.p);
        ID3D12DescriptorHeap* heaps[] = {srv_heap_.p, sampler_heap_.p};
        list_->SetDescriptorHeaps(2, heaps);
        list_->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        const D3D12_INDEX_BUFFER_VIEW ibv{index_buffer_->GetGPUVirtualAddress(),
                                          UINT(align(geometry_.indices.size() * sizeof(std::uint16_t), 4)),
                                          DXGI_FORMAT_R16_UINT};
        list_->IASetIndexBuffer(&ibv);
        list_->SetGraphicsRootConstantBufferView(3, upload_->GetGPUVirtualAddress() + upload_light_);
        list_->SetPipelineState(shadow_pso_.p);
        for (std::size_t d = 0; d < scene.draws.size(); ++d) {
            const Draw& draw = scene.draws[d];
            if (draw.hidden || draw.floor || draw.gpu_instances) continue;   // the floor only receives
            bind_draw(scene, d);
            list_->DrawIndexedInstanced(geometry_.index_count[draw.mesh], UINT(draw.instances.size()),
                                        geometry_.first_index[draw.mesh], 0, 0);
        }
    }
    const D3D12_RESOURCE_BARRIER to_read = transition(shadow_map_.p, write, read);
    list_->ResourceBarrier(1, &to_read);
}

}  // namespace lanterns
