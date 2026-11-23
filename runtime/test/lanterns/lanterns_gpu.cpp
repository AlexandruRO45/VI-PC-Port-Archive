// Lanterns: the GPU side (lanterns_gpu.h). Borrowed pieces, all from this repository under its licence: the
// D3D12XboxCreateDevice import (tiny_frame's, #3488), and from test/d3d12_translated_draw_test.cpp (M1.5) the heap-handle calls
// through slots 9/10, the V# encoding and the upload-to-DEFAULT copies.
#include "lanterns_gpu.h"

#include "lanterns_d3d12.h"

#include <algorithm>
#include <cstdio>
#include <cstring>

// Resolved at link time against test/fixtures/d3d12_xs_stub.cpp (which fails), at run time by gta6rt's shim registry.
extern "C" __declspec(dllimport) HRESULT WINAPI D3D12XboxCreateDevice(IUnknown* adapter, const void* params,
                                                                      const IID* riid, void** out);

using namespace d3d12test;

namespace lanterns {
namespace {

Geometry flatten(const Scene& scene) {
    Geometry g;
    for (const Mesh& mesh : scene.meshes) {
        g.first_vertex.push_back(std::uint32_t(g.vertices.size()));
        g.vertex_count.push_back(std::uint32_t(mesh.vertices.size()));
        g.first_index.push_back(std::uint32_t(g.indices.size()));
        g.vertices.insert(g.vertices.end(), mesh.vertices.begin(), mesh.vertices.end());
        for (const Face& face : mesh.faces)
            for (unsigned k = 1; k + 1 < face.count; ++k)   // indices are relative to the mesh: its V# starts there
                g.indices.insert(g.indices.end(), {face.v[0], face.v[k], face.v[k + 1]});
        g.index_count.push_back(std::uint32_t(g.indices.size()) - g.first_index.back());
    }
    for (const Draw& draw : scene.draws) {
        g.first_instance.push_back(std::uint32_t(g.instances.size()));
        g.instances.insert(g.instances.end(), draw.instances.begin(), draw.instances.end());
    }
    return g;
}

constexpr std::uint64_t kReadbackBytes = std::uint64_t(kPitch) * kHeight;   // the image

}  // namespace

bool load_shaders(const std::filesystem::path& dir, Shaders& out) {
    out.vs = read_file(dir / "vs_lanterns.dxbc");
    out.ps_lit = read_file(dir / "ps_lanterns_lit.dxbc");
    out.ps_floor = read_file(dir / "ps_lanterns_floor.dxbc");
    CHECK(out.vs.size() > 64 && out.ps_lit.size() > 64 && out.ps_floor.size() > 64);
    // The negative control of the lookup: the VS's FSSM program (fssm_unpack.py's copy) with its last byte, in the
    // tail's name, changed. The layer must refuse a container whose program the index does not know.
    auto program = read_file(dir / "fssm" / "vs_lanterns.fssm");
    CHECK(program.size() > 0x200 && std::memcmp(program.data(), "FSSM", 4) == 0);
    program.back() ^= 0x20;
    out.unknown_vs = shader_container(program);
    return true;
}

bool Gpu::create(const Scene& scene, const Shaders& shaders) {
    const std::uint32_t device_params[24] = {};
    const IID device_iid = __uuidof(ID3D12Device);
    const HRESULT created = D3D12XboxCreateDevice(nullptr, device_params, &device_iid, reinterpret_cast<void**>(&device_.p));
    if (FAILED(created) || !device_.p) {
        std::fprintf(stderr, "lanterns: D3D12XboxCreateDevice failed (hr=0x%08lx)\n", static_cast<unsigned long>(created));
        return false;
    }
    geometry_ = flatten(scene);
    CHECK(create_pipelines(shaders));
    CHECK(create_resources(scene));
    CHECK(fill_descriptors(scene));
    return initialize(scene);
}

// The root tables and constants of draw d (the mesh V#, the draw's instance V#, its model matrix and tint).
void Gpu::bind_draw(const Scene& scene, std::size_t d) {
    const Draw& draw = scene.draws[d];
    list_->SetGraphicsRootDescriptorTable(0, D3D12_GPU_DESCRIPTOR_HANDLE{srv_gpu_ + draw.mesh * srv_step_});
    list_->SetGraphicsRootDescriptorTable(1, D3D12_GPU_DESCRIPTOR_HANDLE{srv_gpu_ + (scene.meshes.size() + d) * srv_step_});
    float constants[16];
    std::memcpy(constants, draw.model.data(), sizeof(draw.model));
    std::memcpy(constants + 12, draw.tint.data(), sizeof(draw.tint));
    list_->SetGraphicsRoot32BitConstants(2, 16, constants, 0);
}

bool Gpu::create_pipelines(const Shaders& shaders) {
    // [0] table: the mesh V# (t0)   [1] table: the instance V# (t1)   [2] 16 constants (b0)   [3] root CBV (b1)
    // -> s[0:1], s[2:3], s4..s19, s[20:21]: what vs_lanterns.asm reads.
    D3D12_DESCRIPTOR_RANGE ranges[2] = {{D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0, 0, 0},
                                        {D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 1, 0, 0}};
    D3D12_ROOT_PARAMETER params[4] = {};
    params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    params[0].DescriptorTable = {1, &ranges[0]};
    params[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    params[1].DescriptorTable = {1, &ranges[1]};
    params[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    params[2].Constants = {0, 0, 16};
    params[3].ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
    params[3].Descriptor = {1, 0};
    D3D12_VERSIONED_ROOT_SIGNATURE_DESC root_desc{};
    root_desc.Version = D3D_ROOT_SIGNATURE_VERSION_1_0;
    root_desc.Desc_1_0 = {4, params, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_NONE};
    const auto root_blob = serialize_root_signature(root_desc);
    CHECK(!root_blob.empty());
    CHECK(SUCCEEDED(device_->CreateRootSignature(0, root_blob.data(), root_blob.size(), IID_PPV_ARGS(&root_.p))));

    D3D12_GRAPHICS_PIPELINE_STATE_DESC desc{};
    desc.pRootSignature = root_.p;
    desc.VS = {shaders.vs.data(), shaders.vs.size()};
    desc.BlendState.RenderTarget[0].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
    desc.SampleMask = 0xffffffff;
    desc.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
    desc.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
    desc.RasterizerState.DepthClipEnable = TRUE;
    desc.DepthStencilState.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ALL;
    desc.DepthStencilState.DepthFunc = D3D12_COMPARISON_FUNC_LESS;
    desc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    desc.NumRenderTargets = 1;
    desc.RTVFormats[0] = DXGI_FORMAT_R32G32B32A32_FLOAT;
    desc.DSVFormat = DXGI_FORMAT_D32_FLOAT;
    desc.SampleDesc.Count = 1;
    for (int floor = 0; floor < 2; ++floor)
        for (int depth = 0; depth < 2; ++depth) {
            const auto& ps = floor ? shaders.ps_floor : shaders.ps_lit;
            desc.PS = {ps.data(), ps.size()};
            desc.DepthStencilState.DepthEnable = depth ? TRUE : FALSE;
            CHECK(SUCCEEDED(device_->CreateGraphicsPipelineState(&desc, IID_PPV_ARGS(&pso_[floor][depth].p))));
        }
    std::printf("lanterns: 4 graphics pipelines from 3 FSSM containers; next, a VS the index does not know, which the "
                "layer must refuse (its 'no translated vs' line is expected)\n");
    Ref<ID3D12PipelineState> unknown;
    desc.VS = {shaders.unknown_vs.data(), shaders.unknown_vs.size()};
    CHECK(device_->CreateGraphicsPipelineState(&desc, IID_PPV_ARGS(&unknown.p)) == E_INVALIDARG && !unknown.p);
    return true;
}

bool Gpu::create_resources(const Scene& scene) {
    const D3D12_HEAP_PROPERTIES default_heap{D3D12_HEAP_TYPE_DEFAULT, {}, {}, 0, 0};
    const D3D12_HEAP_PROPERTIES upload_heap{D3D12_HEAP_TYPE_UPLOAD, {}, {}, 0, 0};
    const D3D12_HEAP_PROPERTIES readback_heap{D3D12_HEAP_TYPE_READBACK, {}, {}, 0, 0};
    const std::uint64_t vb_bytes = geometry_.vertices.size() * sizeof(Vertex);
    const std::uint64_t ib_bytes = align(geometry_.indices.size() * sizeof(std::uint16_t), 4);
    const std::uint64_t inst_bytes = std::max<std::uint64_t>(geometry_.instances.size() * sizeof(Instance), 256);
    // Upload layout: vertices, indices, instances, the camera's 64 bytes (the root CBV reads them in place: root views
    // are 64-bit console VAs, docs/d3d12_vulkan.md 3.3).
    upload_vb_ = 0;
    upload_ib_ = align(upload_vb_ + vb_bytes, 256);
    upload_inst_ = align(upload_ib_ + ib_bytes, 256);
    upload_cbv_ = align(upload_inst_ + inst_bytes, 256);
    const std::uint64_t upload_bytes = upload_cbv_ + 256;
    const auto make_buffer = [&](const D3D12_HEAP_PROPERTIES& heap, std::uint64_t bytes, D3D12_RESOURCE_STATES state,
                                 Ref<ID3D12Resource>& out, D3D12_RESOURCE_FLAGS flags = D3D12_RESOURCE_FLAG_NONE) {
        const auto desc = buffer_desc(bytes, flags);
        return SUCCEEDED(device_->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, state, nullptr,
                                                          IID_PPV_ARGS(&out.p)));
    };
    CHECK(make_buffer(default_heap, vb_bytes, D3D12_RESOURCE_STATE_COPY_DEST, vertex_buffer_));
    CHECK(make_buffer(default_heap, ib_bytes, D3D12_RESOURCE_STATE_COPY_DEST, index_buffer_));
    CHECK(make_buffer(default_heap, inst_bytes, D3D12_RESOURCE_STATE_COPY_DEST, instance_buffer_));
    CHECK(make_buffer(upload_heap, upload_bytes, D3D12_RESOURCE_STATE_GENERIC_READ, upload_));
    CHECK(make_buffer(readback_heap, kReadbackBytes, D3D12_RESOURCE_STATE_COPY_DEST, readback_));

    D3D12_RESOURCE_DESC tex{};
    tex.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    tex.Width = kWidth;
    tex.Height = kHeight;
    tex.DepthOrArraySize = 1;
    tex.MipLevels = 1;
    tex.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
    tex.SampleDesc.Count = 1;
    tex.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
    CHECK(SUCCEEDED(device_->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &tex,
                                                     D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&target_.p))));
    tex.Format = DXGI_FORMAT_D32_FLOAT;
    tex.Flags = D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;
    D3D12_CLEAR_VALUE depth_clear{};
    depth_clear.Format = DXGI_FORMAT_D32_FLOAT;
    depth_clear.DepthStencil = {1.0f, 0};
    CHECK(SUCCEEDED(device_->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &tex,
                                                     D3D12_RESOURCE_STATE_DEPTH_WRITE, &depth_clear,
                                                     IID_PPV_ARGS(&depth_.p))));

    const D3D12_DESCRIPTOR_HEAP_DESC srv_desc{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 32,
                                              D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, 0};
    CHECK(SUCCEEDED(device_->CreateDescriptorHeap(&srv_desc, IID_PPV_ARGS(&srv_heap_.p))));
    CHECK(scene.meshes.size() + scene.draws.size() <= 32);
    const D3D12_DESCRIPTOR_HEAP_DESC rtv_desc{D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 1, D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0};
    CHECK(SUCCEEDED(device_->CreateDescriptorHeap(&rtv_desc, IID_PPV_ARGS(&rtv_heap_.p))));
    const D3D12_DESCRIPTOR_HEAP_DESC dsv_desc{D3D12_DESCRIPTOR_HEAP_TYPE_DSV, 1, D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0};
    CHECK(SUCCEEDED(device_->CreateDescriptorHeap(&dsv_desc, IID_PPV_ARGS(&dsv_heap_.p))));
    rtv_ = {SIZE_T(heap_cpu(rtv_heap_.p))};
    dsv_ = {SIZE_T(heap_cpu(dsv_heap_.p))};
    device_->CreateRenderTargetView(target_.p, nullptr, rtv_);
    device_->CreateDepthStencilView(depth_.p, nullptr, dsv_);

    D3D12_COMMAND_QUEUE_DESC queue_desc{};
    CHECK(SUCCEEDED(device_->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&queue_.p))));
    CHECK(SUCCEEDED(device_->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator_.p))));
    CHECK(SUCCEEDED(device_->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator_.p, nullptr,
                                               IID_PPV_ARGS(&list_.p))));
    CHECK(list_->Close() == S_OK);
    CHECK(SUCCEEDED(device_->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence_.p))));
    return true;
}

// The descriptor tables: slot m holds mesh m's V#, slot (meshes + d) draw d's instance V#. Raw V# bytes, written through the CPU handle as the title writes them.
bool Gpu::fill_descriptors(const Scene& scene) {
    srv_cpu_ = heap_cpu(srv_heap_.p);
    srv_gpu_ = heap_gpu(srv_heap_.p);
    srv_step_ = device_->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    CHECK(srv_cpu_ && srv_gpu_ >= 0x10000 && srv_step_ >= 16);
    const auto put = [&](std::size_t index, const std::array<std::uint32_t, 4>& v) {
        std::memcpy(reinterpret_cast<void*>(srv_cpu_ + index * srv_step_), v.data(), 16);
    };
    const std::uint64_t vb = vertex_buffer_->GetGPUVirtualAddress(), inst = instance_buffer_->GetGPUVirtualAddress();
    const std::size_t meshes = scene.meshes.size(), draws = scene.draws.size();
    for (std::size_t m = 0; m < meshes; ++m)
        put(m, make_vsharp(vb + geometry_.first_vertex[m] * sizeof(Vertex), sizeof(Vertex), geometry_.vertex_count[m]));
    for (std::size_t d = 0; d < draws; ++d) {
        const Draw& draw = scene.draws[d];
        const auto count = std::uint32_t(draw.instances.size());
        put(meshes + d, make_vsharp(inst + geometry_.first_instance[d] * sizeof(Instance), sizeof(Instance), count));
    }
    return true;
}

// One direct-queue submission, waited for: the uploads into the DEFAULT buffers, and the target into COPY_SOURCE.
bool Gpu::initialize(const Scene& scene) {
    std::uint8_t* up = nullptr;
    CHECK(SUCCEEDED(upload_->Map(0, nullptr, reinterpret_cast<void**>(&up))));
    std::memcpy(up + upload_vb_, geometry_.vertices.data(), geometry_.vertices.size() * sizeof(Vertex));
    std::memcpy(up + upload_ib_, geometry_.indices.data(), geometry_.indices.size() * sizeof(std::uint16_t));
    if (!geometry_.instances.empty())
        std::memcpy(up + upload_inst_, geometry_.instances.data(), geometry_.instances.size() * sizeof(Instance));
    std::memcpy(up + upload_cbv_, scene.view_projection.data(), sizeof(scene.view_projection));

    CHECK(SUCCEEDED(allocator_->Reset()));
    CHECK(SUCCEEDED(list_->Reset(allocator_.p, nullptr)));
    list_->CopyBufferRegion(vertex_buffer_.p, 0, upload_.p, upload_vb_, geometry_.vertices.size() * sizeof(Vertex));
    list_->CopyBufferRegion(index_buffer_.p, 0, upload_.p, upload_ib_,
                            align(geometry_.indices.size() * sizeof(std::uint16_t), 4));
    if (!geometry_.instances.empty())
        list_->CopyBufferRegion(instance_buffer_.p, 0, upload_.p, upload_inst_,
                                geometry_.instances.size() * sizeof(Instance));
    const std::vector<D3D12_RESOURCE_BARRIER> barriers = {
        transition(vertex_buffer_.p, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE),
        transition(index_buffer_.p, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_INDEX_BUFFER),
        transition(instance_buffer_.p, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE),
        transition(target_.p, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_SOURCE)};
    list_->ResourceBarrier(UINT(barriers.size()), barriers.data());
    CHECK(list_->Close() == S_OK);
    ID3D12CommandList* lists[] = {list_.p};
    queue_->ExecuteCommandLists(1, lists);
    CHECK(SUCCEEDED(queue_->Signal(fence_.p, ++fence_value_)));
    CHECK(SUCCEEDED(fence_->SetEventOnCompletion(fence_value_, nullptr)));
    return true;
}

bool Gpu::submit_frame(const Scene& scene, bool depth_test) {
    CHECK(SUCCEEDED(allocator_->Reset()));
    CHECK(SUCCEEDED(list_->Reset(allocator_.p, nullptr)));
    const std::vector<D3D12_RESOURCE_BARRIER> before = {
        transition(target_.p, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_RENDER_TARGET)};
    list_->ResourceBarrier(UINT(before.size()), before.data());
    list_->ClearRenderTargetView(rtv_, kClear.data(), 0, nullptr);
    list_->ClearDepthStencilView(dsv_, D3D12_CLEAR_FLAG_DEPTH, 1.0f, 0, 0, nullptr);
    list_->SetGraphicsRootSignature(root_.p);
    ID3D12DescriptorHeap* heaps[] = {srv_heap_.p};
    list_->SetDescriptorHeaps(1, heaps);
    list_->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    const D3D12_INDEX_BUFFER_VIEW ibv{index_buffer_->GetGPUVirtualAddress(),
                                      UINT(align(geometry_.indices.size() * sizeof(std::uint16_t), 4)),
                                      DXGI_FORMAT_R16_UINT};
    list_->IASetIndexBuffer(&ibv);
    const D3D12_VIEWPORT viewport{0, 0, float(kWidth), float(kHeight), 0, 1};
    const D3D12_RECT scissor{0, 0, LONG(kWidth), LONG(kHeight)};
    list_->RSSetViewports(1, &viewport);
    list_->RSSetScissorRects(1, &scissor);
    list_->OMSetRenderTargets(1, &rtv_, FALSE, &dsv_);
    list_->SetGraphicsRootConstantBufferView(3, upload_->GetGPUVirtualAddress() + upload_cbv_);
    for (std::size_t d = 0; d < scene.draws.size(); ++d) {
        const Draw& draw = scene.draws[d];
        list_->SetPipelineState(pso_[draw.floor ? 1 : 0][depth_test ? 1 : 0].p);
        bind_draw(scene, d);
        list_->DrawIndexedInstanced(geometry_.index_count[draw.mesh], UINT(draw.instances.size()),
                                    geometry_.first_index[draw.mesh], 0, 0);
    }
    const std::vector<D3D12_RESOURCE_BARRIER> to_copy = {
        transition(target_.p, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_COPY_SOURCE)};
    list_->ResourceBarrier(UINT(to_copy.size()), to_copy.data());
    D3D12_TEXTURE_COPY_LOCATION dst{};
    dst.pResource = readback_.p;
    dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    dst.PlacedFootprint = {0, {DXGI_FORMAT_R32G32B32A32_FLOAT, kWidth, kHeight, 1, kPitch}};
    D3D12_TEXTURE_COPY_LOCATION src{};
    src.pResource = target_.p;
    src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    list_->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
    CHECK(list_->Close() == S_OK);
    ID3D12CommandList* lists[] = {list_.p};
    queue_->ExecuteCommandLists(1, lists);
    CHECK(SUCCEEDED(queue_->Signal(fence_.p, ++fence_value_)));
    return true;
}

bool Gpu::finish_frame(std::vector<Color>& pixels) {
    CHECK(SUCCEEDED(fence_->SetEventOnCompletion(fence_value_, nullptr)));
    void* mapped = nullptr;
    CHECK(SUCCEEDED(readback_->Map(0, nullptr, &mapped)));
    const auto* bytes = static_cast<const std::uint8_t*>(mapped);
    pixels.resize(std::size_t(kWidth) * kHeight);
    for (unsigned y = 0; y < kHeight; ++y)
        std::memcpy(pixels.data() + std::size_t(y) * kWidth, bytes + y * kPitch, kWidth * sizeof(Color));
    readback_->Unmap(0, nullptr);
    return true;
}

bool Gpu::idle() {
    CHECK(SUCCEEDED(fence_->SetEventOnCompletion(fence_value_, nullptr)));
    return true;
}

}  // namespace lanterns
