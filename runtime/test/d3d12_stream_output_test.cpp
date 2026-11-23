// Stream Output on VK_EXT_transform_feedback (SOSetTargets, D3D12_STREAM_OUTPUT_DESC, BufferFilledSizeLocation counters):
// tests PSO validation parity against Windows D3D12, SO-only draws (D3D12_SO_NO_RASTERIZED_STREAM with no PS/RTV/viewport),
// append + reset across draws via BufferFilledSizeLocation, simultaneous SO + rasterization, multi-slot SO (slots 0 and 1),
// SOSetTargets unbind / ClearState, bundle rejection, and standalone device caps.
// No game data.
#include "vk_test_device.h"
#include "d3d12_copy_compare.h"
#include "d3d12_test_shaders.h"
#include "../src/shader_library.h"
#include "../shims/d3d12_com.h"

#include <d3dcompiler.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#define CHECK(cond)                                                                         \
    do {                                                                                    \
        if (!(cond)) {                                                                      \
            std::fprintf(stderr, "line %d: CHECK failed: %s\n", __LINE__, #cond);           \
            return false;                                                                   \
        }                                                                                   \
    } while (0)

#define REQUIRE(cond)                                                                       \
    do {                                                                                    \
        if (!(cond)) {                                                                      \
            std::fprintf(stderr, "line %d: REQUIRE failed: %s\n", __LINE__, #cond);         \
            return 1;                                                                       \
        }                                                                                   \
    } while (0)

namespace {

struct SoInputVertex {
    float x, y, z, w;
    float r, g, b, a;
};

struct SoCapturedVertex {
    float x, y, z, w;
    float r, g, b, a;
};

static_assert(sizeof(SoCapturedVertex) == 32);

struct Fixture {
    copycmp::Run run;
    copycmp::Ptr<ID3D12RootSignature> root;
    D3D12_SHADER_BYTECODE vs_single{};
    D3D12_SHADER_BYTECODE vs_multi{};
    D3D12_SHADER_BYTECODE ps{};
};

bool init_fixture(Fixture& f, ID3D12Device* device, bool ours,
                  D3D12_SHADER_BYTECODE vs_single, D3D12_SHADER_BYTECODE vs_multi,
                  D3D12_SHADER_BYTECODE ps) {
    CHECK(f.run.init(device, ours));
    f.vs_single = vs_single;
    f.vs_multi = vs_multi;
    f.ps = ps;
    D3D12_ROOT_PARAMETER param{};
    param.ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    param.Constants = {0, 0, 2};
    D3D12_VERSIONED_ROOT_SIGNATURE_DESC root_desc{};
    root_desc.Version = D3D_ROOT_SIGNATURE_VERSION_1_0;
    root_desc.Desc_1_0 = {1, &param, 0, nullptr,
                          D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT |
                              D3D12_ROOT_SIGNATURE_FLAG_ALLOW_STREAM_OUTPUT};
    const auto blob = d3d12test::serialize_root_signature(root_desc);
    CHECK(!blob.empty());
    CHECK(SUCCEEDED(device->CreateRootSignature(0, blob.data(), blob.size(), IID_PPV_ARGS(&f.root.p))));
    return true;
}

D3D12_GRAPHICS_PIPELINE_STATE_DESC make_pso_desc(const Fixture& f, bool multi_slot, bool with_ps) {
    static const D3D12_INPUT_ELEMENT_DESC elements[2] = {
        {"POSITION", 0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, 0, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
        {"COLOR", 0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, 16, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
    };
    static const D3D12_SO_DECLARATION_ENTRY single_entries[2] = {
        {0, "TEXCOORD", 0, 0, 4, 0},
        {0, "COLOR", 0, 0, 4, 0},
    };
    static constexpr UINT single_strides[1] = {32};
    static const D3D12_SO_DECLARATION_ENTRY multi_entries[2] = {
        {0, "TEXCOORD", 0, 0, 4, 0},
        {0, "COLOR", 0, 0, 4, 1},
    };
    static constexpr UINT multi_strides[2] = {16, 16};

    D3D12_GRAPHICS_PIPELINE_STATE_DESC desc{};
    desc.pRootSignature = f.root.p;
    desc.VS = multi_slot ? f.vs_multi : f.vs_single;
    if (with_ps) desc.PS = f.ps;
    desc.InputLayout = {elements, 2};
    desc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    desc.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
    desc.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
    desc.RasterizerState.DepthClipEnable = TRUE;
    desc.BlendState.RenderTarget[0].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
    desc.SampleMask = UINT_MAX;
    desc.SampleDesc.Count = 1;
    desc.StreamOutput.pSODeclaration = multi_slot ? multi_entries : single_entries;
    desc.StreamOutput.NumEntries = 2;
    desc.StreamOutput.pBufferStrides = multi_slot ? multi_strides : single_strides;
    desc.StreamOutput.NumStrides = multi_slot ? 2 : 1;
    desc.StreamOutput.RasterizedStream = with_ps ? 0u : D3D12_SO_NO_RASTERIZED_STREAM;
    if (with_ps) {
        desc.NumRenderTargets = 1;
        desc.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
    }
    return desc;
}

bool check_validation_cases(Fixture& ours, Fixture* host) {
    const auto check_invalid = [&](const char* label, bool check_host, auto mutate) -> bool {
        D3D12_GRAPHICS_PIPELINE_STATE_DESC d_ours = make_pso_desc(ours, false, false);
        mutate(d_ours);
        copycmp::Ptr<ID3D12PipelineState> p_ours;
        const HRESULT hr_ours = ours.run.device->CreateGraphicsPipelineState(&d_ours, IID_PPV_ARGS(&p_ours.p));
        if (hr_ours != E_INVALIDARG || p_ours.p != nullptr) {
            std::fprintf(stderr, "validation (%s): ours returned 0x%08x, expected E_INVALIDARG\n",
                         label, unsigned(hr_ours));
            return false;
        }
        if (host && check_host) {
            D3D12_GRAPHICS_PIPELINE_STATE_DESC d_host = make_pso_desc(*host, false, false);
            mutate(d_host);
            copycmp::Ptr<ID3D12PipelineState> p_host;
            const HRESULT hr_host = host->run.device->CreateGraphicsPipelineState(&d_host, IID_PPV_ARGS(&p_host.p));
            if (hr_host != E_INVALIDARG || p_host.p != nullptr) {
                std::fprintf(stderr, "validation (%s): host returned 0x%08x, expected E_INVALIDARG\n",
                             label, unsigned(hr_host));
                return false;
            }
        }
        return true;
    };

    static D3D12_SO_DECLARATION_ENTRY bad_slot{0, "TEXCOORD", 0, 0, 4, 4};
    static D3D12_SO_DECLARATION_ENTRY bad_zero_comp{0, "TEXCOORD", 0, 0, 0, 0};
    static D3D12_SO_DECLARATION_ENTRY bad_comp_range{0, "TEXCOORD", 0, 2, 3, 0};
    static D3D12_SO_DECLARATION_ENTRY bad_gap_start{0, nullptr, 0, 1, 1, 0};
    static constexpr UINT bad_strides[5] = {32, 0, 0, 0, 0};
    static constexpr UINT unaligned_stride[1] = {6};
    static constexpr UINT huge_stride[1] = {2052};

    // Release d3d12Core.dll (without debug layer) does not null-check pSODeclaration / pBufferStrides when counts > 0.
    CHECK(check_invalid("null pSODeclaration", false, [](auto& d) { d.StreamOutput.pSODeclaration = nullptr; }));
    CHECK(check_invalid("NumStrides > 4", true, [](auto& d) {
        d.StreamOutput.NumStrides = 5;
        d.StreamOutput.pBufferStrides = bad_strides;
    }));
    CHECK(check_invalid("null pBufferStrides", false, [](auto& d) { d.StreamOutput.pBufferStrides = nullptr; }));
    CHECK(check_invalid("RasterizedStream == 4", true, [](auto& d) { d.StreamOutput.RasterizedStream = 4; }));
    CHECK(check_invalid("RasterizedStream == 1 without GS", true, [](auto& d) { d.StreamOutput.RasterizedStream = 1; }));
    CHECK(check_invalid("OutputSlot == 4", true, [](auto& d) {
        d.StreamOutput.NumEntries = 1;
        d.StreamOutput.pSODeclaration = &bad_slot;
    }));
    CHECK(check_invalid("ComponentCount == 0", true, [](auto& d) {
        d.StreamOutput.NumEntries = 1;
        d.StreamOutput.pSODeclaration = &bad_zero_comp;
    }));
    CHECK(check_invalid("StartComponent + ComponentCount > 4", true, [](auto& d) {
        d.StreamOutput.NumEntries = 1;
        d.StreamOutput.pSODeclaration = &bad_comp_range;
    }));
    CHECK(check_invalid("null semantic with nonzero StartComponent", true, [](auto& d) {
        d.StreamOutput.NumEntries = 1;
        d.StreamOutput.pSODeclaration = &bad_gap_start;
    }));
    CHECK(check_invalid("unaligned stride 6", true, [](auto& d) { d.StreamOutput.pBufferStrides = unaligned_stride; }));
    CHECK(check_invalid("stride > 2048", true, [](auto& d) { d.StreamOutput.pBufferStrides = huge_stride; }));
    return true;
}

bool run_single_slot_so_append_and_reset(Fixture& f, std::vector<std::uint8_t>& out_so_append,
                                         std::uint64_t& out_counter_append,
                                         std::vector<std::uint8_t>& out_so_reset,
                                         std::uint64_t& out_counter_reset) {
    D3D12_GRAPHICS_PIPELINE_STATE_DESC pso_desc = make_pso_desc(f, false, false);
    copycmp::Ptr<ID3D12PipelineState> pso;
    const HRESULT pso_hr = f.run.device->CreateGraphicsPipelineState(&pso_desc, IID_PPV_ARGS(&pso.p));
    if (FAILED(pso_hr)) {
        std::fprintf(stderr, "CreateGraphicsPipelineState (%s) failed: 0x%08x\n",
                     f.run.ours ? "Vulkan" : "host", unsigned(pso_hr));
        return false;
    }

    constexpr UINT so_size = 512;
    constexpr UINT counter_size = 256;
    const SoInputVertex verts[3] = {
        {-0.5f,  0.5f, 0.25f, 1.0f, 1.0f, 0.0f, 0.0f, 1.0f},
        { 0.5f,  0.5f, 0.25f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f},
        {-0.5f, -0.5f, 0.25f, 1.0f, 0.0f, 0.0f, 1.0f, 1.0f},
    };

    copycmp::Ptr<ID3D12Resource> vb, so_buf, cnt_buf, rb_so, rb_cnt;
    CHECK(f.run.buffer(D3D12_HEAP_TYPE_UPLOAD, sizeof(verts), D3D12_RESOURCE_STATE_GENERIC_READ, vb));
    {
        void* p = nullptr;
        const D3D12_RANGE none{};
        CHECK(SUCCEEDED(vb->Map(0, &none, &p)));
        std::memcpy(p, verts, sizeof(verts));
        vb->Unmap(0, nullptr);
    }
    CHECK(f.run.buffer(D3D12_HEAP_TYPE_DEFAULT, so_size, D3D12_RESOURCE_STATE_COPY_DEST, so_buf));
    CHECK(f.run.buffer(D3D12_HEAP_TYPE_DEFAULT, counter_size, D3D12_RESOURCE_STATE_COPY_DEST, cnt_buf));
    CHECK(f.run.buffer(D3D12_HEAP_TYPE_READBACK, so_size, D3D12_RESOURCE_STATE_COPY_DEST, rb_so));
    CHECK(f.run.buffer(D3D12_HEAP_TYPE_READBACK, counter_size, D3D12_RESOURCE_STATE_COPY_DEST, rb_cnt));

    // Pre-fill SO buffer with 0xcd sentinel and counter buffer with 0.
    f.run.begin();
    std::vector<std::uint8_t> init_so(so_size, 0xcd);
    std::vector<std::uint8_t> init_cnt(counter_size, 0);
    copycmp::Ptr<ID3D12Resource> st_so, st_cnt;
    CHECK(f.run.buffer(D3D12_HEAP_TYPE_UPLOAD, so_size, D3D12_RESOURCE_STATE_GENERIC_READ, st_so));
    CHECK(f.run.buffer(D3D12_HEAP_TYPE_UPLOAD, counter_size, D3D12_RESOURCE_STATE_GENERIC_READ, st_cnt));
    {
        void* p = nullptr;
        const D3D12_RANGE none{};
        CHECK(SUCCEEDED(st_so->Map(0, &none, &p)));
        std::memcpy(p, init_so.data(), so_size);
        st_so->Unmap(0, nullptr);
        CHECK(SUCCEEDED(st_cnt->Map(0, &none, &p)));
        std::memcpy(p, init_cnt.data(), counter_size);
        st_cnt->Unmap(0, nullptr);
    }
    f.run.list->CopyBufferRegion(so_buf.p, 0, st_so.p, 0, so_size);
    f.run.list->CopyBufferRegion(cnt_buf.p, 0, st_cnt.p, 0, counter_size);
    f.run.barrier(so_buf.p, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_STREAM_OUT);
    f.run.barrier(cnt_buf.p, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_STREAM_OUT);

    // Draw 1 and Draw 2 with NO PS, NO RTV, NO viewport — appends via BufferFilledSizeLocation.
    const D3D12_STREAM_OUTPUT_BUFFER_VIEW so_view{
        so_buf->GetGPUVirtualAddress(), so_size, cnt_buf->GetGPUVirtualAddress()};
    const D3D12_VERTEX_BUFFER_VIEW vbv{vb->GetGPUVirtualAddress(), sizeof(verts), sizeof(SoInputVertex)};
    const float off1[2] = {0.125f, 0.25f};
    const float off2[2] = {-0.25f, 0.375f};

    f.run.list->SetPipelineState(pso.p);
    f.run.list->SetGraphicsRootSignature(f.root.p);
    f.run.list->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    f.run.list->IASetVertexBuffers(0, 1, &vbv);
    f.run.list->SOSetTargets(0, 1, &so_view);
    f.run.list->SetGraphicsRoot32BitConstants(0, 2, off1, 0);
    f.run.list->DrawInstanced(3, 1, 0, 0);

    // Second draw appends from counter == 96 -> 192.
    f.run.list->SetGraphicsRoot32BitConstants(0, 2, off2, 0);
    f.run.list->DrawInstanced(3, 1, 0, 0);

    f.run.barrier(so_buf.p, D3D12_RESOURCE_STATE_STREAM_OUT, D3D12_RESOURCE_STATE_COPY_SOURCE);
    f.run.barrier(cnt_buf.p, D3D12_RESOURCE_STATE_STREAM_OUT, D3D12_RESOURCE_STATE_COPY_SOURCE);
    f.run.list->CopyBufferRegion(rb_so.p, 0, so_buf.p, 0, so_size);
    f.run.list->CopyBufferRegion(rb_cnt.p, 0, cnt_buf.p, 0, counter_size);
    const std::string err1 = f.run.submit();
    if (!err1.empty()) {
        std::fprintf(stderr, "SO append submit (%s): %s\n", f.run.ours ? "Vulkan" : "host", err1.c_str());
        return false;
    }

    {
        void* p = nullptr;
        const D3D12_RANGE range{0, so_size};
        CHECK(SUCCEEDED(rb_so->Map(0, &range, &p)));
        out_so_append.assign(static_cast<const std::uint8_t*>(p), static_cast<const std::uint8_t*>(p) + so_size);
        rb_so->Unmap(0, nullptr);
        const D3D12_RANGE crange{0, 8};
        CHECK(SUCCEEDED(rb_cnt->Map(0, &crange, &p)));
        std::memcpy(&out_counter_append, p, sizeof(out_counter_append));
        rb_cnt->Unmap(0, nullptr);
    }

    // Reset counter to 0, fill SO buffer with 0xcd again, draw with off3 -> counter == 96.
    f.run.begin();
    f.run.barrier(so_buf.p, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COPY_DEST);
    f.run.barrier(cnt_buf.p, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COPY_DEST);
    f.run.list->CopyBufferRegion(so_buf.p, 0, st_so.p, 0, so_size);
    f.run.list->CopyBufferRegion(cnt_buf.p, 0, st_cnt.p, 0, counter_size);
    f.run.barrier(so_buf.p, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_STREAM_OUT);
    f.run.barrier(cnt_buf.p, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_STREAM_OUT);

    const float off3[2] = {0.5f, -0.5f};
    f.run.list->SetPipelineState(pso.p);
    f.run.list->SetGraphicsRootSignature(f.root.p);
    f.run.list->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    f.run.list->IASetVertexBuffers(0, 1, &vbv);
    f.run.list->SOSetTargets(0, 1, &so_view);
    f.run.list->SetGraphicsRoot32BitConstants(0, 2, off3, 0);
    f.run.list->DrawInstanced(3, 1, 0, 0);

    f.run.barrier(so_buf.p, D3D12_RESOURCE_STATE_STREAM_OUT, D3D12_RESOURCE_STATE_COPY_SOURCE);
    f.run.barrier(cnt_buf.p, D3D12_RESOURCE_STATE_STREAM_OUT, D3D12_RESOURCE_STATE_COPY_SOURCE);
    f.run.list->CopyBufferRegion(rb_so.p, 0, so_buf.p, 0, so_size);
    f.run.list->CopyBufferRegion(rb_cnt.p, 0, cnt_buf.p, 0, counter_size);
    const std::string err2 = f.run.submit();
    if (!err2.empty()) {
        std::fprintf(stderr, "SO reset submit (%s): %s\n", f.run.ours ? "Vulkan" : "host", err2.c_str());
        return false;
    }
    {
        void* p = nullptr;
        const D3D12_RANGE range{0, so_size};
        CHECK(SUCCEEDED(rb_so->Map(0, &range, &p)));
        out_so_reset.assign(static_cast<const std::uint8_t*>(p), static_cast<const std::uint8_t*>(p) + so_size);
        rb_so->Unmap(0, nullptr);
        const D3D12_RANGE crange{0, 8};
        CHECK(SUCCEEDED(rb_cnt->Map(0, &crange, &p)));
        std::memcpy(&out_counter_reset, p, sizeof(out_counter_reset));
        rb_cnt->Unmap(0, nullptr);
    }
    return true;
}

bool run_simultaneous_so_and_raster(Fixture& f, std::vector<std::uint8_t>& out_so, std::uint64_t& out_counter,
                                    std::vector<std::uint8_t>& out_pixels) {
    D3D12_GRAPHICS_PIPELINE_STATE_DESC pso_desc = make_pso_desc(f, false, true);
    copycmp::Ptr<ID3D12PipelineState> pso;
    const HRESULT pso_hr = f.run.device->CreateGraphicsPipelineState(&pso_desc, IID_PPV_ARGS(&pso.p));
    if (FAILED(pso_hr)) {
        std::fprintf(stderr, "CreateGraphicsPipelineState (%s) failed: 0x%08x\n",
                     f.run.ours ? "Vulkan" : "host", unsigned(pso_hr));
        return false;
    }

    constexpr UINT size = 16;
    constexpr UINT pitch = 256;
    constexpr UINT so_size = 256;
    const SoInputVertex verts[3] = {
        {-1.0f,  1.0f, 0.5f, 1.0f, 1.0f, 0.0f, 0.0f, 1.0f},
        { 1.0f,  1.0f, 0.5f, 1.0f, 1.0f, 0.0f, 0.0f, 1.0f},
        {-1.0f, -1.0f, 0.5f, 1.0f, 1.0f, 0.0f, 0.0f, 1.0f},
    };
    copycmp::Ptr<ID3D12Resource> vb, so_buf, cnt_buf, target, rb_so, rb_cnt, rb_tex;
    CHECK(f.run.buffer(D3D12_HEAP_TYPE_UPLOAD, sizeof(verts), D3D12_RESOURCE_STATE_GENERIC_READ, vb));
    {
        void* p = nullptr;
        const D3D12_RANGE none{};
        CHECK(SUCCEEDED(vb->Map(0, &none, &p)));
        std::memcpy(p, verts, sizeof(verts));
        vb->Unmap(0, nullptr);
    }
    CHECK(f.run.buffer(D3D12_HEAP_TYPE_DEFAULT, so_size, D3D12_RESOURCE_STATE_COMMON, so_buf));
    CHECK(f.run.buffer(D3D12_HEAP_TYPE_DEFAULT, 256, D3D12_RESOURCE_STATE_COMMON, cnt_buf));
    CHECK(f.run.buffer(D3D12_HEAP_TYPE_READBACK, so_size, D3D12_RESOURCE_STATE_COPY_DEST, rb_so));
    CHECK(f.run.buffer(D3D12_HEAP_TYPE_READBACK, 256, D3D12_RESOURCE_STATE_COPY_DEST, rb_cnt));
    CHECK(f.run.buffer(D3D12_HEAP_TYPE_READBACK, size * pitch, D3D12_RESOURCE_STATE_COPY_DEST, rb_tex));

    D3D12_RESOURCE_DESC rt_desc{};
    rt_desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    rt_desc.Width = rt_desc.Height = size;
    rt_desc.DepthOrArraySize = rt_desc.MipLevels = 1;
    rt_desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    rt_desc.SampleDesc.Count = 1;
    rt_desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
    CHECK(f.run.texture(rt_desc, D3D12_RESOURCE_STATE_RENDER_TARGET, target));

    copycmp::Ptr<ID3D12DescriptorHeap> rtv_heap;
    const D3D12_DESCRIPTOR_HEAP_DESC hd{D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 1};
    CHECK(SUCCEEDED(f.run.device->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&rtv_heap.p))));
    const auto rtv = rtv_heap->GetCPUDescriptorHandleForHeapStart();
    f.run.device->CreateRenderTargetView(target.p, nullptr, rtv);

    f.run.begin();
    const FLOAT clear[4] = {0, 0, 1, 1};
    f.run.list->ClearRenderTargetView(rtv, clear, 0, nullptr);
    f.run.barrier(so_buf.p, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_STREAM_OUT);
    f.run.barrier(cnt_buf.p, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_STREAM_OUT);
    f.run.list->SetPipelineState(pso.p);
    f.run.list->SetGraphicsRootSignature(f.root.p);
    f.run.list->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    const D3D12_VERTEX_BUFFER_VIEW vbv{vb->GetGPUVirtualAddress(), sizeof(verts), sizeof(SoInputVertex)};
    f.run.list->IASetVertexBuffers(0, 1, &vbv);
    const D3D12_VIEWPORT vp{0, 0, float(size), float(size), 0, 1};
    const D3D12_RECT sc{0, 0, LONG(size), LONG(size)};
    f.run.list->RSSetViewports(1, &vp);
    f.run.list->RSSetScissorRects(1, &sc);
    f.run.list->OMSetRenderTargets(1, &rtv, FALSE, nullptr);
    const D3D12_STREAM_OUTPUT_BUFFER_VIEW so_view{
        so_buf->GetGPUVirtualAddress(), so_size, cnt_buf->GetGPUVirtualAddress()};
    f.run.list->SOSetTargets(0, 1, &so_view);
    const float off[2] = {0, 0};
    f.run.list->SetGraphicsRoot32BitConstants(0, 2, off, 0);
    f.run.list->DrawInstanced(3, 1, 0, 0);

    f.run.barrier(so_buf.p, D3D12_RESOURCE_STATE_STREAM_OUT, D3D12_RESOURCE_STATE_COPY_SOURCE);
    f.run.barrier(cnt_buf.p, D3D12_RESOURCE_STATE_STREAM_OUT, D3D12_RESOURCE_STATE_COPY_SOURCE);
    f.run.barrier(target.p, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_COPY_SOURCE);
    f.run.list->CopyBufferRegion(rb_so.p, 0, so_buf.p, 0, so_size);
    f.run.list->CopyBufferRegion(rb_cnt.p, 0, cnt_buf.p, 0, 256);
    D3D12_TEXTURE_COPY_LOCATION dst{};
    dst.pResource = rb_tex.p;
    dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    dst.PlacedFootprint = {0, {DXGI_FORMAT_R8G8B8A8_UNORM, size, size, 1, pitch}};
    D3D12_TEXTURE_COPY_LOCATION src{};
    src.pResource = target.p;
    src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    f.run.list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
    const std::string err = f.run.submit();
    if (!err.empty()) {
        std::fprintf(stderr, "SO + raster submit (%s): %s\n", f.run.ours ? "Vulkan" : "host", err.c_str());
        return false;
    }

    void* p = nullptr;
    const D3D12_RANGE r_so{0, 96};
    CHECK(SUCCEEDED(rb_so->Map(0, &r_so, &p)));
    out_so.assign(static_cast<const std::uint8_t*>(p), static_cast<const std::uint8_t*>(p) + 96);
    rb_so->Unmap(0, nullptr);
    const D3D12_RANGE r_cnt{0, 8};
    CHECK(SUCCEEDED(rb_cnt->Map(0, &r_cnt, &p)));
    std::memcpy(&out_counter, p, sizeof(out_counter));
    rb_cnt->Unmap(0, nullptr);
    out_pixels.resize(size * size * 4);
    const D3D12_RANGE r_tex{0, size * pitch};
    CHECK(SUCCEEDED(rb_tex->Map(0, &r_tex, &p)));
    for (UINT row = 0; row < size; ++row)
        std::memcpy(out_pixels.data() + row * size * 4, static_cast<const std::uint8_t*>(p) + row * pitch, size * 4);
    rb_tex->Unmap(0, nullptr);
    return true;
}

bool run_multi_slot_and_unbind(Fixture& f, std::vector<std::uint8_t>& out_slot0, std::uint64_t& out_cnt0,
                               std::vector<std::uint8_t>& out_slot1, std::uint64_t& out_cnt1) {
    D3D12_GRAPHICS_PIPELINE_STATE_DESC pso_desc = make_pso_desc(f, true, false);
    copycmp::Ptr<ID3D12PipelineState> pso;
    const HRESULT pso_hr = f.run.device->CreateGraphicsPipelineState(&pso_desc, IID_PPV_ARGS(&pso.p));
    if (FAILED(pso_hr)) {
        std::fprintf(stderr, "CreateGraphicsPipelineState (%s) failed: 0x%08x\n",
                     f.run.ours ? "Vulkan" : "host", unsigned(pso_hr));
        return false;
    }

    constexpr UINT buf_size = 256;
    const SoInputVertex verts[3] = {
        {0.0f, 0.5f, 0.1f, 1.0f, 0.25f, 0.5f, 0.75f, 1.0f},
        {0.5f, -0.5f, 0.2f, 1.0f, 0.5f, 0.25f, 1.0f, 1.0f},
        {-0.5f, -0.5f, 0.3f, 1.0f, 1.0f, 0.75f, 0.25f, 1.0f},
    };
    copycmp::Ptr<ID3D12Resource> vb, so0, so1, cnt0, cnt1, rb0, rb1, rbc0, rbc1;
    CHECK(f.run.buffer(D3D12_HEAP_TYPE_UPLOAD, sizeof(verts), D3D12_RESOURCE_STATE_GENERIC_READ, vb));
    {
        void* p = nullptr;
        const D3D12_RANGE none{};
        CHECK(SUCCEEDED(vb->Map(0, &none, &p)));
        std::memcpy(p, verts, sizeof(verts));
        vb->Unmap(0, nullptr);
    }
    CHECK(f.run.buffer(D3D12_HEAP_TYPE_DEFAULT, buf_size, D3D12_RESOURCE_STATE_STREAM_OUT, so0));
    CHECK(f.run.buffer(D3D12_HEAP_TYPE_DEFAULT, buf_size, D3D12_RESOURCE_STATE_STREAM_OUT, so1));
    CHECK(f.run.buffer(D3D12_HEAP_TYPE_DEFAULT, 256, D3D12_RESOURCE_STATE_STREAM_OUT, cnt0));
    CHECK(f.run.buffer(D3D12_HEAP_TYPE_DEFAULT, 256, D3D12_RESOURCE_STATE_STREAM_OUT, cnt1));
    CHECK(f.run.buffer(D3D12_HEAP_TYPE_READBACK, buf_size, D3D12_RESOURCE_STATE_COPY_DEST, rb0));
    CHECK(f.run.buffer(D3D12_HEAP_TYPE_READBACK, buf_size, D3D12_RESOURCE_STATE_COPY_DEST, rb1));
    CHECK(f.run.buffer(D3D12_HEAP_TYPE_READBACK, 256, D3D12_RESOURCE_STATE_COPY_DEST, rbc0));
    CHECK(f.run.buffer(D3D12_HEAP_TYPE_READBACK, 256, D3D12_RESOURCE_STATE_COPY_DEST, rbc1));

    f.run.begin();
    f.run.list->SetPipelineState(pso.p);
    f.run.list->SetGraphicsRootSignature(f.root.p);
    f.run.list->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    const D3D12_VERTEX_BUFFER_VIEW vbv{vb->GetGPUVirtualAddress(), sizeof(verts), sizeof(SoInputVertex)};
    f.run.list->IASetVertexBuffers(0, 1, &vbv);
    const D3D12_STREAM_OUTPUT_BUFFER_VIEW so_views[2] = {
        {so0->GetGPUVirtualAddress(), buf_size, cnt0->GetGPUVirtualAddress()},
        {so1->GetGPUVirtualAddress(), buf_size, cnt1->GetGPUVirtualAddress()},
    };
    f.run.list->SOSetTargets(0, 2, so_views);
    const float off[2] = {0.25f, -0.125f};
    f.run.list->SetGraphicsRoot32BitConstants(0, 2, off, 0);
    f.run.list->DrawInstanced(3, 1, 0, 0);

    // Unbind SO targets with nullptr and draw again (must not modify buffers/counters), then also test ClearState.
    f.run.list->SOSetTargets(0, 2, nullptr);
    const D3D12_VIEWPORT vp{0, 0, 16, 16, 0, 1};
    const D3D12_RECT sc{0, 0, 16, 16};
    f.run.list->RSSetViewports(1, &vp);
    f.run.list->RSSetScissorRects(1, &sc);
    f.run.list->DrawInstanced(3, 1, 0, 0);

    // Re-bind slot 0 only with SizeInBytes == 0 after ClearState -> also must remain unbound/safe.
    f.run.list->SOSetTargets(0, 2, so_views);
    f.run.list->ClearState(pso.p);
    f.run.list->SetGraphicsRootSignature(f.root.p);
    f.run.list->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    f.run.list->IASetVertexBuffers(0, 1, &vbv);
    f.run.list->RSSetViewports(1, &vp);
    f.run.list->RSSetScissorRects(1, &sc);
    const D3D12_STREAM_OUTPUT_BUFFER_VIEW empty_view{so0->GetGPUVirtualAddress(), 0, cnt0->GetGPUVirtualAddress()};
    f.run.list->SOSetTargets(0, 1, &empty_view);
    f.run.list->SetGraphicsRoot32BitConstants(0, 2, off, 0);
    f.run.list->DrawInstanced(3, 1, 0, 0);

    f.run.barrier(so0.p, D3D12_RESOURCE_STATE_STREAM_OUT, D3D12_RESOURCE_STATE_COPY_SOURCE);
    f.run.barrier(so1.p, D3D12_RESOURCE_STATE_STREAM_OUT, D3D12_RESOURCE_STATE_COPY_SOURCE);
    f.run.barrier(cnt0.p, D3D12_RESOURCE_STATE_STREAM_OUT, D3D12_RESOURCE_STATE_COPY_SOURCE);
    f.run.barrier(cnt1.p, D3D12_RESOURCE_STATE_STREAM_OUT, D3D12_RESOURCE_STATE_COPY_SOURCE);
    f.run.list->CopyBufferRegion(rb0.p, 0, so0.p, 0, 48);
    f.run.list->CopyBufferRegion(rb1.p, 0, so1.p, 0, 48);
    f.run.list->CopyBufferRegion(rbc0.p, 0, cnt0.p, 0, 8);
    f.run.list->CopyBufferRegion(rbc1.p, 0, cnt1.p, 0, 8);
    const std::string err = f.run.submit();
    if (!err.empty()) {
        std::fprintf(stderr, "multi-slot/unbind submit (%s): %s\n", f.run.ours ? "Vulkan" : "host", err.c_str());
        return false;
    }

    void* p = nullptr;
    const D3D12_RANGE r48{0, 48};
    const D3D12_RANGE r8{0, 8};
    CHECK(SUCCEEDED(rb0->Map(0, &r48, &p)));
    out_slot0.assign(static_cast<const std::uint8_t*>(p), static_cast<const std::uint8_t*>(p) + 48);
    rb0->Unmap(0, nullptr);
    CHECK(SUCCEEDED(rb1->Map(0, &r48, &p)));
    out_slot1.assign(static_cast<const std::uint8_t*>(p), static_cast<const std::uint8_t*>(p) + 48);
    rb1->Unmap(0, nullptr);
    CHECK(SUCCEEDED(rbc0->Map(0, &r8, &p)));
    std::memcpy(&out_cnt0, p, sizeof(out_cnt0));
    rbc0->Unmap(0, nullptr);
    CHECK(SUCCEEDED(rbc1->Map(0, &r8, &p)));
    std::memcpy(&out_cnt1, p, sizeof(out_cnt1));
    rbc1->Unmap(0, nullptr);
    return true;
}

bool check_bundle_rejection(ID3D12Device* device) {
    copycmp::Ptr<ID3D12CommandAllocator> alloc;
    copycmp::Ptr<ID3D12GraphicsCommandList> bundle;
    CHECK(SUCCEEDED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_BUNDLE, IID_PPV_ARGS(&alloc.p))));
    CHECK(SUCCEEDED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_BUNDLE, alloc.p, nullptr,
                                              IID_PPV_ARGS(&bundle.p))));
    bundle->SOSetTargets(0, 1, nullptr);
    CHECK(bundle->Close() == E_INVALIDARG);
    return true;
}

struct SoStatsResult {
    D3D12_QUERY_DATA_SO_STATISTICS normal{};
    D3D12_QUERY_DATA_SO_STATISTICS overflow{};
    D3D12_QUERY_DATA_SO_STATISTICS stream0{};
    D3D12_QUERY_DATA_SO_STATISTICS stream1{};
    D3D12_QUERY_DATA_SO_STATISTICS unended{};
    UINT64 overflow_counter = 0;
};

bool run_so_statistics_queries(Fixture& f, SoStatsResult& out) {
    D3D12_GRAPHICS_PIPELINE_STATE_DESC pso_desc = make_pso_desc(f, false, false);
    copycmp::Ptr<ID3D12PipelineState> pso;
    const HRESULT pso_hr = f.run.device->CreateGraphicsPipelineState(&pso_desc, IID_PPV_ARGS(&pso.p));
    if (FAILED(pso_hr)) {
        std::fprintf(stderr, "CreateGraphicsPipelineState SO query (%s) failed: 0x%08x\n",
                     f.run.ours ? "Vulkan" : "host", unsigned(pso_hr));
        return false;
    }

    const SoInputVertex two_tris[6] = {
        {-0.5f,  0.5f, 0.25f, 1.0f, 1, 0, 0, 1},
        { 0.5f,  0.5f, 0.25f, 1.0f, 0, 1, 0, 1},
        {-0.5f, -0.5f, 0.25f, 1.0f, 0, 0, 1, 1},
        { 0.0f,  0.5f, 0.25f, 1.0f, 1, 1, 0, 1},
        { 0.5f,  0.0f, 0.25f, 1.0f, 0, 1, 1, 1},
        { 0.0f, -0.5f, 0.25f, 1.0f, 1, 0, 1, 1},
    };
    constexpr UINT buf_size = 512;
    copycmp::Ptr<ID3D12Resource> vb, so_buf, cnt0, cnt1, cnt2, rb_stats, rb_cnt1;
    CHECK(f.run.buffer(D3D12_HEAP_TYPE_UPLOAD, sizeof(two_tris), D3D12_RESOURCE_STATE_GENERIC_READ, vb));
    {
        void* p = nullptr;
        const D3D12_RANGE none{};
        CHECK(SUCCEEDED(vb->Map(0, &none, &p)));
        std::memcpy(p, two_tris, sizeof(two_tris));
        vb->Unmap(0, nullptr);
    }
    CHECK(f.run.buffer(D3D12_HEAP_TYPE_DEFAULT, buf_size, D3D12_RESOURCE_STATE_STREAM_OUT, so_buf));
    CHECK(f.run.buffer(D3D12_HEAP_TYPE_DEFAULT, 256, D3D12_RESOURCE_STATE_COPY_DEST, cnt0));
    CHECK(f.run.buffer(D3D12_HEAP_TYPE_DEFAULT, 256, D3D12_RESOURCE_STATE_COPY_DEST, cnt1));
    CHECK(f.run.buffer(D3D12_HEAP_TYPE_DEFAULT, 256, D3D12_RESOURCE_STATE_COPY_DEST, cnt2));
    CHECK(f.run.buffer(D3D12_HEAP_TYPE_READBACK, 256, D3D12_RESOURCE_STATE_COPY_DEST, rb_stats));
    CHECK(f.run.buffer(D3D12_HEAP_TYPE_READBACK, 256, D3D12_RESOURCE_STATE_COPY_DEST, rb_cnt1));
    copycmp::Ptr<ID3D12Resource> st_zero;
    CHECK(f.run.buffer(D3D12_HEAP_TYPE_UPLOAD, 256, D3D12_RESOURCE_STATE_GENERIC_READ, st_zero));
    {
        void* p = nullptr;
        const D3D12_RANGE none{};
        CHECK(SUCCEEDED(st_zero->Map(0, &none, &p)));
        std::memset(p, 0, 256);
        st_zero->Unmap(0, nullptr);
    }

    copycmp::Ptr<ID3D12QueryHeap> heap;
    const D3D12_QUERY_HEAP_DESC qdesc{D3D12_QUERY_HEAP_TYPE_SO_STATISTICS, 5, 0};
    CHECK(SUCCEEDED(f.run.device->CreateQueryHeap(&qdesc, IID_PPV_ARGS(&heap.p))));

    f.run.begin();
    f.run.list->CopyBufferRegion(cnt0.p, 0, st_zero.p, 0, 256);
    f.run.list->CopyBufferRegion(cnt1.p, 0, st_zero.p, 0, 256);
    f.run.list->CopyBufferRegion(cnt2.p, 0, st_zero.p, 0, 256);
    f.run.barrier(cnt0.p, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_STREAM_OUT);
    f.run.barrier(cnt1.p, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_STREAM_OUT);
    f.run.barrier(cnt2.p, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_STREAM_OUT);
    f.run.list->SetPipelineState(pso.p);
    f.run.list->SetGraphicsRootSignature(f.root.p);
    f.run.list->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    const D3D12_VERTEX_BUFFER_VIEW vbv{vb->GetGPUVirtualAddress(), sizeof(two_tris), sizeof(SoInputVertex)};
    f.run.list->IASetVertexBuffers(0, 1, &vbv);
    const float off[2] = {0.0f, 0.0f};
    f.run.list->SetGraphicsRoot32BitConstants(0, 2, off, 0);

    // Query 0: 2 triangles fit inside 512 bytes.
    const D3D12_STREAM_OUTPUT_BUFFER_VIEW view0{so_buf->GetGPUVirtualAddress(), buf_size, cnt0->GetGPUVirtualAddress()};
    f.run.list->SOSetTargets(0, 1, &view0);
    f.run.list->BeginQuery(heap.p, D3D12_QUERY_TYPE_SO_STATISTICS_STREAM0, 0);
    f.run.list->DrawInstanced(6, 1, 0, 0);
    f.run.list->EndQuery(heap.p, D3D12_QUERY_TYPE_SO_STATISTICS_STREAM0, 0);

    // Query 1: 2 triangles drawn into 128-byte buffer -> fits 1 triangle (96 bytes), overflows 2nd.
    const D3D12_STREAM_OUTPUT_BUFFER_VIEW view1{so_buf->GetGPUVirtualAddress(), 128, cnt1->GetGPUVirtualAddress()};
    f.run.list->SOSetTargets(0, 1, &view1);
    f.run.list->BeginQuery(heap.p, D3D12_QUERY_TYPE_SO_STATISTICS_STREAM0, 1);
    f.run.list->DrawInstanced(6, 1, 0, 0);
    f.run.list->EndQuery(heap.p, D3D12_QUERY_TYPE_SO_STATISTICS_STREAM0, 1);

    // Queries 2 & 3: simultaneous STREAM0 + STREAM1 on a 1-triangle VS draw (only stream 0 receives primitives).
    const D3D12_STREAM_OUTPUT_BUFFER_VIEW view2{so_buf->GetGPUVirtualAddress(), buf_size, cnt2->GetGPUVirtualAddress()};
    f.run.list->SOSetTargets(0, 1, &view2);
    f.run.list->BeginQuery(heap.p, D3D12_QUERY_TYPE_SO_STATISTICS_STREAM0, 2);
    f.run.list->BeginQuery(heap.p, D3D12_QUERY_TYPE_SO_STATISTICS_STREAM1, 3);
    f.run.list->DrawInstanced(3, 1, 0, 0);
    f.run.list->EndQuery(heap.p, D3D12_QUERY_TYPE_SO_STATISTICS_STREAM1, 3);
    f.run.list->EndQuery(heap.p, D3D12_QUERY_TYPE_SO_STATISTICS_STREAM0, 2);

    // Resolve ended slots; D3D12 leaves un-ended query slots undefined on the host, while the layer guarantees zeros.
    if (f.run.ours) {
        f.run.list->ResolveQueryData(heap.p, D3D12_QUERY_TYPE_SO_STATISTICS_STREAM0, 0, 5, rb_stats.p, 0);
    } else {
        f.run.list->ResolveQueryData(heap.p, D3D12_QUERY_TYPE_SO_STATISTICS_STREAM0, 0, 3, rb_stats.p, 0);
        f.run.list->ResolveQueryData(heap.p, D3D12_QUERY_TYPE_SO_STATISTICS_STREAM1, 3, 1, rb_stats.p,
                                     3 * sizeof(D3D12_QUERY_DATA_SO_STATISTICS));
    }
    f.run.barrier(cnt1.p, D3D12_RESOURCE_STATE_STREAM_OUT, D3D12_RESOURCE_STATE_COPY_SOURCE);
    f.run.list->CopyBufferRegion(rb_cnt1.p, 0, cnt1.p, 0, 8);

    const std::string err = f.run.submit();
    if (!err.empty()) {
        std::fprintf(stderr, "SO query submit (%s): %s\n", f.run.ours ? "Vulkan" : "host", err.c_str());
        return false;
    }

    void* p = nullptr;
    const D3D12_RANGE r80{0, 5 * sizeof(D3D12_QUERY_DATA_SO_STATISTICS)};
    CHECK(SUCCEEDED(rb_stats->Map(0, &r80, &p)));
    D3D12_QUERY_DATA_SO_STATISTICS stats[5]{};
    std::memcpy(stats, p, sizeof(stats));
    rb_stats->Unmap(0, nullptr);
    out.normal = stats[0];
    out.overflow = stats[1];
    out.stream0 = stats[2];
    out.stream1 = stats[3];
    out.unended = stats[4];

    const D3D12_RANGE r8{0, 8};
    CHECK(SUCCEEDED(rb_cnt1->Map(0, &r8, &p)));
    std::memcpy(&out.overflow_counter, p, sizeof(out.overflow_counter));
    rb_cnt1->Unmap(0, nullptr);
    return true;
}

bool check_float_eq(const std::vector<std::uint8_t>& buf, std::size_t byte_offset,
                    const std::array<float, 8>& expected) {
    float got[8] = {};
    std::memcpy(got, buf.data() + byte_offset, sizeof(got));
    for (std::size_t i = 0; i < 8; ++i) {
        if (std::fabs(got[i] - expected[i]) > 1e-5f) {
            std::fprintf(stderr, "SO float[%zu] at +%zu: got %.6f want %.6f\n",
                         i, byte_offset, got[i], expected[i]);
            return false;
        }
    }
    return true;
}

}  // namespace

int main() {
    vktest::Device d;
    d.request_buffer_device_address = true;
    if (!d.create()) {
        std::puts("SKIP: no suitable Vulkan 1.3 (or 1.2 + KHR) device");
        return 77;
    }
    if (!d.buffer_device_address) {
        std::puts("SKIP: bufferDeviceAddress is required");
        return 77;
    }
    if (!d.transform_feedback) {
        std::puts("SKIP: VK_EXT_transform_feedback is not supported on this device");
        return 77;
    }

    // Verify standalone device populates transform_feedback capability when available.
    {
        copycmp::Ptr<ID3D12Device> standalone;
        if (SUCCEEDED(rt::d3d12::create_vulkan_device_standalone(IID_PPV_ARGS(&standalone.p)))) {
            const auto* sinfo = rt::d3d12::vulkan_device_info(standalone.p);
            REQUIRE(sinfo != nullptr);
            if (sinfo->physical == d.physical) REQUIRE(sinfo->transform_feedback);
        }
    }

    const auto vs_single_prog = d3d12test::fssm_program({0x534f5631, 0xbf810000});
    const auto vs_multi_prog = d3d12test::fssm_program({0x534f5632, 0xbf810000});
    const auto ps_prog = d3d12test::fssm_program({0x534f5031, 0xbf810000});
    const auto vs_single_bytes = d3d12test::shader_container(vs_single_prog);
    const auto vs_multi_bytes = d3d12test::shader_container(vs_multi_prog);
    const auto ps_bytes = d3d12test::shader_container(ps_prog);

    const auto library = std::filesystem::temp_directory_path() /
                         ("leonida_so_test_" + std::to_string(GetCurrentProcessId()));
    std::filesystem::create_directories(library);
    std::filesystem::copy_file(LEONIDA_TEST_VS_SO_SPV, library / "vs_so.spv",
                               std::filesystem::copy_options::overwrite_existing);
    std::filesystem::copy_file(LEONIDA_TEST_VS_SO_MULTI_SPV, library / "vs_so_multi.spv",
                               std::filesystem::copy_options::overwrite_existing);
    std::filesystem::copy_file(LEONIDA_TEST_PS_SO_SPV, library / "ps_so.spv",
                               std::filesystem::copy_options::overwrite_existing);
    {
        const char* sig =
            "[{\"semantic\":\"POSITION\",\"semantic_index\":0,\"register\":0,\"mask\":15,\"rw_mask\":15,"
            "\"system_value\":0,\"component_type\":\"float\"},"
            "{\"semantic\":\"COLOR\",\"semantic_index\":0,\"register\":1,\"mask\":15,\"rw_mask\":15,"
            "\"system_value\":0,\"component_type\":\"float\"}]";
        std::ofstream index(library / "index.json", std::ios::binary);
        index << "{\"vs_so\":{\"stage\":\"vs\",\"fssm_sha256\":\"" << leonida::sha256_hex(vs_single_prog)
              << "\",\"bindings\":[],\"push_constant_bytes\":144,\"input_signature\":" << sig << "},"
                 "\"vs_so_multi\":{\"stage\":\"vs\",\"fssm_sha256\":\"" << leonida::sha256_hex(vs_multi_prog)
              << "\",\"bindings\":[],\"push_constant_bytes\":144,\"input_signature\":" << sig << "},"
                 "\"ps_so\":{\"stage\":\"ps\",\"fssm_sha256\":\"" << leonida::sha256_hex(ps_prog)
              << "\",\"bindings\":[],\"push_constant_bytes\":144,\"outputs\":[0]}}";
    }

    rt::d3d12::VulkanDeviceInfo info{};
    info.get_instance_proc = d.instance_proc;
    info.instance = d.instance;
    info.physical = d.physical;
    info.device = d.device;
    info.queue = d.queue;
    info.queue_family = d.family;
    info.buffer_device_address = d.buffer_device_address;
    info.features = d.enabled;
    info.subgroup = d.subgroup;
    info.transform_feedback = d.transform_feedback;
    info.geometry_streams = d.geometry_streams;
    info.transform_feedback_queries = d.transform_feedback_queries;
    info.transform_feedback_rasterization_stream_select = d.transform_feedback_rasterization_stream_select;
    info.max_transform_feedback_buffers = d.max_transform_feedback_buffers;
    info.max_transform_feedback_streams = d.max_transform_feedback_streams;
    info.max_transform_feedback_buffer_size = d.max_transform_feedback_buffer_size;
    info.shader_dir = library.string();

    int rc = 0;
    {
        copycmp::Ptr<ID3D12Device> ours;
        REQUIRE(SUCCEEDED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&ours.p))));

        // Without VK_EXT_transform_feedback enabled in VulkanDeviceInfo, StreamOutput PSOs return E_NOTIMPL.
        {
            rt::d3d12::VulkanDeviceInfo no_tf = info;
            no_tf.transform_feedback = false;
            no_tf.geometry_streams = false;
            copycmp::Ptr<ID3D12Device> dev_no_tf;
            REQUIRE(SUCCEEDED(rt::d3d12::create_vulkan_device(no_tf, IID_PPV_ARGS(&dev_no_tf.p))));
            Fixture f_no_tf;
            REQUIRE(init_fixture(f_no_tf, dev_no_tf.p, true,
                                 {vs_single_bytes.data(), vs_single_bytes.size()},
                                 {vs_multi_bytes.data(), vs_multi_bytes.size()},
                                 {ps_bytes.data(), ps_bytes.size()}));
            auto desc = make_pso_desc(f_no_tf, false, false);
            copycmp::Ptr<ID3D12PipelineState> pso;
            REQUIRE(dev_no_tf->CreateGraphicsPipelineState(&desc, IID_PPV_ARGS(&pso.p)) == E_NOTIMPL && !pso.p);
        }

        Fixture f_ours;
        REQUIRE(init_fixture(f_ours, ours.p, true,
                             {vs_single_bytes.data(), vs_single_bytes.size()},
                             {vs_multi_bytes.data(), vs_multi_bytes.size()},
                             {ps_bytes.data(), ps_bytes.size()}));

        // Optional native Windows D3D12 host comparison (skipped cleanly under Wine or without d3dcompiler_47).
        copycmp::Ptr<ID3D12Device> host;
        std::string host_name;
        Fixture f_host;
        copycmp::Ptr<ID3DBlob> hlsl_vs, hlsl_ps, hlsl_err;
        bool has_host = false;
        if (const auto* actual_info = rt::d3d12::vulkan_device_info(ours.p)) {
            VkPhysicalDeviceIDProperties id{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES};
            VkPhysicalDeviceProperties2 props2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &id};
            reinterpret_cast<PFN_vkGetPhysicalDeviceProperties2>(
                d.instance_proc(d.instance, "vkGetPhysicalDeviceProperties2"))(actual_info->physical, &props2);
            if (id.deviceLUIDValid) {
                LUID luid{};
                std::memcpy(&luid, id.deviceLUID, sizeof(luid));
                copycmp::layer_adapter() = luid;
            }
        }
        const HMODULE compiler = LoadLibraryExA("d3dcompiler_47.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
        const auto compile = compiler
            ? reinterpret_cast<pD3DCompile>(reinterpret_cast<void*>(GetProcAddress(compiler, "D3DCompile")))
            : nullptr;
        if (compile && copycmp::create_host(host, host_name)) {
            const char* hlsl =
                "cbuffer Push : register(b0) { float2 offset; };\n"
                "struct VSIn { float4 pos : POSITION; float4 color : COLOR; };\n"
                "struct VSOut { float4 pos : TEXCOORD0; float4 color : COLOR0; float4 sv : SV_Position; };\n"
                "VSOut vs(VSIn i) {\n"
                "  VSOut o; float4 t = float4(i.pos.xy + offset, i.pos.zw);\n"
                "  o.sv = t; o.pos = t; o.color = i.color; return o;\n"
                "}\n"
                "float4 ps(VSOut i) : SV_Target { return i.color; }\n";
            if (SUCCEEDED(compile(hlsl, std::strlen(hlsl), nullptr, nullptr, nullptr, "vs", "vs_5_0", 0, 0,
                                  &hlsl_vs.p, &hlsl_err.p)) &&
                SUCCEEDED(compile(hlsl, std::strlen(hlsl), nullptr, nullptr, nullptr, "ps", "ps_5_0", 0, 0,
                                  &hlsl_ps.p, nullptr))) {
                REQUIRE(init_fixture(f_host, host.p, false,
                                     {hlsl_vs->GetBufferPointer(), hlsl_vs->GetBufferSize()},
                                     {hlsl_vs->GetBufferPointer(), hlsl_vs->GetBufferSize()},
                                     {hlsl_ps->GetBufferPointer(), hlsl_ps->GetBufferSize()}));
                has_host = true;
            }
        }

        if (!check_validation_cases(f_ours, has_host ? &f_host : nullptr)) rc = 1;
        if (!check_bundle_rejection(ours.p)) rc = 1;

        std::vector<std::uint8_t> ours_append, ours_reset, host_append, host_reset;
        std::uint64_t ours_cnt_append = 0, ours_cnt_reset = 0, host_cnt_append = 0, host_cnt_reset = 0;
        if (!run_single_slot_so_append_and_reset(f_ours, ours_append, ours_cnt_append, ours_reset, ours_cnt_reset))
            rc = 1;
        if (ours_cnt_append != 192 || ours_cnt_reset != 96) {
            std::fprintf(stderr, "counter mismatch: append=%llu reset=%llu\n",
                         ours_cnt_append, ours_cnt_reset);
            rc = 1;
        }
        if (rc == 0) {
            // Check exact captured floats (draw 1 off=(0.125,0.25), draw 2 off=(-0.25,0.375), reset off=(0.5,-0.5)).
            bool ok = true;
            ok &= check_float_eq(ours_append, 0,   {-0.5f + 0.125f,  0.5f + 0.25f, 0.25f, 1.0f, 1, 0, 0, 1});
            ok &= check_float_eq(ours_append, 32,  { 0.5f + 0.125f,  0.5f + 0.25f, 0.25f, 1.0f, 0, 1, 0, 1});
            ok &= check_float_eq(ours_append, 64,  {-0.5f + 0.125f, -0.5f + 0.25f, 0.25f, 1.0f, 0, 0, 1, 1});
            ok &= check_float_eq(ours_append, 96,  {-0.5f - 0.25f,   0.5f + 0.375f, 0.25f, 1.0f, 1, 0, 0, 1});
            ok &= check_float_eq(ours_append, 128, { 0.5f - 0.25f,   0.5f + 0.375f, 0.25f, 1.0f, 0, 1, 0, 1});
            ok &= check_float_eq(ours_append, 160, {-0.5f - 0.25f,  -0.5f + 0.375f, 0.25f, 1.0f, 0, 0, 1, 1});
            ok &= std::all_of(ours_append.begin() + 192, ours_append.end(), [](std::uint8_t b) { return b == 0xcd; });
            ok &= check_float_eq(ours_reset, 0,  { 0.0f,  0.0f, 0.25f, 1.0f, 1, 0, 0, 1});
            ok &= check_float_eq(ours_reset, 32, { 1.0f,  0.0f, 0.25f, 1.0f, 0, 1, 0, 1});
            ok &= check_float_eq(ours_reset, 64, { 0.0f, -1.0f, 0.25f, 1.0f, 0, 0, 1, 1});
            ok &= std::all_of(ours_reset.begin() + 96, ours_reset.end(), [](std::uint8_t b) { return b == 0xcd; });
            if (!ok) rc = 1;
        }
        if (has_host) {
            if (!run_single_slot_so_append_and_reset(f_host, host_append, host_cnt_append, host_reset, host_cnt_reset))
                rc = 1;
            else if (ours_cnt_append != host_cnt_append || ours_cnt_reset != host_cnt_reset ||
                     ours_append != host_append || ours_reset != host_reset) {
                std::fprintf(stderr, "SO append/reset differed from host D3D12 (%s)\n", host_name.c_str());
                rc = 1;
            }
        }

        // Simultaneous Stream Output + Rasterization (RasterizedStream == 0).
        std::vector<std::uint8_t> ours_sim_so, ours_sim_px, host_sim_so, host_sim_px;
        std::uint64_t ours_sim_cnt = 0, host_sim_cnt = 0;
        if (!run_simultaneous_so_and_raster(f_ours, ours_sim_so, ours_sim_cnt, ours_sim_px) ||
            ours_sim_cnt != 96) {
            rc = 1;
        }
        if (has_host) {
            if (!run_simultaneous_so_and_raster(f_host, host_sim_so, host_sim_cnt, host_sim_px)) {
                rc = 1;
            } else if (ours_sim_cnt != host_sim_cnt || ours_sim_so != host_sim_so || ours_sim_px != host_sim_px) {
                std::fprintf(stderr, "SO + raster differed from host D3D12 (%s)\n", host_name.c_str());
                rc = 1;
            }
        } else if (rc == 0) {
            // Check corner pixels when host is absent: top-left pixel is red, bottom-right is blue clear.
            if (ours_sim_px[0] != 255 || ours_sim_px[1] != 0 || ours_sim_px[2] != 0 ||
                ours_sim_px.back() != 255 || ours_sim_px[ours_sim_px.size() - 2] != 255) {
                std::fprintf(stderr, "simultaneous SO+raster pixels wrong without host\n");
                rc = 1;
            }
        }

        // Multi-slot SO (slots 0 & 1) + SOSetTargets unbind / ClearState.
        std::vector<std::uint8_t> ours_s0, ours_s1, host_s0, host_s1;
        std::uint64_t ours_c0 = 0, ours_c1 = 0, host_c0 = 0, host_c1 = 0;
        if (!run_multi_slot_and_unbind(f_ours, ours_s0, ours_c0, ours_s1, ours_c1) ||
            ours_c0 != 48 || ours_c1 != 48) {
            std::fprintf(stderr, "multi-slot counters wrong: c0=%llu c1=%llu\n", ours_c0, ours_c1);
            rc = 1;
        }
        if (has_host) {
            if (!run_multi_slot_and_unbind(f_host, host_s0, host_c0, host_s1, host_c1)) {
                rc = 1;
            } else if (ours_c0 != host_c0 || ours_c1 != host_c1 || ours_s0 != host_s0 || ours_s1 != host_s1) {
                std::fprintf(stderr, "multi-slot SO / unbind differed from host D3D12 (%s)\n", host_name.c_str());
                rc = 1;
            }
        }

        // Hardware SO_STATISTICS queries (normal, overflow, simultaneous multi-stream, un-ended, and zero fallback).
        if (d.transform_feedback_queries) {
            SoStatsResult ours_q{};
            if (!run_so_statistics_queries(f_ours, ours_q)) {
                rc = 1;
            } else {
                const auto eq = [](const D3D12_QUERY_DATA_SO_STATISTICS& a, UINT64 written, UINT64 needed) {
                    return a.NumPrimitivesWritten == written && a.PrimitivesStorageNeeded == needed;
                };
                if (!eq(ours_q.normal, 2, 2) || !eq(ours_q.overflow, 1, 2) || ours_q.overflow_counter != 96 ||
                    !eq(ours_q.stream0, 1, 1) || !eq(ours_q.stream1, 0, 0) || !eq(ours_q.unended, 0, 0)) {
                    std::fprintf(stderr,
                                 "SO_STATISTICS unexpected: normal=(%llu,%llu) overflow=(%llu,%llu) cnt=%llu "
                                 "stream0=(%llu,%llu) stream1=(%llu,%llu) unended=(%llu,%llu)\n",
                                 ours_q.normal.NumPrimitivesWritten, ours_q.normal.PrimitivesStorageNeeded,
                                 ours_q.overflow.NumPrimitivesWritten, ours_q.overflow.PrimitivesStorageNeeded,
                                 ours_q.overflow_counter,
                                 ours_q.stream0.NumPrimitivesWritten, ours_q.stream0.PrimitivesStorageNeeded,
                                 ours_q.stream1.NumPrimitivesWritten, ours_q.stream1.PrimitivesStorageNeeded,
                                 ours_q.unended.NumPrimitivesWritten, ours_q.unended.PrimitivesStorageNeeded);
                    rc = 1;
                }
            }
            if (has_host && rc == 0) {
                SoStatsResult host_q{};
                if (!run_so_statistics_queries(f_host, host_q)) {
                    rc = 1;
                } else {
                    const auto qeq = [](const D3D12_QUERY_DATA_SO_STATISTICS& a,
                                        const D3D12_QUERY_DATA_SO_STATISTICS& b) {
                        return a.NumPrimitivesWritten == b.NumPrimitivesWritten &&
                               a.PrimitivesStorageNeeded == b.PrimitivesStorageNeeded;
                    };
                    if (!qeq(ours_q.normal, host_q.normal) || !qeq(ours_q.overflow, host_q.overflow) ||
                        !qeq(ours_q.stream0, host_q.stream0) || !qeq(ours_q.stream1, host_q.stream1) ||
                        ours_q.overflow_counter != host_q.overflow_counter) {
                        std::fprintf(stderr, "SO_STATISTICS differed from host D3D12 (%s)\n", host_name.c_str());
                        rc = 1;
                    }
                }
            }
            // Zero-fill fallback when transform_feedback_queries is disabled.
            if (rc == 0) {
                rt::d3d12::VulkanDeviceInfo no_tfq = info;
                no_tfq.transform_feedback_queries = false;
                copycmp::Ptr<ID3D12Device> dev_no_tfq;
                REQUIRE(SUCCEEDED(rt::d3d12::create_vulkan_device(no_tfq, IID_PPV_ARGS(&dev_no_tfq.p))));
                Fixture f_no_tfq;
                REQUIRE(init_fixture(f_no_tfq, dev_no_tfq.p, true,
                                     {vs_single_bytes.data(), vs_single_bytes.size()},
                                     {vs_multi_bytes.data(), vs_multi_bytes.size()},
                                     {ps_bytes.data(), ps_bytes.size()}));
                SoStatsResult fb_q{};
                if (!run_so_statistics_queries(f_no_tfq, fb_q) ||
                    fb_q.normal.NumPrimitivesWritten != 0 || fb_q.normal.PrimitivesStorageNeeded != 0 ||
                    fb_q.overflow.NumPrimitivesWritten != 0 || fb_q.overflow.PrimitivesStorageNeeded != 0) {
                    std::fprintf(stderr, "SO_STATISTICS fallback without transformFeedbackQueries failed\n");
                    rc = 1;
                }
            }
        }

        if (rc == 0) {
            std::printf("stream output: validation parity (11 cases), append/reset counters (%llu/%llu), "
                        "simultaneous SO+raster, multi-slot (2 buffers), unbind/ClearState, bundle rejection%s%s passed\n",
                        ours_cnt_append, ours_cnt_reset,
                        d.transform_feedback_queries ? ", hardware SO_STATISTICS (normal/overflow/multi-stream/fallback)" : "",
                        has_host ? (" (matched host " + host_name + ")").c_str() : "");
        }
    }

    std::error_code ec;
    std::filesystem::remove_all(library, ec);
    if (d.validation_errors()) {
        std::fprintf(stderr, "%u Vulkan validation error(s)\n", d.validation_errors());
        return 1;
    }
    return rc;
}
