// Command-list queue-type recording restrictions (DIRECT vs COMPUTE vs COPY): what Close() returns for every gated
// graphics, compute, pipeline, descriptor-heap, indirect, and raytracing call on the Vulkan layer and on Windows' own
// D3D12 (no debug layer; default hardware adapter and WARP). The layer is held to the exact Close() pass/fail verdict
// of the host wherever the host adapters agree, with 0 disagreements, plus positive end-to-end GPU execution controls
// on COPY and COMPUTE queues under Vulkan validation.
// Both host adapters always run (LEONIDA_HOST_ADAPTER is not read), except under Wine, where the host is skipped.
// No game data.
#include "shader_fixture_path.h"
#include "test_temp_dir.h"
#include "vk_test_device.h"
#include "d3d12_test_shaders.h"
#include "host_wine.h"
#include "../shims/d3d12_com.h"
#include "../src/shader_library.h"

#include <d3dcompiler.h>
#include <dxgi1_4.h>

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <functional>
#include <string>
#include <tuple>
#include <vector>

#if !defined(LEONIDA_TEST_CS_SPV) || !defined(LEONIDA_TEST_VS_SPV) || !defined(LEONIDA_TEST_PS_SPV)
#error "LEONIDA_TEST_CS_SPV, LEONIDA_TEST_VS_SPV and LEONIDA_TEST_PS_SPV must name the compiled test shaders"
#endif

#define CHECK(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return false; } \
} while (0)

#define REQUIRE(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return 1; } \
} while (0)

using namespace d3d12test;

namespace {

constexpr HRESULT kSkipped = HRESULT(0x7fff0001);

bool compile_hlsl(const char* source, const char* entry, const char* profile, std::vector<std::uint8_t>& out) {
    const HMODULE compiler = LoadLibraryExA("d3dcompiler_47.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
    const auto compile = compiler ? reinterpret_cast<pD3DCompile>(
                                        reinterpret_cast<void*>(GetProcAddress(compiler, "D3DCompile")))
                                  : nullptr;
    if (!compile) return false;
    Ref<ID3DBlob> blob, errors;
    const HRESULT hr = compile(source, std::strlen(source), "qtype.hlsl", nullptr, nullptr, entry, profile, 0, 0,
                               &blob.p, &errors.p);
    if (FAILED(hr) || !blob.p) return false;
    const auto* p = static_cast<const std::uint8_t*>(blob->GetBufferPointer());
    out.assign(p, p + blob->GetBufferSize());
    return true;
}

struct Env {
    ID3D12Device* device = nullptr;
    Ref<ID3D12Resource> rt, rt_ms, rt_res, ds, buf, buf2, arg_buf, as_dst, as_src, as_scratch, as_post;
    Ref<ID3D12DescriptorHeap> rtv_heap, dsv_heap, cbv_heap;
    Ref<ID3D12RootSignature> rs;
    Ref<ID3D12PipelineState> cs_pso, gs_pso;
    Ref<ID3D12CommandSignature> csig_draw, csig_dispatch;
    D3D12_CPU_DESCRIPTOR_HANDLE rtv{}, dsv{};
    D3D12_GPU_DESCRIPTOR_HANDLE cbv_gpu{};
    bool has_rt = false;
    bool has_pso = false;

    bool init(ID3D12Device* d, bool is_layer, const std::vector<std::uint8_t>& cs_code,
              const std::vector<std::uint8_t>& vs_code, const std::vector<std::uint8_t>& ps_code) {
        device = d;
        if (is_layer) {
            has_rt = true;  // command-list recording of RT calls does not require Vulkan RT extensions
        } else {
            D3D12_FEATURE_DATA_D3D12_OPTIONS5 opt5{};
            if (SUCCEEDED(device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS5, &opt5, sizeof(opt5))))
                has_rt = opt5.RaytracingTier >= D3D12_RAYTRACING_TIER_1_0;
        }

        const D3D12_HEAP_PROPERTIES default_heap{D3D12_HEAP_TYPE_DEFAULT};
        auto mk_buf = [&](UINT64 size, D3D12_RESOURCE_FLAGS flags, D3D12_RESOURCE_STATES state,
                          ID3D12Resource** out) -> bool {
            const auto rd = buffer_desc(size, flags);
            return SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &rd, state, nullptr,
                                                             IID_PPV_ARGS(out)));
        };
        CHECK(mk_buf(4096, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON, &buf.p));
        CHECK(mk_buf(4096, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON, &buf2.p));
        CHECK(mk_buf(4096, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT, &arg_buf.p));
        if (has_rt) {
            const D3D12_RESOURCE_STATES as_state =
                is_layer ? D3D12_RESOURCE_STATE_COMMON : D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE;
            CHECK(mk_buf(65536, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, as_state, &as_dst.p));
            CHECK(mk_buf(65536, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, as_state, &as_src.p));
            CHECK(mk_buf(65536, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON,
                         &as_scratch.p));
            CHECK(mk_buf(4096, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON, &as_post.p));
        }

        D3D12_RESOURCE_DESC tex{};
        tex.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        tex.Width = 16;
        tex.Height = 16;
        tex.DepthOrArraySize = 1;
        tex.MipLevels = 1;
        tex.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        tex.SampleDesc.Count = 1;
        tex.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
        CHECK(SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &tex,
                                                        D3D12_RESOURCE_STATE_RENDER_TARGET, nullptr,
                                                        IID_PPV_ARGS(&rt.p))));

        tex.SampleDesc.Count = 4;
        CHECK(SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &tex,
                                                        D3D12_RESOURCE_STATE_RESOLVE_SOURCE, nullptr,
                                                        IID_PPV_ARGS(&rt_ms.p))));
        tex.SampleDesc.Count = 1;
        tex.Flags = D3D12_RESOURCE_FLAG_NONE;
        CHECK(SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &tex,
                                                        D3D12_RESOURCE_STATE_RESOLVE_DEST, nullptr,
                                                        IID_PPV_ARGS(&rt_res.p))));

        tex.Format = DXGI_FORMAT_D32_FLOAT;
        tex.Flags = D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;
        CHECK(SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &tex,
                                                        D3D12_RESOURCE_STATE_DEPTH_WRITE, nullptr,
                                                        IID_PPV_ARGS(&ds.p))));

        D3D12_DESCRIPTOR_HEAP_DESC hd{D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 1, D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0};
        CHECK(SUCCEEDED(device->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&rtv_heap.p))));
        rtv = rtv_heap->GetCPUDescriptorHandleForHeapStart();
        device->CreateRenderTargetView(rt.p, nullptr, rtv);

        hd.Type = D3D12_DESCRIPTOR_HEAP_TYPE_DSV;
        CHECK(SUCCEEDED(device->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&dsv_heap.p))));
        dsv = dsv_heap->GetCPUDescriptorHandleForHeapStart();
        device->CreateDepthStencilView(ds.p, nullptr, dsv);

        hd.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
        hd.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
        CHECK(SUCCEEDED(device->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&cbv_heap.p))));
        cbv_gpu = cbv_heap->GetGPUDescriptorHandleForHeapStart();

        // Root signature: [0]=32bit constants(4), [1]=CBV, [2]=SRV, [3]=UAV, [4]=table(SRV).
        D3D12_DESCRIPTOR_RANGE1 range{};
        range.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
        range.NumDescriptors = 1;
        range.BaseShaderRegister = 1;
        D3D12_ROOT_PARAMETER1 params[5]{};
        params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
        params[0].Constants = {0, 0, 4};
        params[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
        params[1].Descriptor = {1, 0, D3D12_ROOT_DESCRIPTOR_FLAG_NONE};
        params[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
        params[2].Descriptor = {0, 0, D3D12_ROOT_DESCRIPTOR_FLAG_NONE};
        params[3].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
        params[3].Descriptor = {0, 0, D3D12_ROOT_DESCRIPTOR_FLAG_NONE};
        params[4].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        params[4].DescriptorTable = {1, &range};
        D3D12_VERSIONED_ROOT_SIGNATURE_DESC rsd{};
        rsd.Version = D3D_ROOT_SIGNATURE_VERSION_1_1;
        rsd.Desc_1_1 = {5, params, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_NONE};
        const auto blob = serialize_root_signature(rsd);
        CHECK(!blob.empty());
        CHECK(SUCCEEDED(device->CreateRootSignature(0, blob.data(), blob.size(), IID_PPV_ARGS(&rs.p))));

        D3D12_COMPUTE_PIPELINE_STATE_DESC cpsd{};
        cpsd.pRootSignature = rs.p;
        cpsd.CS = {cs_code.data(), cs_code.size()};
        const HRESULT cs_hr = device->CreateComputePipelineState(&cpsd, IID_PPV_ARGS(&cs_pso.p));

        static const D3D12_INPUT_ELEMENT_DESC elements[2] = {
            {"POSITION", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 0, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
            {"COLOR", 0, DXGI_FORMAT_R8G8B8A8_UNORM, 0, D3D12_APPEND_ALIGNED_ELEMENT,
             D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0}};
        D3D12_GRAPHICS_PIPELINE_STATE_DESC gpsd{};
        gpsd.pRootSignature = rs.p;
        gpsd.VS = {vs_code.data(), vs_code.size()};
        gpsd.PS = {ps_code.data(), ps_code.size()};
        if (is_layer) gpsd.InputLayout = {elements, 2};
        gpsd.BlendState.RenderTarget[0].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
        gpsd.SampleMask = UINT_MAX;
        gpsd.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
        gpsd.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
        gpsd.RasterizerState.DepthClipEnable = TRUE;
        gpsd.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
        gpsd.NumRenderTargets = 1;
        gpsd.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
        gpsd.SampleDesc.Count = 1;
        const HRESULT gs_hr = device->CreateGraphicsPipelineState(&gpsd, IID_PPV_ARGS(&gs_pso.p));
        // On GPUs with maxPushConstantsSize < 144 (e.g. AMD iGPUs), the layer declines PSO creation; non-PSO calls
        // and queue-type refusals still run in full, while host devices must always create both PSOs.
        if (!is_layer) {
            CHECK(SUCCEEDED(cs_hr));
            CHECK(SUCCEEDED(gs_hr));
        }
        has_pso = SUCCEEDED(cs_hr) && SUCCEEDED(gs_hr);

        D3D12_INDIRECT_ARGUMENT_DESC iad{};
        iad.Type = D3D12_INDIRECT_ARGUMENT_TYPE_DRAW;
        D3D12_COMMAND_SIGNATURE_DESC csd{sizeof(D3D12_DRAW_ARGUMENTS), 1, &iad, 0};
        CHECK(SUCCEEDED(device->CreateCommandSignature(&csd, nullptr, IID_PPV_ARGS(&csig_draw.p))));
        iad.Type = D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH;
        csd.ByteStride = sizeof(D3D12_DISPATCH_ARGUMENTS);
        CHECK(SUCCEEDED(device->CreateCommandSignature(&csd, nullptr, IID_PPV_ARGS(&csig_dispatch.p))));
        return true;
    }
};

struct Call {
    const char* name;
    bool needs_rt;
    bool needs_pso;
    bool expected_direct;
    bool expected_compute;
    bool expected_copy;
    std::function<void(Env&, ID3D12GraphicsCommandList8*)> record;
};

struct Verdict {
    HRESULT direct_close = E_FAIL;
    HRESULT compute_close = E_FAIL;
    HRESULT copy_close = E_FAIL;
};

std::vector<Call> calls() {
    return {
        // Graphics-only calls: allowed on DIRECT, forbidden on COMPUTE and COPY.
        {"DrawInstanced", false, true, true, false, false, [](Env& e, ID3D12GraphicsCommandList8* l) {
            if (l->GetType() == D3D12_COMMAND_LIST_TYPE_DIRECT) {
                l->SetGraphicsRootSignature(e.rs.p);
                l->SetPipelineState(e.gs_pso.p);
                l->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            }
            l->DrawInstanced(3, 1, 0, 0);
        }},
        {"DrawIndexedInstanced", false, true, true, false, false, [](Env& e, ID3D12GraphicsCommandList8* l) {
            if (l->GetType() == D3D12_COMMAND_LIST_TYPE_DIRECT) {
                l->SetGraphicsRootSignature(e.rs.p);
                l->SetPipelineState(e.gs_pso.p);
                l->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
                const D3D12_INDEX_BUFFER_VIEW ibv{e.buf->GetGPUVirtualAddress(), 12, DXGI_FORMAT_R32_UINT};
                l->IASetIndexBuffer(&ibv);
            }
            l->DrawIndexedInstanced(3, 1, 0, 0, 0);
        }},
        {"ClearRenderTargetView", false, false, true, false, false, [](Env& e, ID3D12GraphicsCommandList8* l) {
            const FLOAT color[4] = {0, 0, 0, 1};
            l->ClearRenderTargetView(e.rtv, color, 0, nullptr);
        }},
        {"ClearDepthStencilView", false, false, true, false, false, [](Env& e, ID3D12GraphicsCommandList8* l) {
            l->ClearDepthStencilView(e.dsv, D3D12_CLEAR_FLAG_DEPTH, 1.0f, 0, 0, nullptr);
        }},
        {"ResolveSubresource", false, false, true, false, false, [](Env& e, ID3D12GraphicsCommandList8* l) {
            l->ResolveSubresource(e.rt_res.p, 0, e.rt_ms.p, 0, DXGI_FORMAT_R8G8B8A8_UNORM);
        }},
        {"SetGraphicsRootSignature", false, false, true, false, false, [](Env& e, ID3D12GraphicsCommandList8* l) {
            l->SetGraphicsRootSignature(e.rs.p);
        }},
        {"SetGraphicsRootDescriptorTable", false, false, true, false, false, [](Env& e, ID3D12GraphicsCommandList8* l) {
            if (l->GetType() == D3D12_COMMAND_LIST_TYPE_DIRECT) {
                ID3D12DescriptorHeap* h = e.cbv_heap.p;
                l->SetDescriptorHeaps(1, &h);
                l->SetGraphicsRootSignature(e.rs.p);
            }
            l->SetGraphicsRootDescriptorTable(4, e.cbv_gpu);
        }},
        {"SetGraphicsRoot32BitConstant", false, false, true, false, false, [](Env& e, ID3D12GraphicsCommandList8* l) {
            if (l->GetType() == D3D12_COMMAND_LIST_TYPE_DIRECT) l->SetGraphicsRootSignature(e.rs.p);
            l->SetGraphicsRoot32BitConstant(0, 1, 0);
        }},
        {"SetGraphicsRoot32BitConstants", false, false, true, false, false, [](Env& e, ID3D12GraphicsCommandList8* l) {
            if (l->GetType() == D3D12_COMMAND_LIST_TYPE_DIRECT) l->SetGraphicsRootSignature(e.rs.p);
            const UINT v[2] = {1, 2};
            l->SetGraphicsRoot32BitConstants(0, 2, v, 0);
        }},
        {"SetGraphicsRootCBV", false, false, true, false, false, [](Env& e, ID3D12GraphicsCommandList8* l) {
            if (l->GetType() == D3D12_COMMAND_LIST_TYPE_DIRECT) l->SetGraphicsRootSignature(e.rs.p);
            l->SetGraphicsRootConstantBufferView(1, e.buf->GetGPUVirtualAddress());
        }},
        {"SetGraphicsRootSRV", false, false, true, false, false, [](Env& e, ID3D12GraphicsCommandList8* l) {
            if (l->GetType() == D3D12_COMMAND_LIST_TYPE_DIRECT) l->SetGraphicsRootSignature(e.rs.p);
            l->SetGraphicsRootShaderResourceView(2, e.buf->GetGPUVirtualAddress());
        }},
        {"SetGraphicsRootUAV", false, false, true, false, false, [](Env& e, ID3D12GraphicsCommandList8* l) {
            if (l->GetType() == D3D12_COMMAND_LIST_TYPE_DIRECT) l->SetGraphicsRootSignature(e.rs.p);
            l->SetGraphicsRootUnorderedAccessView(3, e.buf->GetGPUVirtualAddress());
        }},
        {"IASetPrimitiveTopology", false, false, true, false, false, [](Env&, ID3D12GraphicsCommandList8* l) {
            l->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        }},
        {"IASetVertexBuffers", false, false, true, false, false, [](Env& e, ID3D12GraphicsCommandList8* l) {
            const D3D12_VERTEX_BUFFER_VIEW vbv{e.buf->GetGPUVirtualAddress(), 64, 16};
            l->IASetVertexBuffers(0, 1, &vbv);
        }},
        {"IASetIndexBuffer", false, false, true, false, false, [](Env& e, ID3D12GraphicsCommandList8* l) {
            const D3D12_INDEX_BUFFER_VIEW ibv{e.buf->GetGPUVirtualAddress(), 12, DXGI_FORMAT_R32_UINT};
            l->IASetIndexBuffer(&ibv);
        }},
        {"RSSetViewports", false, false, true, false, false, [](Env&, ID3D12GraphicsCommandList8* l) {
            const D3D12_VIEWPORT vp{0, 0, 16, 16, 0, 1};
            l->RSSetViewports(1, &vp);
        }},
        {"RSSetScissorRects", false, false, true, false, false, [](Env&, ID3D12GraphicsCommandList8* l) {
            const D3D12_RECT r{0, 0, 16, 16};
            l->RSSetScissorRects(1, &r);
        }},
        {"OMSetRenderTargets", false, false, true, false, false, [](Env& e, ID3D12GraphicsCommandList8* l) {
            l->OMSetRenderTargets(1, &e.rtv, FALSE, nullptr);
        }},
        {"OMSetBlendFactor", false, false, true, false, false, [](Env&, ID3D12GraphicsCommandList8* l) {
            const FLOAT factor[4] = {1, 1, 1, 1};
            l->OMSetBlendFactor(factor);
        }},
        {"OMSetStencilRef", false, false, true, false, false, [](Env&, ID3D12GraphicsCommandList8* l) {
            l->OMSetStencilRef(1);
        }},
        {"OMSetFrontAndBackStencilRef", false, false, true, false, false, [](Env&, ID3D12GraphicsCommandList8* l) {
            l->OMSetFrontAndBackStencilRef(1, 2);
        }},

        // Compute / pipeline / indirect / RT calls: allowed on DIRECT and COMPUTE, forbidden on COPY.
        {"Dispatch", false, true, true, true, false, [](Env& e, ID3D12GraphicsCommandList8* l) {
            if (l->GetType() != D3D12_COMMAND_LIST_TYPE_COPY) {
                l->SetComputeRootSignature(e.rs.p);
                l->SetPipelineState(e.cs_pso.p);
            }
            l->Dispatch(1, 1, 1);
        }},
        {"ClearState", false, false, true, true, false, [](Env&, ID3D12GraphicsCommandList8* l) {
            l->ClearState(nullptr);
        }},
        {"SetPipelineState", false, false, true, true, false, [](Env& e, ID3D12GraphicsCommandList8* l) {
            l->SetPipelineState(e.cs_pso.p);
        }},
        {"SetComputeRootSignature", false, false, true, true, false, [](Env& e, ID3D12GraphicsCommandList8* l) {
            l->SetComputeRootSignature(e.rs.p);
        }},
        {"SetComputeRootDescriptorTable", false, false, true, true, false, [](Env& e, ID3D12GraphicsCommandList8* l) {
            if (l->GetType() != D3D12_COMMAND_LIST_TYPE_COPY) {
                ID3D12DescriptorHeap* h = e.cbv_heap.p;
                l->SetDescriptorHeaps(1, &h);
                l->SetComputeRootSignature(e.rs.p);
            }
            l->SetComputeRootDescriptorTable(4, e.cbv_gpu);
        }},
        {"SetComputeRoot32BitConstant", false, false, true, true, false, [](Env& e, ID3D12GraphicsCommandList8* l) {
            if (l->GetType() != D3D12_COMMAND_LIST_TYPE_COPY) l->SetComputeRootSignature(e.rs.p);
            l->SetComputeRoot32BitConstant(0, 1, 0);
        }},
        {"SetComputeRoot32BitConstants", false, false, true, true, false, [](Env& e, ID3D12GraphicsCommandList8* l) {
            if (l->GetType() != D3D12_COMMAND_LIST_TYPE_COPY) l->SetComputeRootSignature(e.rs.p);
            const UINT v[2] = {1, 2};
            l->SetComputeRoot32BitConstants(0, 2, v, 0);
        }},
        {"SetComputeRootCBV", false, false, true, true, false, [](Env& e, ID3D12GraphicsCommandList8* l) {
            if (l->GetType() != D3D12_COMMAND_LIST_TYPE_COPY) l->SetComputeRootSignature(e.rs.p);
            l->SetComputeRootConstantBufferView(1, e.buf->GetGPUVirtualAddress());
        }},
        {"SetComputeRootSRV", false, false, true, true, false, [](Env& e, ID3D12GraphicsCommandList8* l) {
            if (l->GetType() != D3D12_COMMAND_LIST_TYPE_COPY) l->SetComputeRootSignature(e.rs.p);
            l->SetComputeRootShaderResourceView(2, e.buf->GetGPUVirtualAddress());
        }},
        {"SetComputeRootUAV", false, false, true, true, false, [](Env& e, ID3D12GraphicsCommandList8* l) {
            if (l->GetType() != D3D12_COMMAND_LIST_TYPE_COPY) l->SetComputeRootSignature(e.rs.p);
            l->SetComputeRootUnorderedAccessView(3, e.buf->GetGPUVirtualAddress());
        }},
        {"SetDescriptorHeaps", false, false, true, true, false, [](Env& e, ID3D12GraphicsCommandList8* l) {
            ID3D12DescriptorHeap* h = e.cbv_heap.p;
            l->SetDescriptorHeaps(1, &h);
        }},
        {"ExecuteIndirect(dispatch)", false, true, true, true, false, [](Env& e, ID3D12GraphicsCommandList8* l) {
            if (l->GetType() != D3D12_COMMAND_LIST_TYPE_COPY) {
                l->SetComputeRootSignature(e.rs.p);
                l->SetPipelineState(e.cs_pso.p);
            }
            l->ExecuteIndirect(e.csig_dispatch.p, 1, e.arg_buf.p, 0, nullptr, 0);
        }},
        {"ExecuteIndirect(draw)", false, true, true, false, false, [](Env& e, ID3D12GraphicsCommandList8* l) {
            if (l->GetType() == D3D12_COMMAND_LIST_TYPE_DIRECT) {
                l->SetGraphicsRootSignature(e.rs.p);
                l->SetPipelineState(e.gs_pso.p);
                l->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            }
            l->ExecuteIndirect(e.csig_draw.p, 1, e.arg_buf.p, 0, nullptr, 0);
        }},
        {"BuildRTAS", true, false, true, true, false, [](Env& e, ID3D12GraphicsCommandList8* l) {
            D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC d{};
            d.DestAccelerationStructureData = e.as_dst->GetGPUVirtualAddress();
            d.ScratchAccelerationStructureData = e.as_scratch->GetGPUVirtualAddress();
            d.Inputs.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL;
            d.Inputs.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
            d.Inputs.NumDescs = 0;
            l->BuildRaytracingAccelerationStructure(&d, 0, nullptr);
        }},
        {"EmitRTASPostbuildInfo", true, false, true, true, false, [](Env& e, ID3D12GraphicsCommandList8* l) {
            D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_DESC d{};
            d.DestBuffer = e.as_post->GetGPUVirtualAddress();
            d.InfoType = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_CURRENT_SIZE;
            const D3D12_GPU_VIRTUAL_ADDRESS src = e.as_dst->GetGPUVirtualAddress();
            l->EmitRaytracingAccelerationStructurePostbuildInfo(&d, 1, &src);
        }},
        {"CopyRTAS", true, false, true, true, false, [](Env& e, ID3D12GraphicsCommandList8* l) {
            l->CopyRaytracingAccelerationStructure(e.as_dst->GetGPUVirtualAddress(), e.as_src->GetGPUVirtualAddress(),
                                                   D3D12_RAYTRACING_ACCELERATION_STRUCTURE_COPY_MODE_CLONE);
        }},

        // Control: AtomicCopyBufferUINT is allowed on DIRECT, COMPUTE, and COPY (only forbidden on BUNDLE).
        {"AtomicCopyBufferUINT", false, false, true, true, true, [](Env& e, ID3D12GraphicsCommandList8* l) {
            ID3D12Resource* dep = e.buf2.p;
            const D3D12_SUBRESOURCE_RANGE_UINT64 range{0, {0, 4}};
            l->AtomicCopyBufferUINT(e.buf.p, 0, e.buf2.p, 0, 1, &dep, &range);
        }},
    };
}

HRESULT record_and_close(Env& env, D3D12_COMMAND_LIST_TYPE type, const Call& call) {
    if (call.needs_rt && !env.has_rt) return kSkipped;
    const bool expected = type == D3D12_COMMAND_LIST_TYPE_DIRECT  ? call.expected_direct
                        : type == D3D12_COMMAND_LIST_TYPE_COMPUTE ? call.expected_compute
                                                                  : call.expected_copy;
    if (call.needs_pso && !env.has_pso && expected) return kSkipped;
    Ref<ID3D12CommandAllocator> alloc;
    Ref<ID3D12GraphicsCommandList8> list;
    if (FAILED(env.device->CreateCommandAllocator(type, IID_PPV_ARGS(&alloc.p)))) return E_UNEXPECTED;
    if (FAILED(env.device->CreateCommandList(0, type, alloc.p, nullptr, IID_PPV_ARGS(&list.p)))) return E_UNEXPECTED;
    call.record(env, list.p);
    return list->Close();
}

bool run_positive_gpu_controls(ID3D12Device* device, const std::vector<std::uint8_t>& cs_code, bool has_pso) {
    // 1. COPY queue positive control: upload -> default -> readback via CopyBufferRegion.
    {
        D3D12_COMMAND_QUEUE_DESC qd{};
        qd.Type = D3D12_COMMAND_LIST_TYPE_COPY;
        Ref<ID3D12CommandQueue> queue;
        CHECK(SUCCEEDED(device->CreateCommandQueue(&qd, IID_PPV_ARGS(&queue.p))));
        Ref<ID3D12Fence> fence;
        CHECK(SUCCEEDED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence.p))));
        Ref<ID3D12CommandAllocator> alloc;
        Ref<ID3D12GraphicsCommandList> list;
        CHECK(SUCCEEDED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COPY, IID_PPV_ARGS(&alloc.p))));
        CHECK(SUCCEEDED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COPY, alloc.p, nullptr,
                                                  IID_PPV_ARGS(&list.p))));

        const auto rd = buffer_desc(256);
        const D3D12_HEAP_PROPERTIES upload_hp{D3D12_HEAP_TYPE_UPLOAD};
        const D3D12_HEAP_PROPERTIES default_hp{D3D12_HEAP_TYPE_DEFAULT};
        const D3D12_HEAP_PROPERTIES readback_hp{D3D12_HEAP_TYPE_READBACK};
        Ref<ID3D12Resource> upload, mid, readback;
        CHECK(SUCCEEDED(device->CreateCommittedResource(&upload_hp, D3D12_HEAP_FLAG_NONE, &rd,
                                                        D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
                                                        IID_PPV_ARGS(&upload.p))));
        CHECK(SUCCEEDED(device->CreateCommittedResource(&default_hp, D3D12_HEAP_FLAG_NONE, &rd,
                                                        D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&mid.p))));
        CHECK(SUCCEEDED(device->CreateCommittedResource(&readback_hp, D3D12_HEAP_FLAG_NONE, &rd,
                                                        D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                        IID_PPV_ARGS(&readback.p))));

        std::uint32_t* mapped = nullptr;
        CHECK(SUCCEEDED(upload->Map(0, nullptr, reinterpret_cast<void**>(&mapped))));
        for (std::uint32_t i = 0; i < 64; ++i) mapped[i] = 0xC0010000u + i;
        upload->Unmap(0, nullptr);

        list->CopyBufferRegion(mid.p, 0, upload.p, 0, 256);
        const auto to_src = transition(mid.p, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COPY_SOURCE);
        list->ResourceBarrier(1, &to_src);
        list->CopyBufferRegion(readback.p, 0, mid.p, 0, 256);
        CHECK(SUCCEEDED(list->Close()));

        ID3D12CommandList* lists[] = {list.p};
        queue->ExecuteCommandLists(1, lists);
        CHECK(SUCCEEDED(queue->Signal(fence.p, 1)));
        CHECK(SUCCEEDED(fence->SetEventOnCompletion(1, nullptr)));

        const std::uint32_t* out = nullptr;
        CHECK(SUCCEEDED(readback->Map(0, nullptr, reinterpret_cast<void**>(const_cast<std::uint32_t**>(&out)))));
        for (std::uint32_t i = 0; i < 64; ++i) CHECK(out[i] == 0xC0010000u + i);
        readback->Unmap(0, nullptr);
    }

    // 2. COMPUTE queue positive control: root signature + PSO + CBV table + Dispatch + CopyBufferRegion readback.
    if (has_pso) {
        D3D12_DESCRIPTOR_RANGE1 range{};
        range.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_CBV;
        range.NumDescriptors = 1;
        range.BaseShaderRegister = 1;
        D3D12_ROOT_PARAMETER1 params[3]{};
        params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
        params[0].Constants = {0, 0, 2};
        params[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
        params[1].Descriptor = {0, 0, D3D12_ROOT_DESCRIPTOR_FLAG_NONE};
        params[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        params[2].DescriptorTable = {1, &range};
        D3D12_VERSIONED_ROOT_SIGNATURE_DESC rsd{};
        rsd.Version = D3D_ROOT_SIGNATURE_VERSION_1_1;
        rsd.Desc_1_1 = {3, params, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_NONE};
        const auto blob = serialize_root_signature(rsd);
        CHECK(!blob.empty());
        Ref<ID3D12RootSignature> rs;
        CHECK(SUCCEEDED(device->CreateRootSignature(0, blob.data(), blob.size(), IID_PPV_ARGS(&rs.p))));

        D3D12_COMPUTE_PIPELINE_STATE_DESC cpsd{};
        cpsd.pRootSignature = rs.p;
        cpsd.CS = {cs_code.data(), cs_code.size()};
        Ref<ID3D12PipelineState> pso;
        CHECK(SUCCEEDED(device->CreateComputePipelineState(&cpsd, IID_PPV_ARGS(&pso.p))));

        const D3D12_HEAP_PROPERTIES upload_hp{D3D12_HEAP_TYPE_UPLOAD};
        const D3D12_HEAP_PROPERTIES default_hp{D3D12_HEAP_TYPE_DEFAULT};
        const D3D12_HEAP_PROPERTIES readback_hp{D3D12_HEAP_TYPE_READBACK};
        const auto cb_rd = buffer_desc(256);
        const auto uav_rd = buffer_desc(256, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        Ref<ID3D12Resource> cb_upload, cb, out_buf, readback;
        CHECK(SUCCEEDED(device->CreateCommittedResource(&upload_hp, D3D12_HEAP_FLAG_NONE, &cb_rd,
                                                        D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
                                                        IID_PPV_ARGS(&cb_upload.p))));
        CHECK(SUCCEEDED(device->CreateCommittedResource(&default_hp, D3D12_HEAP_FLAG_NONE, &cb_rd,
                                                        D3D12_RESOURCE_STATE_COMMON, nullptr,
                                                        IID_PPV_ARGS(&cb.p))));
        CHECK(SUCCEEDED(device->CreateCommittedResource(&default_hp, D3D12_HEAP_FLAG_NONE, &uav_rd,
                                                        D3D12_RESOURCE_STATE_COMMON, nullptr,
                                                        IID_PPV_ARGS(&out_buf.p))));
        CHECK(SUCCEEDED(device->CreateCommittedResource(&readback_hp, D3D12_HEAP_FLAG_NONE, &cb_rd,
                                                        D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                        IID_PPV_ARGS(&readback.p))));

        std::uint32_t* mapped = nullptr;
        CHECK(SUCCEEDED(cb_upload->Map(0, nullptr, reinterpret_cast<void**>(&mapped))));
        for (std::uint32_t i = 0; i < 64; ++i) mapped[i] = 0x5A000000u + i;
        cb_upload->Unmap(0, nullptr);

        D3D12_DESCRIPTOR_HEAP_DESC hd{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 1,
                                      D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, 0};
        Ref<ID3D12DescriptorHeap> heap;
        CHECK(SUCCEEDED(device->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&heap.p))));
        D3D12_CONSTANT_BUFFER_VIEW_DESC cbvd{cb->GetGPUVirtualAddress(), 256};
        device->CreateConstantBufferView(&cbvd, heap->GetCPUDescriptorHandleForHeapStart());

        D3D12_COMMAND_QUEUE_DESC qd{};
        qd.Type = D3D12_COMMAND_LIST_TYPE_COMPUTE;
        Ref<ID3D12CommandQueue> queue;
        CHECK(SUCCEEDED(device->CreateCommandQueue(&qd, IID_PPV_ARGS(&queue.p))));
        Ref<ID3D12Fence> fence;
        CHECK(SUCCEEDED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence.p))));
        Ref<ID3D12CommandAllocator> alloc;
        Ref<ID3D12GraphicsCommandList> list;
        CHECK(SUCCEEDED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE, IID_PPV_ARGS(&alloc.p))));
        CHECK(SUCCEEDED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COMPUTE, alloc.p, nullptr,
                                                  IID_PPV_ARGS(&list.p))));

        list->CopyBufferRegion(cb.p, 0, cb_upload.p, 0, 256);
        const D3D12_RESOURCE_BARRIER pre[2] = {
            transition(cb.p, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER),
            transition(out_buf.p, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS),
        };
        list->ResourceBarrier(2, pre);

        ID3D12DescriptorHeap* heaps[] = {heap.p};
        list->SetDescriptorHeaps(1, heaps);
        list->SetComputeRootSignature(rs.p);
        list->SetPipelineState(pso.p);
        const UINT constants[2] = {0u, 0u};  // CBV V# mode in cs_user_data.comp
        list->SetComputeRoot32BitConstants(0, 2, constants, 0);
        list->SetComputeRootUnorderedAccessView(1, out_buf->GetGPUVirtualAddress());
        list->SetComputeRootDescriptorTable(2, heap->GetGPUDescriptorHandleForHeapStart());
        list->Dispatch(1, 1, 1);
        list->ClearState(nullptr);
        const auto to_src = transition(out_buf.p, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                                       D3D12_RESOURCE_STATE_COPY_SOURCE);
        list->ResourceBarrier(1, &to_src);
        list->CopyBufferRegion(readback.p, 0, out_buf.p, 0, 256);
        CHECK(SUCCEEDED(list->Close()));

        ID3D12CommandList* lists[] = {list.p};
        queue->ExecuteCommandLists(1, lists);
        CHECK(SUCCEEDED(queue->Signal(fence.p, 1)));
        CHECK(SUCCEEDED(fence->SetEventOnCompletion(1, nullptr)));

        const std::uint32_t* out = nullptr;
        CHECK(SUCCEEDED(readback->Map(0, nullptr, reinterpret_cast<void**>(const_cast<std::uint32_t**>(&out)))));
        for (std::uint32_t i = 0; i < 64; ++i) CHECK(out[i] == 0x5A000000u + i);
        readback->Unmap(0, nullptr);
    }
    return true;
}

struct Run {
    const char* name = "";
    bool ran = false;
    std::vector<Verdict> verdicts;
};

}  // namespace

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    vktest::Device vk;
    const auto device_status = vk.initialize();
    if (device_status == vktest::DeviceStatus::unavailable) {
        std::fprintf(stderr, "SKIP: no usable Vulkan 1.3 device\n");
        return 77;
    }
    REQUIRE(device_status == vktest::DeviceStatus::ready);

    const auto cs_prog = fssm_program({0x43533001, 0xbf810000});
    const auto vs_prog = fssm_program({0x56533001, 0xbf810000});
    const auto ps_prog = fssm_program({0x50533001, 0xbf810000});
    const auto cs_dxbc = shader_container(cs_prog);
    const auto vs_dxbc = shader_container(vs_prog);
    const auto ps_dxbc = shader_container(ps_prog);

    const testfiles::ScratchDir scratch("leonida_qtype_restrictions_test");
    const auto& lib_dir = scratch.path();
    REQUIRE(testfiles::copy_file(vktest::shader_fixture_path(LEONIDA_TEST_CS_SPV), lib_dir / "cs.spv"));
    REQUIRE(testfiles::copy_file(vktest::shader_fixture_path(LEONIDA_TEST_VS_SPV), lib_dir / "vs.spv"));
    REQUIRE(testfiles::copy_file(vktest::shader_fixture_path(LEONIDA_TEST_PS_SPV), lib_dir / "ps.spv"));
    {
        std::ofstream idx(lib_dir / "index.json", std::ios::binary);
        idx << "{\"cs\": {\"stage\": \"cs\", \"fssm_sha256\": \"" << leonida::sha256_hex(cs_prog)
            << "\", \"bindings\": [], \"push_constant_bytes\": 144},\n"
               " \"vs\": {\"stage\": \"vs\", \"fssm_sha256\": \"" << leonida::sha256_hex(vs_prog)
            << "\", \"bindings\": [], \"push_constant_bytes\": 144, \"input_signature\": ["
               "{\"semantic\": \"COLOR\", \"semantic_index\": 0, \"register\": 0, \"mask\": 15, \"rw_mask\": 15, "
               "\"system_value\": 0, \"component_type\": \"float\"}, "
               "{\"semantic\": \"POSITION\", \"semantic_index\": 0, \"register\": 1, \"mask\": 3, \"rw_mask\": 3, "
               "\"system_value\": 0, \"component_type\": \"float\"}]},\n"
               " \"ps\": {\"stage\": \"ps\", \"fssm_sha256\": \"" << leonida::sha256_hex(ps_prog)
            << "\", \"bindings\": [], \"push_constant_bytes\": 144, \"outputs\": [0]}}";
    }

    const auto list = calls();
    const unsigned errors_before = vk.validation_errors();

    Run layer;
    layer.name = "layer";
    {
        rt::d3d12::VulkanDeviceInfo info{};
        info.get_instance_proc = vk.instance_proc;
        info.instance = vk.instance;
        info.physical = vk.physical;
        info.device = vk.device;
        info.queue = vk.queue;
        info.queue_family = vk.family;
        info.buffer_device_address = true;
        info.shader_dir = lib_dir.string();
        Ref<ID3D12Device> ours;
        REQUIRE(SUCCEEDED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&ours.p))));

        Env env;
        REQUIRE(env.init(ours.p, true, cs_dxbc, vs_dxbc, ps_dxbc));
        for (const Call& call : list) {
            Verdict v{};
            v.direct_close = record_and_close(env, D3D12_COMMAND_LIST_TYPE_DIRECT, call);
            v.compute_close = record_and_close(env, D3D12_COMMAND_LIST_TYPE_COMPUTE, call);
            v.copy_close = record_and_close(env, D3D12_COMMAND_LIST_TYPE_COPY, call);
            layer.verdicts.push_back(v);
        }
        layer.ran = true;

        // Positive end-to-end execution controls on real COPY and COMPUTE queues.
        REQUIRE(run_positive_gpu_controls(ours.p, cs_dxbc, env.has_pso));
    }

    Run hosts[2];
    hosts[0].name = "default adapter";
    hosts[1].name = "WARP";
    if (!hostwine::host_is_wine()) {
        std::vector<std::uint8_t> hlsl_cs, hlsl_vs, hlsl_ps;
        const bool compiled =
            compile_hlsl("[numthreads(1,1,1)] void main() {}", "main", "cs_5_0", hlsl_cs) &&
            compile_hlsl("float4 main() : SV_Position { return float4(0,0,0,1); }", "main", "vs_5_0", hlsl_vs) &&
            compile_hlsl("float4 main() : SV_Target { return float4(1,0,0,1); }", "main", "ps_5_0", hlsl_ps);
        if (compiled) {
            Ref<IDXGIFactory4> factory;
            REQUIRE(SUCCEEDED(CreateDXGIFactory1(IID_PPV_ARGS(&factory.p))));
            for (int i = 0; i < 2; ++i) {
                Ref<IDXGIAdapter> adapter;
                Ref<ID3D12Device> host;
                if (i == 1 && FAILED(factory->EnumWarpAdapter(IID_PPV_ARGS(&adapter.p)))) continue;
                if (FAILED(D3D12CreateDevice(adapter.p, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&host.p)))) continue;
                Env env;
                REQUIRE(env.init(host.p, false, hlsl_cs, hlsl_vs, hlsl_ps));
                for (const Call& call : list) {
                    Verdict v{};
                    v.direct_close = record_and_close(env, D3D12_COMMAND_LIST_TYPE_DIRECT, call);
                    v.compute_close = record_and_close(env, D3D12_COMMAND_LIST_TYPE_COMPUTE, call);
                    v.copy_close = record_and_close(env, D3D12_COMMAND_LIST_TYPE_COPY, call);
                    hosts[i].verdicts.push_back(v);
                }
                hosts[i].ran = true;
            }
        }
    }

    const auto mark = [](HRESULT hr) -> const char* {
        if (hr == kSkipped) return "-";
        return SUCCEEDED(hr) ? "OK" : "FAIL";
    };

    unsigned failures = 0, checked = 0, compared = 0, disagreed = 0;
    std::printf("%-30s | %-14s | %-14s | %-14s\n", "call (D/C/P)", "layer", hosts[0].name, hosts[1].name);
    for (std::size_t i = 0; i < list.size(); ++i) {
        const Call& c = list[i];
        const Verdict& lv = layer.verdicts[i];
        std::printf("%-30s | %-4s %-4s %-4s", c.name, mark(lv.direct_close), mark(lv.compute_close),
                    mark(lv.copy_close));
        for (int h = 0; h < 2; ++h) {
            if (!hosts[h].ran) {
                std::printf(" | %-14s", "(not run)");
            } else {
                const Verdict& hv = hosts[h].verdicts[i];
                std::printf(" | %-4s %-4s %-4s", mark(hv.direct_close), mark(hv.compute_close), mark(hv.copy_close));
            }
        }
        std::printf("\n");

        const std::array<std::tuple<const char*, HRESULT Verdict::*, bool>, 3> slots = {{
            {"DIRECT", &Verdict::direct_close, c.expected_direct},
            {"COMPUTE", &Verdict::compute_close, c.expected_compute},
            {"COPY", &Verdict::copy_close, c.expected_copy},
        }};
        for (const auto& [qname, field, expected] : slots) {
            if (lv.*field == kSkipped) continue;
            ++checked;
            if (SUCCEEDED(lv.*field) != expected) {
                std::fprintf(stderr, "FAIL %s (%s): layer %s != expected %s\n", c.name, qname,
                             mark(lv.*field), expected ? "OK" : "FAIL");
                ++failures;
            }
            const bool h0 = hosts[0].ran && hosts[0].verdicts[i].*field != kSkipped;
            const bool h1 = hosts[1].ran && hosts[1].verdicts[i].*field != kSkipped;
            if (!h0 && !h1) continue;
            const bool v0 = h0 ? SUCCEEDED(hosts[0].verdicts[i].*field) : SUCCEEDED(hosts[1].verdicts[i].*field);
            if (h0 && h1 && v0 != SUCCEEDED(hosts[1].verdicts[i].*field)) {
                std::printf("  host adapters disagree on %s (%s): not compared\n", c.name, qname);
                ++disagreed;
                continue;
            }
            ++compared;
            if (SUCCEEDED(lv.*field) != v0) {
                std::fprintf(stderr, "FAIL %s (%s): layer %s, host %s\n", c.name, qname,
                             mark(lv.*field), v0 ? "OK" : "FAIL");
                ++failures;
            }
        }
    }

    const unsigned validation_errors = vk.validation_errors() - errors_before;
    if (validation_errors) {
        std::fprintf(stderr, "%u Vulkan validation errors\n", validation_errors);
        ++failures;
    }
    std::printf("queue-type restrictions: %zu calls (%u checks) on layer, %u host-compared, %u host-disagreed, "
                "%u failures, %u validation errors\n",
                list.size(), checked, compared, disagreed, failures, validation_errors);
    return failures ? 1 : 0;
}
