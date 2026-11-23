// Blend state against the render-target formats of a pipeline, compared with the host's D3D12 on the same adapter
// (matched by LUID, the D3D-188 rule) and with WARP.
//
// The PC runtime refuses a pipeline that enables blending for an integer render target (UINT or SINT, which cannot
// blend) that its pixel shader writes: CreateGraphicsPipelineState returns E_INVALIDARG, on NVIDIA and on WARP alike
// (measured: every integer render-target format as target 0, and as target 1 under shared and independent blend
// states). An integer target the pixel shader does not write, a pipeline with no pixel shader, and a target the
// pipeline does not bind are accepted. Vulkan has the same limit (blendEnable needs the format's BLEND feature), so the
// layer has to refuse the first case and must not pass blendEnable on for the others.
//   1. Pipeline creation: a table of blend states over integer and float targets, each created on the layer, the host and
//      WARP; the layer must answer as they do.
//   2. Draws: a full-screen pass writing a float colour to target 0 and integers to target 1, with ONE + ONE blending
//      enabled for target 1 (shared or independent blend state), compared texel for texel with the same draw on the host.
//      The layer's shaders are GLSL stand-ins for translated ones (test/glsl/vs_fullscreen.vert, ps_typeless_float.frag,
//      ps_typeless_uint.frag, ps_blend_mrt.frag), found by the FSSM hash of their containers in a translator-style
//      index; the host's are HLSL compiled by d3dcompiler_47.
// Synthetic data only. LEONIDA_VK_VALIDATION=1 runs it under VK_LAYER_KHRONOS_validation (a blendEnable on an integer
// attachment is a validation error).
#include "vk_test_device.h"
#include "host_wine.h"
#include "d3d12_test_shaders.h"
#include "../shims/d3d12_com.h"
#include "../src/shader_library.h"

#include <d3dcompiler.h>
#include <dxgi1_4.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace {

template<class T>
struct Ref {
    T* p = nullptr;
    ~Ref() { if (p) p->Release(); }
    T* operator->() const { return p; }
};

// One pass for one device: root signature blob and VS / PS bytecode (shader containers for the layer, DXBC for the host).
struct Programs {
    std::vector<std::uint8_t> root, vs, ps_float, ps_uint, ps_mrt;
};

enum class Ps { none, float_out, uint_out, mrt_out };

struct Case {
    const char* name;
    UINT targets;
    DXGI_FORMAT formats[2];
    bool independent;
    bool blend[2];  // BlendEnable of RenderTarget[0] and [1]
    Ps ps;
};

D3D12_GRAPHICS_PIPELINE_STATE_DESC make_desc(const Case& c, ID3D12RootSignature* root, const Programs& p) {
    D3D12_GRAPHICS_PIPELINE_STATE_DESC pd{};
    pd.pRootSignature = root;
    const std::vector<std::uint8_t>& ps = c.ps == Ps::float_out ? p.ps_float : c.ps == Ps::uint_out ? p.ps_uint : p.ps_mrt;
    pd.VS = {p.vs.data(), p.vs.size()};
    if (c.ps != Ps::none) pd.PS = {ps.data(), ps.size()};
    pd.BlendState.IndependentBlendEnable = c.independent;
    for (int i = 0; i < 2; ++i) {
        auto& rt = pd.BlendState.RenderTarget[i];
        rt.BlendEnable = c.blend[i];
        rt.SrcBlend = rt.DestBlend = rt.SrcBlendAlpha = rt.DestBlendAlpha = D3D12_BLEND_ONE;
        rt.BlendOp = rt.BlendOpAlpha = D3D12_BLEND_OP_ADD;
        rt.LogicOp = D3D12_LOGIC_OP_NOOP;
        rt.RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
    }
    pd.SampleMask = UINT_MAX;
    pd.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
    pd.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
    pd.RasterizerState.DepthClipEnable = TRUE;
    pd.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    pd.NumRenderTargets = c.targets;
    for (UINT i = 0; i < c.targets && i < 2; ++i) pd.RTVFormats[i] = c.formats[i];
    pd.SampleDesc = {1, 0};
    return pd;
}

HRESULT create_pso(ID3D12Device* device, const Case& c, const Programs& p) {
    Ref<ID3D12RootSignature> root;
    HRESULT hr = device->CreateRootSignature(0, p.root.data(), p.root.size(), IID_PPV_ARGS(&root.p));
    if (FAILED(hr)) return hr;
    const auto pd = make_desc(c, root.p, p);
    Ref<ID3D12PipelineState> pso;
    return device->CreateGraphicsPipelineState(&pd, IID_PPV_ARGS(&pso.p));
}

// A full-screen draw into two 4x4 targets cleared to a known value; the bytes of both, row by row.
bool draw_and_read(ID3D12Device* device, const Case& c, const Programs& p, std::vector<unsigned char> (&bytes)[2]) {
    const D3D12_HEAP_PROPERTIES def_heap{D3D12_HEAP_TYPE_DEFAULT}, rb_heap{D3D12_HEAP_TYPE_READBACK};
    Ref<ID3D12Resource> texture[2], readback[2];
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint[2]{};
    UINT rows[2]{};
    UINT64 row_bytes[2]{};
    Ref<ID3D12DescriptorHeap> heap;
    const D3D12_DESCRIPTOR_HEAP_DESC hd{D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 2};
    if (FAILED(device->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&heap.p)))) return false;
    const UINT step = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
    D3D12_CPU_DESCRIPTOR_HANDLE handles[2];
    for (UINT i = 0; i < 2; ++i) {
        D3D12_RESOURCE_DESC desc{};
        desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        desc.Width = desc.Height = 4;
        desc.DepthOrArraySize = 1;
        desc.MipLevels = 1;
        desc.Format = c.formats[i];
        desc.SampleDesc = {1, 0};
        desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
        if (FAILED(device->CreateCommittedResource(&def_heap, D3D12_HEAP_FLAG_NONE, &desc,
                                                   D3D12_RESOURCE_STATE_RENDER_TARGET, nullptr,
                                                   IID_PPV_ARGS(&texture[i].p))))
            return false;
        UINT64 total = 0;
        device->GetCopyableFootprints(&desc, 0, 1, 0, &footprint[i], &rows[i], &row_bytes[i], &total);
        D3D12_RESOURCE_DESC buffer{};
        buffer.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        buffer.Width = total;
        buffer.Height = buffer.DepthOrArraySize = buffer.MipLevels = 1;
        buffer.SampleDesc = {1, 0};
        buffer.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        if (FAILED(device->CreateCommittedResource(&rb_heap, D3D12_HEAP_FLAG_NONE, &buffer,
                                                   D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                   IID_PPV_ARGS(&readback[i].p))))
            return false;
        handles[i] = heap->GetCPUDescriptorHandleForHeapStart();
        handles[i].ptr += SIZE_T(i) * step;
        D3D12_RENDER_TARGET_VIEW_DESC rtv{};
        rtv.Format = c.formats[i];
        rtv.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2D;
        device->CreateRenderTargetView(texture[i].p, &rtv, handles[i]);
    }
    Ref<ID3D12RootSignature> root;
    Ref<ID3D12PipelineState> pso;
    if (FAILED(device->CreateRootSignature(0, p.root.data(), p.root.size(), IID_PPV_ARGS(&root.p)))) return false;
    const auto pd = make_desc(c, root.p, p);
    if (FAILED(device->CreateGraphicsPipelineState(&pd, IID_PPV_ARGS(&pso.p)))) return false;
    Ref<ID3D12CommandQueue> queue;
    Ref<ID3D12CommandAllocator> alloc;
    Ref<ID3D12GraphicsCommandList> list;
    Ref<ID3D12Fence> fence;
    const D3D12_COMMAND_QUEUE_DESC qd{D3D12_COMMAND_LIST_TYPE_DIRECT};
    if (FAILED(device->CreateCommandQueue(&qd, IID_PPV_ARGS(&queue.p))) ||
        FAILED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&alloc.p))) ||
        FAILED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, alloc.p, nullptr, IID_PPV_ARGS(&list.p))) ||
        FAILED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence.p))))
        return false;
    // Target 0 starts at 0.25 in every channel, target 1 at 100: a blended result differs from the shader's constant.
    const float float_clear[4] = {0.25f, 0.25f, 0.25f, 0.25f}, uint_clear[4] = {100.f, 100.f, 100.f, 100.f};
    list->ClearRenderTargetView(handles[0], float_clear, 0, nullptr);
    list->ClearRenderTargetView(handles[1], uint_clear, 0, nullptr);
    const D3D12_VIEWPORT viewport{0, 0, 4, 4, 0, 1};
    const D3D12_RECT scissor{0, 0, 4, 4};
    list->SetGraphicsRootSignature(root.p);
    list->SetPipelineState(pso.p);
    list->OMSetRenderTargets(2, handles, FALSE, nullptr);
    list->RSSetViewports(1, &viewport);
    list->RSSetScissorRects(1, &scissor);
    list->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    list->DrawInstanced(3, 1, 0, 0);
    for (UINT i = 0; i < 2; ++i) {
        D3D12_RESOURCE_BARRIER barrier{};
        barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        barrier.Transition = {texture[i].p, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, D3D12_RESOURCE_STATE_RENDER_TARGET,
                              D3D12_RESOURCE_STATE_COPY_SOURCE};
        list->ResourceBarrier(1, &barrier);
        D3D12_TEXTURE_COPY_LOCATION dst{readback[i].p, D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT};
        dst.PlacedFootprint = footprint[i];
        D3D12_TEXTURE_COPY_LOCATION src{texture[i].p, D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX};
        list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
    }
    if (FAILED(list->Close())) return false;
    ID3D12CommandList* lists[] = {list.p};
    queue->ExecuteCommandLists(1, lists);
    if (FAILED(queue->Signal(fence.p, 1)) || FAILED(fence->SetEventOnCompletion(1, nullptr))) return false;
    for (UINT i = 0; i < 2; ++i) {
        void* mapped = nullptr;
        if (FAILED(readback[i]->Map(0, nullptr, &mapped))) return false;
        bytes[i].clear();
        for (UINT y = 0; y < rows[i]; ++y) {
            const auto* row = static_cast<const unsigned char*>(mapped) + footprint[i].Offset +
                              y * footprint[i].Footprint.RowPitch;
            bytes[i].insert(bytes[i].end(), row, row + row_bytes[i]);
        }
        const D3D12_RANGE none{};
        readback[i]->Unmap(0, &none);
    }
    return true;
}

// The host's D3D12 device on the adapter whose LUID the Vulkan device reports (not the default adapter: D3D-188).
bool host_device(const vktest::Device& vk, Ref<ID3D12Device>& out, std::string& name) {
    VkPhysicalDeviceIDProperties id{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES};
    VkPhysicalDeviceProperties2 props{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &id};
    vk.instance_fn<PFN_vkGetPhysicalDeviceProperties2>("vkGetPhysicalDeviceProperties2")(vk.physical, &props);
    Ref<IDXGIFactory4> factory;
    if (!id.deviceLUIDValid || FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory.p)))) return false;
    LUID luid{};
    std::memcpy(&luid, id.deviceLUID, sizeof(luid));
    Ref<IDXGIAdapter1> adapter;
    if (FAILED(factory->EnumAdapterByLuid(luid, IID_PPV_ARGS(&adapter.p)))) return false;
    DXGI_ADAPTER_DESC1 desc{};
    adapter->GetDesc1(&desc);
    char utf8[256]{};
    WideCharToMultiByte(CP_UTF8, 0, desc.Description, -1, utf8, sizeof(utf8), nullptr, nullptr);
    name = utf8;
    return SUCCEEDED(D3D12CreateDevice(adapter.p, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&out.p)));
}

std::string hex(const std::vector<unsigned char>& bytes, std::size_t count) {
    std::string s;
    char b[4];
    for (std::size_t i = 0; i < count && i < bytes.size(); ++i) {
        std::snprintf(b, sizeof(b), "%02x", bytes[i]);
        s += b;
    }
    return s;
}

// HLSL for the host's passes, the same constants as the GLSL stand-ins.
constexpr char kHostVs[] =
    "float4 main(uint id : SV_VertexID) : SV_Position {\n"
    "    float2 corner = float2(id & 2, (id << 1) & 2);\n"
    "    return float4(corner * 2.0 - 1.0, 0.0, 1.0);\n"
    "}\n";
constexpr char kHostPsFloat[] = "float4 main() : SV_Target { return float4(0.5, -2.0, 1.5, 0.25); }\n";
constexpr char kHostPsUint[] = "uint4 main() : SV_Target { return uint4(7, 300, 1000, 3); }\n";
constexpr char kHostPsMrt[] =
    "struct Out { float4 a : SV_Target0; uint4 b : SV_Target1; };\n"
    "Out main() { Out o; o.a = float4(0.25, 0.5, 0.0, 0.0); o.b = uint4(7, 300, 1000, 3); return o; }\n";

bool compile_hlsl(pD3DCompile compile, const char* source, std::vector<std::uint8_t>& out, bool vertex) {
    ID3DBlob* code = nullptr;
    ID3DBlob* errors = nullptr;
    const HRESULT hr = compile(source, std::strlen(source), nullptr, nullptr, nullptr, "main", vertex ? "vs_5_0" : "ps_5_0",
                               0, 0, &code, &errors);
    if (errors) {
        std::fprintf(stderr, "%s\n", static_cast<const char*>(errors->GetBufferPointer()));
        errors->Release();
    }
    if (FAILED(hr) || !code) return false;
    const auto* bytes = static_cast<const std::uint8_t*>(code->GetBufferPointer());
    out.assign(bytes, bytes + code->GetBufferSize());
    code->Release();
    return true;
}

}  // namespace

int main() {
    vktest::Device vk;
    vk.request_buffer_device_address = true;
    const auto status = vk.initialize();
    if (status == vktest::DeviceStatus::unavailable) return 77;
    if (status != vktest::DeviceStatus::ready) return 1;
    if (hostwine::host_is_wine()) return 77;  // the host D3D12 runtime is the reference

    // The layer's passes: shader containers around synthetic FSSM programs, and the index that maps their hashes to
    // the compiled GLSL stand-ins.
    const auto vs_program = d3d12test::fssm_program({0x56530401, 0xbf810000});
    const auto ps_float_program = d3d12test::fssm_program({0x50530401, 0xbf810000});
    const auto ps_uint_program = d3d12test::fssm_program({0x50530402, 0xbf810000});
    const auto ps_mrt_program = d3d12test::fssm_program({0x50530403, 0xbf810000});
    const auto library = std::filesystem::temp_directory_path() / "leonida_blend_format_test";
    std::filesystem::create_directories(library);
    const auto overwrite = std::filesystem::copy_options::overwrite_existing;
    std::filesystem::copy_file(LEONIDA_BLEND_VS_SPV, library / "vs.spv", overwrite);
    std::filesystem::copy_file(LEONIDA_BLEND_PS_FLOAT_SPV, library / "ps_float.spv", overwrite);
    std::filesystem::copy_file(LEONIDA_BLEND_PS_UINT_SPV, library / "ps_uint.spv", overwrite);
    std::filesystem::copy_file(LEONIDA_BLEND_PS_MRT_SPV, library / "ps_mrt.spv", overwrite);
    {
        std::ofstream index(library / "index.json", std::ios::binary);
        index << "{\"vs\": {\"stage\": \"vs\", \"fssm_sha256\": \"" << leonida::sha256_hex(vs_program)
              << "\", \"bindings\": [], \"push_constant_bytes\": 144, \"input_signature\": []},\n"
                 " \"ps_float\": {\"stage\": \"ps\", \"fssm_sha256\": \"" << leonida::sha256_hex(ps_float_program)
              << "\", \"bindings\": [], \"push_constant_bytes\": 144, \"outputs\": [0]},\n"
                 " \"ps_uint\": {\"stage\": \"ps\", \"fssm_sha256\": \"" << leonida::sha256_hex(ps_uint_program)
              << "\", \"bindings\": [], \"push_constant_bytes\": 144, \"outputs\": [0]},\n"
                 " \"ps_mrt\": {\"stage\": \"ps\", \"fssm_sha256\": \"" << leonida::sha256_hex(ps_mrt_program)
              << "\", \"bindings\": [], \"push_constant_bytes\": 144, \"outputs\": [0, 1]}}";
    }
    rt::d3d12::VulkanDeviceInfo info{};
    info.get_instance_proc = vk.instance_proc;
    info.instance = vk.instance;
    info.physical = vk.physical;
    info.device = vk.device;
    info.queue = vk.queue;
    info.queue_family = vk.family;
    info.bindless = vk.bindless;
    info.bindless_partially_bound = vk.partially_bound;
    info.host_image_copy = vk.host_image_copy;
    info.buffer_device_address = vk.buffer_device_address;
    info.features.independentBlend = vk.enabled.independentBlend;
    info.shader_dir = library.string();
    Ref<ID3D12Device> ours, host, warp;
    std::string adapter;
    if (FAILED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&ours.p)))) return 1;
    if (!host_device(vk, host, adapter)) {
        std::fprintf(stderr, "no host D3D12 device on the Vulkan device's adapter\n");
        return 77;
    }
    {
        Ref<IDXGIFactory4> factory;
        Ref<IDXGIAdapter> warp_adapter;
        if (SUCCEEDED(CreateDXGIFactory1(IID_PPV_ARGS(&factory.p))) &&
            SUCCEEDED(factory->EnumWarpAdapter(IID_PPV_ARGS(&warp_adapter.p))))
            D3D12CreateDevice(warp_adapter.p, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&warp.p));
    }
    std::printf("host reference: %s%s\n", adapter.c_str(), warp.p ? ", and WARP" : "");
    if (!vk.enabled.independentBlend) {
        std::fprintf(stderr, "the Vulkan device lacks independentBlend\n");
        return 77;
    }

    const HMODULE compiler = LoadLibraryExA("d3dcompiler_47.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
    const auto compile = compiler ? reinterpret_cast<pD3DCompile>(reinterpret_cast<void*>(
                                        GetProcAddress(compiler, "D3DCompile")))
                                  : nullptr;
    D3D12_VERSIONED_ROOT_SIGNATURE_DESC root_desc{};
    root_desc.Version = D3D_ROOT_SIGNATURE_VERSION_1_0;
    Programs host_programs, our_programs;
    host_programs.root = our_programs.root = d3d12test::serialize_root_signature(root_desc);
    if (!compile || host_programs.root.empty() || !compile_hlsl(compile, kHostVs, host_programs.vs, true) ||
        !compile_hlsl(compile, kHostPsFloat, host_programs.ps_float, false) ||
        !compile_hlsl(compile, kHostPsUint, host_programs.ps_uint, false) ||
        !compile_hlsl(compile, kHostPsMrt, host_programs.ps_mrt, false)) {
        std::fprintf(stderr, "d3dcompiler_47 could not compile the host's passes\n");
        return 1;
    }
    our_programs.vs = d3d12test::shader_container(vs_program);
    our_programs.ps_float = d3d12test::shader_container(ps_float_program);
    our_programs.ps_uint = d3d12test::shader_container(ps_uint_program);
    our_programs.ps_mrt = d3d12test::shader_container(ps_mrt_program);

    constexpr DXGI_FORMAT F = DXGI_FORMAT_R8G8B8A8_UNORM, U = DXGI_FORMAT_R32_UINT, UK = DXGI_FORMAT_UNKNOWN;
    const Case cases[] = {
        // Refused by the host: blending on for an integer render target that the pixel shader writes.
        {"R32_UINT, blend", 1, {U, UK}, false, {true, false}, Ps::uint_out},
        {"R32_UINT, blend, independent", 1, {U, UK}, true, {true, false}, Ps::uint_out},
        {"R8G8B8A8_UINT, blend", 1, {DXGI_FORMAT_R8G8B8A8_UINT, UK}, false, {true, false}, Ps::uint_out},
        {"R10G10B10A2_UINT, blend", 1, {DXGI_FORMAT_R10G10B10A2_UINT, UK}, false, {true, false}, Ps::uint_out},
        {"R16G16_UINT, blend", 1, {DXGI_FORMAT_R16G16_UINT, UK}, false, {true, false}, Ps::uint_out},
        {"R8_UINT, blend", 1, {DXGI_FORMAT_R8_UINT, UK}, false, {true, false}, Ps::uint_out},
        {"R32G32B32A32_UINT, blend", 1, {DXGI_FORMAT_R32G32B32A32_UINT, UK}, false, {true, false}, Ps::uint_out},
        {"[UINT, UNORM], blend on target 0, independent", 2, {U, F}, true, {true, false}, Ps::uint_out},
        {"[UNORM, UINT], blend, shared, shader writes both", 2, {F, U}, false, {true, false}, Ps::mrt_out},
        {"[UNORM, UINT], blend on target 1 only, independent", 2, {F, U}, true, {false, true}, Ps::mrt_out},
        {"[UNORM, UINT], blend on both, independent", 2, {F, U}, true, {true, true}, Ps::mrt_out},
        // Accepted: blending where it applies, and blending on integer targets that nothing writes.
        {"R32_UINT, no blend", 1, {U, UK}, false, {false, false}, Ps::uint_out},
        {"R32_UINT, blend, no pixel shader", 1, {U, UK}, false, {true, false}, Ps::none},
        {"R8G8B8A8_UNORM, blend", 1, {F, UK}, false, {true, false}, Ps::float_out},
        {"R16G16B16A16_FLOAT, blend", 1, {DXGI_FORMAT_R16G16B16A16_FLOAT, UK}, false, {true, false}, Ps::float_out},
        {"[UNORM, UINT], blend, shared, shader writes target 0", 2, {F, U}, false, {true, false}, Ps::float_out},
        {"[UNORM, UINT], blend on target 0 only, independent", 2, {F, U}, true, {true, false}, Ps::mrt_out},
        {"[UNORM], blend on an unbound UINT target 1, independent", 1, {F, U}, true, {false, true}, Ps::float_out},
        {"[UNKNOWN], blend", 1, {UK, UK}, false, {true, false}, Ps::float_out},
        {"no targets, blend", 0, {UK, UK}, false, {true, false}, Ps::float_out},
    };
    unsigned compared = 0, differ = 0, warp_differs = 0;
    for (const Case& c : cases) {
        const HRESULT reference = create_pso(host.p, c, host_programs);
        const HRESULT mine = create_pso(ours.p, c, our_programs);
        const bool same = SUCCEEDED(mine) == SUCCEEDED(reference) && (SUCCEEDED(mine) || mine == reference);
        const HRESULT software = warp.p ? create_pso(warp.p, c, host_programs) : reference;
        if (SUCCEEDED(software) != SUCCEEDED(reference)) ++warp_differs;
        ++compared;
        differ += !same;
        std::printf("PSO %-56s host 0x%08x, layer 0x%08x%s\n", c.name, static_cast<unsigned>(reference),
                    static_cast<unsigned>(mine), same ? "" : "  MISMATCH");
    }
    std::printf("%u pipelines compared with %s, %u differ; WARP differs from the host in %u\n", compared, adapter.c_str(),
                differ, warp_differs);

    // Draws into an RGBA8 target 0 and an R32_UINT target 1 with blending on in the pipeline. The host blends target 0
    // and leaves target 1 as the shader wrote it (7), or untouched (100) where the shader does not write it.
    struct Draw {
        Case pipeline;
        unsigned target1_first_byte;  // the host's R32_UINT texel 0, low byte
    };
    const Draw draws[] = {
        {{"[UNORM, UINT], blend on target 0 only, independent", 2, {F, U}, true, {true, false}, Ps::mrt_out}, 7},
        {{"[UNORM, UINT], blend, shared, shader writes target 0", 2, {F, U}, false, {true, false}, Ps::float_out}, 100},
        {{"[UNORM, UINT], blend on target 1 only, independent, unwritten", 2, {F, U}, true, {false, true}, Ps::float_out},
         100},
        {{"[UNORM, UINT], no blend", 2, {F, U}, true, {false, false}, Ps::mrt_out}, 7},
    };
    unsigned drawn = 0, draws_failed = 0;
    for (const Draw& d : draws) {
        const Case& c = d.pipeline;
        std::vector<unsigned char> reference[2], mine[2];
        if (!draw_and_read(host.p, c, host_programs, reference)) {
            std::fprintf(stderr, "draw %s: the host failed\n", c.name);
            return 1;
        }
        if (!draw_and_read(ours.p, c, our_programs, mine)) {
            std::fprintf(stderr, "draw %s: the layer failed\n", c.name);
            return 1;
        }
        ++drawn;
        const bool same = mine[0] == reference[0] && mine[1] == reference[1];
        const bool as_expected = reference[1].size() >= 4 && reference[1][0] == d.target1_first_byte;
        draws_failed += !same || !as_expected;
        std::printf("draw %-62s target0 host %s, layer %s; target1 host %s, layer %s%s%s\n", c.name,
                    hex(reference[0], 4).c_str(), hex(mine[0], 4).c_str(), hex(reference[1], 4).c_str(),
                    hex(mine[1], 4).c_str(), same ? "" : "  MISMATCH", as_expected ? "" : "  UNEXPECTED HOST TEXEL");
    }
    std::printf("%u draws compared with %s, %u differ\n", drawn, adapter.c_str(), draws_failed);
    if (vk.validation_errors()) {
        std::fprintf(stderr, "%u Vulkan validation errors\n", vk.validation_errors());
        return 1;
    }
    return differ || draws_failed ? 1 : 0;
}
