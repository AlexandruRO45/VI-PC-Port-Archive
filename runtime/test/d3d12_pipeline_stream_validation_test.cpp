// ID3D12Device2::CreatePipelineState & parse_pipeline_stream validation parity against live Windows D3D12 (WARP & HW).
// Covers null desc / null out / riid precedence, compute + graphics subobject/stage rules, mesh/amplification vs
// legacy geometry stage mixing, RASTERIZER1/RASTERIZER2 float DepthBias edge cases (NaN, +/-Inf, out-of-range),
// truncated subobjects, and duplicate subobject families.
// No game data. Set LEONIDA_VK_VALIDATION=1 to check Vulkan usage under VK_LAYER_KHRONOS_validation.
#include "d3d12_test_shaders.h"
#include "host_adapter.h"
#include "host_wine.h"
#include "shader_fixture_path.h"
#include "test_temp_dir.h"
#include "vk_test_device.h"
#include "../shims/d3d12_com.h"
#include "../src/shader_library.h"

#include <d3dcompiler.h>
#include <dxgi1_4.h>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#if !defined(LEONIDA_TEST_VS_FULLSCREEN_SPV) || !defined(LEONIDA_TEST_PS_SPV) || !defined(LEONIDA_TEST_CS_SPV) || \
    !defined(LEONIDA_TEST_MS_SPV)
#error "Test shader SPIR-V paths must be defined"
#endif

#define EXPECT(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return 1; } \
} while (0)

using namespace d3d12test;

namespace {

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4324)  // structure was padded due to alignment specifier (matches d3dx12 subobject layout)
#endif
template <D3D12_PIPELINE_STATE_SUBOBJECT_TYPE Type, typename Inner>
struct alignas(void*) Subobject {
    D3D12_PIPELINE_STATE_SUBOBJECT_TYPE type = Type;
    Inner inner{};
};
#ifdef _MSC_VER
#pragma warning(pop)
#endif

struct Stream {
    std::vector<std::uint8_t> bytes;
    template <typename S>
    Stream& add(const S& subobject) {
        static_assert(sizeof(S) % sizeof(void*) == 0);
        const std::size_t at = bytes.size();
        bytes.resize(at + sizeof(S));
        std::memcpy(bytes.data() + at, &subobject, sizeof(S));
        return *this;
    }
    D3D12_PIPELINE_STATE_STREAM_DESC desc(std::size_t trim = 0) const {
        return {bytes.size() - trim, const_cast<std::uint8_t*>(bytes.data())};
    }
};

struct HostContext {
    std::string name;
    bool warp = false;
    Ref<IDXGIAdapter> adapter;
    Ref<ID3D12Device2> device;
    Ref<ID3D12RootSignature> root;
    std::vector<std::uint8_t> root_blob;
    D3D12_SHADER_BYTECODE vs{};
    D3D12_SHADER_BYTECODE ps{};
    D3D12_SHADER_BYTECODE cs{};
    D3D12_SHADER_BYTECODE gs{};
    D3D12_SHADER_BYTECODE hs{};
    D3D12_SHADER_BYTECODE ds{};

    bool ensure_alive() {
        if (device.p && device->GetDeviceRemovedReason() == S_OK) return true;
        root.reset();
        device.reset();
        if (FAILED(D3D12CreateDevice(adapter.p, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&device.p)))) return false;
        return SUCCEEDED(
            device->CreateRootSignature(0, root_blob.data(), root_blob.size(), IID_PPV_ARGS(&root.p)));
    }
};

bool compile_hlsl(pD3DCompile compile, const char* src, const char* profile, Ref<ID3DBlob>& out) {
    Ref<ID3DBlob> err;
    const HRESULT hr = compile(src, std::strlen(src), nullptr, nullptr, nullptr, "main", profile, 0, 0, &out.p, &err.p);
    if (SUCCEEDED(hr) && out.p) return true;
    std::fprintf(stderr, "D3DCompile (%s) failed: 0x%08lx\n", profile, static_cast<unsigned long>(hr));
    return false;
}

D3D12_RT_FORMAT_ARRAY one_rt() {
    D3D12_RT_FORMAT_ARRAY rts{};
    rts.NumRenderTargets = 1;
    rts.RTFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
    return rts;
}

D3D12_RASTERIZER_DESC default_raster() {
    D3D12_RASTERIZER_DESC r{};
    r.FillMode = D3D12_FILL_MODE_SOLID;
    r.CullMode = D3D12_CULL_MODE_NONE;
    r.DepthClipEnable = TRUE;
    return r;
}

D3D12_RASTERIZER_DESC1 default_raster1(float depth_bias = 0.0f) {
    D3D12_RASTERIZER_DESC1 r{};
    r.FillMode = D3D12_FILL_MODE_SOLID;
    r.CullMode = D3D12_CULL_MODE_NONE;
    r.DepthClipEnable = TRUE;
    r.DepthBias = depth_bias;
    return r;
}

D3D12_RASTERIZER_DESC2 default_raster2(float depth_bias = 0.0f) {
    D3D12_RASTERIZER_DESC2 r{};
    r.FillMode = D3D12_FILL_MODE_SOLID;
    r.CullMode = D3D12_CULL_MODE_NONE;
    r.DepthClipEnable = TRUE;
    r.DepthBias = depth_bias;
    return r;
}

}  // namespace

int main() {
    vktest::Device vk_dev;
    vk_dev.request_mesh_shader = true;
    const auto status = vk_dev.initialize();
    if (status == vktest::DeviceStatus::unavailable) {
        std::fprintf(stderr, "SKIP: no usable Vulkan 1.3 device\n");
        return 77;
    }
    EXPECT(status == vktest::DeviceStatus::ready);

    const auto vs_prog = fssm_program({0x56532001, 0xbf810000});
    const auto ps_prog = fssm_program({0x50532001, 0xbf810000});
    const auto cs_prog = fssm_program({0x43532001, 0xbf810000});
    const auto ms_prog = fssm_program({0x4d532001, 0xbf810000});

    const auto vs_container = shader_container(vs_prog);
    const auto ps_container = shader_container(ps_prog);
    const auto cs_container = shader_container(cs_prog);
    const auto ms_container = shader_container(ms_prog);

    const testfiles::ScratchDir scratch("leonida_pso_stream_validation_test");
    const auto& lib_dir = scratch.path();
    EXPECT(testfiles::copy_file(vktest::shader_fixture_path(LEONIDA_TEST_VS_FULLSCREEN_SPV),
                                lib_dir / "vs_fullscreen.spv"));
    EXPECT(testfiles::copy_file(vktest::shader_fixture_path(LEONIDA_TEST_PS_SPV), lib_dir / "ps_draw.spv"));
    EXPECT(testfiles::copy_file(vktest::shader_fixture_path(LEONIDA_TEST_CS_SPV), lib_dir / "cs_user_data.spv"));
    EXPECT(testfiles::copy_file(vktest::shader_fixture_path(LEONIDA_TEST_MS_SPV), lib_dir / "ms_draw.spv"));
    {
        std::ofstream index(lib_dir / "index.json", std::ios::binary);
        index << "{\"vs_fullscreen\": {\"stage\": \"vs\", \"fssm_sha256\": \"" << leonida::sha256_hex(vs_prog)
              << "\", \"bindings\": [], \"push_constant_bytes\": 144, \"input_signature\": []},\n"
                 " \"ps_draw\": {\"stage\": \"ps\", \"fssm_sha256\": \"" << leonida::sha256_hex(ps_prog)
              << "\", \"bindings\": [], \"push_constant_bytes\": 144, \"outputs\": [0]},\n"
                 " \"cs_user_data\": {\"stage\": \"cs\", \"fssm_sha256\": \"" << leonida::sha256_hex(cs_prog)
              << "\", \"bindings\": [], \"push_constant_bytes\": 144},\n"
                 " \"ms_draw\": {\"stage\": \"ms\", \"fssm_sha256\": \"" << leonida::sha256_hex(ms_prog)
              << "\", \"bindings\": [], \"push_constant_bytes\": 144}}";
    }

    rt::d3d12::VulkanDeviceInfo info{};
    info.get_instance_proc = vk_dev.instance_proc;
    info.instance = vk_dev.instance;
    info.physical = vk_dev.physical;
    info.device = vk_dev.device;
    info.queue = vk_dev.queue;
    info.queue_family = vk_dev.family;
    info.buffer_device_address = true;
    info.subgroup = vk_dev.subgroup;
    info.mesh_shader = vk_dev.mesh_shader;
    info.shader_dir = lib_dir.string();

    Ref<ID3D12Device2> layer_dev;
    EXPECT(SUCCEEDED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&layer_dev.p))));

    D3D12_VERSIONED_ROOT_SIGNATURE_DESC root_desc{};
    root_desc.Version = D3D_ROOT_SIGNATURE_VERSION_1_0;
    root_desc.Desc_1_0.Flags = D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT;
    const auto root_blob = serialize_root_signature(root_desc);
    EXPECT(!root_blob.empty());
    Ref<ID3D12RootSignature> layer_root;
    EXPECT(SUCCEEDED(
        layer_dev->CreateRootSignature(0, root_blob.data(), root_blob.size(), IID_PPV_ARGS(&layer_root.p))));

    const D3D12_SHADER_BYTECODE layer_vs{vs_container.data(), vs_container.size()};
    const D3D12_SHADER_BYTECODE layer_ps{ps_container.data(), ps_container.size()};
    const D3D12_SHADER_BYTECODE layer_cs{cs_container.data(), cs_container.size()};
    const D3D12_SHADER_BYTECODE layer_ms{ms_container.data(), ms_container.size()};
    static const std::uint8_t dummy_bytes[16] = {1, 2, 3, 4};
    const D3D12_SHADER_BYTECODE dummy_bc{dummy_bytes, sizeof(dummy_bytes)};
    const D3D12_SHADER_BYTECODE empty_bc{nullptr, 0};

    // Live host D3D12 devices (WARP and HW) on native Windows for exact HRESULT parity.
    Ref<ID3DBlob> hlsl_vs, hlsl_ps, hlsl_cs, hlsl_gs, hlsl_hs, hlsl_ds;
    std::vector<std::unique_ptr<HostContext>> hosts;
    HMODULE compiler_dll = nullptr;
    if (!hostwine::host_is_wine()) {
        compiler_dll = LoadLibraryA("d3dcompiler_47.dll");
        auto compile = compiler_dll ? reinterpret_cast<pD3DCompile>(GetProcAddress(compiler_dll, "D3DCompile")) : nullptr;
        if (compile) {
            const char* kVs = "float4 main(uint id : SV_VertexID) : SV_Position { return float4(0,0,0,1); }";
            const char* kPs = "float4 main() : SV_Target { return float4(1,0,0,1); }";
            const char* kCs = "[numthreads(1,1,1)] void main() {}";
            const char* kGs =
                "struct V { float4 p : SV_Position; }; "
                "[maxvertexcount(3)] void main(triangle V input[3], inout TriangleStream<V> s) { s.Append(input[0]); }";
            const char* kHs =
                "struct V { float4 p : SV_Position; }; "
                "struct PC { float e[3] : SV_TessFactor; float i : SV_InsideTessFactor; }; "
                "PC pc() { PC o; o.e[0]=o.e[1]=o.e[2]=o.i=1; return o; } "
                "[domain(\"tri\")][partitioning(\"integer\")][outputtopology(\"triangle_cw\")]"
                "[outputcontrolpoints(3)][patchconstantfunc(\"pc\")] "
                "V main(InputPatch<V,3> p, uint i : SV_OutputControlPointID) { return p[i]; }";
            const char* kDs =
                "struct V { float4 p : SV_Position; }; "
                "struct PC { float e[3] : SV_TessFactor; float i : SV_InsideTessFactor; }; "
                "[domain(\"tri\")] V main(PC c, float3 u : SV_DomainLocation, const OutputPatch<V,3> p) { return p[0]; }";
            if (compile_hlsl(compile, kVs, "vs_5_0", hlsl_vs) && compile_hlsl(compile, kPs, "ps_5_0", hlsl_ps) &&
                compile_hlsl(compile, kCs, "cs_5_0", hlsl_cs) && compile_hlsl(compile, kGs, "gs_5_0", hlsl_gs) &&
                compile_hlsl(compile, kHs, "hs_5_0", hlsl_hs) && compile_hlsl(compile, kDs, "ds_5_0", hlsl_ds)) {
                Ref<IDXGIFactory4> factory;
                if (SUCCEEDED(CreateDXGIFactory1(IID_PPV_ARGS(&factory.p)))) {
                    for (const bool warp : {false, true}) {
                        Ref<IDXGIAdapter> adapter;
                        std::string name;
                        if (!hostadapter::reference(factory.p, vk_dev, warp, &adapter.p, name)) continue;
                        auto hc = std::make_unique<HostContext>();
                        hc->name = name;
                        hc->warp = warp;
                        hc->adapter.p = adapter.p;
                        adapter.p = nullptr;
                        hc->root_blob = root_blob;
                        if (!hc->ensure_alive()) continue;
                        hc->vs = {hlsl_vs->GetBufferPointer(), hlsl_vs->GetBufferSize()};
                        hc->ps = {hlsl_ps->GetBufferPointer(), hlsl_ps->GetBufferSize()};
                        hc->cs = {hlsl_cs->GetBufferPointer(), hlsl_cs->GetBufferSize()};
                        hc->gs = {hlsl_gs->GetBufferPointer(), hlsl_gs->GetBufferSize()};
                        hc->hs = {hlsl_hs->GetBufferPointer(), hlsl_hs->GetBufferSize()};
                        hc->ds = {hlsl_ds->GetBufferPointer(), hlsl_ds->GetBufferSize()};
                        hosts.push_back(std::move(hc));
                    }
                }
            }
        }
    }

    auto make_cs_stream = [&](ID3D12RootSignature* rs, D3D12_SHADER_BYTECODE cs) {
        Stream s;
        s.add(Subobject<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_ROOT_SIGNATURE, ID3D12RootSignature*>{
            D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_ROOT_SIGNATURE, rs});
        s.add(Subobject<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_CS, D3D12_SHADER_BYTECODE>{
            D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_CS, cs});
        return s;
    };

    auto make_gfx_stream = [&](ID3D12RootSignature* rs, D3D12_SHADER_BYTECODE vs, D3D12_SHADER_BYTECODE ps) {
        Stream s;
        s.add(Subobject<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_ROOT_SIGNATURE, ID3D12RootSignature*>{
            D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_ROOT_SIGNATURE, rs});
        s.add(Subobject<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_VS, D3D12_SHADER_BYTECODE>{
            D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_VS, vs});
        s.add(Subobject<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PS, D3D12_SHADER_BYTECODE>{
            D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PS, ps});
        s.add(Subobject<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PRIMITIVE_TOPOLOGY, D3D12_PRIMITIVE_TOPOLOGY_TYPE>{
            D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PRIMITIVE_TOPOLOGY, D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE});
        s.add(Subobject<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RENDER_TARGET_FORMATS, D3D12_RT_FORMAT_ARRAY>{
            D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RENDER_TARGET_FORMATS, one_rt()});
        return s;
    };

    unsigned checked = 0, host_compared = 0;

    auto check_layer_only = [&](const char* label, HRESULT expected, auto builder, std::size_t trim = 0) -> bool {
        Stream ls = builder(layer_root.p, layer_vs, layer_ps, layer_cs, layer_ms, dummy_bc, dummy_bc, dummy_bc);
        const D3D12_PIPELINE_STATE_STREAM_DESC ld = ls.desc(trim);
        Ref<ID3D12PipelineState> lpso;
        const HRESULT actual = layer_dev->CreatePipelineState(&ld, IID_PPV_ARGS(&lpso.p));
        if (actual != expected) {
            std::fprintf(stderr, "FAIL [%s]: layer returned 0x%08lx, expected 0x%08lx\n", label,
                         static_cast<unsigned long>(actual), static_cast<unsigned long>(expected));
            return false;
        }
        if (FAILED(actual) && lpso.p) {
            std::fprintf(stderr, "FAIL [%s]: layer left out non-null on failure\n", label);
            return false;
        }
        ++checked;
        return true;
    };

    auto check_parity = [&](const char* label, HRESULT expected, auto builder, std::size_t trim = 0) -> bool {
        Stream ls = builder(layer_root.p, layer_vs, layer_ps, layer_cs, layer_ms, dummy_bc, dummy_bc, dummy_bc);
        const D3D12_PIPELINE_STATE_STREAM_DESC ld = ls.desc(trim);
        Ref<ID3D12PipelineState> lpso;
        const HRESULT actual = layer_dev->CreatePipelineState(&ld, IID_PPV_ARGS(&lpso.p));
        if (actual != expected) {
            std::fprintf(stderr, "FAIL [%s]: layer returned 0x%08lx, expected 0x%08lx\n", label,
                         static_cast<unsigned long>(actual), static_cast<unsigned long>(expected));
            return false;
        }
        if (FAILED(actual) && lpso.p) {
            std::fprintf(stderr, "FAIL [%s]: layer left out non-null on failure\n", label);
            return false;
        }
        ++checked;
        for (auto& hcp : hosts) {
            EXPECT(hcp->ensure_alive());
            Stream hs = builder(hcp->root.p, hcp->vs, hcp->ps, hcp->cs, dummy_bc, hcp->gs, hcp->hs, hcp->ds);
            const D3D12_PIPELINE_STATE_STREAM_DESC hd = hs.desc(trim);
            Ref<ID3D12PipelineState> hpso;
            const HRESULT hhr = hcp->device->CreatePipelineState(&hd, IID_PPV_ARGS(&hpso.p));
            if (hhr != actual) {
                std::fprintf(stderr, "FAIL [%s]: layer 0x%08lx != host %s 0x%08lx\n", label,
                             static_cast<unsigned long>(actual), hcp->name.c_str(), static_cast<unsigned long>(hhr));
                return false;
            }
            ++host_compared;
        }
        return true;
    };

    // 1. Null desc / null out / riid precedence
    {
        ID3D12PipelineState* sentinel = reinterpret_cast<ID3D12PipelineState*>(0x1234);
        EXPECT(layer_dev->CreatePipelineState(nullptr, IID_PPV_ARGS(&sentinel)) == E_INVALIDARG);
        EXPECT(sentinel == nullptr);
        EXPECT(layer_dev->CreatePipelineState(nullptr, __uuidof(ID3D12PipelineState), nullptr) == E_INVALIDARG);
        checked += 2;

        const Stream vcs = make_cs_stream(layer_root.p, layer_cs);
        const D3D12_PIPELINE_STATE_STREAM_DESC vcs_d = vcs.desc();
        const Stream vgfx = make_gfx_stream(layer_root.p, layer_vs, layer_ps);
        const D3D12_PIPELINE_STATE_STREAM_DESC vgfx_d = vgfx.desc();
        const D3D12_PIPELINE_STATE_STREAM_DESC empty_d{0, nullptr};

        EXPECT(layer_dev->CreatePipelineState(&vcs_d, __uuidof(ID3D12PipelineState), nullptr) == S_FALSE);
        EXPECT(layer_dev->CreatePipelineState(&vgfx_d, __uuidof(ID3D12PipelineState), nullptr) == S_FALSE);
        EXPECT(layer_dev->CreatePipelineState(&empty_d, __uuidof(ID3D12PipelineState), nullptr) == E_INVALIDARG);
        EXPECT(layer_dev->CreatePipelineState(&vcs_d, __uuidof(ID3D12Resource), nullptr) == S_FALSE);
        sentinel = reinterpret_cast<ID3D12PipelineState*>(0x1234);
        EXPECT(layer_dev->CreatePipelineState(&vcs_d, __uuidof(ID3D12Resource), reinterpret_cast<void**>(&sentinel)) ==
               E_NOINTERFACE);
        EXPECT(sentinel == nullptr);
        checked += 5;

        for (auto& hcp : hosts) {
            EXPECT(hcp->ensure_alive());
            const Stream hvcs = make_cs_stream(hcp->root.p, hcp->cs);
            const D3D12_PIPELINE_STATE_STREAM_DESC hvcs_d = hvcs.desc();
            const Stream hvgfx = make_gfx_stream(hcp->root.p, hcp->vs, hcp->ps);
            const D3D12_PIPELINE_STATE_STREAM_DESC hvgfx_d = hvgfx.desc();
            EXPECT(hcp->device->CreatePipelineState(&hvcs_d, __uuidof(ID3D12PipelineState), nullptr) == S_FALSE);
            EXPECT(hcp->device->CreatePipelineState(&hvgfx_d, __uuidof(ID3D12PipelineState), nullptr) == S_FALSE);
            EXPECT(hcp->device->CreatePipelineState(&empty_d, __uuidof(ID3D12PipelineState), nullptr) == E_INVALIDARG);
            EXPECT(hcp->device->CreatePipelineState(&hvcs_d, __uuidof(ID3D12Resource), nullptr) == S_FALSE);
            ID3D12PipelineState* hsent = reinterpret_cast<ID3D12PipelineState*>(0x1234);
            EXPECT(hcp->device->CreatePipelineState(&hvcs_d, __uuidof(ID3D12Resource),
                                                    reinterpret_cast<void**>(&hsent)) == E_NOINTERFACE);
            EXPECT(hsent == nullptr);
            host_compared += 5;
        }
    }

    // 2. Stream framing, null/zero, truncation, duplicate families, unknown subobject
    {
        int buf[4]{};
        const D3D12_PIPELINE_STATE_STREAM_DESC zero_sz{0, buf};
        const D3D12_PIPELINE_STATE_STREAM_DESC null_ptr{16, nullptr};
        Ref<ID3D12PipelineState> pso;
        EXPECT(layer_dev->CreatePipelineState(&zero_sz, IID_PPV_ARGS(&pso.p)) == E_INVALIDARG);
        EXPECT(layer_dev->CreatePipelineState(&null_ptr, IID_PPV_ARGS(&pso.p)) == E_INVALIDARG);
        checked += 2;
        for (auto& hcp : hosts) {
            EXPECT(hcp->ensure_alive());
            Ref<ID3D12PipelineState> hpso;
            EXPECT(hcp->device->CreatePipelineState(&zero_sz, IID_PPV_ARGS(&hpso.p)) == E_INVALIDARG);
            EXPECT(hcp->device->CreatePipelineState(&null_ptr, IID_PPV_ARGS(&hpso.p)) == E_INVALIDARG);
            host_compared += 2;
        }
    }

    EXPECT(check_parity("valid_cs", S_OK, [&](auto* rs, auto, auto, auto cs, auto, auto, auto, auto) {
        return make_cs_stream(rs, cs);
    }));
    EXPECT(check_parity("valid_gfx", S_OK, [&](auto* rs, auto vs, auto ps, auto, auto, auto, auto, auto) {
        return make_gfx_stream(rs, vs, ps);
    }));
    EXPECT(check_layer_only("truncated_cs_4b", E_INVALIDARG,
                            [&](auto* rs, auto, auto, auto cs, auto, auto, auto, auto) { return make_cs_stream(rs, cs); },
                            4));
    EXPECT(check_layer_only("truncated_raster1", E_INVALIDARG,
                            [&](auto* rs, auto vs, auto ps, auto, auto, auto, auto, auto) {
                                auto s = make_gfx_stream(rs, vs, ps);
                                s.add(Subobject<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RASTERIZER1, D3D12_RASTERIZER_DESC1>{
                                    D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RASTERIZER1, default_raster1(1.5f)});
                                return s;
                            },
                            8));
    EXPECT(check_layer_only("truncated_raster2", E_INVALIDARG,
                            [&](auto* rs, auto vs, auto ps, auto, auto, auto, auto, auto) {
                                auto s = make_gfx_stream(rs, vs, ps);
                                s.add(Subobject<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RASTERIZER2, D3D12_RASTERIZER_DESC2>{
                                    D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RASTERIZER2, default_raster2(1.5f)});
                                return s;
                            },
                            8));
    EXPECT(check_parity("duplicate_raster_and_raster1", E_INVALIDARG,
                        [&](auto* rs, auto vs, auto ps, auto, auto, auto, auto, auto) {
                            auto s = make_gfx_stream(rs, vs, ps);
                            s.add(Subobject<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RASTERIZER, D3D12_RASTERIZER_DESC>{
                                D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RASTERIZER, default_raster()});
                            s.add(Subobject<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RASTERIZER1, D3D12_RASTERIZER_DESC1>{
                                D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RASTERIZER1, default_raster1()});
                            return s;
                        }));
    EXPECT(check_parity("duplicate_ds_and_ds2", E_INVALIDARG,
                        [&](auto* rs, auto vs, auto ps, auto, auto, auto, auto, auto) {
                            auto s = make_gfx_stream(rs, vs, ps);
                            D3D12_DEPTH_STENCIL_DESC d0{};
                            D3D12_DEPTH_STENCIL_DESC2 d2{};
                            s.add(Subobject<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL, D3D12_DEPTH_STENCIL_DESC>{
                                D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL, d0});
                            s.add(Subobject<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL2,
                                            D3D12_DEPTH_STENCIL_DESC2>{
                                D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL2, d2});
                            return s;
                        }));
    EXPECT(check_parity("unknown_subobject_999", E_INVALIDARG,
                        [&](auto* rs, auto, auto, auto cs, auto, auto, auto, auto) {
                            auto s = make_cs_stream(rs, cs);
                            s.add(Subobject<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE(999), UINT>{
                                D3D12_PIPELINE_STATE_SUBOBJECT_TYPE(999), 0});
                            return s;
                        }));
    EXPECT(check_parity("cs_empty_only", E_INVALIDARG, [&](auto* rs, auto, auto, auto, auto, auto, auto, auto) {
        return make_cs_stream(rs, empty_bc);
    }));
    EXPECT(check_parity("vs_empty_only", E_INVALIDARG, [&](auto* rs, auto, auto, auto, auto, auto, auto, auto) {
        Stream s;
        s.add(Subobject<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_ROOT_SIGNATURE, ID3D12RootSignature*>{
            D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_ROOT_SIGNATURE, rs});
        s.add(Subobject<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_VS, D3D12_SHADER_BYTECODE>{
            D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_VS, empty_bc});
        return s;
    }));

    // 3. Compute stream + graphics-only subobjects & stage conflicts (exact Windows 11 WARP + HW parity)
    EXPECT(check_parity("cs_plus_blend", S_OK, [&](auto* rs, auto, auto, auto cs, auto, auto, auto, auto) {
        auto s = make_cs_stream(rs, cs);
        D3D12_BLEND_DESC bd{};
        bd.RenderTarget[0].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
        s.add(Subobject<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_BLEND, D3D12_BLEND_DESC>{
            D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_BLEND, bd});
        return s;
    }));
    EXPECT(check_parity("cs_plus_rasterizer", S_OK, [&](auto* rs, auto, auto, auto cs, auto, auto, auto, auto) {
        auto s = make_cs_stream(rs, cs);
        s.add(Subobject<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RASTERIZER, D3D12_RASTERIZER_DESC>{
            D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RASTERIZER, default_raster()});
        return s;
    }));
    EXPECT(check_parity("cs_plus_ds1_depth_bounds", S_OK, [&](auto* rs, auto, auto, auto cs, auto, auto, auto, auto) {
        auto s = make_cs_stream(rs, cs);
        D3D12_DEPTH_STENCIL_DESC1 d{};
        d.DepthBoundsTestEnable = TRUE;
        s.add(Subobject<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL1, D3D12_DEPTH_STENCIL_DESC1>{
            D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL1, d});
        return s;
    }));
    EXPECT(check_parity("cs_plus_view_instancing", S_OK, [&](auto* rs, auto, auto, auto cs, auto, auto, auto, auto) {
        auto s = make_cs_stream(rs, cs);
        static const D3D12_VIEW_INSTANCE_LOCATION locs[2] = {{0, 0}, {1, 1}};
        D3D12_VIEW_INSTANCING_DESC v{2, locs, D3D12_VIEW_INSTANCING_FLAG_NONE};
        s.add(Subobject<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_VIEW_INSTANCING, D3D12_VIEW_INSTANCING_DESC>{
            D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_VIEW_INSTANCING, v});
        return s;
    }));
    EXPECT(check_parity("cs_plus_rt_formats_valid", S_OK, [&](auto* rs, auto, auto, auto cs, auto, auto, auto, auto) {
        auto s = make_cs_stream(rs, cs);
        s.add(Subobject<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RENDER_TARGET_FORMATS, D3D12_RT_FORMAT_ARRAY>{
            D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RENDER_TARGET_FORMATS, one_rt()});
        return s;
    }));
    EXPECT(check_parity("cs_plus_rt_formats_num9", E_INVALIDARG,
                        [&](auto* rs, auto, auto, auto cs, auto, auto, auto, auto) {
                            auto s = make_cs_stream(rs, cs);
                            D3D12_RT_FORMAT_ARRAY rts{};
                            rts.NumRenderTargets = 9;
                            s.add(Subobject<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RENDER_TARGET_FORMATS,
                                            D3D12_RT_FORMAT_ARRAY>{
                                D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RENDER_TARGET_FORMATS, rts});
                            return s;
                        }));
    EXPECT(check_parity("cs_plus_vs_empty", S_OK, [&](auto* rs, auto, auto, auto cs, auto, auto, auto, auto) {
        auto s = make_cs_stream(rs, cs);
        s.add(Subobject<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_VS, D3D12_SHADER_BYTECODE>{
            D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_VS, empty_bc});
        return s;
    }));
    EXPECT(check_parity("cs_plus_ms_empty", S_OK, [&](auto* rs, auto, auto, auto cs, auto, auto, auto, auto) {
        auto s = make_cs_stream(rs, cs);
        s.add(Subobject<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_MS, D3D12_SHADER_BYTECODE>{
            D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_MS, empty_bc});
        return s;
    }));
    EXPECT(check_parity("cs_plus_as_empty", S_OK, [&](auto* rs, auto, auto, auto cs, auto, auto, auto, auto) {
        auto s = make_cs_stream(rs, cs);
        s.add(Subobject<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_AS, D3D12_SHADER_BYTECODE>{
            D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_AS, empty_bc});
        return s;
    }));
    EXPECT(check_parity("gfx_plus_cs_empty", S_OK, [&](auto* rs, auto vs, auto ps, auto, auto, auto, auto, auto) {
        auto s = make_gfx_stream(rs, vs, ps);
        s.add(Subobject<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_CS, D3D12_SHADER_BYTECODE>{
            D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_CS, empty_bc});
        return s;
    }));
    EXPECT(check_parity("gfx_plus_as_empty", S_OK, [&](auto* rs, auto vs, auto ps, auto, auto, auto, auto, auto) {
        auto s = make_gfx_stream(rs, vs, ps);
        s.add(Subobject<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_AS, D3D12_SHADER_BYTECODE>{
            D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_AS, empty_bc});
        return s;
    }));
    EXPECT(check_parity("cs_plus_vs_valid", E_INVALIDARG,
                        [&](auto* rs, auto vs, auto, auto cs, auto, auto, auto, auto) {
                            auto s = make_cs_stream(rs, cs);
                            s.add(Subobject<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_VS, D3D12_SHADER_BYTECODE>{
                                D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_VS, vs});
                            return s;
                        }));
    EXPECT(check_parity("cs_plus_ms_nonempty", E_INVALIDARG,
                        [&](auto* rs, auto, auto, auto cs, auto ms, auto, auto, auto) {
                            auto s = make_cs_stream(rs, cs);
                            s.add(Subobject<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_MS, D3D12_SHADER_BYTECODE>{
                                D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_MS, ms});
                            return s;
                        }));
    EXPECT(check_parity("cs_plus_as_nonempty", E_INVALIDARG,
                        [&](auto* rs, auto, auto, auto cs, auto, auto, auto, auto) {
                            auto s = make_cs_stream(rs, cs);
                            s.add(Subobject<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_AS, D3D12_SHADER_BYTECODE>{
                                D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_AS, dummy_bc});
                            return s;
                        }));

    // 4. Mesh / Amplification vs Legacy Geometry stage mixing
    EXPECT(check_parity("gfx_plus_ms_nonempty", E_INVALIDARG,
                        [&](auto* rs, auto vs, auto ps, auto, auto ms, auto, auto, auto) {
                            auto s = make_gfx_stream(rs, vs, ps);
                            s.add(Subobject<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_MS, D3D12_SHADER_BYTECODE>{
                                D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_MS, ms});
                            return s;
                        }));
    EXPECT(check_parity("gfx_plus_ms_empty", S_OK,
                        [&](auto* rs, auto vs, auto ps, auto, auto, auto, auto, auto) {
                            auto s = make_gfx_stream(rs, vs, ps);
                            s.add(Subobject<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_MS, D3D12_SHADER_BYTECODE>{
                                D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_MS, empty_bc});
                            return s;
                        }));
    EXPECT(check_parity("gfx_plus_as_nonempty", E_INVALIDARG,
                        [&](auto* rs, auto vs, auto ps, auto, auto, auto, auto, auto) {
                            auto s = make_gfx_stream(rs, vs, ps);
                            s.add(Subobject<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_AS, D3D12_SHADER_BYTECODE>{
                                D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_AS, dummy_bc});
                            return s;
                        }));
    EXPECT(check_parity("as_plus_gs", E_INVALIDARG, [&](auto* rs, auto, auto, auto, auto, auto gs, auto, auto) {
        Stream s;
        s.add(Subobject<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_ROOT_SIGNATURE, ID3D12RootSignature*>{
            D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_ROOT_SIGNATURE, rs});
        s.add(Subobject<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_AS, D3D12_SHADER_BYTECODE>{
            D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_AS, dummy_bc});
        s.add(Subobject<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_GS, D3D12_SHADER_BYTECODE>{
            D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_GS, gs});
        return s;
    }));
    EXPECT(check_parity("as_plus_hs_ds", E_INVALIDARG, [&](auto* rs, auto, auto, auto, auto, auto, auto hs, auto ds) {
        Stream s;
        s.add(Subobject<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_ROOT_SIGNATURE, ID3D12RootSignature*>{
            D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_ROOT_SIGNATURE, rs});
        s.add(Subobject<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_AS, D3D12_SHADER_BYTECODE>{
            D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_AS, dummy_bc});
        s.add(Subobject<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_HS, D3D12_SHADER_BYTECODE>{
            D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_HS, hs});
        s.add(Subobject<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DS, D3D12_SHADER_BYTECODE>{
            D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DS, ds});
        return s;
    }));
    EXPECT(check_parity("as_plus_ps_no_ms", E_INVALIDARG, [&](auto* rs, auto, auto ps, auto, auto, auto, auto, auto) {
        Stream s;
        s.add(Subobject<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_ROOT_SIGNATURE, ID3D12RootSignature*>{
            D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_ROOT_SIGNATURE, rs});
        s.add(Subobject<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_AS, D3D12_SHADER_BYTECODE>{
            D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_AS, dummy_bc});
        s.add(Subobject<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PS, D3D12_SHADER_BYTECODE>{
            D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PS, ps});
        return s;
    }));
    EXPECT(check_parity("ms_plus_gs", E_INVALIDARG, [&](auto* rs, auto, auto, auto, auto ms, auto gs, auto, auto) {
        Stream s;
        s.add(Subobject<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_ROOT_SIGNATURE, ID3D12RootSignature*>{
            D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_ROOT_SIGNATURE, rs});
        s.add(Subobject<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_MS, D3D12_SHADER_BYTECODE>{
            D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_MS, ms});
        s.add(Subobject<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_GS, D3D12_SHADER_BYTECODE>{
            D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_GS, gs});
        return s;
    }));
    EXPECT(check_parity("ms_plus_stream_output", E_INVALIDARG,
                        [&](auto* rs, auto, auto, auto, auto ms, auto, auto, auto) {
                            Stream s;
                            s.add(Subobject<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_ROOT_SIGNATURE, ID3D12RootSignature*>{
                                D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_ROOT_SIGNATURE, rs});
                            s.add(Subobject<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_MS, D3D12_SHADER_BYTECODE>{
                                D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_MS, ms});
                            static const D3D12_SO_DECLARATION_ENTRY entry{0, "SV_Position", 0, 0, 4, 0};
                            D3D12_STREAM_OUTPUT_DESC so{&entry, 1, nullptr, 0, 0};
                            s.add(Subobject<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_STREAM_OUTPUT, D3D12_STREAM_OUTPUT_DESC>{
                                D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_STREAM_OUTPUT, so});
                            return s;
                        }));
    EXPECT(check_parity("ms_empty_plus_ps", E_INVALIDARG, [&](auto* rs, auto, auto ps, auto, auto, auto, auto, auto) {
        Stream s;
        s.add(Subobject<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_ROOT_SIGNATURE, ID3D12RootSignature*>{
            D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_ROOT_SIGNATURE, rs});
        s.add(Subobject<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_MS, D3D12_SHADER_BYTECODE>{
            D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_MS, empty_bc});
        s.add(Subobject<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PS, D3D12_SHADER_BYTECODE>{
            D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PS, ps});
        return s;
    }));

    // 5. RASTERIZER1 / RASTERIZER2 float DepthBias edge cases (zero UB, exact parity with WARP & HW)
    const struct {
        const char* name;
        float value;
    } kBiases[] = {
        {"0.0f", 0.0f},
        {"1.5f", 1.5f},
        {"-1.5f", -1.5f},
        {"NaN", std::numeric_limits<float>::quiet_NaN()},
        {"+Inf", std::numeric_limits<float>::infinity()},
        {"-Inf", -std::numeric_limits<float>::infinity()},
        {"1e20f", 1e20f},
        {"-1e20f", -1e20f},
        {"FLT_MAX", (std::numeric_limits<float>::max)()},
        {"-FLT_MAX", -(std::numeric_limits<float>::max)()},
        {"2^31", 2147483648.0f},
        {"-2^31", -2147483648.0f},
    };
    for (const auto& b : kBiases) {
        char lbl1[64], lbl2[64];
        std::snprintf(lbl1, sizeof(lbl1), "raster1_depth_bias_%s", b.name);
        std::snprintf(lbl2, sizeof(lbl2), "raster2_depth_bias_%s", b.name);
        EXPECT(check_parity(lbl1, S_OK, [&](auto* rs, auto vs, auto ps, auto, auto, auto, auto, auto) {
            auto s = make_gfx_stream(rs, vs, ps);
            s.add(Subobject<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RASTERIZER1, D3D12_RASTERIZER_DESC1>{
                D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RASTERIZER1, default_raster1(b.value)});
            return s;
        }));
        EXPECT(check_parity(lbl2, S_OK, [&](auto* rs, auto vs, auto ps, auto, auto, auto, auto, auto) {
            auto s = make_gfx_stream(rs, vs, ps);
            s.add(Subobject<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RASTERIZER2, D3D12_RASTERIZER_DESC2>{
                D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RASTERIZER2, default_raster2(b.value)});
            return s;
        }));
    }

    hosts.clear();
    hlsl_vs.reset();
    hlsl_ps.reset();
    hlsl_cs.reset();
    hlsl_gs.reset();
    hlsl_hs.reset();
    hlsl_ds.reset();
    if (compiler_dll) FreeLibrary(compiler_dll);
    std::printf("d3d12_pipeline_stream_validation_test: PASS (%u checks, %u host parity comparisons)\n", checked,
                host_compared);
    return 0;
}
