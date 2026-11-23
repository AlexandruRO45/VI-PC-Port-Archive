// Synthetic unit + host-parity tests for ID3D12Device::CreateCommandList and ID3D12GraphicsCommandList::Reset
// initial-PSO handling, argument/state precedence, foreign-PSO rejection, and bundle Reset double-bind/state-leak fixes.
// Compares the layer against live Windows D3D12 (both the default adapter and WARP, skipped under Wine).
// No game data.
#include "d3d12_test_shaders.h"
#include "host_wine.h"
#include "shader_fixture_path.h"
#include "test_temp_dir.h"
#include "vk_test_device.h"
#include "../shims/d3d12_com.h"
#include "../shims/d3d12_trace.h"
#include "../src/shader_library.h"

#include <d3dcompiler.h>
#include <dxgi1_4.h>
#include <io.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#if !defined(LEONIDA_TEST_CS_SPV) || !defined(LEONIDA_TEST_VS_SPV) || !defined(LEONIDA_TEST_PS_SPV)
#error "LEONIDA_TEST_CS_SPV, LEONIDA_TEST_VS_SPV and LEONIDA_TEST_PS_SPV must be defined"
#endif

#define CHECK(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return false; } \
} while (0)

#define REQUIRE(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return 1; } \
} while (0)

using namespace d3d12test;

namespace {

bool compile_hlsl(const char* source, const char* entry, const char* profile, std::vector<std::uint8_t>& out) {
    const HMODULE compiler = LoadLibraryExA("d3dcompiler_47.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
    const auto compile = compiler ? reinterpret_cast<pD3DCompile>(GetProcAddress(compiler, "D3DCompile")) : nullptr;
    if (!compile) return false;
    ID3DBlob* blob = nullptr;
    ID3DBlob* errors = nullptr;
    const HRESULT hr = compile(source, std::strlen(source), "cl_reset_pso.hlsl", nullptr, nullptr, entry, profile,
                               0, 0, &blob, &errors);
    if (FAILED(hr) && errors)
        std::fprintf(stderr, "%s\n", static_cast<const char*>(errors->GetBufferPointer()));
    if (errors) errors->Release();
    if (FAILED(hr) || !blob) return false;
    const auto* bytes = static_cast<const std::uint8_t*>(blob->GetBufferPointer());
    out.assign(bytes, bytes + blob->GetBufferSize());
    blob->Release();
    return true;
}

// Minimal non-layer ID3D12PipelineState implementation to verify foreign PSO rejection without pinning or crashing.
class ForeignPipelineState final : public ID3D12PipelineState {
public:
    HRESULT STDMETHODCALLTYPE QueryInterface(REFIID riid, void** out) override {
        if (!out) return E_POINTER;
        *out = nullptr;
        if (riid == __uuidof(IUnknown) || riid == __uuidof(ID3D12Object) || riid == __uuidof(ID3D12DeviceChild) ||
            riid == __uuidof(ID3D12Pageable) || riid == __uuidof(ID3D12PipelineState)) {
            AddRef();
            *out = static_cast<ID3D12PipelineState*>(this);
            return S_OK;
        }
        return E_NOINTERFACE;
    }
    ULONG STDMETHODCALLTYPE AddRef() override { return ++refs_; }
    ULONG STDMETHODCALLTYPE Release() override { return --refs_; }
    HRESULT STDMETHODCALLTYPE GetPrivateData(REFGUID, UINT*, void*) override { return DXGI_ERROR_NOT_FOUND; }
    HRESULT STDMETHODCALLTYPE SetPrivateData(REFGUID, UINT, const void*) override { return E_NOTIMPL; }
    HRESULT STDMETHODCALLTYPE SetPrivateDataInterface(REFGUID, const IUnknown*) override { return E_NOTIMPL; }
    HRESULT STDMETHODCALLTYPE SetName(LPCWSTR) override { return S_OK; }
    HRESULT STDMETHODCALLTYPE GetDevice(REFIID, void**) override { return E_NOINTERFACE; }
    HRESULT STDMETHODCALLTYPE GetCachedBlob(ID3DBlob**) override { return E_NOTIMPL; }

    ULONG ref_count() const { return refs_.load(); }

private:
    std::atomic<ULONG> refs_{1};
};

// Captures the D3D12 trace to a temporary file while alive, matching d3d12_bundle_test's TraceCapture.
class TraceCapture {
public:
    explicit TraceCapture(const std::filesystem::path& path) : path_(path) {
        std::fflush(stderr);
        saved_ = _dup(_fileno(stderr));
        std::FILE* file = nullptr;
        if (fopen_s(&file, path_.string().c_str(), "wb") == 0 && file) {
            _dup2(_fileno(file), _fileno(stderr));
            std::fclose(file);
        }
        rt::d3d12_trace::set_enabled(true);
    }
    ~TraceCapture() { finish(); }

    std::string finish() {
        if (saved_ < 0) return {};
        rt::d3d12_trace::set_enabled(false);
        std::fflush(stderr);
        _dup2(saved_, _fileno(stderr));
        _close(saved_);
        saved_ = -1;
        std::ifstream in(path_, std::ios::binary);
        return std::string(std::istreambuf_iterator<char>(in), {});
    }

private:
    std::filesystem::path path_;
    int saved_ = -1;
};

int count_of(const std::string& text, const char* needle) {
    int count = 0;
    const std::size_t length = std::strlen(needle);
    for (std::size_t at = 0; (at = text.find(needle, at)) != std::string::npos; at += length) ++count;
    return count;
}

ULONG current_refcount(IUnknown* object) {
    object->AddRef();
    return object->Release();
}

struct MatrixResults {
    std::vector<std::pair<std::string, HRESULT>> entries;
};

const char* type_name(D3D12_COMMAND_LIST_TYPE t) {
    switch (t) {
    case D3D12_COMMAND_LIST_TYPE_DIRECT: return "DIRECT";
    case D3D12_COMMAND_LIST_TYPE_BUNDLE: return "BUNDLE";
    case D3D12_COMMAND_LIST_TYPE_COMPUTE: return "COMPUTE";
    case D3D12_COMMAND_LIST_TYPE_COPY: return "COPY";
    default: return "OTHER";
    }
}

bool run_parity_matrix(ID3D12Device* device, ID3D12PipelineState* gfx_pso, ID3D12PipelineState* cs_pso,
                       MatrixResults& out) {
    const D3D12_COMMAND_LIST_TYPE types[] = {
        D3D12_COMMAND_LIST_TYPE_DIRECT,
        D3D12_COMMAND_LIST_TYPE_BUNDLE,
        D3D12_COMMAND_LIST_TYPE_COMPUTE,
        D3D12_COMMAND_LIST_TYPE_COPY,
    };

    auto record = [&](const std::string& label, HRESULT hr) { out.entries.emplace_back(label, hr); };

    // 1. CreateCommandList & Reset across all 4 types x (null, gfx, cs) PSOs.
    for (D3D12_COMMAND_LIST_TYPE t : types) {
        for (int p = 0; p < 3; ++p) {
            ID3D12PipelineState* pso = (p == 0) ? nullptr : (p == 1 ? gfx_pso : cs_pso);
            const char* pname = (p == 0) ? "null" : (p == 1 ? "gfx" : "cs");

            Ref<ID3D12CommandAllocator> alloc;
            CHECK(SUCCEEDED(device->CreateCommandAllocator(t, IID_PPV_ARGS(&alloc.p))));
            Ref<ID3D12GraphicsCommandList> cl;
            const HRESULT create_hr = device->CreateCommandList(0, t, alloc.p, pso, IID_PPV_ARGS(&cl.p));
            record(std::string("Create(") + type_name(t) + "," + pname + ")", create_hr);
            CHECK(SUCCEEDED(create_hr) && cl.p);
            const HRESULT close1_hr = cl->Close();
            record(std::string("CreateClose(") + type_name(t) + "," + pname + ")", close1_hr);

            const HRESULT reset_hr = cl->Reset(alloc.p, pso);
            record(std::string("Reset(") + type_name(t) + "," + pname + ")", reset_hr);
            CHECK(SUCCEEDED(reset_hr));
            const HRESULT close2_hr = cl->Close();
            record(std::string("ResetClose(") + type_name(t) + "," + pname + ")", close2_hr);
        }
    }

    // 2. Error & precedence cases.
    {
        Ref<ID3D12GraphicsCommandList> cl;
        record("Create(DIRECT,null_alloc,null)",
               device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, nullptr, nullptr, IID_PPV_ARGS(&cl.p)));
    }
    {
        Ref<ID3D12CommandAllocator> cs_alloc;
        Ref<ID3D12GraphicsCommandList> cl;
        CHECK(SUCCEEDED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE, IID_PPV_ARGS(&cs_alloc.p))));
        record("Create(DIRECT,COMPUTE_alloc,null)",
               device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, cs_alloc.p, nullptr, IID_PPV_ARGS(&cl.p)));
    }
    {
        Ref<ID3D12CommandAllocator> alloc;
        Ref<ID3D12GraphicsCommandList> cl1, cl2;
        CHECK(SUCCEEDED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&alloc.p))));
        CHECK(SUCCEEDED(
            device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, alloc.p, nullptr, IID_PPV_ARGS(&cl1.p))));
        record("Create(DIRECT,busy_alloc,null)",
               device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, alloc.p, nullptr, IID_PPV_ARGS(&cl2.p)));
        record("Create(DIRECT,busy_alloc,null,out=null)",
               device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, alloc.p, nullptr,
                                         __uuidof(ID3D12GraphicsCommandList), nullptr));
    }
    {
        Ref<ID3D12CommandAllocator> copy_alloc;
        CHECK(SUCCEEDED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COPY, IID_PPV_ARGS(&copy_alloc.p))));
        record("Create(COPY,match,gfx,out=null)",
               device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COPY, copy_alloc.p, gfx_pso,
                                         __uuidof(ID3D12GraphicsCommandList), nullptr));
    }
    {
        Ref<ID3D12CommandAllocator> alloc, cs_alloc, busy_alloc;
        Ref<ID3D12GraphicsCommandList> cl, busy_cl;
        CHECK(SUCCEEDED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&alloc.p))));
        CHECK(SUCCEEDED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE, IID_PPV_ARGS(&cs_alloc.p))));
        CHECK(SUCCEEDED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&busy_alloc.p))));
        CHECK(SUCCEEDED(
            device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, busy_alloc.p, nullptr, IID_PPV_ARGS(&busy_cl.p))));
        CHECK(SUCCEEDED(
            device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, alloc.p, nullptr, IID_PPV_ARGS(&cl.p))));

        // Open-list Reset precedence: E_FAIL must beat null, mismatched, and busy allocators.
        record("Reset(OPEN,match,null)", cl->Reset(alloc.p, nullptr));
        record("Reset(OPEN,null_alloc,null)", cl->Reset(nullptr, nullptr));
        record("Reset(OPEN,COMPUTE_alloc,null)", cl->Reset(cs_alloc.p, nullptr));
        record("Reset(OPEN,busy_alloc,null)", cl->Reset(busy_alloc.p, nullptr));

        // Closed-list Reset with mismatched or busy allocator: E_INVALIDARG, leaving list closed.
        CHECK(cl->Close() == S_OK);
        record("Reset(CLOSED,COMPUTE_alloc,null)", cl->Reset(cs_alloc.p, nullptr));
        record("Reset(CLOSED,busy_alloc,null)", cl->Reset(busy_alloc.p, nullptr));
        // Still closed -> valid Reset succeeds.
        record("Reset(CLOSED_after_err,match,null)", cl->Reset(alloc.p, nullptr));
        CHECK(cl->Close() == S_OK);
    }
    return true;
}

bool run_layer_only_checks(ID3D12Device* device, ID3D12PipelineState* gfx_pso, ID3D12PipelineState* cs_pso,
                           const std::filesystem::path& scratch) {
    // 1. Closed-list Reset with null allocator safely returns E_INVALIDARG (retail crashes) and leaves list closed.
    {
        Ref<ID3D12CommandAllocator> alloc;
        Ref<ID3D12GraphicsCommandList> cl;
        CHECK(SUCCEEDED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&alloc.p))));
        CHECK(SUCCEEDED(
            device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, alloc.p, nullptr, IID_PPV_ARGS(&cl.p))));
        CHECK(cl->Close() == S_OK);
        CHECK(cl->Reset(nullptr, nullptr) == E_INVALIDARG);
        CHECK(SUCCEEDED(cl->Reset(alloc.p, nullptr)));
        CHECK(cl->Close() == S_OK);
    }

    // 2. Foreign (non-layer) ID3D12PipelineState* is rejected cleanly with E_INVALIDARG on CreateCommandList and Reset,
    //    without pinning the object, mutating list open/closed state, or marking the allocator busy.
    {
        ForeignPipelineState foreign;
        CHECK(foreign.ref_count() == 1);

        for (D3D12_COMMAND_LIST_TYPE t : {D3D12_COMMAND_LIST_TYPE_DIRECT, D3D12_COMMAND_LIST_TYPE_BUNDLE}) {
            Ref<ID3D12CommandAllocator> alloc;
            Ref<ID3D12GraphicsCommandList> cl;
            CHECK(SUCCEEDED(device->CreateCommandAllocator(t, IID_PPV_ARGS(&alloc.p))));
            CHECK(device->CreateCommandList(0, t, alloc.p, &foreign, IID_PPV_ARGS(&cl.p)) == E_INVALIDARG);
            CHECK(!cl.p);
            CHECK(foreign.ref_count() == 1);

            // Allocator must still be free.
            CHECK(SUCCEEDED(device->CreateCommandList(0, t, alloc.p, nullptr, IID_PPV_ARGS(&cl.p))));
            CHECK(cl->Close() == S_OK);

            // Reset with foreign PSO must fail with E_INVALIDARG, leave list closed, and not pin foreign PSO.
            CHECK(cl->Reset(alloc.p, &foreign) == E_INVALIDARG);
            CHECK(foreign.ref_count() == 1);
            CHECK(SUCCEEDED(cl->Reset(alloc.p, nullptr)));
            CHECK(cl->Close() == S_OK);
        }
    }

    // 3. Bundle Reset(alloc, non_null_pso) double-bind & state-leak regression + parent inheritance.
    {
        const std::uintptr_t gfx_id = rt::d3d12::graphics_pipeline_id(gfx_pso);
        const std::uintptr_t cs_id = rt::d3d12::compute_pipeline_id(cs_pso);
        CHECK(gfx_id != 0 && cs_id != 0);

        const ULONG gfx_base_refs = current_refcount(gfx_pso);
        const ULONG cs_base_refs = current_refcount(cs_pso);

        Ref<ID3D12CommandAllocator> bundle_alloc, direct_alloc;
        Ref<ID3D12GraphicsCommandList> bundle, direct;
        CHECK(SUCCEEDED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_BUNDLE, IID_PPV_ARGS(&bundle_alloc.p))));
        CHECK(SUCCEEDED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&direct_alloc.p))));

        // Create bundle with gfx_pso: bundle itself must NOT bind gfx_pso directly, and must hold exactly 1 ref.
        CHECK(SUCCEEDED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_BUNDLE, bundle_alloc.p, gfx_pso,
                                                  IID_PPV_ARGS(&bundle.p))));
        std::uintptr_t g = 1, c = 1;
        CHECK(rt::d3d12::bound_pipelines(bundle.p, g, c) && g == 0 && c == 0);
        CHECK(current_refcount(gfx_pso) == gfx_base_refs + 1);
        CHECK(bundle->Close() == S_OK);

        // Reset bundle with gfx_pso: still no direct bind on bundle, and still exactly 1 ref (no double-hold).
        CHECK(SUCCEEDED(bundle->Reset(bundle_alloc.p, gfx_pso)));
        CHECK(rt::d3d12::bound_pipelines(bundle.p, g, c) && g == 0 && c == 0);
        CHECK(current_refcount(gfx_pso) == gfx_base_refs + 1);
        CHECK(bundle->Close() == S_OK);

        // Execute bundle on a direct list that has no PSO bound: parent receives gfx_pso.
        CHECK(SUCCEEDED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, direct_alloc.p, nullptr,
                                                  IID_PPV_ARGS(&direct.p))));
        CHECK(rt::d3d12::bound_pipelines(direct.p, g, c) && g == 0 && c == 0);
        direct->ExecuteBundle(bundle.p);
        CHECK(rt::d3d12::bound_pipelines(direct.p, g, c) && g == gfx_id && c == 0);

        // Reset bundle with nullptr: releases gfx_pso back to baseline, does not clobber parent's bound PSO.
        CHECK(SUCCEEDED(bundle->Reset(bundle_alloc.p, nullptr)));
        CHECK(rt::d3d12::bound_pipelines(bundle.p, g, c) && g == 0 && c == 0);
        // direct still holds gfx_pso from ExecuteBundle; bundle released its ref, so gfx_pso is base + 1 (direct only).
        CHECK(current_refcount(gfx_pso) == gfx_base_refs + 1);
        CHECK(bundle->Close() == S_OK);
        direct->ExecuteBundle(bundle.p);
        CHECK(rt::d3d12::bound_pipelines(direct.p, g, c) && g == gfx_id && c == 0);

        // Reset bundle with cs_pso: holds cs_pso once, bundle's own pipelines stay 0, parent receives cs_pso.
        CHECK(SUCCEEDED(bundle->Reset(bundle_alloc.p, cs_pso)));
        CHECK(rt::d3d12::bound_pipelines(bundle.p, g, c) && g == 0 && c == 0);
        CHECK(current_refcount(cs_pso) == cs_base_refs + 1);
        CHECK(bundle->Close() == S_OK);
        direct->ExecuteBundle(bundle.p);
        CHECK(rt::d3d12::bound_pipelines(direct.p, g, c) && g == 0 && c == cs_id);
        CHECK(direct->Close() == S_OK);
    }

    // 4. Trace hygiene: CreateCommandList and Reset with initial PSO on DIRECT must not emit synthetic
    //    SetPipelineState trace lines.
    {
        TraceCapture capture(scratch / "cl_reset_trace.txt");
        Ref<ID3D12CommandAllocator> alloc;
        Ref<ID3D12GraphicsCommandList> cl;
        CHECK(SUCCEEDED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&alloc.p))));
        CHECK(SUCCEEDED(
            device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, alloc.p, gfx_pso, IID_PPV_ARGS(&cl.p))));
        std::uintptr_t g = 0, c = 0;
        CHECK(rt::d3d12::bound_pipelines(cl.p, g, c) && g == rt::d3d12::graphics_pipeline_id(gfx_pso) && c == 0);
        CHECK(cl->Close() == S_OK);
        CHECK(SUCCEEDED(cl->Reset(alloc.p, cs_pso)));
        CHECK(rt::d3d12::bound_pipelines(cl.p, g, c) && g == 0 && c == rt::d3d12::compute_pipeline_id(cs_pso));
        CHECK(cl->Close() == S_OK);
        const std::string trace = capture.finish();
        CHECK(count_of(trace, "GraphicsCommandList::SetPipelineState called") == 0);
    }

    return true;
}

bool make_host_psos(ID3D12Device* device, const std::vector<std::uint8_t>& vs, const std::vector<std::uint8_t>& ps,
                    const std::vector<std::uint8_t>& cs, Ref<ID3D12RootSignature>& rs,
                    Ref<ID3D12PipelineState>& gfx_pso, Ref<ID3D12PipelineState>& cs_pso) {
    D3D12_VERSIONED_ROOT_SIGNATURE_DESC rsd{};
    rsd.Version = D3D_ROOT_SIGNATURE_VERSION_1_0;
    const auto blob = serialize_root_signature(rsd);
    CHECK(!blob.empty());
    CHECK(SUCCEEDED(device->CreateRootSignature(0, blob.data(), blob.size(), IID_PPV_ARGS(&rs.p))));

    D3D12_GRAPHICS_PIPELINE_STATE_DESC gpso{};
    gpso.pRootSignature = rs.p;
    gpso.VS = {vs.data(), vs.size()};
    gpso.PS = {ps.data(), ps.size()};
    gpso.BlendState.RenderTarget[0].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
    gpso.SampleMask = 0xffffffff;
    gpso.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
    gpso.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
    gpso.RasterizerState.DepthClipEnable = TRUE;
    gpso.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    gpso.NumRenderTargets = 1;
    gpso.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
    gpso.SampleDesc.Count = 1;
    CHECK(SUCCEEDED(device->CreateGraphicsPipelineState(&gpso, IID_PPV_ARGS(&gfx_pso.p))));

    D3D12_COMPUTE_PIPELINE_STATE_DESC cpso{};
    cpso.pRootSignature = rs.p;
    cpso.CS = {cs.data(), cs.size()};
    CHECK(SUCCEEDED(device->CreateComputePipelineState(&cpso, IID_PPV_ARGS(&cs_pso.p))));
    return true;
}

}  // namespace

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    vktest::Device vk;
    const auto status = vk.initialize();
    if (status == vktest::DeviceStatus::unavailable) {
        std::fprintf(stderr, "SKIP: no usable Vulkan 1.3 device\n");
        return 77;
    }
    REQUIRE(status == vktest::DeviceStatus::ready);

    const auto cs_prog = fssm_program({0x43533001, 0xbf810000});
    const auto vs_prog = fssm_program({0x56533001, 0xbf810000});
    const auto ps_prog = fssm_program({0x50533001, 0xbf810000});
    const auto cs_dxbc = shader_container(cs_prog);
    const auto vs_dxbc = shader_container(vs_prog);
    const auto ps_dxbc = shader_container(ps_prog);

    const testfiles::ScratchDir scratch("leonida_cl_reset_pso_test");
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

    const unsigned errors_before = vk.validation_errors();
    MatrixResults layer_results;
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
        if (rt::d3d12::device_push_constants_need_fallback(ours.p)) {
            std::printf("d3d12_cl_reset_pso_test: skipped, push constants < 144 bytes (%u)\n",
                        rt::d3d12::device_max_push_constants_size(ours.p));
            return 77;
        }

        // Build one graphics PSO and one compute PSO on the layer.
        D3D12_ROOT_PARAMETER gfx_params[2]{};
        gfx_params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
        gfx_params[0].Constants = {0, 0, 2};
        gfx_params[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
        gfx_params[1].Descriptor = {1, 0};
        D3D12_VERSIONED_ROOT_SIGNATURE_DESC gfx_rs_desc{};
        gfx_rs_desc.Version = D3D_ROOT_SIGNATURE_VERSION_1_0;
        gfx_rs_desc.Desc_1_0 = {2, gfx_params, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT};
        const auto gfx_rs_blob = serialize_root_signature(gfx_rs_desc);
        Ref<ID3D12RootSignature> gfx_rs;
        REQUIRE(SUCCEEDED(ours->CreateRootSignature(0, gfx_rs_blob.data(), gfx_rs_blob.size(), IID_PPV_ARGS(&gfx_rs.p))));

        static const D3D12_INPUT_ELEMENT_DESC elements[2] = {
            {"POSITION", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 0, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
            {"COLOR", 0, DXGI_FORMAT_R8G8B8A8_UNORM, 0, D3D12_APPEND_ALIGNED_ELEMENT,
             D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0}};
        D3D12_GRAPHICS_PIPELINE_STATE_DESC gpso_desc{};
        gpso_desc.pRootSignature = gfx_rs.p;
        gpso_desc.VS = {vs_dxbc.data(), vs_dxbc.size()};
        gpso_desc.PS = {ps_dxbc.data(), ps_dxbc.size()};
        gpso_desc.BlendState.RenderTarget[0].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
        gpso_desc.SampleMask = 0xffffffff;
        gpso_desc.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
        gpso_desc.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
        gpso_desc.RasterizerState.DepthClipEnable = TRUE;
        gpso_desc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
        gpso_desc.NumRenderTargets = 1;
        gpso_desc.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
        gpso_desc.SampleDesc.Count = 1;
        gpso_desc.InputLayout = {elements, 2};
        Ref<ID3D12PipelineState> gfx_pso;
        REQUIRE(SUCCEEDED(ours->CreateGraphicsPipelineState(&gpso_desc, IID_PPV_ARGS(&gfx_pso.p))));

        D3D12_DESCRIPTOR_RANGE uav_range{D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 1, 0, 0};
        D3D12_ROOT_PARAMETER cs_params[3]{};
        cs_params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
        cs_params[0].Constants = {0, 0, 1};
        cs_params[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
        cs_params[1].Descriptor = {0, 0};
        cs_params[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        cs_params[2].DescriptorTable = {1, &uav_range};
        D3D12_VERSIONED_ROOT_SIGNATURE_DESC cs_rs_desc{};
        cs_rs_desc.Version = D3D_ROOT_SIGNATURE_VERSION_1_0;
        cs_rs_desc.Desc_1_0 = {3, cs_params, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_NONE};
        const auto cs_rs_blob = serialize_root_signature(cs_rs_desc);
        Ref<ID3D12RootSignature> cs_rs;
        REQUIRE(SUCCEEDED(ours->CreateRootSignature(0, cs_rs_blob.data(), cs_rs_blob.size(), IID_PPV_ARGS(&cs_rs.p))));

        D3D12_COMPUTE_PIPELINE_STATE_DESC cpso_desc{};
        cpso_desc.pRootSignature = cs_rs.p;
        cpso_desc.CS = {cs_dxbc.data(), cs_dxbc.size()};
        Ref<ID3D12PipelineState> cs_pso;
        REQUIRE(SUCCEEDED(ours->CreateComputePipelineState(&cpso_desc, IID_PPV_ARGS(&cs_pso.p))));

        REQUIRE(run_parity_matrix(ours.p, gfx_pso.p, cs_pso.p, layer_results));
        REQUIRE(run_layer_only_checks(ours.p, gfx_pso.p, cs_pso.p, scratch.path()));
    }

    // Verify expected layer HRESULTs directly (even if host is skipped under Wine).
    unsigned failures = 0;
    for (const auto& [name, hr] : layer_results.entries) {
        HRESULT expected = S_OK;
        if (name.find("out=null") != std::string::npos)
            expected = E_UNEXPECTED;
        else if (name.find("Reset(OPEN,") != std::string::npos)
            expected = E_FAIL;
        else if (name.find("null_alloc") != std::string::npos || name.find("COMPUTE_alloc") != std::string::npos ||
                 name.find("busy_alloc") != std::string::npos)
            expected = E_INVALIDARG;
        if (hr != expected) {
            std::fprintf(stderr, "FAIL layer %s: got 0x%08lx, expected 0x%08lx\n", name.c_str(),
                         static_cast<unsigned long>(hr), static_cast<unsigned long>(expected));
            ++failures;
        }
    }

    // Compare against live Windows D3D12 (default adapter + WARP) when not running under Wine.
    unsigned host_comparisons = 0;
    const bool wine = hostwine::host_is_wine();
    std::vector<std::uint8_t> host_vs, host_ps, host_cs;
    const char* hlsl_vs = "float4 main(uint id : SV_VertexID) : SV_Position { return float4(0,0,0,1); }";
    const char* hlsl_ps = "float4 main() : SV_Target { return float4(1,0,0,1); }";
    const char* hlsl_cs = "[numthreads(1,1,1)] void main() {}";
    if (!wine && compile_hlsl(hlsl_vs, "main", "vs_5_0", host_vs) &&
        compile_hlsl(hlsl_ps, "main", "ps_5_0", host_ps) &&
        compile_hlsl(hlsl_cs, "main", "cs_5_0", host_cs)) {
        Ref<IDXGIFactory4> factory;
        REQUIRE(SUCCEEDED(CreateDXGIFactory1(IID_PPV_ARGS(&factory.p))));
        const char* host_names[2] = {"default adapter", "WARP"};
        for (int i = 0; i < 2; ++i) {
            Ref<IDXGIAdapter> adapter;
            if (i == 1 && FAILED(factory->EnumWarpAdapter(IID_PPV_ARGS(&adapter.p)))) continue;
            Ref<ID3D12Device> host;
            if (FAILED(D3D12CreateDevice(adapter.p, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&host.p)))) continue;
            Ref<ID3D12RootSignature> rs;
            Ref<ID3D12PipelineState> gfx_pso, cs_pso;
            REQUIRE(make_host_psos(host.p, host_vs, host_ps, host_cs, rs, gfx_pso, cs_pso));
            MatrixResults host_results;
            REQUIRE(run_parity_matrix(host.p, gfx_pso.p, cs_pso.p, host_results));
            REQUIRE(host_results.entries.size() == layer_results.entries.size());
            for (std::size_t k = 0; k < layer_results.entries.size(); ++k) {
                if (host_results.entries[k].second != layer_results.entries[k].second) {
                    std::fprintf(stderr, "FAIL parity vs %s for %s: layer 0x%08lx vs host 0x%08lx\n", host_names[i],
                                 layer_results.entries[k].first.c_str(),
                                 static_cast<unsigned long>(layer_results.entries[k].second),
                                 static_cast<unsigned long>(host_results.entries[k].second));
                    ++failures;
                }
                ++host_comparisons;
            }
        }
    }

    const unsigned validation_errors = vk.validation_errors() - errors_before;
    if (validation_errors) {
        std::fprintf(stderr, "%u Vulkan validation errors\n", validation_errors);
        ++failures;
    }

    std::printf("d3d12_cl_reset_pso: %zu matrix cases, %u host comparisons, %u failures\n",
                layer_results.entries.size(), host_comparisons, failures);
    return failures ? 1 : 0;
}
