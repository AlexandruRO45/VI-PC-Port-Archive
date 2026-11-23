// End-to-end Vulkan device test for optional SPIR-V extension capability preflight (src/spirv_ext_caps.h) and
// D3D12_FEATURE_D3D12_OPTIONS3::BarycentricsSupported.
//
// Verifies on a real Vulkan device (with zero glslangValidator or game-file dependencies):
//   - plain graphics (VS+PS) and compute (CS) PSOs succeed as positive controls;
//   - graphics and compute PSOs whose SPIR-V declares FragmentBarycentricKHR, FragmentShaderPixelInterlockEXT, or
//     ShaderClockKHR (bare, Subgroup-scoped, or Device-scoped) are refused with E_NOTIMPL before vkCreateShaderModule
//     when the corresponding VulkanDeviceInfo feature flag is false (zero validation errors);
//   - CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS3)::BarycentricsSupported reflects
//     VulkanDeviceInfo::fragment_shader_barycentric on custom and standalone devices.
#include "d3d12_test_shaders.h"
#include "test_temp_dir.h"
#include "vk_test_device.h"
#include "../shims/d3d12_com.h"
#include "../src/shader_library.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <initializer_list>
#include <string>
#include <vector>

#define EXPECT(condition)                                                                       \
    do {                                                                                        \
        if (!(condition)) {                                                                     \
            std::fprintf(stderr, "d3d12_ext_caps_test: line %d: %s\n", __LINE__, #condition);   \
            return 1;                                                                           \
        }                                                                                       \
    } while (0)

using namespace d3d12test;

namespace {

using Words = std::vector<std::uint32_t>;

constexpr std::uint16_t OpExtension = 10, OpMemoryModel = 14, OpEntryPoint = 15, OpExecutionMode = 16,
                        OpCapability = 17, OpTypeVoid = 19, OpTypeInt = 21, OpTypeVector = 23, OpTypeFunction = 33,
                        OpConstant = 43, OpFunction = 54, OpFunctionEnd = 56, OpLabel = 248, OpReturn = 253,
                        OpReadClockKHR = 5056;
constexpr std::uint32_t CapShader = 1, CapShaderClockKHR = 5055, CapFragmentBarycentricKHR = 5284,
                        CapFragmentShaderPixelInterlockEXT = 5378;
constexpr std::uint32_t ExecVertex = 0, ExecFragment = 4, ExecGLCompute = 5;
constexpr std::uint32_t ModeOriginUpperLeft = 7, ModeLocalSize = 17, ModePixelInterlockOrderedEXT = 5366;
constexpr std::uint32_t ScopeDevice = 1, ScopeSubgroup = 3;

void push_inst(Words& out, std::uint16_t op, std::initializer_list<std::uint32_t> operands) {
    out.push_back((static_cast<std::uint32_t>(operands.size() + 1) << 16) | op);
    out.insert(out.end(), operands);
}

void push_string_inst(Words& out, std::uint16_t op, std::initializer_list<std::uint32_t> prefix, const char* text) {
    const std::size_t bytes = std::strlen(text) + 1;
    const std::size_t str_words = (bytes + 3) / 4;
    const std::size_t total = 1 + prefix.size() + str_words;
    out.push_back((static_cast<std::uint32_t>(total) << 16) | op);
    out.insert(out.end(), prefix);
    for (std::size_t w = 0; w < str_words; ++w) {
        std::uint32_t word = 0;
        for (std::size_t b = 0; b < 4; ++b) {
            const std::size_t idx = w * 4 + b;
            if (idx < bytes) word |= static_cast<std::uint32_t>(static_cast<unsigned char>(text[idx])) << (8 * b);
        }
        out.push_back(word);
    }
}

// Builds a minimal, valid SPIR-V 1.5 shader module for `exec_model` with optional extra capability/extension and
// optional OpReadClockKHR instruction (when `clock_scope != 0`).
Words build_minimal_spv(std::uint32_t exec_model, std::uint32_t extra_cap = 0, const char* ext_name = nullptr,
                        std::uint32_t clock_scope = 0) {
    constexpr std::uint32_t void_t = 1, fn_t = 2, main_id = 3, label_id = 4;
    constexpr std::uint32_t uint_t = 5, uvec2_t = 6, scope_c = 7, clock_id = 8;
    Words w = {0x07230203u, 0x00010500u, 0u, 16u, 0u};
    push_inst(w, OpCapability, {CapShader});
    if (extra_cap != 0) push_inst(w, OpCapability, {extra_cap});
    if (ext_name != nullptr) push_string_inst(w, OpExtension, {}, ext_name);
    push_inst(w, OpMemoryModel, {0u, 1u});  // Logical, GLSL450
    push_string_inst(w, OpEntryPoint, {exec_model, main_id}, "main");
    if (exec_model == ExecFragment) {
        push_inst(w, OpExecutionMode, {main_id, ModeOriginUpperLeft});
        if (extra_cap == CapFragmentShaderPixelInterlockEXT) {
            push_inst(w, OpExecutionMode, {main_id, ModePixelInterlockOrderedEXT});
        }
    } else if (exec_model == ExecGLCompute) {
        push_inst(w, OpExecutionMode, {main_id, ModeLocalSize, 1u, 1u, 1u});
    }
    push_inst(w, OpTypeVoid, {void_t});
    push_inst(w, OpTypeFunction, {fn_t, void_t});
    if (clock_scope != 0) {
        push_inst(w, OpTypeInt, {uint_t, 32u, 0u});
        push_inst(w, OpTypeVector, {uvec2_t, uint_t, 2u});
        push_inst(w, OpConstant, {uint_t, scope_c, clock_scope});
    }
    push_inst(w, OpFunction, {void_t, main_id, 0u, fn_t});
    push_inst(w, OpLabel, {label_id});
    if (clock_scope != 0) push_inst(w, OpReadClockKHR, {uvec2_t, clock_id, scope_c});
    push_inst(w, OpReturn, {});
    push_inst(w, OpFunctionEnd, {});
    return w;
}

bool write_spv(const std::filesystem::path& path, const Words& words) {
    std::ofstream out(path, std::ios::binary);
    if (!out) return false;
    out.write(reinterpret_cast<const char*>(words.data()), static_cast<std::streamsize>(words.size() * 4));
    return out.good();
}

D3D12_GRAPHICS_PIPELINE_STATE_DESC make_graphics_desc(ID3D12RootSignature* root,
                                                      const std::vector<std::uint8_t>& vs,
                                                      const std::vector<std::uint8_t>& ps) {
    D3D12_GRAPHICS_PIPELINE_STATE_DESC desc{};
    desc.pRootSignature = root;
    desc.VS = {vs.data(), vs.size()};
    desc.PS = {ps.data(), ps.size()};
    desc.BlendState.RenderTarget[0].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
    desc.SampleMask = UINT_MAX;
    desc.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
    desc.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
    desc.RasterizerState.DepthClipEnable = TRUE;
    desc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    desc.NumRenderTargets = 1;
    desc.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
    desc.SampleDesc.Count = 1;
    return desc;
}

}  // namespace

int main() {
    vktest::Device d;
    const auto status = d.initialize();
    if (status == vktest::DeviceStatus::unavailable) {
        std::fprintf(stderr, "SKIP: no usable Vulkan 1.3 device\n");
        return 77;
    }
    EXPECT(status == vktest::DeviceStatus::ready);

    // 1. Build a synthetic shader library with minimal valid SPIR-V modules (no glslang required).
    const testfiles::ScratchDir scratch("leonida_ext_caps_test");
    const auto& lib = scratch.path();

    struct ShaderEntrySpec {
        const char* name;
        const char* stage;
        std::uint32_t tag;
        Words spv;
        std::vector<std::uint8_t> fssm = {};
        std::vector<std::uint8_t> container = {};
    };

    std::vector<ShaderEntrySpec> specs = {
        {"vs_plain", "vs", 0xE001u, build_minimal_spv(ExecVertex), {}, {}},
        {"ps_plain", "ps", 0xE002u, build_minimal_spv(ExecFragment), {}, {}},
        {"cs_plain", "cs", 0xE003u, build_minimal_spv(ExecGLCompute), {}, {}},
        {"ps_bary", "ps", 0xE004u,
         build_minimal_spv(ExecFragment, CapFragmentBarycentricKHR, "SPV_KHR_fragment_shader_barycentric"), {}, {}},
        {"ps_interlock", "ps", 0xE005u,
         build_minimal_spv(ExecFragment, CapFragmentShaderPixelInterlockEXT, "SPV_EXT_fragment_shader_interlock"), {},
         {}},
        {"ps_clock_sub", "ps", 0xE006u,
         build_minimal_spv(ExecFragment, CapShaderClockKHR, "SPV_KHR_shader_clock", ScopeSubgroup), {}, {}},
        {"ps_clock_dev", "ps", 0xE007u,
         build_minimal_spv(ExecFragment, CapShaderClockKHR, "SPV_KHR_shader_clock", ScopeDevice), {}, {}},
        {"vs_clock", "vs", 0xE008u, build_minimal_spv(ExecVertex, CapShaderClockKHR, "SPV_KHR_shader_clock"), {}, {}},
        {"cs_clock_sub", "cs", 0xE009u,
         build_minimal_spv(ExecGLCompute, CapShaderClockKHR, "SPV_KHR_shader_clock", ScopeSubgroup), {}, {}},
        {"cs_clock_dev", "cs", 0xE00Au,
         build_minimal_spv(ExecGLCompute, CapShaderClockKHR, "SPV_KHR_shader_clock", ScopeDevice), {}, {}},
    };

    {
        std::ofstream index(lib / "index.json", std::ios::binary);
        EXPECT(index.good());
        index << "{";
        for (std::size_t i = 0; i < specs.size(); ++i) {
            auto& s = specs[i];
            s.fssm = fssm_program({s.tag, 0xbf810000u});
            s.container = shader_container(s.fssm);
            EXPECT(write_spv(lib / (std::string(s.name) + ".spv"), s.spv));
            if (i > 0) index << ",\n ";
            index << "\"" << s.name << "\": {\"stage\": \"" << s.stage << "\", \"fssm_sha256\": \""
                  << leonida::sha256_hex(s.fssm) << "\", \"bindings\": [], \"push_constant_bytes\": 144";
            if (std::strcmp(s.stage, "vs") == 0) index << ", \"input_signature\": []";
            if (std::strcmp(s.stage, "ps") == 0) index << ", \"outputs\": [0]";
            index << "}";
        }
        index << "}";
        EXPECT(index.good());
    }

    const auto& vs_plain = specs[0].container;
    const auto& ps_plain = specs[1].container;
    const auto& cs_plain = specs[2].container;
    const auto& ps_bary = specs[3].container;
    const auto& ps_interlock = specs[4].container;
    const auto& ps_clock_sub = specs[5].container;
    const auto& ps_clock_dev = specs[6].container;
    const auto& vs_clock = specs[7].container;
    const auto& cs_clock_sub = specs[8].container;
    const auto& cs_clock_dev = specs[9].container;

    D3D12_VERSIONED_ROOT_SIGNATURE_DESC root_desc{};
    root_desc.Version = D3D_ROOT_SIGNATURE_VERSION_1_0;
    const auto root_blob = serialize_root_signature(root_desc);
    EXPECT(!root_blob.empty());

    // 2. Standalone device: verify OPTIONS3::BarycentricsSupported and live positive/negative PSO outcomes match
    // the GPU's enabled extension features.
    {
        struct ScopedEnv {
            std::string saved;
            bool had = false;
            explicit ScopedEnv(const char* value) {
                if (const DWORD need = GetEnvironmentVariableA("LEONIDA_SHADERS", nullptr, 0); need > 0) {
                    had = true;
                    saved.resize(need);
                    const DWORD got = GetEnvironmentVariableA("LEONIDA_SHADERS", saved.data(), need);
                    saved.resize(got);
                }
                SetEnvironmentVariableA("LEONIDA_SHADERS", value);
            }
            ~ScopedEnv() { SetEnvironmentVariableA("LEONIDA_SHADERS", had ? saved.c_str() : nullptr); }
        };
        const ScopedEnv shaders_env(lib.string().c_str());

        Ref<ID3D12Device> standalone;
        EXPECT(SUCCEEDED(rt::d3d12::create_vulkan_device_standalone(IID_PPV_ARGS(&standalone.p))));

        const auto* made = rt::d3d12::vulkan_device_info(standalone.p);
        EXPECT(made != nullptr);
        D3D12_FEATURE_DATA_D3D12_OPTIONS3 opts3{};
        std::memset(&opts3, 0xAA, sizeof(opts3));
        EXPECT(SUCCEEDED(standalone->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS3, &opts3, sizeof(opts3))));
        EXPECT(opts3.BarycentricsSupported == (made->fragment_shader_barycentric ? TRUE : FALSE));
        std::fprintf(stderr,
                     "d3d12_ext_caps_test: standalone GPU flags: barycentric=%d interlock=%d subgroup_clock=%d "
                     "device_clock=%d OPTIONS3.BarycentricsSupported=%d\n",
                     int(made->fragment_shader_barycentric), int(made->fragment_shader_interlock),
                     int(made->shader_subgroup_clock), int(made->shader_device_clock),
                     int(opts3.BarycentricsSupported));

        Ref<ID3D12RootSignature> root;
        EXPECT(SUCCEEDED(standalone->CreateRootSignature(0, root_blob.data(), root_blob.size(), IID_PPV_ARGS(&root.p))));

        const auto check_gfx = [&](const std::vector<std::uint8_t>& ps, bool supported) -> bool {
            const auto gfx = make_graphics_desc(root.p, vs_plain, ps);
            Ref<ID3D12PipelineState> pso;
            const HRESULT hr = standalone->CreateGraphicsPipelineState(&gfx, IID_PPV_ARGS(&pso.p));
            return supported ? (SUCCEEDED(hr) && pso.p != nullptr) : (hr == E_NOTIMPL && pso.p == nullptr);
        };
        const auto check_cs = [&](const std::vector<std::uint8_t>& cs_bytes, bool supported) -> bool {
            D3D12_COMPUTE_PIPELINE_STATE_DESC cs{};
            cs.pRootSignature = root.p;
            cs.CS = {cs_bytes.data(), cs_bytes.size()};
            Ref<ID3D12PipelineState> pso;
            const HRESULT hr = standalone->CreateComputePipelineState(&cs, IID_PPV_ARGS(&pso.p));
            return supported ? (SUCCEEDED(hr) && pso.p != nullptr) : (hr == E_NOTIMPL && pso.p == nullptr);
        };

        EXPECT(check_gfx(ps_bary, made->fragment_shader_barycentric));
        EXPECT(check_gfx(ps_interlock, made->fragment_shader_interlock));
        EXPECT(check_gfx(ps_clock_sub, made->shader_subgroup_clock));
        EXPECT(check_gfx(ps_clock_dev, made->shader_device_clock));
        EXPECT(check_cs(cs_clock_sub, made->shader_subgroup_clock));
        EXPECT(check_cs(cs_clock_dev, made->shader_device_clock));
    }

    rt::d3d12::VulkanDeviceInfo base_info{};
    base_info.get_instance_proc = d.instance_proc;
    base_info.instance = d.instance;
    base_info.physical = d.physical;
    base_info.device = d.device;
    base_info.queue = d.queue;
    base_info.queue_family = d.family;
    base_info.buffer_device_address = true;
    base_info.subgroup = d.subgroup;
    base_info.shader_dir = lib.string();

    // 3. All 3 extension flags false (matches Pascal / GTX 1070 for barycentric/interlock and any GPU without clock).
    {
        Ref<ID3D12Device> device;
        EXPECT(SUCCEEDED(rt::d3d12::create_vulkan_device(base_info, IID_PPV_ARGS(&device.p))));
        D3D12_FEATURE_DATA_D3D12_OPTIONS3 opts3{};
        EXPECT(SUCCEEDED(device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS3, &opts3, sizeof(opts3))));
        EXPECT(opts3.BarycentricsSupported == FALSE);

        Ref<ID3D12RootSignature> root;
        EXPECT(SUCCEEDED(device->CreateRootSignature(0, root_blob.data(), root_blob.size(), IID_PPV_ARGS(&root.p))));

        // Positive controls: plain VS+PS graphics PSO and plain CS compute PSO succeed.
        {
            const auto gfx = make_graphics_desc(root.p, vs_plain, ps_plain);
            Ref<ID3D12PipelineState> pso;
            EXPECT(SUCCEEDED(device->CreateGraphicsPipelineState(&gfx, IID_PPV_ARGS(&pso.p))) && pso.p != nullptr);
        }
        {
            D3D12_COMPUTE_PIPELINE_STATE_DESC cs{};
            cs.pRootSignature = root.p;
            cs.CS = {cs_plain.data(), cs_plain.size()};
            Ref<ID3D12PipelineState> pso;
            EXPECT(SUCCEEDED(device->CreateComputePipelineState(&cs, IID_PPV_ARGS(&pso.p))) && pso.p != nullptr);
        }

        // Preflight refusals: missing extension capabilities return E_NOTIMPL before vkCreateShaderModule.
        for (const auto* ps : {&ps_bary, &ps_interlock, &ps_clock_sub, &ps_clock_dev}) {
            const auto gfx = make_graphics_desc(root.p, vs_plain, *ps);
            Ref<ID3D12PipelineState> refused;
            EXPECT(device->CreateGraphicsPipelineState(&gfx, IID_PPV_ARGS(&refused.p)) == E_NOTIMPL && !refused.p);
        }
        {
            const auto gfx = make_graphics_desc(root.p, vs_clock, ps_plain);
            Ref<ID3D12PipelineState> refused;
            EXPECT(device->CreateGraphicsPipelineState(&gfx, IID_PPV_ARGS(&refused.p)) == E_NOTIMPL && !refused.p);
        }
        for (const auto* cs_bytes : {&cs_clock_sub, &cs_clock_dev}) {
            D3D12_COMPUTE_PIPELINE_STATE_DESC cs{};
            cs.pRootSignature = root.p;
            cs.CS = {cs_bytes->data(), cs_bytes->size()};
            Ref<ID3D12PipelineState> refused;
            EXPECT(device->CreateComputePipelineState(&cs, IID_PPV_ARGS(&refused.p)) == E_NOTIMPL && !refused.p);
        }
    }

    // 4. Toggled OPTIONS3::BarycentricsSupported = TRUE when fragment_shader_barycentric is enabled in VulkanDeviceInfo.
    {
        rt::d3d12::VulkanDeviceInfo bary_info = base_info;
        bary_info.fragment_shader_barycentric = true;
        Ref<ID3D12Device> device;
        EXPECT(SUCCEEDED(rt::d3d12::create_vulkan_device(bary_info, IID_PPV_ARGS(&device.p))));
        D3D12_FEATURE_DATA_D3D12_OPTIONS3 opts3{};
        EXPECT(SUCCEEDED(device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS3, &opts3, sizeof(opts3))));
        EXPECT(opts3.BarycentricsSupported == TRUE);
    }

    // 5. Scope-specific ShaderClockKHR refusals: subgroup-only rejects device-scoped clock, and vice versa.
    {
        rt::d3d12::VulkanDeviceInfo sub_only = base_info;
        sub_only.shader_subgroup_clock = true;
        sub_only.shader_device_clock = false;
        Ref<ID3D12Device> device;
        EXPECT(SUCCEEDED(rt::d3d12::create_vulkan_device(sub_only, IID_PPV_ARGS(&device.p))));
        Ref<ID3D12RootSignature> root;
        EXPECT(SUCCEEDED(device->CreateRootSignature(0, root_blob.data(), root_blob.size(), IID_PPV_ARGS(&root.p))));

        const auto gfx = make_graphics_desc(root.p, vs_plain, ps_clock_dev);
        Ref<ID3D12PipelineState> refused_gfx;
        EXPECT(device->CreateGraphicsPipelineState(&gfx, IID_PPV_ARGS(&refused_gfx.p)) == E_NOTIMPL && !refused_gfx.p);

        D3D12_COMPUTE_PIPELINE_STATE_DESC cs{};
        cs.pRootSignature = root.p;
        cs.CS = {cs_clock_dev.data(), cs_clock_dev.size()};
        Ref<ID3D12PipelineState> refused_cs;
        EXPECT(device->CreateComputePipelineState(&cs, IID_PPV_ARGS(&refused_cs.p)) == E_NOTIMPL && !refused_cs.p);
    }
    {
        rt::d3d12::VulkanDeviceInfo dev_only = base_info;
        dev_only.shader_subgroup_clock = false;
        dev_only.shader_device_clock = true;
        Ref<ID3D12Device> device;
        EXPECT(SUCCEEDED(rt::d3d12::create_vulkan_device(dev_only, IID_PPV_ARGS(&device.p))));
        Ref<ID3D12RootSignature> root;
        EXPECT(SUCCEEDED(device->CreateRootSignature(0, root_blob.data(), root_blob.size(), IID_PPV_ARGS(&root.p))));

        const auto gfx = make_graphics_desc(root.p, vs_plain, ps_clock_sub);
        Ref<ID3D12PipelineState> refused_gfx;
        EXPECT(device->CreateGraphicsPipelineState(&gfx, IID_PPV_ARGS(&refused_gfx.p)) == E_NOTIMPL && !refused_gfx.p);

        D3D12_COMPUTE_PIPELINE_STATE_DESC cs{};
        cs.pRootSignature = root.p;
        cs.CS = {cs_clock_sub.data(), cs_clock_sub.size()};
        Ref<ID3D12PipelineState> refused_cs;
        EXPECT(device->CreateComputePipelineState(&cs, IID_PPV_ARGS(&refused_cs.p)) == E_NOTIMPL && !refused_cs.p);
    }

    std::fprintf(stderr, "d3d12_ext_caps_test: OK\n");
    return 0;
}
