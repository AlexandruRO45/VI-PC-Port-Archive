// VK_KHR_shader_fma (OpFmaKHR / shaderFmaFloat32) capability guard in ID3D12Device::CreateComputePipelineState
// and CreateGraphicsPipelineState:
//   1. Caller-owned device with shader_fma = false: refuses compute and graphics (VS/PS) shaders requiring
//      VK_KHR_shader_fma via index.json `features`, `capabilities`, or unindexed SPIR-V OpCapability FMAKHR with
//      E_NOTIMPL, *out == nullptr, and zero validation errors, while accepting plain shaders (S_OK).
//   2. Standalone runtime device (create_vulkan_device_standalone): verifies FMAKHR compute/graphics shaders (both
//      indexed and unindexed SPIR-V) are refused with E_NOTIMPL when !info->shader_fma (e.g. NVIDIA Pascal / GTX 1070)
//      or created when supported, while plain shaders always succeed with zero validation errors.
// Minimal hand-assembled SPIR-V modules (validated with spirv-val); nothing is drawn. No game data.
#include "d3d12_test_shaders.h"
#include "test_temp_dir.h"
#include "vk_test_device.h"
#include "../shims/d3d12_com.h"
#include "../src/shader_library.h"

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <span>
#include <string>
#include <vector>

#define EXPECT(condition) do {                                                        \
    if (!(condition)) {                                                               \
        std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition);                  \
        return 1;                                                                     \
    }                                                                                 \
} while (0)

using namespace d3d12test;

namespace {

std::atomic<unsigned> g_standalone_errors{0};

VKAPI_ATTR VkBool32 VKAPI_CALL on_standalone_message(VkDebugUtilsMessageSeverityFlagBitsEXT severity,
                                                     VkDebugUtilsMessageTypeFlagsEXT,
                                                     const VkDebugUtilsMessengerCallbackDataEXT* data, void*) {
    if (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) {
        ++g_standalone_errors;
        std::fprintf(stderr, "validation: %s\n", data && data->pMessage ? data->pMessage : "(no message)");
    }
    return VK_FALSE;
}

bool validation_layer_installed() {
    const HMODULE loader = LoadLibraryA("vulkan-1.dll");
    const auto get = loader ? reinterpret_cast<PFN_vkGetInstanceProcAddr>(
                                  reinterpret_cast<void*>(GetProcAddress(loader, "vkGetInstanceProcAddr")))
                            : nullptr;
    const auto enumerate = get ? reinterpret_cast<PFN_vkEnumerateInstanceLayerProperties>(
                                     get(nullptr, "vkEnumerateInstanceLayerProperties"))
                               : nullptr;
    if (!enumerate) return false;
    std::uint32_t count = 0;
    enumerate(&count, nullptr);
    std::vector<VkLayerProperties> layers(count);
    enumerate(&count, layers.data());
    for (std::uint32_t i = 0; i < count && i < layers.size(); ++i)
        if (!std::strcmp(layers[i].layerName, "VK_LAYER_KHRONOS_validation")) return true;
    return false;
}

// Minimal Vulkan 1.2 SPIR-V modules (assembled with spirv-as --target-env vulkan1.2 and verified with spirv-val).
constexpr std::uint32_t kCsPlain[] = {
    0x07230203u, 0x00010500u, 0x00070000u, 0x00000005u, 0x00000000u, 0x00020011u, 0x00000001u, 0x0003000eu,
    0x00000000u, 0x00000001u, 0x0005000fu, 0x00000005u, 0x00000001u, 0x6e69616du, 0x00000000u, 0x00060010u,
    0x00000001u, 0x00000011u, 0x00000001u, 0x00000001u, 0x00000001u, 0x00020013u, 0x00000002u, 0x00030021u,
    0x00000003u, 0x00000002u, 0x00050036u, 0x00000002u, 0x00000001u, 0x00000000u, 0x00000003u, 0x000200f8u,
    0x00000004u, 0x000100fdu, 0x00010038u,
};

constexpr std::uint32_t kCsFma[] = {
    0x07230203u, 0x00010500u, 0x00070000u, 0x00000005u, 0x00000000u, 0x00020011u, 0x00000001u, 0x00020011u,
    0x0000178eu, 0x0004000au, 0x5f565053u, 0x5f52484bu, 0x00616d66u, 0x0003000eu, 0x00000000u, 0x00000001u,
    0x0005000fu, 0x00000005u, 0x00000001u, 0x6e69616du, 0x00000000u, 0x00060010u, 0x00000001u, 0x00000011u,
    0x00000001u, 0x00000001u, 0x00000001u, 0x00020013u, 0x00000002u, 0x00030021u, 0x00000003u, 0x00000002u,
    0x00050036u, 0x00000002u, 0x00000001u, 0x00000000u, 0x00000003u, 0x000200f8u, 0x00000004u, 0x000100fdu,
    0x00010038u,
};

// Hand-written compute module with a double-precision OpFmaKHR of three constants.
constexpr std::uint32_t kCsFma64[] = {
    0x07230203u, 0x00010500u, 0x00070000u, 0x00000008u, 0x00000000u, 0x00020011u, 0x00000001u, 0x00020011u,
    0x0000000au, 0x00020011u, 0x0000178eu, 0x0004000au, 0x5f565053u, 0x5f52484bu, 0x00616d66u, 0x0003000eu,
    0x00000000u, 0x00000001u, 0x0005000fu, 0x00000005u, 0x00000001u, 0x6e69616du, 0x00000000u, 0x00060010u,
    0x00000001u, 0x00000011u, 0x00000001u, 0x00000001u, 0x00000001u, 0x00020013u, 0x00000002u, 0x00030021u,
    0x00000003u, 0x00000002u, 0x00030016u, 0x00000004u, 0x00000040u, 0x0005002bu, 0x00000004u, 0x00000005u,
    0x00000000u, 0x3ff00000u, 0x00050036u, 0x00000002u, 0x00000001u, 0x00000000u, 0x00000003u, 0x000200f8u,
    0x00000006u, 0x0006114bu, 0x00000004u, 0x00000007u, 0x00000005u, 0x00000005u, 0x00000005u, 0x000100fdu,
    0x00010038u,
};

constexpr std::uint32_t kVsPlain[] = {
    0x07230203u, 0x00010500u, 0x00070000u, 0x0000000cu, 0x00000000u, 0x00020011u, 0x00000001u, 0x0003000eu,
    0x00000000u, 0x00000001u, 0x0006000fu, 0x00000000u, 0x00000001u, 0x6e69616du, 0x00000000u, 0x00000002u,
    0x00040047u, 0x00000002u, 0x0000000bu, 0x00000000u, 0x00020013u, 0x00000003u, 0x00030021u, 0x00000004u,
    0x00000003u, 0x00030016u, 0x00000005u, 0x00000020u, 0x00040017u, 0x00000006u, 0x00000005u, 0x00000004u,
    0x00040020u, 0x00000007u, 0x00000003u, 0x00000006u, 0x0004003bu, 0x00000007u, 0x00000002u, 0x00000003u,
    0x0004002bu, 0x00000005u, 0x00000008u, 0x00000000u, 0x0004002bu, 0x00000005u, 0x00000009u, 0x3f800000u,
    0x0007002cu, 0x00000006u, 0x0000000au, 0x00000008u, 0x00000008u, 0x00000008u, 0x00000009u, 0x00050036u,
    0x00000003u, 0x00000001u, 0x00000000u, 0x00000004u, 0x000200f8u, 0x0000000bu, 0x0003003eu, 0x00000002u,
    0x0000000au, 0x000100fdu, 0x00010038u,
};

constexpr std::uint32_t kVsFma[] = {
    0x07230203u, 0x00010500u, 0x00070000u, 0x0000000cu, 0x00000000u, 0x00020011u, 0x00000001u, 0x00020011u,
    0x0000178eu, 0x0004000au, 0x5f565053u, 0x5f52484bu, 0x00616d66u, 0x0003000eu, 0x00000000u, 0x00000001u,
    0x0006000fu, 0x00000000u, 0x00000001u, 0x6e69616du, 0x00000000u, 0x00000002u, 0x00040047u, 0x00000002u,
    0x0000000bu, 0x00000000u, 0x00020013u, 0x00000003u, 0x00030021u, 0x00000004u, 0x00000003u, 0x00030016u,
    0x00000005u, 0x00000020u, 0x00040017u, 0x00000006u, 0x00000005u, 0x00000004u, 0x00040020u, 0x00000007u,
    0x00000003u, 0x00000006u, 0x0004003bu, 0x00000007u, 0x00000002u, 0x00000003u, 0x0004002bu, 0x00000005u,
    0x00000008u, 0x00000000u, 0x0004002bu, 0x00000005u, 0x00000009u, 0x3f800000u, 0x0007002cu, 0x00000006u,
    0x0000000au, 0x00000008u, 0x00000008u, 0x00000008u, 0x00000009u, 0x00050036u, 0x00000003u, 0x00000001u,
    0x00000000u, 0x00000004u, 0x000200f8u, 0x0000000bu, 0x0003003eu, 0x00000002u, 0x0000000au, 0x000100fdu,
    0x00010038u,
};

constexpr std::uint32_t kPsPlain[] = {
    0x07230203u, 0x00010500u, 0x00070000u, 0x0000000cu, 0x00000000u, 0x00020011u, 0x00000001u, 0x0003000eu,
    0x00000000u, 0x00000001u, 0x0006000fu, 0x00000004u, 0x00000001u, 0x6e69616du, 0x00000000u, 0x00000002u,
    0x00030010u, 0x00000001u, 0x00000007u, 0x00040047u, 0x00000002u, 0x0000001eu, 0x00000000u, 0x00020013u,
    0x00000003u, 0x00030021u, 0x00000004u, 0x00000003u, 0x00030016u, 0x00000005u, 0x00000020u, 0x00040017u,
    0x00000006u, 0x00000005u, 0x00000004u, 0x00040020u, 0x00000007u, 0x00000003u, 0x00000006u, 0x0004003bu,
    0x00000007u, 0x00000002u, 0x00000003u, 0x0004002bu, 0x00000005u, 0x00000008u, 0x00000000u, 0x0004002bu,
    0x00000005u, 0x00000009u, 0x3f800000u, 0x0007002cu, 0x00000006u, 0x0000000au, 0x00000008u, 0x00000008u,
    0x00000008u, 0x00000009u, 0x00050036u, 0x00000003u, 0x00000001u, 0x00000000u, 0x00000004u, 0x000200f8u,
    0x0000000bu, 0x0003003eu, 0x00000002u, 0x0000000au, 0x000100fdu, 0x00010038u,
};

constexpr std::uint32_t kPsFma[] = {
    0x07230203u, 0x00010500u, 0x00070000u, 0x0000000cu, 0x00000000u, 0x00020011u, 0x00000001u, 0x00020011u,
    0x0000178eu, 0x0004000au, 0x5f565053u, 0x5f52484bu, 0x00616d66u, 0x0003000eu, 0x00000000u, 0x00000001u,
    0x0006000fu, 0x00000004u, 0x00000001u, 0x6e69616du, 0x00000000u, 0x00000002u, 0x00030010u, 0x00000001u,
    0x00000007u, 0x00040047u, 0x00000002u, 0x0000001eu, 0x00000000u, 0x00020013u, 0x00000003u, 0x00030021u,
    0x00000004u, 0x00000003u, 0x00030016u, 0x00000005u, 0x00000020u, 0x00040017u, 0x00000006u, 0x00000005u,
    0x00000004u, 0x00040020u, 0x00000007u, 0x00000003u, 0x00000006u, 0x0004003bu, 0x00000007u, 0x00000002u,
    0x00000003u, 0x0004002bu, 0x00000005u, 0x00000008u, 0x00000000u, 0x0004002bu, 0x00000005u, 0x00000009u,
    0x3f800000u, 0x0007002cu, 0x00000006u, 0x0000000au, 0x00000008u, 0x00000008u, 0x00000008u, 0x00000009u,
    0x00050036u, 0x00000003u, 0x00000001u, 0x00000000u, 0x00000004u, 0x000200f8u, 0x0000000bu, 0x0003003eu,
    0x00000002u, 0x0000000au, 0x000100fdu, 0x00010038u,
};

bool write_spv(const std::filesystem::path& path, std::span<const std::uint32_t> words) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out.write(reinterpret_cast<const char*>(words.data()), std::streamsize(words.size() * sizeof(std::uint32_t)));
    return bool(out);
}

D3D12_COMPUTE_PIPELINE_STATE_DESC compute_desc(ID3D12RootSignature* root, const std::vector<std::uint8_t>& cs) {
    D3D12_COMPUTE_PIPELINE_STATE_DESC desc{};
    desc.pRootSignature = root;
    desc.CS = {cs.data(), cs.size()};
    return desc;
}

D3D12_GRAPHICS_PIPELINE_STATE_DESC graphics_desc(ID3D12RootSignature* root, const std::vector<std::uint8_t>& vs,
                                                 const std::vector<std::uint8_t>& ps) {
    D3D12_GRAPHICS_PIPELINE_STATE_DESC desc{};
    desc.pRootSignature = root;
    desc.VS = {vs.data(), vs.size()};
    desc.PS = {ps.data(), ps.size()};
    desc.BlendState.RenderTarget[0].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
    desc.SampleMask = 0xffffffffu;
    desc.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
    desc.RasterizerState.CullMode = D3D12_CULL_MODE_BACK;
    desc.RasterizerState.DepthClipEnable = TRUE;
    desc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    desc.NumRenderTargets = 1;
    desc.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
    desc.SampleDesc.Count = 1;
    return desc;
}

HRESULT make_root(ID3D12Device* device, ID3D12RootSignature** root) {
    D3D12_VERSIONED_ROOT_SIGNATURE_DESC desc{};
    desc.Version = D3D_ROOT_SIGNATURE_VERSION_1_0;
    desc.Desc_1_0 = {0, nullptr, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_NONE};
    const auto blob = serialize_root_signature(desc);
    if (blob.empty()) return E_FAIL;
    return device->CreateRootSignature(0, blob.data(), blob.size(), IID_PPV_ARGS(root));
}

}  // namespace

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
#pragma warning(push)
#pragma warning(disable : 4996)
    const char* const requested = std::getenv("LEONIDA_VK_VALIDATION");
#pragma warning(pop)
    const bool validating = requested ? requested[0] && requested[0] != '0' : validation_layer_installed();
    if (validating && !requested) _putenv_s("LEONIDA_VK_VALIDATION", "1");

    // Synthetic shader library covering FMA via index `features`, `capabilities`, and unindexed SPIR-V OpCapability.
    const auto p_cs_plain = fssm_program({0x43530001u, 0xbf810000u});
    const auto p_cs_fma_feat = fssm_program({0x43530002u, 0xbf810000u});
    const auto p_cs_fma_cap = fssm_program({0x43530003u, 0xbf810000u});
    const auto p_cs_fma_spv = fssm_program({0x43530004u, 0xbf810000u});
    const auto p_cs_fma64 = fssm_program({0x43530005u, 0xbf810000u});
    const auto c_cs_fma64 = shader_container(p_cs_fma64);
    const auto p_vs_plain = fssm_program({0x56530001u, 0xbf810000u});
    const auto p_vs_fma = fssm_program({0x56530002u, 0xbf810000u});
    const auto p_ps_plain = fssm_program({0x50530001u, 0xbf810000u});
    const auto p_ps_fma = fssm_program({0x50530002u, 0xbf810000u});

    const auto c_cs_plain = shader_container(p_cs_plain);
    const auto c_cs_fma_feat = shader_container(p_cs_fma_feat);
    const auto c_cs_fma_cap = shader_container(p_cs_fma_cap);
    const auto c_cs_fma_spv = shader_container(p_cs_fma_spv);
    const auto c_vs_plain = shader_container(p_vs_plain);
    const auto c_vs_fma = shader_container(p_vs_fma);
    const auto c_ps_plain = shader_container(p_ps_plain);
    const auto c_ps_fma = shader_container(p_ps_fma);

    const testfiles::ScratchDir scratch("leonida_shader_fma_guard_test");
    const auto& library = scratch.path();
    std::filesystem::create_directories(library);
    EXPECT(write_spv(library / "cs_plain.spv", kCsPlain));
    EXPECT(write_spv(library / "cs_fma_feat.spv", kCsFma));
    EXPECT(write_spv(library / "cs_fma_cap.spv", kCsFma));
    EXPECT(write_spv(library / "cs_fma_spv.spv", kCsFma));
    EXPECT(write_spv(library / "cs_fma64.spv", kCsFma64));
    EXPECT(write_spv(library / "vs_plain.spv", kVsPlain));
    EXPECT(write_spv(library / "vs_fma.spv", kVsFma));
    EXPECT(write_spv(library / "ps_plain.spv", kPsPlain));
    EXPECT(write_spv(library / "ps_fma.spv", kPsFma));

    {
        std::ofstream index(library / "index.json", std::ios::binary | std::ios::trunc);
        index << "{\n"
              << "  \"cs_plain\": {\"stage\": \"cs\", \"fssm_sha256\": \"" << leonida::sha256_hex(p_cs_plain)
              << "\", \"bindings\": [], \"push_constant_bytes\": 144, \"local_size\": [1, 1, 1]},\n"
              << "  \"cs_fma_feat\": {\"stage\": \"cs\", \"fssm_sha256\": \"" << leonida::sha256_hex(p_cs_fma_feat)
              << "\", \"fma\": \"khr\", \"features\": [\"shaderFmaFloat32\"], \"bindings\": [], \"local_size\": [1, 1, 1]},\n"
              << "  \"cs_fma_cap\": {\"stage\": \"cs\", \"fssm_sha256\": \"" << leonida::sha256_hex(p_cs_fma_cap)
              << "\", \"capabilities\": [\"Shader\", \"FMAKHR\"], \"bindings\": [], \"local_size\": [1, 1, 1]},\n"
              << "  \"cs_fma_spv\": {\"stage\": \"cs\", \"fssm_sha256\": \"" << leonida::sha256_hex(p_cs_fma_spv)
              << "\", \"bindings\": [], \"local_size\": [1, 1, 1]},\n"
              << "  \"cs_fma64\": {\"stage\": \"cs\", \"fssm_sha256\": \"" << leonida::sha256_hex(p_cs_fma64)
              << "\", \"capabilities\": [\"FMAKHR\"], \"bindings\": [], \"local_size\": [1, 1, 1]},\n"
              << "  \"vs_plain\": {\"stage\": \"vs\", \"fssm_sha256\": \"" << leonida::sha256_hex(p_vs_plain)
              << "\", \"bindings\": [], \"input_signature\": []},\n"
              << "  \"vs_fma\": {\"stage\": \"vs\", \"fssm_sha256\": \"" << leonida::sha256_hex(p_vs_fma)
              << "\", \"features\": [\"shaderFmaFloat32\"], \"capabilities\": [\"FMAKHR\"], \"bindings\": [], \"input_signature\": []},\n"
              << "  \"ps_plain\": {\"stage\": \"ps\", \"fssm_sha256\": \"" << leonida::sha256_hex(p_ps_plain)
              << "\", \"bindings\": [], \"outputs\": [0]},\n"
              << "  \"ps_fma\": {\"stage\": \"ps\", \"fssm_sha256\": \"" << leonida::sha256_hex(p_ps_fma)
              << "\", \"features\": [\"shaderFmaFloat32\"], \"capabilities\": [\"FMAKHR\"], \"bindings\": [], \"outputs\": [0]}\n"
              << "}\n";
        EXPECT(bool(index));
    }

    int ran = 0;

    // 1. Caller-owned device without shader_fma: refuses FMAKHR compute/VS/PS shaders (metadata & SPIR-V scan).
    {
        vktest::Device d;
        const auto status = d.initialize();
        if (status == vktest::DeviceStatus::unavailable) {
            std::fprintf(stderr, "SKIP: no usable Vulkan 1.3 device\n");
            return 77;
        }
        EXPECT(status == vktest::DeviceStatus::ready);

        rt::d3d12::VulkanDeviceInfo info{};
        info.get_instance_proc = d.instance_proc;
        info.instance = d.instance;
        info.physical = d.physical;
        info.device = d.device;
        info.queue = d.queue;
        info.queue_family = d.family;
        info.buffer_device_address = true;
        info.bindless = d.bindless;
        info.bindless_partially_bound = d.partially_bound;
        info.subgroup = d.subgroup;
        info.shader_dir = library.string();
        info.features = d.enabled;
        info.shader_fma = false;  // explicitly off, simulating GPUs without VK_KHR_shader_fma

        Ref<ID3D12Device> device;
        EXPECT(SUCCEEDED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&device.p))));
        Ref<ID3D12RootSignature> root;
        EXPECT(SUCCEEDED(make_root(device.p, &root.p)));

        for (const auto* fma_cs : {&c_cs_fma_feat, &c_cs_fma_cap, &c_cs_fma_spv}) {
            const auto desc = compute_desc(root.p, *fma_cs);
            ID3D12PipelineState* refused = reinterpret_cast<ID3D12PipelineState*>(std::uintptr_t{1});
            EXPECT(device->CreateComputePipelineState(&desc, IID_PPV_ARGS(&refused)) == E_NOTIMPL && !refused);
        }
        {
            const auto desc = graphics_desc(root.p, c_vs_fma, c_ps_plain);
            ID3D12PipelineState* refused = reinterpret_cast<ID3D12PipelineState*>(std::uintptr_t{1});
            EXPECT(device->CreateGraphicsPipelineState(&desc, IID_PPV_ARGS(&refused)) == E_NOTIMPL && !refused);
        }
        {
            const auto desc = graphics_desc(root.p, c_vs_plain, c_ps_fma);
            ID3D12PipelineState* refused = reinterpret_cast<ID3D12PipelineState*>(std::uintptr_t{1});
            EXPECT(device->CreateGraphicsPipelineState(&desc, IID_PPV_ARGS(&refused)) == E_NOTIMPL && !refused);
        }
        // Control: plain compute and graphics pipelines still succeed.
        {
            const auto cs = compute_desc(root.p, c_cs_plain);
            Ref<ID3D12PipelineState> ok_cs;
            EXPECT(SUCCEEDED(device->CreateComputePipelineState(&cs, IID_PPV_ARGS(&ok_cs.p))) && ok_cs.p);
            const auto gfx = graphics_desc(root.p, c_vs_plain, c_ps_plain);
            Ref<ID3D12PipelineState> ok_gfx;
            EXPECT(SUCCEEDED(device->CreateGraphicsPipelineState(&gfx, IID_PPV_ARGS(&ok_gfx.p))) && ok_gfx.p);
        }

        root.reset();
        device.reset();
        if (d.shader_fma) {
            // Expose only shaderFmaFloat32 through the caller-owned device contract.
            info.shader_fma = true;
            EXPECT(SUCCEEDED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&device.p))));
            EXPECT(SUCCEEDED(make_root(device.p, &root.p)));
            const auto cs = compute_desc(root.p, c_cs_fma64);
            ID3D12PipelineState* refused = reinterpret_cast<ID3D12PipelineState*>(std::uintptr_t{1});
            EXPECT(device->CreateComputePipelineState(&cs, IID_PPV_ARGS(&refused)) == E_NOTIMPL && !refused);
            root.reset();
            device.reset();
            std::printf("caller-owned f32-only contract: double OpFmaKHR refused\n");
        }
        EXPECT(d.validation_errors() == 0);
        std::printf("shader_fma=false device: FMAKHR CS/VS/PS refused (E_NOTIMPL), plain CS/VS+PS created%s\n",
                    validating ? "; validation errors: 0" : "");
        ++ran;
    }

    // 2. Standalone runtime device: FMAKHR PSO creation succeeds iff info->shader_fma, else E_NOTIMPL cleanly with
    //    zero validation errors, while plain pipelines always succeed.
    {
        _putenv_s("LEONIDA_SHADERS", library.string().c_str());
        Ref<ID3D12Device> device;
        EXPECT(SUCCEEDED(rt::d3d12::create_vulkan_device_standalone(IID_PPV_ARGS(&device.p))));
        const rt::d3d12::VulkanDeviceInfo* info = rt::d3d12::vulkan_device_info(device.p);
        EXPECT(info != nullptr);

        VkDebugUtilsMessengerEXT messenger = VK_NULL_HANDLE;
        const auto create_messenger = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(
            info->get_instance_proc(info->instance, "vkCreateDebugUtilsMessengerEXT"));
        const auto destroy_messenger = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(
            info->get_instance_proc(info->instance, "vkDestroyDebugUtilsMessengerEXT"));
        const bool watching = validating && info->debug_utils && create_messenger && destroy_messenger;
        if (watching) {
            VkDebugUtilsMessengerCreateInfoEXT messenger_info{};
            messenger_info.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
            messenger_info.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
            messenger_info.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT;
            messenger_info.pfnUserCallback = on_standalone_message;
            EXPECT(create_messenger(info->instance, &messenger_info, nullptr, &messenger) == VK_SUCCESS);
        }

        Ref<ID3D12RootSignature> root;
        EXPECT(SUCCEEDED(make_root(device.p, &root.p)));

        for (const auto* fma_cs : {&c_cs_fma_feat, &c_cs_fma_cap, &c_cs_fma_spv}) {
            const auto cs = compute_desc(root.p, *fma_cs);
            ID3D12PipelineState* raw = reinterpret_cast<ID3D12PipelineState*>(std::uintptr_t{1});
            const HRESULT hr = device->CreateComputePipelineState(&cs, IID_PPV_ARGS(&raw));
            if (info->shader_fma) {
                EXPECT(hr == S_OK && raw != nullptr);
                raw->Release();
            } else {
                EXPECT(hr == E_NOTIMPL && raw == nullptr);
            }
        }
        {
            const auto cs = compute_desc(root.p, c_cs_fma64);
            ID3D12PipelineState* raw = reinterpret_cast<ID3D12PipelineState*>(std::uintptr_t{1});
            const HRESULT hr = device->CreateComputePipelineState(&cs, IID_PPV_ARGS(&raw));
            if (info->shader_fma_float64) {
                EXPECT(hr == S_OK && raw != nullptr);
                raw->Release();
            } else {
                EXPECT(hr == E_NOTIMPL && raw == nullptr);
            }
            std::printf("standalone double OpFmaKHR pipeline: shaderFmaFloat64=%d\n", info->shader_fma_float64);
        }
        {
            const auto gfx = graphics_desc(root.p, c_vs_plain, c_ps_fma);
            ID3D12PipelineState* raw = reinterpret_cast<ID3D12PipelineState*>(std::uintptr_t{1});
            const HRESULT hr = device->CreateGraphicsPipelineState(&gfx, IID_PPV_ARGS(&raw));
            if (info->shader_fma) {
                EXPECT(hr == S_OK && raw != nullptr);
                raw->Release();
            } else {
                EXPECT(hr == E_NOTIMPL && raw == nullptr);
            }
        }
        {
            const auto cs = compute_desc(root.p, c_cs_plain);
            Ref<ID3D12PipelineState> ok_cs;
            EXPECT(SUCCEEDED(device->CreateComputePipelineState(&cs, IID_PPV_ARGS(&ok_cs.p))) && ok_cs.p);
            const auto gfx = graphics_desc(root.p, c_vs_plain, c_ps_plain);
            Ref<ID3D12PipelineState> ok_gfx;
            EXPECT(SUCCEEDED(device->CreateGraphicsPipelineState(&gfx, IID_PPV_ARGS(&ok_gfx.p))) && ok_gfx.p);
        }

        const bool has_fma = info->shader_fma;
        root.reset();
        const unsigned errors = g_standalone_errors.load();
        if (messenger) destroy_messenger(info->instance, messenger, nullptr);
        device.reset();
        EXPECT(errors == 0);
        std::printf("standalone device: shader_fma=%d%s\n", has_fma ? 1 : 0,
                    watching ? "; validation errors: 0" : " (no validation layer: errors not checked)");
        ++ran;
    }

    std::printf("d3d12_shader_fma_guard_test: %d device(s) passed\n", ran);
    return 0;
}
