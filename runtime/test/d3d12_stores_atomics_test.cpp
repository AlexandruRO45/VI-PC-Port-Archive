// A pixel shader that declares a writable storage image (an image atomic, as 1,018 of the corpus's pixel shaders
// do) needs the Vulkan feature fragmentStoresAndAtomics: VUID-RuntimeSpirv-NonWritable-06340. Two devices:
//   - a device without the feature (this test's own Vulkan device does not enable it): CreateGraphicsPipelineState
//     refuses the PSO with E_NOTIMPL instead of creating a pipeline Vulkan does not allow, and still creates the
//     same PSO with a pixel shader that has no storage image;
//   - the runtime's own device (create_vulkan_device_standalone, the one D3D12XboxCreateDevice hands a title with
//     LEONIDA_D3D12=vulkan): it enables fragmentStoresAndAtomics and vertexPipelineStoresAndAtomics exactly where
//     the GPU has them, and with the first one creates the PSO.
// When VK_LAYER_KHRONOS_validation is installed, both devices run under it (the test turns it on for itself unless
// LEONIDA_VK_VALIDATION=0) and any validation error fails the test: before the fix the second device gave -06340
// and S_OK. Without the layer the test still checks the HRESULTs and the feature bits.
// The shaders are GLSL written to the translator's runtime contract; nothing is drawn. No game data.
#include "d3d12_test_shaders.h"
#include "shader_fixture_path.h"
#include "vk_test_device.h"
#include "../shims/d3d12_com.h"
#include "../src/shader_library.h"
#include "../src/spirv_stores.h"
#include "../src/spirv_words.h"

#include <atomic>
#include <cstdlib>
#include <cstring>

#if !defined(LEONIDA_TEST_VS_FULLSCREEN_SPV) || !defined(LEONIDA_TEST_PS_SPV) || !defined(LEONIDA_TEST_PS_ATOMIC_SPV)
#error "LEONIDA_TEST_VS_FULLSCREEN_SPV, LEONIDA_TEST_PS_SPV and LEONIDA_TEST_PS_ATOMIC_SPV must name the test shaders"
#endif

#define EXPECT(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return 1; } \
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

D3D12_GRAPHICS_PIPELINE_STATE_DESC pipeline_desc(ID3D12RootSignature* root, const std::vector<std::uint8_t>& vs,
                                                 const std::vector<std::uint8_t>& ps) {
    D3D12_GRAPHICS_PIPELINE_STATE_DESC desc{};
    desc.pRootSignature = root;
    desc.VS = {vs.data(), vs.size()};
    desc.PS = {ps.data(), ps.size()};
    desc.BlendState.RenderTarget[0].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
    desc.SampleMask = 0xffffffff;
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
    D3D12_ROOT_PARAMETER parameter{};
    parameter.ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    parameter.Constants = {0, 0, 1};  // s0: the value the atomic writes
    D3D12_VERSIONED_ROOT_SIGNATURE_DESC desc{};
    desc.Version = D3D_ROOT_SIGNATURE_VERSION_1_0;
    desc.Desc_1_0 = {1, &parameter, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_NONE};
    const auto blob = serialize_root_signature(desc);
    if (blob.empty()) return E_FAIL;
    return device->CreateRootSignature(0, blob.data(), blob.size(), IID_PPV_ARGS(root));
}

std::vector<std::uint32_t> read_words(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) return {};
    return leonida::read_spirv_words(file, file.tellg());
}

}  // namespace

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    // Under the validation layer wherever it is installed: that is what shows the pipeline is one Vulkan allows.
    // Unlike the other tests, this one turns the layer on for itself when LEONIDA_VK_VALIDATION is not set; set to
    // 0 it stays off, and the test then checks the HRESULTs and the feature bits only.
#pragma warning(push)
#pragma warning(disable : 4996)  // getenv: read once, at startup
    const char* const requested = std::getenv("LEONIDA_VK_VALIDATION");
#pragma warning(pop)
    const bool validating = requested ? requested[0] && requested[0] != '0' : validation_layer_installed();
    if (validating && !requested) _putenv_s("LEONIDA_VK_VALIDATION", "1");

    // The shader library: three programs and the translator's index for them. The atomic pixel shader lists the
    // set-0 binding it declares (4: the R32ui storage images).
    const auto vs_program = fssm_program({0x56530002, 0xbf810000});
    const auto ps_program = fssm_program({0x50530001, 0xbf810000});
    const auto atomic_program = fssm_program({0x50530a70, 0xbf810000});
    const auto vs_container = shader_container(vs_program), ps_container = shader_container(ps_program);
    const auto atomic_container = shader_container(atomic_program);
    const auto library = std::filesystem::temp_directory_path() / "leonida_stores_atomics_test";
    std::filesystem::create_directories(library);
    const struct { const char* compiled; const char* name; } files[] = {
        {LEONIDA_TEST_VS_FULLSCREEN_SPV, "vs_fullscreen.spv"},
        {LEONIDA_TEST_PS_SPV, "ps_draw.spv"},
        {LEONIDA_TEST_PS_ATOMIC_SPV, "ps_image_atomic.spv"}};
    for (const auto& file : files)
        std::filesystem::copy_file(vktest::shader_fixture_path(file.compiled), library / file.name,
                                   std::filesystem::copy_options::overwrite_existing);
    {
        std::ofstream index(library / "index.json", std::ios::binary);
        index << "{\"vs_fullscreen\": {\"stage\": \"vs\", \"fssm_sha256\": \"" << leonida::sha256_hex(vs_program)
              << "\", \"bindings\": [], \"push_constant_bytes\": 144, \"input_signature\": []},\n"
                 " \"ps_draw\": {\"stage\": \"ps\", \"fssm_sha256\": \"" << leonida::sha256_hex(ps_program)
              << "\", \"bindings\": [], \"push_constant_bytes\": 144, \"outputs\": [0]},\n"
                 " \"ps_image_atomic\": {\"stage\": \"ps\", \"fssm_sha256\": \"" << leonida::sha256_hex(atomic_program)
              << "\", \"bindings\": [4], \"push_constant_bytes\": 144, \"outputs\": [0]}}";
    }
    // What the device's check reads off the modules: only the atomic shader declares writable storage.
    EXPECT(leonida::spirv_declares_writable_storage(read_words(library / "ps_image_atomic.spv")));
    EXPECT(!leonida::spirv_declares_writable_storage(read_words(library / "ps_draw.spv")));
    EXPECT(!leonida::spirv_declares_writable_storage(read_words(library / "vs_fullscreen.spv")));

    int ran = 0;

    // 1. A device without the feature: the test's own Vulkan device leaves fragmentStoresAndAtomics off.
    {
        vktest::Device d;
        const auto status = d.initialize();
        if (status == vktest::DeviceStatus::unavailable) {
            std::fprintf(stderr, "SKIP: no usable Vulkan 1.3 device\n");
            return 77;
        }
        EXPECT(status == vktest::DeviceStatus::ready);
        if (!d.bindless) {
            std::printf("SKIP: the device lacks the descriptor indexing features of the bindless set 0\n");
            return 77;
        }
        EXPECT(d.enabled.fragmentStoresAndAtomics == VK_FALSE);
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
        Ref<ID3D12Device> device;
        EXPECT(SUCCEEDED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&device.p))));
        Ref<ID3D12RootSignature> root;
        EXPECT(SUCCEEDED(make_root(device.p, &root.p)));
        const auto atomic_desc = pipeline_desc(root.p, vs_container, atomic_container);
        ID3D12PipelineState* refused = reinterpret_cast<ID3D12PipelineState*>(std::uintptr_t{1});
        EXPECT(device->CreateGraphicsPipelineState(&atomic_desc, IID_PPV_ARGS(&refused)) == E_NOTIMPL && !refused);
        // The refusal is the storage image's: the same PSO with a pixel shader that has none is created.
        const auto plain_desc = pipeline_desc(root.p, vs_container, ps_container);
        Ref<ID3D12PipelineState> plain;
        EXPECT(SUCCEEDED(device->CreateGraphicsPipelineState(&plain_desc, IID_PPV_ARGS(&plain.p))) && plain.p);
        plain.reset();
        root.reset();
        device.reset();
        EXPECT(d.validation_errors() == 0);
        std::printf("without fragmentStoresAndAtomics: the PSO with the atomic pixel shader is refused (E_NOTIMPL), "
                    "the plain one is created%s\n", validating ? "; validation errors: 0" : "");
        ++ran;
    }

    // 2. The runtime's own device: the features are on where the GPU has them, and the PSO is created.
    {
        _putenv_s("LEONIDA_SHADERS", library.string().c_str());
        Ref<ID3D12Device> device;
        EXPECT(SUCCEEDED(rt::d3d12::create_vulkan_device_standalone(IID_PPV_ARGS(&device.p))));
        const rt::d3d12::VulkanDeviceInfo* info = rt::d3d12::vulkan_device_info(device.p);
        EXPECT(info != nullptr);
        const auto get_features = reinterpret_cast<PFN_vkGetPhysicalDeviceFeatures>(
            info->get_instance_proc(info->instance, "vkGetPhysicalDeviceFeatures"));
        EXPECT(get_features != nullptr);
        VkPhysicalDeviceFeatures have{};
        get_features(info->physical, &have);
        EXPECT(info->features.fragmentStoresAndAtomics == have.fragmentStoresAndAtomics);
        EXPECT(info->features.vertexPipelineStoresAndAtomics == have.vertexPipelineStoresAndAtomics);

        VkDebugUtilsMessengerEXT messenger = VK_NULL_HANDLE;
        const auto create_messenger = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(
            info->get_instance_proc(info->instance, "vkCreateDebugUtilsMessengerEXT"));
        const auto destroy_messenger = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(
            info->get_instance_proc(info->instance, "vkDestroyDebugUtilsMessengerEXT"));
        const bool watching = validating && info->debug_utils && create_messenger && destroy_messenger;
        if (watching) {
            VkDebugUtilsMessengerCreateInfoEXT messenger_info{VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT};
            messenger_info.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
            messenger_info.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT;
            messenger_info.pfnUserCallback = on_standalone_message;
            EXPECT(create_messenger(info->instance, &messenger_info, nullptr, &messenger) == VK_SUCCESS);
        }
        Ref<ID3D12RootSignature> root;
        EXPECT(SUCCEEDED(make_root(device.p, &root.p)));
        const auto atomic_desc = pipeline_desc(root.p, vs_container, atomic_container);
        Ref<ID3D12PipelineState> pso;
        const HRESULT hr = device->CreateGraphicsPipelineState(&atomic_desc, IID_PPV_ARGS(&pso.p));
        if (have.fragmentStoresAndAtomics) {
            EXPECT(hr == S_OK && pso.p);
        } else {
            EXPECT(hr == E_NOTIMPL && !pso.p);  // a GPU without the feature: refused here too
        }
        pso.reset();
        root.reset();
        const unsigned errors = g_standalone_errors.load();
        if (messenger) destroy_messenger(info->instance, messenger, nullptr);
        device.reset();
        EXPECT(errors == 0);
    std::printf("the runtime's own device: fragmentStoresAndAtomics %s, vertexPipelineStoresAndAtomics %s; the PSO "
                    "with the atomic pixel shader is %s%s\n",
                    have.fragmentStoresAndAtomics ? "enabled" : "absent on this GPU",
                    have.vertexPipelineStoresAndAtomics ? "enabled" : "absent on this GPU",
                    have.fragmentStoresAndAtomics ? "created" : "refused",
                    watching ? "; validation errors: 0" : " (no validation layer: errors not checked)");
        ++ran;
    }

    // 3. Vertex and mesh shader stages: vertexPipelineStoresAndAtomics covers vertex/geometry stages
    //    (VUID-RuntimeSpirv-NonWritable-06341), whereas mesh shaders (VK_SHADER_STAGE_MESH_BIT_EXT) do not need it.
    {
        auto push_op = [](std::vector<std::uint32_t>& w, std::uint16_t op, std::initializer_list<std::uint32_t> args) {
            w.push_back((static_cast<std::uint32_t>(args.size() + 1) << 16) | op);
            w.insert(w.end(), args);
        };
        auto push_str_op = [](std::vector<std::uint32_t>& w, std::uint16_t op, std::initializer_list<std::uint32_t> pre,
                              const char* text, std::initializer_list<std::uint32_t> post) {
            const std::size_t len = std::strlen(text) + 1;
            const std::size_t str_words = (len + 3) / 4;
            w.push_back((static_cast<std::uint32_t>(1 + pre.size() + str_words + post.size()) << 16) | op);
            w.insert(w.end(), pre);
            for (std::size_t i = 0; i < str_words; ++i) {
                std::uint32_t word = 0;
                for (std::size_t b = 0; b < 4 && i * 4 + b < len; ++b)
                    word |= static_cast<std::uint32_t>(static_cast<std::uint8_t>(text[i * 4 + b])) << (8 * b);
                w.push_back(word);
            }
            w.insert(w.end(), post);
        };
        auto build_vs_atomic = [&]() {
            std::vector<std::uint32_t> w{0x07230203u, 0x00010500u, 0u, 20u, 0u};
            push_op(w, 17, {1});                                           // OpCapability Shader
            push_op(w, 17, {5302});                                        // OpCapability RuntimeDescriptorArray
            push_str_op(w, 10, {}, "SPV_EXT_descriptor_indexing", {});     // OpExtension
            push_op(w, 14, {0, 1});                                        // OpMemoryModel Logical GLSL450
            push_str_op(w, 15, {0, 4}, "main", {9, 17, 19});               // OpEntryPoint Vertex
            push_op(w, 71, {9, 34, 0});                                    // DescriptorSet 0
            push_op(w, 71, {9, 33, 4});                                    // Binding 4
            push_op(w, 71, {17, 11, 0});                                   // BuiltIn Position
            push_op(w, 71, {19, 30, 0});                                   // Location 0
            push_op(w, 19, {2});                                           // void
            push_op(w, 33, {3, 2});                                        // void()
            push_op(w, 21, {5, 32, 0});                                    // uint
            push_op(w, 25, {6, 5, 1, 0, 0, 0, 2, 33});                     // OpTypeImage %uint 2D ... 2 R32ui
            push_op(w, 29, {7, 6});                                        // OpTypeRuntimeArray
            push_op(w, 32, {8, 0, 7});                                     // OpTypePointer UniformConstant
            push_op(w, 59, {8, 9, 0});                                     // %counters
            push_op(w, 22, {11, 32});                                      // float
            push_op(w, 23, {12, 11, 4});                                   // vec4
            push_op(w, 43, {11, 13, 0});                                   // 0.0f
            push_op(w, 43, {11, 14, 0x3f800000u});                         // 1.0f
            push_op(w, 44, {12, 15, 13, 13, 13, 14});                      // vec4(0,0,0,1)
            push_op(w, 32, {16, 3, 12});                                   // ptr Output vec4
            push_op(w, 59, {16, 17, 3});                                   // %pos
            push_op(w, 44, {12, 18, 14, 13, 13, 14});                      // vec4(1,0,0,1)
            push_op(w, 59, {16, 19, 3});                                   // %out_color
            push_op(w, 54, {2, 4, 0, 3});
            push_op(w, 248, {1});
            push_op(w, 62, {17, 15});
            push_op(w, 62, {19, 18});
            push_op(w, 253, {});
            push_op(w, 56, {});
            return w;
        };
        auto build_ms_atomic = [&]() {
            std::vector<std::uint32_t> w{0x07230203u, 0x00010500u, 0u, 26u, 0u};
            push_op(w, 17, {1});                                           // OpCapability Shader
            push_op(w, 17, {5302});                                        // OpCapability RuntimeDescriptorArray
            push_op(w, 17, {5283});                                        // OpCapability MeshShadingEXT
            push_str_op(w, 10, {}, "SPV_EXT_descriptor_indexing", {});     // OpExtension
            push_str_op(w, 10, {}, "SPV_EXT_mesh_shader", {});             // OpExtension
            push_op(w, 14, {0, 1});                                        // OpMemoryModel Logical GLSL450
            push_str_op(w, 15, {5365, 4}, "main", {9, 17, 23, 25});        // OpEntryPoint MeshEXT
            push_op(w, 16, {4, 17, 32, 1, 1});                             // LocalSize 32 1 1
            push_op(w, 16, {4, 26, 4});                                    // OutputVertices 4
            push_op(w, 16, {4, 5270, 2});                                  // OutputPrimitivesEXT 2
            push_op(w, 16, {4, 5298});                                     // OutputTrianglesEXT
            push_op(w, 71, {9, 34, 0});                                    // DescriptorSet 0
            push_op(w, 71, {9, 33, 4});                                    // Binding 4
            push_op(w, 71, {14, 2});                                       // Block
            push_op(w, 72, {14, 0, 11, 0});                                // Member 0 BuiltIn Position
            push_op(w, 71, {23, 11, 5299});                                // BuiltIn PrimitiveTriangleIndicesEXT
            push_op(w, 71, {25, 30, 0});                                   // Location 0
            push_op(w, 19, {2});
            push_op(w, 33, {3, 2});
            push_op(w, 21, {5, 32, 0});
            push_op(w, 25, {6, 5, 1, 0, 0, 0, 2, 33});
            push_op(w, 29, {7, 6});
            push_op(w, 32, {8, 0, 7});
            push_op(w, 59, {8, 9, 0});
            push_op(w, 43, {5, 10, 4});
            push_op(w, 22, {11, 32});
            push_op(w, 23, {12, 11, 4});
            push_op(w, 30, {14, 12});
            push_op(w, 28, {15, 14, 10});
            push_op(w, 32, {16, 3, 15});
            push_op(w, 59, {16, 17, 3});
            push_op(w, 43, {5, 18, 2});
            push_op(w, 23, {19, 5, 3});
            push_op(w, 28, {20, 19, 18});
            push_op(w, 32, {21, 3, 20});
            push_op(w, 59, {21, 23, 3});
            push_op(w, 28, {22, 12, 10});
            push_op(w, 32, {24, 3, 22});
            push_op(w, 59, {24, 25, 3});
            push_op(w, 54, {2, 4, 0, 3});
            push_op(w, 248, {1});
            push_op(w, 5295, {10, 18});                                    // OpSetMeshOutputsEXT 4 2
            push_op(w, 253, {});
            push_op(w, 56, {});
            return w;
        };
        const auto vs_atomic_words = build_vs_atomic();
        const auto ms_atomic_words = build_ms_atomic();
        EXPECT(leonida::spirv_declares_writable_storage(vs_atomic_words));
        EXPECT(leonida::spirv_declares_writable_storage(ms_atomic_words));

        const auto vs_atomic_prog = fssm_program({0x56530a71u, 0xbf810000u});
        const auto ms_atomic_prog = fssm_program({0x4d530a71u, 0xbf810000u});
        const auto vs_atomic_container = shader_container(vs_atomic_prog);
        const auto ms_atomic_container = shader_container(ms_atomic_prog);

        const auto stage_library = library / "stages";
        std::filesystem::create_directories(stage_library);
        for (const auto& file : files)
            std::filesystem::copy_file(vktest::shader_fixture_path(file.compiled), stage_library / file.name,
                                       std::filesystem::copy_options::overwrite_existing);
        const auto write_spv = [](const std::filesystem::path& p, const std::vector<std::uint32_t>& words) {
            std::ofstream out(p, std::ios::binary);
            out.write(reinterpret_cast<const char*>(words.data()),
                      static_cast<std::streamsize>(words.size() * sizeof(std::uint32_t)));
        };
        write_spv(stage_library / "vs_atomic.spv", vs_atomic_words);
        write_spv(stage_library / "ms_atomic.spv", ms_atomic_words);
        {
            std::ofstream index(stage_library / "index.json", std::ios::binary);
            index << "{\"vs_atomic\": {\"stage\": \"vs\", \"fssm_sha256\": \"" << leonida::sha256_hex(vs_atomic_prog)
                  << "\", \"bindings\": [4], \"push_constant_bytes\": 144, \"input_signature\": []},\n"
                     " \"ms_atomic\": {\"stage\": \"ms\", \"fssm_sha256\": \"" << leonida::sha256_hex(ms_atomic_prog)
                  << "\", \"bindings\": [4], \"push_constant_bytes\": 144, \"local_size\": [32, 1, 1], "
                     "\"max_vertices\": 4, \"max_primitives\": 2},\n"
                     " \"ps_draw\": {\"stage\": \"ps\", \"fssm_sha256\": \"" << leonida::sha256_hex(ps_program)
                  << "\", \"bindings\": [], \"push_constant_bytes\": 144, \"outputs\": [0]},\n"
                     " \"ps_image_atomic\": {\"stage\": \"ps\", \"fssm_sha256\": \""
                  << leonida::sha256_hex(atomic_program)
                  << "\", \"bindings\": [4], \"push_constant_bytes\": 144, \"outputs\": [0]}}";
        }

        const auto create_mesh_pso = [&](ID3D12Device* dev, ID3D12RootSignature* root,
                                         const std::vector<std::uint8_t>& ps_bytes,
                                         ID3D12PipelineState** out_pso) -> HRESULT {
            struct MeshStream {
                alignas(void*) struct {
                    D3D12_PIPELINE_STATE_SUBOBJECT_TYPE type = D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_ROOT_SIGNATURE;
                    ID3D12RootSignature* inner;
                } root_sig;
                alignas(void*) struct {
                    D3D12_PIPELINE_STATE_SUBOBJECT_TYPE type = D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_MS;
                    D3D12_SHADER_BYTECODE inner;
                } ms;
                alignas(void*) struct {
                    D3D12_PIPELINE_STATE_SUBOBJECT_TYPE type = D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PS;
                    D3D12_SHADER_BYTECODE inner;
                } ps;
                alignas(void*) struct {
                    D3D12_PIPELINE_STATE_SUBOBJECT_TYPE type =
                        D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RENDER_TARGET_FORMATS;
                    D3D12_RT_FORMAT_ARRAY inner;
                } rtvs;
            };
            MeshStream stream{};
            stream.root_sig.inner = root;
            stream.ms.inner = {ms_atomic_container.data(), ms_atomic_container.size()};
            stream.ps.inner = {ps_bytes.data(), ps_bytes.size()};
            stream.rtvs.inner.NumRenderTargets = 1;
            stream.rtvs.inner.RTFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
            Ref<ID3D12Device2> device2;
            if (FAILED(dev->QueryInterface(IID_PPV_ARGS(&device2.p)))) return E_NOINTERFACE;
            const D3D12_PIPELINE_STATE_STREAM_DESC stream_desc{sizeof(stream), &stream};
            return device2->CreatePipelineState(&stream_desc, IID_PPV_ARGS(out_pso));
        };

        // 3a. Device without vertexPipelineStoresAndAtomics or fragmentStoresAndAtomics (with mesh_shader if supported).
        {
            vktest::Device d;
            d.request_mesh_shader = true;
            const auto status = d.initialize();
            EXPECT(status == vktest::DeviceStatus::ready);
            if (!d.bindless) return 77;
            EXPECT(d.enabled.vertexPipelineStoresAndAtomics == VK_FALSE);
            EXPECT(d.enabled.fragmentStoresAndAtomics == VK_FALSE);
            const unsigned errors_before = d.validation_errors();
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
            info.mesh_shader = d.mesh_shader;
            info.subgroup = d.subgroup;
            info.shader_dir = stage_library.string();
            info.features = d.enabled;
            Ref<ID3D12Device> device;
            EXPECT(SUCCEEDED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&device.p))));
            Ref<ID3D12RootSignature> root;
            EXPECT(SUCCEEDED(make_root(device.p, &root.p)));

            // Vertex stage with writable storage still requires vertexPipelineStoresAndAtomics.
            const auto vs_atomic_desc = pipeline_desc(root.p, vs_atomic_container, ps_container);
            ID3D12PipelineState* refused = reinterpret_cast<ID3D12PipelineState*>(std::uintptr_t{1});
            EXPECT(device->CreateGraphicsPipelineState(&vs_atomic_desc, IID_PPV_ARGS(&refused)) == E_NOTIMPL &&
                   !refused);

            if (d.mesh_shader) {
                // Mesh stage does not require vertexPipelineStoresAndAtomics (-06341 exempts MeshEXT).
                Ref<ID3D12PipelineState> mesh_pso;
                const HRESULT mesh_hr = create_mesh_pso(device.p, root.p, ps_container, &mesh_pso.p);
                EXPECT(mesh_hr == S_OK && mesh_pso.p);
                mesh_pso.reset();

                // Pairing with a writable fragment stage is still refused without fragmentStoresAndAtomics.
                ID3D12PipelineState* refused_mesh = reinterpret_cast<ID3D12PipelineState*>(std::uintptr_t{1});
                EXPECT(create_mesh_pso(device.p, root.p, atomic_container, &refused_mesh) == E_NOTIMPL &&
                       !refused_mesh);
            }
            root.reset();
            device.reset();
            EXPECT(d.validation_errors() == errors_before);
            std::printf("stage check (no stores features): VS writable refused (E_NOTIMPL), MS writable %s%s\n",
                        d.mesh_shader ? "created (S_OK)" : "skipped (no VK_EXT_mesh_shader)",
                        validating ? "; validation errors: 0" : "");
        }

        // 3b. Standalone device: enables vertexPipelineStoresAndAtomics where available + mesh_shader where available.
        {
            g_standalone_errors.store(0);
            _putenv_s("LEONIDA_SHADERS", stage_library.string().c_str());
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
                VkDebugUtilsMessengerCreateInfoEXT messenger_info{
                    VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT};
                messenger_info.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
                messenger_info.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT;
                messenger_info.pfnUserCallback = on_standalone_message;
                EXPECT(create_messenger(info->instance, &messenger_info, nullptr, &messenger) == VK_SUCCESS);
            }
            Ref<ID3D12RootSignature> root;
            EXPECT(SUCCEEDED(make_root(device.p, &root.p)));
            const auto vs_atomic_desc = pipeline_desc(root.p, vs_atomic_container, ps_container);
            Ref<ID3D12PipelineState> vs_pso;
            const HRESULT vs_hr = device->CreateGraphicsPipelineState(&vs_atomic_desc, IID_PPV_ARGS(&vs_pso.p));
            if (info->features.vertexPipelineStoresAndAtomics) {
                EXPECT(vs_hr == S_OK && vs_pso.p);
            } else {
                EXPECT(vs_hr == E_NOTIMPL && !vs_pso.p);
            }
            vs_pso.reset();
            if (info->mesh_shader) {
                Ref<ID3D12PipelineState> mesh_pso;
                EXPECT(create_mesh_pso(device.p, root.p, ps_container, &mesh_pso.p) == S_OK && mesh_pso.p);
                mesh_pso.reset();
            }
            root.reset();
            const unsigned errors = g_standalone_errors.load();
            if (messenger) destroy_messenger(info->instance, messenger, nullptr);
            device.reset();
            EXPECT(errors == 0);
        }
        std::filesystem::remove_all(stage_library);
    }

    std::printf("d3d12_stores_atomics_test: %d device(s) passed\n", ran);
    return 0;
}
