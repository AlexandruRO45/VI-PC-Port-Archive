// WARP versus Vulkan, synthetic VS writes SV_ViewportArrayIndex=1 for all vertices.
// Reuse the existing bundle test's self-contained D3D12 PSO/vertex/readback rig without game assets.
#define main bundle_original_main
#include "d3d12_bundle_test.cpp"
#undef main

namespace {
struct ViewportCase {
    const char* name;
    UINT count;
    bool clipped_scissor;
    bool bundle;
    bool mutate_source;
};
constexpr char viewport_hlsl[] =
    "cbuffer Offset : register(b0) { float2 offset; };\n"
    "cbuffer Tint : register(b1) { float3 tint; };\n"
    "struct V { float4 position : SV_Position; float4 color : COLOR; uint vp : SV_ViewportArrayIndex; };\n"
    "V vs(float2 position : POSITION, float4 color : COLOR) {\n"
    "    V v; v.position = float4(position + offset, 0, 1); v.color = color; v.vp = 1; return v; }\n"
    "float4 ps(float4 position : SV_Position, float4 color : COLOR) : SV_Target {\n"
    "    return float4(color.r * tint.x, tint.y, color.b * tint.z, 1); }\n";

bool run_viewport_case(ID3D12Device* device, const std::vector<std::uint8_t>& vs,
                       const std::vector<std::uint8_t>& ps, const ViewportCase& scenario,
                       std::string& grid, HRESULT& closed) {
    Rig rig;
    if (!rig.init(device, vs, ps)) return false;
    return rig.render([&](Rig& r, ID3D12GraphicsCommandList* l) {
        r.targets(l);
        l->SetPipelineState(r.pso_a.p);
        l->SetGraphicsRootSignature(r.root1.p);
        l->IASetPrimitiveTopology(strip);
        r.vertices(l, vb_quad);
        r.offset(l, 0, 0);
        r.tint(l, tint_red);
        std::array<D3D12_VIEWPORT, 16> views{};
        std::array<D3D12_RECT, 16> scissors{};
        for (UINT i = 0; i < scenario.count; ++i) {
            views[i] = {0, 0, 32, 32, 0, 1};
            scissors[i] = {0, 0, 64, 64};
        }
        views[1] = {32, 0, 32, 32, 0, 1};
        if (scenario.clipped_scissor) scissors[1] = {32, 0, 40, 64};
        l->RSSetViewports(scenario.count, views.data());
        l->RSSetScissorRects(scenario.count, scissors.data());
        if (scenario.mutate_source) {
            // D3D12 snapshots viewports and scissors at the API call, not at Draw/ExecuteBundle.
            views[1] = {0, 0, 1, 1, 0, 1};
            scissors[1] = {0, 0, 0, 0};
        }
        if (scenario.bundle) {
            auto* b = r.bundle(r.pso_a.p, [](ID3D12GraphicsCommandList* bundle) {
                bundle->IASetPrimitiveTopology(strip);
                bundle->DrawInstanced(4, 1, 0, 0);
            });
            r.execute(l, b);
        } else {
            l->DrawInstanced(4, 1, 0, 0);
        }
    }, grid, closed);
}
}

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    vktest::Device vk;
    vk.request_buffer_device_address = true;
    if (vk.initialize() == vktest::DeviceStatus::unavailable) return 77;
    if (!vk.device) return 1;
    if (!vk.enabled.multiViewport) {
        std::printf("SKIP: Vulkan device does not support multiViewport\n");
        return 77;
    }
    std::vector<std::uint8_t> host_vs, host_ps;
    if (!compile_hlsl(viewport_hlsl, "vs", "vs_5_0", host_vs) ||
        !compile_hlsl(viewport_hlsl, "ps", "ps_5_0", host_ps)) return 1;
    Ref<IDXGIFactory4> factory;
    Ref<IDXGIAdapter> adapter;
    Ref<ID3D12Device> warp;
    if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory.p))) ||
        FAILED(factory->EnumWarpAdapter(IID_PPV_ARGS(&adapter.p))) ||
        FAILED(D3D12CreateDevice(adapter.p, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&warp.p)))) return 77;
    const std::array<ViewportCase, 5> cases{{
        {"two viewports, viewport 1", 2, false, false, false},
        {"scissor 1 clips half of viewport 1", 2, true, false, false},
        {"sixteen viewport/scissor slots, viewport 1", 16, false, false, false},
        {"parent viewports inherited into a bundle", 2, false, true, false},
        {"binding arrays snapshotted before draw", 2, false, false, true},
    }};

    // Real COM layer, original synthetic shader library hashed by unique artificial FSSM bytes.
    const auto vs_program = fssm_program({0x56530102, 0xbf810001});
    const auto ps_program = fssm_program({0x50530102, 0xbf810001});
    const auto vs_container = shader_container(vs_program);
    const auto ps_container = shader_container(ps_program);
    const auto library = std::filesystem::temp_directory_path() / "leonida_multi_viewport_test";
    std::filesystem::create_directories(library);
    struct Cleanup {
        std::filesystem::path path;
        ~Cleanup() { std::error_code ec; std::filesystem::remove_all(path, ec); }
    } cleanup{library};
    std::filesystem::copy_file(LEONIDA_TEST_VS_SPV, library / "vs_draw.spv",
                               std::filesystem::copy_options::overwrite_existing);
    std::filesystem::copy_file(LEONIDA_TEST_PS_SPV, library / "ps_draw.spv",
                               std::filesystem::copy_options::overwrite_existing);
    {
        std::ofstream index(library / "index.json");
        index << "{\"vs_draw\": {\"stage\": \"vs\", \"fssm_sha256\": \"" << leonida::sha256_hex(vs_program)
              << "\", \"bindings\": [], \"push_constant_bytes\": 144, \"input_signature\": ["
                 "{\"semantic\":\"COLOR\",\"semantic_index\":0,\"register\":0,\"mask\":15,\"rw_mask\":15,"
                 "\"system_value\":0,\"component_type\":\"float\"},"
                 "{\"semantic\":\"POSITION\",\"semantic_index\":0,\"register\":1,\"mask\":3,\"rw_mask\":3,"
                 "\"system_value\":0,\"component_type\":\"float\"}]},"
                 "\"ps_draw\": {\"stage\":\"ps\",\"fssm_sha256\":\"" << leonida::sha256_hex(ps_program)
              << "\",\"bindings\":[],\"push_constant_bytes\":144,\"outputs\":[0]}}";
    }
    rt::d3d12::VulkanDeviceInfo info{};
    info.get_instance_proc = vk.instance_proc;
    info.instance = vk.instance;
    info.physical = vk.physical;
    info.device = vk.device;
    info.queue = vk.queue;
    info.queue_family = vk.family;
    info.features = vk.enabled;
    info.buffer_device_address = vk.buffer_device_address;
    info.shader_output_viewport_index = true;
    info.shader_dir = library.string();
    Ref<ID3D12Device> ours;
    if (FAILED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&ours.p)))) return 1;
    const unsigned before = vk.validation_errors();
    unsigned failures = 0;
    for (const auto& scenario : cases) {
        std::string host_grid, layer_grid;
        HRESULT host_close = S_OK, layer_close = S_OK;
        if (!run_viewport_case(warp.p, host_vs, host_ps, scenario, host_grid, host_close) ||
            !run_viewport_case(ours.p, vs_container, ps_container, scenario, layer_grid, layer_close))
            return 1;
        const std::string expected = scenario.clipped_scissor
            ? "BBBBRBBB BBBBRBBB BBBBBBBB BBBBBBBB BBBBBBBB BBBBBBBB BBBBBBBB BBBBBBBB"
            : "BBBBRRBB BBBBRRBB BBBBBBBB BBBBBBBB BBBBBBBB BBBBBBBB BBBBBBBB BBBBBBBB";
        const bool good = host_close == S_OK && layer_close == S_OK &&
                          host_grid == expected && layer_grid == expected;
        std::printf("%s: WARP Close=%08x pixels=%s; Leonida Close=%08x pixels=%s: %s\n",
                    scenario.name, unsigned(host_close), host_grid.c_str(),
                    unsigned(layer_close), layer_grid.c_str(), good ? "PASS" : "FAIL");
        if (!good) ++failures;
    }
    const unsigned errors = vk.validation_errors() - before;
    std::printf("multi-viewport: %zu scenarios, %u failures, Vulkan validation errors=%u\n",
                cases.size(), failures, errors);
    return failures || errors ? 1 : 0;
}
