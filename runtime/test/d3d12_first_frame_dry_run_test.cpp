// First-frame dry run: the contiguous boot order docs/boot.md (sections 3 and 5) and docs/d3d12_vulkan.md (M0/M1)
// expect from the retail executable, run without the game or its data:
//
//   D3D12XboxCreateDevice (the measured 24-word params block)
//     -> CreateCommandQueue, CreateFence
//     -> CreateCommandAllocator, CreateCommandList
//     -> CreateCommittedResource (vertex buffer, upload, render target, readback), CreateDescriptorHeap
//     -> CreateRootSignature (one 32BIT_CONSTANTS parameter: s0..s31; the layer's 144-byte block adds heap base)
//     -> CreateGraphicsPipelineState (VS+PS found by FSSM hash in a translator-style index)
//     -> SetGraphicsRoot32BitConstants, IASetVertexBuffers, DrawInstanced
//     -> ResourceBarrier, CopyTextureRegion, ExecuteCommandLists, Signal, SetEventOnCompletion, Map, compare
//
// The test deliberately takes the export through rt::shims::find, which is the binding a mapped import gets, instead
// of calling create_vulkan_device / create_device_proxy directly as the other D3D12 tests do. That makes this the
// one test that walks device creation, root signature, PSO, vertex buffer, draw and readback as one sequence on the
// device the title is actually handed. The pixel reference is computed on the CPU: the PS hashes s0..s31 with FNV-1a
// and reports whether the heap-base tail is non-zero, so every one of the 32 user dwords and the tail is checked.
//
// No game data and no text: the shaders are GLSL written to the translator's contract (test/glsl/vs_first_frame.vert,
// ps_first_frame.frag) compiled by glslangValidator at build time. The shader containers are synthetic FSSM
// programs keyed by their SHA-256, exactly as d3d12_draw_test does.
//
// Usage: d3d12_first_frame_dry_run_test [--through=device|queues|allocator|resources|root|pso|draw|readback]
// The stage runs the prefix and stops; readback is the default. Exit 0: every requested stage passed. Exit 1: a
// stage printed GAP. Exit 77: no usable Vulkan device (ctest skips). Set LEONIDA_VK_DEVICE to pick a GPU and
// LEONIDA_D3D12_TRACE=1 to write the [d3d12-trace] lines to stderr.
#include "d3d12_test_shaders.h"
#include "shader_fixture_path.h"
#include "test_temp_dir.h"
#include "../shims/d3d12_com.h"
#include "../shims/d3d12_trace.h"
#include "../shims/d3d12_xs_device.h"
#include "../shims/shims.h"
#include "../src/root_signature.h"
#include "../src/shader_library.h"

#include <windows.h>
#include <d3d12.h>

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <string_view>
#include <vector>

#ifndef LEONIDA_TEST_VS_SPV
#error "LEONIDA_TEST_VS_SPV must name the compiled first-frame vertex shader"
#endif
#ifndef LEONIDA_TEST_PS_SPV
#error "LEONIDA_TEST_PS_SPV must name the compiled first-frame pixel shader"
#endif

using namespace d3d12test;
using XboxCreateDeviceFn = HRESULT(WINAPI*)(IUnknown*, const void*, const IID*, void**);
using XboxHeapHandleFn = std::uint64_t(WINAPI*)(void* heap);  // heap slot 9: the 8-byte handle comes back in RAX

namespace {

constexpr UINT kWidth = 8, kHeight = 8;
constexpr UINT kRed = 0, kGreen = 1, kBlue = 2, kAlpha = 3;

int g_passed_stages = 0;
std::vector<std::string> g_gaps;

const char* const kStages[] = {"device", "queues", "allocator", "resources", "root", "pso", "draw", "readback"};
constexpr int kStageCount = int(std::size(kStages));

int stage_index(std::string_view name) {
    for (int i = 0; i < kStageCount; ++i)
        if (name == kStages[i]) return i;
    return -1;
}

int finish();

int gap(const char* stage, const char* what, HRESULT hr = S_OK) {
    char line[512];
    if (hr != S_OK)
        std::snprintf(line, sizeof(line), "%s: %s (hr=0x%08lx)", stage, what, static_cast<unsigned long>(hr));
    else
        std::snprintf(line, sizeof(line), "%s: %s", stage, what);
    std::printf("GAP: %s\n", line);
    std::fflush(stdout);
    g_gaps.emplace_back(line);
    return finish();
}

void stage_ok(const char* stage) {
    ++g_passed_stages;
    std::printf("[stage] %s: ok\n", stage);
    std::fflush(stdout);
}

int finish() {
    std::printf("first-frame dry run: %d/%d stages passed, %zu gap(s)\n", g_passed_stages, kStageCount,
                g_gaps.size());
    for (const auto& gap_line : g_gaps) std::printf("  %s\n", gap_line.c_str());
    std::fflush(stdout);
    return g_gaps.empty() ? 0 : 1;
}

template <class Fn>
Fn shim_export(const char* name) {
    return reinterpret_cast<Fn>(rt::shims::find("d3d12_xs.dll", name));
}

std::uint32_t fnv1a(const std::array<std::uint32_t, 32>& s) {
    std::uint32_t h = 2166136261u;  // the PS's seed
    for (const std::uint32_t value : s) {
        h ^= value;
        h *= 16777619u;
    }
    return h;
}

}  // namespace

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);

    std::string_view through = "readback";
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg = argv[i];
        if (arg.starts_with("--through=")) through = arg.substr(std::strlen("--through="));
    }
    const int stop = stage_index(through);
    if (stop < 0) {
        std::printf("unknown --through=%.*s (expected one of:", int(through.size()), through.data());
        for (const char* stage : kStages) std::printf(" %s", stage);
        std::printf(")\n");
        return 2;
    }
    std::printf("d3d12 first-frame dry run: through=%s\n", std::string(through).c_str());

    // This harness is about the Vulkan layer: the shader containers it feeds to CreateGraphicsPipelineState are
    // FSSM/SPIR-V, which only the layer's PSO path understands. Force the backend the runbook selects.
    if (_putenv_s("LEONIDA_D3D12", "vulkan") != 0) return gap("setup", "cannot set LEONIDA_D3D12");
    if (std::getenv("LEONIDA_D3D12_TRACE")) rt::d3d12_trace::set_enabled(true);

    // A translator-style library: the synthetic FSSM programs the containers carry, and the compiled modules their
    // hashes map to. CreateGraphicsPipelineState reads this folder through LEONIDA_SHADERS.
    testfiles::ScratchDir scratch("leonida_first_frame");
    const std::filesystem::path& library_dir = scratch.path();
    if (!testfiles::copy_file(vktest::shader_fixture_path(LEONIDA_TEST_VS_SPV),
                              library_dir / "vs_first_frame.spv") ||
        !testfiles::copy_file(vktest::shader_fixture_path(LEONIDA_TEST_PS_SPV),
                              library_dir / "ps_first_frame.spv"))
        return gap("setup", "cannot stage the compiled shaders");
    if (_putenv_s("LEONIDA_SHADERS", library_dir.string().c_str()) != 0)
        return gap("setup", "cannot set LEONIDA_SHADERS");

    const auto vs_program = fssm_program({0x56530010, 0xbf810000});
    const auto ps_program = fssm_program({0x50530010, 0xbf810000});
    {
        std::ofstream index(library_dir / "index.json", std::ios::binary);
        index << "{\"vs_first_frame\": {\"stage\": \"vs\", \"fssm_sha256\": \"" << leonida::sha256_hex(vs_program)
              << "\", \"bindings\": [], \"push_constant_bytes\": 144, \"input_signature\": ["
                 "{\"semantic\": \"POSITION\", \"semantic_index\": 0, \"register\": 0, \"mask\": 3, \"rw_mask\": 3, "
                 "\"system_value\": 0, \"component_type\": \"float\"}]},\n"
                 " \"ps_first_frame\": {\"stage\": \"ps\", \"fssm_sha256\": \""
              << leonida::sha256_hex(ps_program)
              << "\", \"bindings\": [], \"push_constant_bytes\": 144, \"outputs\": [0]}}";
        if (!index) return gap("setup", "cannot write index.json");
    }
    const auto vs_container = shader_container(vs_program), ps_container = shader_container(ps_program);

    // Device availability first: a standalone Vulkan device is what the export creates, so a failure here is a
    // skip, not a first-frame gap. It is released before the export call, which creates its own.
    {
        ID3D12Device* probe = nullptr;
        const HRESULT probe_hr = rt::d3d12::create_vulkan_device_standalone(IID_PPV_ARGS(&probe));
        if (probe) probe->Release();
        if (FAILED(probe_hr) || !probe) {
            const char* wanted = std::getenv("LEONIDA_VK_DEVICE");
            std::printf("SKIP: no usable Vulkan 1.3 device%s%s (hr=0x%08lx)\n", wanted ? " matching " : "",
                        wanted ? wanted : "", static_cast<unsigned long>(probe_hr));
            return 77;
        }
    }

    // ---- device ---------------------------------------------------------------------------------------------
    // The documented measured block: 24 32-bit words (96 bytes), only logged by the shim; the field meanings are
    // still open (docs/d3d12_vulkan.md question 1). Print it ourselves so the run's report carries it either way.
    std::array<std::uint32_t, 24> params{};
    static_assert(sizeof(params) == 96, "D3D12XboxCreateDevice params block is 24 words");
    for (std::size_t i = 0; i < params.size(); ++i) params[i] = 0x4C454F4Eu + std::uint32_t(i);
    std::printf("  D3D12XboxCreateDevice params[24] =");
    for (const std::uint32_t word : params) std::printf(" %08x", word);
    std::printf("\n");

    const auto create_device = shim_export<XboxCreateDeviceFn>("D3D12XboxCreateDevice");
    if (!create_device) return gap("device", "D3D12XboxCreateDevice is not in the d3d12_xs shim registry");
    const IID device_iid = __uuidof(ID3D12Device);
    Ref<ID3D12Device> device;
    HRESULT hr = create_device(nullptr, params.data(), &device_iid, reinterpret_cast<void**>(&device.p));
    if (FAILED(hr) || !device.p)
        return gap("device", "D3D12XboxCreateDevice did not return an ID3D12Device", hr);
    ID3D12Device* const inner = rt::d3d12_xs::device_proxy_real_device(device.p);
    if (!inner) return gap("device", "the returned object is not the Xbox-ABI device proxy");
    const std::uint32_t max_push_constants = rt::d3d12::device_max_push_constants_size(inner);
    const bool push_fallback = rt::d3d12::device_push_constants_need_fallback(inner);
    std::printf("  device proxy %p, Vulkan device %p, maxPushConstantsSize=%u, fallback=%s\n",
                static_cast<void*>(device.p), static_cast<void*>(inner), max_push_constants,
                push_fallback ? "yes" : "no");
    if (max_push_constants < 144 || push_fallback)
        return gap("device", "the device cannot carry the 144-byte push-constant block natively");
    stage_ok("device");
    if (stop == 0) return finish();

    // ---- queues and fence -----------------------------------------------------------------------------------
    D3D12_COMMAND_QUEUE_DESC queue_desc{};
    queue_desc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    Ref<ID3D12CommandQueue> queue;
    hr = device->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&queue.p));
    if (FAILED(hr) || !queue.p) return gap("queues", "CreateCommandQueue(DIRECT)", hr);
    Ref<ID3D12Fence> fence;
    hr = device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence.p));
    if (FAILED(hr) || !fence.p) return gap("queues", "CreateFence", hr);
    std::printf("  queue %p, fence %p\n", static_cast<void*>(queue.p), static_cast<void*>(fence.p));
    stage_ok("queues");
    if (stop == 1) return finish();

    // ---- allocator and list ---------------------------------------------------------------------------------
    Ref<ID3D12CommandAllocator> allocator;
    hr = device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator.p));
    if (FAILED(hr) || !allocator.p) return gap("allocator", "CreateCommandAllocator(DIRECT)", hr);
    Ref<ID3D12GraphicsCommandList> list;
    hr = device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.p, nullptr, IID_PPV_ARGS(&list.p));
    if (FAILED(hr) || !list.p) return gap("allocator", "CreateCommandList(DIRECT)", hr);
    std::printf("  allocator %p, list %p\n", static_cast<void*>(allocator.p), static_cast<void*>(list.p));
    stage_ok("allocator");
    if (stop == 2) return finish();

    // ---- resources: committed vertex buffer, targets, descriptors -------------------------------------------
    // A quad 0.25 to the left of the target. The VS adds s0 (0.25) / s1 (0.0), so without those root constants the
    // right quarter of the target would keep the clear colour and the pixel compare would fail.
    const float vertices[6][2] = {{-1.25f, -1.0f}, {0.75f, -1.0f}, {-1.25f, 1.0f},
                                  {-1.25f, 1.0f},  {0.75f, -1.0f}, {0.75f, 1.0f}};
    const D3D12_HEAP_PROPERTIES default_heap{D3D12_HEAP_TYPE_DEFAULT};
    const D3D12_HEAP_PROPERTIES upload_heap{D3D12_HEAP_TYPE_UPLOAD};
    const D3D12_HEAP_PROPERTIES readback_heap{D3D12_HEAP_TYPE_READBACK};

    Ref<ID3D12Resource> vertex_buffer;
    const auto vb_desc = buffer_desc(sizeof(vertices));
    hr = device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &vb_desc,
                                         D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&vertex_buffer.p));
    if (FAILED(hr) || !vertex_buffer.p) return gap("resources", "CreateCommittedResource(vertex, DEFAULT)", hr);
    Ref<ID3D12Resource> upload;
    const auto upload_desc = buffer_desc(sizeof(vertices));
    hr = device->CreateCommittedResource(&upload_heap, D3D12_HEAP_FLAG_NONE, &upload_desc,
                                         D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&upload.p));
    if (FAILED(hr) || !upload.p) return gap("resources", "CreateCommittedResource(vertex upload, UPLOAD)", hr);
    std::uint8_t* mapped = nullptr;
    hr = upload->Map(0, nullptr, reinterpret_cast<void**>(&mapped));
    if (FAILED(hr) || !mapped) return gap("resources", "Map of the vertex upload buffer", hr);
    std::memcpy(mapped, vertices, sizeof(vertices));

    D3D12_RESOURCE_DESC target_desc{};
    target_desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    target_desc.Width = kWidth;
    target_desc.Height = kHeight;
    target_desc.DepthOrArraySize = 1;
    target_desc.MipLevels = 1;
    target_desc.Format = DXGI_FORMAT_R32G32B32A32_UINT;
    target_desc.SampleDesc.Count = 1;
    target_desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
    Ref<ID3D12Resource> target;
    hr = device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &target_desc,
                                         D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&target.p));
    if (FAILED(hr) || !target.p) return gap("resources", "CreateCommittedResource(R32G32B32A32_UINT target)", hr);

    D3D12_PLACED_SUBRESOURCE_FOOTPRINT placed{};
    UINT rows = 0;
    UINT64 row_bytes = 0, readback_bytes = 0;
    device->GetCopyableFootprints(&target_desc, 0, 1, 0, &placed, &rows, &row_bytes, &readback_bytes);
    std::printf("  GetCopyableFootprints: rows=%u row_bytes=%llu pitch=%u total=%llu\n", rows,
                static_cast<unsigned long long>(row_bytes), placed.Footprint.RowPitch,
                static_cast<unsigned long long>(readback_bytes));
    if (rows != kHeight || row_bytes != kWidth * 16 || placed.Footprint.RowPitch < kWidth * 16 ||
        readback_bytes < UINT64(placed.Footprint.RowPitch) * (kHeight - 1) + kWidth * 16)
        return gap("resources", "GetCopyableFootprints does not describe the target");
    const UINT pitch = placed.Footprint.RowPitch;
    Ref<ID3D12Resource> readback;
    const auto readback_desc = buffer_desc(readback_bytes);
    hr = device->CreateCommittedResource(&readback_heap, D3D12_HEAP_FLAG_NONE, &readback_desc,
                                         D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&readback.p));
    if (FAILED(hr) || !readback.p) return gap("resources", "CreateCommittedResource(readback, READBACK)", hr);

    // The RTV heap's start handle is read the console way: slot 9 returns the 8-byte handle in RAX.
    D3D12_DESCRIPTOR_HEAP_DESC heap_desc{D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 1, D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0};
    Ref<ID3D12DescriptorHeap> rtv_heap;
    hr = device->CreateDescriptorHeap(&heap_desc, IID_PPV_ARGS(&rtv_heap.p));
    if (FAILED(hr) || !rtv_heap.p) return gap("resources", "CreateDescriptorHeap(RTV)", hr);
    const std::uint64_t start =
        reinterpret_cast<XboxHeapHandleFn>(slot(rtv_heap.p, 9))(rtv_heap.p);
    if (!start) return gap("resources", "the RTV heap's start handle is 0 (heap slot 9)");
    const D3D12_CPU_DESCRIPTOR_HANDLE rtv{static_cast<SIZE_T>(start)};
    device->CreateRenderTargetView(target.p, nullptr, rtv);

    // Record the upload and the resource transitions; they belong to resource creation before the first draw.
    list->CopyBufferRegion(vertex_buffer.p, 0, upload.p, 0, sizeof(vertices));
    D3D12_RESOURCE_BARRIER barriers[2] = {
        transition(vertex_buffer.p, D3D12_RESOURCE_STATE_COPY_DEST,
                   D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER),
        transition(target.p, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_RENDER_TARGET)};
    list->ResourceBarrier(2, barriers);
    const FLOAT clear[4] = {0.25f, 0.5f, 0.75f, 1.0f};  // for a UINT target: 0, 0, 0, 1 (the sentinel)
    list->ClearRenderTargetView(rtv, clear, 0, nullptr);
    std::printf("  vertex buffer %p (%zu bytes), target %p (%ux%u, R32G32B32A32_UINT), readback pitch %u\n",
                static_cast<void*>(vertex_buffer.p), sizeof(vertices), static_cast<void*>(target.p), kWidth,
                kHeight, pitch);
    stage_ok("resources");
    if (stop == 3) return finish();

    // ---- root signature: one 32BIT_CONSTANTS parameter, all 32 user dwords -----------------------------------
    // The measured Xbox flag high byte (0x03000000) is set deliberately: the d3d12_xs serializer must clear it on a
    // copy, and the layer's RTS0 parse must still map the parameter to dwords 0..31.
    D3D12_ROOT_PARAMETER root_param{};
    root_param.ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    root_param.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    root_param.Constants = {0, 0, 32};
    D3D12_VERSIONED_ROOT_SIGNATURE_DESC root_desc{};
    root_desc.Version = D3D_ROOT_SIGNATURE_VERSION_1_0;
    root_desc.Desc_1_0.NumParameters = 1;
    root_desc.Desc_1_0.pParameters = &root_param;
    root_desc.Desc_1_0.NumStaticSamplers = 0;
    root_desc.Desc_1_0.pStaticSamplers = nullptr;
    root_desc.Desc_1_0.Flags = static_cast<D3D12_ROOT_SIGNATURE_FLAGS>(
        D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT | 0x03000000u);
    const auto serialize = shim_export<d3d12test::SerializeFn>("D3D12SerializeVersionedRootSignature");
    if (!serialize) return gap("root", "D3D12SerializeVersionedRootSignature is not in the shim registry");
    Ref<ID3DBlob> blob, error;
    hr = serialize(&root_desc, &blob.p, &error.p);
    if (FAILED(hr) || !blob.p) {
        if (error.p)
            std::printf("  serializer: %s\n", static_cast<const char*>(error.p->GetBufferPointer()));
        return gap("root", "D3D12SerializeVersionedRootSignature with the Xbox flag high byte", hr);
    }
    const auto parsed = rt::rootsig::parse_rts0(static_cast<const std::uint8_t*>(blob.p->GetBufferPointer()),
                                                blob.p->GetBufferSize());
    if (!parsed.ok) return gap("root", ("the layer cannot parse the serialized RTS0: " + parsed.error).c_str());
    if ((parsed.sig.flags & 0xFF000000u) != 0)
        return gap("root", "the serializer did not clear the Xbox flag high byte");
    if (parsed.sig.params.size() != 1 || parsed.sig.params[0].type != rt::rootsig::ParamType::Constants ||
        parsed.sig.params[0].num_values != 32)
        return gap("root", "the RTS0 does not hold one 32-value constants parameter");
    const auto packed = rt::rootsig::pack(parsed.sig);
    if (!packed.ok || packed.dword_count != 32)
        return gap("root", ("the parameter does not pack to s0..s31: " + packed.error).c_str());

    Ref<ID3D12RootSignature> root;
    hr = device->CreateRootSignature(0, blob.p->GetBufferPointer(), blob.p->GetBufferSize(), IID_PPV_ARGS(&root.p));
    if (FAILED(hr) || !root.p) return gap("root", "CreateRootSignature(32 constants)", hr);

    // The 32 dwords: s0 = 0.25f (the VS's x offset), s1 = 0.0f, s2..s31 a recognisable pattern for the PS hash.
    std::array<std::uint32_t, 32> user_data{};
    user_data[0] = 0x3E800000u;  // 0.25f
    user_data[1] = 0x00000000u;  // 0.0f
    for (std::size_t i = 2; i < user_data.size(); ++i)
        user_data[i] = 0x5A5A0000u + std::uint32_t(i);
    list->SetGraphicsRootSignature(root.p);
    list->SetGraphicsRoot32BitConstants(0, 32, user_data.data(), 0);
    std::printf("  root signature %p: 1 parameter -> s0..s31, %u packed dwords, flags 0x%08x\n",
                static_cast<void*>(root.p), packed.dword_count, parsed.sig.flags);
    stage_ok("root");
    if (stop == 4) return finish();

    // ---- pipeline state -------------------------------------------------------------------------------------
    const D3D12_INPUT_ELEMENT_DESC element = {"POSITION", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 0,
                                              D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0};
    D3D12_GRAPHICS_PIPELINE_STATE_DESC pso_desc{};
    pso_desc.pRootSignature = root.p;
    pso_desc.VS = {vs_container.data(), vs_container.size()};
    pso_desc.PS = {ps_container.data(), ps_container.size()};
    pso_desc.BlendState.RenderTarget[0].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
    pso_desc.SampleMask = 0xffffffff;
    pso_desc.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
    pso_desc.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
    pso_desc.RasterizerState.DepthClipEnable = TRUE;
    pso_desc.InputLayout = {&element, 1};
    pso_desc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    pso_desc.NumRenderTargets = 1;
    pso_desc.RTVFormats[0] = DXGI_FORMAT_R32G32B32A32_UINT;
    pso_desc.SampleDesc.Count = 1;
    Ref<ID3D12PipelineState> pso;
    hr = device->CreateGraphicsPipelineState(&pso_desc, IID_PPV_ARGS(&pso.p));
    if (FAILED(hr) || !pso.p) return gap("pso", "CreateGraphicsPipelineState(VS+PS by FSSM hash)", hr);
    std::printf("  pipeline state %p\n", static_cast<void*>(pso.p));
    stage_ok("pso");
    if (stop == 5) return finish();

    // ---- draw ------------------------------------------------------------------------------------------------
    list->SetPipelineState(pso.p);
    list->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    const D3D12_VERTEX_BUFFER_VIEW vbv{vertex_buffer->GetGPUVirtualAddress(), UINT(sizeof(vertices)), 2 * sizeof(float)};
    list->IASetVertexBuffers(0, 1, &vbv);
    const D3D12_VIEWPORT viewport{0, 0, float(kWidth), float(kHeight), 0, 1};
    const D3D12_RECT scissor{0, 0, LONG(kWidth), LONG(kHeight)};
    list->RSSetViewports(1, &viewport);
    list->RSSetScissorRects(1, &scissor);
    list->OMSetRenderTargets(1, &rtv, FALSE, nullptr);
    list->DrawInstanced(6, 1, 0, 0);
    D3D12_RESOURCE_BARRIER to_copy =
        transition(target.p, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_COPY_SOURCE);
    list->ResourceBarrier(1, &to_copy);
    stage_ok("draw");
    if (stop == 6) return finish();

    // ---- readback: submit, wait on a fence, compare with the CPU reference ------------------------------------
    D3D12_TEXTURE_COPY_LOCATION dst{};
    dst.pResource = readback.p;
    dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    dst.PlacedFootprint = placed;
    D3D12_TEXTURE_COPY_LOCATION src{};
    src.pResource = target.p;
    src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    src.SubresourceIndex = 0;
    list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);

    hr = list->Close();
    if (FAILED(hr)) return gap("readback", "Close", hr);
    ID3D12CommandList* lists[] = {list.p};
    queue->ExecuteCommandLists(1, lists);
    HANDLE done = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    hr = queue->Signal(fence.p, 1);
    if (FAILED(hr)) return gap("readback", "Signal", hr);
    hr = fence->SetEventOnCompletion(1, done);
    if (FAILED(hr) || !done || WaitForSingleObject(done, 30000) != WAIT_OBJECT_0)
        return gap("readback", "the fence did not reach value 1", hr);
    CloseHandle(done);
    if (fence->GetCompletedValue() < 1) return gap("readback", "GetCompletedValue is below the signal");

    void* bytes = nullptr;
    hr = readback->Map(0, nullptr, &bytes);
    if (FAILED(hr) || !bytes) return gap("readback", "Map of the readback buffer", hr);
    const std::array<std::uint32_t, 4> expected = {fnv1a(user_data), 0x48454150u, 0x5EED5EEDu, 0xFFFFFFFFu};
    unsigned wrong = 0, first_x = 0, first_y = 0;
    std::array<std::uint32_t, 4> first_got{};
    const auto* base = static_cast<const std::uint8_t*>(bytes) + placed.Offset;
    for (UINT y = 0; y < kHeight; ++y) {
        for (UINT x = 0; x < kWidth; ++x) {
            std::array<std::uint32_t, 4> got{};
            std::memcpy(got.data(), base + y * pitch + x * 16, 16);
            if (got != expected) {
                if (!wrong) {
                    first_x = x;
                    first_y = y;
                    first_got = got;
                }
                ++wrong;
            }
        }
    }
    readback->Unmap(0, nullptr);
    if (wrong) {
        char detail[384];
        std::snprintf(detail, sizeof(detail),
                      "%u of %u pixels differ from the CPU reference; first (%u,%u) got %08x %08x %08x %08x, "
                      "want %08x %08x %08x %08x",
                      wrong, kWidth * kHeight, first_x, first_y, first_got[0], first_got[1], first_got[2],
                      first_got[3], expected[0], expected[1], expected[2], expected[3]);
        return gap("readback", detail);
    }
    std::printf("  readback: all %u pixels match the CPU FNV-1a reference (s0..s31) and a non-zero heap base\n",
                kWidth * kHeight);
    stage_ok("readback");
    return finish();
}
