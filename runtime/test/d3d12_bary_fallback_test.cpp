// A --bary-fallback pixel shader through a graphics PSO (SHADER-98): the layer adds the pass-through geometry shader
// that writes its barycentric input (runtime/src/bary_fallback_gs.h), on every device, with or without
// VK_KHR_fragment_shader_barycentric. The PS (test/glsl/ps_bary_fallback.frag, the interface of a translated one)
// outputs its barycentric input as red, green and blue. One triangle with corners at clip (-1,1), (1,1) and (-1,-1)
// on a 64x64 target cleared to zero:
//   - every pixel inside it holds the barycentrics of its centre, (1-u-v, u, v) with u = (x+1)/2 and v = (1-y)/2,
//     within 2/255 (w is 1, so perspective interpolation is linear), and alpha 255 from the colour's green: the
//     geometry stage copied the VS output at location 0 as well;
//   - pixels outside it stay zero. Pixels within half a pixel of the diagonal are not judged.
// A second PS (test/glsl/ps_bary_fallback_primid.frag) also reads gl_PrimitiveID, which behind a GS must come from
// the GS (VUID-VkGraphicsPipelineCreateInfo-pStages-00740). A quad of two triangles through it: every pixel's
// barycentrics sum to one, and alpha is 0 in primitive 0 and 255 in primitive 1, so the stage forwarded each ID.
// Also refused with E_NOTIMPL: the PS with a line topology, with a PSO's own geometry shader, and with a VS that
// writes Layer (none of which the pass-through stage can serve). Skipped where the GPU lacks geometryShader.
// Shaders: test/glsl/vs_draw.vert, ps_bary_fallback.frag, ps_bary_fallback_primid.frag, gs_double.geom. No game
// data.
// Set LEONIDA_VK_VALIDATION=1 to run under VK_LAYER_KHRONOS_validation.
#include "vk_test_device.h"
#include "d3d12_test_shaders.h"
#include "../shims/d3d12_com.h"
#include "../shims/d3d12_xs_device.h"
#include "../src/shader_library.h"

#include <cmath>
#include <cstring>

#if !defined(LEONIDA_TEST_VS_SPV) || !defined(LEONIDA_TEST_GS_SPV) || !defined(LEONIDA_TEST_PS_BARY_SPV) || \
    !defined(LEONIDA_TEST_PS_BARY_PRIMID_SPV)
#error "LEONIDA_TEST_VS_SPV, LEONIDA_TEST_GS_SPV, LEONIDA_TEST_PS_BARY_SPV and LEONIDA_TEST_PS_BARY_PRIMID_SPV must name the compiled test shaders"
#endif

#define EXPECT(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return 1; } \
} while (0)

using namespace d3d12test;
using XboxHeapHandleFn = std::uint64_t(WINAPI*)(void* heap);  // heap slot 9: handle in RAX

namespace {

struct Vertex {
    float x, y;
    std::uint8_t r, g, b, a;
};

}  // namespace

int main() {
    vktest::Device d;
    EXPECT(d.create());
    if (!d.enabled.geometryShader) {
        std::printf("SKIP: no geometryShader on this GPU\n");
        return 77;  // ctest's SKIP_RETURN_CODE
    }

    const auto vs_program = fssm_program({0x56530001, 0xbf810000});
    const auto vs_layer_program = fssm_program({0x56530002, 0xbf810000});
    const auto gs_program = fssm_program({0x47530001, 0xbf810000});
    const auto ps_program = fssm_program({0x50530001, 0xbf810000});
    const auto ps_primid_program = fssm_program({0x50530002, 0xbf810000});
    const auto vs_container = shader_container(vs_program), vs_layer_container = shader_container(vs_layer_program),
               gs_container = shader_container(gs_program), ps_container = shader_container(ps_program),
               ps_primid_container = shader_container(ps_primid_program);
    const auto library = std::filesystem::temp_directory_path() / "leonida_bary_fallback_test";
    std::filesystem::create_directories(library);
    const auto overwrite = std::filesystem::copy_options::overwrite_existing;
    std::filesystem::copy_file(LEONIDA_TEST_VS_SPV, library / "vs_draw.spv", overwrite);
    std::filesystem::copy_file(LEONIDA_TEST_VS_SPV, library / "vs_layer.spv", overwrite);
    std::filesystem::copy_file(LEONIDA_TEST_GS_SPV, library / "gs_double.spv", overwrite);
    std::filesystem::copy_file(LEONIDA_TEST_PS_BARY_SPV, library / "ps_bary_fallback.spv", overwrite);
    std::filesystem::copy_file(LEONIDA_TEST_PS_BARY_PRIMID_SPV, library / "ps_bary_primid.spv", overwrite);
    {
        const char* signature =
            "\"input_signature\": ["
            "{\"semantic\": \"COLOR\", \"semantic_index\": 0, \"register\": 0, \"mask\": 15, \"rw_mask\": 15, "
            "\"system_value\": 0, \"component_type\": \"float\"}, "
            "{\"semantic\": \"POSITION\", \"semantic_index\": 0, \"register\": 1, \"mask\": 3, \"rw_mask\": 3, "
            "\"system_value\": 0, \"component_type\": \"float\"}]";
        std::ofstream index(library / "index.json", std::ios::binary);
        index << "{\"vs_draw\": {\"stage\": \"vs\", \"fssm_sha256\": \"" << leonida::sha256_hex(vs_program)
              << "\", \"bindings\": [], \"push_constant_bytes\": 144, \"outputs\": [0], " << signature << "},\n"
              << " \"vs_layer\": {\"stage\": \"vs\", \"fssm_sha256\": \"" << leonida::sha256_hex(vs_layer_program)
              << "\", \"bindings\": [], \"push_constant_bytes\": 144, \"outputs\": [0], "
                 "\"capabilities\": [\"Shader\", \"ShaderLayer\"], " << signature << "},\n"
              << " \"gs_double\": {\"stage\": \"gs\", \"fssm_sha256\": \"" << leonida::sha256_hex(gs_program)
              << "\", \"bindings\": [], \"push_constant_bytes\": 144},\n"
              << " \"ps_bary_fallback\": {\"stage\": \"ps\", \"fssm_sha256\": \"" << leonida::sha256_hex(ps_program)
              << "\", \"bindings\": [], \"push_constant_bytes\": 144, \"inputs\": [0], \"outputs\": [0], "
                 "\"bary_fallback_input\": {\"location\": 1}},\n"
              << " \"ps_bary_primid\": {\"stage\": \"ps\", \"fssm_sha256\": \"" << leonida::sha256_hex(ps_primid_program)
              << "\", \"bindings\": [], \"push_constant_bytes\": 144, \"inputs\": [0], \"outputs\": [0], "
                 "\"bary_fallback_input\": {\"location\": 1}}}";
    }

    constexpr UINT size = 64;
    std::vector<std::uint8_t> pixels(size * 256), primid_pixels(size * 256);
    {
        rt::d3d12::VulkanDeviceInfo info{};
        info.get_instance_proc = d.instance_proc;
        info.instance = d.instance;
        info.physical = d.physical;
        info.device = d.device;
        info.queue = d.queue;
        info.queue_family = d.family;
        info.buffer_device_address = true;
        info.features = d.enabled;
        info.subgroup = d.subgroup;
        info.shader_dir = library.string();
        ID3D12Device* inner = nullptr;
        EXPECT(SUCCEEDED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&inner))));
        Ref<ID3D12Device> device;
        EXPECT(SUCCEEDED(rt::d3d12_xs::create_device_proxy(inner, __uuidof(ID3D12Device),
                                                           reinterpret_cast<void**>(&device.p))));
        rt::d3d12::set_device_outer(inner, device.p);

        D3D12_ROOT_PARAMETER param{};
        param.ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
        param.Constants = {0, 0, 2};  // s0, s1: the VS's clip-space offset, zero here
        D3D12_VERSIONED_ROOT_SIGNATURE_DESC root_desc{};
        root_desc.Version = D3D_ROOT_SIGNATURE_VERSION_1_0;
        root_desc.Desc_1_0 = {1, &param, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT};
        const auto root_blob = serialize_root_signature(root_desc);
        EXPECT(!root_blob.empty());
        Ref<ID3D12RootSignature> root;
        EXPECT(SUCCEEDED(device->CreateRootSignature(0, root_blob.data(), root_blob.size(), IID_PPV_ARGS(&root.p))));

        const D3D12_INPUT_ELEMENT_DESC elements[2] = {
            {"POSITION", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 0, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
            {"COLOR", 0, DXGI_FORMAT_R8G8B8A8_UNORM, 0, D3D12_APPEND_ALIGNED_ELEMENT,
             D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0}};
        D3D12_GRAPHICS_PIPELINE_STATE_DESC pso_desc{};
        pso_desc.pRootSignature = root.p;
        pso_desc.VS = {vs_container.data(), vs_container.size()};
        pso_desc.PS = {ps_container.data(), ps_container.size()};
        pso_desc.BlendState.RenderTarget[0].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
        pso_desc.SampleMask = 0xffffffff;
        pso_desc.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
        pso_desc.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
        pso_desc.RasterizerState.DepthClipEnable = TRUE;
        pso_desc.InputLayout = {elements, 2};
        pso_desc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
        pso_desc.NumRenderTargets = 1;
        pso_desc.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
        pso_desc.SampleDesc.Count = 1;
        Ref<ID3D12PipelineState> pso, primid_pso, refused;
        EXPECT(SUCCEEDED(device->CreateGraphicsPipelineState(&pso_desc, IID_PPV_ARGS(&pso.p))));
        D3D12_GRAPHICS_PIPELINE_STATE_DESC primid_desc = pso_desc;
        primid_desc.PS = {ps_primid_container.data(), ps_primid_container.size()};
        EXPECT(SUCCEEDED(device->CreateGraphicsPipelineState(&primid_desc, IID_PPV_ARGS(&primid_pso.p))));
        // What the pass-through stage can't serve.
        D3D12_GRAPHICS_PIPELINE_STATE_DESC lines = pso_desc;
        lines.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_LINE;
        EXPECT(device->CreateGraphicsPipelineState(&lines, IID_PPV_ARGS(&refused.p)) == E_NOTIMPL && !refused.p);
        D3D12_GRAPHICS_PIPELINE_STATE_DESC own_gs = pso_desc;
        own_gs.GS = {gs_container.data(), gs_container.size()};
        EXPECT(device->CreateGraphicsPipelineState(&own_gs, IID_PPV_ARGS(&refused.p)) == E_NOTIMPL && !refused.p);
        D3D12_GRAPHICS_PIPELINE_STATE_DESC layer_vs = pso_desc;
        layer_vs.VS = {vs_layer_container.data(), vs_layer_container.size()};
        EXPECT(device->CreateGraphicsPipelineState(&layer_vs, IID_PPV_ARGS(&refused.p)) == E_NOTIMPL && !refused.p);

        const Vertex vertices[3] = {{-1, 1, 0, 255, 0, 255}, {1, 1, 0, 255, 0, 255}, {-1, -1, 0, 255, 0, 255}};
        const std::uint16_t indices[3] = {0, 1, 2};
        // The quad: the same triangle (primitive 0), then the one below the diagonal (primitive 1).
        const Vertex quad[4] = {{-1, 1, 0, 255, 0, 255}, {1, 1, 0, 255, 0, 255}, {-1, -1, 0, 255, 0, 255},
                                {1, -1, 0, 255, 0, 255}};
        const std::uint16_t quad_indices[6] = {0, 1, 2, 2, 1, 3};

        const D3D12_HEAP_PROPERTIES default_heap{D3D12_HEAP_TYPE_DEFAULT};
        const D3D12_HEAP_PROPERTIES upload_heap{D3D12_HEAP_TYPE_UPLOAD};
        const D3D12_HEAP_PROPERTIES readback_heap{D3D12_HEAP_TYPE_READBACK};
        Ref<ID3D12Resource> upload, readback, target;
        const auto upload_desc = buffer_desc(4096);
        EXPECT(SUCCEEDED(device->CreateCommittedResource(&upload_heap, D3D12_HEAP_FLAG_NONE, &upload_desc,
                                                         D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
                                                         IID_PPV_ARGS(&upload.p))));
        const auto readback_desc = buffer_desc(size * 256);
        EXPECT(SUCCEEDED(device->CreateCommittedResource(&readback_heap, D3D12_HEAP_FLAG_NONE, &readback_desc,
                                                         D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                         IID_PPV_ARGS(&readback.p))));
        D3D12_RESOURCE_DESC rt_desc{};
        rt_desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        rt_desc.Width = rt_desc.Height = size;
        rt_desc.DepthOrArraySize = 1;
        rt_desc.MipLevels = 1;
        rt_desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        rt_desc.SampleDesc.Count = 1;
        rt_desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
        EXPECT(SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &rt_desc,
                                                         D3D12_RESOURCE_STATE_RENDER_TARGET, nullptr,
                                                         IID_PPV_ARGS(&target.p))));
        std::uint8_t* up = nullptr;
        EXPECT(SUCCEEDED(upload->Map(0, nullptr, reinterpret_cast<void**>(&up))));
        std::memcpy(up, vertices, sizeof(vertices));
        std::memcpy(up + 256, indices, sizeof(indices));
        std::memcpy(up + 512, quad, sizeof(quad));
        std::memcpy(up + 768, quad_indices, sizeof(quad_indices));
        const D3D12_GPU_VIRTUAL_ADDRESS upload_va = upload->GetGPUVirtualAddress();

        const D3D12_DESCRIPTOR_HEAP_DESC heap_desc{D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 1, D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0};
        Ref<ID3D12DescriptorHeap> rtv_heap;
        EXPECT(SUCCEEDED(device->CreateDescriptorHeap(&heap_desc, IID_PPV_ARGS(&rtv_heap.p))));
        const D3D12_CPU_DESCRIPTOR_HANDLE rtv{
            SIZE_T(reinterpret_cast<XboxHeapHandleFn>(slot(rtv_heap.p, 9))(rtv_heap.p))};
        device->CreateRenderTargetView(target.p, nullptr, rtv);

        D3D12_COMMAND_QUEUE_DESC queue_desc{};
        Ref<ID3D12CommandQueue> queue;
        Ref<ID3D12CommandAllocator> allocator;
        Ref<ID3D12GraphicsCommandList> list;
        Ref<ID3D12Fence> fence;
        EXPECT(SUCCEEDED(device->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&queue.p))));
        EXPECT(SUCCEEDED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator.p))));
        EXPECT(SUCCEEDED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.p, pso.p,
                                                   IID_PPV_ARGS(&list.p))));
        EXPECT(SUCCEEDED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence.p))));

        const FLOAT zero[4] = {0, 0, 0, 0};
        list->ClearRenderTargetView(rtv, zero, 0, nullptr);
        list->SetGraphicsRootSignature(root.p);
        list->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        const D3D12_VERTEX_BUFFER_VIEW vbv{upload_va, sizeof(vertices), sizeof(Vertex)};
        list->IASetVertexBuffers(0, 1, &vbv);
        const D3D12_INDEX_BUFFER_VIEW ibv{upload_va + 256, sizeof(indices), DXGI_FORMAT_R16_UINT};
        list->IASetIndexBuffer(&ibv);
        const D3D12_VIEWPORT viewport{0, 0, float(size), float(size), 0, 1};
        const D3D12_RECT scissor{0, 0, LONG(size), LONG(size)};
        list->RSSetViewports(1, &viewport);
        list->RSSetScissorRects(1, &scissor);
        list->OMSetRenderTargets(1, &rtv, FALSE, nullptr);
        const float no_offset[2] = {0, 0};
        list->SetGraphicsRoot32BitConstants(0, 2, no_offset, 0);
        list->DrawIndexedInstanced(3, 1, 0, 0, 0);

        D3D12_RESOURCE_BARRIER to_copy = transition(target.p, D3D12_RESOURCE_STATE_RENDER_TARGET,
                                                    D3D12_RESOURCE_STATE_COPY_SOURCE);
        list->ResourceBarrier(1, &to_copy);
        D3D12_TEXTURE_COPY_LOCATION dst{};
        dst.pResource = readback.p;
        dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        dst.PlacedFootprint = {0, {DXGI_FORMAT_R8G8B8A8_UNORM, size, size, 1, 256}};
        D3D12_TEXTURE_COPY_LOCATION src{};
        src.pResource = target.p;
        src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
        EXPECT(list->Close() == S_OK);
        ID3D12CommandList* lists[] = {list.p};
        queue->ExecuteCommandLists(1, lists);
        EXPECT(rt::d3d12::last_execute_result(queue.p) == leonida::ReplayResult::success);
        EXPECT(SUCCEEDED(queue->Signal(fence.p, 1)));
        EXPECT(SUCCEEDED(fence->SetEventOnCompletion(1, nullptr)));
        void* mapped = nullptr;
        EXPECT(SUCCEEDED(readback->Map(0, nullptr, &mapped)));
        std::memcpy(pixels.data(), mapped, pixels.size());

        // Second pass: the quad through the PS that reads gl_PrimitiveID.
        EXPECT(SUCCEEDED(allocator->Reset()) && SUCCEEDED(list->Reset(allocator.p, primid_pso.p)));
        D3D12_RESOURCE_BARRIER to_target = transition(target.p, D3D12_RESOURCE_STATE_COPY_SOURCE,
                                                      D3D12_RESOURCE_STATE_RENDER_TARGET);
        list->ResourceBarrier(1, &to_target);
        list->ClearRenderTargetView(rtv, zero, 0, nullptr);
        list->SetGraphicsRootSignature(root.p);
        list->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        const D3D12_VERTEX_BUFFER_VIEW quad_vbv{upload_va + 512, sizeof(quad), sizeof(Vertex)};
        list->IASetVertexBuffers(0, 1, &quad_vbv);
        const D3D12_INDEX_BUFFER_VIEW quad_ibv{upload_va + 768, sizeof(quad_indices), DXGI_FORMAT_R16_UINT};
        list->IASetIndexBuffer(&quad_ibv);
        list->RSSetViewports(1, &viewport);
        list->RSSetScissorRects(1, &scissor);
        list->OMSetRenderTargets(1, &rtv, FALSE, nullptr);
        list->SetGraphicsRoot32BitConstants(0, 2, no_offset, 0);
        list->DrawIndexedInstanced(6, 1, 0, 0, 0);
        list->ResourceBarrier(1, &to_copy);
        list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
        EXPECT(list->Close() == S_OK);
        queue->ExecuteCommandLists(1, lists);
        EXPECT(rt::d3d12::last_execute_result(queue.p) == leonida::ReplayResult::success);
        EXPECT(SUCCEEDED(queue->Signal(fence.p, 2)) && SUCCEEDED(fence->SetEventOnCompletion(2, nullptr)));
        std::memcpy(primid_pixels.data(), mapped, primid_pixels.size());

        for (auto* r : {&upload, &readback, &target}) r->reset();
        pso.reset();
        primid_pso.reset();
        root.reset();
        rtv_heap.reset();
        list.reset();
        allocator.reset();
        queue.reset();
        fence.reset();
        EXPECT(rt::d3d12_xs::device_proxy_ref_count(device.p) == 1);
    }
    std::filesystem::remove_all(library);

    unsigned inside = 0, outside = 0, mismatches = 0;
    for (UINT row = 0; row < size; ++row) {
        for (UINT col = 0; col < size; ++col) {
            const double x = (col + 0.5) / (size / 2.0) - 1, y = 1 - (row + 0.5) / (size / 2.0);
            const double u = (x + 1) / 2, v = (1 - y) / 2, w = 1 - u - v;
            if (std::fabs(w) < 0.5 / size) continue;  // on the diagonal edge: coverage is the rasterizer's call
            const std::uint8_t* got = pixels.data() + row * 256 + col * 4;
            int want[4] = {0, 0, 0, 0};
            if (w > 0) {
                want[0] = int(std::lround(w * 255));
                want[1] = int(std::lround(u * 255));
                want[2] = int(std::lround(v * 255));
                want[3] = 255;
                ++inside;
            } else {
                ++outside;
            }
            bool ok = true;
            for (int c = 0; c < 4; ++c) ok = ok && std::abs(int(got[c]) - want[c]) <= 2;
            if (ok) continue;
            if (mismatches++ < 6)
                std::fprintf(stderr, "pixel (%u, %u): got %u,%u,%u,%u want %d,%d,%d,%d\n", col, row, got[0], got[1],
                             got[2], got[3], want[0], want[1], want[2], want[3]);
        }
    }
    EXPECT(!mismatches);
    // The quad: alpha says which primitive the fragment's PrimitiveId named, and the barycentrics sum to one.
    unsigned primitive_pixels[2] = {0, 0};
    for (UINT row = 0; row < size; ++row) {
        for (UINT col = 0; col < size; ++col) {
            const double x = (col + 0.5) / (size / 2.0) - 1, y = 1 - (row + 0.5) / (size / 2.0);
            const double w = 1 - (x + 1) / 2 - (1 - y) / 2;  // > 0 in primitive 0, < 0 in primitive 1
            if (std::fabs(w) < 0.5 / size) continue;
            const std::uint8_t* got = primid_pixels.data() + row * 256 + col * 4;
            const int primitive = w > 0 ? 0 : 1;
            const int sum = got[0] + got[1] + got[2];
            const bool ok = got[3] == (primitive ? 255 : 0) && std::abs(sum - 255) <= 3;
            ++primitive_pixels[primitive];
            if (ok) continue;
            if (mismatches++ < 6)
                std::fprintf(stderr, "primitive %d pixel (%u, %u): got %u,%u,%u,%u\n", primitive, col, row, got[0],
                             got[1], got[2], got[3]);
        }
    }
    EXPECT(!mismatches);
    if (d.validation_errors()) {
        std::fprintf(stderr, "%u validation errors\n", d.validation_errors());
        return 1;
    }
    std::printf("bary_fallback: the layer drew a --bary-fallback PS through its pass-through geometry shader, %u "
                "pixels inside the triangle hold their barycentrics and location 0, %u outside stayed clear; a line "
                "topology, a PSO's own GS and a VS writing Layer refused; a PS reading gl_PrimitiveID got 0 on %u "
                "pixels and 1 on %u through the stage\n",
                inside, outside, primitive_pixels[0], primitive_pixels[1]);
    return 0;
}
