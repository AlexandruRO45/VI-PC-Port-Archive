// --vs-compute on the GPU (issue #94, atom 3): a tier-C wave64 vertex shader drawn through a D3D12 graphics PSO.
// test/shaders/vs_w64_c_wave_color.asm colours each full-target triangle by the number of active lanes of its vertex's
// 64-lane wave / 64; ps_flat_color.asm writes that value. The build translates both with --vs-compute
// (LEONIDA_VS_COMPUTE=1 in test/shaders/build.py), which adds vs_w64_c_wave_color.vs_compute.spv (the VS as an emulated
// compute prepass) and .vs_compute_draw.spv (the VS that reads the slots back, tools/vs_compute_vs.py).
//
// DrawInstanced(48, 2, 0, 0): every triangle covers the whole 64x64 target and the last one drawn (vertices 45..47 of
// instance 1) decides every pixel. Its wave has 48 active lanes, so the hardware gives 48 / 64 = 0.75. A device that
// cannot give the VS 64-wide subgroups gets the prepass (d3d12_vulkan_device.cpp, build_vs_compute) and must give 0.75
// too; without it, a 32-wide subgroup puts vertices 45..47 in a half of 16 lanes (0.25; NVIDIA's batching even gives
// 0.09375, 6 lanes). AMD's Windows driver on an RDNA2 iGPU reports a default subgroup size of 64 but runs the VS
// 32-wide (0.5 without the prepass), so the layer uses the prepass wherever 64 cannot be required for the vertex
// stage. Every pixel is compared bit for bit.
#include "vk_test_device.h"
#include "d3d12_test_shaders.h"
#include "../shims/d3d12_com.h"
#include "../shims/d3d12_xs_device.h"
#include "../src/shader_library.h"

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <vector>

#ifndef LEONIDA_TEST_VSC_DIR
#error "LEONIDA_TEST_VSC_DIR must name the folder build.py wrote (spirv/index.json, fssm/*.fssm)"
#endif

#define EXPECT(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return 1; } \
} while (0)

using namespace d3d12test;
using XboxHeapHandleFn = std::uint64_t(WINAPI*)(void* heap);  // heap slot 9: handle in RAX

namespace {

std::vector<std::uint8_t> read_file(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

using Pixel = std::array<float, 4>;
constexpr UINT target_extent = 64;
constexpr UINT pitch = target_extent * sizeof(Pixel);  // 1024, a multiple of 256

}  // namespace

int main() {
    vktest::Device d;
    const auto device_status = d.initialize();
    if (device_status == vktest::DeviceStatus::unavailable) return 77;  // no Vulkan GPU: ctest skips
    EXPECT(device_status == vktest::DeviceStatus::ready);

    const std::filesystem::path root_dir = LEONIDA_TEST_VSC_DIR;
    const auto vs_program = read_file(root_dir / "fssm" / "vs_w64_c_wave_color.fssm");
    const auto ps_program = read_file(root_dir / "fssm" / "ps_flat_color.fssm");
    EXPECT(!vs_program.empty() && !ps_program.empty());
    leonida::ShaderLibrary library;
    EXPECT(library.load((root_dir / "spirv").string()));
    const leonida::ShaderEntry* vs_entry = library.find("vs", leonida::sha256_hex(vs_program));
    EXPECT(vs_entry && library.find("ps", leonida::sha256_hex(ps_program)));
    EXPECT(vs_entry->wave64 && vs_entry->vs_compute_stride == 32);   // position + param0
    EXPECT(!library.spirv(*vs_entry, ".vs_compute").empty() && !library.spirv(*vs_entry, ".vs_compute_draw").empty());
    const auto vs_container = shader_container(vs_program), ps_container = shader_container(ps_program);

    std::vector<Pixel> pixels(target_extent * target_extent);
    {
        rt::d3d12::VulkanDeviceInfo info{};
        info.get_instance_proc = d.instance_proc;
        info.instance = d.instance;
        info.physical = d.physical;
        info.device = d.device;
        info.queue = d.queue;
        info.queue_family = d.family;
        info.buffer_device_address = true;
        info.shader_dir = (root_dir / "spirv").string();
        ID3D12Device* inner = nullptr;
        EXPECT(SUCCEEDED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&inner))));
        Ref<ID3D12Device> device;
        EXPECT(SUCCEEDED(rt::d3d12_xs::create_device_proxy(inner, __uuidof(ID3D12Device),
                                                           reinterpret_cast<void**>(&device.p))));
        rt::d3d12::set_device_outer(inner, device.p);

        D3D12_VERSIONED_ROOT_SIGNATURE_DESC root_desc{};
        root_desc.Version = D3D_ROOT_SIGNATURE_VERSION_1_0;
        root_desc.Desc_1_0 = {0, nullptr, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_NONE};
        const auto root_blob = serialize_root_signature(root_desc);
        EXPECT(!root_blob.empty());
        Ref<ID3D12RootSignature> root;
        EXPECT(SUCCEEDED(device->CreateRootSignature(0, root_blob.data(), root_blob.size(), IID_PPV_ARGS(&root.p))));

        D3D12_GRAPHICS_PIPELINE_STATE_DESC pso_desc{};
        pso_desc.pRootSignature = root.p;
        pso_desc.VS = {vs_container.data(), vs_container.size()};
        pso_desc.PS = {ps_container.data(), ps_container.size()};
        pso_desc.BlendState.RenderTarget[0].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
        pso_desc.SampleMask = 0xffffffff;
        pso_desc.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
        pso_desc.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
        pso_desc.RasterizerState.DepthClipEnable = TRUE;
        pso_desc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
        pso_desc.NumRenderTargets = 1;
        pso_desc.RTVFormats[0] = DXGI_FORMAT_R32G32B32A32_FLOAT;
        pso_desc.SampleDesc.Count = 1;
        Ref<ID3D12PipelineState> pso;
        EXPECT(SUCCEEDED(device->CreateGraphicsPipelineState(&pso_desc, IID_PPV_ARGS(&pso.p))));

        const D3D12_HEAP_PROPERTIES default_heap{D3D12_HEAP_TYPE_DEFAULT};
        const D3D12_HEAP_PROPERTIES readback_heap{D3D12_HEAP_TYPE_READBACK};
        Ref<ID3D12Resource> readback, target;
        const auto readback_desc = buffer_desc(pitch * target_extent);
        EXPECT(SUCCEEDED(device->CreateCommittedResource(&readback_heap, D3D12_HEAP_FLAG_NONE, &readback_desc,
                                                         D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                         IID_PPV_ARGS(&readback.p))));
        D3D12_RESOURCE_DESC rt_desc{};
        rt_desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        rt_desc.Width = rt_desc.Height = target_extent;
        rt_desc.DepthOrArraySize = 1;
        rt_desc.MipLevels = 1;
        rt_desc.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
        rt_desc.SampleDesc.Count = 1;
        rt_desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
        EXPECT(SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &rt_desc,
                                                         D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&target.p))));

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

        D3D12_RESOURCE_BARRIER to_target = transition(target.p, D3D12_RESOURCE_STATE_COMMON,
                                                      D3D12_RESOURCE_STATE_RENDER_TARGET);
        list->ResourceBarrier(1, &to_target);
        const FLOAT clear[4] = {0.5f, 0.125f, 0.25f, 0.5f};
        list->ClearRenderTargetView(rtv, clear, 0, nullptr);
        list->SetGraphicsRootSignature(root.p);
        list->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        const D3D12_VIEWPORT viewport{0, 0, float(target_extent), float(target_extent), 0, 1};
        const D3D12_RECT scissor{0, 0, LONG(target_extent), LONG(target_extent)};
        list->RSSetViewports(1, &viewport);
        list->RSSetScissorRects(1, &scissor);
        list->OMSetRenderTargets(1, &rtv, FALSE, nullptr);
        list->DrawInstanced(48, 2, 0, 0);

        D3D12_RESOURCE_BARRIER to_copy = transition(target.p, D3D12_RESOURCE_STATE_RENDER_TARGET,
                                                    D3D12_RESOURCE_STATE_COPY_SOURCE);
        list->ResourceBarrier(1, &to_copy);
        D3D12_TEXTURE_COPY_LOCATION dst{};
        dst.pResource = readback.p;
        dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        dst.PlacedFootprint = {0, {DXGI_FORMAT_R32G32B32A32_FLOAT, target_extent, target_extent, 1, pitch}};
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
        std::memcpy(pixels.data(), mapped, pixels.size() * sizeof(Pixel));
        readback->Unmap(0, nullptr);

        for (auto* r : {&readback, &target}) r->reset();
        pso.reset();
        root.reset();
        rtv_heap.reset();
        list.reset();
        allocator.reset();
        queue.reset();
        fence.reset();
        EXPECT(rt::d3d12_xs::device_proxy_ref_count(device.p) == 1);
    }

    const Pixel want = {0.75f, 0.75f, 0.75f, 0.75f};
    unsigned mismatches = 0;
    for (UINT i = 0; i < target_extent * target_extent; ++i) {
        if (std::memcmp(&pixels[i], &want, sizeof(Pixel)) == 0) continue;
        if (mismatches++ < 8)
            std::fprintf(stderr, "pixel (%u, %u): got %g,%g,%g,%g want 0.75 (48 of the wave's 64 lanes)\n",
                         i % target_extent, i / target_extent, pixels[i][0], pixels[i][1], pixels[i][2], pixels[i][3]);
    }
    EXPECT(!mismatches);
    if (d.validation_errors()) {
        std::fprintf(stderr, "%u validation errors\n", d.validation_errors());
        return 1;
    }
    std::printf("--vs-compute: a tier-C wave64 VS drew %u of %u pixels with its 64-lane wave's lane count (0.75)\n",
                target_extent * target_extent, target_extent * target_extent);
    return 0;
}
