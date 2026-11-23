// Host-parity and Vulkan-validation tests for RSSetScissorRects and closed-list RSSetScissorRects / RSSetViewports.
//
// Part 1: Compares Close() and closed-list no-op/Reset/Execute behavior on the layer against live Windows 11 WARP
//         and live native hardware D3D12 (recreating a host device if GetDeviceRemovedReason() != S_OK).
// Part 2: Verifies the layer's null-pointer guard on RSSetScissorRects(1, nullptr) (fails Close() with E_INVALIDARG
//         without crashing; native D3D12 access-violates on this call so it is not issued to host devices).
// Part 3: Executes real GPU draws on the layer with negative, inverted, empty, and extreme INT_MIN/INT_MAX scissor
//         rectangles and post-Close RSSetScissorRects/RSSetViewports calls, verifying 0 Vulkan validation errors
//         (VUID-VkRect2D-offset-00617, VUID-vkCmdSetScissor-x-00595/00596) and exact pixel clipping via readback.
// No game data. Set LEONIDA_VK_VALIDATION=1 to run under VK_LAYER_KHRONOS_validation.
#include "shader_fixture_path.h"
#include "test_temp_dir.h"
#include "vk_test_device.h"
#include "d3d12_test_shaders.h"
#include "host_wine.h"
#include "../shims/d3d12_com.h"
#include "../src/shader_library.h"

#include <dxgi1_4.h>

#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <functional>
#include <string>
#include <vector>

#if !defined(LEONIDA_TEST_VS_FULLSCREEN_SPV) || !defined(LEONIDA_TEST_PS_SPV)
#error "LEONIDA_TEST_VS_FULLSCREEN_SPV and LEONIDA_TEST_PS_SPV must name the compiled test shaders"
#endif

#define CHECK(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return false; } \
} while (0)

#define REQUIRE(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return 1; } \
} while (0)

using namespace d3d12test;

namespace {

struct ParityCase {
    const char* name;
    bool expect_ok;
    bool closed_list_case;
    std::function<void(ID3D12GraphicsCommandList*)> record;
};

std::vector<ParityCase> parity_cases() {
    return {
        {"RSSetScissorRects(0, nullptr)", true, false,
         [](ID3D12GraphicsCommandList* l) { l->RSSetScissorRects(0, nullptr); }},
        {"RSSetScissorRects(0, &rc)", true, false,
         [](ID3D12GraphicsCommandList* l) {
             const D3D12_RECT r{0, 0, 10, 10};
             l->RSSetScissorRects(0, &r);
         }},
        {"RSSetScissorRects(16, rects)", true, false,
         [](ID3D12GraphicsCommandList* l) {
             D3D12_RECT r[16]{};
             for (auto& x : r) x = {0, 0, 10, 10};
             l->RSSetScissorRects(16, r);
         }},
        {"RSSetScissorRects(17, rects)", true, false,
         [](ID3D12GraphicsCommandList* l) {
             D3D12_RECT r[17]{};
             for (auto& x : r) x = {0, 0, 10, 10};
             l->RSSetScissorRects(17, r);
         }},
        {"RSSetScissorRects(1, {-1, 0, 10, 10})", true, false,
         [](ID3D12GraphicsCommandList* l) {
             const D3D12_RECT r{-1, 0, 10, 10};
             l->RSSetScissorRects(1, &r);
         }},
        {"RSSetScissorRects(1, {0, -1, 10, 10})", true, false,
         [](ID3D12GraphicsCommandList* l) {
             const D3D12_RECT r{0, -1, 10, 10};
             l->RSSetScissorRects(1, &r);
         }},
        {"RSSetScissorRects(1, {-10, -10, -5, -5})", true, false,
         [](ID3D12GraphicsCommandList* l) {
             const D3D12_RECT r{-10, -10, -5, -5};
             l->RSSetScissorRects(1, &r);
         }},
        {"RSSetScissorRects(1, {10, 0, 5, 10})", true, false,
         [](ID3D12GraphicsCommandList* l) {
             const D3D12_RECT r{10, 0, 5, 10};
             l->RSSetScissorRects(1, &r);
         }},
        {"RSSetScissorRects(1, {0, 10, 10, 5})", true, false,
         [](ID3D12GraphicsCommandList* l) {
             const D3D12_RECT r{0, 10, 10, 5};
             l->RSSetScissorRects(1, &r);
         }},
        {"RSSetScissorRects(2, {valid, negative})", true, false,
         [](ID3D12GraphicsCommandList* l) {
             const D3D12_RECT r[2] = {{0, 0, 10, 10}, {-1, 0, 10, 10}};
             l->RSSetScissorRects(2, r);
         }},
        {"RSSetScissorRects(2, {valid, inverted})", true, false,
         [](ID3D12GraphicsCommandList* l) {
             const D3D12_RECT r[2] = {{0, 0, 10, 10}, {10, 0, 5, 10}};
             l->RSSetScissorRects(2, r);
         }},
        {"RSSetScissorRects(1, {0, 0, 0, 0})", true, false,
         [](ID3D12GraphicsCommandList* l) {
             const D3D12_RECT r{0, 0, 0, 0};
             l->RSSetScissorRects(1, &r);
         }},
        {"RSSetScissorRects(1, {INT_MIN, INT_MIN, INT_MAX, INT_MAX})", true, false,
         [](ID3D12GraphicsCommandList* l) {
             const D3D12_RECT r{INT_MIN, INT_MIN, INT_MAX, INT_MAX};
             l->RSSetScissorRects(1, &r);
         }},
        {"RSSetScissorRects(1, {INT_MAX, INT_MAX, INT_MIN, INT_MIN})", true, false,
         [](ID3D12GraphicsCommandList* l) {
             const D3D12_RECT r{INT_MAX, INT_MAX, INT_MIN, INT_MIN};
             l->RSSetScissorRects(1, &r);
         }},
        // Closed-list no-op checks (valid and invalid arguments after Close must not crash or corrupt state).
        {"closed DIRECT: RSSetScissorRects(1, valid)", true, true,
         [](ID3D12GraphicsCommandList* l) {
             const D3D12_RECT r{0, 0, 10, 10};
             l->RSSetScissorRects(1, &r);
         }},
        {"closed DIRECT: RSSetScissorRects(17, rects)", true, true,
         [](ID3D12GraphicsCommandList* l) {
             const D3D12_RECT r[17]{};
             l->RSSetScissorRects(17, r);
         }},
        {"closed DIRECT: RSSetScissorRects(1, negative)", true, true,
         [](ID3D12GraphicsCommandList* l) {
             const D3D12_RECT r{-5, -5, 10, 10};
             l->RSSetScissorRects(1, &r);
         }},
        {"closed DIRECT: RSSetScissorRects(1, nullptr)", true, true,
         [](ID3D12GraphicsCommandList* l) { l->RSSetScissorRects(1, nullptr); }},
        {"closed DIRECT: RSSetViewports(1, valid)", true, true,
         [](ID3D12GraphicsCommandList* l) {
             const D3D12_VIEWPORT v{0, 0, 8, 8, 0, 1};
             l->RSSetViewports(1, &v);
         }},
        {"closed DIRECT: RSSetViewports(17, viewports)", true, true,
         [](ID3D12GraphicsCommandList* l) {
             const D3D12_VIEWPORT v[17]{};
             l->RSSetViewports(17, v);
         }},
        {"closed DIRECT: RSSetViewports(1, nullptr)", true, true,
         [](ID3D12GraphicsCommandList* l) { l->RSSetViewports(1, nullptr); }},
    };
}

bool run_parity_suite(const char* label, ID3D12Device* initial_device,
                      const std::function<bool(Ref<ID3D12Device>&)>& recreate,
                      const std::vector<ParityCase>& table, std::vector<bool>& results) {
    Ref<ID3D12Device> owned;
    ID3D12Device* dev = initial_device;
    results.assign(table.size(), false);
    for (std::size_t i = 0; i < table.size(); ++i) {
        if (dev->GetDeviceRemovedReason() != S_OK) {
            CHECK(recreate && recreate(owned));
            dev = owned.p;
        }
        Ref<ID3D12CommandAllocator> alloc;
        Ref<ID3D12GraphicsCommandList> list;
        CHECK(SUCCEEDED(dev->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&alloc.p))));
        CHECK(SUCCEEDED(dev->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, alloc.p, nullptr,
                                               IID_PPV_ARGS(&list.p))));
        if (table[i].closed_list_case) {
            const HRESULT close1 = list->Close();
            table[i].record(list.p);
            const HRESULT reset_hr = list->Reset(alloc.p, nullptr);
            const HRESULT close2 = SUCCEEDED(reset_hr) ? list->Close() : E_FAIL;
            results[i] = SUCCEEDED(close1) && SUCCEEDED(reset_hr) && SUCCEEDED(close2);
        } else {
            table[i].record(list.p);
            results[i] = SUCCEEDED(list->Close());
        }
        if (dev->GetDeviceRemovedReason() != S_OK) {
            std::fprintf(stderr, "%s: device removed on case '%s' (0x%08lx)\n", label, table[i].name,
                         static_cast<unsigned long>(dev->GetDeviceRemovedReason()));
            CHECK(recreate && recreate(owned));
            dev = owned.p;
        }
    }
    return true;
}

bool wait_queue(ID3D12Device* device, ID3D12CommandQueue* queue, UINT64& fence_value, ID3D12Fence* fence) {
    ++fence_value;
    CHECK(SUCCEEDED(queue->Signal(fence, fence_value)));
    if (fence->GetCompletedValue() < fence_value) {
        HANDLE ev = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        CHECK(ev != nullptr);
        CHECK(SUCCEEDED(fence->SetEventOnCompletion(fence_value, ev)));
        const DWORD wait = WaitForSingleObject(ev, 10000);
        CloseHandle(ev);
        CHECK(wait == WAIT_OBJECT_0);
    }
    CHECK(device->GetDeviceRemovedReason() == S_OK);
    return true;
}

// Executes draws on the layer with edge-case scissor rects and verifies both pixel clipping and 0 VUIDs.
bool test_gpu_scissor_draws(vktest::Device& vk) {
    const auto vs_program = fssm_program({0x56530101, 0xbf810000});
    const auto ps_program = fssm_program({0x50530101, 0xbf810000});
    const auto vs_container = shader_container(vs_program);
    const auto ps_container = shader_container(ps_program);

    const testfiles::ScratchDir scratch("leonida_scissor_val_test");
    const auto& library = scratch.path();
    CHECK(testfiles::copy_file(vktest::shader_fixture_path(LEONIDA_TEST_VS_FULLSCREEN_SPV),
                               library / "vs_fullscreen.spv"));
    CHECK(testfiles::copy_file(vktest::shader_fixture_path(LEONIDA_TEST_PS_SPV), library / "ps_draw.spv"));
    {
        std::ofstream index(library / "index.json", std::ios::binary);
        index << "{\"vs_fullscreen\": {\"stage\": \"vs\", \"fssm_sha256\": \"" << leonida::sha256_hex(vs_program)
              << "\", \"bindings\": [], \"push_constant_bytes\": 144, \"input_signature\": []},\n"
                 " \"ps_draw\": {\"stage\": \"ps\", \"fssm_sha256\": \"" << leonida::sha256_hex(ps_program)
              << "\", \"bindings\": [], \"push_constant_bytes\": 144, \"input_signature\": []}}\n";
        CHECK(index.good());
    }

    rt::d3d12::VulkanDeviceInfo info{};
    info.get_instance_proc = vk.instance_proc;
    info.instance = vk.instance;
    info.physical = vk.physical;
    info.device = vk.device;
    info.queue = vk.queue;
    info.queue_family = vk.family;
    info.buffer_device_address = true;
    info.shader_dir = library.string();

    Ref<ID3D12Device> device;
    CHECK(SUCCEEDED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&device.p))));

    D3D12_COMMAND_QUEUE_DESC qdesc{};
    qdesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    Ref<ID3D12CommandQueue> queue;
    CHECK(SUCCEEDED(device->CreateCommandQueue(&qdesc, IID_PPV_ARGS(&queue.p))));

    Ref<ID3D12Fence> fence;
    UINT64 fence_value = 0;
    CHECK(SUCCEEDED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence.p))));

    // Root signature: 2 root constants (s0..s1) + root CBV (s2..s3) matching ps_draw.frag.
    D3D12_ROOT_PARAMETER1 params[2]{};
    params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    params[0].Constants = {0, 0, 2};
    params[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    params[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
    params[1].Descriptor = {1, 0, D3D12_ROOT_DESCRIPTOR_FLAG_DATA_STATIC};
    params[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;

    D3D12_VERSIONED_ROOT_SIGNATURE_DESC rs_desc{};
    rs_desc.Version = D3D_ROOT_SIGNATURE_VERSION_1_1;
    rs_desc.Desc_1_1.NumParameters = 2;
    rs_desc.Desc_1_1.pParameters = params;
    const auto rs_blob = serialize_root_signature(rs_desc);
    CHECK(!rs_blob.empty());
    Ref<ID3D12RootSignature> root_sig;
    CHECK(SUCCEEDED(device->CreateRootSignature(0, rs_blob.data(), rs_blob.size(), IID_PPV_ARGS(&root_sig.p))));

    D3D12_GRAPHICS_PIPELINE_STATE_DESC pso_desc{};
    pso_desc.pRootSignature = root_sig.p;
    pso_desc.VS = {vs_container.data(), vs_container.size()};
    pso_desc.PS = {ps_container.data(), ps_container.size()};
    pso_desc.BlendState.RenderTarget[0].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
    pso_desc.SampleMask = UINT_MAX;
    pso_desc.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
    pso_desc.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
    pso_desc.RasterizerState.DepthClipEnable = TRUE;
    pso_desc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    pso_desc.NumRenderTargets = 1;
    pso_desc.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
    pso_desc.SampleDesc.Count = 1;
    Ref<ID3D12PipelineState> pso;
    CHECK(SUCCEEDED(device->CreateGraphicsPipelineState(&pso_desc, IID_PPV_ARGS(&pso.p))));

    constexpr UINT kW = 16, kH = 16, kRowPitch = 256;
    const D3D12_HEAP_PROPERTIES default_heap{D3D12_HEAP_TYPE_DEFAULT};
    const D3D12_HEAP_PROPERTIES upload_heap{D3D12_HEAP_TYPE_UPLOAD};
    const D3D12_HEAP_PROPERTIES readback_heap{D3D12_HEAP_TYPE_READBACK};

    D3D12_RESOURCE_DESC rt_desc{};
    rt_desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    rt_desc.Width = kW;
    rt_desc.Height = kH;
    rt_desc.DepthOrArraySize = 1;
    rt_desc.MipLevels = 1;
    rt_desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    rt_desc.SampleDesc.Count = 1;
    rt_desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
    Ref<ID3D12Resource> target;
    CHECK(SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &rt_desc,
                                                    D3D12_RESOURCE_STATE_RENDER_TARGET, nullptr,
                                                    IID_PPV_ARGS(&target.p))));

    const auto cbv_desc = buffer_desc(256);
    Ref<ID3D12Resource> cbv;
    CHECK(SUCCEEDED(device->CreateCommittedResource(&upload_heap, D3D12_HEAP_FLAG_NONE, &cbv_desc,
                                                    D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&cbv.p))));
    {
        void* mapped = nullptr;
        CHECK(SUCCEEDED(cbv->Map(0, nullptr, &mapped)));
        const float tint[3] = {1.0f, 0.0f, 0.0f};  // pure red output
        std::memcpy(mapped, tint, sizeof(tint));
        cbv->Unmap(0, nullptr);
    }

    const auto rb_desc = buffer_desc(UINT64(kRowPitch) * kH);
    Ref<ID3D12Resource> readback;
    CHECK(SUCCEEDED(device->CreateCommittedResource(&readback_heap, D3D12_HEAP_FLAG_NONE, &rb_desc,
                                                    D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                    IID_PPV_ARGS(&readback.p))));

    const D3D12_DESCRIPTOR_HEAP_DESC rtv_desc{D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 1, D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0};
    Ref<ID3D12DescriptorHeap> rtv_heap;
    CHECK(SUCCEEDED(device->CreateDescriptorHeap(&rtv_desc, IID_PPV_ARGS(&rtv_heap.p))));
    const D3D12_CPU_DESCRIPTOR_HANDLE rtv = rtv_heap->GetCPUDescriptorHandleForHeapStart();
    device->CreateRenderTargetView(target.p, nullptr, rtv);

    Ref<ID3D12CommandAllocator> alloc;
    Ref<ID3D12GraphicsCommandList> list;
    CHECK(SUCCEEDED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&alloc.p))));
    CHECK(SUCCEEDED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, alloc.p, nullptr,
                                              IID_PPV_ARGS(&list.p))));
    CHECK(SUCCEEDED(list->Close()));

    auto draw_and_count_red = [&](D3D12_RECT rect, bool post_close_tamper, UINT& red_pixels) -> bool {
        CHECK(SUCCEEDED(alloc->Reset()));
        CHECK(SUCCEEDED(list->Reset(alloc.p, pso.p)));
        const FLOAT blue[4] = {0.0f, 0.0f, 1.0f, 1.0f};
        list->ClearRenderTargetView(rtv, blue, 0, nullptr);
        list->OMSetRenderTargets(1, &rtv, FALSE, nullptr);
        const D3D12_VIEWPORT vp{0, 0, float(kW), float(kH), 0, 1};
        list->RSSetViewports(1, &vp);
        list->RSSetScissorRects(1, &rect);
        list->SetGraphicsRootSignature(root_sig.p);
        list->SetGraphicsRootConstantBufferView(1, cbv->GetGPUVirtualAddress());
        list->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        list->DrawInstanced(3, 1, 0, 0);

        const auto to_src = transition(target.p, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_COPY_SOURCE);
        list->ResourceBarrier(1, &to_src);
        D3D12_TEXTURE_COPY_LOCATION dst{};
        dst.pResource = readback.p;
        dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        dst.PlacedFootprint.Footprint = {DXGI_FORMAT_R8G8B8A8_UNORM, kW, kH, 1, kRowPitch};
        D3D12_TEXTURE_COPY_LOCATION src{};
        src.pResource = target.p;
        src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        src.SubresourceIndex = 0;
        list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
        const auto to_rt = transition(target.p, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_RENDER_TARGET);
        list->ResourceBarrier(1, &to_rt);
        CHECK(SUCCEEDED(list->Close()));

        if (post_close_tamper) {
            // Post-Close calls must be no-ops and must not poison or alter the already-closed list.
            list->RSSetScissorRects(0, nullptr);
            list->RSSetScissorRects(1, nullptr);
            list->RSSetViewports(0, nullptr);
            list->RSSetViewports(1, nullptr);
            const D3D12_VIEWPORT v17[17]{};
            list->RSSetViewports(17, v17);
        }

        ID3D12CommandList* lists[] = {list.p};
        queue->ExecuteCommandLists(1, lists);
        CHECK(wait_queue(device.p, queue.p, fence_value, fence.p));

        void* mapped = nullptr;
        CHECK(SUCCEEDED(readback->Map(0, nullptr, &mapped)));
        const auto* bytes = static_cast<const std::uint8_t*>(mapped);
        red_pixels = 0;
        for (UINT y = 0; y < kH; ++y) {
            for (UINT x = 0; x < kW; ++x) {
                const std::uint8_t* px = bytes + y * kRowPitch + x * 4;
                if (px[0] > 200 && px[2] < 50) ++red_pixels;
            }
        }
        readback->Unmap(0, nullptr);
        return true;
    };

    UINT red = 0;
    // Negative left/top clamps to 0 without VUID-VkRect2D-offset-00617: draws [0..8) x [0..8) = 64 pixels.
    CHECK(draw_and_count_red({-8, -8, 8, 8}, false, red));
    CHECK(red == 64);

    // All-negative rect clips every fragment -> 0 red pixels, 0 VUIDs.
    CHECK(draw_and_count_red({-16, -16, -4, -4}, false, red));
    CHECK(red == 0);

    // Inverted rect clips every fragment -> 0 red pixels, 0 VUIDs.
    CHECK(draw_and_count_red({10, 0, 5, 10}, false, red));
    CHECK(red == 0);

    // Empty zero-area rect clips every fragment -> 0 red pixels, 0 VUIDs.
    CHECK(draw_and_count_red({0, 0, 0, 0}, false, red));
    CHECK(red == 0);

    // Extreme INT_MIN/INT_MAX rects must not overflow int32 subtraction or violate vkCmdSetScissor VUIDs.
    CHECK(draw_and_count_red({INT_MIN, INT_MIN, INT_MAX, INT_MAX}, false, red));
    CHECK(red == kW * kH);
    CHECK(draw_and_count_red({INT_MAX, INT_MAX, INT_MIN, INT_MIN}, false, red));
    CHECK(red == 0);

    // Closed-list RSSetScissorRects / RSSetViewports calls after Close() must not poison ExecuteCommandLists.
    CHECK(draw_and_count_red({0, 0, 8, 8}, true, red));
    CHECK(red == 64);

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
    const unsigned errors_before = vk.validation_errors();

    const auto table = parity_cases();
    std::vector<bool> layer_ok;
    {
        rt::d3d12::VulkanDeviceInfo info{};
        info.get_instance_proc = vk.instance_proc;
        info.instance = vk.instance;
        info.physical = vk.physical;
        info.device = vk.device;
        info.queue = vk.queue;
        info.queue_family = vk.family;
        info.buffer_device_address = true;
        Ref<ID3D12Device> ours;
        REQUIRE(SUCCEEDED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&ours.p))));

        REQUIRE(run_parity_suite("layer", ours.p, nullptr, table, layer_ok));

        // Layer-only safety check: RSSetScissorRects(1, nullptr) on an open list must not crash and must fail Close().
        Ref<ID3D12CommandAllocator> alloc;
        Ref<ID3D12GraphicsCommandList> list;
        REQUIRE(SUCCEEDED(ours->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&alloc.p))));
        REQUIRE(SUCCEEDED(ours->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, alloc.p, nullptr,
                                                  IID_PPV_ARGS(&list.p))));
        list->RSSetScissorRects(1, nullptr);
        REQUIRE(list->Close() == E_INVALIDARG);
    }

    // Compare against live WARP and live native hardware D3D12 when available on Windows.
    bool warp_ran = false;
    bool hw_ran = false;
    std::vector<bool> warp_ok, hw_ok;
    if (!hostwine::host_is_wine()) {
        Ref<IDXGIFactory4> factory;
        if (SUCCEEDED(CreateDXGIFactory1(IID_PPV_ARGS(&factory.p)))) {
            Ref<IDXGIAdapter> warp_adapter;
            if (SUCCEEDED(factory->EnumWarpAdapter(IID_PPV_ARGS(&warp_adapter.p)))) {
                auto make_warp = [&](Ref<ID3D12Device>& out) {
                    out.reset();
                    return SUCCEEDED(D3D12CreateDevice(warp_adapter.p, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&out.p)));
                };
                Ref<ID3D12Device> warp_dev;
                if (make_warp(warp_dev)) {
                    REQUIRE(run_parity_suite("WARP", warp_dev.p, make_warp, table, warp_ok));
                    warp_ran = true;
                }
            }

            // Pick the first hardware D3D12 adapter (e.g. NVIDIA GeForce RTX 5090).
            for (UINT i = 0;; ++i) {
                Ref<IDXGIAdapter1> adapter;
                if (factory->EnumAdapters1(i, &adapter.p) == DXGI_ERROR_NOT_FOUND) break;
                DXGI_ADAPTER_DESC1 desc{};
                if (FAILED(adapter->GetDesc1(&desc))) continue;
                if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) continue;
                auto make_hw = [a = adapter.p](Ref<ID3D12Device>& out) {
                    out.reset();
                    return SUCCEEDED(D3D12CreateDevice(a, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&out.p)));
                };
                Ref<ID3D12Device> hw_dev;
                if (make_hw(hw_dev)) {
                    REQUIRE(run_parity_suite("HW-D3D12", hw_dev.p, make_hw, table, hw_ok));
                    hw_ran = true;
                    break;
                }
            }
        }
    }

    unsigned failures = 0;
    for (std::size_t i = 0; i < table.size(); ++i) {
        const bool expected = warp_ran ? warp_ok[i] : table[i].expect_ok;
        if (layer_ok[i] != expected) {
            std::fprintf(stderr, "FAIL '%s': layer=%s, expected=%s\n", table[i].name,
                         layer_ok[i] ? "S_OK" : "FAIL", expected ? "S_OK" : "FAIL");
            ++failures;
        }
        if (hw_ran && layer_ok[i] != hw_ok[i]) {
            std::fprintf(stderr, "FAIL '%s': layer=%s, hw-d3d12=%s\n", table[i].name,
                         layer_ok[i] ? "S_OK" : "FAIL", hw_ok[i] ? "S_OK" : "FAIL");
            ++failures;
        }
    }

    REQUIRE(test_gpu_scissor_draws(vk));

    const unsigned validation_errors = vk.validation_errors() - errors_before;
    if (validation_errors) {
        std::fprintf(stderr, "%u Vulkan validation errors\n", validation_errors);
        ++failures;
    }

    std::printf("d3d12_scissor_validation: %zu parity cases (warp=%s, hw=%s), GPU draw+clip checks passed, %u failures\n",
                table.size(), warp_ran ? "live" : "measured", hw_ran ? "live" : "none", failures);
    return failures ? 1 : 0;
}
