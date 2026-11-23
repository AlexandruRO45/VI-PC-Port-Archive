// Graphics draws into 3D volume render-target views (D3D12_RTV_DIMENSION_TEXTURE3D) through the Vulkan device:
//   - single-slice draw on mip 0 with FirstWSlice = 2, WSize = 1 leaves slices 0, 1, 3 untouched and colours slice 2;
//   - single-slice draw on mip 1 with FirstWSlice = 1, WSize = 1 leaves slice 0 untouched and colours slice 1;
//   - layered draw (when geometryShader is enabled) with FirstWSlice = 1, WSize = 2 writes gl_Layer 0 to slice 1 and
//     gl_Layer 1 to slice 2 without touching slices 0 or 3;
//   - layered draw with FirstWSlice = 2, WSize = UINT_MAX writes gl_Layer 0 to slice 2 and gl_Layer 1 to slice 3.
// Regression test for prepare_draw passing the 3D layout range (baseArrayLayer = 0, layerCount = 1) to image_view
// and VkRenderingInfo::layerCount instead of target.first_layer and target.layer_count.
// Shaders: test/glsl/vs_fullscreen.vert, gs_volume_layers.geom, ps_draw.frag. No game data.
// Set LEONIDA_VK_VALIDATION=1 to run under VK_LAYER_KHRONOS_validation.
#include "shader_fixture_path.h"
#include "test_temp_dir.h"
#include "vk_test_device.h"
#include "d3d12_test_shaders.h"
#include "../shims/d3d12_com.h"
#include "../src/shader_library.h"

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <vector>

#if !defined(LEONIDA_TEST_VS_FULLSCREEN_SPV) || !defined(LEONIDA_TEST_GS_VOLUME_LAYERS_SPV) || \
    !defined(LEONIDA_TEST_PS_SPV)
#error "LEONIDA_TEST_VS_FULLSCREEN_SPV, LEONIDA_TEST_GS_VOLUME_LAYERS_SPV and LEONIDA_TEST_PS_SPV must be defined"
#endif

#define EXPECT(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return 1; } \
} while (0)

using namespace d3d12test;

namespace {

bool check_slice(const std::uint8_t* data, UINT width, UINT height, UINT row_pitch, UINT slice,
                 const std::array<std::uint8_t, 4>& expected, const char* label) {
    const std::size_t slice_pitch = std::size_t(height) * row_pitch;
    for (UINT y = 0; y < height; ++y) {
        for (UINT x = 0; x < width; ++x) {
            const std::uint8_t* px = data + std::size_t(slice) * slice_pitch + std::size_t(y) * row_pitch + x * 4;
            if (px[0] != expected[0] || px[1] != expected[1] || px[2] != expected[2] || px[3] != expected[3]) {
                std::fprintf(stderr,
                             "%s: slice %u (%u,%u) got (%u,%u,%u,%u), expected (%u,%u,%u,%u)\n",
                             label, slice, x, y, px[0], px[1], px[2], px[3],
                             expected[0], expected[1], expected[2], expected[3]);
                return false;
            }
        }
    }
    return true;
}

}  // namespace

int main() {
    vktest::Device d;
    const auto device_status = d.initialize();
    if (device_status == vktest::DeviceStatus::unavailable) {
        std::printf("SKIP: no usable Vulkan 1.3 device\n");
        return 77;
    }
    EXPECT(device_status == vktest::DeviceStatus::ready);

    const auto vs_program = fssm_program({0x56530301, 0xbf810000});
    const auto gs_program = fssm_program({0x47530301, 0xbf810000});
    const auto ps_program = fssm_program({0x50530301, 0xbf810000});
    const auto vs_container = shader_container(vs_program);
    const auto gs_container = shader_container(gs_program);
    const auto ps_container = shader_container(ps_program);

    const testfiles::ScratchDir scratch("leonida_volume_draw_test");
    const auto& library = scratch.path();
    EXPECT(testfiles::copy_file(vktest::shader_fixture_path(LEONIDA_TEST_VS_FULLSCREEN_SPV),
                                library / "vs_fullscreen.spv"));
    EXPECT(testfiles::copy_file(vktest::shader_fixture_path(LEONIDA_TEST_GS_VOLUME_LAYERS_SPV),
                                library / "gs_volume_layers.spv"));
    EXPECT(testfiles::copy_file(vktest::shader_fixture_path(LEONIDA_TEST_PS_SPV), library / "ps_draw.spv"));
    {
        std::ofstream index(library / "index.json", std::ios::binary);
        index << "{\"vs_fullscreen\": {\"stage\": \"vs\", \"fssm_sha256\": \"" << leonida::sha256_hex(vs_program)
              << "\", \"bindings\": [], \"push_constant_bytes\": 144, \"input_signature\": []},\n"
                 " \"gs_volume_layers\": {\"stage\": \"gs\", \"fssm_sha256\": \"" << leonida::sha256_hex(gs_program)
              << "\", \"bindings\": [], \"push_constant_bytes\": 144},\n"
                 " \"ps_draw\": {\"stage\": \"ps\", \"fssm_sha256\": \"" << leonida::sha256_hex(ps_program)
              << "\", \"bindings\": [], \"push_constant_bytes\": 144, \"outputs\": [0]}}";
    }

    const bool has_gs = d.enabled.geometryShader == VK_TRUE;

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
        Ref<ID3D12Device> device;
        EXPECT(SUCCEEDED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&device.p))));

        // Root signature: [0] two unused constants (s0, s1), [1] root CBV (s2:s3: tint for ps_draw.frag).
        D3D12_ROOT_PARAMETER params[2] = {};
        params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
        params[0].Constants = {0, 0, 2};
        params[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
        params[1].Descriptor = {1, 0};
        D3D12_VERSIONED_ROOT_SIGNATURE_DESC root_desc{};
        root_desc.Version = D3D_ROOT_SIGNATURE_VERSION_1_0;
        root_desc.Desc_1_0 = {2, params, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_NONE};
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
        pso_desc.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
        pso_desc.SampleDesc.Count = 1;
        Ref<ID3D12PipelineState> single_pso, layered_pso;
        EXPECT(SUCCEEDED(device->CreateGraphicsPipelineState(&pso_desc, IID_PPV_ARGS(&single_pso.p))));
        if (has_gs) {
            pso_desc.GS = {gs_container.data(), gs_container.size()};
            EXPECT(SUCCEEDED(device->CreateGraphicsPipelineState(&pso_desc, IID_PPV_ARGS(&layered_pso.p))));
        }

        const D3D12_HEAP_PROPERTIES default_heap{D3D12_HEAP_TYPE_DEFAULT};
        const D3D12_HEAP_PROPERTIES upload_heap{D3D12_HEAP_TYPE_UPLOAD};
        const D3D12_HEAP_PROPERTIES readback_heap{D3D12_HEAP_TYPE_READBACK};

        // Upload buffer holding three tint CBVs (256-byte aligned each):
        //   [0]: (1, 0, 0) -> ps_draw outputs red (vs_fullscreen has r=1, b=0)
        //   [1]: (0, 1, 0) -> ps_draw outputs green
        //   [2]: (1, 0, 1) -> ps_draw outputs (color.r, 0, color.b, 1): red for layer 0, blue for layer 1
        Ref<ID3D12Resource> upload;
        const auto upload_desc = buffer_desc(1024);
        EXPECT(SUCCEEDED(device->CreateCommittedResource(&upload_heap, D3D12_HEAP_FLAG_NONE, &upload_desc,
                                                         D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
                                                         IID_PPV_ARGS(&upload.p))));
        void* mapped = nullptr;
        EXPECT(SUCCEEDED(upload->Map(0, nullptr, &mapped)));
        const float tints[3][3] = {{1.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, {1.0f, 0.0f, 1.0f}};
        for (int i = 0; i < 3; ++i)
            std::memcpy(static_cast<std::uint8_t*>(mapped) + i * 256, tints[i], sizeof(tints[i]));
        upload->Unmap(0, nullptr);
        const D3D12_GPU_VIRTUAL_ADDRESS cbv_base = upload->GetGPUVirtualAddress();

        constexpr UINT kWidth = 16, kHeight = 16, kDepth = 4;
        D3D12_RESOURCE_DESC vol_desc{};
        vol_desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE3D;
        vol_desc.Width = kWidth;
        vol_desc.Height = kHeight;
        vol_desc.DepthOrArraySize = kDepth;
        vol_desc.MipLevels = 2;
        vol_desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        vol_desc.SampleDesc.Count = 1;
        vol_desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
        Ref<ID3D12Resource> volume;
        EXPECT(SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &vol_desc,
                                                         D3D12_RESOURCE_STATE_RENDER_TARGET, nullptr,
                                                         IID_PPV_ARGS(&volume.p))));

        D3D12_PLACED_SUBRESOURCE_FOOTPRINT fp_mip0{}, fp_mip1{};
        UINT64 bytes_mip0 = 0, bytes_mip1 = 0;
        device->GetCopyableFootprints(&vol_desc, 0, 1, 0, &fp_mip0, nullptr, nullptr, &bytes_mip0);
        device->GetCopyableFootprints(&vol_desc, 1, 1, 0, &fp_mip1, nullptr, nullptr, &bytes_mip1);
        const UINT64 readback_size = std::max(bytes_mip0, bytes_mip1);
        Ref<ID3D12Resource> readback;
        const auto rb_desc = buffer_desc(readback_size);
        EXPECT(SUCCEEDED(device->CreateCommittedResource(&readback_heap, D3D12_HEAP_FLAG_NONE, &rb_desc,
                                                         D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                         IID_PPV_ARGS(&readback.p))));

        // RTV descriptor heap:
        //   0: mip 0, all 4 slices (0..4)
        //   1: mip 0, slice 2 only (FirstWSlice=2, WSize=1)
        //   2: mip 1, all 2 slices (0..2)
        //   3: mip 1, slice 1 only (FirstWSlice=1, WSize=1)
        //   4: mip 0, slices 1..2  (FirstWSlice=1, WSize=2)
        //   5: mip 0, slices 2..3  (FirstWSlice=2, WSize=UINT_MAX)
        D3D12_DESCRIPTOR_HEAP_DESC rtv_heap_desc{D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 6, D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0};
        Ref<ID3D12DescriptorHeap> rtv_heap;
        EXPECT(SUCCEEDED(device->CreateDescriptorHeap(&rtv_heap_desc, IID_PPV_ARGS(&rtv_heap.p))));
        const UINT rtv_inc = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
        const auto rtv_at = [&](UINT slot) {
            D3D12_CPU_DESCRIPTOR_HANDLE h = rtv_heap->GetCPUDescriptorHandleForHeapStart();
            h.ptr += SIZE_T(slot) * rtv_inc;
            return h;
        };
        const auto make_rtv = [&](UINT slot, UINT mip, UINT first_w, UINT w_size) {
            D3D12_RENDER_TARGET_VIEW_DESC d{};
            d.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
            d.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE3D;
            d.Texture3D = {mip, first_w, w_size};
            device->CreateRenderTargetView(volume.p, &d, rtv_at(slot));
        };
        make_rtv(0, 0, 0, kDepth);
        make_rtv(1, 0, 2, 1);
        make_rtv(2, 1, 0, kDepth / 2);
        make_rtv(3, 1, 1, 1);
        make_rtv(4, 0, 1, 2);
        make_rtv(5, 0, 2, UINT_MAX);

        D3D12_COMMAND_QUEUE_DESC qdesc{D3D12_COMMAND_LIST_TYPE_DIRECT, 0, D3D12_COMMAND_QUEUE_FLAG_NONE, 0};
        Ref<ID3D12CommandQueue> queue;
        EXPECT(SUCCEEDED(device->CreateCommandQueue(&qdesc, IID_PPV_ARGS(&queue.p))));
        Ref<ID3D12CommandAllocator> alloc;
        EXPECT(SUCCEEDED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&alloc.p))));
        Ref<ID3D12GraphicsCommandList> list;
        EXPECT(SUCCEEDED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, alloc.p, nullptr,
                                                   IID_PPV_ARGS(&list.p))));
        Ref<ID3D12Fence> fence;
        EXPECT(SUCCEEDED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence.p))));
        HANDLE event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        EXPECT(event != nullptr);
        UINT64 fence_val = 0;

        auto submit_and_read = [&](UINT mip, const D3D12_PLACED_SUBRESOURCE_FOOTPRINT& fp, UINT64 copy_bytes,
                                   std::vector<std::uint8_t>& out) -> bool {
            D3D12_RESOURCE_BARRIER to_src{};
            to_src.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
            to_src.Transition = {volume.p, mip, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_COPY_SOURCE};
            list->ResourceBarrier(1, &to_src);

            D3D12_TEXTURE_COPY_LOCATION dst_loc{readback.p, D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT};
            dst_loc.PlacedFootprint = fp;
            D3D12_TEXTURE_COPY_LOCATION src_loc{volume.p, D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX};
            src_loc.SubresourceIndex = mip;
            list->CopyTextureRegion(&dst_loc, 0, 0, 0, &src_loc, nullptr);

            D3D12_RESOURCE_BARRIER to_rt{};
            to_rt.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
            to_rt.Transition = {volume.p, mip, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_RENDER_TARGET};
            list->ResourceBarrier(1, &to_rt);

            if (FAILED(list->Close())) return false;
            ID3D12CommandList* lists[] = {list.p};
            queue->ExecuteCommandLists(1, lists);
            ++fence_val;
            if (FAILED(queue->Signal(fence.p, fence_val))) return false;
            if (FAILED(fence->SetEventOnCompletion(fence_val, event))) return false;
            if (WaitForSingleObject(event, 10000) != WAIT_OBJECT_0) return false;

            out.resize(std::size_t(copy_bytes));
            void* p = nullptr;
            if (FAILED(readback->Map(0, nullptr, &p))) return false;
            std::memcpy(out.data(), p, std::size_t(copy_bytes));
            readback->Unmap(0, nullptr);

            if (FAILED(alloc->Reset())) return false;
            if (FAILED(list->Reset(alloc.p, nullptr))) return false;
            return true;
        };

        auto record_draw = [&](ID3D12PipelineState* pso, UINT clear_rtv, const float clear_color[4],
                               UINT draw_rtv, UINT w, UINT h, UINT cbv_index) {
            list->ClearRenderTargetView(rtv_at(clear_rtv), clear_color, 0, nullptr);
            D3D12_CPU_DESCRIPTOR_HANDLE rt = rtv_at(draw_rtv);
            list->OMSetRenderTargets(1, &rt, FALSE, nullptr);
            D3D12_VIEWPORT vp{0.0f, 0.0f, float(w), float(h), 0.0f, 1.0f};
            D3D12_RECT sc{0, 0, LONG(w), LONG(h)};
            list->RSSetViewports(1, &vp);
            list->RSSetScissorRects(1, &sc);
            list->SetPipelineState(pso);
            list->SetGraphicsRootSignature(root.p);
            list->SetGraphicsRootConstantBufferView(1, cbv_base + UINT64(cbv_index) * 256);
            list->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            list->DrawInstanced(3, 1, 0, 0);
        };

        constexpr std::array<std::uint8_t, 4> kRed{255, 0, 0, 255};
        constexpr std::array<std::uint8_t, 4> kGreen{0, 255, 0, 255};
        constexpr std::array<std::uint8_t, 4> kBlue{0, 0, 255, 255};
        const float blue[4] = {0.0f, 0.0f, 1.0f, 1.0f};
        const float green[4] = {0.0f, 1.0f, 0.0f, 1.0f};

        std::vector<std::uint8_t> data;

        // 1. Mip 0, FirstWSlice = 2, WSize = 1 -> red in slice 2, blue in slices 0, 1, 3.
        record_draw(single_pso.p, 0, blue, 1, kWidth, kHeight, 0);
        EXPECT(submit_and_read(0, fp_mip0, bytes_mip0, data));
        EXPECT(check_slice(data.data(), kWidth, kHeight, fp_mip0.Footprint.RowPitch, 0, kBlue, "mip0 FirstWSlice=2"));
        EXPECT(check_slice(data.data(), kWidth, kHeight, fp_mip0.Footprint.RowPitch, 1, kBlue, "mip0 FirstWSlice=2"));
        EXPECT(check_slice(data.data(), kWidth, kHeight, fp_mip0.Footprint.RowPitch, 2, kRed, "mip0 FirstWSlice=2"));
        EXPECT(check_slice(data.data(), kWidth, kHeight, fp_mip0.Footprint.RowPitch, 3, kBlue, "mip0 FirstWSlice=2"));

        // 2. Mip 1, FirstWSlice = 1, WSize = 1 -> green in slice 1, blue in slice 0.
        record_draw(single_pso.p, 2, blue, 3, kWidth / 2, kHeight / 2, 1);
        EXPECT(submit_and_read(1, fp_mip1, bytes_mip1, data));
        EXPECT(check_slice(data.data(), kWidth / 2, kHeight / 2, fp_mip1.Footprint.RowPitch, 0, kBlue, "mip1 FirstWSlice=1"));
        EXPECT(check_slice(data.data(), kWidth / 2, kHeight / 2, fp_mip1.Footprint.RowPitch, 1, kGreen, "mip1 FirstWSlice=1"));

        if (has_gs) {
            // 3. Layered draw into mip 0, FirstWSlice = 1, WSize = 2 -> red in slice 1, blue in slice 2, green in 0 and 3.
            record_draw(layered_pso.p, 0, green, 4, kWidth, kHeight, 2);
            EXPECT(submit_and_read(0, fp_mip0, bytes_mip0, data));
            EXPECT(check_slice(data.data(), kWidth, kHeight, fp_mip0.Footprint.RowPitch, 0, kGreen, "layered FirstWSlice=1 WSize=2"));
            EXPECT(check_slice(data.data(), kWidth, kHeight, fp_mip0.Footprint.RowPitch, 1, kRed, "layered FirstWSlice=1 WSize=2"));
            EXPECT(check_slice(data.data(), kWidth, kHeight, fp_mip0.Footprint.RowPitch, 2, kBlue, "layered FirstWSlice=1 WSize=2"));
            EXPECT(check_slice(data.data(), kWidth, kHeight, fp_mip0.Footprint.RowPitch, 3, kGreen, "layered FirstWSlice=1 WSize=2"));

            // 4. Layered draw into mip 0, FirstWSlice = 2, WSize = UINT_MAX -> red in slice 2, blue in slice 3, green in 0 and 1.
            record_draw(layered_pso.p, 0, green, 5, kWidth, kHeight, 2);
            EXPECT(submit_and_read(0, fp_mip0, bytes_mip0, data));
            EXPECT(check_slice(data.data(), kWidth, kHeight, fp_mip0.Footprint.RowPitch, 0, kGreen, "layered FirstWSlice=2 WSize=-1"));
            EXPECT(check_slice(data.data(), kWidth, kHeight, fp_mip0.Footprint.RowPitch, 1, kGreen, "layered FirstWSlice=2 WSize=-1"));
            EXPECT(check_slice(data.data(), kWidth, kHeight, fp_mip0.Footprint.RowPitch, 2, kRed, "layered FirstWSlice=2 WSize=-1"));
            EXPECT(check_slice(data.data(), kWidth, kHeight, fp_mip0.Footprint.RowPitch, 3, kBlue, "layered FirstWSlice=2 WSize=-1"));
        }

        CloseHandle(event);
    }

    EXPECT(d.validation_errors() == 0);
    std::printf("d3d12_volume_draw_test: PASS (mip 0/1 single-slice%s)\n",
                has_gs ? " + layered WSize=2 / WSize=UINT_MAX" : ", GS skipped");
    return 0;
}
