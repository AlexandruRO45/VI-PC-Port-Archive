// Regression test for ResourceMinLODClamp on depth and stencil shader-resource views.
// bind_image previously gated VK_EXT_image_view_min_lod and the empty-mip-set swizzle on
// VK_IMAGE_ASPECT_COLOR_BIT, silently ignoring ResourceMinLODClamp on depth and stencil SRVs.
// Covers:
//   - 3-mip D32_FLOAT 2D depth SRVs (clamps 0.0, 1.0, 2.0, sub-range MostDetailedMip=1 clamp 2.0, empty clamp 2.5)
//   - 3-mip D32_FLOAT_S8X24_UINT depth-plane and stencil-plane SRVs (clamps 0.0, 1.0, 2.0, empty clamp 3.0)
//   - 3-mip 2-layer D32_FLOAT 2D-array depth SRV (MostDetailedMip=1, clamp 2.0)
//   - Device without VK_EXT_image_view_min_lod refuses an in-range depth clamp instead of binding unclamped
// No game data. Set LEONIDA_VK_VALIDATION=1 to run under VK_LAYER_KHRONOS_validation.
#include "shader_fixture_path.h"
#include "vk_test_device.h"
#include "../shims/d3d12_com.h"
#include "../src/gfx10_descriptors.h"

#include <array>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <span>
#include <vector>

#define EXPECT(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return 1; } \
} while (0)

namespace {

template<class T>
struct Ref {
    T* p = nullptr;
    ~Ref() { reset(); }
    void reset() {
        if (p) p->Release();
        p = nullptr;
    }
    T* operator->() const { return p; }
};

D3D12_RESOURCE_BARRIER transition(ID3D12Resource* resource, D3D12_RESOURCE_STATES before,
                                  D3D12_RESOURCE_STATES after) {
    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition.pResource = resource;
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    barrier.Transition.StateBefore = before;
    barrier.Transition.StateAfter = after;
    return barrier;
}

#ifdef LEONIDA_TEST_CS_SAMPLE_SPV
struct SampleKeys {
    std::uint32_t tsharp[8];
    std::uint32_t ssharp[4];
    std::uint32_t first;
    std::uint32_t mode;  // 0: normalized, 1: uint
    std::uint32_t layer;
};

VkShaderModule load_shader(vktest::Device& d, const char* path) {
    std::ifstream file(vktest::shader_fixture_path(path), std::ios::binary | std::ios::ate);
    if (!file) return VK_NULL_HANDLE;
    std::vector<std::uint32_t> code(std::size_t(file.tellg()) / 4);
    file.seekg(0);
    if (code.empty() || !file.read(reinterpret_cast<char*>(code.data()), std::streamsize(code.size() * 4)))
        return VK_NULL_HANDLE;
    VkShaderModuleCreateInfo info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    info.codeSize = code.size() * 4;
    info.pCode = code.data();
    VkShaderModule module{};
    return d.fn<PFN_vkCreateShaderModule>("vkCreateShaderModule")(d.device, &info, nullptr, &module) == VK_SUCCESS
               ? module
               : VK_NULL_HANDLE;
}

bool sample_through_set(vktest::Device& d, VkDescriptorSet set0, VkDescriptorSetLayout set0_layout,
                        std::span<const SampleKeys> keys, void* results) {
    const VkDevice dev = d.device;
    constexpr VkDeviceSize texels = 64;
    void* mapped = nullptr;
    const VkBuffer output = d.buffer(keys.size() * texels * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &mapped);
    if (!output) return false;

    const VkShaderModule module = load_shader(d, LEONIDA_TEST_CS_SAMPLE_SPV);
    VkDescriptorSetLayout set1_layout = VK_NULL_HANDLE;
    VkDescriptorPool pool = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkDescriptorSet set1 = VK_NULL_HANDLE;

    const VkDescriptorSetLayoutBinding binding{0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT,
                                               nullptr};
    VkDescriptorSetLayoutCreateInfo set1_info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    set1_info.bindingCount = 1;
    set1_info.pBindings = &binding;
    const VkDescriptorPoolSize pool_size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1};
    VkDescriptorPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pool_info.maxSets = 1;
    pool_info.poolSizeCount = 1;
    pool_info.pPoolSizes = &pool_size;
    bool ok = module &&
              d.fn<PFN_vkCreateDescriptorSetLayout>("vkCreateDescriptorSetLayout")(dev, &set1_info, nullptr,
                                                                                   &set1_layout) == VK_SUCCESS &&
              d.fn<PFN_vkCreateDescriptorPool>("vkCreateDescriptorPool")(dev, &pool_info, nullptr, &pool) == VK_SUCCESS;
    if (ok) {
        VkDescriptorSetAllocateInfo allocate{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        allocate.descriptorPool = pool;
        allocate.descriptorSetCount = 1;
        allocate.pSetLayouts = &set1_layout;
        ok = d.fn<PFN_vkAllocateDescriptorSets>("vkAllocateDescriptorSets")(dev, &allocate, &set1) == VK_SUCCESS;
    }
    if (ok) {
        const VkDescriptorBufferInfo buffer_info{output, 0, VK_WHOLE_SIZE};
        VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        write.dstSet = set1;
        write.descriptorCount = 1;
        write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        write.pBufferInfo = &buffer_info;
        d.fn<PFN_vkUpdateDescriptorSets>("vkUpdateDescriptorSets")(dev, 1, &write, 0, nullptr);

        const VkDescriptorSetLayout layouts[2] = {set0_layout, set1_layout};
        const VkPushConstantRange push{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(SampleKeys)};
        VkPipelineLayoutCreateInfo layout_info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        layout_info.setLayoutCount = 2;
        layout_info.pSetLayouts = layouts;
        layout_info.pushConstantRangeCount = 1;
        layout_info.pPushConstantRanges = &push;
        VkComputePipelineCreateInfo pipeline_info{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        pipeline_info.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        pipeline_info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        pipeline_info.stage.module = module;
        pipeline_info.stage.pName = "main";
        ok = d.fn<PFN_vkCreatePipelineLayout>("vkCreatePipelineLayout")(dev, &layout_info, nullptr, &layout) ==
             VK_SUCCESS;
        pipeline_info.layout = layout;
        ok = ok && d.fn<PFN_vkCreateComputePipelines>("vkCreateComputePipelines")(
                       dev, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &pipeline) == VK_SUCCESS;
    }
    if (ok) {
        const auto bind_sets = d.fn<PFN_vkCmdBindDescriptorSets>("vkCmdBindDescriptorSets");
        const auto push_constants = d.fn<PFN_vkCmdPushConstants>("vkCmdPushConstants");
        const auto dispatch = d.fn<PFN_vkCmdDispatch>("vkCmdDispatch");
        ok = d.run([&](VkCommandBuffer cb) {
            d.fn<PFN_vkCmdBindPipeline>("vkCmdBindPipeline")(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
            const VkDescriptorSet sets[2] = {set0, set1};
            bind_sets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 2, sets, 0, nullptr);
            for (const SampleKeys& k : keys) {
                push_constants(cb, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(k), &k);
                dispatch(cb, 1, 1, 1);
            }
            return true;
        });
    }
    if (ok) std::memcpy(results, mapped, keys.size() * texels * 4);
    if (pipeline) d.fn<PFN_vkDestroyPipeline>("vkDestroyPipeline")(dev, pipeline, nullptr);
    if (layout) d.fn<PFN_vkDestroyPipelineLayout>("vkDestroyPipelineLayout")(dev, layout, nullptr);
    if (pool) d.fn<PFN_vkDestroyDescriptorPool>("vkDestroyDescriptorPool")(dev, pool, nullptr);
    if (set1_layout) d.fn<PFN_vkDestroyDescriptorSetLayout>("vkDestroyDescriptorSetLayout")(dev, set1_layout, nullptr);
    if (module) d.fn<PFN_vkDestroyShaderModule>("vkDestroyShaderModule")(dev, module, nullptr);
    return ok;
}
#endif

}  // namespace

int main() {
    vktest::Device d;
    const auto status = d.initialize();
    if (status == vktest::DeviceStatus::unavailable) return 77;
    if (status != vktest::DeviceStatus::ready) return 1;
    if (!d.bindless) {
        std::printf("SKIP: device lacks bindless descriptor indexing\n");
        return 77;
    }

    const unsigned validation_before = d.validation_errors();
    constexpr UINT texels = 64, bytes = texels * 4;

    {
        rt::d3d12::VulkanDeviceInfo info{};
        info.get_instance_proc = d.instance_proc;
        info.instance = d.instance;
        info.physical = d.physical;
        info.device = d.device;
        info.queue = d.queue;
        info.queue_family = d.family;
        info.bindless = d.bindless;
        info.bindless_partially_bound = d.partially_bound;
        info.image_view_min_lod = d.image_view_min_lod;
        Ref<ID3D12Device> device;
        EXPECT(SUCCEEDED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&device.p))));

        D3D12_COMMAND_QUEUE_DESC qd{};
        qd.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
        Ref<ID3D12CommandQueue> queue;
        Ref<ID3D12CommandAllocator> allocator;
        Ref<ID3D12GraphicsCommandList> list;
        Ref<ID3D12Fence> fence;
        EXPECT(SUCCEEDED(device->CreateCommandQueue(&qd, IID_PPV_ARGS(&queue.p))));
        EXPECT(SUCCEEDED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator.p))));
        EXPECT(SUCCEEDED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.p, nullptr,
                                                   IID_PPV_ARGS(&list.p))));
        EXPECT(SUCCEEDED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence.p))));

        const D3D12_HEAP_PROPERTIES default_heap{D3D12_HEAP_TYPE_DEFAULT};
        const auto make_depth = [&](DXGI_FORMAT format, UINT16 layers, Ref<ID3D12Resource>& out) {
            D3D12_RESOURCE_DESC rd{};
            rd.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
            rd.Width = 8;
            rd.Height = 8;
            rd.DepthOrArraySize = layers;
            rd.MipLevels = 3;
            rd.Format = format;
            rd.SampleDesc.Count = 1;
            rd.Flags = D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;
            return SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &rd,
                                                             D3D12_RESOURCE_STATE_DEPTH_WRITE, nullptr,
                                                             IID_PPV_ARGS(&out.p)));
        };

        Ref<ID3D12Resource> d32_tex, d32s8_tex, d32_arr;
        EXPECT(make_depth(DXGI_FORMAT_D32_FLOAT, 1, d32_tex));
        EXPECT(make_depth(DXGI_FORMAT_D32_FLOAT_S8X24_UINT, 1, d32s8_tex));
        EXPECT(make_depth(DXGI_FORMAT_D32_FLOAT, 2, d32_arr));

        // DSVs: 3 mips of d32_tex, 3 mips of d32s8_tex, 6 subresources (2 layers x 3 mips) of d32_arr.
        constexpr UINT dsv_count = 12;
        const D3D12_DESCRIPTOR_HEAP_DESC dsv_hd{D3D12_DESCRIPTOR_HEAP_TYPE_DSV, dsv_count,
                                                D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0};
        Ref<ID3D12DescriptorHeap> dsv_heap;
        EXPECT(SUCCEEDED(device->CreateDescriptorHeap(&dsv_hd, IID_PPV_ARGS(&dsv_heap.p))));
        const SIZE_T dsv_base = dsv_heap->GetCPUDescriptorHandleForHeapStart().ptr;
        const UINT dsv_step = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_DSV);
        const auto dsv_slot = [&](UINT i) { return D3D12_CPU_DESCRIPTOR_HANDLE{dsv_base + SIZE_T(i) * dsv_step}; };

        constexpr std::uint8_t d32_vals[3] = {51, 128, 204};
        constexpr std::uint8_t ds_depth_vals[3] = {40, 110, 190};
        constexpr std::uint8_t ds_stencil_vals[3] = {17, 42, 99};
        constexpr std::uint8_t arr_vals[2][3] = {{30, 90, 150}, {60, 120, 210}};

        for (UINT mip = 0; mip < 3; ++mip) {
            D3D12_DEPTH_STENCIL_VIEW_DESC dsv{};
            dsv.Format = DXGI_FORMAT_D32_FLOAT;
            dsv.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2D;
            dsv.Texture2D.MipSlice = mip;
            device->CreateDepthStencilView(d32_tex.p, &dsv, dsv_slot(mip));
            list->ClearDepthStencilView(dsv_slot(mip), D3D12_CLEAR_FLAG_DEPTH, float(d32_vals[mip]) / 255.0f, 0, 0,
                                        nullptr);

            dsv.Format = DXGI_FORMAT_D32_FLOAT_S8X24_UINT;
            device->CreateDepthStencilView(d32s8_tex.p, &dsv, dsv_slot(3 + mip));
            list->ClearDepthStencilView(dsv_slot(3 + mip), D3D12_CLEAR_FLAG_DEPTH | D3D12_CLEAR_FLAG_STENCIL,
                                        float(ds_depth_vals[mip]) / 255.0f, ds_stencil_vals[mip], 0, nullptr);

            for (UINT layer = 0; layer < 2; ++layer) {
                D3D12_DEPTH_STENCIL_VIEW_DESC adsv{};
                adsv.Format = DXGI_FORMAT_D32_FLOAT;
                adsv.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2DARRAY;
                adsv.Texture2DArray = {mip, layer, 1};
                const UINT slot_idx = 6 + layer * 3 + mip;
                device->CreateDepthStencilView(d32_arr.p, &adsv, dsv_slot(slot_idx));
                list->ClearDepthStencilView(dsv_slot(slot_idx), D3D12_CLEAR_FLAG_DEPTH,
                                            float(arr_vals[layer][mip]) / 255.0f, 0, 0, nullptr);
            }
        }
        const D3D12_RESOURCE_BARRIER barriers[] = {
            transition(d32_tex.p, D3D12_RESOURCE_STATE_DEPTH_WRITE, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE),
            transition(d32s8_tex.p, D3D12_RESOURCE_STATE_DEPTH_WRITE, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE),
            transition(d32_arr.p, D3D12_RESOURCE_STATE_DEPTH_WRITE, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE),
        };
        list->ResourceBarrier(3, barriers);
        EXPECT(list->Close() == S_OK);

        // SRV + sampler heaps.
        // 0..4: d32_tex (clamp 0, 1, 2; first=1 count=2 clamp=2; first=0 count=2 clamp=2.5 empty)
        // 5..8: d32s8_tex depth plane (clamp 0, 1, 2, 3 empty)
        // 9..12: d32s8_tex stencil plane (clamp 0, 1, 2, 3 empty)
        // 13: d32_arr (first=1 count=2 layers 0..2 clamp=2)
        constexpr UINT srv_count = 14;
        const D3D12_DESCRIPTOR_HEAP_DESC srv_hd{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, srv_count,
                                                D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, 0};
        const D3D12_DESCRIPTOR_HEAP_DESC smp_hd{D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER, 1,
                                                D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, 0};
        Ref<ID3D12DescriptorHeap> srv_heap, smp_heap;
        EXPECT(SUCCEEDED(device->CreateDescriptorHeap(&srv_hd, IID_PPV_ARGS(&srv_heap.p))));
        EXPECT(SUCCEEDED(device->CreateDescriptorHeap(&smp_hd, IID_PPV_ARGS(&smp_heap.p))));
        const SIZE_T srv_base = srv_heap->GetCPUDescriptorHandleForHeapStart().ptr;
        const UINT srv_step = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        const auto srv_slot = [&](UINT i) { return D3D12_CPU_DESCRIPTOR_HANDLE{srv_base + SIZE_T(i) * srv_step}; };
        const auto srv_bytes = [&](UINT i) { return reinterpret_cast<const std::uint8_t*>(srv_slot(i).ptr); };

        const auto make_srv2d = [&](ID3D12Resource* res, DXGI_FORMAT fmt, UINT first, UINT count, UINT plane,
                                    float clamp, UINT slot_idx) {
            D3D12_SHADER_RESOURCE_VIEW_DESC srv{};
            srv.Format = fmt;
            srv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
            srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
            srv.Texture2D = {first, count, plane, clamp};
            device->CreateShaderResourceView(res, &srv, srv_slot(slot_idx));
        };

        make_srv2d(d32_tex.p, DXGI_FORMAT_R32_FLOAT, 0, 3, 0, 0.0f, 0);
        make_srv2d(d32_tex.p, DXGI_FORMAT_R32_FLOAT, 0, 3, 0, 1.0f, 1);
        make_srv2d(d32_tex.p, DXGI_FORMAT_R32_FLOAT, 0, 3, 0, 2.0f, 2);
        make_srv2d(d32_tex.p, DXGI_FORMAT_R32_FLOAT, 1, 2, 0, 2.0f, 3);
        make_srv2d(d32_tex.p, DXGI_FORMAT_R32_FLOAT, 0, 2, 0, 2.5f, 4);

        make_srv2d(d32s8_tex.p, DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS, 0, 3, 0, 0.0f, 5);
        make_srv2d(d32s8_tex.p, DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS, 0, 3, 0, 1.0f, 6);
        make_srv2d(d32s8_tex.p, DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS, 0, 3, 0, 2.0f, 7);
        make_srv2d(d32s8_tex.p, DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS, 0, 3, 0, 3.0f, 8);

        make_srv2d(d32s8_tex.p, DXGI_FORMAT_X32_TYPELESS_G8X24_UINT, 0, 3, 1, 0.0f, 9);
        make_srv2d(d32s8_tex.p, DXGI_FORMAT_X32_TYPELESS_G8X24_UINT, 0, 3, 1, 1.0f, 10);
        make_srv2d(d32s8_tex.p, DXGI_FORMAT_X32_TYPELESS_G8X24_UINT, 0, 3, 1, 2.0f, 11);
        make_srv2d(d32s8_tex.p, DXGI_FORMAT_X32_TYPELESS_G8X24_UINT, 0, 3, 1, 3.0f, 12);

        {
            D3D12_SHADER_RESOURCE_VIEW_DESC srv{};
            srv.Format = DXGI_FORMAT_R32_FLOAT;
            srv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2DARRAY;
            srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
            srv.Texture2DArray = {1, 2, 0, 2, 0, 2.0f};
            device->CreateShaderResourceView(d32_arr.p, &srv, srv_slot(13));
        }

        D3D12_SAMPLER_DESC sd{};
        sd.Filter = D3D12_FILTER_MIN_MAG_MIP_POINT;
        sd.AddressU = sd.AddressV = sd.AddressW = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
        sd.MaxAnisotropy = 1;
        sd.ComparisonFunc = D3D12_COMPARISON_FUNC_NEVER;
        sd.MaxLOD = D3D12_FLOAT32_MAX;
        const D3D12_CPU_DESCRIPTOR_HANDLE smp_slot = smp_heap->GetCPUDescriptorHandleForHeapStart();
        device->CreateSampler(&sd, smp_slot);

        // Execute clears + transitions (also flushes the bindless descriptor set).
        ID3D12CommandList* lists[] = {list.p};
        queue->ExecuteCommandLists(1, lists);
        EXPECT(rt::d3d12::last_execute_result(queue.p) == leonida::ReplayResult::success);
        EXPECT(SUCCEEDED(queue->Signal(fence.p, 1)) && SUCCEEDED(fence->SetEventOnCompletion(1, nullptr)));

#ifdef LEONIDA_TEST_CS_SAMPLE_SPV
        const VkDescriptorSet set0 = rt::d3d12::bindless_set(device.p);
        const VkDescriptorSetLayout set0_layout = rt::d3d12::bindless_set_layout(device.p);
        EXPECT(set0 && set0_layout);

        std::array<SampleKeys, srv_count> keys{};
        for (UINT k = 0; k < srv_count; ++k) {
            std::memcpy(keys[k].tsharp, srv_bytes(k), sizeof(keys[k].tsharp));
            std::memcpy(keys[k].ssharp, reinterpret_cast<const void*>(smp_slot.ptr), sizeof(keys[k].ssharp));
            keys[k].first = k * texels;
            keys[k].mode = (k >= 9 && k <= 12) ? 1u : 0u;  // stencil plane uses uint sampling
        }
        std::vector<std::uint8_t> pixels(srv_count * bytes);
        EXPECT(sample_through_set(d, set0, set0_layout, keys, pixels.data()));

        using Pixel = std::array<std::uint8_t, 4>;
        const Pixel fallback{};
        const bool min_lod = d.image_view_min_lod;
        const Pixel expected_2d[13] = {
            // D32_FLOAT 2D
            {d32_vals[0], 0, 0, 255},
            min_lod ? Pixel{d32_vals[1], 0, 0, 255} : fallback,
            min_lod ? Pixel{d32_vals[2], 0, 0, 255} : fallback,
            min_lod ? Pixel{d32_vals[2], 0, 0, 255} : fallback,
            {0, 0, 0, 255},
            // D32_FLOAT_S8X24_UINT depth plane
            {ds_depth_vals[0], 0, 0, 255},
            min_lod ? Pixel{ds_depth_vals[1], 0, 0, 255} : fallback,
            min_lod ? Pixel{ds_depth_vals[2], 0, 0, 255} : fallback,
            {0, 0, 0, 255},
            // D32_FLOAT_S8X24_UINT stencil plane (mode=1: R, G, B, A bytes; stencil is in G, A=1)
            {0, ds_stencil_vals[0], 0, 1},
            min_lod ? Pixel{0, ds_stencil_vals[1], 0, 1} : fallback,
            min_lod ? Pixel{0, ds_stencil_vals[2], 0, 1} : fallback,
            {0, 0, 0, 1},
        };
        UINT mismatches = 0;
        for (UINT k = 0; k < 13; ++k) {
            for (UINT i = 0; i < texels; ++i) {
                const auto* got = pixels.data() + k * bytes + i * 4;
                if (std::memcmp(got, expected_2d[k].data(), 4) != 0) {
                    if (mismatches++ < 16)
                        std::fprintf(stderr, "view %u texel %u: got %u,%u,%u,%u, expected %u,%u,%u,%u\n", k, i,
                                     got[0], got[1], got[2], got[3], expected_2d[k][0], expected_2d[k][1],
                                     expected_2d[k][2], expected_2d[k][3]);
                    break;
                }
            }
        }
        // View 13: 2D array (first 32 texels sample layer 0, next 32 sample layer 1).
        for (UINT i = 0; i < texels; ++i) {
            const UINT layer = i / 32;
            const Pixel want = min_lod ? Pixel{arr_vals[layer][2], 0, 0, 255} : fallback;
            const auto* got = pixels.data() + 13 * bytes + i * 4;
            if (std::memcmp(got, want.data(), 4) != 0) {
                if (mismatches++ < 16)
                    std::fprintf(stderr, "array view layer %u texel %u: got %u,%u,%u,%u, expected %u,%u,%u,%u\n",
                                 layer, i, got[0], got[1], got[2], got[3], want[0], want[1], want[2], want[3]);
                break;
            }
        }
        EXPECT(mismatches == 0);
        std::printf("depth/stencil SRV ResourceMinLODClamp: all %u views matched (%s)\n", srv_count,
                    min_lod ? "VK_EXT_image_view_min_lod on" : "no VK_EXT_image_view_min_lod");
#endif
    }

    // Negative check: when VK_EXT_image_view_min_lod is disabled on the device, an in-range depth clamp
    // must not bind an unclamped view into the bindless hash table, while unclamped (0.0) and empty-range
    // (past last mip) views still bind.
    {
        rt::d3d12::VulkanDeviceInfo info{};
        info.get_instance_proc = d.instance_proc;
        info.instance = d.instance;
        info.physical = d.physical;
        info.device = d.device;
        info.queue = d.queue;
        info.queue_family = d.family;
        info.bindless = d.bindless;
        info.bindless_partially_bound = d.partially_bound;
        info.image_view_min_lod = false;
        Ref<ID3D12Device> device;
        EXPECT(SUCCEEDED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&device.p))));

        const D3D12_HEAP_PROPERTIES default_heap{D3D12_HEAP_TYPE_DEFAULT};
        D3D12_RESOURCE_DESC rd{};
        rd.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        rd.Width = 8;
        rd.Height = 8;
        rd.DepthOrArraySize = 1;
        rd.MipLevels = 3;
        rd.Format = DXGI_FORMAT_D32_FLOAT;
        rd.SampleDesc.Count = 1;
        rd.Flags = D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;
        Ref<ID3D12Resource> depth_tex;
        EXPECT(SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &rd,
                                                         D3D12_RESOURCE_STATE_DEPTH_WRITE, nullptr,
                                                         IID_PPV_ARGS(&depth_tex.p))));

        const D3D12_DESCRIPTOR_HEAP_DESC hd{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 3,
                                            D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, 0};
        Ref<ID3D12DescriptorHeap> heap;
        EXPECT(SUCCEEDED(device->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&heap.p))));
        const SIZE_T base = heap->GetCPUDescriptorHandleForHeapStart().ptr;
        const UINT step = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        const auto slot = [&](UINT i) { return D3D12_CPU_DESCRIPTOR_HANDLE{base + SIZE_T(i) * step}; };
        const auto lookup = [&](UINT i) {
            std::uint32_t key[8]{};
            std::memcpy(key, reinterpret_cast<const void*>(slot(i).ptr), sizeof(key));
            return rt::d3d12::bindless_hash_lookup(device.p, key);
        };

        const float clamps[3] = {0.0f, 1.0f, 3.0f};
        for (UINT i = 0; i < 3; ++i) {
            D3D12_SHADER_RESOURCE_VIEW_DESC srv{};
            srv.Format = DXGI_FORMAT_R32_FLOAT;
            srv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
            srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
            srv.Texture2D = {0, 3, 0, clamps[i]};
            device->CreateShaderResourceView(depth_tex.p, &srv, slot(i));
        }
        EXPECT(lookup(0) != 0);
        EXPECT(lookup(1) == 0);
        EXPECT(lookup(2) != 0);
        std::printf("depth SRV without VK_EXT_image_view_min_lod: clamp 0 bound, clamp 1 refused, clamp 3 (empty) bound\n");
    }

    EXPECT(d.validation_errors() == validation_before);
    return 0;
}
