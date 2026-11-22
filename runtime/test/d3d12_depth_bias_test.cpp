// DepthBias is exactly one step of the depth format, as in D3D12: r = 1 / (2^n - 1) for an n-bit UNORM (2^-16 and
// 2^-24 within half a step for the biases here), 2^(exponent(z) - 23) for D32_FLOAT. Vulkan allows a fixed-point
// attachment up to twice that unless VK_EXT_depth_bias_control's depthBiasExact is asked for, and the RTX 5070 uses
// twice (D24: +3 stored as +6, +1000 as +2000). The layer asks for it where the device enabled the feature
// (VulkanDeviceInfo::depth_bias_exact); where it can't, this test reports what the device does instead of failing.
//
// A flat full-screen triangle at a depth on one of the format's steps (so no rounding tie decides the code) is drawn
// with DepthFunc ALWAYS and several constant biases into D16_UNORM, D24_UNORM_S8_UINT and D32_FLOAT targets, and the
// stored depth is read back through a footprint. D24 is skipped where the layer stores it as D32 (no D24 on the
// GPU): that path has its own bias handling (D3D-309).
//
// The vertex shader is GLSL written to the translator's contract (test/glsl/vs_depth.vert). No game data.
#include "vk_test_device.h"
#include "d3d12_test_shaders.h"
#include "../shims/d3d12_com.h"
#include "../shims/d3d12_xs_device.h"
#include "../src/shader_library.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>

#ifndef LEONIDA_TEST_VS_DEPTH_SPV
#error "LEONIDA_TEST_VS_DEPTH_SPV must name the compiled test shader"
#endif

#define EXPECT(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return 1; } \
} while (0)

using namespace d3d12test;
using XboxHeapHandleFn = std::uint64_t(WINAPI*)(void* heap);  // heap slot 9: handle in RAX

namespace {

constexpr UINT size = 16;
int g_failures = 0;

std::uint32_t f2u(float f) {
    std::uint32_t u;
    std::memcpy(&u, &f, 4);
    return u;
}
float u2f(std::uint32_t u) {
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

struct Format {
    const char* name;
    DXGI_FORMAT format;
    std::uint32_t bits;  // UNORM bits, 0 for float
    std::uint32_t texel_bytes;
};

}  // namespace

int main() {
    vktest::Device d;
    const auto device_status = d.initialize();
    if (device_status == vktest::DeviceStatus::unavailable) {
        std::fprintf(stderr, "SKIP: no usable Vulkan 1.3 device\n");
        return 77;
    }
    EXPECT(device_status == vktest::DeviceStatus::ready);
    std::printf("VK_EXT_depth_bias_control depthBiasExact: %s\n", d.depth_bias_exact ? "enabled" : "not available");
    bool d24_native = true;
    {
        const auto props = reinterpret_cast<PFN_vkGetPhysicalDeviceFormatProperties>(
            d.instance_proc(d.instance, "vkGetPhysicalDeviceFormatProperties"));
        VkFormatProperties d24{};
        props(d.physical, VK_FORMAT_D24_UNORM_S8_UINT, &d24);
        d24_native = d24.optimalTilingFeatures & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT;
    }

    const auto vs_program = fssm_program({0x56530311, 0xbf810000});
    const auto library = std::filesystem::temp_directory_path() / "leonida_depth_bias_test";
    std::filesystem::create_directories(library);
    std::filesystem::copy_file(LEONIDA_TEST_VS_DEPTH_SPV, library / "vs_depth.spv",
                               std::filesystem::copy_options::overwrite_existing);
    {
        std::ofstream index(library / "index.json", std::ios::binary);
        index << "{\"vs_depth\": {\"stage\": \"vs\", \"fssm_sha256\": \"" << leonida::sha256_hex(vs_program)
              << "\", \"bindings\": [], \"push_constant_bytes\": 144, \"input_signature\": []}}";
    }

    {
        rt::d3d12::VulkanDeviceInfo info{};
        info.get_instance_proc = d.instance_proc;
        info.instance = d.instance;
        info.physical = d.physical;
        info.device = d.device;
        info.queue = d.queue;
        info.queue_family = d.family;
        info.buffer_device_address = true;
        info.subgroup = d.subgroup;
        info.shader_dir = library.string();
        info.depth_bias_exact = d.depth_bias_exact;
        ID3D12Device* inner = nullptr;
        EXPECT(SUCCEEDED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&inner))));
        Ref<ID3D12Device> device;
        EXPECT(SUCCEEDED(rt::d3d12_xs::create_device_proxy(inner, __uuidof(ID3D12Device),
                                                           reinterpret_cast<void**>(&device.p))));
        rt::d3d12::set_device_outer(inner, device.p);

        D3D12_ROOT_PARAMETER param{};
        param.ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
        param.Constants = {0, 0, 2};
        D3D12_VERSIONED_ROOT_SIGNATURE_DESC root_desc{};
        root_desc.Version = D3D_ROOT_SIGNATURE_VERSION_1_0;
        root_desc.Desc_1_0 = {1, &param, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_NONE};
        const auto root_blob = serialize_root_signature(root_desc);
        EXPECT(!root_blob.empty());
        Ref<ID3D12RootSignature> root;
        EXPECT(SUCCEEDED(device->CreateRootSignature(0, root_blob.data(), root_blob.size(), IID_PPV_ARGS(&root.p))));
        const auto vs = shader_container(vs_program);

        D3D12_COMMAND_QUEUE_DESC queue_desc{};
        Ref<ID3D12CommandQueue> queue;
        Ref<ID3D12CommandAllocator> allocator;
        Ref<ID3D12GraphicsCommandList> list;
        Ref<ID3D12Fence> fence;
        EXPECT(SUCCEEDED(device->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&queue.p))));
        EXPECT(SUCCEEDED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator.p))));
        EXPECT(SUCCEEDED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.p, nullptr,
                                                   IID_PPV_ARGS(&list.p))));
        EXPECT(SUCCEEDED(list->Close()));
        EXPECT(SUCCEEDED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence.p))));
        UINT64 fence_value = 0;

        const Format formats[] = {{"D16_UNORM", DXGI_FORMAT_D16_UNORM, 16, 2},
                                  {"D24_UNORM_S8_UINT", DXGI_FORMAT_D24_UNORM_S8_UINT, 24, 4},
                                  {"D32_FLOAT", DXGI_FORMAT_D32_FLOAT, 0, 4}};
        for (const Format& f : formats) {
            if (f.format == DXGI_FORMAT_D24_UNORM_S8_UINT && !d24_native) {
                std::printf("SKIP %s: stored as D32_SFLOAT_S8_UINT on this GPU (D3D-309 covers that path)\n", f.name);
                continue;
            }
            D3D12_RESOURCE_DESC desc{};
            desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
            desc.Width = desc.Height = size;
            desc.DepthOrArraySize = 1;
            desc.MipLevels = 1;
            desc.Format = f.format;
            desc.SampleDesc.Count = 1;
            desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;
            D3D12_HEAP_PROPERTIES default_heap{D3D12_HEAP_TYPE_DEFAULT};
            D3D12_CLEAR_VALUE clear{};
            clear.Format = f.format;
            clear.DepthStencil.Depth = 1.0f;
            Ref<ID3D12Resource> depth, readback;
            EXPECT(SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &desc,
                                                             D3D12_RESOURCE_STATE_DEPTH_WRITE, &clear,
                                                             IID_PPV_ARGS(&depth.p))));
            D3D12_PLACED_SUBRESOURCE_FOOTPRINT foot{};
            UINT64 total = 0;
            device->GetCopyableFootprints(&desc, 0, 1, 0, &foot, nullptr, nullptr, &total);
            D3D12_HEAP_PROPERTIES readback_heap{D3D12_HEAP_TYPE_READBACK};
            const D3D12_RESOURCE_DESC readback_desc = buffer_desc(total);
            EXPECT(SUCCEEDED(device->CreateCommittedResource(&readback_heap, D3D12_HEAP_FLAG_NONE, &readback_desc,
                                                             D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                             IID_PPV_ARGS(&readback.p))));
            const D3D12_DESCRIPTOR_HEAP_DESC heap_desc{D3D12_DESCRIPTOR_HEAP_TYPE_DSV, 1,
                                                       D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0};
            Ref<ID3D12DescriptorHeap> dsv_heap;
            EXPECT(SUCCEEDED(device->CreateDescriptorHeap(&heap_desc, IID_PPV_ARGS(&dsv_heap.p))));
            const D3D12_CPU_DESCRIPTOR_HANDLE dsv{
                SIZE_T(reinterpret_cast<XboxHeapHandleFn>(slot(dsv_heap.p, 9))(dsv_heap.p))};
            device->CreateDepthStencilView(depth.p, nullptr, dsv);

            // A depth on one of the format's steps: UNORM codes 20000 / 5033000, and 0.375 for float.
            const std::uint32_t max_code = f.bits ? (1u << f.bits) - 1 : 0;
            const std::uint32_t base_code = f.bits == 16 ? 20000u : 5033000u;
            const float z = f.bits ? float(double(base_code) / double(max_code)) : 0.375f;
            for (int bias : {0, 1, 3, -5, 1000}) {
                D3D12_GRAPHICS_PIPELINE_STATE_DESC pso_desc{};
                pso_desc.pRootSignature = root.p;
                pso_desc.VS = {vs.data(), vs.size()};
                pso_desc.SampleMask = 0xffffffff;
                pso_desc.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
                pso_desc.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
                pso_desc.RasterizerState.DepthClipEnable = TRUE;
                pso_desc.RasterizerState.DepthBias = bias;
                pso_desc.DepthStencilState.DepthEnable = TRUE;
                pso_desc.DepthStencilState.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ALL;
                pso_desc.DepthStencilState.DepthFunc = D3D12_COMPARISON_FUNC_ALWAYS;
                pso_desc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
                pso_desc.DSVFormat = f.format;
                pso_desc.SampleDesc.Count = 1;
                Ref<ID3D12PipelineState> pso;
                EXPECT(SUCCEEDED(device->CreateGraphicsPipelineState(&pso_desc, IID_PPV_ARGS(&pso.p))));

                EXPECT(SUCCEEDED(allocator->Reset()) && SUCCEEDED(list->Reset(allocator.p, pso.p)));
                list->ClearDepthStencilView(dsv, D3D12_CLEAR_FLAG_DEPTH, 1.0f, 0, 0, nullptr);
                list->SetGraphicsRootSignature(root.p);
                const std::uint32_t constants[2] = {f2u(z), 0};
                list->SetGraphicsRoot32BitConstants(0, 2, constants, 0);
                list->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
                const D3D12_VIEWPORT viewport{0, 0, float(size), float(size), 0, 1};
                const D3D12_RECT scissor{0, 0, LONG(size), LONG(size)};
                list->RSSetViewports(1, &viewport);
                list->RSSetScissorRects(1, &scissor);
                list->OMSetRenderTargets(0, nullptr, FALSE, &dsv);
                list->DrawInstanced(3, 1, 0, 0);
                D3D12_RESOURCE_BARRIER to_copy = transition(depth.p, D3D12_RESOURCE_STATE_DEPTH_WRITE,
                                                            D3D12_RESOURCE_STATE_COPY_SOURCE);
                list->ResourceBarrier(1, &to_copy);
                D3D12_TEXTURE_COPY_LOCATION dst{};
                dst.pResource = readback.p;
                dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
                dst.PlacedFootprint = foot;
                D3D12_TEXTURE_COPY_LOCATION src{};
                src.pResource = depth.p;
                src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
                list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
                D3D12_RESOURCE_BARRIER back = transition(depth.p, D3D12_RESOURCE_STATE_COPY_SOURCE,
                                                         D3D12_RESOURCE_STATE_DEPTH_WRITE);
                list->ResourceBarrier(1, &back);
                EXPECT(list->Close() == S_OK);
                ID3D12CommandList* lists[] = {list.p};
                queue->ExecuteCommandLists(1, lists);
                EXPECT(rt::d3d12::last_execute_result(queue.p) == leonida::ReplayResult::success);
                EXPECT(SUCCEEDED(queue->Signal(fence.p, ++fence_value)));
                EXPECT(SUCCEEDED(fence->SetEventOnCompletion(fence_value, nullptr)));
                std::uint8_t* mapped = nullptr;
                EXPECT(SUCCEEDED(readback->Map(0, nullptr, reinterpret_cast<void**>(&mapped))));
                std::vector<std::uint32_t> texels(size * size);
                for (UINT y = 0; y < size; ++y)
                    for (UINT x = 0; x < size; ++x) {
                        std::uint32_t word = 0;
                        std::memcpy(&word, mapped + foot.Offset + y * foot.Footprint.RowPitch + x * f.texel_bytes,
                                    f.texel_bytes);
                        texels[y * size + x] = f.bits == 24 ? word & 0xffffffu : word;
                    }
                readback->Unmap(0, nullptr);

                // D3D: UNORM stores code + bias; float stores z + bias * 2^(exponent(z) - 23).
                std::uint32_t expected;
                if (f.bits) {
                    expected = std::uint32_t(std::int64_t(base_code) + bias);
                } else {
                    int exponent = 0;
                    std::frexp(z, &exponent);  // z = m * 2^exponent with m in [0.5, 1): IEEE exponent is exponent - 1
                    expected = f2u(z + float(bias) * std::ldexp(1.0f, exponent - 1 - 23));
                }
                int wrong = 0;
                for (std::uint32_t t : texels) wrong += t != expected;
                char line[240];
                if (f.bits)
                    std::snprintf(line, sizeof(line), "%s DepthBias %+d: stored code %u, D3D gives %u (%+lld steps)",
                                  f.name, bias, texels[0], expected,
                                  static_cast<long long>(std::int64_t(texels[0]) - std::int64_t(base_code)));
                else
                    std::snprintf(line, sizeof(line), "%s DepthBias %+d: stored %.9g, D3D gives %.9g", f.name, bias,
                                  u2f(texels[0]), u2f(expected));
                if (wrong == 0) {
                    std::printf("PASS %s\n", line);
                } else if (!d.depth_bias_exact) {
                    std::printf("INFO %s (no depthBiasExact on this device: not checked)\n", line);
                } else {
                    std::printf("FAIL %s, %d of %u texels\n", line, wrong, size * size);
                    ++g_failures;
                }
            }
        }
    }
    std::filesystem::remove_all(library);
    std::printf("%d failure(s)\n", g_failures);
    return g_failures ? 1 : 0;
}
