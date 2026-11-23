// A translated compute shader stores to a DXGI_FORMAT_B5G5R5A1_UNORM UAV texture through the D3D12 layer, and the
// texture reads back as the packed 16-bit texels of those floats (B in bits 0-4, G 5-9, R 10-14, A 15; each channel
// clamped to [0, 1] and rounded to nearest).
//
// On a GPU whose VK_FORMAT_A1R5G5B5_UNORM_PACK16 has no STORAGE_IMAGE support the texture is a mutable R16_UINT image
// (d3d12_b5g5r5a1_uav_test) whose storage view is in the uint class of set 0, while its T# still names GFX10's
// 1_5_5_5_UNORM. The translator's image_store therefore also stores such a T#'s texel packed through the uint view
// (docs/design.md, "Packed 1_5_5_5 image stores"); test/shaders/cs_b5g5r5a1_store.asm is hand-written RDNA2 that
// stores four floats a texel from a raw buffer. On a GPU with native A1R5G5B5 storage the same texels come from the
// float view; the inputs keep clear of rounding ties, which that conversion may break either way.
//
// The inputs: every level of every channel, values a quarter of a step off a level, values outside [0, 1], infinities
// and a NaN, and a spread of multiples of 1/1024. Compared with a CPU reference, texel by texel, error 0.
// Synthetic data only; no game files. Set LEONIDA_VK_VALIDATION=1 to run under VK_LAYER_KHRONOS_validation (0 errors
// required).
#include "vk_test_device.h"
#include "d3d12_test_shaders.h"
#include "../shims/d3d12_com.h"
#include "../shims/d3d12_xs_device.h"
#include "../src/gfx10_descriptors.h"
#include "../src/shader_library.h"

#include <array>
#include <cmath>
#include <cstring>
#include <limits>

#ifndef LEONIDA_TEST_B5G5R5A1_STORE_DIR
#error "LEONIDA_TEST_B5G5R5A1_STORE_DIR must name the folder build.py wrote (spirv/index.json, fssm/cs_*.fssm)"
#endif

#define EXPECT(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return 1; } \
} while (0)

using namespace d3d12test;
using XboxHeapHandleFn = std::uint64_t(WINAPI*)(void* heap);  // heap slots 9/10: handle in RAX

namespace {

std::vector<std::uint8_t> read_file(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

constexpr UINT image_dim = 16, texels = image_dim * image_dim;   // 4 groups of 64 threads, one texel a thread
constexpr UINT image_pitch = D3D12_TEXTURE_DATA_PITCH_ALIGNMENT;
constexpr UINT64 input_bytes = texels * 16;

// The four floats texel g stores (R, G, B, A).
std::array<float, 4> input(UINT g) {
    const float inf = std::numeric_limits<float>::infinity(), nan = std::numeric_limits<float>::quiet_NaN();
    const UINT k = g & 31;
    switch (g >> 5) {
    case 0: return {float(k) / 31.0f, 0.0f, 0.0f, 0.0f};                  // every level of R
    case 1: return {0.0f, float(k) / 31.0f, 0.0f, 1.0f};                  // of G, with A set
    case 2: return {0.0f, 0.0f, float(k) / 31.0f, 0.0f};                  // of B
    case 3:                                                               // a quarter of a step either side
        return {(float(k) + 0.25f) / 31.0f, (float(k) + 0.75f) / 31.0f, (float(k) - 0.25f) / 31.0f,
                k & 1 ? 0.51f : 0.49f};
    case 4: {                                                             // outside [0, 1], and not numbers
        const float special[8] = {-1.0f, 2.0f, inf, -inf, nan, -0.0f, 1.0e30f, -1.0e-30f};
        return {special[k & 7], special[(k + 3) & 7], special[(k + 5) & 7], special[(k >> 3) + (k & 4)]};
    }
    default: {                                                            // multiples of 1/1024 in [-0.25, 1.25]
        const auto value = [&](UINT salt) {
            UINT n = (g * 0x9e3779b1u + salt * 0x85ebca6bu) >> 7 & 2047;  // 0..2047
            n = n * 1536 / 2047;                                          // 0..1536
            if (n == 256 + 512) ++n;                                      // 0.5 is a tie for every channel
            return (float(n) - 256.0f) / 1024.0f;
        };
        return {value(1), value(2), value(3), value(4)};
    }
    }
}

// What the store leaves: clamp to [0, 1] (a NaN is 0), scale, round to nearest; the f32 steps the shader takes.
std::uint16_t packed(const std::array<float, 4>& c) {
    const auto level = [](float x, float top) {
        x = std::isnan(x) ? 0.0f : x < 0.0f ? 0.0f : x > 1.0f ? 1.0f : x;
        const float scaled = x * top;
        return std::uint32_t(scaled + 0.5f);
    };
    return std::uint16_t(level(c[3], 1.0f) << 15 | level(c[0], 31.0f) << 10 | level(c[1], 31.0f) << 5 |
                         level(c[2], 31.0f));
}

}  // namespace

int main() {
    vktest::Device d;
    const auto device_status = d.initialize();
    if (device_status == vktest::DeviceStatus::unavailable) return 77;   // no Vulkan GPU: ctest skips
    if (device_status != vktest::DeviceStatus::ready) return 1;
    if (!d.bindless) {
        std::printf("SKIP: the device lacks the descriptor indexing features of the bindless set 0\n");
        return 77;
    }
    VkFormatProperties a1r5g5b5{};
    d.instance_fn<PFN_vkGetPhysicalDeviceFormatProperties>("vkGetPhysicalDeviceFormatProperties")(
        d.physical, VK_FORMAT_A1R5G5B5_UNORM_PACK16, &a1r5g5b5);
    const bool native_storage = (a1r5g5b5.optimalTilingFeatures & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT) != 0;

    const std::filesystem::path root_dir = LEONIDA_TEST_B5G5R5A1_STORE_DIR;
    leonida::ShaderLibrary library;
    EXPECT(library.load((root_dir / "spirv").string()));
    const auto program = read_file(root_dir / "fssm" / "cs_b5g5r5a1_store.fssm");
    EXPECT(program.size() >= 0x150);
    EXPECT(library.find("cs", leonida::sha256_hex(program)));
    const auto container = shader_container(program);

    // The self-check of the reference: the values the layer's render-target clears read back as.
    EXPECT(packed({1.0f, 0.0f, 0.0f, 1.0f}) == 0xFC00 && packed({0.0f, 1.0f, 0.0f, 0.0f}) == 0x03E0 &&
           packed({0.0f, 0.0f, 1.0f, 0.0f}) == 0x001F);

    std::vector<std::uint8_t> got;
    {
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
        info.image_view_min_lod = d.image_view_min_lod;
        info.host_image_copy = d.host_image_copy;
        info.shader_dir = (root_dir / "spirv").string();
        ID3D12Device* inner = nullptr;
        EXPECT(SUCCEEDED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&inner))));
        Ref<ID3D12Device> device;
        EXPECT(SUCCEEDED(rt::d3d12_xs::create_device_proxy(inner, __uuidof(ID3D12Device),
                                                           reinterpret_cast<void**>(&device.p))));
        rt::d3d12::set_device_outer(inner, device.p);
        EXPECT(rt::d3d12::bindless_set(device.p) != VK_NULL_HANDLE);

        const D3D12_HEAP_PROPERTIES default_heap{D3D12_HEAP_TYPE_DEFAULT};
        const D3D12_HEAP_PROPERTIES upload_heap{D3D12_HEAP_TYPE_UPLOAD};
        const D3D12_HEAP_PROPERTIES readback_heap{D3D12_HEAP_TYPE_READBACK};
        constexpr auto uav_flag = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
        constexpr auto uav_state = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;

        D3D12_RESOURCE_DESC image_desc{};
        image_desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        image_desc.Width = image_desc.Height = image_dim;
        image_desc.DepthOrArraySize = 1;
        image_desc.MipLevels = 1;
        image_desc.Format = DXGI_FORMAT_B5G5R5A1_UNORM;
        image_desc.SampleDesc.Count = 1;
        image_desc.Flags = uav_flag;
        if (device->GetResourceAllocationInfo(0, 1, &image_desc).SizeInBytes == UINT64_MAX) {
            std::printf("SKIP: this GPU has no B5G5R5A1_UNORM UAV texture, native or as R16_UINT\n");
            return 77;
        }
        Ref<ID3D12Resource> upload, input_buffer, image, readback;
        EXPECT(SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &image_desc, uav_state,
                                                         nullptr, IID_PPV_ARGS(&image.p))));
        const auto create_buffer = [&](Ref<ID3D12Resource>& out, const D3D12_HEAP_PROPERTIES& heap, UINT64 bytes,
                                       D3D12_RESOURCE_FLAGS flags, D3D12_RESOURCE_STATES state) {
            const auto desc = buffer_desc(bytes, flags);
            return SUCCEEDED(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, state, nullptr,
                                                             IID_PPV_ARGS(&out.p)));
        };
        EXPECT(create_buffer(upload, upload_heap, input_bytes, D3D12_RESOURCE_FLAG_NONE,
                             D3D12_RESOURCE_STATE_GENERIC_READ));
        EXPECT(create_buffer(input_buffer, default_heap, input_bytes, uav_flag, D3D12_RESOURCE_STATE_COPY_DEST));
        EXPECT(create_buffer(readback, readback_heap, UINT64(image_dim) * image_pitch, D3D12_RESOURCE_FLAG_NONE,
                             D3D12_RESOURCE_STATE_COPY_DEST));
        {
            float* mapped = nullptr;
            EXPECT(SUCCEEDED(upload->Map(0, nullptr, reinterpret_cast<void**>(&mapped))));
            for (UINT g = 0; g < texels; ++g) {
                const auto value = input(g);
                std::memcpy(mapped + g * 4, value.data(), sizeof(value));
            }
            upload->Unmap(0, nullptr);
        }

        // [0] four 32-bit constants (s4..s7, unused), [1] a table of two UAVs (s8, its 32-bit VA): the console compute
        // layout (D3D-448). The fixture is built with LEONIDA_USER_DATA_SHIFT=4 (runtime/CMakeLists.txt).
        D3D12_DESCRIPTOR_RANGE uav_range{D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 2, 0, 0, 0};
        D3D12_ROOT_PARAMETER params[2] = {};
        params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
        params[0].Constants = {0, 0, 4};
        params[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
        params[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        params[1].DescriptorTable = {1, &uav_range};
        params[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
        D3D12_VERSIONED_ROOT_SIGNATURE_DESC root_desc{};
        root_desc.Version = D3D_ROOT_SIGNATURE_VERSION_1_0;
        root_desc.Desc_1_0 = {2, params, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_NONE};
        const auto root_blob = serialize_root_signature(root_desc);
        EXPECT(!root_blob.empty());
        Ref<ID3D12RootSignature> root;
        EXPECT(SUCCEEDED(device->CreateRootSignature(0, root_blob.data(), root_blob.size(), IID_PPV_ARGS(&root.p))));
        D3D12_COMPUTE_PIPELINE_STATE_DESC pso_desc{};
        pso_desc.pRootSignature = root.p;
        pso_desc.CS = {container.data(), container.size()};
        Ref<ID3D12PipelineState> pso;
        EXPECT(SUCCEEDED(device->CreateComputePipelineState(&pso_desc, IID_PPV_ARGS(&pso.p))));

        // The table: slot 0 the image, slot 1 the raw input.
        const D3D12_DESCRIPTOR_HEAP_DESC heap_desc{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 2,
                                                   D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, 0};
        Ref<ID3D12DescriptorHeap> heap;
        EXPECT(SUCCEEDED(device->CreateDescriptorHeap(&heap_desc, IID_PPV_ARGS(&heap.p))));
        const std::uint64_t cpu = reinterpret_cast<XboxHeapHandleFn>(slot(heap.p, 9))(heap.p);
        const std::uint64_t gpu = reinterpret_cast<XboxHeapHandleFn>(slot(heap.p, 10))(heap.p);
        const UINT step = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        EXPECT(cpu && gpu >= 0x10000 && gpu < (1ull << 32) && step == 32);   // the shader steps 32 bytes a slot
        const auto handle = [&](UINT i) { return D3D12_CPU_DESCRIPTOR_HANDLE{SIZE_T(cpu + std::uint64_t(i) * step)}; };
        D3D12_UNORDERED_ACCESS_VIEW_DESC image_view{};
        image_view.Format = DXGI_FORMAT_B5G5R5A1_UNORM;
        image_view.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
        device->CreateUnorderedAccessView(image.p, nullptr, &image_view, handle(0));
        D3D12_UNORDERED_ACCESS_VIEW_DESC input_view{};
        input_view.Format = DXGI_FORMAT_R32_TYPELESS;
        input_view.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
        input_view.Buffer = {0, texels * 4, 0, 0, D3D12_BUFFER_UAV_FLAG_RAW};
        device->CreateUnorderedAccessView(input_buffer.p, nullptr, &input_view, handle(1));
        // The T# the shader loads names the texture's own format, whatever the image behind it is.
        std::uint32_t tsharp[8];
        std::memcpy(tsharp, reinterpret_cast<const void*>(handle(0).ptr), sizeof(tsharp));
        EXPECT((tsharp[1] >> 20 & 0x1ff) == leonida::gfx10::format_from_vulkan(VK_FORMAT_A1R5G5B5_UNORM_PACK16));
        EXPECT((tsharp[1] >> 20 & 0x1ff) == 134);

        D3D12_COMMAND_QUEUE_DESC queue_desc{};
        queue_desc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
        Ref<ID3D12CommandQueue> queue;
        EXPECT(SUCCEEDED(device->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&queue.p))));
        Ref<ID3D12CommandAllocator> allocator;
        Ref<ID3D12GraphicsCommandList> list;
        EXPECT(SUCCEEDED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator.p))));
        EXPECT(SUCCEEDED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.p, nullptr,
                                                   IID_PPV_ARGS(&list.p))));
        Ref<ID3D12Fence> fence;
        EXPECT(SUCCEEDED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence.p))));

        list->CopyBufferRegion(input_buffer.p, 0, upload.p, 0, input_bytes);
        const auto to_uav = transition(input_buffer.p, D3D12_RESOURCE_STATE_COPY_DEST, uav_state);
        list->ResourceBarrier(1, &to_uav);
        ID3D12DescriptorHeap* heaps[] = {heap.p};
        list->SetDescriptorHeaps(1, heaps);
        list->SetComputeRootSignature(root.p);
        list->SetPipelineState(pso.p);
        const std::uint32_t constants[4] = {};
        list->SetComputeRoot32BitConstants(0, 4, constants, 0);
        list->SetComputeRootDescriptorTable(1, D3D12_GPU_DESCRIPTOR_HANDLE{gpu});
        list->Dispatch(texels / 64, 1, 1);
        const auto to_source = transition(image.p, uav_state, D3D12_RESOURCE_STATE_COPY_SOURCE);
        list->ResourceBarrier(1, &to_source);
        D3D12_TEXTURE_COPY_LOCATION dst{}, src{};
        dst.pResource = readback.p;
        dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        dst.PlacedFootprint = {0, {DXGI_FORMAT_B5G5R5A1_UNORM, image_dim, image_dim, 1, image_pitch}};
        src.pResource = image.p;
        src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
        EXPECT(list->Close() == S_OK);
        ID3D12CommandList* lists[] = {list.p};
        queue->ExecuteCommandLists(1, lists);
        EXPECT(rt::d3d12::last_execute_result(queue.p) == leonida::ReplayResult::success);
        EXPECT(SUCCEEDED(queue->Signal(fence.p, 1)) && SUCCEEDED(fence->SetEventOnCompletion(1, nullptr)));

        void* mapped = nullptr;
        EXPECT(SUCCEEDED(readback->Map(0, nullptr, &mapped)));
        got.assign(static_cast<const std::uint8_t*>(mapped),
                   static_cast<const std::uint8_t*>(mapped) + std::size_t(image_dim) * image_pitch);
        readback->Unmap(0, nullptr);

        for (auto* r : {&upload, &input_buffer, &image, &readback}) r->reset();
        list.reset(); allocator.reset(); fence.reset(); pso.reset(); root.reset(); heap.reset(); queue.reset();
        EXPECT(rt::d3d12_xs::device_proxy_ref_count(device.p) == 1);
    }

    unsigned wrong = 0;
    for (UINT g = 0; g < texels; ++g) {
        std::uint16_t texel = 0;
        std::memcpy(&texel, got.data() + std::size_t(g >> 4) * image_pitch + (g & 15) * 2, sizeof(texel));
        const auto value = input(g);
        const std::uint16_t want = packed(value);
        if (texel != want && wrong++ < 8)
            std::fprintf(stderr, "texel %u (%g, %g, %g, %g) = 0x%04x, want 0x%04x\n", g, value[0], value[1], value[2],
                         value[3], texel, want);
    }
    if (wrong) std::fprintf(stderr, "%u of %u texels differ\n", wrong, texels);
    EXPECT(!wrong);
    if (d.validation_errors()) {
        std::fprintf(stderr, "%u validation errors\n", d.validation_errors());
        return 1;
    }
    std::printf("d3d12_b5g5r5a1_store_test: OK (a translated image_store wrote %u B5G5R5A1 texels through the %s "
                "storage view; all match the packed CPU reference, 0 validation errors)\n",
                texels, native_storage ? "A1R5G5B5" : "R16_UINT");
    return 0;
}
