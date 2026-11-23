// Mesh PSOs and DispatchMesh on a device without VK_EXT_mesh_shader, through `--mesh-compute` (docs/d3d12_vulkan.md,
// "Mesh shaders without VK_EXT_mesh_shader"). The layer's device is made without the extension on every GPU, so this
// runs everywhere d3d12_draw_test does. The mesh shader is test/glsl/ms_draw_compute.comp: d3d12_mesh_test's quad
// shader written to the --mesh-compute contract (a compute shader writing one output slot per workgroup), with the
// index entry's "mesh_compute" layout. The layer builds the vertex shader that draws the slots (mesh_compute_vs.h).
//   - MeshShaderTier is 1 (the index has --mesh-compute mesh shaders), and an MS + PS PSO from a pipeline state
//     stream is created. A mesh shader without a mesh_compute layout is still refused (E_NOTIMPL).
//   - Draws, as in d3d12_mesh_test: an empty grid records nothing; DispatchMesh(2, 2, 1) with root constants (0, 0)
//     and a red tint fills the top-left quadrant; DispatchMesh(1, 1, 2) moved by (1, -1) with a green tint fills the
//     top half of the bottom-right one, so the X, Y and Z counts each reach the slot index. The rest stays blue.
//     Each pixel is read back and compared.
//   - A vertex draw with the mesh PSO fails at execution (ReplayResult::invalid_draw).
// Shaders: test/glsl/ms_draw_compute.comp, ps_draw.frag. No game data.
// Set LEONIDA_VK_VALIDATION=1 to run under VK_LAYER_KHRONOS_validation.
#include "vk_test_device.h"
#include "d3d12_test_shaders.h"
#include "../shims/d3d12_com.h"
#include "../shims/d3d12_xs_device.h"
#include "../src/shader_library.h"

#include <array>
#include <cstring>

#if !defined(LEONIDA_TEST_MS_COMPUTE_SPV) || !defined(LEONIDA_TEST_PS_SPV)
#error "LEONIDA_TEST_MS_COMPUTE_SPV and LEONIDA_TEST_PS_SPV must name the compiled test shaders"
#endif

#define EXPECT(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return 1; } \
} while (0)

using namespace d3d12test;
using XboxHeapHandleFn = std::uint64_t(WINAPI*)(void* heap);  // heap slot 9: handle in RAX

namespace {

// One pipeline state stream subobject as d3dx12.h declares it (CD3DX12_PIPELINE_STATE_STREAM_SUBOBJECT).
template <D3D12_PIPELINE_STATE_SUBOBJECT_TYPE Type, typename Inner>
struct alignas(void*) Subobject {
    D3D12_PIPELINE_STATE_SUBOBJECT_TYPE type = Type;
    Inner inner;
};
using RootSubobject = Subobject<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_ROOT_SIGNATURE, ID3D12RootSignature*>;
using PsSubobject = Subobject<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PS, D3D12_SHADER_BYTECODE>;
using MsSubobject = Subobject<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_MS, D3D12_SHADER_BYTECODE>;
using TargetsSubobject = Subobject<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RENDER_TARGET_FORMATS, D3D12_RT_FORMAT_ARRAY>;

struct Stream {
    std::vector<std::uint8_t> bytes;
    template <typename S>
    Stream& add(const S& subobject) {
        static_assert(sizeof(S) % sizeof(void*) == 0);
        const std::size_t at = bytes.size();
        bytes.resize(at + sizeof(S));
        std::memcpy(bytes.data() + at, &subobject, sizeof(S));
        return *this;
    }
};

HRESULT create(ID3D12Device* device, ID3D12RootSignature* root, const D3D12_SHADER_BYTECODE& ms,
               const D3D12_SHADER_BYTECODE& ps, Ref<ID3D12PipelineState>& out) {
    D3D12_RT_FORMAT_ARRAY formats{};
    formats.RTFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
    formats.NumRenderTargets = 1;
    Stream stream;
    stream.add(RootSubobject{.inner = root}).add(MsSubobject{.inner = ms}).add(PsSubobject{.inner = ps});
    stream.add(TargetsSubobject{.inner = formats});
    Ref<ID3D12Device2> device2;
    if (FAILED(device->QueryInterface(IID_PPV_ARGS(&device2.p)))) return E_NOINTERFACE;
    const D3D12_PIPELINE_STATE_STREAM_DESC desc{stream.bytes.size(), stream.bytes.data()};
    out.reset();
    return device2->CreatePipelineState(&desc, IID_PPV_ARGS(&out.p));
}

D3D12_MESH_SHADER_TIER mesh_tier(ID3D12Device* device) {
    D3D12_FEATURE_DATA_D3D12_OPTIONS7 options{};
    if (FAILED(device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS7, &options, sizeof(options))))
        return D3D12_MESH_SHADER_TIER(-1);
    return options.MeshShaderTier;
}

}  // namespace

int main() {
    vktest::Device d;
    const auto device_status = d.initialize();
    if (device_status == vktest::DeviceStatus::unavailable) {
        std::fprintf(stderr, "SKIP: no usable Vulkan 1.3 device\n");
        return 77;
    }
    EXPECT(device_status == vktest::DeviceStatus::ready);

    const auto ms_program = fssm_program({0x4d530001, 0xbf810000});
    const auto ms_plain_program = fssm_program({0x4d530002, 0xbf810000});
    const auto ps_program = fssm_program({0x50530001, 0xbf810000});
    const auto ms_container = shader_container(ms_program), ms_plain_container = shader_container(ms_plain_program),
               ps_container = shader_container(ps_program);
    const auto library = std::filesystem::temp_directory_path() / "leonida_mesh_compute_test";
    std::filesystem::create_directories(library);
    const auto overwrite = std::filesystem::copy_options::overwrite_existing;
    std::filesystem::copy_file(LEONIDA_TEST_MS_COMPUTE_SPV, library / "ms_draw_compute.spv", overwrite);
    std::filesystem::copy_file(LEONIDA_TEST_MS_COMPUTE_SPV, library / "ms_plain.spv", overwrite);
    std::filesystem::copy_file(LEONIDA_TEST_PS_SPV, library / "ps_draw.spv", overwrite);
    {
        // ms_draw_compute's layout is rdna2spirv.mesh_compute_layout(4, 2, 0, 0, [0]). ms_plain has the same module
        // but no layout, as a mesh shader translated without --mesh-compute has.
        std::ofstream index(library / "index.json", std::ios::binary);
        index << "{\"ms_draw_compute\": {\"stage\": \"ms\", \"fssm_sha256\": \"" << leonida::sha256_hex(ms_program)
              << "\", \"bindings\": [], \"push_constant_bytes\": 152, \"local_size\": [32, 1, 1], "
                 "\"max_vertices\": 4, \"max_primitives\": 2, \"mesh_compute\": {\"slot_bytes\": 176, "
                 "\"vertices\": 16, \"vertex_stride\": 32, \"primitives\": 144, \"primitive_stride\": 16, "
                 "\"max_vertices\": 4, \"max_primitives\": 2, \"clip_count\": 0, \"clip_offset\": 16, "
                 "\"cull_count\": 0, \"cull_offset\": 16, \"params\": {\"0\": 16}, \"push_constant_bytes\": 152, "
                 "\"address_dwords\": [36, 37]}},\n"
                 " \"ms_plain\": {\"stage\": \"ms\", \"fssm_sha256\": \"" << leonida::sha256_hex(ms_plain_program)
              << "\", \"bindings\": [], \"push_constant_bytes\": 144, \"local_size\": [32, 1, 1], "
                 "\"max_vertices\": 4, \"max_primitives\": 2},\n"
                 " \"ps_draw\": {\"stage\": \"ps\", \"fssm_sha256\": \"" << leonida::sha256_hex(ps_program)
              << "\", \"bindings\": [], \"push_constant_bytes\": 144, \"inputs\": [0], \"outputs\": [0]}}";
    }

    constexpr UINT size = 64;
    std::vector<std::uint8_t> pixels(size * 256);
    {
        rt::d3d12::VulkanDeviceInfo info{};
        info.get_instance_proc = d.instance_proc;
        info.instance = d.instance;
        info.physical = d.physical;
        info.device = d.device;
        info.queue = d.queue;
        info.queue_family = d.family;
        info.buffer_device_address = true;
        info.shader_dir = library.string();
        info.mesh_shader = false;  // as on a GPU without VK_EXT_mesh_shader, whatever this one has
        ID3D12Device* inner = nullptr;
        EXPECT(SUCCEEDED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&inner))));
        Ref<ID3D12Device> device;
        EXPECT(SUCCEEDED(rt::d3d12_xs::create_device_proxy(inner, __uuidof(ID3D12Device),
                                                           reinterpret_cast<void**>(&device.p))));
        rt::d3d12::set_device_outer(inner, device.p);
        EXPECT(mesh_tier(device.p) == D3D12_MESH_SHADER_TIER_1);

        // [0] two constants (s0, s1: offset), [1] a root CBV (s2:s3: tint), as in d3d12_mesh_test.
        D3D12_ROOT_PARAMETER params[2] = {};
        params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
        params[0].Constants = {0, 0, 2};
        params[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
        params[1].Descriptor = {1, 0};
        D3D12_VERSIONED_ROOT_SIGNATURE_DESC root_desc{};
        root_desc.Version = D3D_ROOT_SIGNATURE_VERSION_1_0;
        root_desc.Desc_1_0 = {2, params, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_NONE};
        const auto blob = serialize_root_signature(root_desc);
        EXPECT(!blob.empty());
        Ref<ID3D12RootSignature> root;
        EXPECT(SUCCEEDED(device->CreateRootSignature(0, blob.data(), blob.size(), IID_PPV_ARGS(&root.p))));

        const D3D12_SHADER_BYTECODE ms{ms_container.data(), ms_container.size()};
        const D3D12_SHADER_BYTECODE ms_plain{ms_plain_container.data(), ms_plain_container.size()};
        const D3D12_SHADER_BYTECODE ps{ps_container.data(), ps_container.size()};
        Ref<ID3D12PipelineState> mesh_pso, refused;
        EXPECT(SUCCEEDED(create(device.p, root.p, ms, ps, mesh_pso)));
        EXPECT(create(device.p, root.p, ms_plain, ps, refused) == E_NOTIMPL && !refused.p);

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
                                                         D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&target.p))));

        // Tints at 0 (red stays red) and 256 (green); CBVs are 256-byte aligned.
        const float tints[2][3] = {{1, 0, 1}, {0, 1, 0}};
        std::uint8_t* up = nullptr;
        EXPECT(SUCCEEDED(upload->Map(0, nullptr, reinterpret_cast<void**>(&up))));
        for (int i = 0; i < 2; ++i) std::memcpy(up + 256 * i, tints[i], sizeof(tints[i]));
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
        Ref<ID3D12GraphicsCommandList6> list6;
        Ref<ID3D12Fence> fence;
        EXPECT(SUCCEEDED(device->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&queue.p))));
        EXPECT(SUCCEEDED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator.p))));
        EXPECT(SUCCEEDED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.p, mesh_pso.p,
                                                   IID_PPV_ARGS(&list.p))));
        EXPECT(SUCCEEDED(list->QueryInterface(IID_PPV_ARGS(&list6.p))));
        EXPECT(SUCCEEDED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence.p))));

        D3D12_RESOURCE_BARRIER to_target =
            transition(target.p, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_RENDER_TARGET);
        list->ResourceBarrier(1, &to_target);
        const FLOAT blue[4] = {0, 0, 1, 1};
        list->ClearRenderTargetView(rtv, blue, 0, nullptr);
        list->SetGraphicsRootSignature(root.p);
        const D3D12_VIEWPORT viewport{0, 0, float(size), float(size), 0, 1};
        const D3D12_RECT scissor{0, 0, LONG(size), LONG(size)};
        list->RSSetViewports(1, &viewport);
        list->RSSetScissorRects(1, &scissor);
        list->OMSetRenderTargets(1, &rtv, FALSE, nullptr);
        // Mesh draw 1: a 2x2 grid of quads, no offset, red -> the top-left quadrant, after an empty grid.
        const float origin[2] = {0, 0}, down_right[2] = {1, -1};
        list->SetGraphicsRoot32BitConstants(0, 2, origin, 0);
        list->SetGraphicsRootConstantBufferView(1, upload_va);
        list6->DispatchMesh(0, 1, 1);
        list6->DispatchMesh(2, 2, 1);
        // Mesh draw 2: groups (0,0,0) and (0,0,1), columns 0 and 1 of row 0, moved by (1, -1), green -> the top half
        // of the bottom-right quadrant.
        list->SetGraphicsRoot32BitConstants(0, 2, down_right, 0);
        list->SetGraphicsRootConstantBufferView(1, upload_va + 256);
        list6->DispatchMesh(1, 1, 2);
        D3D12_RESOURCE_BARRIER to_copy =
            transition(target.p, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_COPY_SOURCE);
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
        EXPECT(SUCCEEDED(queue->Signal(fence.p, 1)) && SUCCEEDED(fence->SetEventOnCompletion(1, nullptr)));
        void* mapped = nullptr;
        EXPECT(SUCCEEDED(readback->Map(0, nullptr, &mapped)));
        std::memcpy(pixels.data(), mapped, pixels.size());

        // A vertex draw with the mesh PSO fails at execution.
        EXPECT(SUCCEEDED(allocator->Reset()) && SUCCEEDED(list->Reset(allocator.p, mesh_pso.p)));
        list->SetGraphicsRootSignature(root.p);
        list->RSSetViewports(1, &viewport);
        list->RSSetScissorRects(1, &scissor);
        list->OMSetRenderTargets(1, &rtv, FALSE, nullptr);
        list->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        list->DrawInstanced(3, 1, 0, 0);
        EXPECT(list->Close() == S_OK);
        queue->ExecuteCommandLists(1, lists);
        EXPECT(rt::d3d12::last_execute_result(queue.p) == leonida::ReplayResult::invalid_draw);
        EXPECT(SUCCEEDED(queue->Signal(fence.p, 2)) && SUCCEEDED(fence->SetEventOnCompletion(2, nullptr)));

        for (auto* r : {&upload, &readback, &target}) r->reset();
        for (auto* p : {&mesh_pso, &refused}) p->reset();
        root.reset();
        rtv_heap.reset();
        list6.reset();
        list.reset();
        allocator.reset();
        queue.reset();
        fence.reset();
        EXPECT(rt::d3d12_xs::device_proxy_ref_count(device.p) == 1);
    }
    std::filesystem::remove_all(library);

    unsigned mismatches = 0;
    for (UINT row = 0; row < size; ++row) {
        for (UINT col = 0; col < size; ++col) {
            const bool top = row < size / 2, left = col < size / 2;
            using Pixel = std::array<std::uint8_t, 4>;
            const Pixel want = top && left                            ? Pixel{255, 0, 0, 255}
                               : !top && !left && row < size * 3 / 4  ? Pixel{0, 255, 0, 255}
                                                                      : Pixel{0, 0, 255, 255};
            const std::uint8_t* got = pixels.data() + row * 256 + col * 4;
            if (std::equal(want.begin(), want.end(), got)) continue;
            if (mismatches++ < 6)
                std::fprintf(stderr, "pixel (%u, %u): got %u,%u,%u,%u want %u,%u,%u,%u\n", col, row, got[0], got[1],
                             got[2], got[3], want[0], want[1], want[2], want[3]);
        }
    }
    EXPECT(!mismatches);
    if (d.validation_errors()) {
        std::fprintf(stderr, "%u validation errors\n", d.validation_errors());
        return 1;
    }
    std::printf("mesh compute: without VK_EXT_mesh_shader, mesh tier 1 through --mesh-compute; a stream MS + PS PSO "
                "drew DispatchMesh over X, Y and Z (each slot a workgroup's quad) and matched all %u pixels; a mesh "
                "shader without a mesh_compute layout refused; a vertex draw with the mesh PSO fails at execution\n",
                size * size);
    return 0;
}
