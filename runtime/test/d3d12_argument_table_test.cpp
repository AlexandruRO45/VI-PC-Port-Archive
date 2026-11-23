// Root arguments in the table behind s[0:1], through the Vulkan device (docs/d3d12_vulkan.md section 3.5):
//   - a compute program whose index entry says it launches with the 4 driver SGPRs and uses s0 as a pointer
//     (as XboxBvhBuilder.dll's 52 programs do) gets every root argument in a packed table, and s0 and the low byte
//     of s1 point at it;
//   - two dispatches in one list each keep the arguments they were recorded with;
//   - the allocator's Reset gives the table memory back, and a second recording works;
//   - a signature too large for the user SGPRs is accepted at the compute bind point for such a program, and still
//     refused for a program that reads its arguments from the SGPRs.
// The shader (test/glsl/cs_argument_table.comp) is written to the translator's push-constant and address contract.
// No game data. Set LEONIDA_VK_VALIDATION=1 to run under VK_LAYER_KHRONOS_validation.
#include "shader_fixture_path.h"
#include "test_temp_dir.h"
#include "vk_test_device.h"
#include "../shims/d3d12_com.h"
#include "../src/shader_library.h"

#include <cstdint>
#include <cstring>
#include <fstream>
#include <vector>

#ifndef LEONIDA_TEST_CS_TABLE_SPV
#error "LEONIDA_TEST_CS_TABLE_SPV must name the compiled test shader"
#endif

#define EXPECT(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return 1; } \
} while (0)

namespace {

using SerializeFn = HRESULT(WINAPI*)(const D3D12_VERSIONED_ROOT_SIGNATURE_DESC*, ID3DBlob**, ID3DBlob**);

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

void put32(std::vector<std::uint8_t>& out, std::uint32_t v) {
    for (int i = 0; i < 4; ++i) out.push_back(std::uint8_t(v >> (8 * i)));
}

// A stand-in program: the magic and a word that makes its hash its own.
std::vector<std::uint8_t> program(std::uint32_t word) {
    std::vector<std::uint8_t> fssm = {'F', 'S', 'S', 'M'};
    put32(fssm, word);
    for (std::uint32_t i = 0; i < 14; ++i) put32(fssm, 0xbf810000u + i);
    return fssm;
}

std::vector<std::uint8_t> serialize(const D3D12_ROOT_PARAMETER* params, UINT count) {
    D3D12_VERSIONED_ROOT_SIGNATURE_DESC desc{};
    desc.Version = D3D_ROOT_SIGNATURE_VERSION_1_0;
    desc.Desc_1_0 = {count, params, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_NONE};
    const HMODULE d3d12 = LoadLibraryExA("d3d12.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
    const auto fn = d3d12 ? reinterpret_cast<SerializeFn>(
                                reinterpret_cast<void*>(GetProcAddress(d3d12, "D3D12SerializeVersionedRootSignature")))
                          : nullptr;
    std::vector<std::uint8_t> out;
    ID3DBlob* blob = nullptr;
    if (fn && SUCCEEDED(fn(&desc, &blob, nullptr)) && blob) {
        const auto* p = static_cast<const std::uint8_t*>(blob->GetBufferPointer());
        out.assign(p, p + blob->GetBufferSize());
        blob->Release();
    }
    return out;
}

D3D12_RESOURCE_DESC buffer_desc(UINT64 bytes, D3D12_RESOURCE_FLAGS flags) {
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width = bytes;
    desc.Height = 1;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    desc.Flags = flags;
    return desc;
}

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

D3D12_ROOT_PARAMETER uav_parameter(UINT reg) {
    D3D12_ROOT_PARAMETER p{};
    p.ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
    p.Descriptor = {reg, 0};
    return p;
}

D3D12_ROOT_PARAMETER constants_parameter(UINT count) {
    D3D12_ROOT_PARAMETER p{};
    p.ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    p.Constants = {0, 0, count};
    return p;
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

    // Two programs with the same module: the index says one reads the argument table and the other the user SGPRs.
    const auto table_program = program(1), sgpr_program = program(2);
    const testfiles::ScratchDir scratch("leonida_argument_table_test");
    const auto& library = scratch.path();
    for (const char* name : {"cs_argument_table.spv", "cs_sgprs.spv"})
        EXPECT(testfiles::copy_file(vktest::shader_fixture_path(LEONIDA_TEST_CS_TABLE_SPV), library / name));
    {
        std::ofstream index(library / "index.json", std::ios::binary);
        index << "{\"cs_argument_table\": {\"stage\": \"cs\", \"fssm_sha256\": \"" << leonida::sha256_hex(table_program)
              << "\", \"bindings\": [], \"push_constant_bytes\": 144, \"local_size\": [64, 1, 1], \"user_sgprs\": 4, "
                 "\"user_sgpr_reads\": [{\"kind\": \"ptr64_lo\", \"sgpr\": 0}, "
                 "{\"kind\": \"const32\", \"sgpr\": 1}]},\n"
                 " \"cs_sgprs\": {\"stage\": \"cs\", \"fssm_sha256\": \"" << leonida::sha256_hex(sgpr_program)
              << "\", \"bindings\": [], \"push_constant_bytes\": 144, \"local_size\": [64, 1, 1], \"user_sgprs\": 14}}";
    }

    constexpr UINT elements = 64, bytes = elements * 4;
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
    Ref<ID3D12Device> device;
    EXPECT(SUCCEEDED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&device.p))));

    // [0] a root UAV, [1] three constants, [2] a root UAV: 7 dwords, table offsets 0, 8 and 20. It also fits the
    // user SGPRs from s4, as 48 of the 52 measured signatures do, and the table is used all the same.
    const D3D12_ROOT_PARAMETER small_params[] = {uav_parameter(0), constants_parameter(3), uav_parameter(1)};
    const auto small_blob = serialize(small_params, 3);
    // Eight root UAVs and 14 constants, the largest measured shape: 30 dwords, more than the 28 from s4.
    D3D12_ROOT_PARAMETER large_params[9];
    for (UINT i = 0; i < 8; ++i) large_params[i] = uav_parameter(i);
    large_params[8] = constants_parameter(14);
    const auto large_blob = serialize(large_params, 9);
    EXPECT(!small_blob.empty() && !large_blob.empty());
    Ref<ID3D12RootSignature> small_root, large_root;
    EXPECT(SUCCEEDED(device->CreateRootSignature(0, small_blob.data(), small_blob.size(),
                                                 IID_PPV_ARGS(&small_root.p))));
    EXPECT(SUCCEEDED(device->CreateRootSignature(0, large_blob.data(), large_blob.size(),
                                                 IID_PPV_ARGS(&large_root.p))));
    const rt::d3d12::RootArgumentLayout* small_layout = rt::d3d12::root_signature_layout(small_root.p);
    const rt::d3d12::RootArgumentLayout* large_layout = rt::d3d12::root_signature_layout(large_root.p);
    EXPECT(small_layout && small_layout->compute_ok && small_layout->table_ok && small_layout->table_dwords == 7);
    EXPECT(small_layout->parameters[1].table_offset == 2 && small_layout->parameters[2].table_offset == 5);
    EXPECT(large_layout && !large_layout->compute_ok && large_layout->table_ok && large_layout->table_dwords == 30);

    const auto pipeline = [&](const std::vector<std::uint8_t>& fssm, ID3D12RootSignature* root,
                              Ref<ID3D12PipelineState>& out) {
        // A bare FSSM program is what device slot 67 hands the PC CreateComputePipelineState.
        D3D12_COMPUTE_PIPELINE_STATE_DESC desc{};
        desc.pRootSignature = root;
        desc.CS = {fssm.data(), fssm.size()};
        return device->CreateComputePipelineState(&desc, IID_PPV_ARGS(&out.p));
    };
    Ref<ID3D12PipelineState> table_pso, sgpr_pso, large_table_pso, large_sgpr_pso;
    EXPECT(SUCCEEDED(pipeline(table_program, small_root.p, table_pso)));
    EXPECT(SUCCEEDED(pipeline(sgpr_program, small_root.p, sgpr_pso)));
    EXPECT(SUCCEEDED(pipeline(table_program, large_root.p, large_table_pso)));
    EXPECT(SUCCEEDED(pipeline(sgpr_program, large_root.p, large_sgpr_pso)));
    EXPECT(rt::d3d12::compute_pipeline_reads_argument_table(table_pso.p));
    EXPECT(!rt::d3d12::compute_pipeline_reads_argument_table(sgpr_pso.p));
    EXPECT(!rt::d3d12::compute_pipeline_reads_argument_table(nullptr));

    const D3D12_HEAP_PROPERTIES default_heap{D3D12_HEAP_TYPE_DEFAULT};
    const D3D12_HEAP_PROPERTIES upload_heap{D3D12_HEAP_TYPE_UPLOAD};
    const D3D12_HEAP_PROPERTIES readback_heap{D3D12_HEAP_TYPE_READBACK};
    Ref<ID3D12Resource> out_a, out_b, extra, readback;
    const auto uav = buffer_desc(bytes, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    EXPECT(SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &uav,
                                                     D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr,
                                                     IID_PPV_ARGS(&out_a.p))));
    EXPECT(SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &uav,
                                                     D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr,
                                                     IID_PPV_ARGS(&out_b.p))));
    const auto extra_desc = buffer_desc(256, D3D12_RESOURCE_FLAG_NONE);
    EXPECT(SUCCEEDED(device->CreateCommittedResource(&upload_heap, D3D12_HEAP_FLAG_NONE, &extra_desc,
                                                     D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
                                                     IID_PPV_ARGS(&extra.p))));
    const auto readback_desc = buffer_desc(2 * bytes, D3D12_RESOURCE_FLAG_NONE);
    EXPECT(SUCCEEDED(device->CreateCommittedResource(&readback_heap, D3D12_HEAP_FLAG_NONE, &readback_desc,
                                                     D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                     IID_PPV_ARGS(&readback.p))));
    constexpr std::uint32_t extra_word = 7000;
    void* mapped = nullptr;
    EXPECT(SUCCEEDED(extra->Map(0, nullptr, &mapped)));
    std::memcpy(mapped, &extra_word, sizeof(extra_word));
    const D3D12_GPU_VIRTUAL_ADDRESS va_a = out_a->GetGPUVirtualAddress(), va_b = out_b->GetGPUVirtualAddress(),
                                    va_extra = extra->GetGPUVirtualAddress();

    D3D12_COMMAND_QUEUE_DESC queue_desc{};
    Ref<ID3D12CommandQueue> queue;
    Ref<ID3D12CommandAllocator> allocator;
    Ref<ID3D12GraphicsCommandList> list;
    Ref<ID3D12Fence> fence;
    EXPECT(SUCCEEDED(device->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&queue.p))));
    EXPECT(SUCCEEDED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator.p))));
    EXPECT(SUCCEEDED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.p, nullptr,
                                               IID_PPV_ARGS(&list.p))));
    EXPECT(SUCCEEDED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence.p))));

    // Two dispatches, the second with another output and step; then the same again after the allocator's Reset,
    // with other constants, so the table memory is taken a second time.
    std::vector<std::uint32_t> first(elements), second(elements);
    UINT64 fence_value = 0;
    for (const std::uint32_t base : {100u, 9000u}) {
        const std::uint32_t constants[3] = {base, 3, 50};
        list->SetPipelineState(table_pso.p);
        list->SetComputeRootSignature(small_root.p);
        list->SetComputeRootUnorderedAccessView(0, va_a);
        list->SetComputeRoot32BitConstants(1, 3, constants, 0);
        list->SetComputeRootUnorderedAccessView(2, va_extra);
        list->Dispatch(1, 1, 1);
        list->SetComputeRootUnorderedAccessView(0, va_b);
        list->SetComputeRoot32BitConstant(1, 5, 1);
        list->Dispatch(1, 1, 1);
        D3D12_RESOURCE_BARRIER to_copy[2] = {
            transition(out_a.p, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE),
            transition(out_b.p, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE)};
        list->ResourceBarrier(2, to_copy);
        list->CopyBufferRegion(readback.p, 0, out_a.p, 0, bytes);
        list->CopyBufferRegion(readback.p, bytes, out_b.p, 0, bytes);
        D3D12_RESOURCE_BARRIER back[2] = {
            transition(out_a.p, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS),
            transition(out_b.p, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS)};
        list->ResourceBarrier(2, back);
        EXPECT(list->Close() == S_OK);
        ID3D12CommandList* lists[] = {list.p};
        queue->ExecuteCommandLists(1, lists);
        EXPECT(rt::d3d12::last_execute_result(queue.p) == leonida::ReplayResult::success);
        EXPECT(SUCCEEDED(queue->Signal(fence.p, ++fence_value)));
        EXPECT(SUCCEEDED(fence->SetEventOnCompletion(fence_value, nullptr)));
        void* results = nullptr;
        EXPECT(SUCCEEDED(readback->Map(0, nullptr, &results)));
        std::memcpy(first.data(), results, bytes);
        std::memcpy(second.data(), static_cast<const std::uint8_t*>(results) + bytes, bytes);
        readback->Unmap(0, nullptr);
        for (UINT i = 0; i < elements; ++i) {
            EXPECT(first[i] == base + 3 * i + 50 + extra_word);
            EXPECT(second[i] == base + 5 * i + 50 + extra_word);
        }
        EXPECT(allocator->Reset() == S_OK);
        EXPECT(SUCCEEDED(list->Reset(allocator.p, nullptr)));
    }

    // The large signature: a program that reads the table can be dispatched with it, one that reads the user SGPRs
    // cannot, and the small one serves both.
    const auto closes = [&](ID3D12PipelineState* pso, ID3D12RootSignature* root) {
        list->SetPipelineState(pso);
        list->SetComputeRootSignature(root);
        list->Dispatch(1, 1, 1);
        const HRESULT hr = list->Close();
        list->Reset(allocator.p, nullptr);
        return hr;
    };
    EXPECT(closes(large_table_pso.p, large_root.p) == S_OK);
    EXPECT(closes(large_sgpr_pso.p, large_root.p) == E_INVALIDARG);
    EXPECT(closes(sgpr_pso.p, small_root.p) == S_OK);
    EXPECT(closes(table_pso.p, small_root.p) == S_OK);

    std::printf("d3d12_argument_table_test: two dispatches and a second recording read their arguments from the "
                "table behind s[0:1]; a 30-dword signature is usable by a program that reads the table\n");
    return 0;
}
