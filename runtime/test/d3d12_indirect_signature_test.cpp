// ExecuteIndirect with a root constant before the draw or dispatch, on the Vulkan-backed device: the signatures
// [CONSTANT, DISPATCH], [CONSTANT, DRAW] and [CONSTANT, DRAW_INDEXED].
// The layer replays only the final argument, read from the start of the record, so until it executes such signatures
// (#2077) CreateCommandSignature has to refuse them (E_NOTIMPL) rather than run them wrong.
//   - [CONSTANT, DISPATCH] passes when it is refused with E_NOTIMPL or runs as D3D12 says: the record {7, 1, 1, 1}
//     runs one group that sees root constant 7. The list sets the constant to 100 directly before the call, so a
//     dispatch that drops the record's constant shows 100. A plain [DISPATCH] control runs first and must run one group
//     with 100, so the harness itself is checked.
//   - [CONSTANT, DRAW] and [CONSTANT, DRAW_INDEXED] pass when they are refused with E_NOTIMPL or accepted; an accepted
//     one is not run. A draw ExecuteIndirect is not replayed with the list's pipeline, render targets and root
//     arguments until #1865 (GraphicsCommandList::ExecuteIndirect records them for a dispatch only), so running one
//     here would test #1865, not the signature. When #1865 lands, run the draws too.
// The test stops at the first case that fails. The shader is GLSL written to the translator's contract
// (test/glsl/cs_indirect_constant.comp). No game data. Set LEONIDA_VK_VALIDATION=1 to run under
// VK_LAYER_KHRONOS_validation.
#include "shader_fixture_path.h"
#include "test_temp_dir.h"
#include "vk_test_device.h"
#include "d3d12_test_shaders.h"
#include "../shims/d3d12_com.h"
#include "../src/shader_library.h"

#include <array>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#ifndef LEONIDA_TEST_INDIRECT_CS_SPV
#error "LEONIDA_TEST_INDIRECT_CS_SPV must name the compiled test shader"
#endif

using namespace d3d12test;

namespace {

constexpr UINT kConstant = 7, kDirect = 100;  // the record's constant; the one set directly before

struct Shared {
    ID3D12Device* device = nullptr;
    ID3D12CommandQueue* queue = nullptr;
    ID3D12Fence* fence = nullptr;
    UINT64 fence_value = 0;
    ID3D12RootSignature* root = nullptr;  // [0] a root UAV (s0:s1), [1] one root constant (s2)
    ID3D12PipelineState* groups = nullptr;
};

// What one ExecuteIndirect(DISPATCH) did: the number of groups, and the constant they saw + 1.
struct Observed {
    UINT group_count = 0, group_value = 0;
    bool operator==(const Observed&) const = default;
};

std::string describe(const Observed& o) {
    return std::to_string(o.group_count) + " group(s), constant + 1 = " + std::to_string(o.group_value);
}

bool create_buffer(ID3D12Device* dev, D3D12_HEAP_TYPE type, UINT64 bytes, D3D12_RESOURCE_STATES state,
                   D3D12_RESOURCE_FLAGS flags, ID3D12Resource** out) {
    D3D12_HEAP_PROPERTIES heap{};
    heap.Type = type;
    const auto desc = buffer_desc(bytes, flags);
    return SUCCEEDED(dev->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, state, nullptr,
                                                  IID_PPV_ARGS(out)));
}

bool write_buffer(ID3D12Resource* buffer, const void* data, std::size_t bytes) {
    void* mapped = nullptr;
    if (FAILED(buffer->Map(0, nullptr, &mapped)) || !mapped) return false;
    std::memcpy(mapped, data, bytes);
    buffer->Unmap(0, nullptr);
    return true;
}

bool submit(Shared& s, ID3D12GraphicsCommandList* list) {
    if (FAILED(list->Close())) return false;
    ID3D12CommandList* lists[] = {list};
    s.queue->ExecuteCommandLists(1, lists);
    if (rt::d3d12::last_execute_result(s.queue) != leonida::ReplayResult::success) return false;
    ++s.fence_value;
    const HANDLE event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    if (!event) return false;
    const bool done = SUCCEEDED(s.queue->Signal(s.fence, s.fence_value)) &&
                      SUCCEEDED(s.fence->SetEventOnCompletion(s.fence_value, event)) &&
                      WaitForSingleObject(event, 20000) == WAIT_OBJECT_0;
    CloseHandle(event);
    if (!done) std::fprintf(stderr, "the list did not finish in 20 s\n");
    return done;
}

HRESULT create_root(ID3D12Device* dev, ID3D12RootSignature** out) {
    D3D12_ROOT_PARAMETER params[2] = {};
    params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
    params[0].Descriptor = {0, 0};
    params[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    params[1].Constants = {0, 0, 1};
    for (auto& p : params) p.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    D3D12_VERSIONED_ROOT_SIGNATURE_DESC desc{};
    desc.Version = D3D_ROOT_SIGNATURE_VERSION_1_0;
    desc.Desc_1_0 = {2, params, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_NONE};
    const auto blob = serialize_root_signature(desc);
    if (blob.empty()) return E_FAIL;
    return dev->CreateRootSignature(0, blob.data(), blob.size(), IID_PPV_ARGS(out));
}

// The record: the constant (when the signature has one), then the final argument's words.
std::vector<UINT> record(D3D12_INDIRECT_ARGUMENT_TYPE final, bool with_constant) {
    std::vector<UINT> words;
    if (with_constant) words.push_back(kConstant);
    if (final == D3D12_INDIRECT_ARGUMENT_TYPE_DRAW) words.insert(words.end(), {3, 1, 0, 0});
    if (final == D3D12_INDIRECT_ARGUMENT_TYPE_DRAW_INDEXED) words.insert(words.end(), {3, 1, 0, 0, 0});
    if (final == D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH) words.insert(words.end(), {1, 1, 1});
    return words;
}

// [CONSTANT into root parameter 1, final] or [final] alone; one root signature serves every case.
HRESULT create_signature(const Shared& s, D3D12_INDIRECT_ARGUMENT_TYPE final, bool with_constant,
                         ID3D12CommandSignature** out) {
    D3D12_INDIRECT_ARGUMENT_DESC args[2] = {};
    args[0].Type = D3D12_INDIRECT_ARGUMENT_TYPE_CONSTANT;
    args[0].Constant = {1, 0, 1};
    args[1].Type = final;
    const UINT stride = UINT(record(final, with_constant).size() * sizeof(UINT));
    const D3D12_COMMAND_SIGNATURE_DESC desc{stride, with_constant ? 2u : 1u, with_constant ? args : args + 1, 0};
    return s.device->CreateCommandSignature(&desc, with_constant ? s.root : nullptr, IID_PPV_ARGS(out));
}

// Sets the direct constant, runs ExecuteIndirect(signature, 1 dispatch record) and reads the two words back.
bool execute(Shared& s, ID3D12CommandSignature* signature, bool with_constant, Observed& seen) {
    const auto words = record(D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH, with_constant);
    std::array<UINT, 64> upload_words{};  // [0, 16): the record; [32, 34): zeros for the UAV
    std::memcpy(upload_words.data(), words.data(), words.size() * sizeof(UINT));
    Ref<ID3D12Resource> upload, args, output, readback;
    Ref<ID3D12CommandAllocator> allocator;
    Ref<ID3D12GraphicsCommandList> list;
    if (!create_buffer(s.device, D3D12_HEAP_TYPE_UPLOAD, 256, D3D12_RESOURCE_STATE_GENERIC_READ,
                       D3D12_RESOURCE_FLAG_NONE, &upload.p) ||
        !write_buffer(upload.p, upload_words.data(), sizeof(upload_words)) ||
        !create_buffer(s.device, D3D12_HEAP_TYPE_DEFAULT, 256, D3D12_RESOURCE_STATE_COPY_DEST,
                       D3D12_RESOURCE_FLAG_NONE, &args.p) ||
        !create_buffer(s.device, D3D12_HEAP_TYPE_DEFAULT, 256, D3D12_RESOURCE_STATE_COPY_DEST,
                       D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, &output.p) ||
        !create_buffer(s.device, D3D12_HEAP_TYPE_READBACK, 256, D3D12_RESOURCE_STATE_COPY_DEST,
                       D3D12_RESOURCE_FLAG_NONE, &readback.p) ||
        FAILED(s.device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator.p))) ||
        FAILED(s.device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.p, nullptr,
                                           IID_PPV_ARGS(&list.p))))
        return false;
    list->CopyBufferRegion(args.p, 0, upload.p, 0, 64);
    list->CopyBufferRegion(output.p, 0, upload.p, 128, 8);
    const D3D12_RESOURCE_BARRIER ready[] = {
        transition(args.p, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT),
        transition(output.p, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS)};
    list->ResourceBarrier(2, ready);
    list->SetComputeRootSignature(s.root);
    list->SetPipelineState(s.groups);
    list->SetComputeRootUnorderedAccessView(0, output->GetGPUVirtualAddress());
    list->SetComputeRoot32BitConstant(1, kDirect, 0);
    list->ExecuteIndirect(signature, 1, args.p, 0, nullptr, 0);
    const D3D12_RESOURCE_BARRIER to_copy =
        transition(output.p, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
    list->ResourceBarrier(1, &to_copy);
    list->CopyBufferRegion(readback.p, 0, output.p, 0, 8);
    if (!submit(s, list.p)) return false;
    std::uint8_t* mapped = nullptr;
    if (FAILED(readback->Map(0, nullptr, reinterpret_cast<void**>(&mapped))) || !mapped) return false;
    std::memcpy(&seen.group_count, mapped, 4);
    std::memcpy(&seen.group_value, mapped + 4, 4);
    readback->Unmap(0, nullptr);
    return true;
}

// What D3D12 does with a dispatch record {1, 1, 1}: one group, which sees this constant.
Observed expected(UINT constant) { return {1, constant + 1}; }

// The plain [DISPATCH] control: one group with the directly set constant.
int run_control(Shared& s) {
    Ref<ID3D12CommandSignature> plain;
    Observed seen;
    if (FAILED(create_signature(s, D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH, false, &plain.p)) ||
        !execute(s, plain.p, false, seen) || !(seen == expected(kDirect))) {
        std::fprintf(stderr, "[FAIL] control [DISPATCH]: %s, want %s\n", describe(seen).c_str(),
                     describe(expected(kDirect)).c_str());
        return 1;
    }
    std::printf("[ok]   control [DISPATCH]: %s\n", describe(seen).c_str());
    return 0;
}

// [CONSTANT, DISPATCH]: refused with E_NOTIMPL, or run as D3D12 says.
int run_dispatch(Shared& s) {
    Ref<ID3D12CommandSignature> prefixed;
    const HRESULT hr = create_signature(s, D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH, true, &prefixed.p);
    if (hr == E_NOTIMPL && !prefixed.p) {
        std::printf("[ok]   [CONSTANT, DISPATCH]: refused with E_NOTIMPL\n");
        return 0;
    }
    Observed seen;
    if (FAILED(hr) || !prefixed.p || !execute(s, prefixed.p, true, seen)) {
        std::fprintf(stderr, "[FAIL] [CONSTANT, DISPATCH]: CreateCommandSignature 0x%08lx, or the list did not run\n",
                     static_cast<unsigned long>(hr));
        return 1;
    }
    const bool right = seen == expected(kConstant);
    std::fprintf(right ? stdout : stderr, "%s [CONSTANT, DISPATCH]: accepted; %s, want %s\n",
                 right ? "[ok]  " : "[FAIL]", describe(seen).c_str(), describe(expected(kConstant)).c_str());
    return right ? 0 : 1;
}

// [CONSTANT, DRAW] and [CONSTANT, DRAW_INDEXED]: refused with E_NOTIMPL, or accepted (not run until #1865).
int run_draw(Shared& s, const char* name, D3D12_INDIRECT_ARGUMENT_TYPE final) {
    Ref<ID3D12CommandSignature> prefixed;
    const HRESULT hr = create_signature(s, final, true, &prefixed.p);
    if (hr == E_NOTIMPL && !prefixed.p) {
        std::printf("[ok]   [CONSTANT, %s]: refused with E_NOTIMPL\n", name);
        return 0;
    }
    if (SUCCEEDED(hr) && prefixed.p) {
        std::printf("[ok]   [CONSTANT, %s]: accepted; not run, a draw ExecuteIndirect replays without the list's "
                    "state until #1865\n", name);
        return 0;
    }
    std::fprintf(stderr, "[FAIL] [CONSTANT, %s]: CreateCommandSignature 0x%08lx, want S_OK or E_NOTIMPL\n", name,
                 static_cast<unsigned long>(hr));
    return 1;
}

bool write_library(const std::filesystem::path& library, const std::vector<std::uint8_t>& program) {
    if (!testfiles::copy_file(vktest::shader_fixture_path(LEONIDA_TEST_INDIRECT_CS_SPV), library / "cs_ic.spv"))
        return false;
    std::ofstream index(library / "index.json", std::ios::binary);
    index << "{\"cs_ic\": {\"stage\": \"cs\", \"fssm_sha256\": \"" << leonida::sha256_hex(program)
          << "\", \"bindings\": [], \"push_constant_bytes\": 144, \"local_size\": [1, 1, 1], "
             "\"uses_code_base\": false, \"wave64\": false}}";
    return bool(index);
}

}  // namespace

int main() {
    vktest::Device vk;
    vk.request_buffer_device_address = true;
    const auto status = vk.initialize();
    if (status == vktest::DeviceStatus::unavailable) return 77;
    if (status != vktest::DeviceStatus::ready) return 1;
    if (!vk.buffer_device_address) return 77;

    const auto program = fssm_program({0x43530359, 0xbf810000});
    const testfiles::ScratchDir scratch("leonida_indirect_signature_test");
    if (!write_library(scratch.path(), program)) return 1;
    int failures = 0;
    {
        rt::d3d12::VulkanDeviceInfo info{};
        info.get_instance_proc = vk.instance_proc;
        info.instance = vk.instance;
        info.physical = vk.physical;
        info.device = vk.device;
        info.queue = vk.queue;
        info.queue_family = vk.family;
        info.buffer_device_address = vk.buffer_device_address;
        info.shader_dir = scratch.path().string();
        Ref<ID3D12Device> device;
        Ref<ID3D12CommandQueue> queue;
        Ref<ID3D12Fence> fence;
        Ref<ID3D12RootSignature> root;
        Ref<ID3D12PipelineState> groups;
        if (FAILED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&device.p)))) return 1;
        const D3D12_COMMAND_QUEUE_DESC queue_desc{};  // a direct queue
        const auto container = shader_container(program);
        D3D12_COMPUTE_PIPELINE_STATE_DESC pso{};
        pso.CS = {container.data(), container.size()};
        if (FAILED(device->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&queue.p))) ||
            FAILED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence.p))) ||
            FAILED(create_root(device.p, &root.p)))
            return 1;
        pso.pRootSignature = root.p;
        if (FAILED(device->CreateComputePipelineState(&pso, IID_PPV_ARGS(&groups.p)))) return 1;
        Shared s{device.p, queue.p, fence.p, 0, root.p, groups.p};
        failures = run_control(s);
        if (!failures) failures = run_dispatch(s);
        if (!failures) failures = run_draw(s, "DRAW", D3D12_INDIRECT_ARGUMENT_TYPE_DRAW);
        if (!failures) failures = run_draw(s, "DRAW_INDEXED", D3D12_INDIRECT_ARGUMENT_TYPE_DRAW_INDEXED);
    }
    if (vk.validation_errors()) {
        std::fprintf(stderr, "validation errors: %u\n", vk.validation_errors());
        return 1;
    }
    return failures ? 1 : 0;
}
