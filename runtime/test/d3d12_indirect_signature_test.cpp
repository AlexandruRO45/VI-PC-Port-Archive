// ExecuteIndirect with a root constant before the draw or dispatch, on the Vulkan-backed device: the signatures
// [CONSTANT, DRAW], [CONSTANT, DRAW_INDEXED] and [CONSTANT, DISPATCH]. Each record is {7, then the draw or dispatch:
// 3 vertices or indices and 1 instance, or 1x1x1 groups}; D3D12 runs it once with root constant 7. Before that the list
// sets the constant to 100 directly, so a draw that drops the record's constant shows 100.
// The layer replays only the final argument, read from the start of the record, so until it executes such signatures
// (#2077) CreateCommandSignature has to refuse them (E_NOTIMPL) rather than run them wrong. A case passes when the
// signature is refused with E_NOTIMPL or runs as D3D12 says; the test stops at the first case that fails. The plain
// [DISPATCH] signature runs first as a control: it must dispatch exactly what its record says. Draws have no control:
// ExecuteIndirect draws do not render through the command list yet (#1865), so a draw case runs only when its
// signature is accepted, and a list that does not finish in 20 s fails it.
// What is drawn is read back: vertex v of instance n is a point on pixel (v, n) of a 16x4 R32_FLOAT target, written
// with the constant + 1. A dispatch counts its groups and records the constant + 1. Shaders are GLSL written to the
// translator's contract (test/glsl/cs_indirect_constant.comp, vs_indirect_points.vert, ps_indirect_constant.frag).
// No game data. Set LEONIDA_VK_VALIDATION=1 to run under VK_LAYER_KHRONOS_validation.
#include "shader_fixture_path.h"
#include "test_temp_dir.h"
#include "vk_test_device.h"
#include "d3d12_test_shaders.h"
#include "../shims/d3d12_com.h"
#include "../src/shader_library.h"

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#if !defined(LEONIDA_TEST_INDIRECT_CS_SPV) || !defined(LEONIDA_TEST_INDIRECT_VS_SPV) || \
    !defined(LEONIDA_TEST_INDIRECT_PS_SPV)
#error "LEONIDA_TEST_INDIRECT_CS_SPV, _VS_SPV and _PS_SPV must name the compiled test shaders"
#endif

using namespace d3d12test;

namespace {

constexpr UINT kWidth = 16, kHeight = 4, kPitch = 256;  // a row of 16 floats, padded to the copy pitch
constexpr UINT kConstant = 7, kDirect = 100;             // the record's constant; the one set directly before

struct Shared {
    ID3D12Device* device = nullptr;
    ID3D12CommandQueue* queue = nullptr;
    ID3D12Fence* fence = nullptr;
    UINT64 fence_value = 0;
    ID3D12RootSignature* graphics_root = nullptr;  // [0] one root constant (s0)
    ID3D12RootSignature* compute_root = nullptr;   // [0] a root UAV (s0:s1), [1] one root constant (s2)
    ID3D12PipelineState* points = nullptr;
    ID3D12PipelineState* groups = nullptr;
};

// What one ExecuteIndirect did: the drawn pixels (value = constant + 1), or the group count and constant + 1.
struct Observed {
    std::vector<std::array<UINT, 3>> pixels;  // col, row, value
    UINT group_count = 0, group_value = 0;
    bool operator==(const Observed&) const = default;
};

std::string describe(const Observed& o, bool dispatch) {
    if (dispatch) return std::to_string(o.group_count) + " group(s), constant + 1 = " + std::to_string(o.group_value);
    if (o.pixels.empty()) return "nothing drawn";
    UINT c0 = kWidth, c1 = 0, r0 = kHeight, r1 = 0;
    std::string values;
    for (const auto& p : o.pixels) {
        c0 = std::min(c0, p[0]), c1 = std::max(c1, p[0]), r0 = std::min(r0, p[1]), r1 = std::max(r1, p[1]);
        const std::string v = " " + std::to_string(p[2]);
        if ((values + " ").find(v + " ") == std::string::npos) values += v;
    }
    return std::to_string(o.pixels.size()) + " point(s): vertices " + std::to_string(c0) + ".." + std::to_string(c1) +
           ", instances " + std::to_string(r0) + ".." + std::to_string(r1) + ", constant + 1 =" + values;
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

HRESULT create_root(ID3D12Device* dev, bool compute, ID3D12RootSignature** out) {
    D3D12_ROOT_PARAMETER params[2] = {};
    params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
    params[0].Descriptor = {0, 0};
    D3D12_ROOT_PARAMETER& constant = params[compute ? 1 : 0];
    constant.ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    constant.Constants = {0, 0, 1};
    for (auto& p : params) p.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    D3D12_VERSIONED_ROOT_SIGNATURE_DESC desc{};
    desc.Version = D3D_ROOT_SIGNATURE_VERSION_1_0;
    desc.Desc_1_0 = {compute ? 2u : 1u, params, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_NONE};
    const auto blob = serialize_root_signature(desc);
    if (blob.empty()) return E_FAIL;
    return dev->CreateRootSignature(0, blob.data(), blob.size(), IID_PPV_ARGS(out));
}

HRESULT create_pipelines(Shared& s, const std::vector<std::uint8_t>& cs, const std::vector<std::uint8_t>& vs,
                         const std::vector<std::uint8_t>& ps) {
    D3D12_COMPUTE_PIPELINE_STATE_DESC compute{};
    compute.pRootSignature = s.compute_root;
    compute.CS = {cs.data(), cs.size()};
    HRESULT hr = s.device->CreateComputePipelineState(&compute, IID_PPV_ARGS(&s.groups));
    if (FAILED(hr)) return hr;
    D3D12_GRAPHICS_PIPELINE_STATE_DESC desc{};
    desc.pRootSignature = s.graphics_root;
    desc.VS = {vs.data(), vs.size()};
    desc.PS = {ps.data(), ps.size()};
    desc.BlendState.RenderTarget[0].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
    desc.SampleMask = 0xffffffff;
    desc.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
    desc.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
    desc.RasterizerState.DepthClipEnable = TRUE;
    desc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_POINT;
    desc.NumRenderTargets = 1;
    desc.RTVFormats[0] = DXGI_FORMAT_R32_FLOAT;
    desc.SampleDesc.Count = 1;
    return s.device->CreateGraphicsPipelineState(&desc, IID_PPV_ARGS(&s.points));
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

HRESULT create_signature(const Shared& s, D3D12_INDIRECT_ARGUMENT_TYPE final, bool with_constant,
                         ID3D12CommandSignature** out) {
    const bool dispatch = final == D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH;
    D3D12_INDIRECT_ARGUMENT_DESC args[2] = {};
    args[0].Type = D3D12_INDIRECT_ARGUMENT_TYPE_CONSTANT;
    args[0].Constant = {dispatch ? 1u : 0u, 0, 1};
    args[1].Type = final;
    const UINT stride = UINT(record(final, with_constant).size() * sizeof(UINT));
    const D3D12_COMMAND_SIGNATURE_DESC desc{stride, with_constant ? 2u : 1u, with_constant ? args : args + 1, 0};
    ID3D12RootSignature* root = with_constant ? (dispatch ? s.compute_root : s.graphics_root) : nullptr;
    return s.device->CreateCommandSignature(&desc, root, IID_PPV_ARGS(out));
}

// Records the state a draw or a dispatch needs, with the constant set directly to kDirect.
void bind(Shared& s, ID3D12GraphicsCommandList* list, bool dispatch, ID3D12Resource* output, ID3D12Resource* upload,
          D3D12_CPU_DESCRIPTOR_HANDLE rtv) {
    if (dispatch) {
        list->SetComputeRootSignature(s.compute_root);
        list->SetPipelineState(s.groups);
        list->SetComputeRootUnorderedAccessView(0, output->GetGPUVirtualAddress());
        list->SetComputeRoot32BitConstant(1, kDirect, 0);
        return;
    }
    const FLOAT clear[4] = {};
    list->ClearRenderTargetView(rtv, clear, 0, nullptr);
    list->SetGraphicsRootSignature(s.graphics_root);
    list->SetPipelineState(s.points);
    list->SetGraphicsRoot32BitConstant(0, kDirect, 0);
    list->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_POINTLIST);
    const D3D12_INDEX_BUFFER_VIEW indices{upload->GetGPUVirtualAddress() + 64, 64, DXGI_FORMAT_R32_UINT};
    list->IASetIndexBuffer(&indices);
    const D3D12_VIEWPORT viewport{0, 0, float(kWidth), float(kHeight), 0, 1};
    const D3D12_RECT scissor{0, 0, LONG(kWidth), LONG(kHeight)};
    list->RSSetViewports(1, &viewport);
    list->RSSetScissorRects(1, &scissor);
    list->OMSetRenderTargets(1, &rtv, FALSE, nullptr);
}

Observed read_back(const std::uint8_t* mapped, bool dispatch) {
    Observed seen;
    if (dispatch) {
        std::memcpy(&seen.group_count, mapped, 4);
        std::memcpy(&seen.group_value, mapped + 4, 4);
        return seen;
    }
    for (UINT row = 0; row < kHeight; ++row)
        for (UINT col = 0; col < kWidth; ++col) {
            float v = 0;
            std::memcpy(&v, mapped + row * kPitch + col * 4, 4);
            if (v != 0) seen.pixels.push_back({col, row, UINT(v)});
        }
    return seen;
}

// Clears the target, sets the direct constant, runs ExecuteIndirect(signature, 1 record) and reads the result back.
bool execute(Shared& s, ID3D12CommandSignature* signature, D3D12_INDIRECT_ARGUMENT_TYPE final, bool with_constant,
             Observed& seen) {
    const bool dispatch = final == D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH;
    const auto words = record(final, with_constant);
    std::array<UINT, 64> upload_words{};  // [0, 16): the record; [16, 32): indices 0..15; [32, 34): zeros for the UAV
    std::memcpy(upload_words.data(), words.data(), words.size() * sizeof(UINT));
    for (UINT i = 0; i < 16; ++i) upload_words[16 + i] = i;
    Ref<ID3D12Resource> upload, args, output, target, readback;
    Ref<ID3D12CommandAllocator> allocator;
    Ref<ID3D12GraphicsCommandList> list;
    Ref<ID3D12DescriptorHeap> rtv_heap;
    D3D12_RESOURCE_DESC rt_desc{};
    rt_desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    rt_desc.Width = kWidth;
    rt_desc.Height = kHeight;
    rt_desc.DepthOrArraySize = rt_desc.MipLevels = 1;
    rt_desc.Format = DXGI_FORMAT_R32_FLOAT;
    rt_desc.SampleDesc.Count = 1;
    rt_desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
    D3D12_HEAP_PROPERTIES default_heap{};
    default_heap.Type = D3D12_HEAP_TYPE_DEFAULT;
    const D3D12_DESCRIPTOR_HEAP_DESC rtv_desc{D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 1, D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0};
    if (!create_buffer(s.device, D3D12_HEAP_TYPE_UPLOAD, 256, D3D12_RESOURCE_STATE_GENERIC_READ,
                       D3D12_RESOURCE_FLAG_NONE, &upload.p) ||
        !write_buffer(upload.p, upload_words.data(), sizeof(upload_words)) ||
        !create_buffer(s.device, D3D12_HEAP_TYPE_DEFAULT, 256, D3D12_RESOURCE_STATE_COPY_DEST,
                       D3D12_RESOURCE_FLAG_NONE, &args.p) ||
        !create_buffer(s.device, D3D12_HEAP_TYPE_DEFAULT, 256, D3D12_RESOURCE_STATE_COPY_DEST,
                       D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, &output.p) ||
        !create_buffer(s.device, D3D12_HEAP_TYPE_READBACK, kPitch * kHeight, D3D12_RESOURCE_STATE_COPY_DEST,
                       D3D12_RESOURCE_FLAG_NONE, &readback.p) ||
        FAILED(s.device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &rt_desc,
                                                 D3D12_RESOURCE_STATE_RENDER_TARGET, nullptr, IID_PPV_ARGS(&target.p))) ||
        FAILED(s.device->CreateDescriptorHeap(&rtv_desc, IID_PPV_ARGS(&rtv_heap.p))) ||
        FAILED(s.device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator.p))) ||
        FAILED(s.device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.p, nullptr,
                                           IID_PPV_ARGS(&list.p))))
        return false;
    const D3D12_CPU_DESCRIPTOR_HANDLE rtv = rtv_heap->GetCPUDescriptorHandleForHeapStart();
    s.device->CreateRenderTargetView(target.p, nullptr, rtv);
    list->CopyBufferRegion(args.p, 0, upload.p, 0, 64);
    list->CopyBufferRegion(output.p, 0, upload.p, 128, 8);
    const D3D12_RESOURCE_BARRIER ready[] = {
        transition(args.p, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT),
        transition(output.p, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS)};
    list->ResourceBarrier(2, ready);
    bind(s, list.p, dispatch, output.p, upload.p, rtv);
    list->ExecuteIndirect(signature, 1, args.p, 0, nullptr, 0);
    const D3D12_RESOURCE_BARRIER to_copy[] = {
        transition(output.p, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE),
        transition(target.p, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_COPY_SOURCE)};
    list->ResourceBarrier(2, to_copy);
    if (dispatch) {
        list->CopyBufferRegion(readback.p, 0, output.p, 0, 8);
    } else {
        D3D12_TEXTURE_COPY_LOCATION dst{}, src{};
        dst.pResource = readback.p;
        dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        dst.PlacedFootprint = {0, {DXGI_FORMAT_R32_FLOAT, kWidth, kHeight, 1, kPitch}};
        src.pResource = target.p;
        src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
    }
    if (!submit(s, list.p)) return false;
    std::uint8_t* mapped = nullptr;
    if (FAILED(readback->Map(0, nullptr, reinterpret_cast<void**>(&mapped))) || !mapped) return false;
    seen = read_back(mapped, dispatch);
    readback->Unmap(0, nullptr);
    return true;
}

// What D3D12 does with the record: 3 points on vertices 0..2 of instance 0, or one group, with this constant.
Observed expected(bool dispatch, UINT constant) {
    Observed o;
    if (dispatch) return {{}, 1, constant + 1};
    for (UINT v = 0; v < 3; ++v) o.pixels.push_back({v, 0, constant + 1});
    return o;
}

struct Case {
    const char* name;
    D3D12_INDIRECT_ARGUMENT_TYPE final;
};

// The plain [DISPATCH] control: one group with the directly set constant.
int run_control(Shared& s) {
    const auto dispatch = D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH;
    Ref<ID3D12CommandSignature> plain;
    Observed seen;
    if (FAILED(create_signature(s, dispatch, false, &plain.p)) || !execute(s, plain.p, dispatch, false, seen) ||
        !(seen == expected(true, kDirect))) {
        std::fprintf(stderr, "[FAIL] control [DISPATCH]: %s, want %s\n", describe(seen, true).c_str(),
                     describe(expected(true, kDirect), true).c_str());
        return 1;
    }
    std::printf("[ok]   control [DISPATCH]: %s\n", describe(seen, true).c_str());
    return 0;
}

// The signature with a constant before the final argument: refused with E_NOTIMPL, or run as D3D12 says.
int run_case(Shared& s, const Case& c) {
    const bool dispatch = c.final == D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH;
    Ref<ID3D12CommandSignature> prefixed;
    Observed seen;
    const HRESULT hr = create_signature(s, c.final, true, &prefixed.p);
    if (hr == E_NOTIMPL && !prefixed.p) {
        std::printf("[ok]   [CONSTANT, %s]: refused with E_NOTIMPL\n", c.name);
        return 0;
    }
    std::printf("       [CONSTANT, %s]: CreateCommandSignature 0x%08lx\n", c.name, static_cast<unsigned long>(hr));
    std::fflush(stdout);
    if (FAILED(hr) || !execute(s, prefixed.p, c.final, true, seen)) {
        std::fprintf(stderr, "[FAIL] [CONSTANT, %s]: CreateCommandSignature 0x%08lx, or the list did not run\n", c.name,
                     static_cast<unsigned long>(hr));
        return 1;
    }
    const bool right = seen == expected(dispatch, kConstant);
    std::fprintf(right ? stdout : stderr, "%s [CONSTANT, %s]: accepted; %s, want %s\n", right ? "[ok]  " : "[FAIL]",
                 c.name, describe(seen, dispatch).c_str(), describe(expected(dispatch, kConstant), dispatch).c_str());
    return right ? 0 : 1;
}

bool write_library(const std::filesystem::path& library, const std::vector<std::uint8_t> (&programs)[3]) {
    if (!testfiles::copy_file(vktest::shader_fixture_path(LEONIDA_TEST_INDIRECT_CS_SPV), library / "cs_ic.spv") ||
        !testfiles::copy_file(vktest::shader_fixture_path(LEONIDA_TEST_INDIRECT_VS_SPV), library / "vs_ic.spv") ||
        !testfiles::copy_file(vktest::shader_fixture_path(LEONIDA_TEST_INDIRECT_PS_SPV), library / "ps_ic.spv"))
        return false;
    std::ofstream index(library / "index.json", std::ios::binary);
    index << "{\"cs_ic\": {\"stage\": \"cs\", \"fssm_sha256\": \"" << leonida::sha256_hex(programs[0])
          << "\", \"bindings\": [], \"push_constant_bytes\": 144, \"local_size\": [1, 1, 1], "
             "\"uses_code_base\": false, \"wave64\": false},\n"
             " \"vs_ic\": {\"stage\": \"vs\", \"fssm_sha256\": \"" << leonida::sha256_hex(programs[1])
          << "\", \"bindings\": [], \"push_constant_bytes\": 144, \"input_signature\": []},\n"
             " \"ps_ic\": {\"stage\": \"ps\", \"fssm_sha256\": \"" << leonida::sha256_hex(programs[2])
          << "\", \"bindings\": [], \"push_constant_bytes\": 144, \"outputs\": [0]}}";
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

    const std::vector<std::uint8_t> programs[3] = {fssm_program({0x43530359, 0xbf810000}),   // cs
                                                   fssm_program({0x56530359, 0xbf810000}),   // vs
                                                   fssm_program({0x50530359, 0xbf810000})};  // ps
    const testfiles::ScratchDir scratch("leonida_indirect_signature_test");
    if (!write_library(scratch.path(), programs)) return 1;
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
        Ref<ID3D12RootSignature> graphics_root, compute_root;
        Ref<ID3D12PipelineState> points, groups;
        if (FAILED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&device.p)))) return 1;
        const D3D12_COMMAND_QUEUE_DESC queue_desc{};  // a direct queue
        if (FAILED(device->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&queue.p))) ||
            FAILED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence.p))) ||
            FAILED(create_root(device.p, false, &graphics_root.p)) || FAILED(create_root(device.p, true, &compute_root.p)))
            return 1;
        Shared s{device.p, queue.p, fence.p, 0, graphics_root.p, compute_root.p};
        const HRESULT pipelines = create_pipelines(s, shader_container(programs[0]), shader_container(programs[1]),
                                                   shader_container(programs[2]));
        points.p = s.points;
        groups.p = s.groups;
        if (FAILED(pipelines)) {
            std::fprintf(stderr, "pipelines: 0x%08lx\n", static_cast<unsigned long>(pipelines));
            return 1;
        }
        const Case cases[] = {{"DISPATCH", D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH},
                              {"DRAW", D3D12_INDIRECT_ARGUMENT_TYPE_DRAW},
                              {"DRAW_INDEXED", D3D12_INDIRECT_ARGUMENT_TYPE_DRAW_INDEXED}};
        failures = run_control(s);
        for (const Case& c : cases)
            if (!failures) failures = run_case(s, c);
    }
    if (vk.validation_errors()) {
        std::fprintf(stderr, "validation errors: %u\n", vk.validation_errors());
        return 1;
    }
    return failures ? 1 : 0;
}
