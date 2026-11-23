// D3D-314: a bindless flush that replaces the set (the table grows) does not wait for the device. D3D12 queues on
// distinct Vulkan queues, through the Xbox-ABI proxy, with the set-0 compute shader of d3d12_set0_compute_test
// (test/glsl/cs_set0_sample.comp: the T# and the S# from the descriptor tables, looked up in binding 2):
//   1. queue A executes a dispatch that samples the gradient behind a Wait on a gate fence the GPU signals: queue C
//      (COPY) runs 64 MiB copies between host buffers, sized from a measured run to last about 300 ms, then signals
//      the gate. C's Signal is submitted first, so A's batch reaches Vulkan at once, with a Vulkan wait and the set
//      as it is bound (rt::d3d12::submitted_retire_ticket advances at its ExecuteCommandLists: nothing is held on the
//      CPU, queue_holds.h). With two Vulkan queues only, the gate runs on A itself, ahead of the dispatch;
//   2. 40 more textures get views: the hash table outgrows its buffer, so the next flush replaces the set;
//   3. queue B executes a dispatch that samples the reflected texture: its ExecuteCommandLists returns while the gate
//      is still closed and A's dispatch pending on the GPU (a flush that waits for A, in place or for the device,
//      returns only once the gate has opened), with the old set and table retired, not freed
//      (rt::d3d12::bindless_retired_count);
//   4. queue A executes a second dispatch of the reflected texture, through the new set: that ExecuteCommandLists,
//      while the first dispatch still waits, keeps what was retired, and its batch waits for B's, which uploads the
//      new table;
//   5. B's work completes while the gate is closed; the gate opens: A's first dispatch reads the gradient through the
//      old set and table, the second the reflected texture through the new ones; the next ExecuteCommandLists, once
//      both have completed, frees what was retired;
//   6. 40 more views, and B's next batch waits for a second gate: B's flush replaces the set again and B's batch
//      uploads the new table only once the gate opens. A's next dispatch binds the new set while the gate is still
//      closed: it must complete after the gate (it waits for B's batch), with the reflected texture.
// No game data. Set LEONIDA_VK_VALIDATION=1 to run under VK_LAYER_KHRONOS_validation with synchronization validation
// (0 errors required): it also reports a set or table freed under A's pending dispatch.
#include "vk_test_device.h"
#include "d3d12_test_shaders.h"
#include "../shims/d3d12_com.h"
#include "../shims/d3d12_xs_device.h"
#include "../src/shader_library.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <thread>
#include <vector>

#ifndef LEONIDA_TEST_CS_SET0_SAMPLE_SPV
#error "LEONIDA_TEST_CS_SET0_SAMPLE_SPV must name the compiled test shader"
#endif

#define EXPECT(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return 1; } \
} while (0)

using namespace d3d12test;
using XboxHeapHandleFn = std::uint64_t(WINAPI*)(void* heap);  // heap slots 9/10: handle in RAX

namespace {

using Texel = std::array<std::uint8_t, 4>;
constexpr UINT texture_size = 8, texels = texture_size * texture_size, run_bytes = texels * 4;
constexpr UINT extra_textures = 40;  // with the four keys of the setup, past half of the table's first 64 slots
constexpr UINT views = 2 + 2 * extra_textures;  // twice 40 extra textures: the table grows twice
constexpr UINT64 gate_bytes = 64ull << 20;
constexpr double gate_target_ms = 300;

Texel gradient(UINT x, UINT y) {
    return {std::uint8_t(x * 36), std::uint8_t(y * 36), std::uint8_t(255 - (x + y) * 16), 255};
}
Texel reflected(UINT x, UINT y) { return gradient(texture_size - 1 - x, texture_size - 1 - y); }

// Waits for the fence on the CPU, at most `timeout`.
bool wait_for(ID3D12Fence* fence, UINT64 value, std::chrono::milliseconds timeout) {
    const auto end = std::chrono::steady_clock::now() + timeout;
    while (fence->GetCompletedValue() < value) {
        if (std::chrono::steady_clock::now() > end) return false;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return true;
}

double ms_since(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
}
}  // namespace

int main() {
    vktest::Device d;
    d.queue_count_wanted = 3;  // B, A and the gate's queue C; with two, the gate runs on A
    const auto device_status = d.initialize();
    if (device_status == vktest::DeviceStatus::unavailable) return 77;   // no Vulkan GPU: ctest skips
    if (device_status != vktest::DeviceStatus::ready) return 1;
    if (!d.bindless) {
        std::printf("SKIP: the device lacks the descriptor indexing features of the bindless set 0\n");
        return 77;
    }
    if (d.queues.size() < 2) {
        std::printf("SKIP: the queue family has one queue; the test needs two\n");
        return 77;
    }

    const auto program = fssm_program({0x43530003, 0xbf810000});
    const auto container = shader_container(program);
    const auto library = std::filesystem::temp_directory_path() / "leonida_bindless_retire_test";
    std::filesystem::create_directories(library);
    std::filesystem::copy_file(LEONIDA_TEST_CS_SET0_SAMPLE_SPV, library / "cs_set0_sample.spv",
                               std::filesystem::copy_options::overwrite_existing);
    {
        std::ofstream index(library / "index.json", std::ios::binary);
        index << "{\"cs_set0_sample\": {\"stage\": \"cs\", \"fssm_sha256\": \"" << leonida::sha256_hex(program)
              << "\", \"bindings\": [0, 1, 2], \"push_constant_bytes\": 144, \"local_size\": [64, 1, 1], "
                 "\"uses_code_base\": false, \"wave64\": false}}";
    }

    std::vector<std::uint8_t> results[4];  // A's first dispatch, B's, A's second, A's third
    std::size_t retired_at_flush = 0, retired_while_pending = 0, retired_after = 0;
    UINT keys_found = 0;
    bool third_waited = false;  // step 6: A's third dispatch waited for B's batch, which uploads the table
    {
        rt::d3d12::VulkanDeviceInfo info{};
        info.get_instance_proc = d.instance_proc;
        info.instance = d.instance;
        info.physical = d.physical;
        info.device = d.device;
        info.queue = d.queue;
        info.command_queues = d.queues;
        info.queue_family = d.family;
        info.buffer_device_address = true;
        info.bindless = d.bindless;
        info.bindless_partially_bound = d.partially_bound;
        info.image_view_min_lod = d.image_view_min_lod;
        info.shader_dir = library.string();
        ID3D12Device* inner = nullptr;
        EXPECT(SUCCEEDED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&inner))));
        Ref<ID3D12Device> device;
        EXPECT(SUCCEEDED(rt::d3d12_xs::create_device_proxy(inner, __uuidof(ID3D12Device),
                                                           reinterpret_cast<void**>(&device.p))));
        rt::d3d12::set_device_outer(inner, device.p);

        // Root signature: [0] two constants (the view slot's offset in dwords, the first output element), [1] a root
        // UAV (the output buffer), [2] the table of views, [3] the table of samplers.
        D3D12_DESCRIPTOR_RANGE view_range{D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 2, 0, 0, 0};
        D3D12_DESCRIPTOR_RANGE sampler_range{D3D12_DESCRIPTOR_RANGE_TYPE_SAMPLER, 1, 0, 0, 0};
        D3D12_ROOT_PARAMETER params[4] = {};
        params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
        params[0].Constants = {0, 0, 2};
        params[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
        params[1].Descriptor = {0, 0};
        params[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        params[2].DescriptorTable = {1, &view_range};
        params[3].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        params[3].DescriptorTable = {1, &sampler_range};
        D3D12_VERSIONED_ROOT_SIGNATURE_DESC root_desc{};
        root_desc.Version = D3D_ROOT_SIGNATURE_VERSION_1_0;
        root_desc.Desc_1_0 = {4, params, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_NONE};
        const auto root_blob = serialize_root_signature(root_desc);
        EXPECT(!root_blob.empty());
        Ref<ID3D12RootSignature> root;
        EXPECT(SUCCEEDED(device->CreateRootSignature(0, root_blob.data(), root_blob.size(), IID_PPV_ARGS(&root.p))));
        D3D12_COMPUTE_PIPELINE_STATE_DESC pso_desc{};
        pso_desc.pRootSignature = root.p;
        pso_desc.CS = {container.data(), container.size()};
        Ref<ID3D12PipelineState> pso;
        EXPECT(SUCCEEDED(device->CreateComputePipelineState(&pso_desc, IID_PPV_ARGS(&pso.p))));

        const D3D12_HEAP_PROPERTIES default_heap{D3D12_HEAP_TYPE_DEFAULT};
        const D3D12_HEAP_PROPERTIES upload_heap{D3D12_HEAP_TYPE_UPLOAD};
        const D3D12_HEAP_PROPERTIES readback_heap{D3D12_HEAP_TYPE_READBACK};
        D3D12_RESOURCE_DESC texture_desc{};
        texture_desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        texture_desc.Width = texture_desc.Height = texture_size;
        texture_desc.DepthOrArraySize = 1;
        texture_desc.MipLevels = 1;
        texture_desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        texture_desc.SampleDesc.Count = 1;
        Ref<ID3D12Resource> textures[2], upload, outputs[4], readbacks[4];
        for (auto& texture : textures)
            EXPECT(SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &texture_desc,
                                                             D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                             IID_PPV_ARGS(&texture.p))));
        constexpr UINT64 b_offset = 2048;
        const auto upload_desc = buffer_desc(4096);
        EXPECT(SUCCEEDED(device->CreateCommittedResource(&upload_heap, D3D12_HEAP_FLAG_NONE, &upload_desc,
                                                         D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
                                                         IID_PPV_ARGS(&upload.p))));
        const auto output_desc = buffer_desc(run_bytes, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        const auto readback_desc = buffer_desc(run_bytes);
        for (UINT q = 0; q < 4; ++q) {
            EXPECT(SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &output_desc,
                                                             D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr,
                                                             IID_PPV_ARGS(&outputs[q].p))));
            EXPECT(SUCCEEDED(device->CreateCommittedResource(&readback_heap, D3D12_HEAP_FLAG_NONE, &readback_desc,
                                                             D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                             IID_PPV_ARGS(&readbacks[q].p))));
        }
        D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
        device->GetCopyableFootprints(&texture_desc, 0, 1, 0, &footprint, nullptr, nullptr, nullptr);
        EXPECT(footprint.Footprint.RowPitch == 256 && footprint.Footprint.RowPitch * texture_size <= b_offset);
        std::uint8_t* up = nullptr;
        EXPECT(SUCCEEDED(upload->Map(0, nullptr, reinterpret_cast<void**>(&up))));
        for (UINT y = 0; y < texture_size; ++y)
            for (UINT x = 0; x < texture_size; ++x) {
                const Texel a = gradient(x, y), b = reflected(x, y);
                std::memcpy(up + y * footprint.Footprint.RowPitch + x * 4, a.data(), 4);
                std::memcpy(up + b_offset + y * footprint.Footprint.RowPitch + x * 4, b.data(), 4);
            }

        // Views: the two textures in slots 0 and 1, the extra textures after them. Samplers: one that wraps.
        const D3D12_DESCRIPTOR_HEAP_DESC view_heap_desc{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, views,
                                                        D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, 0};
        const D3D12_DESCRIPTOR_HEAP_DESC sampler_heap_desc{D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER, 1,
                                                           D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, 0};
        Ref<ID3D12DescriptorHeap> view_heap, sampler_heap;
        EXPECT(SUCCEEDED(device->CreateDescriptorHeap(&view_heap_desc, IID_PPV_ARGS(&view_heap.p))));
        EXPECT(SUCCEEDED(device->CreateDescriptorHeap(&sampler_heap_desc, IID_PPV_ARGS(&sampler_heap.p))));
        const std::uint64_t view_cpu = reinterpret_cast<XboxHeapHandleFn>(slot(view_heap.p, 9))(view_heap.p);
        const std::uint64_t view_gpu = reinterpret_cast<XboxHeapHandleFn>(slot(view_heap.p, 10))(view_heap.p);
        const std::uint64_t sampler_cpu = reinterpret_cast<XboxHeapHandleFn>(slot(sampler_heap.p, 9))(sampler_heap.p);
        const std::uint64_t sampler_gpu = reinterpret_cast<XboxHeapHandleFn>(slot(sampler_heap.p, 10))(sampler_heap.p);
        const UINT view_step = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        EXPECT(view_cpu && sampler_cpu && view_gpu >= 0x10000 && sampler_gpu >= 0x10000);
        D3D12_SHADER_RESOURCE_VIEW_DESC srv{};
        srv.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        srv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
        srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        srv.Texture2D = {0, 1, 0, 0.0f};
        for (UINT i = 0; i < 2; ++i)
            device->CreateShaderResourceView(textures[i].p, &srv,
                                             D3D12_CPU_DESCRIPTOR_HANDLE{SIZE_T(view_cpu + i * view_step)});
        D3D12_SAMPLER_DESC sampler_desc{};
        sampler_desc.Filter = D3D12_FILTER_MIN_MAG_MIP_POINT;
        sampler_desc.AddressU = sampler_desc.AddressV = sampler_desc.AddressW = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
        sampler_desc.MaxAnisotropy = 1;
        sampler_desc.ComparisonFunc = D3D12_COMPARISON_FUNC_NEVER;
        sampler_desc.MaxLOD = D3D12_FLOAT32_MAX;
        device->CreateSampler(&sampler_desc, D3D12_CPU_DESCRIPTOR_HANDLE{SIZE_T(sampler_cpu)});

        // Queue B takes the first Vulkan queue, queue A the second, the gate's COPY queue C the third; with two
        // Vulkan queues the gate runs on A, ahead of A's dispatch.
        D3D12_COMMAND_QUEUE_DESC queue_desc{};
        Ref<ID3D12CommandQueue> queue_b, queue_a, queue_c;
        EXPECT(SUCCEEDED(device->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&queue_b.p))));
        EXPECT(SUCCEEDED(device->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&queue_a.p))));
        D3D12_COMMAND_LIST_TYPE gate_type = D3D12_COMMAND_LIST_TYPE_DIRECT;
        if (d.queues.size() >= 3) {
            gate_type = D3D12_COMMAND_LIST_TYPE_COPY;
            const D3D12_COMMAND_QUEUE_DESC copy_desc{D3D12_COMMAND_LIST_TYPE_COPY};
            EXPECT(SUCCEEDED(device->CreateCommandQueue(&copy_desc, IID_PPV_ARGS(&queue_c.p))));
        }
        ID3D12CommandQueue* gate_queue = queue_c.p ? queue_c.p : queue_a.p;
        std::printf("gate on %s\n", queue_c.p ? "a COPY queue of its own" : "queue A, ahead of its dispatch");
        Ref<ID3D12CommandAllocator> allocators[4], gate_allocator;
        Ref<ID3D12GraphicsCommandList> lists[4], gate_list;
        for (UINT i = 0; i < 4; ++i) {
            EXPECT(SUCCEEDED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
                                                            IID_PPV_ARGS(&allocators[i].p))));
            EXPECT(SUCCEEDED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocators[i].p, nullptr,
                                                       IID_PPV_ARGS(&lists[i].p))));
        }
        EXPECT(SUCCEEDED(device->CreateCommandAllocator(gate_type, IID_PPV_ARGS(&gate_allocator.p))));
        EXPECT(SUCCEEDED(device->CreateCommandList(0, gate_type, gate_allocator.p, nullptr,
                                                   IID_PPV_ARGS(&gate_list.p))));
        EXPECT(gate_list->Close() == S_OK);
        Ref<ID3D12Fence> uploaded, gate, done_a, done_b;
        for (auto* fence : {&uploaded, &gate, &done_a, &done_b})
            EXPECT(SUCCEEDED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence->p))));
        const auto gate_desc = buffer_desc(gate_bytes);
        Ref<ID3D12Resource> gate_src, gate_dst;
        EXPECT(SUCCEEDED(device->CreateCommittedResource(&upload_heap, D3D12_HEAP_FLAG_NONE, &gate_desc,
                                                         D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
                                                         IID_PPV_ARGS(&gate_src.p))));
        EXPECT(SUCCEEDED(device->CreateCommittedResource(&readback_heap, D3D12_HEAP_FLAG_NONE, &gate_desc,
                                                         D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                         IID_PPV_ARGS(&gate_dst.p))));

        // The gate: `copies` copies of 64 MiB from an upload to a readback buffer, one after the other (a null
        // aliasing barrier between them is a full memory barrier), then Signal(gate). Returns the value, 0 on failure.
        UINT64 gate_value = 0;
        const auto submit_gate = [&](UINT copies) -> UINT64 {
            if (gate_list->Reset(gate_allocator.p, nullptr) != S_OK) return 0;
            D3D12_RESOURCE_BARRIER between{};
            between.Type = D3D12_RESOURCE_BARRIER_TYPE_ALIASING;
            for (UINT i = 0; i < copies; ++i) {
                if (i) gate_list->ResourceBarrier(1, &between);
                gate_list->CopyBufferRegion(gate_dst.p, 0, gate_src.p, 0, gate_bytes);
            }
            if (gate_list->Close() != S_OK) return 0;
            ID3D12CommandList* batch[] = {gate_list.p};
            gate_queue->ExecuteCommandLists(1, batch);
            return SUCCEEDED(gate_queue->Signal(gate.p, ++gate_value)) ? gate_value : 0;
        };
        // Two copies, timed, size the gate to about gate_target_ms.
        const auto calibration_start = std::chrono::steady_clock::now();
        const UINT64 calibrated = submit_gate(2);
        EXPECT(calibrated && wait_for(gate.p, calibrated, std::chrono::seconds(30)));
        const double two_copies_ms = std::max(ms_since(calibration_start), 0.5);
        const UINT gate_copies = std::clamp(UINT(2 * gate_target_ms / two_copies_ms) + 1, 2u, 400u);

        // The uploads, on B.
        for (UINT i = 0; i < 2; ++i) {
            D3D12_TEXTURE_COPY_LOCATION dst{};
            dst.pResource = textures[i].p;
            dst.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
            D3D12_TEXTURE_COPY_LOCATION src{};
            src.pResource = upload.p;
            src.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
            src.PlacedFootprint = footprint;
            src.PlacedFootprint.Offset = i ? b_offset : 0;
            lists[2]->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
        }
        D3D12_RESOURCE_BARRIER sampled[2] = {
            transition(textures[0].p, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE),
            transition(textures[1].p, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE)};
        lists[2]->ResourceBarrier(2, sampled);
        EXPECT(lists[2]->Close() == S_OK);
        ID3D12CommandList* upload_lists[] = {lists[2].p};
        queue_b->ExecuteCommandLists(1, upload_lists);
        EXPECT(rt::d3d12::last_execute_result(queue_b.p) == leonida::ReplayResult::success);
        EXPECT(SUCCEEDED(queue_b->Signal(uploaded.p, 1)));
        EXPECT(wait_for(uploaded.p, 1, std::chrono::seconds(10)));

        // One dispatch of a texture's slot into a queue's output, then the copy to its readback buffer.
        ID3D12DescriptorHeap* heaps[] = {view_heap.p, sampler_heap.p};
        const auto record = [&](ID3D12GraphicsCommandList* list, UINT view_slot, UINT q) {
            list->SetDescriptorHeaps(2, heaps);
            list->SetComputeRootSignature(root.p);
            list->SetPipelineState(pso.p);
            list->SetComputeRootUnorderedAccessView(1, outputs[q]->GetGPUVirtualAddress());
            list->SetComputeRootDescriptorTable(2, D3D12_GPU_DESCRIPTOR_HANDLE{view_gpu});
            list->SetComputeRootDescriptorTable(3, D3D12_GPU_DESCRIPTOR_HANDLE{sampler_gpu});
            const std::uint32_t offset_and_first[2] = {view_slot * view_step / 4, 0};
            list->SetComputeRoot32BitConstants(0, 2, offset_and_first, 0);
            list->Dispatch(1, 1, 1);
            D3D12_RESOURCE_BARRIER to_copy = transition(outputs[q].p, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                                                        D3D12_RESOURCE_STATE_COPY_SOURCE);
            list->ResourceBarrier(1, &to_copy);
            list->CopyBufferRegion(readbacks[q].p, 0, outputs[q].p, 0, run_bytes);
            return list->Close() == S_OK;
        };

        // 1. A: the gradient, behind the gate. The gate's Signal is submitted first, so A's batch goes to Vulkan now
        // with a Vulkan wait for it, bound to the set as it is: the ticket of the last submission advances at A's
        // ExecuteCommandLists. A batch held on the CPU would bind the set as it is when the gate releases it.
        EXPECT(record(lists[0].p, 0, 0));
        const auto gate_start = std::chrono::steady_clock::now();
        const UINT64 gate_open = submit_gate(gate_copies);
        EXPECT(gate_open);
        EXPECT(SUCCEEDED(queue_a->Wait(uploaded.p, 1)) && SUCCEEDED(queue_a->Wait(gate.p, gate_open)));
        ID3D12CommandList* a_lists[] = {lists[0].p};
        const std::uint64_t ticket_before_a = rt::d3d12::submitted_retire_ticket(device.p);
        queue_a->ExecuteCommandLists(1, a_lists);
        const std::uint64_t ticket_after_a = rt::d3d12::submitted_retire_ticket(device.p);
        EXPECT(rt::d3d12::last_execute_result(queue_a.p) == leonida::ReplayResult::success);
        EXPECT(SUCCEEDED(queue_a->Signal(done_a.p, 1)));
        const bool gate_closed_at_a = gate->GetCompletedValue() < gate_open;
        std::printf("A's ExecuteCommandLists behind the gate: submitted retire ticket %llu -> %llu (%s), gate %s\n",
                    static_cast<unsigned long long>(ticket_before_a), static_cast<unsigned long long>(ticket_after_a),
                    ticket_after_a > ticket_before_a ? "submitted to Vulkan" : "HELD on the CPU",
                    gate_closed_at_a ? "closed" : "ALREADY OPEN");
        EXPECT(ticket_after_a > ticket_before_a);
        EXPECT(gate_closed_at_a);  // else the gate is too short to show anything
        const std::size_t retired_before = rt::d3d12::bindless_retired_count(device.p);

        // 2. More views than the table's first buffer holds.
        Ref<ID3D12Resource> extra[2 * extra_textures];
        const auto add_views = [&](UINT first) {
            for (UINT i = first; i < first + extra_textures; ++i) {
                if (FAILED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &texture_desc,
                                                           D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, nullptr,
                                                           IID_PPV_ARGS(&extra[i].p))))
                    return false;
                device->CreateShaderResourceView(extra[i].p, &srv,
                                                 D3D12_CPU_DESCRIPTOR_HANDLE{SIZE_T(view_cpu + (2 + i) * view_step)});
            }
            return true;
        };
        EXPECT(add_views(0));

        // 3. B: the reflected texture through the new set, while A's dispatch is pending on the GPU.
        EXPECT(record(lists[1].p, 1, 1));
        ID3D12CommandList* b_lists[] = {lists[1].p};
        const auto start = std::chrono::steady_clock::now();
        queue_b->ExecuteCommandLists(1, b_lists);
        const double execute_ms = ms_since(start);
        const bool a_pending_after_b = gate->GetCompletedValue() < gate_open && done_a->GetCompletedValue() == 0;
        const bool b_executed = rt::d3d12::last_execute_result(queue_b.p) == leonida::ReplayResult::success;
        const bool b_signalled = SUCCEEDED(queue_b->Signal(done_b.p, 1));
        retired_at_flush = rt::d3d12::bindless_retired_count(device.p);

        // 4. A: the reflected texture through the new set, after A's first dispatch. This ExecuteCommandLists polls
        // the retired sets and tables while the first dispatch still binds the old ones.
        EXPECT(record(lists[3].p, 1, 2));
        ID3D12CommandList* a2_lists[] = {lists[3].p};
        queue_a->ExecuteCommandLists(1, a2_lists);
        const bool a2_executed = rt::d3d12::last_execute_result(queue_a.p) == leonida::ReplayResult::success;
        const bool a2_signalled = SUCCEEDED(queue_a->Signal(done_a.p, 2));
        retired_while_pending = rt::d3d12::bindless_retired_count(device.p);
        const bool a_pending_after_a2 = gate->GetCompletedValue() < gate_open && done_a->GetCompletedValue() == 0;

        // 5. B completes while the gate is closed; then the gate opens and A's dispatches run.
        const bool b_done = b_signalled && wait_for(done_b.p, 1, std::chrono::seconds(10));
        const bool b_done_while_gated = b_done && gate->GetCompletedValue() < gate_open;
        EXPECT(wait_for(gate.p, gate_open, std::chrono::seconds(30)));
        const double gate_ms = ms_since(gate_start);
        EXPECT(a2_signalled && wait_for(done_a.p, 2, std::chrono::seconds(30)));
        std::printf("gate: 2 copies of 64 MiB took %.1f ms, so %u copies; it opened %.1f ms after it was submitted\n",
                    two_copies_ms, gate_copies, gate_ms);
        std::printf("flush that replaced the set with A's dispatch pending on the GPU: ExecuteCommandLists took %.2f "
                    "ms; A's dispatch %s when it returned and %s after A's next ExecuteCommandLists; B's work %s\n",
                    execute_ms, a_pending_after_b ? "still pending" : "NOT pending",
                    a_pending_after_a2 ? "still pending" : "NOT pending",
                    b_done_while_gated ? "completed while the gate was closed" : "did NOT complete while gated");
        EXPECT(b_executed && a2_executed && a_pending_after_b && a_pending_after_a2 && b_done_while_gated);
        EXPECT(retired_before == 0 && retired_at_flush >= 1 && retired_while_pending >= 1);

        // The next ExecuteCommandLists frees what was retired: every submission that could bind it is done.
        EXPECT(lists[2]->Reset(allocators[2].p, nullptr) == S_OK && lists[2]->Close() == S_OK);
        queue_b->ExecuteCommandLists(1, upload_lists);
        EXPECT(rt::d3d12::last_execute_result(queue_b.p) == leonida::ReplayResult::success);
        retired_after = rt::d3d12::bindless_retired_count(device.p);
        EXPECT(SUCCEEDED(queue_b->Signal(done_b.p, 2)));
        EXPECT(wait_for(done_b.p, 2, std::chrono::seconds(10)));

        // 6. The cross-queue wait. 40 more views after the gate's ExecuteCommandLists (which would flush them): B's
        // next flush replaces the set again, and B's batch waits for the gate, so the copy in it that uploads the new
        // table runs only once the gate opens. A's next dispatch, of the reflected texture, binds the new set right
        // after: its batch must wait for B's, not look up a table that is not uploaded yet (it would find the
        // fallback).
        const UINT64 gate_again = submit_gate(gate_copies);
        EXPECT(gate_again && SUCCEEDED(queue_b->Wait(gate.p, gate_again)));
        EXPECT(add_views(extra_textures));
        EXPECT(lists[2]->Reset(allocators[2].p, nullptr) == S_OK && lists[2]->Close() == S_OK);
        queue_b->ExecuteCommandLists(1, upload_lists);
        EXPECT(rt::d3d12::last_execute_result(queue_b.p) == leonida::ReplayResult::success);
        EXPECT(SUCCEEDED(queue_b->Signal(done_b.p, 3)));
        const std::size_t retired_second = rt::d3d12::bindless_retired_count(device.p);
        EXPECT(lists[0]->Reset(allocators[0].p, nullptr) == S_OK && record(lists[0].p, 1, 3));
        queue_a->ExecuteCommandLists(1, a_lists);
        EXPECT(rt::d3d12::last_execute_result(queue_a.p) == leonida::ReplayResult::success);
        EXPECT(SUCCEEDED(queue_a->Signal(done_a.p, 3)));
        const bool gated_at_a3 = gate->GetCompletedValue() < gate_again;
        EXPECT(wait_for(done_a.p, 3, std::chrono::seconds(30)));
        const bool a3_done_while_gated = gate->GetCompletedValue() < gate_again;
        EXPECT(wait_for(done_b.p, 3, std::chrono::seconds(30)));
        std::printf("second replacement, B's upload behind the gate: %zu retired after B's flush; A's next "
                    "ExecuteCommandLists with the gate %s, its dispatch completed %s\n", retired_second,
                    gated_at_a3 ? "closed" : "ALREADY OPEN",
                    a3_done_while_gated ? "BEFORE the gate opened (no wait for B)" : "after the gate opened");
        third_waited = retired_second >= 1 && gated_at_a3 && !a3_done_while_gated;

        // Every key the views made is in the table the device keeps.
        for (UINT i = 0; i < views; ++i) {
            std::uint32_t key[8];
            std::memcpy(key, reinterpret_cast<const void*>(view_cpu + i * view_step), sizeof(key));
            keys_found += rt::d3d12::bindless_hash_lookup(device.p, key) != 0;
        }
        for (UINT q = 0; q < 4; ++q) {
            void* mapped = nullptr;
            EXPECT(SUCCEEDED(readbacks[q]->Map(0, nullptr, &mapped)));
            results[q].assign(static_cast<const std::uint8_t*>(mapped), static_cast<const std::uint8_t*>(mapped) +
                                                                             run_bytes);
        }
    }
    std::filesystem::remove_all(library);

    static const char* const names[4] = {"A's first dispatch", "B's dispatch", "A's second dispatch",
                                         "A's third dispatch"};
    unsigned mismatches = 0;
    for (UINT q = 0; q < 4; ++q)
        for (UINT i = 0; i < texels; ++i) {
            const UINT x = i % texture_size, y = i / texture_size;
            const Texel want = q ? reflected(x, y) : gradient(x, y);
            const std::uint8_t* got = results[q].data() + i * 4;
            if (std::memcmp(got, want.data(), 4) == 0) continue;
            if (mismatches++ < 8)
                std::fprintf(stderr, "%s, texel %u: got %u,%u,%u,%u want %u,%u,%u,%u\n", names[q], i, got[0], got[1],
                             got[2], got[3], want[0], want[1], want[2], want[3]);
        }
    std::printf("retired sets and tables: %zu after the flush, %zu after A's next ExecuteCommandLists with its first "
                "dispatch pending, %zu once both completed; %u of %u keys found; %u texel mismatches\n",
                retired_at_flush, retired_while_pending, retired_after, keys_found, views, mismatches);
    EXPECT(retired_after == 0 && keys_found == views && !mismatches && third_waited);
    std::printf("validation errors: %u\n", d.validation_errors());
    if (d.validation_errors()) return 1;
    std::printf("bindless set replaced while a dispatch that bound the old one was pending on the GPU: it read the "
                "gradient through the old set and table, B and A's next dispatches the reflected texture through the "
                "new ones, the last one after the upload in B's batch that waited for the gate\n");
    return 0;
}
