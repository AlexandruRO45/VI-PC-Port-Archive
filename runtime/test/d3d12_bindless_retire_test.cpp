// D3D-314: a bindless flush that replaces the set (the table grows) does not wait for the device. Two D3D12 queues on
// two Vulkan queues, through the Xbox-ABI proxy, with the set-0 compute shader of d3d12_set0_compute_test
// (test/glsl/cs_set0_sample.comp: the T# and the S# from the descriptor tables, looked up in binding 2):
//   1. queue A executes a dispatch that samples the gradient behind a Wait on a gate fence nobody has signalled, so
//      the submission stays pending with the set and table as they were;
//   2. 40 more textures get views: the hash table outgrows its buffer, so the next flush replaces the set;
//   3. queue B executes a dispatch that samples the reflected texture: the ExecuteCommandLists returns and B's work
//      completes while A still waits (before D3D-314 the flush waited for the device, here forever), and the old set
//      and table are still alive (rt::d3d12::bindless_retired_count);
//   4. the gate is signalled: A's dispatch reads the old table and gives the gradient; the next ExecuteCommandLists
//      frees what was retired.
// A watchdog signals the gate after 10 s so a flush that waits for A fails the test instead of hanging it. No game
// data. Set LEONIDA_VK_VALIDATION=1 to run under VK_LAYER_KHRONOS_validation with synchronization validation
// (0 errors required).
#include "vk_test_device.h"
#include "d3d12_test_shaders.h"
#include "../shims/d3d12_com.h"
#include "../shims/d3d12_xs_device.h"
#include "../src/shader_library.h"

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <mutex>
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

// Opens the gate (signals 1) after 10 s unless stopped first, and when destroyed with the gate still closed: a flush
// that waits for the gated queue fails the test instead of hanging it, and an early return leaves no queue waiting.
class Watchdog {
public:
    explicit Watchdog(ID3D12Fence* gate) : gate_(gate), thread_([this] {
        std::unique_lock lock(mutex_);
        if (wake_.wait_for(lock, std::chrono::seconds(10), [this] { return stop_; })) return;
        fired_ = true;
        gate_->Signal(1);
    }) {}
    ~Watchdog() {
        stop();
        if (gate_->GetCompletedValue() == 0) gate_->Signal(1);
    }
    Watchdog(const Watchdog&) = delete;
    Watchdog& operator=(const Watchdog&) = delete;

    void stop() {
        {
            std::lock_guard lock(mutex_);
            stop_ = true;
        }
        wake_.notify_all();
        if (thread_.joinable()) thread_.join();
    }
    bool fired() const { return fired_; }

private:
    ID3D12Fence* gate_;
    std::mutex mutex_;
    std::condition_variable wake_;
    bool stop_ = false;
    std::atomic<bool> fired_{false};
    std::thread thread_;  // last: starts once the members above exist
};
}  // namespace

int main() {
    vktest::Device d;
    d.queue_count_wanted = 2;
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

    std::vector<std::uint8_t> results[2];
    double execute_ms = 0;
    std::size_t retired_while_pending = 0, retired_after = 0;
    UINT keys_found = 0;
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
        Ref<ID3D12Resource> textures[2], upload, outputs[2], readbacks[2];
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
        for (UINT q = 0; q < 2; ++q) {
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
        const D3D12_DESCRIPTOR_HEAP_DESC view_heap_desc{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 2 + extra_textures,
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

        // Queue B takes the first Vulkan queue, queue A the second.
        D3D12_COMMAND_QUEUE_DESC queue_desc{};
        Ref<ID3D12CommandQueue> queue_b, queue_a;
        EXPECT(SUCCEEDED(device->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&queue_b.p))));
        EXPECT(SUCCEEDED(device->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&queue_a.p))));
        Ref<ID3D12CommandAllocator> allocators[3];
        Ref<ID3D12GraphicsCommandList> lists[3];
        for (UINT i = 0; i < 3; ++i) {
            EXPECT(SUCCEEDED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
                                                            IID_PPV_ARGS(&allocators[i].p))));
            EXPECT(SUCCEEDED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocators[i].p, nullptr,
                                                       IID_PPV_ARGS(&lists[i].p))));
        }
        Ref<ID3D12Fence> uploaded, gate, done_a, done_b;
        for (auto* fence : {&uploaded, &gate, &done_a, &done_b})
            EXPECT(SUCCEEDED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence->p))));

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

        // 1. A: the gradient, held back by the gate.
        EXPECT(record(lists[0].p, 0, 0));
        Watchdog watchdog(gate.p);
        EXPECT(SUCCEEDED(queue_a->Wait(uploaded.p, 1)) && SUCCEEDED(queue_a->Wait(gate.p, 1)));
        ID3D12CommandList* a_lists[] = {lists[0].p};
        queue_a->ExecuteCommandLists(1, a_lists);
        EXPECT(rt::d3d12::last_execute_result(queue_a.p) == leonida::ReplayResult::success);
        EXPECT(SUCCEEDED(queue_a->Signal(done_a.p, 1)));
        const std::size_t retired_before = rt::d3d12::bindless_retired_count(device.p);

        // 2. More views than the table's first buffer holds.
        Ref<ID3D12Resource> extra[extra_textures];
        for (UINT i = 0; i < extra_textures; ++i) {
            EXPECT(SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &texture_desc,
                                                             D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, nullptr,
                                                             IID_PPV_ARGS(&extra[i].p))));
            device->CreateShaderResourceView(extra[i].p, &srv,
                                             D3D12_CPU_DESCRIPTOR_HANDLE{SIZE_T(view_cpu + (2 + i) * view_step)});
        }

        // 3. B: the reflected texture through the new set, while A still waits.
        EXPECT(record(lists[1].p, 1, 1));
        ID3D12CommandList* b_lists[] = {lists[1].p};
        const auto start = std::chrono::steady_clock::now();
        queue_b->ExecuteCommandLists(1, b_lists);
        execute_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        const bool b_executed = rt::d3d12::last_execute_result(queue_b.p) == leonida::ReplayResult::success;
        const bool b_signalled = SUCCEEDED(queue_b->Signal(done_b.p, 1));
        const bool b_done = b_signalled && wait_for(done_b.p, 1, std::chrono::seconds(10));
        const bool a_waiting = done_a->GetCompletedValue() == 0 && !watchdog.fired();
        retired_while_pending = rt::d3d12::bindless_retired_count(device.p);
        watchdog.stop();

        // 4. Open the gate: A samples through the set it bound.
        if (!watchdog.fired()) EXPECT(SUCCEEDED(gate->Signal(1)));
        EXPECT(wait_for(done_a.p, 1, std::chrono::seconds(10)));
        std::printf("flush with a pending submission on another queue: ExecuteCommandLists took %.2f ms, the other "
                    "queue's submission %s\n", execute_ms, a_waiting ? "still pending" : "NOT pending");
        EXPECT(b_executed && b_done && a_waiting);
        EXPECT(retired_before == 0 && retired_while_pending >= 1);

        // The next ExecuteCommandLists frees what was retired: every submission that could bind it is done.
        EXPECT(lists[2]->Reset(allocators[2].p, nullptr) == S_OK && lists[2]->Close() == S_OK);
        queue_b->ExecuteCommandLists(1, upload_lists);
        EXPECT(rt::d3d12::last_execute_result(queue_b.p) == leonida::ReplayResult::success);
        retired_after = rt::d3d12::bindless_retired_count(device.p);
        EXPECT(SUCCEEDED(queue_b->Signal(done_b.p, 2)));
        EXPECT(wait_for(done_b.p, 2, std::chrono::seconds(10)));

        // Every key the views made is in the table the device keeps.
        for (UINT i = 0; i < 2 + extra_textures; ++i) {
            std::uint32_t key[8];
            std::memcpy(key, reinterpret_cast<const void*>(view_cpu + i * view_step), sizeof(key));
            keys_found += rt::d3d12::bindless_hash_lookup(device.p, key) != 0;
        }
        for (UINT q = 0; q < 2; ++q) {
            void* mapped = nullptr;
            EXPECT(SUCCEEDED(readbacks[q]->Map(0, nullptr, &mapped)));
            results[q].assign(static_cast<const std::uint8_t*>(mapped), static_cast<const std::uint8_t*>(mapped) +
                                                                             run_bytes);
        }
    }
    std::filesystem::remove_all(library);

    unsigned mismatches = 0;
    for (UINT q = 0; q < 2; ++q)
        for (UINT i = 0; i < texels; ++i) {
            const UINT x = i % texture_size, y = i / texture_size;
            const Texel want = q ? reflected(x, y) : gradient(x, y);
            const std::uint8_t* got = results[q].data() + i * 4;
            if (std::memcmp(got, want.data(), 4) == 0) continue;
            if (mismatches++ < 8)
                std::fprintf(stderr, "queue %c, texel %u: got %u,%u,%u,%u want %u,%u,%u,%u\n", q ? 'B' : 'A', i, got[0],
                             got[1], got[2], got[3], want[0], want[1], want[2], want[3]);
        }
    std::printf("retired sets and tables: %zu while the old submission was pending, %zu after it completed; %u of %u "
                "keys found; %u texel mismatches\n", retired_while_pending, retired_after, keys_found,
                2 + extra_textures, mismatches);
    EXPECT(retired_after == 0 && keys_found == 2 + extra_textures && !mismatches);
    std::printf("validation errors: %u\n", d.validation_errors());
    if (d.validation_errors()) return 1;
    std::printf("bindless set replaced without waiting for the device: the pending dispatch read the old table "
                "(gradient), the new one the grown table (reflected)\n");
    return 0;
}
