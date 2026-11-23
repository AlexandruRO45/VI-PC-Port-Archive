// Synthetic validation and null-offset tests for UpdateTileMappings / CopyTileMappings / append_tile_regions.
// Runs valid null-offset and UseBox GPU readback cases on the Vulkan layer, native D3D12, and WARP, and verifies
// invalid cases are cleanly rejected without mutating existing mappings on the Vulkan layer.
#include "../shims/d3d12_com.h"

#include <dxgi1_4.h>
#include <wrl/client.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <string>

#define EXPECT(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return false; } \
} while (0)

namespace {

using Microsoft::WRL::ComPtr;
constexpr UINT64 tile_bytes = D3D12_TILED_RESOURCE_TILE_SIZE_IN_BYTES;
constexpr UINT tiles_per_buffer = 4;
constexpr UINT64 buffer_bytes = tiles_per_buffer * tile_bytes;
constexpr UINT snapshots = 4;
constexpr std::array<std::uint32_t, tiles_per_buffer> patterns{
    0x11111111u, 0x22222222u, 0x33333333u, 0x44444444u};
using Snapshot = std::array<std::uint32_t, tiles_per_buffer>;
using Results = std::array<Snapshot, snapshots>;
constexpr Results expected{{
    {patterns[0], patterns[1], 0, 0},
    {patterns[0], patterns[0], patterns[0], 0},
    {0, patterns[0], patterns[1], 0},
    {patterns[0], patterns[1], 0, 0},
}};

D3D12_RESOURCE_DESC buffer_desc(UINT64 bytes) {
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width = bytes;
    desc.Height = desc.DepthOrArraySize = desc.MipLevels = 1;
    desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    return desc;
}

struct Event {
    HANDLE handle = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    ~Event() { if (handle) CloseHandle(handle); }
};

bool wait_idle(ID3D12CommandQueue& queue, ID3D12Fence& fence, UINT64& value, HANDLE event) {
    ++value;
    EXPECT(SUCCEEDED(queue.Signal(&fence, value)));
    EXPECT(SUCCEEDED(fence.SetEventOnCompletion(value, event)));
    EXPECT(WaitForSingleObject(event, 15000) == WAIT_OBJECT_0);
    return true;
}

bool copy_buffer_sync(ID3D12Device& device, ID3D12CommandQueue& queue, ID3D12Fence& fence, UINT64& fence_val,
                      HANDLE event, ID3D12Resource* dst, UINT64 dst_offset, ID3D12Resource* src, UINT64 src_offset,
                      UINT64 bytes) {
    ComPtr<ID3D12CommandAllocator> allocator;
    ComPtr<ID3D12GraphicsCommandList> list;
    EXPECT(SUCCEEDED(device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator))));
    EXPECT(SUCCEEDED(device.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.Get(), nullptr,
                                              IID_PPV_ARGS(&list))));
    list->CopyBufferRegion(dst, dst_offset, src, src_offset, bytes);
    EXPECT(SUCCEEDED(list->Close()));
    ID3D12CommandList* lists[] = {list.Get()};
    queue.ExecuteCommandLists(1, lists);
    return wait_idle(queue, fence, fence_val, event);
}

bool read_tiles(ID3D12Device& device, ID3D12CommandQueue& queue, ID3D12Fence& fence, UINT64& fence_val, HANDLE event,
                ID3D12Resource* src, ID3D12Resource* readback, Snapshot& out, const Snapshot& want, const char* label) {
    EXPECT(copy_buffer_sync(device, queue, fence, fence_val, event, readback, 0, src, 0, buffer_bytes));
    void* mapped = nullptr;
    EXPECT(SUCCEEDED(readback->Map(0, nullptr, &mapped)));
    const auto* data = static_cast<const std::uint32_t*>(mapped);
    constexpr auto words_per_tile = std::size_t(tile_bytes / sizeof(std::uint32_t));
    UINT wrong_tiles = 0;
    for (UINT tile = 0; tile < tiles_per_buffer; ++tile) {
        const auto* first = data + tile * words_per_tile;
        out[tile] = *first;
        const auto* bad = std::find_if(first, first + words_per_tile, [&](auto word) { return word != want[tile]; });
        if (bad == first + words_per_tile) continue;
        ++wrong_tiles;
        std::fprintf(stderr, "%s tile %u: word %u reads %#010x, want %#010x\n",
                     label, tile, unsigned(bad - first), unsigned(*bad), unsigned(want[tile]));
    }
    const D3D12_RANGE empty{0, 0};
    readback->Unmap(0, &empty);
    EXPECT(wrong_tiles == 0);
    return true;
}

bool run_valid_suite(ID3D12Device& device, Results& results, bool layer, const char* backend_name) {
    ComPtr<ID3D12Resource> a, b, upload, readback, committed, placed;
    const auto reserved = buffer_desc(buffer_bytes);
    EXPECT(SUCCEEDED(device.CreateReservedResource(&reserved, D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&a))));
    EXPECT(SUCCEEDED(device.CreateReservedResource(&reserved, D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&b))));

    D3D12_HEAP_DESC heap_desc{};
    heap_desc.SizeInBytes = tiles_per_buffer * tile_bytes;
    heap_desc.Properties.Type = D3D12_HEAP_TYPE_DEFAULT;
    heap_desc.Properties.CreationNodeMask = heap_desc.Properties.VisibleNodeMask = 1;
    heap_desc.Flags = D3D12_HEAP_FLAG_ALLOW_ONLY_BUFFERS;
    ComPtr<ID3D12Heap> heap;
    EXPECT(SUCCEEDED(device.CreateHeap(&heap_desc, IID_PPV_ARGS(&heap))));

    D3D12_HEAP_PROPERTIES default_heap = heap_desc.Properties;
    D3D12_HEAP_PROPERTIES upload_heap = default_heap;
    upload_heap.Type = D3D12_HEAP_TYPE_UPLOAD;
    D3D12_HEAP_PROPERTIES readback_heap = default_heap;
    readback_heap.Type = D3D12_HEAP_TYPE_READBACK;
    EXPECT(SUCCEEDED(device.CreateCommittedResource(&upload_heap, D3D12_HEAP_FLAG_NONE, &reserved,
                                                    D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&upload))));
    EXPECT(SUCCEEDED(device.CreateCommittedResource(&readback_heap, D3D12_HEAP_FLAG_NONE, &reserved,
                                                    D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&readback))));
    ComPtr<ID3D12Heap> placed_heap;
    if (layer) {
        EXPECT(SUCCEEDED(device.CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &reserved,
                                                        D3D12_RESOURCE_STATE_COMMON, nullptr,
                                                        IID_PPV_ARGS(&committed))));
        EXPECT(SUCCEEDED(device.CreateHeap(&heap_desc, IID_PPV_ARGS(&placed_heap))));
        EXPECT(SUCCEEDED(device.CreatePlacedResource(placed_heap.Get(), 0, &reserved, D3D12_RESOURCE_STATE_COMMON,
                                                     nullptr, IID_PPV_ARGS(&placed))));
    }

    constexpr auto words_per_tile = std::size_t(tile_bytes / sizeof(std::uint32_t));
    const D3D12_RANGE empty{0, 0};
    void* mapped = nullptr;
    EXPECT(SUCCEEDED(upload->Map(0, &empty, &mapped)));
    auto* words = static_cast<std::uint32_t*>(mapped);
    for (std::size_t i = 0; i < patterns.size(); ++i)
        std::fill_n(words + i * words_per_tile, words_per_tile, patterns[i]);
    upload->Unmap(0, nullptr);

    ComPtr<ID3D12CommandQueue> queue;
    const D3D12_COMMAND_QUEUE_DESC queue_desc{};
    EXPECT(SUCCEEDED(device.CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&queue))));
    ComPtr<ID3D12Fence> fence;
    EXPECT(SUCCEEDED(device.CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence))));
    Event event;
    EXPECT(event.handle);
    UINT64 fence_val = 0;

    const D3D12_TILED_RESOURCE_COORDINATE origin{};
    const D3D12_TILE_REGION_SIZE all{tiles_per_buffer, FALSE, 0, 0, 0};
    const D3D12_TILE_RANGE_FLAGS unmap = D3D12_TILE_RANGE_FLAG_NULL;

    // Populate heap tiles 0..3 using reserved buffer a.
    const UINT init_off = 0, init_cnt = tiles_per_buffer;
    queue->UpdateTileMappings(a.Get(), 1, &origin, &all, heap.Get(), 1, nullptr, &init_off, &init_cnt,
                              D3D12_TILE_MAPPING_FLAG_NONE);
    EXPECT(copy_buffer_sync(device, *queue.Get(), *fence.Get(), fence_val, event.handle, a.Get(), 0, upload.Get(), 0,
                            buffer_bytes));

    auto reset_a = [&]() {
        queue->UpdateTileMappings(a.Get(), 1, nullptr, nullptr, nullptr, 1, &unmap, nullptr, nullptr,
                                  D3D12_TILE_MAPPING_FLAG_NONE);
    };

    // Snapshot 0: pHeapRangeStartOffsets == nullptr with D3D12_TILE_RANGE_FLAG_NONE (defaults to heap offset 0).
    reset_a();
    const D3D12_TILE_REGION_SIZE two{2, FALSE, 0, 0, 0};
    const D3D12_TILE_RANGE_FLAGS flag_none = D3D12_TILE_RANGE_FLAG_NONE;
    const UINT count_two = 2;
    queue->UpdateTileMappings(a.Get(), 1, &origin, &two, heap.Get(), 1, &flag_none, nullptr, &count_two,
                              D3D12_TILE_MAPPING_FLAG_NONE);
    EXPECT(SUCCEEDED(device.GetDeviceRemovedReason()));
    EXPECT(read_tiles(device, *queue.Get(), *fence.Get(), fence_val, event.handle, a.Get(), readback.Get(),
                      results[0], expected[0], backend_name));

    // Snapshot 1: pHeapRangeStartOffsets == nullptr with D3D12_TILE_RANGE_FLAG_REUSE_SINGLE_TILE.
    reset_a();
    const D3D12_TILE_REGION_SIZE three{3, FALSE, 0, 0, 0};
    const D3D12_TILE_RANGE_FLAGS flag_reuse = D3D12_TILE_RANGE_FLAG_REUSE_SINGLE_TILE;
    const UINT count_three = 3;
    queue->UpdateTileMappings(a.Get(), 1, &origin, &three, heap.Get(), 1, &flag_reuse, nullptr, &count_three,
                              D3D12_TILE_MAPPING_FLAG_NONE);
    EXPECT(SUCCEEDED(device.GetDeviceRemovedReason()));
    EXPECT(read_tiles(device, *queue.Get(), *fence.Get(), fence_val, event.handle, a.Get(), readback.Get(),
                      results[1], expected[1], backend_name));

    // Snapshot 2: pHeapRangeStartOffsets == nullptr with mixed NULL + NONE ranges.
    reset_a();
    const D3D12_TILE_RANGE_FLAGS mixed_flags[2] = {D3D12_TILE_RANGE_FLAG_NULL, D3D12_TILE_RANGE_FLAG_NONE};
    const UINT mixed_counts[2] = {1, 2};
    queue->UpdateTileMappings(a.Get(), 1, &origin, &three, heap.Get(), 2, mixed_flags, nullptr, mixed_counts,
                              D3D12_TILE_MAPPING_FLAG_NONE);
    EXPECT(SUCCEEDED(device.GetDeviceRemovedReason()));
    EXPECT(read_tiles(device, *queue.Get(), *fence.Get(), fence_val, event.handle, a.Get(), readback.Get(),
                      results[2], expected[2], backend_name));

    // Snapshot 3: UseBox == TRUE with matching NumTiles == Width * Height * Depth and null offsets.
    reset_a();
    const D3D12_TILE_REGION_SIZE box_ok{2, TRUE, 2, 1, 1};
    queue->UpdateTileMappings(a.Get(), 1, &origin, &box_ok, heap.Get(), 1, &flag_none, nullptr, &count_two,
                              D3D12_TILE_MAPPING_FLAG_NONE);
    EXPECT(SUCCEEDED(device.GetDeviceRemovedReason()));
    EXPECT(read_tiles(device, *queue.Get(), *fence.Get(), fence_val, event.handle, a.Get(), readback.Get(),
                      results[3], expected[3], backend_name));

    if (layer) {
        // Zero-count range with a valid flag followed by a backed range succeeds (sets a to {patterns[2], patterns[3], 0, 0}).
        reset_a();
        const D3D12_TILE_RANGE_FLAGS zero_valid_flags[2] = {D3D12_TILE_RANGE_FLAG_SKIP, D3D12_TILE_RANGE_FLAG_NONE};
        const UINT zero_valid_offsets[2] = {0, 2};
        const UINT zero_valid_counts[2] = {0, 2};
        queue->UpdateTileMappings(a.Get(), 1, &origin, &two, heap.Get(), 2, zero_valid_flags, zero_valid_offsets,
                                  zero_valid_counts, D3D12_TILE_MAPPING_FLAG_NONE);
        EXPECT(SUCCEEDED(device.GetDeviceRemovedReason()));
        const Snapshot baseline{patterns[2], patterns[3], 0, 0};
        Snapshot got_baseline{};
        EXPECT(read_tiles(device, *queue.Get(), *fence.Get(), fence_val, event.handle, a.Get(), readback.Get(),
                          got_baseline, baseline, "Vulkan layer (zero-count valid flag)"));

        // Every invalid call below must be rejected without mutating a ({patterns[2], patterns[3], 0, 0}) or b ({0}).
        const UINT zero_off = 0, one_cnt = 1;
        const D3D12_TILE_REGION_SIZE one{1, FALSE, 0, 0, 0};

        // 1. Null heap with NONE / REUSE_SINGLE_TILE (with or without null offsets).
        queue->UpdateTileMappings(a.Get(), 1, &origin, &two, nullptr, 1, &flag_none, nullptr, &count_two,
                                  D3D12_TILE_MAPPING_FLAG_NONE);
        queue->UpdateTileMappings(a.Get(), 1, &origin, &two, nullptr, 1, &flag_reuse, nullptr, &count_two,
                                  D3D12_TILE_MAPPING_FLAG_NONE);

        // 2. Non-reserved committed and placed buffers in UpdateTileMappings and CopyTileMappings.
        queue->UpdateTileMappings(committed.Get(), 1, &origin, &one, heap.Get(), 1, &flag_none, &zero_off, &one_cnt,
                                  D3D12_TILE_MAPPING_FLAG_NONE);
        queue->UpdateTileMappings(placed.Get(), 1, &origin, &one, heap.Get(), 1, &flag_none, &zero_off, &one_cnt,
                                  D3D12_TILE_MAPPING_FLAG_NONE);
        queue->CopyTileMappings(committed.Get(), &origin, a.Get(), &origin, &one, D3D12_TILE_MAPPING_FLAG_NONE);
        queue->CopyTileMappings(b.Get(), &origin, committed.Get(), &origin, &one, D3D12_TILE_MAPPING_FLAG_NONE);
        queue->CopyTileMappings(placed.Get(), &origin, a.Get(), &origin, &one, D3D12_TILE_MAPPING_FLAG_NONE);
        queue->CopyTileMappings(b.Get(), &origin, placed.Get(), &origin, &one, D3D12_TILE_MAPPING_FLAG_NONE);

        // 3. UseBox == TRUE with mismatched NumTiles (too small, too large, zero) in UpdateTileMappings & CopyTileMappings.
        const D3D12_TILE_REGION_SIZE box_small{1, TRUE, 2, 1, 1};
        const D3D12_TILE_REGION_SIZE box_large{3, TRUE, 2, 1, 1};
        const D3D12_TILE_REGION_SIZE box_zero{0, TRUE, 2, 1, 1};
        queue->UpdateTileMappings(a.Get(), 1, &origin, &box_small, heap.Get(), 1, &flag_none, nullptr, &count_two,
                                  D3D12_TILE_MAPPING_FLAG_NONE);
        queue->UpdateTileMappings(a.Get(), 1, &origin, &box_large, heap.Get(), 1, &flag_none, nullptr, &count_two,
                                  D3D12_TILE_MAPPING_FLAG_NONE);
        queue->UpdateTileMappings(a.Get(), 1, &origin, &box_zero, heap.Get(), 1, &flag_none, nullptr, &count_two,
                                  D3D12_TILE_MAPPING_FLAG_NONE);
        queue->CopyTileMappings(b.Get(), &origin, a.Get(), &origin, &box_small, D3D12_TILE_MAPPING_FLAG_NONE);
        queue->CopyTileMappings(b.Get(), &origin, a.Get(), &origin, &box_large, D3D12_TILE_MAPPING_FLAG_NONE);
        queue->CopyTileMappings(b.Get(), &origin, a.Get(), &origin, &box_zero, D3D12_TILE_MAPPING_FLAG_NONE);

        // 4. counts[0] == 0 with an invalid D3D12_TILE_RANGE_FLAGS value must reject the entire call.
        const D3D12_TILE_RANGE_FLAGS bad_zero_flags[2] = {
            static_cast<D3D12_TILE_RANGE_FLAGS>(0x99), D3D12_TILE_RANGE_FLAG_NULL};
        const UINT bad_zero_counts[2] = {0, 2};
        queue->UpdateTileMappings(a.Get(), 1, &origin, &two, heap.Get(), 2, bad_zero_flags, nullptr, bad_zero_counts,
                                  D3D12_TILE_MAPPING_FLAG_NONE);

        // 5. CopyTiles on a non-reserved committed buffer must fail list recording.
        ComPtr<ID3D12CommandAllocator> bad_alloc;
        ComPtr<ID3D12GraphicsCommandList> bad_list;
        EXPECT(SUCCEEDED(device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&bad_alloc))));
        EXPECT(SUCCEEDED(device.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, bad_alloc.Get(), nullptr,
                                                  IID_PPV_ARGS(&bad_list))));
        bad_list->CopyTiles(committed.Get(), &origin, &one, upload.Get(), 0,
                            D3D12_TILE_COPY_FLAG_LINEAR_BUFFER_TO_SWIZZLED_TILED_RESOURCE);
        EXPECT(FAILED(bad_list->Close()));

        // Verify neither a nor b was mutated by any rejected call above.
        Snapshot after_a{}, after_b{};
        const Snapshot zeros{0, 0, 0, 0};
        EXPECT(read_tiles(device, *queue.Get(), *fence.Get(), fence_val, event.handle, a.Get(), readback.Get(),
                          after_a, baseline, "Vulkan layer (a after invalid calls)"));
        EXPECT(read_tiles(device, *queue.Get(), *fence.Get(), fence_val, event.handle, b.Get(), readback.Get(),
                          after_b, zeros, "Vulkan layer (b after invalid calls)"));
    }

    return true;
}

ComPtr<ID3D12Device> tiled_device(IDXGIFactory4& factory, LUID luid) {
    ComPtr<IDXGIAdapter1> adapter;
    if (!luid.LowPart && !luid.HighPart) return {};
    if (FAILED(factory.EnumAdapterByLuid(luid, IID_PPV_ARGS(&adapter)))) return {};
    DXGI_ADAPTER_DESC1 desc{};
    ComPtr<ID3D12Device> device;
    D3D12_FEATURE_DATA_D3D12_OPTIONS options{};
    if (FAILED(adapter->GetDesc1(&desc)) || (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) ||
        FAILED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&device))) ||
        FAILED(device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS, &options, sizeof(options))) ||
        options.TiledResourcesTier < D3D12_TILED_RESOURCES_TIER_2)
        return {};
    std::printf("Native D3D12: %ls, tiled resources tier %u\n", desc.Description, unsigned(options.TiledResourcesTier));
    return device;
}

ComPtr<ID3D12Device> warp_device(IDXGIFactory4& factory) {
    ComPtr<IDXGIAdapter1> adapter;
    if (FAILED(factory.EnumWarpAdapter(IID_PPV_ARGS(&adapter)))) return {};
    ComPtr<ID3D12Device> device;
    D3D12_FEATURE_DATA_D3D12_OPTIONS options{};
    if (FAILED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&device))) ||
        FAILED(device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS, &options, sizeof(options))) ||
        options.TiledResourcesTier < D3D12_TILED_RESOURCES_TIER_2)
        return {};
    std::printf("WARP D3D12: tiled resources tier %u\n", unsigned(options.TiledResourcesTier));
    return device;
}

}  // namespace

int main() {
    ComPtr<ID3D12Device> vulkan;
    if (FAILED(rt::d3d12::create_vulkan_device_standalone(IID_PPV_ARGS(&vulkan)))) {
        std::puts("Vulkan standalone device unavailable");
        return 77;
    }
    if (!rt::d3d12::device_sparse_buffer_tiles(vulkan.Get())) {
        std::puts("Vulkan device lacks sparse buffer residency with aliasing and strict unbound reads");
        return 77;
    }

    Results observed{};
    if (!run_valid_suite(*vulkan.Get(), observed, true, "Vulkan layer")) return 1;
    std::puts("Vulkan layer: null-offset, UseBox, zero-count, and invalid-mapping checks passed");

    ComPtr<IDXGIFactory4> factory;
    if (SUCCEEDED(CreateDXGIFactory1(IID_PPV_ARGS(&factory)))) {
        if (auto native = tiled_device(*factory.Get(), vulkan->GetAdapterLuid())) {
            Results host_results{};
            if (!run_valid_suite(*native.Get(), host_results, false, "Native D3D12") || host_results != observed)
                return 1;
            std::puts("Native D3D12 matched all 4 GPU readbacks");
        }
        if (auto warp = warp_device(*factory.Get())) {
            Results warp_results{};
            if (!run_valid_suite(*warp.Get(), warp_results, false, "WARP D3D12") || warp_results != observed)
                return 1;
            std::puts("WARP D3D12 matched all 4 GPU readbacks");
        }
    }
    return 0;
}
