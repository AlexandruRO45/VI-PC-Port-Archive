// Multiple render targets (PR C): one draw writes several colour attachments, the way deferred rendering does.
//
// The pipeline-state stream declares two render-target formats (R8G8B8A8_UNORM and B8G8R8A8_UNORM, so a per-slot
// format mix-up would show), OMSetRenderTargets binds two RTVs and emit() opens one dynamic-rendering scope with both
// of them. The test shader has a single fragment output, so attachment 1 is intentionally left undefined by the
// pipeline: Vulkan keeps whatever the attachment holds there, which is the layer's zero fill for a committed
// resource. The check is therefore: attachment 0 has the triangle's orange, attachment 1 keeps its zeros, and the
// whole replay is validation-clean (run with LEONIDA_VK_VALIDATION=1).
//
// No game data, no keys. Exit 77 means the machine has no usable Vulkan device.

#include "d3d12_draw_scene.h"

#include <cstdint>
#include <cstdio>
#include <cstring>

namespace {

using namespace d3d12_scene;

constexpr UINT kSize = 64;

int failures = 0;

#define EXPECT(condition)                                                \
    do {                                                                 \
        if (!(condition)) {                                              \
            std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); \
            ++failures;                                                  \
        }                                                                \
    } while (0)

}  // namespace

int main() {
    Ref<ID3D12Device> device;
    if (FAILED(rt::d3d12::create_vulkan_device_standalone(__uuidof(ID3D12Device), device.as_void()))) {
        std::printf("mrt_test: no Vulkan device\n");
        return 77;
    }

    // Two render targets with different formats, and one RTV heap with a slot for each.
    const DXGI_FORMAT formats[2] = {DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_FORMAT_B8G8R8A8_UNORM};
    const D3D12_RESOURCE_DESC descs[2] = {
        texture_desc(kSize, kSize, formats[0], D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET),
        texture_desc(kSize, kSize, formats[1], D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET),
    };
    Ref<ID3D12Resource> targets[2];
    for (int i = 0; i < 2; ++i)
        EXPECT(SUCCEEDED(create_texture(device.p, descs[i], D3D12_RESOURCE_STATE_RENDER_TARGET, targets[i].out())));

    D3D12_DESCRIPTOR_HEAP_DESC rtv_heap_desc{};
    rtv_heap_desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
    rtv_heap_desc.NumDescriptors = 2;
    Ref<ID3D12DescriptorHeap> rtv_heap;
    EXPECT(SUCCEEDED(device->CreateDescriptorHeap(&rtv_heap_desc, __uuidof(ID3D12DescriptorHeap), rtv_heap.as_void())));
    const UINT stride = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
    D3D12_CPU_DESCRIPTOR_HANDLE rtvs[2] = {rtv_heap->GetCPUDescriptorHandleForHeapStart(),
                                           rtv_heap->GetCPUDescriptorHandleForHeapStart()};
    rtvs[1].ptr += stride;
    if (targets[0].p && rtv_heap.p) device->CreateRenderTargetView(targets[0].p, nullptr, rtvs[0]);
    if (targets[1].p && rtv_heap.p) device->CreateRenderTargetView(targets[1].p, nullptr, rtvs[1]);

    Ref<ID3D12PipelineState> pipeline;
    EXPECT(SUCCEEDED(create_pipeline(device.p, std::span<const DXGI_FORMAT>(formats, 2), pipeline.out())));

    Ref<ID3D12Resource> vertex_buffer;
    EXPECT(SUCCEEDED(create_buffer(device.p, D3D12_HEAP_TYPE_UPLOAD, sizeof(kTriangle), vertex_buffer.out())));
    if (vertex_buffer.p) {
        void* mapped = nullptr;
        EXPECT(SUCCEEDED(vertex_buffer->Map(0, nullptr, &mapped)));
        if (mapped) std::memcpy(mapped, kTriangle, sizeof(kTriangle));
        vertex_buffer->Unmap(0, nullptr);
    }
    D3D12_VERTEX_BUFFER_VIEW vertex_view{};
    if (vertex_buffer.p) {
        vertex_view.BufferLocation = vertex_buffer->GetGPUVirtualAddress();
        vertex_view.SizeInBytes = sizeof(kTriangle);
        vertex_view.StrideInBytes = 2 * sizeof(float);
    }

    // Readback for both attachments, each laid out by the device's own footprint.
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprints[2]{};
    UINT64 slot_bytes[2]{};
    UINT64 total = 0;
    for (int i = 0; i < 2; ++i) {
        UINT rows = 0;
        UINT64 row_size = 0;
        device->GetCopyableFootprints(&descs[i], 0, 1, 0, &footprints[i], &rows, &row_size, &slot_bytes[i]);
        EXPECT(rows == kSize && row_size == kSize * 4);
        footprints[i].Offset = total;
        total += slot_bytes[i];
    }
    Ref<ID3D12Resource> readback;
    EXPECT(SUCCEEDED(create_buffer(device.p, D3D12_HEAP_TYPE_READBACK, total, readback.out())));

    D3D12_COMMAND_QUEUE_DESC queue_desc{};
    queue_desc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    Ref<ID3D12CommandQueue> queue;
    EXPECT(SUCCEEDED(device->CreateCommandQueue(&queue_desc, __uuidof(ID3D12CommandQueue), queue.as_void())));
    Ref<ID3D12CommandAllocator> allocator;
    EXPECT(SUCCEEDED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
                                                    __uuidof(ID3D12CommandAllocator), allocator.as_void())));
    Ref<ID3D12GraphicsCommandList> list;
    EXPECT(SUCCEEDED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.p, nullptr,
                                               __uuidof(ID3D12GraphicsCommandList), list.as_void())));
    Ref<ID3D12Fence> fence;
    EXPECT(SUCCEEDED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, __uuidof(ID3D12Fence), fence.as_void())));

    // One list: bind both attachments, draw the triangle, then read both back.
    list->OMSetRenderTargets(2, rtvs, FALSE, nullptr);
    list->IASetVertexBuffers(0, 1, &vertex_view);
    list->SetPipelineState(pipeline.p);
    list->DrawInstanced(3, 1, 0, 0);
    for (int i = 0; i < 2; ++i) {
        D3D12_RESOURCE_BARRIER to_copy{};
        to_copy.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        to_copy.Transition.pResource = targets[i].p;
        to_copy.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        to_copy.Transition.StateBefore = D3D12_RESOURCE_STATE_RENDER_TARGET;
        to_copy.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
        list->ResourceBarrier(1, &to_copy);
        D3D12_TEXTURE_COPY_LOCATION destination{};
        destination.pResource = readback.p;
        destination.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        destination.PlacedFootprint = footprints[i];
        D3D12_TEXTURE_COPY_LOCATION source{};
        source.pResource = targets[i].p;
        source.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        source.SubresourceIndex = 0;
        list->CopyTextureRegion(&destination, 0, 0, 0, &source, nullptr);
    }
    EXPECT(list->Close() == S_OK);

    ID3D12CommandList* lists[] = {list.p};
    queue->ExecuteCommandLists(1, lists);
    EXPECT(rt::d3d12::last_execute_result(queue.p) == leonida::ReplayResult::success);
    EXPECT(wait_for(queue.p, fence.p, 1));

    void* mapped = nullptr;
    if (readback.p) EXPECT(SUCCEEDED(readback->Map(0, nullptr, &mapped)));
    const auto* pixels = static_cast<const std::uint8_t*>(mapped);
    std::size_t orange0 = 0, non_zero1 = 0;
    if (pixels) {
        const std::uint8_t* first = pixels + footprints[0].Offset;
        for (UINT row = 0; row < kSize; ++row) {
            const std::uint8_t* line = first + std::size_t(row) * footprints[0].Footprint.RowPitch;
            for (UINT col = 0; col < kSize; ++col) {
                const std::uint8_t r = line[col * 4], g = line[col * 4 + 1], b = line[col * 4 + 2];
                if (r > 200 && g > 30 && g < 120 && b < 40) ++orange0;
            }
        }
        const std::uint8_t* second = pixels + footprints[1].Offset;
        for (UINT row = 0; row < kSize; ++row) {
            const std::uint8_t* line = second + std::size_t(row) * footprints[1].Footprint.RowPitch;
            for (UINT col = 0; col < kSize; ++col)
                if (line[col * 4] || line[col * 4 + 1] || line[col * 4 + 2] || line[col * 4 + 3]) ++non_zero1;
        }
    }
    std::printf("mrt_test: attachment 0 has %zu orange pixels, attachment 1 has %zu non-zero\n", orange0, non_zero1);

    EXPECT(orange0 > 1000);  // the triangle reached the first attachment
    EXPECT(non_zero1 == 0);  // the second one kept its zero fill: nothing else wrote into it
    EXPECT(rt::d3d12::vulkan_image(targets[0].p) != VK_NULL_HANDLE);
    EXPECT(rt::d3d12::vulkan_image(targets[1].p) != VK_NULL_HANDLE);

    if (failures) {
        std::fprintf(stderr, "mrt_test: %d failure(s)\n", failures);
        return 1;
    }
    std::printf("mrt_test: all tests passed\n");
    return 0;
}
