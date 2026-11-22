// A frame through the whole D3D12 surface (PR C, step 4): CreateCommittedResource -> Map -> a command list that
// binds a render target, a vertex buffer and a pipeline state and draws -> ExecuteCommandLists -> fence -> readback.
//
// Everything goes through the COM objects of the device that leonida_shims.dll exports, on the Vulkan device the
// standalone factory created, so this is the path the game's own calls take: the vertex buffer is an upload-heap
// committed resource filled through Map(), the render target is a default-heap texture with an RTV in a descriptor
// heap, the pipeline comes from CreatePipelineState and is found again by the replay through the device's pipeline
// registry, and the readback is a CopyTextureRegion into a readback-heap committed resource whose footprint the
// device computed. The triangle's pixels are checked on the CPU after a fence wait.
//
// No game data, no keys. Exit 77 means the machine has no usable Vulkan device (the other GPU tests skip the same
// way). Set LEONIDA_VK_VALIDATION=1 to run under VK_LAYER_KHRONOS_validation; any validation error fails the test.

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "d3d12_com.h"
#include "first_draw_shaders.h"

namespace {

constexpr UINT kSize = 64;

int failures = 0;

#define EXPECT(condition)                                                \
    do {                                                                 \
        if (!(condition)) {                                              \
            std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); \
            ++failures;                                                  \
        }                                                                \
    } while (0)

template<typename T>
void add(std::vector<std::uint8_t>& stream, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE type, const T& payload) {
    const std::size_t start = stream.size();
    const std::size_t total = (8 + sizeof(T) + 7) & ~std::size_t(7);
    stream.resize(start + total, 0);
    std::memcpy(stream.data() + start, &type, sizeof(type));
    std::memcpy(stream.data() + start + 8, &payload, sizeof(T));
}

template<class T>
struct Ref {
    T* p = nullptr;
    ~Ref() {
        if (p) p->Release();
    }
    T* operator->() const { return p; }
    T** out() { return &p; }
    void** as_void() { return reinterpret_cast<void**>(&p); }
};

D3D12_HEAP_PROPERTIES heap_properties(D3D12_HEAP_TYPE type) {
    D3D12_HEAP_PROPERTIES heap{};
    heap.Type = type;
    heap.CreationNodeMask = 1;
    heap.VisibleNodeMask = 1;
    return heap;
}

D3D12_RESOURCE_DESC buffer_desc(UINT64 size) {
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width = size;
    desc.Height = 1;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.Format = DXGI_FORMAT_UNKNOWN;
    desc.SampleDesc = {1, 0};
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    return desc;
}

D3D12_RESOURCE_DESC texture_desc(DXGI_FORMAT format, D3D12_RESOURCE_FLAGS flags) {
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    desc.Width = kSize;
    desc.Height = kSize;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.Format = format;
    desc.SampleDesc = {1, 0};
    desc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    desc.Flags = flags;
    return desc;
}

// D3D12_HEAP_TYPE_DEFAULT, an initial state the layer ignores (it tracks the resource from UNDEFINED and zero-fills
// it on first use, like a committed resource).
HRESULT create_texture(ID3D12Device* device, const D3D12_RESOURCE_DESC& desc, D3D12_RESOURCE_STATES state,
                       ID3D12Resource** out) {
    const D3D12_HEAP_PROPERTIES heap = heap_properties(D3D12_HEAP_TYPE_DEFAULT);
    return device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, state, nullptr,
                                           __uuidof(ID3D12Resource), reinterpret_cast<void**>(out));
}

HRESULT create_buffer(ID3D12Device* device, D3D12_HEAP_TYPE type, UINT64 size, ID3D12Resource** out) {
    const D3D12_HEAP_PROPERTIES heap = heap_properties(type);
    const D3D12_RESOURCE_DESC desc = buffer_desc(size);
    return device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COMMON, nullptr,
                                           __uuidof(ID3D12Resource), reinterpret_cast<void**>(out));
}

// A graphics pipeline for the two test shaders: one R32G32_FLOAT attribute, no culling, one RGBA8 target.
HRESULT create_pipeline(ID3D12Device* device, DXGI_FORMAT color_format, ID3D12PipelineState** out) {
    D3D12_SHADER_BYTECODE vs{first_draw::triangle_vert_spv, sizeof(first_draw::triangle_vert_spv)};
    D3D12_SHADER_BYTECODE ps{first_draw::triangle_frag_spv, sizeof(first_draw::triangle_frag_spv)};
    D3D12_INPUT_ELEMENT_DESC element{"POSITION", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 0,
                                     D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0};
    D3D12_INPUT_LAYOUT_DESC layout{&element, 1};
    D3D12_BLEND_DESC blend{};
    blend.RenderTarget[0].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
    D3D12_DEPTH_STENCIL_DESC depth{};
    D3D12_RASTERIZER_DESC raster{};
    raster.FillMode = D3D12_FILL_MODE_SOLID;
    raster.CullMode = D3D12_CULL_MODE_NONE;
    D3D12_RT_FORMAT_ARRAY formats{};
    formats.NumRenderTargets = 1;
    formats.RTFormats[0] = color_format;

    std::vector<std::uint8_t> stream;
    add(stream, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_VS, vs);
    add(stream, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PS, ps);
    add(stream, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_INPUT_LAYOUT, layout);
    add(stream, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_BLEND, blend);
    add(stream, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL, depth);
    add(stream, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RASTERIZER, raster);
    add(stream, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PRIMITIVE_TOPOLOGY, D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE);
    add(stream, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RENDER_TARGET_FORMATS, formats);
    add(stream, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL_FORMAT, DXGI_FORMAT_UNKNOWN);
    add(stream, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_SAMPLE_DESC, DXGI_SAMPLE_DESC{1, 0});

    D3D12_PIPELINE_STATE_STREAM_DESC desc{};
    desc.SizeInBytes = stream.size();
    desc.pPipelineStateSubobjectStream = stream.data();

    Ref<ID3D12Device2> device2;
    if (FAILED(device->QueryInterface(IID_PPV_ARGS(device2.out())))) return E_NOINTERFACE;
    return device2->CreatePipelineState(&desc, __uuidof(ID3D12PipelineState), reinterpret_cast<void**>(out));
}

// Blocks until the queue has reached `value` on `fence`, the way a frame does before it reads a readback buffer.
bool wait_for(ID3D12CommandQueue* queue, ID3D12Fence* fence, UINT64 value) {
    const HANDLE event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    if (!event) return false;
    bool ok = SUCCEEDED(queue->Signal(fence, value)) && SUCCEEDED(fence->SetEventOnCompletion(value, event));
    ok = ok && WaitForSingleObject(event, 10'000) == WAIT_OBJECT_0;
    CloseHandle(event);
    return ok;
}

}  // namespace

int main() {
    Ref<ID3D12Device> device;
    if (FAILED(rt::d3d12::create_vulkan_device_standalone(__uuidof(ID3D12Device), device.as_void()))) {
        std::printf("com_draw_test: no Vulkan device\n");
        return 77;
    }

    // The render target, its view, the pipeline and the vertex buffer, all through the D3D12 surface.
    const D3D12_RESOURCE_DESC rt_desc = texture_desc(DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET);
    Ref<ID3D12Resource> render_target;
    EXPECT(SUCCEEDED(create_texture(device.p, rt_desc, D3D12_RESOURCE_STATE_RENDER_TARGET, render_target.out())));

    D3D12_DESCRIPTOR_HEAP_DESC rtv_heap_desc{};
    rtv_heap_desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
    rtv_heap_desc.NumDescriptors = 1;
    Ref<ID3D12DescriptorHeap> rtv_heap;
    EXPECT(SUCCEEDED(device->CreateDescriptorHeap(&rtv_heap_desc, __uuidof(ID3D12DescriptorHeap), rtv_heap.as_void())));
    const D3D12_CPU_DESCRIPTOR_HANDLE rtv = rtv_heap->GetCPUDescriptorHandleForHeapStart();
    if (render_target.p && rtv_heap.p) device->CreateRenderTargetView(render_target.p, nullptr, rtv);

    Ref<ID3D12PipelineState> pipeline;
    EXPECT(SUCCEEDED(create_pipeline(device.p, DXGI_FORMAT_R8G8B8A8_UNORM, pipeline.out())));

    const float vertices[6] = {-0.8f, -0.8f, 0.8f, -0.8f, 0.0f, 0.8f};
    Ref<ID3D12Resource> vertex_buffer;
    EXPECT(SUCCEEDED(create_buffer(device.p, D3D12_HEAP_TYPE_UPLOAD, sizeof(vertices), vertex_buffer.out())));
    if (vertex_buffer.p) {
        void* mapped = nullptr;
        EXPECT(SUCCEEDED(vertex_buffer->Map(0, nullptr, &mapped)));
        if (mapped) std::memcpy(mapped, vertices, sizeof(vertices));
        vertex_buffer->Unmap(0, nullptr);
    }
    D3D12_VERTEX_BUFFER_VIEW vertex_view{};
    if (vertex_buffer.p) {
        vertex_view.BufferLocation = vertex_buffer->GetGPUVirtualAddress();
        vertex_view.SizeInBytes = sizeof(vertices);
        vertex_view.StrideInBytes = 2 * sizeof(float);
    }

    // The readback buffer: its footprint comes from the device, as a frame's would.
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
    UINT rows = 0;
    UINT64 row_size = 0, total = 0;
    device->GetCopyableFootprints(&rt_desc, 0, 1, 0, &footprint, &rows, &row_size, &total);
    EXPECT(rows == kSize && row_size == kSize * 4 && total >= kSize * footprint.Footprint.RowPitch);
    Ref<ID3D12Resource> readback;
    EXPECT(SUCCEEDED(create_buffer(device.p, D3D12_HEAP_TYPE_READBACK, total, readback.out())));

    D3D12_COMMAND_QUEUE_DESC queue_desc{};
    queue_desc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    Ref<ID3D12CommandQueue> queue;
    EXPECT(SUCCEEDED(device->CreateCommandQueue(&queue_desc, __uuidof(ID3D12CommandQueue), queue.as_void())));
    Ref<ID3D12CommandAllocator> draw_allocator, copy_allocator;
    EXPECT(SUCCEEDED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
                                                    __uuidof(ID3D12CommandAllocator), draw_allocator.as_void())));
    EXPECT(SUCCEEDED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
                                                    __uuidof(ID3D12CommandAllocator), copy_allocator.as_void())));
    Ref<ID3D12GraphicsCommandList> draw_list, copy_list;
    EXPECT(SUCCEEDED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, draw_allocator.p, nullptr,
                                               __uuidof(ID3D12GraphicsCommandList), draw_list.as_void())));
    EXPECT(SUCCEEDED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, copy_allocator.p, nullptr,
                                               __uuidof(ID3D12GraphicsCommandList), copy_list.as_void())));
    Ref<ID3D12Fence> fence;
    EXPECT(SUCCEEDED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, __uuidof(ID3D12Fence), fence.as_void())));

    // Frame recording: bind the target, the vertex buffer and the pipeline, then draw one triangle.
    draw_list->OMSetRenderTargets(1, &rtv, FALSE, nullptr);
    draw_list->IASetVertexBuffers(0, 1, &vertex_view);
    draw_list->SetPipelineState(pipeline.p);
    draw_list->DrawInstanced(3, 1, 0, 0);
    EXPECT(draw_list->Close() == S_OK);

    // Readback recording: the target leaves the render-target state and is copied into the placed footprint.
    D3D12_RESOURCE_BARRIER to_copy{};
    to_copy.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    to_copy.Transition.pResource = render_target.p;
    to_copy.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    to_copy.Transition.StateBefore = D3D12_RESOURCE_STATE_RENDER_TARGET;
    to_copy.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
    copy_list->ResourceBarrier(1, &to_copy);
    D3D12_TEXTURE_COPY_LOCATION destination{};
    destination.pResource = readback.p;
    destination.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    destination.PlacedFootprint = footprint;
    D3D12_TEXTURE_COPY_LOCATION source{};
    source.pResource = render_target.p;
    source.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    source.SubresourceIndex = 0;
    copy_list->CopyTextureRegion(&destination, 0, 0, 0, &source, nullptr);
    EXPECT(copy_list->Close() == S_OK);

    // Both lists travel together, in order, exactly as a frame submits them.
    ID3D12CommandList* lists[] = {draw_list.p, copy_list.p};
    queue->ExecuteCommandLists(2, lists);
    EXPECT(rt::d3d12::last_execute_result(queue.p) == leonida::ReplayResult::success);
    EXPECT(wait_for(queue.p, fence.p, 1));
    EXPECT(fence->GetCompletedValue() == 1);

    void* mapped = nullptr;
    if (readback.p) EXPECT(SUCCEEDED(readback->Map(0, nullptr, &mapped)));
    const auto* pixels = static_cast<const std::uint8_t*>(mapped);
    std::size_t orange = 0, lit = 0, not_zeroed = 0, alpha_missing = 0;
    if (pixels) {
        for (UINT row = 0; row < kSize; ++row) {
            const std::uint8_t* line = pixels + footprint.Offset + std::size_t(row) * footprint.Footprint.RowPitch;
            for (UINT col = 0; col < kSize; ++col) {
                const std::uint8_t r = line[col * 4], g = line[col * 4 + 1], b = line[col * 4 + 2],
                                    a = line[col * 4 + 3];
                if (r || g || b || a) ++lit;
                const bool is_orange = r > 200 && g > 30 && g < 120 && b < 40;  // vec4(1.0, 0.25, 0.0, 1.0)
                if (is_orange) {
                    ++orange;
                    if (a != 255) ++alpha_missing;
                }
                // Outside the triangle the committed resource must be zero-filled: colour zero, not 0xcd (the
                // pattern the layer writes into host-visible buffers at creation).
                if (!(r || g || b) && a != 0) ++not_zeroed;
            }
        }
    }
    std::printf("com_draw_test: %zu/%u pixels lit, %zu orange\n", lit, kSize * kSize, orange);

    if (pipeline.p) EXPECT(rt::d3d12::vulkan_pipeline(pipeline.p) != VK_NULL_HANDLE);
    if (pixels) {
        EXPECT(lit > 0);
        EXPECT(orange > 0);
        EXPECT(lit == orange);      // every lit pixel is the shader's orange
        EXPECT(!not_zeroed);        // the rest of the target was zero-filled
        EXPECT(!alpha_missing);     // and the shader's alpha reached the target
    }

    if (failures) {
        std::fprintf(stderr, "com_draw_test: %d failure(s)\n", failures);
        return 1;
    }
    std::printf("com_draw_test: all tests passed\n");
    return 0;
}
