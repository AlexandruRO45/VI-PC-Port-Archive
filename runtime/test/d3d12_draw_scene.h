// Helpers shared by the tests that drive a frame through the D3D12 COM surface (com_draw_test, window_draw_test):
// committed resources, the pipeline-state stream of the two test shaders, and a fence wait.
#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <span>
#include <vector>

#include "d3d12_com.h"
#include "first_draw_shaders.h"

namespace d3d12_scene {

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

inline D3D12_HEAP_PROPERTIES heap_properties(D3D12_HEAP_TYPE type) {
    D3D12_HEAP_PROPERTIES heap{};
    heap.Type = type;
    heap.CreationNodeMask = 1;
    heap.VisibleNodeMask = 1;
    return heap;
}

inline D3D12_RESOURCE_DESC buffer_desc(UINT64 size) {
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

inline D3D12_RESOURCE_DESC texture_desc(UINT width, UINT height, DXGI_FORMAT format, D3D12_RESOURCE_FLAGS flags) {
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    desc.Width = width;
    desc.Height = height;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.Format = format;
    desc.SampleDesc = {1, 0};
    desc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    desc.Flags = flags;
    return desc;
}

// A default-heap texture. `state` is the initial state the layer does not need: it tracks the resource from
// UNDEFINED and zero-fills it on first use, like a committed resource.
inline HRESULT create_texture(ID3D12Device* device, const D3D12_RESOURCE_DESC& desc, D3D12_RESOURCE_STATES state,
                              ID3D12Resource** out) {
    const D3D12_HEAP_PROPERTIES heap = heap_properties(D3D12_HEAP_TYPE_DEFAULT);
    return device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, state, nullptr,
                                           __uuidof(ID3D12Resource), reinterpret_cast<void**>(out));
}

inline HRESULT create_buffer(ID3D12Device* device, D3D12_HEAP_TYPE type, UINT64 size, ID3D12Resource** out) {
    const D3D12_HEAP_PROPERTIES heap = heap_properties(type);
    const D3D12_RESOURCE_DESC desc = buffer_desc(size);
    return device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COMMON, nullptr,
                                           __uuidof(ID3D12Resource), reinterpret_cast<void**>(out));
}

template<typename T>
inline void add(std::vector<std::uint8_t>& stream, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE type, const T& payload) {
    const std::size_t start = stream.size();
    const std::size_t total = (8 + sizeof(T) + 7) & ~std::size_t(7);
    stream.resize(start + total, 0);
    std::memcpy(stream.data() + start, &type, sizeof(type));
    std::memcpy(stream.data() + start + 8, &payload, sizeof(T));
}

// A graphics pipeline for the two test shaders: one R32G32_FLOAT attribute, no culling, one attachment per colour
// format in `color_formats` (up to eight).
inline HRESULT create_pipeline(ID3D12Device* device, std::span<const DXGI_FORMAT> color_formats,
                               ID3D12PipelineState** out) {
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
    formats.NumRenderTargets = std::uint32_t(std::min<std::size_t>(color_formats.size(), 8));
    for (std::uint32_t i = 0; i < formats.NumRenderTargets; ++i) formats.RTFormats[i] = color_formats[i];

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

inline HRESULT create_pipeline(ID3D12Device* device, DXGI_FORMAT color_format, ID3D12PipelineState** out) {
    return create_pipeline(device, std::span<const DXGI_FORMAT>(&color_format, 1), out);
}

// Blocks until the queue has reached `value` on `fence`, the way a frame does before it touches what the GPU wrote.
inline bool wait_for(ID3D12CommandQueue* queue, ID3D12Fence* fence, UINT64 value) {
    const HANDLE event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    if (!event) return false;
    bool ok = SUCCEEDED(queue->Signal(fence, value)) && SUCCEEDED(fence->SetEventOnCompletion(value, event));
    ok = ok && WaitForSingleObject(event, 10'000) == WAIT_OBJECT_0;
    CloseHandle(event);
    return ok;
}

// The triangle the shaders draw: three corners at +-0.8, one attribute of two floats.
inline constexpr float kTriangle[6] = {-0.8f, -0.8f, 0.8f, -0.8f, 0.0f, 0.8f};

}  // namespace d3d12_scene
