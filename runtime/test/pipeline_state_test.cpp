// GPU test for CreatePipelineState: a D3D12 pipeline-state stream becomes a real VkPipeline.
//
// The stream is built the way D3D12's helper builds one (8-byte type header, payload, rounded to 8) with the two
// hand-written shaders, one vertex attribute, default blend/depth and an R8G8B8A8 render target. No game data.
// Exit 77: no Vulkan device.

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "d3d12_com.h"
#include "first_draw_shaders.h"

namespace {

int failures = 0;

#define EXPECT(condition)                                                \
    do {                                                                 \
        if (!(condition)) {                                              \
            std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); \
            ++failures;                                                  \
        }                                                                \
    } while (0)

template <typename T>
void add(std::vector<std::uint8_t>& stream, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE type, const T& payload) {
    const std::size_t start = stream.size();
    const std::size_t total = (8 + sizeof(T) + 7) & ~std::size_t(7);
    stream.resize(start + total, 0);
    std::memcpy(stream.data() + start, &type, sizeof(type));
    std::memcpy(stream.data() + start + 8, &payload, sizeof(T));
}

}  // namespace

int main() {
    ID3D12Device* device = nullptr;
    if (FAILED(rt::d3d12::create_vulkan_device_standalone(__uuidof(ID3D12Device),
                                                          reinterpret_cast<void**>(&device)))) {
        std::printf("pipeline_state_test: no Vulkan device\n");
        return 77;
    }

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
    formats.RTFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;

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

    ID3D12Device2* device2 = nullptr;
    EXPECT(SUCCEEDED(device->QueryInterface(IID_PPV_ARGS(&device2))));
    ID3D12PipelineState* pipeline_state = nullptr;
    const HRESULT hr = device2 ? device2->CreatePipelineState(&desc, IID_PPV_ARGS(&pipeline_state)) : E_NOINTERFACE;
    EXPECT(SUCCEEDED(hr));
    EXPECT(pipeline_state != nullptr);
    if (pipeline_state) {
        EXPECT(rt::d3d12::vulkan_pipeline(pipeline_state) != VK_NULL_HANDLE);
        pipeline_state->Release();
    }
    if (device2) device2->Release();
    device->Release();

    if (failures) {
        std::fprintf(stderr, "pipeline_state_test: %d failure(s)\n", failures);
        return 1;
    }
    std::printf("pipeline_state_test: all tests passed\n");
    return 0;
}
