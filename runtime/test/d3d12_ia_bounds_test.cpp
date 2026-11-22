// Input-assembler reads past a buffer view, on Windows D3D12 (default adapter and WARP) and on the Vulkan layer.
// D3D12 treats a vertex or an index fetched outside its view as zero. The layer keeps all buffer resources in one
// GPU address arena buffer, so what lies past a view in the arena is real data unless the layer bounds the binding
// (an index buffer bound with a size, robust buffer access for vertex fetches).
//
// Every case draws the same kind of scene on each device, into a 64x64 target cleared to blue, and compares probe
// pixels: the resource that holds a view's data also holds distinctive data right after the view, so a read past the
// view draws a visible triangle (green) instead of what D3D12 gives.
//   A16 / A32  indexed draw whose R16 / R32 index view covers 6 indices (the left half, red) while the resource holds
//              3 more that form a green triangle; the draw asks for 9.
//   B1         vertex view covers 3 vertices (a red triangle), the draw indexes vertices 3..5 beyond it (green data).
//   B2         the same, not indexed: Draw(6).
//   B3         mixed: one triangle takes one vertex from beyond the view; what D3D12 fetches there decides how it looks.
//   C          indexed draw with no index buffer bound and more indices than the layer's zero page holds (a decoy
//              buffer is placed right behind the page, the arena's first allocation).
//   D1 / D2    in-range controls, indexed and not.
//   E          the vertex view ends 8 bytes into vertex 3 (position inside, colour outside): D3D12 keeps the position
//              and zeroes the colour; robustBufferAccess2 may zero more, so another driver may differ.
// A second layer device with LEONIDA_NO_IA_BOUNDS=1 (no sized index bind, no robustness) must still draw the in-range
// cases (D1, D2) as the host does; what it draws for the cases that read past a view is printed, not asserted.
// The measured outcomes are in `expected` (docs/d3d12_vulkan.md). A case counts only where the default adapter and WARP
// agree; where they do not the numbers are printed as information. Under Wine the host is not run (host_wine.h) and the
// layer is checked against `expected` alone. The layer device is the standalone one (create_vulkan_device_standalone).
// The shaders are GLSL written to the translator's contract (test/glsl/vs_ia_bounds.vert, ps_ia_bounds.frag), and
// HLSL with the same behaviour for the host. No game data.
#include "d3d12_test_shaders.h"
#include "host_wine.h"
#include "../shims/d3d12_com.h"
#include "../src/shader_library.h"

#include <d3dcompiler.h>
#include <dxgi1_4.h>

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#if !defined(LEONIDA_TEST_VS_SPV) || !defined(LEONIDA_TEST_PS_SPV)
#error "LEONIDA_TEST_VS_SPV and LEONIDA_TEST_PS_SPV must name the compiled test shaders"
#endif

#define EXPECT(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return false; } \
} while (0)

using namespace d3d12test;

namespace {

constexpr UINT target_extent = 64;
constexpr UINT probe_count = 4;
constexpr std::uint64_t arena_reserved = 64 * 1024;  // the layer's zero page (kArenaReserved)
constexpr DXGI_FORMAT no_index_buffer = DXGI_FORMAT_UNKNOWN;

struct Vertex {
    float x, y;
    std::uint8_t r, g, b, a;
};
static_assert(sizeof(Vertex) == 12);

struct Pixel {
    std::uint8_t r, g, b, a;
};
using Probes = std::array<Pixel, probe_count>;

constexpr Pixel blue{0, 0, 255, 255}, red{255, 0, 0, 255};

// Two vertex sets of seven vertices. Set 0: a quad over the left half (red) and a triangle in the top right (green).
// Set 1: a big triangle over the top left (red) and a triangle in the bottom right (green).
constexpr Vertex vertex_sets[2][7] = {
    {{-1, 1, 255, 0, 0, 255}, {0, 1, 255, 0, 0, 255}, {-1, -1, 255, 0, 0, 255}, {0, -1, 255, 0, 0, 255},
     {0.25f, 0.75f, 0, 255, 0, 255}, {0.75f, 0.75f, 0, 255, 0, 255}, {0.5f, 0.25f, 0, 255, 0, 255}},
    {{-0.9f, 0.9f, 255, 0, 0, 255}, {0.9f, 0.9f, 255, 0, 0, 255}, {-0.9f, -0.9f, 255, 0, 0, 255},
     {0.1f, -0.1f, 0, 255, 0, 255}, {0.9f, -0.1f, 0, 255, 0, 255}, {0.5f, -0.9f, 0, 255, 0, 255},
     {0, 0, 0, 255, 0, 255}}};
constexpr UINT probe_xy[3][probe_count][2] = {{{16, 16}, {16, 48}, {48, 12}, {48, 48}},
                                              {{12, 12}, {48, 44}, {12, 52}, {32, 8}},
                                              {{32, 6}, {32, 16}, {32, 28}, {48, 12}}};

struct Case {
    const char* name;
    int set;                      // vertex set
    UINT vertex_view;             // vertices the vertex buffer view covers
    DXGI_FORMAT index_format;     // no_index_buffer: none bound (indexed draws only)
    std::vector<std::uint32_t> indices;  // the index resource's contents, view first
    UINT index_view;              // indices the view covers
    bool indexed;
    UINT draw_count;
    int tolerance;                // allowed difference per channel between devices (interpolated colours)
    bool decoy;                   // case C: the index draw reads the decoy right behind the zero page
    bool defined;                 // the measured outcome is a pass/fail condition; otherwise information only
    Probes expected;              // measured on Windows D3D12, default adapter and WARP agreeing
    UINT vertex_extra_bytes = 0;  // bytes of one more vertex the vertex view covers (a partial vertex)
    int probe_set = -1;           // the probes to read, when not those of `set`
    bool in_range = false;        // nothing is read past a view: also checked on a layer device without the bounds
};

const std::vector<Case>& cases() {
    static const std::vector<Case> all = {
        {"A16 index view 6 of 9, R16", 0, 7, DXGI_FORMAT_R16_UINT, {0, 1, 2, 2, 1, 3, 4, 5, 6}, 6, true, 9, 0, false,
         true, {red, red, blue, blue}},
        {"A32 index view 6 of 9, R32", 0, 7, DXGI_FORMAT_R32_UINT, {0, 1, 2, 2, 1, 3, 4, 5, 6}, 6, true, 9, 0, false,
         true, {red, red, blue, blue}},
        {"B1 indexed, vertices 3..5 past a view of 3", 1, 3, DXGI_FORMAT_R16_UINT, {0, 1, 2, 3, 4, 5}, 6, true, 6, 0,
         false, true, {red, blue, blue, red}},
        {"B2 Draw(6), vertices 3..5 past a view of 3", 1, 3, no_index_buffer, {}, 0, false, 6, 0, false, true,
         {red, blue, blue, red}},
        {"B3 one vertex from past the view", 1, 3, DXGI_FORMAT_R16_UINT, {0, 1, 2, 0, 1, 3}, 6, true, 6, 4, false,
         true, {Pixel{173, 0, 0, 173}, blue, blue, Pixel{208, 0, 0, 208}}},
        {"C no index buffer, count past the zero page", 0, 7, no_index_buffer, {}, 0, true,
         UINT(arena_reserved / 2) + 4, 0, true, true, {blue, blue, blue, blue}},
        {"D1 in-range indexed control", 0, 7, DXGI_FORMAT_R16_UINT, {0, 1, 2, 2, 1, 3}, 6, true, 6, 0, false, true,
         {red, red, blue, blue}, 0, -1, true},
        {"D2 in-range Draw control", 1, 3, no_index_buffer, {}, 0, false, 3, 0, false, true, {red, blue, blue, red},
         0, -1, true},
        // The view ends 8 bytes into vertex 3: its position is inside, its colour outside. Measured: D3D12 keeps the
        // position and zeroes the colour (per element). Vulkan allows other vertex inputs of a binding to read as zero
        // too, so the layer gives each element its own binding (D3D-353) and the bound is per element on any driver.
        {"E partial vertex: view of 3 vertices + 8 bytes, uses vertex 3", 1, 3, DXGI_FORMAT_R16_UINT, {0, 1, 3}, 3,
         true, 3, 4, false, true,
         {Pixel{229, 0, 0, 229}, Pixel{149, 0, 0, 149}, Pixel{53, 0, 0, 53}, Pixel{181, 0, 0, 181}}, 8, 2},
    };
    return all;
}

bool close_enough(const Pixel& a, const Pixel& b, int tolerance) {
    return std::abs(int(a.r) - int(b.r)) <= tolerance && std::abs(int(a.g) - int(b.g)) <= tolerance &&
           std::abs(int(a.b) - int(b.b)) <= tolerance && std::abs(int(a.a) - int(b.a)) <= tolerance;
}

bool same(const Probes& a, const Probes& b, int tolerance) {
    for (UINT i = 0; i < probe_count; ++i)
        if (!close_enough(a[i], b[i], tolerance)) return false;
    return true;
}

std::string text(const Probes& probes) {
    std::string out;
    for (const Pixel& p : probes) {
        char buffer[40];
        std::snprintf(buffer, sizeof(buffer), "(%3u,%3u,%3u,%3u) ", p.r, p.g, p.b, p.a);
        out += buffer;
    }
    return out;
}

// A temporary folder that goes away on every way out of main.
struct TempFolder {
    std::filesystem::path path;
    ~TempFolder() {
        std::error_code ignored;
        if (!path.empty()) std::filesystem::remove_all(path, ignored);
    }
};

struct Event {
    HANDLE handle = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    ~Event() {
        if (handle) CloseHandle(handle);
    }
};

const char* const vertex_hlsl =
    "struct In { float2 position : POSITION; float4 color : COLOR; };\n"
    "struct Out { float4 position : SV_Position; float4 color : COLOR; };\n"
    "Out main(In v) { Out o; o.position = float4(v.position, 0, 1); o.color = v.color; return o; }\n";
const char* const pixel_hlsl =
    "float4 main(float4 position : SV_Position, float4 color : COLOR) : SV_Target { return color; }\n";

bool compile_hlsl(const char* source, const char* target, std::vector<std::uint8_t>& out) {
    const HMODULE compiler = LoadLibraryExA("d3dcompiler_47.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
    EXPECT(compiler != nullptr);
    const auto compile = reinterpret_cast<pD3DCompile>(reinterpret_cast<void*>(GetProcAddress(compiler, "D3DCompile")));
    EXPECT(compile != nullptr);
    ID3DBlob* blob = nullptr;
    ID3DBlob* errors = nullptr;
    const HRESULT hr = compile(source, std::strlen(source), "d3d12_ia_bounds_test.hlsl", nullptr, nullptr, "main",
                               target, D3DCOMPILE_ENABLE_STRICTNESS, 0, &blob, &errors);
    if (FAILED(hr) && errors) std::fprintf(stderr, "HLSL: %s\n", static_cast<const char*>(errors->GetBufferPointer()));
    if (errors) errors->Release();
    EXPECT(SUCCEEDED(hr) && blob);
    const auto* bytes = static_cast<const std::uint8_t*>(blob->GetBufferPointer());
    out.assign(bytes, bytes + blob->GetBufferSize());
    blob->Release();
    return true;
}

// One device with the root signature and PSO its cases draw with.
struct Context {
    Ref<ID3D12Device> device;
    Ref<ID3D12CommandQueue> queue;
    Ref<ID3D12RootSignature> root;
    Ref<ID3D12PipelineState> pso;
};

bool make_context(Context& c, const std::vector<std::uint8_t>& vs, const std::vector<std::uint8_t>& ps) {
    D3D12_COMMAND_QUEUE_DESC queue_desc{};
    EXPECT(SUCCEEDED(c.device->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&c.queue.p))));
    D3D12_VERSIONED_ROOT_SIGNATURE_DESC root_desc{};
    root_desc.Version = D3D_ROOT_SIGNATURE_VERSION_1_0;
    root_desc.Desc_1_0 = {0, nullptr, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT};
    const auto root_blob = serialize_root_signature(root_desc);
    EXPECT(!root_blob.empty());
    EXPECT(SUCCEEDED(c.device->CreateRootSignature(0, root_blob.data(), root_blob.size(), IID_PPV_ARGS(&c.root.p))));
    const D3D12_INPUT_ELEMENT_DESC elements[2] = {
        {"POSITION", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 0, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
        {"COLOR", 0, DXGI_FORMAT_R8G8B8A8_UNORM, 0, 8, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0}};
    D3D12_GRAPHICS_PIPELINE_STATE_DESC desc{};
    desc.pRootSignature = c.root.p;
    desc.VS = {vs.data(), vs.size()};
    desc.PS = {ps.data(), ps.size()};
    desc.BlendState.RenderTarget[0].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
    desc.SampleMask = 0xffffffff;
    desc.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
    desc.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
    desc.RasterizerState.DepthClipEnable = TRUE;
    desc.InputLayout = {elements, 2};
    desc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    desc.NumRenderTargets = 1;
    desc.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
    desc.SampleDesc.Count = 1;
    EXPECT(SUCCEEDED(c.device->CreateGraphicsPipelineState(&desc, IID_PPV_ARGS(&c.pso.p))));
    return true;
}

const D3D12_HEAP_PROPERTIES default_heap{D3D12_HEAP_TYPE_DEFAULT, D3D12_CPU_PAGE_PROPERTY_UNKNOWN, D3D12_MEMORY_POOL_UNKNOWN, 1, 1};
const D3D12_HEAP_PROPERTIES upload_heap{D3D12_HEAP_TYPE_UPLOAD, D3D12_CPU_PAGE_PROPERTY_UNKNOWN, D3D12_MEMORY_POOL_UNKNOWN, 1, 1};
const D3D12_HEAP_PROPERTIES readback_heap{D3D12_HEAP_TYPE_READBACK, D3D12_CPU_PAGE_PROPERTY_UNKNOWN, D3D12_MEMORY_POOL_UNKNOWN, 1, 1};

// Runs `list` on the queue and waits; false if the wait times out or the device was removed (`why` says which).
bool execute(Context& c, ID3D12GraphicsCommandList* list, const char*& why) {
    Ref<ID3D12Fence> fence;
    if (FAILED(c.device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence.p)))) return why = "fence", false;
    ID3D12CommandList* lists[] = {list};
    c.queue->ExecuteCommandLists(1, lists);
    if (FAILED(c.queue->Signal(fence.p, 1))) return why = "signal", false;
    Event event;
    if (!event.handle || FAILED(fence->SetEventOnCompletion(1, event.handle))) return why = "event", false;
    if (WaitForSingleObject(event.handle, 10000) != WAIT_OBJECT_0) return why = "timeout", false;
    if (FAILED(c.device->GetDeviceRemovedReason())) return why = "device removed", false;
    return true;
}

// Draws `cs` and reads the probe pixels back. `why` names the failure when it returns false.
bool run_case(Context& c, const Case& cs, Probes& out, const char*& why) {
    why = "setup";
    ID3D12Device* device = c.device.p;
    constexpr UINT index_offset = 256;
    Ref<ID3D12Resource> upload, vertices, indices, target, readback;
    const auto upload_desc = buffer_desc(1024), small_desc = buffer_desc(256);
    const auto readback_desc = buffer_desc(target_extent * 256);
    EXPECT(SUCCEEDED(device->CreateCommittedResource(&upload_heap, D3D12_HEAP_FLAG_NONE, &upload_desc,
        D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&upload.p))));
    EXPECT(SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &small_desc,
        D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&vertices.p))));
    EXPECT(SUCCEEDED(device->CreateCommittedResource(&readback_heap, D3D12_HEAP_FLAG_NONE, &readback_desc,
        D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&readback.p))));
    const bool has_index_resource = !cs.indices.empty();
    if (has_index_resource)
        EXPECT(SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &small_desc,
            D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&indices.p))));
    D3D12_RESOURCE_DESC target_desc{};
    target_desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    target_desc.Width = target_desc.Height = target_extent;
    target_desc.DepthOrArraySize = target_desc.MipLevels = 1;
    target_desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    target_desc.SampleDesc.Count = 1;
    target_desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
    EXPECT(SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &target_desc,
        D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&target.p))));

    std::uint8_t* mapped = nullptr;
    EXPECT(SUCCEEDED(upload->Map(0, nullptr, reinterpret_cast<void**>(&mapped))));
    std::memset(mapped, 0, 1024);
    std::memcpy(mapped, vertex_sets[cs.set], sizeof(vertex_sets[cs.set]));
    const bool wide = cs.index_format == DXGI_FORMAT_R32_UINT;
    for (std::size_t i = 0; i < cs.indices.size(); ++i) {
        if (wide) std::memcpy(mapped + index_offset + 4 * i, &cs.indices[i], 4);
        else {
            const auto narrow = static_cast<std::uint16_t>(cs.indices[i]);
            std::memcpy(mapped + index_offset + 2 * i, &narrow, 2);
        }
    }
    upload->Unmap(0, nullptr);

    Ref<ID3D12DescriptorHeap> rtv_heap;
    const D3D12_DESCRIPTOR_HEAP_DESC heap_desc{D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 1, D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0};
    EXPECT(SUCCEEDED(device->CreateDescriptorHeap(&heap_desc, IID_PPV_ARGS(&rtv_heap.p))));
    const D3D12_CPU_DESCRIPTOR_HANDLE rtv = rtv_heap->GetCPUDescriptorHandleForHeapStart();
    device->CreateRenderTargetView(target.p, nullptr, rtv);

    Ref<ID3D12CommandAllocator> allocator;
    Ref<ID3D12GraphicsCommandList> list;
    EXPECT(SUCCEEDED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator.p))));
    EXPECT(SUCCEEDED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.p, c.pso.p,
                                               IID_PPV_ARGS(&list.p))));
    list->CopyBufferRegion(vertices.p, 0, upload.p, 0, 256);
    if (has_index_resource) list->CopyBufferRegion(indices.p, 0, upload.p, index_offset, 256);
    D3D12_RESOURCE_BARRIER barriers[3] = {
        transition(vertices.p, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER),
        transition(target.p, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_RENDER_TARGET)};
    UINT barrier_count = 2;
    if (has_index_resource)
        barriers[barrier_count++] = transition(indices.p, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_INDEX_BUFFER);
    list->ResourceBarrier(barrier_count, barriers);
    const FLOAT clear[4] = {0, 0, 1, 1};
    list->ClearRenderTargetView(rtv, clear, 0, nullptr);
    list->SetGraphicsRootSignature(c.root.p);
    list->SetPipelineState(c.pso.p);
    list->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    const D3D12_VERTEX_BUFFER_VIEW vertex_view{vertices->GetGPUVirtualAddress(), UINT(cs.vertex_view * sizeof(Vertex) + cs.vertex_extra_bytes),
                                               sizeof(Vertex)};
    list->IASetVertexBuffers(0, 1, &vertex_view);
    if (cs.index_format == no_index_buffer) {
        list->IASetIndexBuffer(nullptr);
    } else {
        const D3D12_INDEX_BUFFER_VIEW index_view{indices->GetGPUVirtualAddress(),
                                                 cs.index_view * (wide ? 4u : 2u), cs.index_format};
        list->IASetIndexBuffer(&index_view);
    }
    const D3D12_VIEWPORT viewport{0, 0, float(target_extent), float(target_extent), 0, 1};
    const D3D12_RECT scissor{0, 0, LONG(target_extent), LONG(target_extent)};
    list->RSSetViewports(1, &viewport);
    list->RSSetScissorRects(1, &scissor);
    list->OMSetRenderTargets(1, &rtv, FALSE, nullptr);
    if (cs.indexed) list->DrawIndexedInstanced(cs.draw_count, 1, 0, 0, 0);
    else list->DrawInstanced(cs.draw_count, 1, 0, 0);
    const auto to_copy = transition(target.p, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_COPY_SOURCE);
    list->ResourceBarrier(1, &to_copy);
    D3D12_TEXTURE_COPY_LOCATION source{}, destination{};
    source.pResource = target.p;
    source.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    destination.pResource = readback.p;
    destination.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    destination.PlacedFootprint.Footprint = {DXGI_FORMAT_R8G8B8A8_UNORM, target_extent, target_extent, 1,
                                             target_extent * 4};
    list->CopyTextureRegion(&destination, 0, 0, 0, &source, nullptr);
    EXPECT(SUCCEEDED(list->Close()));
    if (!execute(c, list.p, why)) return false;

    const std::uint8_t* pixels = nullptr;
    EXPECT(SUCCEEDED(readback->Map(0, nullptr, reinterpret_cast<void**>(const_cast<std::uint8_t**>(&pixels)))));
    for (UINT i = 0; i < probe_count; ++i) {
        const auto& xy = probe_xy[cs.probe_set >= 0 ? cs.probe_set : cs.set][i];
        const std::uint8_t* p = pixels + xy[1] * target_extent * 4 + xy[0] * 4;
        out[i] = {p[0], p[1], p[2], p[3]};
    }
    const D3D12_RANGE none{};
    readback->Unmap(0, &none);
    return true;
}

// Case C's decoy: a default-heap buffer filled with R16 indices {0, 4, 5, 6}, so that indices 32769..32771 of an
// unbounded read past the zero page are the green triangle's.
bool make_decoy(ID3D12Device* device, Ref<ID3D12Resource>& decoy, std::uint64_t& va) {
    Ref<ID3D12Resource> upload;
    const auto desc = buffer_desc(256), upload_desc = buffer_desc(256);
    EXPECT(SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &desc,
        D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&decoy.p))));
    va = decoy->GetGPUVirtualAddress();  // the arena's first allocation lands right behind the page, unless not
    EXPECT(SUCCEEDED(device->CreateCommittedResource(&upload_heap, D3D12_HEAP_FLAG_NONE, &upload_desc,
        D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&upload.p))));
    std::uint8_t* mapped = nullptr;
    EXPECT(SUCCEEDED(upload->Map(0, nullptr, reinterpret_cast<void**>(&mapped))));
    std::memset(mapped, 0, 256);
    const std::uint16_t data[4] = {0, 4, 5, 6};
    std::memcpy(mapped, data, sizeof(data));
    upload->Unmap(0, nullptr);
    D3D12_COMMAND_QUEUE_DESC queue_desc{};
    Context c;
    c.device.p = device;
    device->AddRef();
    EXPECT(SUCCEEDED(device->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&c.queue.p))));
    Ref<ID3D12CommandAllocator> allocator;
    Ref<ID3D12GraphicsCommandList> list;
    EXPECT(SUCCEEDED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator.p))));
    EXPECT(SUCCEEDED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.p, nullptr,
                                               IID_PPV_ARGS(&list.p))));
    list->CopyBufferRegion(decoy.p, 0, upload.p, 0, 256);
    EXPECT(SUCCEEDED(list->Close()));
    const char* why = "";
    if (!execute(c, list.p, why)) std::fprintf(stderr, "decoy upload: %s\n", why);
    return std::strcmp(why, "") == 0;
}

enum class layer_status { done, skip, failed };

// Runs every case on a fresh standalone layer device. With `need_exact` the device must bound input-assembler reads
// (else skip); `decoy_behind_page` tells whether case C's decoy landed right behind the zero page.
layer_status run_layer(bool need_exact, const std::vector<std::uint8_t>& vs, const std::vector<std::uint8_t>& ps,
                       std::vector<Probes>& out, bool& decoy_behind_page) {
    Context layer;
    if (FAILED(rt::d3d12::create_vulkan_device_standalone(IID_PPV_ARGS(&layer.device.p)))) {
        std::puts("SKIP: no Vulkan device the layer can use");
        return layer_status::skip;
    }
    const bool exact = rt::d3d12::device_index_bounds_exact(layer.device.p);
    if (need_exact && !exact) {
        std::puts("SKIP: the Vulkan device lacks VK_KHR_maintenance5 or robustBufferAccess2 (or robustBufferAccess with "
                  "update-after-bind buffers), so reads past a view stay unbounded (docs/d3d12_vulkan.md)");
        return layer_status::skip;
    }
    if (!need_exact && exact) {
        std::fprintf(stderr, "LEONIDA_NO_IA_BOUNDS did not switch the bounds off\n");
        return layer_status::failed;
    }
    Ref<ID3D12Resource> decoy;
    std::uint64_t decoy_va = 0;
    if (!make_decoy(layer.device.p, decoy, decoy_va)) return layer_status::failed;
    decoy_behind_page = decoy_va == arena_reserved;
    if (!decoy_behind_page)
        std::printf("the decoy is at 0x%llx, not right behind the zero page: case C is information only\n",
                    static_cast<unsigned long long>(decoy_va));
    if (!make_context(layer, vs, ps)) return layer_status::failed;
    out.assign(cases().size(), {});
    for (std::size_t i = 0; i < cases().size(); ++i) {
        const char* why = "";
        if (!run_case(layer, cases()[i], out[i], why)) {
            std::fprintf(stderr, "%s: the layer failed (%s)\n", cases()[i].name, why);
            return layer_status::failed;
        }
    }
    return layer_status::done;
}

}  // namespace

int main() {
    const bool wine = hostwine::host_is_wine();
    std::vector<std::uint8_t> host_vs, host_ps;
    if (!wine && (!compile_hlsl(vertex_hlsl, "vs_5_0", host_vs) || !compile_hlsl(pixel_hlsl, "ps_5_0", host_ps))) return 1;

    // The layer's shaders: synthetic FSSM programs found in a translator-style index by their hashes.
    const auto vs_program = fssm_program({0x56530010, 0xbf810000});
    const auto ps_program = fssm_program({0x50530010, 0xbf810000});
    const auto vs_container = shader_container(vs_program), ps_container = shader_container(ps_program);
    TempFolder library;
    library.path = std::filesystem::temp_directory_path() / ("leonida_ia_bounds_" + std::to_string(GetCurrentProcessId()));
    std::filesystem::create_directories(library.path);
    std::filesystem::copy_file(LEONIDA_TEST_VS_SPV, library.path / "vs_ia_bounds.spv",
                               std::filesystem::copy_options::overwrite_existing);
    std::filesystem::copy_file(LEONIDA_TEST_PS_SPV, library.path / "ps_ia_bounds.spv",
                               std::filesystem::copy_options::overwrite_existing);
    {
        std::ofstream index(library.path / "index.json", std::ios::binary);
        index << "{\"vs_ia_bounds\": {\"stage\": \"vs\", \"fssm_sha256\": \"" << leonida::sha256_hex(vs_program)
              << "\", \"bindings\": [], \"push_constant_bytes\": 144, \"input_signature\": ["
                 "{\"semantic\": \"COLOR\", \"semantic_index\": 0, \"register\": 0, \"mask\": 15, \"rw_mask\": 15, "
                 "\"system_value\": 0, \"component_type\": \"float\"}, "
                 "{\"semantic\": \"POSITION\", \"semantic_index\": 0, \"register\": 1, \"mask\": 3, \"rw_mask\": 3, "
                 "\"system_value\": 0, \"component_type\": \"float\"}]},\n"
                 " \"ps_ia_bounds\": {\"stage\": \"ps\", \"fssm_sha256\": \"" << leonida::sha256_hex(ps_program)
              << "\", \"bindings\": [], \"push_constant_bytes\": 144, \"outputs\": [0]}}";
        if (!index) {
            std::fprintf(stderr, "cannot write the shader index\n");
            return 1;
        }
    }
    _putenv_s("LEONIDA_SHADERS", library.path.string().c_str());

    // The layer, on the standalone device. No Vulkan device, or one without what the layer needs: skip.
    std::vector<Probes> layer_result, fallback_result;
    bool decoy_behind_page = false;
    switch (run_layer(true, vs_container, ps_container, layer_result, decoy_behind_page)) {
    case layer_status::skip: return 77;
    case layer_status::failed: return 1;
    case layer_status::done: break;
    }
    const auto& all = cases();
    unsigned failures = 0;

    // A second layer device with the guarantees forced off (LEONIDA_NO_IA_BOUNDS=1): the old behaviour, which the cases
    // that read past a view do not assert. Cases that stay inside their views must still draw as the host does.
    bool have_fallback = false;
    {
        _putenv_s("LEONIDA_NO_IA_BOUNDS", "1");
        bool ignored = false;
        const auto status = run_layer(false, vs_container, ps_container, fallback_result, ignored);
        _putenv_s("LEONIDA_NO_IA_BOUNDS", "");
        if (status == layer_status::failed) return 1;
        have_fallback = status == layer_status::done;
    }

    // The host: Windows D3D12 on the default adapter and on WARP.
    struct Host {
        const char* name;
        bool ran = false;
        std::vector<Probes> result;
        std::vector<const char*> failure;
    };
    Host hosts[2] = {{"default adapter", false, {}, {}}, {"WARP", false, {}, {}}};
    if (!wine) {
        Ref<IDXGIFactory4> factory;
        if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory.p)))) {
            std::fprintf(stderr, "no DXGI factory\n");
            return 1;
        }
        for (int h = 0; h < 2; ++h) {
            Ref<IDXGIAdapter> adapter;
            Context host;
            if (h == 1 && FAILED(factory->EnumWarpAdapter(IID_PPV_ARGS(&adapter.p)))) continue;
            if (FAILED(D3D12CreateDevice(adapter.p, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&host.device.p)))) continue;
            if (!make_context(host, host_vs, host_ps)) return 1;
            hosts[h].ran = true;
            hosts[h].result.resize(all.size());
            hosts[h].failure.assign(all.size(), nullptr);
            for (std::size_t i = 0; i < all.size(); ++i) {
                const char* why = "";
                if (!run_case(host, all[i], hosts[h].result[i], why)) {
                    hosts[h].failure[i] = why;
                    // A removed or hung device cannot run the other cases.
                    if (std::strcmp(why, "device removed") == 0 || std::strcmp(why, "timeout") == 0) {
                        for (std::size_t rest = i + 1; rest < all.size(); ++rest) hosts[h].failure[rest] = "not run";
                        break;
                    }
                }
            }
        }
    }

    // The table: what each device drew at the probes.
    std::printf("%-46s %-18s %s\n", "case", "device", "probe pixels RGBA");
    for (std::size_t i = 0; i < all.size(); ++i) {
        for (const Host& h : hosts)
            if (h.ran)
                std::printf("%-46s %-18s %s\n", all[i].name, h.name,
                            h.failure[i] ? h.failure[i] : text(h.result[i]).c_str());
        std::printf("%-46s %-18s %s\n", all[i].name, "layer", text(layer_result[i]).c_str());
        if (have_fallback)
            std::printf("%-46s %-18s %s\n", all[i].name, "layer, no bounds", text(fallback_result[i]).c_str());
    }

    unsigned compared = 0, informational = 0;
    for (std::size_t i = 0; i < all.size(); ++i) {
        const Case& cs = all[i];
        bool have_host = false, hosts_agree = true;
        const Probes* reference = nullptr;
        for (const Host& h : hosts) {
            if (!h.ran || h.failure[i]) continue;
            if (!reference) reference = &h.result[i];
            else if (!same(*reference, h.result[i], cs.tolerance)) hosts_agree = false;
            have_host = true;
        }
        const bool defined = cs.defined && !(cs.decoy && !decoy_behind_page);
        if (have_fallback && cs.in_range && !same(fallback_result[i], cs.expected, cs.tolerance)) {
            std::fprintf(stderr, "%s: without the bounds the layer drew %sbut the in-range outcome is %s\n", cs.name,
                         text(fallback_result[i]).c_str(), text(cs.expected).c_str());
            ++failures;
        }
        if (wine || !defined) {
            // Layer against the measured table (the only check under Wine). Undefined cases are information.
            if (defined && !same(layer_result[i], cs.expected, cs.tolerance)) {
                std::fprintf(stderr, "%s: the layer drew %sbut the measured host outcome is %s\n", cs.name,
                             text(layer_result[i]).c_str(), text(cs.expected).c_str());
                ++failures;
            } else if (!defined) {
                std::printf("info: %s: no pass/fail condition (%s; outcomes are above)\n", cs.name,
                            cs.defined ? "the decoy is not right behind the zero page, so the read is not placed"
                                       : "the host's per-element outcome is not claimed");
                ++informational;
            } else {
                ++compared;
            }
            continue;
        }
        if (!have_host || !hosts_agree) {
            std::printf("info: %s: the default adapter and WARP %s, not a pass/fail condition\n", cs.name,
                        have_host ? "disagree" : "did not run it");
            ++informational;
            continue;
        }
        if (!same(layer_result[i], *reference, cs.tolerance)) {
            std::fprintf(stderr, "%s: the layer drew %sbut D3D12 drew %s\n", cs.name,
                         text(layer_result[i]).c_str(), text(*reference).c_str());
            ++failures;
            continue;
        }
        if (defined && !same(*reference, cs.expected, cs.tolerance))
            std::printf("note: %s: the host drew %sinstead of the recorded %s\n", cs.name, text(*reference).c_str(),
                        text(cs.expected).c_str());
        ++compared;
    }
    if (!wine && !hosts[0].ran && !hosts[1].ran) {
        std::fprintf(stderr, "no D3D12 reference device (default adapter or WARP)\n");
        return 1;
    }
    if (failures) {
        std::fprintf(stderr, "d3d12_ia_bounds_test: %u case(s) differ\n", failures);
        return 1;
    }
    std::printf("input-assembler bounds: %u cases match%s, %u informational\n", compared,
                wine ? " the recorded host outcomes" : " the host", informational);
    return 0;
}
