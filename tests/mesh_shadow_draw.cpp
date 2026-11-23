// Two-pass shadow draw with translated mesh and pixel programs.
// Pass 1 writes light-space depth into a depth texture; its color writes are masked off. Pass 2 draws a receiver
// whose pixel program samples that texture with a comparison sampler, then a lit caster, both depth tested.
// The case file holds every draw's inputs, the expected mesh traces, the expected color image and the
// expected light depth, all from an independent CPU reference.
#include "desktop_gpu.h"
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <vector>
using Microsoft::WRL::ComPtr;

namespace {
std::vector<char> read_file(const char* path)
{
    std::ifstream stream(path, std::ios::binary);
    if (!stream) throw std::runtime_error("Cannot open shadow draw input");
    return {std::istreambuf_iterator<char>(stream), {}};
}

template<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE Kind, typename T>
struct alignas(void*) Field {
    D3D12_PIPELINE_STATE_SUBOBJECT_TYPE kind = Kind;
    T value{};
};

struct ShadowPipeline {
    Field<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_ROOT_SIGNATURE, ID3D12RootSignature*> root;
    Field<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_MS, D3D12_SHADER_BYTECODE> ms;
    Field<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PS, D3D12_SHADER_BYTECODE> ps;
    Field<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RENDER_TARGET_FORMATS, D3D12_RT_FORMAT_ARRAY> formats;
    Field<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_BLEND, D3D12_BLEND_DESC> blend;
    Field<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RASTERIZER, D3D12_RASTERIZER_DESC> raster;
    Field<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL, D3D12_DEPTH_STENCIL_DESC> depth;
    Field<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL_FORMAT, DXGI_FORMAT> depth_format;
    Field<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PRIMITIVE_TOPOLOGY, D3D12_PRIMITIVE_TOPOLOGY_TYPE> topology;
};

ComPtr<ID3D12PipelineState> shadow_pipeline(ID3D12Device2* device, ID3D12RootSignature* root,
                                            const std::vector<char>& mesh, const std::vector<char>& pixel,
                                            bool write_color)
{
    ShadowPipeline stream;
    stream.root.value = root;
    stream.ms.value = {mesh.data(), mesh.size()};
    stream.ps.value = {pixel.data(), pixel.size()};
    stream.formats.value.NumRenderTargets = 1;
    stream.formats.value.RTFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
    // The light pass keeps the caster's pixel program and target but writes depth only. On WARP
    // (10.0.26100.9818) a mesh pipeline without a pixel shader removed the device in this fixture, with or
    // without a bound render target and with no debug-layer message; the RTX 3070 drew it correctly.
    auto& target = stream.blend.value.RenderTarget[0];
    target.SrcBlend = target.SrcBlendAlpha = D3D12_BLEND_ONE;
    target.DestBlend = target.DestBlendAlpha = D3D12_BLEND_ZERO;
    target.BlendOp = target.BlendOpAlpha = D3D12_BLEND_OP_ADD;
    target.LogicOp = D3D12_LOGIC_OP_NOOP;
    target.RenderTargetWriteMask = write_color ? D3D12_COLOR_WRITE_ENABLE_ALL : 0;
    stream.raster.value.FillMode = D3D12_FILL_MODE_SOLID;
    stream.raster.value.CullMode = D3D12_CULL_MODE_NONE;
    stream.raster.value.DepthClipEnable = TRUE;
    stream.depth.value.DepthEnable = TRUE;
    stream.depth.value.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ALL;
    stream.depth.value.DepthFunc = D3D12_COMPARISON_FUNC_LESS;
    stream.depth_format.value = DXGI_FORMAT_D32_FLOAT;
    stream.topology.value = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    D3D12_PIPELINE_STATE_STREAM_DESC desc{sizeof(stream), &stream};
    ComPtr<ID3D12PipelineState> pipeline;
    check(device->CreatePipelineState(&desc, IID_PPV_ARGS(&pipeline)), "Create shadow pipeline");
    return pipeline;
}

void copy_to_readback(ID3D12GraphicsCommandList* list, ID3D12Resource* texture, ID3D12Resource* readback,
                      const D3D12_PLACED_SUBRESOURCE_FOOTPRINT& footprint)
{
    D3D12_TEXTURE_COPY_LOCATION src{}, dst{};
    src.pResource = texture;
    dst.pResource = readback;
    dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    dst.PlacedFootprint = footprint;
    list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
}

unsigned count_mismatches(const char* rows, const D3D12_PLACED_SUBRESOURCE_FOOTPRINT& footprint,
                          const char* expected, UINT side)
{
    unsigned mismatches = 0;
    for (UINT y = 0; y < side; ++y)
        for (UINT x = 0; x < side; ++x)
            mismatches += std::memcmp(rows + footprint.Offset + y * footprint.Footprint.RowPitch + 4 * x,
                                      expected + (y * side + x) * 4, 4) != 0;
    return mismatches;
}
}

bool check_mesh_shadow_draw(ID3D12Device* device, const char* mesh_path, const char* receiver_path,
                            const char* caster_path, const char* case_path, bool stale,
                            std::vector<unsigned char>* frames)
{
    D3D12_FEATURE_DATA_D3D12_OPTIONS7 options{};
    check(check_feature(device, D3D12_FEATURE_D3D12_OPTIONS7, &options, sizeof(options)), "Query mesh tier");
    if (options.MeshShaderTier < D3D12_MESH_SHADER_TIER_1) {
        std::puts("mesh_shader_tier=unsupported");
        return false;
    }
    constexpr UINT cb_bytes = 512;
    constexpr UINT draw_bytes = 3 * cb_bytes;  // constant buffers b0 and b1, then the vertex buffer t0
    constexpr UINT trace_stride = 80;
    const auto input = read_file(case_path);
    std::array<uint32_t, 8> header{};
    if (input.size() < sizeof(header)) throw std::runtime_error("Invalid shadow case header");
    std::memcpy(header.data(), input.data(), sizeof(header));
    const UINT side = header[0], previous = header[1], current = header[2];
    const auto filter = D3D12_FILTER(header[3]);
    const auto func = D3D12_COMPARISON_FUNC(header[4]);
    // Light draws for the previous and the current shadow map, then the receiver and the lit caster.
    const UINT draws = previous + current + (frames ? 4 : 2);
    if (side == 0 || side > 256 || previous == 0 || current == 0 || draws > 16)
        throw std::runtime_error("Invalid shadow case header");
    const UINT image_bytes = side * side * 4;
    const size_t inputs_at = sizeof(header);
    const size_t traces_at = inputs_at + size_t(draws) * draw_bytes;
    const size_t image_at = traces_at + size_t(draws) * trace_stride;
    const size_t depth_at = image_at + image_bytes;
    if (input.size() != (frames ? image_at : depth_at + image_bytes)) throw std::runtime_error("Shadow case size does not match its header");
    const auto mesh = read_file(mesh_path);
    const auto receiver = read_file(receiver_path);
    const auto caster = read_file(caster_path);

    // b0 and b1 as root CBVs for both stages, the trace UAV and the vertex SRV for the mesh stage, and the
    // shadow map at t0 with a comparison sampler at s0 for the pixel stage.
    D3D12_ROOT_PARAMETER params[5]{};
    for (UINT i = 0; i < 2; ++i) {
        params[i].ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
        params[i].Descriptor.ShaderRegister = i;
    }
    params[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
    params[2].ShaderVisibility = D3D12_SHADER_VISIBILITY_MESH;
    params[3].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
    params[3].ShaderVisibility = D3D12_SHADER_VISIBILITY_MESH;
    D3D12_DESCRIPTOR_RANGE map_range{D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0, 0, 0};
    params[4].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    params[4].DescriptorTable = {1, &map_range};
    params[4].ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
    D3D12_STATIC_SAMPLER_DESC sampler{};
    sampler.Filter = filter;
    sampler.AddressU = sampler.AddressV = sampler.AddressW = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
    sampler.MaxAnisotropy = 1;
    sampler.ComparisonFunc = func;
    sampler.MaxLOD = D3D12_FLOAT32_MAX;
    sampler.ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
    D3D12_ROOT_SIGNATURE_DESC layout{5, params, 1, &sampler, D3D12_ROOT_SIGNATURE_FLAG_NONE};
    ComPtr<ID3DBlob> blob;
    check(D3D12SerializeRootSignature(&layout, D3D_ROOT_SIGNATURE_VERSION_1, &blob, nullptr), "Serialize shadow root");
    ComPtr<ID3D12RootSignature> root;
    check(device->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(), IID_PPV_ARGS(&root)),
          "Shadow root");
    ComPtr<ID3D12Device2> device2;
    check(device->QueryInterface(IID_PPV_ARGS(&device2)), "Query mesh pipeline device");
    const auto light_pass = shadow_pipeline(device2.Get(), root.Get(), mesh, caster, false);
    const auto receiver_pass = shadow_pipeline(device2.Get(), root.Get(), mesh, receiver, true);
    const auto caster_pass = shadow_pipeline(device2.Get(), root.Get(), mesh, caster, true);

    D3D12_RESOURCE_DESC color_desc{};
    color_desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    color_desc.Width = color_desc.Height = side;
    color_desc.DepthOrArraySize = color_desc.MipLevels = color_desc.SampleDesc.Count = 1;
    color_desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    color_desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
    D3D12_CLEAR_VALUE clear_color{color_desc.Format, {0, 0, 0, 1}};
    auto target = resource(device, color_desc, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_RENDER_TARGET,
                           &clear_color);
    // Two shadow maps (previous and current light) and the camera depth. The maps are typeless so the
    // camera pass can read them through an R32_FLOAT view.
    auto depth_desc = color_desc;
    depth_desc.Format = DXGI_FORMAT_R32_TYPELESS;
    depth_desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;
    D3D12_CLEAR_VALUE clear_depth{};
    clear_depth.Format = DXGI_FORMAT_D32_FLOAT;
    clear_depth.DepthStencil.Depth = 1.0f;
    std::array<ComPtr<ID3D12Resource>, 3> depths;
    for (auto& depth : depths)
        depth = resource(device, depth_desc, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_DEPTH_WRITE, &clear_depth);
    D3D12_DESCRIPTOR_HEAP_DESC rtv_desc{D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 1};
    ComPtr<ID3D12DescriptorHeap> rtv_heap, dsv_heap, srv_heap;
    check(device->CreateDescriptorHeap(&rtv_desc, IID_PPV_ARGS(&rtv_heap)), "Create shadow target heap");
    const auto rtv = rtv_heap->GetCPUDescriptorHandleForHeapStart();
    device->CreateRenderTargetView(target.Get(), nullptr, rtv);
    D3D12_DESCRIPTOR_HEAP_DESC dsv_desc{D3D12_DESCRIPTOR_HEAP_TYPE_DSV, UINT(depths.size())};
    check(device->CreateDescriptorHeap(&dsv_desc, IID_PPV_ARGS(&dsv_heap)), "Create shadow depth heap");
    const UINT dsv_step = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_DSV);
    std::array<D3D12_CPU_DESCRIPTOR_HANDLE, 3> dsv{};
    D3D12_DEPTH_STENCIL_VIEW_DESC dsv_view{};
    dsv_view.Format = DXGI_FORMAT_D32_FLOAT;
    dsv_view.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2D;
    for (UINT i = 0; i < depths.size(); ++i) {
        dsv[i] = {dsv_heap->GetCPUDescriptorHandleForHeapStart().ptr + i * dsv_step};
        device->CreateDepthStencilView(depths[i].Get(), &dsv_view, dsv[i]);
    }
    D3D12_DESCRIPTOR_HEAP_DESC srv_desc{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 2,
                                        D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, 0};
    check(device->CreateDescriptorHeap(&srv_desc, IID_PPV_ARGS(&srv_heap)), "Create shadow map heap");
    D3D12_SHADER_RESOURCE_VIEW_DESC map_view{};
    map_view.Format = DXGI_FORMAT_R32_FLOAT;
    map_view.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    map_view.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    map_view.Texture2D.MipLevels = 1;
    const UINT srv_step = device->GetDescriptorHandleIncrementSize(srv_desc.Type);
    for (UINT i = 0; i < 2; ++i) {
        auto handle = srv_heap->GetCPUDescriptorHandleForHeapStart();
        handle.ptr += i * srv_step;
        device->CreateShaderResourceView(depths[i].Get(), &map_view, handle);
    }

    D3D12_PLACED_SUBRESOURCE_FOOTPRINT color_footprint{}, depth_footprint{};
    UINT64 color_size = 0, depth_size = 0;
    device->GetCopyableFootprints(&color_desc, 0, 1, 0, &color_footprint, nullptr, nullptr, &color_size);
    device->GetCopyableFootprints(&depth_desc, 0, 1, 0, &depth_footprint, nullptr, nullptr, &depth_size);
    const auto align_placement = [](UINT64 size) {
        constexpr UINT64 alignment = D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT;
        return (size + alignment - 1) / alignment * alignment;
    };
    const UINT64 color_stride = align_placement(color_size), depth_stride = align_placement(depth_size);
    if (frames) {
        color_size += color_stride;
        depth_size += depth_stride;
    }
    const UINT trace_bytes = draws * trace_stride;
    auto color_readback = make_buffer(device, color_size, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST);
    auto depth_readback = make_buffer(device, depth_size, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST);
    auto trace_readback = make_buffer(device, trace_bytes, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST);
    const UINT upload_bytes = draws * draw_bytes;
    auto upload = make_buffer(device, upload_bytes + trace_bytes, D3D12_HEAP_TYPE_UPLOAD,
                              D3D12_RESOURCE_STATE_GENERIC_READ);
    void* mapped = nullptr;
    D3D12_RANGE empty{};
    check(upload->Map(0, &empty, &mapped), "Map shadow input");
    std::memcpy(mapped, input.data() + inputs_at, upload_bytes);
    std::memset(static_cast<char*>(mapped) + upload_bytes, 0xcd, trace_bytes);
    upload->Unmap(0, nullptr);
    auto trace = make_buffer(device, trace_bytes, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COMMON,
                             D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);

    ComputeCommands commands(device, light_pass.Get(), D3D12_COMMAND_LIST_TYPE_DIRECT);
    auto* list = commands.list.Get();
    ComPtr<ID3D12GraphicsCommandList6> mesh_list;
    check(commands.list.As(&mesh_list), "Query mesh dispatch list");
    list->CopyBufferRegion(trace.Get(), 0, upload.Get(), upload_bytes, trace_bytes);
    transition(list, trace.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    list->SetGraphicsRootSignature(root.Get());
    const D3D12_VIEWPORT viewport{0, 0, float(side), float(side), 0, 1};
    const D3D12_RECT scissor{0, 0, LONG(side), LONG(side)};
    list->RSSetViewports(1, &viewport);
    list->RSSetScissorRects(1, &scissor);
    const auto base = upload->GetGPUVirtualAddress();
    UINT draw = 0;
    auto dispatch = [&] {
        const auto at = base + UINT64(draw) * draw_bytes;
        list->SetGraphicsRootConstantBufferView(0, at);
        list->SetGraphicsRootConstantBufferView(1, at + cb_bytes);
        list->SetGraphicsRootUnorderedAccessView(2, trace->GetGPUVirtualAddress() + UINT64(draw) * trace_stride);
        list->SetGraphicsRootShaderResourceView(3, at + 2 * cb_bytes);
        mesh_list->DispatchMesh(1, 1, 1);
        ++draw;
    };
    for (UINT frame = frames ? 0u : 1u; frame < 2; ++frame) {
        // The standalone fixture renders both light maps before its single camera pass.
        for (UINT map = frames ? frame : 0u; map <= frame; ++map) {
            draw = map ? previous : 0;
            list->SetPipelineState(light_pass.Get());
            list->OMSetRenderTargets(1, &rtv, FALSE, &dsv[map]);
            list->ClearDepthStencilView(dsv[map], D3D12_CLEAR_FLAG_DEPTH, 1.0f, 0, 0, nullptr);
            for (UINT i = 0; i < (map ? current : previous); ++i) dispatch();
            transition(list, depths[map].Get(), D3D12_RESOURCE_STATE_DEPTH_WRITE,
                       D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
        }
        ID3D12DescriptorHeap* heaps[]{srv_heap.Get()};
        list->SetDescriptorHeaps(1, heaps);
        auto bound = srv_heap->GetGPUDescriptorHandleForHeapStart();
        if (frame && !stale) bound.ptr += srv_step;
        list->SetGraphicsRootDescriptorTable(4, bound);
        list->OMSetRenderTargets(1, &rtv, FALSE, &dsv[2]);
        list->ClearRenderTargetView(rtv, clear_color.Color, 0, nullptr);
        list->ClearDepthStencilView(dsv[2], D3D12_CLEAR_FLAG_DEPTH, 1.0f, 0, 0, nullptr);
        draw = previous + current + (frames ? 2 * frame : 0);
        list->SetPipelineState(receiver_pass.Get());
        dispatch();
        list->SetPipelineState(caster_pass.Get());
        dispatch();

        auto color_copy = color_footprint, depth_copy = depth_footprint;
        if (frames && frame) {
            color_copy.Offset += color_stride;
            depth_copy.Offset += depth_stride;
        }
        transition(list, target.Get(), D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_COPY_SOURCE);
        copy_to_readback(list, target.Get(), color_readback.Get(), color_copy);
        transition(list, depths[frame].Get(), D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COPY_SOURCE);
        copy_to_readback(list, depths[frame].Get(), depth_readback.Get(), depth_copy);
        if (frames && !frame) {
            transition(list, target.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_RENDER_TARGET);
            transition(list, depths[frame].Get(), D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
        }
    }
    transition(list, trace.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
    list->CopyBufferRegion(trace_readback.Get(), 0, trace.Get(), 0, trace_bytes);
    commands.submit_and_wait(device);

    check(trace_readback->Map(0, nullptr, &mapped), "Map shadow trace");
    const bool trace_ok = std::memcmp(mapped, input.data() + traces_at, trace_bytes) == 0;
    trace_readback->Unmap(0, &empty);
    if (!trace_ok) throw std::runtime_error("Mesh export trace mismatch");
    if (frames) {
        auto append = [&](ID3D12Resource* readback, D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint, UINT64 stride) {
            check(readback->Map(0, nullptr, &mapped), "Map shadow frames");
            auto* source = static_cast<const unsigned char*>(mapped) + footprint.Offset;
            for (UINT frame = 0; frame < 2; ++frame) {
                for (UINT y = 0; y < side; ++y) {
                    const auto* row = source + frame * stride + y * footprint.Footprint.RowPitch;
                    frames->insert(frames->end(), row, row + side * 4);
                }
            }
            readback->Unmap(0, &empty);
        };
        append(color_readback.Get(), color_footprint, color_stride);
        append(depth_readback.Get(), depth_footprint, depth_stride);
        unsigned releases = 0;
        auto release = [&](ComPtr<ID3D12Resource>& surface) {
            if (surface) {
                surface.Reset();
                ++releases;
            }
        };
        for (auto& depth : depths) release(depth);
        release(target);
        frames->push_back(static_cast<unsigned char>(releases));
        return true;
    }
    std::printf("light_draws=%u+%u camera_draws=2 bound_map=%s\n", previous, current, stale ? "previous" : "current");
    check(color_readback->Map(0, nullptr, &mapped), "Map shadow color");
    const unsigned mismatches = count_mismatches(static_cast<const char*>(mapped), color_footprint,
                                                 input.data() + image_at, side);
    color_readback->Unmap(0, &empty);
    check(depth_readback->Map(0, nullptr, &mapped), "Map shadow depth");
    const unsigned depth_mismatches = count_mismatches(static_cast<const char*>(mapped), depth_footprint,
                                                       input.data() + depth_at, side);
    depth_readback->Unmap(0, &empty);
    std::printf("pixels=%u mismatches=%u\n", side * side, mismatches);
    std::printf("shadow_map_texels=%u shadow_map_mismatches=%u\n", side * side, depth_mismatches);
    if (depth_mismatches) throw std::runtime_error("Shadow map reference mismatch");
    if (mismatches) throw std::runtime_error("Draw reference mismatch");
    return true;
}
