#include "desktop_gpu.h"
#include <array>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <utility>
#include <vector>
using Microsoft::WRL::ComPtr;

namespace {
std::vector<char> read_bytes(const char* path)
{
    std::ifstream stream(path, std::ios::binary);
    if (!stream) throw std::runtime_error("Cannot open draw input");
    return {std::istreambuf_iterator<char>(stream), {}};
}

template<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE Kind, typename T>
struct alignas(void*) Field {
    D3D12_PIPELINE_STATE_SUBOBJECT_TYPE kind = Kind;
    T value{};
};

struct MeshPipeline {
    Field<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_ROOT_SIGNATURE, ID3D12RootSignature*> root;
    Field<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_MS, D3D12_SHADER_BYTECODE> ms;
    Field<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PS, D3D12_SHADER_BYTECODE> ps;
    Field<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RENDER_TARGET_FORMATS, D3D12_RT_FORMAT_ARRAY> formats;
    Field<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RASTERIZER, D3D12_RASTERIZER_DESC> raster;
    Field<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL, D3D12_DEPTH_STENCIL_DESC> depth;
    Field<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PRIMITIVE_TOPOLOGY, D3D12_PRIMITIVE_TOPOLOGY_TYPE> topology;
};
}

bool check_mesh_entry_draw(ID3D12Device* device, const char* first, const char* second, const char* fixture,
                           bool vertices, MeshConstantBinding binding)
{
    D3D12_FEATURE_DATA_D3D12_OPTIONS7 options{};
    check(check_feature(device, D3D12_FEATURE_D3D12_OPTIONS7, &options, sizeof(options)), "Query mesh tier");
    if (options.MeshShaderTier < D3D12_MESH_SHADER_TIER_1) {
        std::puts("mesh_shader_tier=unsupported");
        return false;
    }
    constexpr UINT side = 32;
    constexpr UINT cb_bytes = 512;
    constexpr UINT trace_bytes = 80;
    const auto input = read_bytes(fixture);
    const UINT input_bytes = 2 * cb_bytes + (vertices ? 512 : 0);
    if (input.size() != input_bytes + trace_bytes + side * side * 4)
        throw std::runtime_error("Invalid draw fixture size");
    auto shader = read_bytes(first);
    auto pixel = read_bytes(second);
    D3D12_ROOT_PARAMETER params[4]{};
    D3D12_DESCRIPTOR_RANGE cbv_range{D3D12_DESCRIPTOR_RANGE_TYPE_CBV, 2, 0, 0, 0};
    const bool uses_table = binding == MeshConstantBinding::table || binding == MeshConstantBinding::swapped_table;
    if (uses_table) {
        params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        params[0].DescriptorTable = {1, &cbv_range};
    } else {
        for (UINT i = 0; i < 2; ++i) {
            params[i].ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
            params[i].Descriptor.ShaderRegister = i;
        }
    }
    const UINT trace_slot = uses_table ? 1 : 2;
    const UINT vertex_slot = trace_slot + 1;
    params[trace_slot].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
    D3D12_ROOT_SIGNATURE_DESC layout{};
    params[vertex_slot].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
    layout.NumParameters = vertex_slot + vertices;
    layout.pParameters = params;
    ComPtr<ID3DBlob> blob;
    check(D3D12SerializeRootSignature(&layout, D3D_ROOT_SIGNATURE_VERSION_1, &blob, nullptr), "Serialize draw root");
    ComPtr<ID3D12RootSignature> root;
    check(device->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(), IID_PPV_ARGS(&root)), "Draw root");
    ComPtr<ID3D12PipelineState> pipeline;
    MeshPipeline stream;
    stream.root.value = root.Get();
    stream.ms.value = {shader.data(), shader.size()};
    stream.ps.value = {pixel.data(), pixel.size()};
    stream.formats.value.NumRenderTargets = 1;
    stream.formats.value.RTFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
    stream.raster.value.FillMode = D3D12_FILL_MODE_SOLID;
    stream.raster.value.CullMode = D3D12_CULL_MODE_NONE;
    stream.raster.value.DepthClipEnable = TRUE;
    stream.topology.value = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    D3D12_PIPELINE_STATE_STREAM_DESC desc{sizeof(stream), &stream};
    ComPtr<ID3D12Device2> device2;
    check(device->QueryInterface(IID_PPV_ARGS(&device2)), "Query mesh pipeline device");
    check(device2->CreatePipelineState(&desc, IID_PPV_ARGS(&pipeline)), "Create mesh pipeline");
    D3D12_RESOURCE_DESC texture{};
    texture.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    texture.Width = texture.Height = side;
    texture.DepthOrArraySize = texture.MipLevels = texture.SampleDesc.Count = 1;
    texture.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    texture.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
    constexpr auto draw_state = D3D12_RESOURCE_STATE_RENDER_TARGET;
    D3D12_CLEAR_VALUE clear_value{};
    clear_value.Format = texture.Format;
    clear_value.Color[3] = 1;
    auto target = resource(device, texture, D3D12_HEAP_TYPE_DEFAULT, draw_state, &clear_value);
    D3D12_DESCRIPTOR_HEAP_DESC heap_desc{};
    heap_desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
    heap_desc.NumDescriptors = 1;
    ComPtr<ID3D12DescriptorHeap> heap;
    check(device->CreateDescriptorHeap(&heap_desc, IID_PPV_ARGS(&heap)), "Create target heap");
    auto handle = heap->GetCPUDescriptorHandleForHeapStart();
    device->CreateRenderTargetView(target.Get(), nullptr, handle);
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
    UINT64 size = 0;
    device->GetCopyableFootprints(&texture, 0, 1, 0, &footprint, nullptr, nullptr, &size);
    auto readback = make_buffer(device, size + trace_bytes, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST);
    auto upload = make_buffer(device, input_bytes + trace_bytes, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ);
    void* mapped = nullptr;
    D3D12_RANGE empty{};
    check(upload->Map(0, &empty, &mapped), "Map draw input");
    std::memcpy(mapped, input.data(), input_bytes);
    std::memset(static_cast<char*>(mapped) + input_bytes, 0xcd, trace_bytes);
    upload->Unmap(0, nullptr);
    ComputeCommands commands(device, pipeline.Get(), D3D12_COMMAND_LIST_TYPE_DIRECT);
    auto* list = commands.list.Get();
    list->SetGraphicsRootSignature(root.Get());
    const auto base = upload->GetGPUVirtualAddress();
    std::array<ComPtr<ID3D12Resource>, 2> constants;
    ComPtr<ID3D12DescriptorHeap> cbv_heap;
    ComPtr<ID3D12Resource> buffer_b;
    if (uses_table) {
        D3D12_DESCRIPTOR_HEAP_DESC cbv_heap_desc{};
        cbv_heap_desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
        cbv_heap_desc.NumDescriptors = static_cast<UINT>(constants.size());
        cbv_heap_desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
        check(device->CreateDescriptorHeap(&cbv_heap_desc, IID_PPV_ARGS(&cbv_heap)), "Create mesh CBV heap");
        const UINT increment = device->GetDescriptorHandleIncrementSize(cbv_heap_desc.Type);
        const auto first_slot = cbv_heap->GetCPUDescriptorHandleForHeapStart();
        for (UINT i = 0; i < constants.size(); ++i) {
            constants[i] = make_buffer(device, cb_bytes, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ);
            check(constants[i]->Map(0, &empty, &mapped), "Map mesh CBV");
            std::memcpy(mapped, input.data() + i * cb_bytes, cb_bytes);
            constants[i]->Unmap(0, nullptr);
            const UINT slot = binding == MeshConstantBinding::swapped_table ? 1 - i : i;
            const D3D12_CONSTANT_BUFFER_VIEW_DESC view{constants[i]->GetGPUVirtualAddress(), cb_bytes};
            device->CreateConstantBufferView(&view, {first_slot.ptr + slot * increment});
        }
        ID3D12DescriptorHeap* heaps[]{cbv_heap.Get()};
        list->SetDescriptorHeaps(1, heaps);
        list->SetGraphicsRootDescriptorTable(0, cbv_heap->GetGPUDescriptorHandleForHeapStart());
    } else {
        D3D12_GPU_VIRTUAL_ADDRESS cbv[2]{base, base + cb_bytes};
        if (binding != MeshConstantBinding::root) {
            // Keep A and B in distinct resources with identical contents across binding orders.
            buffer_b = make_buffer(device, cb_bytes, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ);
            check(buffer_b->Map(0, &empty, &mapped), "Map constant buffer B");
            std::memcpy(mapped, input.data() + cb_bytes, cb_bytes);
            buffer_b->Unmap(0, nullptr);
            cbv[1] = buffer_b->GetGPUVirtualAddress();
            if (binding == MeshConstantBinding::swapped_root_pair) std::swap(cbv[0], cbv[1]);
        }
        list->SetGraphicsRootConstantBufferView(0, cbv[0]);
        list->SetGraphicsRootConstantBufferView(1, cbv[1]);
    }
    if (vertices) list->SetGraphicsRootShaderResourceView(vertex_slot, base + 2 * cb_bytes);
    const D3D12_VIEWPORT viewport{0, 0, side, side, 0, 1};
    const D3D12_RECT scissor{0, 0, side, side};
    list->RSSetViewports(1, &viewport);
    list->RSSetScissorRects(1, &scissor);
    auto trace = make_buffer(device, trace_bytes, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COMMON,
                             D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    list->CopyBufferRegion(trace.Get(), 0, upload.Get(), input_bytes, trace_bytes);
    transition(list, trace.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    list->SetGraphicsRootUnorderedAccessView(trace_slot, trace->GetGPUVirtualAddress());
    list->OMSetRenderTargets(1, &handle, FALSE, nullptr);
    const float clear[4]{0, 0, 0, 1};
    list->ClearRenderTargetView(handle, clear, 0, nullptr);
    ComPtr<ID3D12GraphicsCommandList6> mesh_list;
    check(commands.list.As(&mesh_list), "Query mesh dispatch list");
    mesh_list->DispatchMesh(1, 1, 1);
    transition(list, trace.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
    list->CopyBufferRegion(readback.Get(), size, trace.Get(), 0, trace_bytes);
    transition(list, target.Get(), draw_state, D3D12_RESOURCE_STATE_COPY_SOURCE);
    D3D12_TEXTURE_COPY_LOCATION src{}, dst{};
    src.pResource = target.Get();
    dst.pResource = readback.Get();
    dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    dst.PlacedFootprint = footprint;
    list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
    commands.submit_and_wait(device);
    check(readback->Map(0, nullptr, &mapped), "Map draw result");
    const char* rows = static_cast<const char*>(mapped);
    const char* reference = input.data() + input_bytes;
    const bool trace_ok = std::memcmp(rows + size, reference, trace_bytes) == 0;
    reference += trace_bytes;
    unsigned mismatches = 0;
    for (UINT y = 0; y < side; ++y)
        for (UINT x = 0; x < side; ++x)
            mismatches += std::memcmp(rows + footprint.Offset + y * footprint.Footprint.RowPitch + 4*x,
                                      reference + (y * side + x) * 4, 4) != 0;
    readback->Unmap(0, &empty);
    std::printf("pixels=1024 mismatches=%u\n", mismatches);
    if (!trace_ok) throw std::runtime_error("Mesh export trace mismatch");
    if (mismatches) throw std::runtime_error("Draw reference mismatch");
    return true;
}
