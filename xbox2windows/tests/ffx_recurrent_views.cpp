#include "ffx_recurrent.h"
#include "bvh_fixture.h"
#include "SampleMip.h"
#include "WriteRecurrent.h"

namespace {
using bvh_fixture::Word;
constexpr auto heap_type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;

// A fresh context assigns slot 1 in each of the three distinct heaps.
D3D12_CPU_DESCRIPTOR_HANDLE slot_one(ID3D12Device* device, ID3D12DescriptorHeap* heap)
{
    auto handle = heap->GetCPUDescriptorHandleForHeapStart();
    handle.ptr += device->GetDescriptorHandleIncrementSize(heap_type);
    return handle;
}

Word expected_pixel(UINT x, UINT y, UINT seed)
{
    return ((x + seed) & 255u) | (((y + 3 * seed) & 255u) << 8) |
           ((((x ^ y) + seed) & 255u) << 16) | (((193 + seed) & 255u) << 24);
}
}

void RecurrentBinding::create_srv(uintptr_t self, uintptr_t texture, uintptr_t description, uintptr_t dest)
{
    D3D12_SHADER_RESOURCE_VIEW_DESC desc{};
    std::memcpy(&desc, reinterpret_cast<const void*>(description), sizeof desc);
    if (self != reinterpret_cast<uintptr_t>(device) || texture != reinterpret_cast<uintptr_t>(resource.Get()) ||
        srv.ptr || dest != slot_one(device, probe.heaps[0].Get()).ptr ||
        desc.Format != DXGI_FORMAT_R8G8B8A8_UNORM || desc.ViewDimension != D3D12_SRV_DIMENSION_TEXTURE2D ||
        desc.Shader4ComponentMapping != D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING || desc.Texture2D.MostDetailedMip ||
        desc.Texture2D.MipLevels != 1 || desc.Texture2D.PlaneSlice || desc.Texture2D.ResourceMinLODClamp != 0)
        throw std::runtime_error("Unexpected recurrent SRV description, resource or destination");
    srv.ptr = dest;
    device->CreateShaderResourceView(resource.Get(), &desc, srv);
}

void RecurrentBinding::create_uav(uintptr_t self, uintptr_t texture, uintptr_t counter, uintptr_t description, uintptr_t dest)
{
    D3D12_UNORDERED_ACCESS_VIEW_DESC desc{};
    std::memcpy(&desc, reinterpret_cast<const void*>(description), sizeof desc);
    const bool mirror = visible_uav.ptr != 0;
    if (self != reinterpret_cast<uintptr_t>(device) || texture != reinterpret_cast<uintptr_t>(resource.Get()) ||
        counter || cpu_uav.ptr || dest != slot_one(device, probe.heaps[mirror ? 1 : 2].Get()).ptr ||
        desc.Format != DXGI_FORMAT_R8G8B8A8_UNORM || desc.ViewDimension != D3D12_UAV_DIMENSION_TEXTURE2D ||
        desc.Texture2D.MipSlice || desc.Texture2D.PlaneSlice)
        throw std::runtime_error("Unexpected recurrent UAV description, resource or destination");
    auto& handle = mirror ? cpu_uav : visible_uav;
    handle.ptr = dest;
    device->CreateUnorderedAccessView(resource.Get(), nullptr, &desc, handle);
}

void check_recurrent_views(const RecurrentBinding& binding)
{
    if (!binding.srv.ptr || !binding.visible_uav.ptr || !binding.cpu_uav.ptr)
        throw std::runtime_error("Missing recurrent view");
    auto* device = binding.device;
    auto* texture = binding.resource.Get();
    const auto desc = texture->GetDesc();
    const UINT width = static_cast<UINT>(desc.Width), height = desc.Height;
    const UINT64 bytes = UINT64(width) * height * sizeof(Word);
    const UINT stride = device->GetDescriptorHandleIncrementSize(heap_type);
    D3D12_DESCRIPTOR_HEAP_DESC heap_desc{heap_type, 2, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, 0};
    Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> staging;
    check(device->CreateDescriptorHeap(&heap_desc, IID_PPV_ARGS(&staging)), "Create recurrent test heap");
    auto cpu = staging->GetCPUDescriptorHandleForHeapStart();
    device->CopyDescriptorsSimple(1, cpu, binding.srv, heap_type);
    cpu.ptr += stride;
    device->CopyDescriptorsSimple(1, cpu, binding.cpu_uav, heap_type);

    bvh_fixture::Kernel writer(device, {WriteRecurrent, sizeof(WriteRecurrent)});
    bvh_fixture::Kernel reader(device, {SampleMip, sizeof(SampleMip)});
    auto sampled = make_buffer(device, bytes, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COMMON,
                               D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT layout{};
    const UINT64 aligned = (bytes + D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT - 1) &
                           ~UINT64(D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT - 1);
    device->GetCopyableFootprints(&desc, 0, 1, aligned, &layout, nullptr, nullptr, nullptr);
    const UINT64 total = layout.Offset + UINT64(layout.Footprint.RowPitch) * height;
    auto readback = make_buffer(device, total, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST);
    for (const bool mirror : {false, true}) {
        const UINT seed = mirror ? 141u : 7u;
        ComputeCommands cmd(device, writer.pipeline.Get(), D3D12_COMMAND_LIST_TYPE_DIRECT);
        auto* list = cmd.list.Get();
        ID3D12DescriptorHeap* write_heap = mirror ? staging.Get() : binding.probe.heaps[2].Get();
        auto write_at = write_heap->GetGPUDescriptorHandleForHeapStart();
        write_at.ptr += stride;
        list->SetDescriptorHeaps(1, &write_heap);
        list->SetComputeRootSignature(writer.root.Get());
        list->SetComputeRootDescriptorTable(0, write_at);
        const UINT pattern[] = {width, height, seed};
        list->SetComputeRoot32BitConstants(1, 3, pattern, 0);
        list->Dispatch((width+7)/8, (height+7)/8, 1);
        transition(list, texture, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        transition(list, sampled.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

        ID3D12DescriptorHeap* read_heap = staging.Get();
        list->SetDescriptorHeaps(1, &read_heap);
        list->SetPipelineState(reader.pipeline.Get());
        list->SetComputeRootSignature(reader.root.Get());
        list->SetComputeRootDescriptorTable(0, read_heap->GetGPUDescriptorHandleForHeapStart());
        list->SetComputeRootUnorderedAccessView(1, sampled->GetGPUVirtualAddress());
        const UINT params[] = {0, width, height, 0};
        list->SetComputeRoot32BitConstants(2, 4, params, 0);
        list->Dispatch((width+7)/8, (height+7)/8, 1);
        transition(list, sampled.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
        list->CopyBufferRegion(readback.Get(), 0, sampled.Get(), 0, bytes);
        transition(list, sampled.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON);

        transition(list, texture, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COPY_SOURCE);
        D3D12_TEXTURE_COPY_LOCATION src{}, dst{};
        src.pResource = texture;
        src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        dst.pResource = readback.Get();
        dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        dst.PlacedFootprint = layout;
        list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
        transition(list, texture, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        cmd.submit_and_wait(device);
        const auto words = bvh_fixture::read(readback.Get(), static_cast<size_t>(total / sizeof(Word)));
        for (UINT y = 0; y < height; ++y) {
            const auto row = (layout.Offset + UINT64(y) * layout.Footprint.RowPitch) / sizeof(Word);
            for (UINT x = 0; x < width; ++x) {
                const UINT i = y * width + x;
                const auto pixel = expected_pixel(x, y, seed);
                bvh_fixture::equal(words[i], pixel, i, 0);       // Shader result through the SRV.
                bvh_fixture::equal(words[row+x], pixel, i, 1);   // Direct copy of the original texture.
            }
        }
        std::printf("recurrent_views uav=%s slot=1 srv_slot=1 stride=%u seed=%u srv_pixels=%u direct_pixels=%u mismatches=0\n",
                    mirror ? "cpu-mirror" : "shader-visible", stride, seed, width*height, width*height);
    }
}
