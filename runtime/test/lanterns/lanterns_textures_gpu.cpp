// Lanterns stage 6: texture upload (lanterns_textures_gpu.h). The footprint-per-subresource upload follows
// test/d3d12_m2_formats_test.cpp (M2), which uploads hand-built BC1 and BC7 mips the same way.
#include "lanterns_textures_gpu.h"

#include "lanterns_d3d12.h"

#include <cstring>

using namespace d3d12test;

namespace lanterns {

bool upload_texture(ID3D12Device* device, ID3D12CommandQueue* queue, const BcTexture& tex, Ref<ID3D12Resource>& out) {
    const UINT mips = UINT(tex.blocks.size());
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    desc.Width = tex.width;
    desc.Height = tex.height;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = UINT16(mips);
    desc.Format = tex.format == BcFormat::bc1 ? DXGI_FORMAT_BC1_UNORM : DXGI_FORMAT_BC7_UNORM;
    desc.SampleDesc.Count = 1;
    const D3D12_HEAP_PROPERTIES default_heap{D3D12_HEAP_TYPE_DEFAULT, {}, {}, 0, 0};
    const D3D12_HEAP_PROPERTIES upload_heap{D3D12_HEAP_TYPE_UPLOAD, {}, {}, 0, 0};
    CHECK(SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &desc,
                                                    D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&out.p))));
    std::vector<D3D12_PLACED_SUBRESOURCE_FOOTPRINT> layouts(mips);
    std::vector<UINT> rows(mips);
    std::vector<UINT64> row_bytes(mips);
    UINT64 total = 0;
    device->GetCopyableFootprints(&desc, 0, mips, 0, layouts.data(), rows.data(), row_bytes.data(), &total);
    CHECK(total != ~0ull && total > 0);
    Ref<ID3D12Resource> upload;
    const auto upload_desc = buffer_desc(total);
    CHECK(SUCCEEDED(device->CreateCommittedResource(&upload_heap, D3D12_HEAP_FLAG_NONE, &upload_desc,
                                                    D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&upload.p))));
    std::uint8_t* mapped = nullptr;
    CHECK(SUCCEEDED(upload->Map(0, nullptr, reinterpret_cast<void**>(&mapped))));
    for (UINT m = 0; m < mips; ++m) {
        // A footprint row is a row of 4x4 blocks.
        const std::size_t block_row = std::size_t((tex.mip_width(m) + 3) / 4) * tex.block_bytes();
        const bool fits = row_bytes[m] == block_row && rows[m] == (tex.mip_height(m) + 3) / 4;
        if (!fits) {
            upload->Unmap(0, nullptr);
            CHECK(fits);
        }
        for (UINT r = 0; r < rows[m]; ++r)
            std::memcpy(mapped + layouts[m].Offset + std::size_t(r) * layouts[m].Footprint.RowPitch,
                        tex.blocks[m].data() + r * block_row, block_row);
    }
    upload->Unmap(0, nullptr);

    Ref<ID3D12CommandAllocator> allocator;
    Ref<ID3D12GraphicsCommandList> list;
    Ref<ID3D12Fence> fence;
    CHECK(SUCCEEDED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator.p))));
    CHECK(SUCCEEDED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.p, nullptr,
                                              IID_PPV_ARGS(&list.p))));
    CHECK(SUCCEEDED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence.p))));
    for (UINT m = 0; m < mips; ++m) {
        D3D12_TEXTURE_COPY_LOCATION dst{};
        dst.pResource = out.p;
        dst.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        dst.SubresourceIndex = m;
        D3D12_TEXTURE_COPY_LOCATION src{};
        src.pResource = upload.p;
        src.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        src.PlacedFootprint = layouts[m];
        list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
    }
    const D3D12_RESOURCE_BARRIER ready = transition(out.p, D3D12_RESOURCE_STATE_COPY_DEST,
                                                    D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
    list->ResourceBarrier(1, &ready);
    CHECK(list->Close() == S_OK);
    ID3D12CommandList* lists[] = {list.p};
    queue->ExecuteCommandLists(1, lists);
    CHECK(SUCCEEDED(queue->Signal(fence.p, 1)));
    CHECK(wait_fence(fence, 1));   // the upload buffer is released after this
    return true;
}

}  // namespace lanterns
