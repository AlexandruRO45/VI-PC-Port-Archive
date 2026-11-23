// Block-compressed (BC1/BC7) CopyTextureRegion footprint bounds and block-alignment validation on a real Vulkan GPU,
// cross-checked against live Windows WARP. Valid padded and edge-block uploads/readbacks must round-trip exact bytes;
// oversized or interior non-block-aligned copies must be rejected without modifying the destination texture.
// No game data.
#include "../shims/d3d12_com.h"
#include "host_wine.h"

#include <dxgi1_4.h>

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

template<class T>
struct Ref {
    T* p = nullptr;
    ~Ref() {
        if (p) p->Release();
    }
    T* operator->() const { return p; }
};

int g_checks = 0;
int g_failures = 0;

void expect(bool ok, const char* what) {
    ++g_checks;
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++g_failures;
    }
}

bool wait_queue(ID3D12Device* device, ID3D12CommandQueue* queue) {
    Ref<ID3D12Fence> fence;
    if (FAILED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence.p))) ||
        FAILED(queue->Signal(fence.p, 1)))
        return false;
    HANDLE ev = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    if (!ev) return false;
    const bool ok = SUCCEEDED(fence->SetEventOnCompletion(1, ev)) && WaitForSingleObject(ev, 10000) == WAIT_OBJECT_0;
    CloseHandle(ev);
    return ok;
}

D3D12_RESOURCE_DESC buffer_desc(UINT64 width) {
    D3D12_RESOURCE_DESC d{};
    d.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    d.Width = width;
    d.Height = 1;
    d.DepthOrArraySize = 1;
    d.MipLevels = 1;
    d.SampleDesc.Count = 1;
    d.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    return d;
}

D3D12_RESOURCE_DESC bc_tex_desc(DXGI_FORMAT fmt, UINT w, UINT h, UINT16 mips = 1) {
    D3D12_RESOURCE_DESC d{};
    d.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    d.Width = w;
    d.Height = h;
    d.DepthOrArraySize = 1;
    d.MipLevels = mips;
    d.Format = fmt;
    d.SampleDesc.Count = 1;
    return d;
}

struct InvalidCase {
    const char* name;
    DXGI_FORMAT format;
    UINT tex_w, tex_h, mips, sub;
    bool upload;
    UINT dst_x, dst_y;
    UINT fp_w, fp_h;
    bool has_box;
    D3D12_BOX box;
};

constexpr InvalidCase kInvalidCases[] = {
    {"8x8 BC1 oversized 16x16 upload", DXGI_FORMAT_BC1_UNORM, 8, 8, 1, 0, true, 0, 0, 16, 16, false, {}},
    {"8x8 BC1 8x8 upload at x=4", DXGI_FORMAT_BC1_UNORM, 8, 8, 1, 0, true, 4, 0, 8, 8, false, {}},
    {"8x8 BC1 8x8 upload at y=4", DXGI_FORMAT_BC1_UNORM, 8, 8, 1, 0, true, 0, 4, 8, 8, false, {}},
    {"7x5 BC1 oversized 12x8 upload", DXGI_FORMAT_BC1_UNORM, 7, 5, 1, 0, true, 0, 0, 12, 8, false, {}},
    {"7x5 BC1 8x4 upload at x=4", DXGI_FORMAT_BC1_UNORM, 7, 5, 1, 0, true, 4, 0, 8, 4, false, {}},
    {"1x1 BC1 mip oversized 8x4 upload", DXGI_FORMAT_BC1_UNORM, 8, 8, 4, 3, true, 0, 0, 8, 4, false, {}},
    {"8x8 BC1 oversized 12x8 readback", DXGI_FORMAT_BC1_UNORM, 8, 8, 1, 0, false, 0, 0, 12, 8, true, {0, 0, 0, 12, 8, 1}},
    {"8x8 BC1 interior unaligned 3x4 upload", DXGI_FORMAT_BC1_UNORM, 8, 8, 1, 0, true, 0, 0, 8, 8, true, {0, 0, 0, 3, 4, 1}},
    {"8x8 BC1 interior unaligned 4x3 upload", DXGI_FORMAT_BC1_UNORM, 8, 8, 1, 0, true, 0, 0, 8, 8, true, {0, 0, 0, 4, 3, 1}},
    {"8x8 BC1 interior unaligned 6x4 readback", DXGI_FORMAT_BC1_UNORM, 8, 8, 1, 0, false, 0, 0, 8, 8, true, {0, 0, 0, 6, 4, 1}},
    {"7x5 BC1 interior unaligned 6x4 upload", DXGI_FORMAT_BC1_UNORM, 7, 5, 1, 0, true, 0, 0, 8, 8, true, {0, 0, 0, 6, 4, 1}},
    {"7x5 BC1 interior unaligned 6x4 readback", DXGI_FORMAT_BC1_UNORM, 7, 5, 1, 0, false, 0, 0, 8, 8, true, {0, 0, 0, 6, 4, 1}},
    {"7x5 BC7 8x4 upload at x=4", DXGI_FORMAT_BC7_UNORM, 7, 5, 1, 0, true, 4, 0, 8, 4, false, {}},
};

void run_invalid_cases(ID3D12Device* dev, bool is_layer, const char* label) {
    const D3D12_HEAP_PROPERTIES uh{D3D12_HEAP_TYPE_UPLOAD}, rh{D3D12_HEAP_TYPE_READBACK}, dh{D3D12_HEAP_TYPE_DEFAULT};
    const auto bdesc = buffer_desc(65536);
    Ref<ID3D12Resource> up_buf, rb_buf;
    expect(SUCCEEDED(dev->CreateCommittedResource(&uh, D3D12_HEAP_FLAG_NONE, &bdesc, D3D12_RESOURCE_STATE_GENERIC_READ,
                                                  nullptr, IID_PPV_ARGS(&up_buf.p))),
           "create upload buffer");
    expect(SUCCEEDED(dev->CreateCommittedResource(&rh, D3D12_HEAP_FLAG_NONE, &bdesc, D3D12_RESOURCE_STATE_COPY_DEST,
                                                  nullptr, IID_PPV_ARGS(&rb_buf.p))),
           "create readback buffer");
    Ref<ID3D12CommandQueue> queue;
    D3D12_COMMAND_QUEUE_DESC qd{};
    qd.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    expect(SUCCEEDED(dev->CreateCommandQueue(&qd, IID_PPV_ARGS(&queue.p))), "create queue");

    for (const InvalidCase& c : kInvalidCases) {
        Ref<ID3D12Resource> tex;
        const auto tdesc = bc_tex_desc(c.format, c.tex_w, c.tex_h, static_cast<UINT16>(c.mips));
        if (FAILED(dev->CreateCommittedResource(&dh, D3D12_HEAP_FLAG_NONE, &tdesc,
                                                c.upload ? D3D12_RESOURCE_STATE_COPY_DEST
                                                         : D3D12_RESOURCE_STATE_COPY_SOURCE,
                                                nullptr, IID_PPV_ARGS(&tex.p)))) {
            expect(false, c.name);
            continue;
        }
        Ref<ID3D12CommandAllocator> alloc;
        Ref<ID3D12GraphicsCommandList> list;
        expect(SUCCEEDED(dev->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&alloc.p))) &&
                   SUCCEEDED(dev->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, alloc.p, nullptr,
                                                    IID_PPV_ARGS(&list.p))),
               "create allocator and list");
        D3D12_TEXTURE_COPY_LOCATION tloc{}, bloc{};
        tloc.pResource = tex.p;
        tloc.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        tloc.SubresourceIndex = c.sub;
        bloc.pResource = c.upload ? up_buf.p : rb_buf.p;
        bloc.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        bloc.PlacedFootprint.Offset = 0;
        bloc.PlacedFootprint.Footprint = {c.format, c.fp_w, c.fp_h, 1, 256};
        const auto& dst = c.upload ? tloc : bloc;
        const auto& src = c.upload ? bloc : tloc;
        list->CopyTextureRegion(&dst, c.dst_x, c.dst_y, 0, &src, c.has_box ? &c.box : nullptr);
        const HRESULT close_hr = list->Close();
        bool rejected = FAILED(close_hr);
        if (!rejected && is_layer) {
            ID3D12CommandList* lists[] = {list.p};
            queue->ExecuteCommandLists(1, lists);
            rejected = rt::d3d12::last_execute_result(queue.p) == leonida::ReplayResult::invalid_copy;
        }
        if (!rejected) {
            std::fprintf(stderr, "FAIL [%s]: %s was not rejected (Close=0x%08lx)\n", label, c.name,
                         static_cast<unsigned long>(close_hr));
            ++g_failures;
        }
        ++g_checks;
    }
}

void run_gpu_roundtrip_and_preservation(ID3D12Device* dev) {
    const D3D12_HEAP_PROPERTIES uh{D3D12_HEAP_TYPE_UPLOAD}, rh{D3D12_HEAP_TYPE_READBACK}, dh{D3D12_HEAP_TYPE_DEFAULT};
    const auto bdesc = buffer_desc(65536);
    Ref<ID3D12Resource> up_buf, bad_buf, rb_buf, tex_7x5, tex_8x8_mips;
    expect(SUCCEEDED(dev->CreateCommittedResource(&uh, D3D12_HEAP_FLAG_NONE, &bdesc, D3D12_RESOURCE_STATE_GENERIC_READ,
                                                  nullptr, IID_PPV_ARGS(&up_buf.p))),
           "roundtrip upload buffer");
    expect(SUCCEEDED(dev->CreateCommittedResource(&uh, D3D12_HEAP_FLAG_NONE, &bdesc, D3D12_RESOURCE_STATE_GENERIC_READ,
                                                  nullptr, IID_PPV_ARGS(&bad_buf.p))),
           "roundtrip bad upload buffer");
    expect(SUCCEEDED(dev->CreateCommittedResource(&rh, D3D12_HEAP_FLAG_NONE, &bdesc, D3D12_RESOURCE_STATE_COPY_DEST,
                                                  nullptr, IID_PPV_ARGS(&rb_buf.p))),
           "roundtrip readback buffer");
    const auto desc_7x5 = bc_tex_desc(DXGI_FORMAT_BC1_UNORM, 7, 5, 1);
    const auto desc_8x8 = bc_tex_desc(DXGI_FORMAT_BC1_UNORM, 8, 8, 4);
    expect(SUCCEEDED(dev->CreateCommittedResource(&dh, D3D12_HEAP_FLAG_NONE, &desc_7x5, D3D12_RESOURCE_STATE_COPY_DEST,
                                                  nullptr, IID_PPV_ARGS(&tex_7x5.p))),
           "create 7x5 BC1 texture");
    expect(SUCCEEDED(dev->CreateCommittedResource(&dh, D3D12_HEAP_FLAG_NONE, &desc_8x8, D3D12_RESOURCE_STATE_COPY_DEST,
                                                  nullptr, IID_PPV_ARGS(&tex_8x8_mips.p))),
           "create 8x8 BC1 mip-chain texture");

    // 7x5 BC1 has 2x2 = 4 blocks (8 bytes each), row pitch 256 -> row 0 at 0..15, row 1 at 256..271.
    // Fill up_buf with a distinct pattern per block and bad_buf with 0xEE.
    std::array<std::uint8_t, 32> pattern{};
    for (std::size_t i = 0; i < pattern.size(); ++i) pattern[i] = static_cast<std::uint8_t>(0x30u + i * 5u);
    void* mapped = nullptr;
    D3D12_RANGE none{0, 0};
    expect(SUCCEEDED(up_buf->Map(0, &none, &mapped)) && mapped, "map up_buf");
    auto* up_bytes = static_cast<std::uint8_t*>(mapped);
    std::memset(up_bytes, 0, 65536);
    std::memcpy(up_bytes + 0, pattern.data(), 16);
    std::memcpy(up_bytes + 256, pattern.data() + 16, 16);
    // Offset 1024: single 4x4 block for edge-block update (pattern XOR 0x55) and 1x1 mip upload.
    std::array<std::uint8_t, 8> edge_block{};
    for (std::size_t i = 0; i < 8; ++i) edge_block[i] = static_cast<std::uint8_t>(0xA1u + i * 7u);
    std::memcpy(up_bytes + 1024, edge_block.data(), 8);
    up_buf->Unmap(0, nullptr);

    expect(SUCCEEDED(bad_buf->Map(0, &none, &mapped)) && mapped, "map bad_buf");
    std::memset(mapped, 0xEE, 65536);
    bad_buf->Unmap(0, nullptr);

    Ref<ID3D12CommandQueue> queue;
    D3D12_COMMAND_QUEUE_DESC qd{};
    qd.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    expect(SUCCEEDED(dev->CreateCommandQueue(&qd, IID_PPV_ARGS(&queue.p))), "roundtrip queue");

    auto submit_one = [&](auto&& record) -> leonida::ReplayResult {
        Ref<ID3D12CommandAllocator> alloc;
        Ref<ID3D12GraphicsCommandList> list;
        if (FAILED(dev->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&alloc.p))) ||
            FAILED(dev->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, alloc.p, nullptr, IID_PPV_ARGS(&list.p))))
            return leonida::ReplayResult::unsupported;
        record(list.p);
        if (FAILED(list->Close())) return leonida::ReplayResult::invalid_copy;
        ID3D12CommandList* lists[] = {list.p};
        queue->ExecuteCommandLists(1, lists);
        const auto res = rt::d3d12::last_execute_result(queue.p);
        expect(wait_queue(dev, queue.p), "wait queue");
        return res;
    };

    // 1. Upload valid 8x8 padded footprint into 7x5 BC1 texture, then overwrite corner block (4,4) with edge_block.
    expect(submit_one([&](ID3D12GraphicsCommandList* list) {
               D3D12_TEXTURE_COPY_LOCATION dst{}, src{};
               dst.pResource = tex_7x5.p;
               dst.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
               src.pResource = up_buf.p;
               src.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
               src.PlacedFootprint.Offset = 0;
               src.PlacedFootprint.Footprint = {DXGI_FORMAT_BC1_UNORM, 8, 8, 1, 256};
               list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);

               src.PlacedFootprint.Offset = 1024;
               src.PlacedFootprint.Footprint = {DXGI_FORMAT_BC1_UNORM, 4, 4, 1, 256};
               list->CopyTextureRegion(&dst, 4, 4, 0, &src, nullptr);

               // Also upload 1x1 mip 3 of 8x8 BC1 texture via 4x4 padded footprint.
               dst.pResource = tex_8x8_mips.p;
               dst.SubresourceIndex = 3;
               list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
           }) == leonida::ReplayResult::success,
           "valid 7x5 whole + (4,4) edge + 1x1 mip BC1 uploads succeed on GPU");

    // 2. Attempt an oversized 12x8 upload from bad_buf (0xEE) into tex_7x5: must be rejected and NOT overwrite tex_7x5!
    expect(submit_one([&](ID3D12GraphicsCommandList* list) {
               D3D12_TEXTURE_COPY_LOCATION dst{}, src{};
               dst.pResource = tex_7x5.p;
               dst.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
               src.pResource = bad_buf.p;
               src.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
               src.PlacedFootprint.Offset = 0;
               src.PlacedFootprint.Footprint = {DXGI_FORMAT_BC1_UNORM, 12, 8, 1, 256};
               list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
           }) == leonida::ReplayResult::invalid_copy,
           "oversized 12x8 BC1 upload is rejected");

    // 3. Read back tex_7x5 (padded 8x8 box) and tex_8x8_mips mip 3 (4x4 box) and verify exact preserved bytes.
    expect(submit_one([&](ID3D12GraphicsCommandList* list) {
               D3D12_RESOURCE_BARRIER b[2]{};
               b[0].Type = b[1].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
               b[0].Transition = {tex_7x5.p, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, D3D12_RESOURCE_STATE_COPY_DEST,
                                  D3D12_RESOURCE_STATE_COPY_SOURCE};
               b[1].Transition = {tex_8x8_mips.p, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                                  D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COPY_SOURCE};
               list->ResourceBarrier(2, b);

               D3D12_TEXTURE_COPY_LOCATION dst{}, src{};
               dst.pResource = rb_buf.p;
               dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
               dst.PlacedFootprint.Offset = 0;
               dst.PlacedFootprint.Footprint = {DXGI_FORMAT_BC1_UNORM, 8, 8, 1, 256};
               src.pResource = tex_7x5.p;
               src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
               const D3D12_BOX box_8x8{0, 0, 0, 8, 8, 1};
               list->CopyTextureRegion(&dst, 0, 0, 0, &src, &box_8x8);

               dst.PlacedFootprint.Offset = 1024;
               dst.PlacedFootprint.Footprint = {DXGI_FORMAT_BC1_UNORM, 4, 4, 1, 256};
               src.pResource = tex_8x8_mips.p;
               src.SubresourceIndex = 3;
               const D3D12_BOX box_4x4{0, 0, 0, 4, 4, 1};
               list->CopyTextureRegion(&dst, 0, 0, 0, &src, &box_4x4);
           }) == leonida::ReplayResult::success,
           "readback of 7x5 BC1 and 1x1 BC1 mip succeeds");

    D3D12_RANGE all{0, 65536};
    expect(SUCCEEDED(rb_buf->Map(0, &all, &mapped)) && mapped, "map rb_buf");
    const auto* rb = static_cast<const std::uint8_t*>(mapped);
    expect(std::memcmp(rb + 0, pattern.data(), 16) == 0, "7x5 BC1 row 0 blocks preserved intact");
    expect(std::memcmp(rb + 256, pattern.data() + 16, 8) == 0, "7x5 BC1 block (0,1) preserved intact");
    expect(std::memcmp(rb + 256 + 8, edge_block.data(), 8) == 0, "7x5 BC1 corner block (1,1) matches edge upload");
    expect(std::memcmp(rb + 1024, edge_block.data(), 8) == 0, "8x8 BC1 mip 3 (1x1) matches upload");
    rb_buf->Unmap(0, &none);
}

}  // namespace

int main() {
    if (!hostwine::host_is_wine()) {
        Ref<IDXGIFactory4> factory;
        Ref<IDXGIAdapter> warp;
        Ref<ID3D12Device> host;
        if (SUCCEEDED(CreateDXGIFactory1(IID_PPV_ARGS(&factory.p))) &&
            SUCCEEDED(factory->EnumWarpAdapter(IID_PPV_ARGS(&warp.p))) &&
            SUCCEEDED(D3D12CreateDevice(warp.p, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&host.p)))) {
            run_invalid_cases(host.p, false, "WARP");
        }
    }

    Ref<ID3D12Device> layer;
    if (FAILED(rt::d3d12::create_vulkan_device_standalone(IID_PPV_ARGS(&layer.p)))) return 77;
    run_invalid_cases(layer.p, true, "Leonida Vulkan");
    run_gpu_roundtrip_and_preservation(layer.p);

    std::printf("d3d12_bc_copy_bounds_test: %d checks, %d failures\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
