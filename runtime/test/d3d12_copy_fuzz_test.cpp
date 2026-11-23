// Differential copy fuzzer: one seeded random program of CopyBufferRegion, CopyTextureRegion (buffer to texture,
// texture to buffer, texture to texture, with boxes) and CopyResource over two buffers and three textures of one format
// (12 formats, BC1/3/7 among them; mips and arrays), filled with known bytes, runs on the layer and on WARP, and every
// byte of every resource is read back and compared. Copies that touch a resource back to back while it stays in
// COPY_DEST or COPY_SOURCE get no barrier between them: D3D12 runs them in program order (WARP does), and the layer
// must too. The reference is WARP, not the GPU's own D3D12: NVIDIA's driver (621.14) lets a texture-to-buffer copy
// land after a later CopyBufferRegion over the same bytes, where WARP keeps program order (#4997).
// --seeds N --start S choose the programs (default 30 from 1); --hardware compares with the GPU's D3D12 instead;
// --reduce SEED drops copies and list breaks while the layer still differs and prints the smallest program left.
// With LEONIDA_VK_VALIDATION=1 the layer runs under synchronization validation, which reports a missing barrier
// between two copies even where the bytes come out right. Needs a Vulkan GPU (exit 77 otherwise, and under Wine).
// No game data.
#include "../shims/d3d12_com.h"

#include <dxgi1_4.h>
#include <wrl/client.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

using Microsoft::WRL::ComPtr;

namespace {

struct FormatInfo {
    DXGI_FORMAT format;
    UINT block;   // texels per block side (1 or 4)
    UINT bytes;   // bytes per texel or block
    const char* name;
};
constexpr FormatInfo kFormats[] = {
    {DXGI_FORMAT_R8G8B8A8_UNORM, 1, 4, "R8G8B8A8_UNORM"},     {DXGI_FORMAT_B8G8R8A8_UNORM, 1, 4, "B8G8R8A8_UNORM"},
    {DXGI_FORMAT_R16G16B16A16_FLOAT, 1, 8, "R16G16B16A16_FLOAT"}, {DXGI_FORMAT_R32_FLOAT, 1, 4, "R32_FLOAT"},
    {DXGI_FORMAT_R32G32_UINT, 1, 8, "R32G32_UINT"},           {DXGI_FORMAT_R8_UNORM, 1, 1, "R8_UNORM"},
    {DXGI_FORMAT_R16_UINT, 1, 2, "R16_UINT"},                 {DXGI_FORMAT_R10G10B10A2_UNORM, 1, 4, "R10G10B10A2_UNORM"},
    {DXGI_FORMAT_R32G32B32A32_UINT, 1, 16, "R32G32B32A32_UINT"}, {DXGI_FORMAT_BC1_UNORM, 4, 8, "BC1_UNORM"},
    {DXGI_FORMAT_BC3_UNORM, 4, 16, "BC3_UNORM"},              {DXGI_FORMAT_BC7_UNORM, 4, 16, "BC7_UNORM"},
};

// WARP by default; the first hardware adapter with --hardware.
ComPtr<ID3D12Device> reference_device(bool hardware, std::wstring& name) {
    ComPtr<IDXGIFactory4> factory;
    if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory)))) return {};
    if (!hardware) {
        ComPtr<IDXGIAdapter> warp;
        ComPtr<ID3D12Device> device;
        if (FAILED(factory->EnumWarpAdapter(IID_PPV_ARGS(&warp))) ||
            FAILED(D3D12CreateDevice(warp.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&device))))
            return {};
        name = L"WARP";
        return device;
    }
    for (UINT i = 0;; ++i) {
        ComPtr<IDXGIAdapter1> adapter;
        if (factory->EnumAdapters1(i, &adapter) != S_OK) return {};
        DXGI_ADAPTER_DESC1 desc{};
        if (FAILED(adapter->GetDesc1(&desc)) || (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE)) continue;
        ComPtr<ID3D12Device> device;
        if (FAILED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&device)))) continue;
        name = desc.Description;
        return device;
    }
}

constexpr UINT kBuffers = 2, kTextures = 3;
constexpr UINT64 kBufferBytes = 1 << 18;
constexpr UINT64 kStagingBytes = 1 << 23;  // upload and readback buffers

struct TextureShape {
    UINT width, height, layers, mips;
};

// One copy. Resource indices: 0..1 buffers, 2..4 textures.
struct Copy {
    enum Kind { buffer_to_buffer, buffer_to_texture, texture_to_texture, texture_to_buffer, whole_resource } kind;
    UINT dst = 0, src = 0, dst_sub = 0, src_sub = 0;
    UINT64 dst_offset = 0, src_offset = 0, bytes = 0;  // buffers
    UINT dst_x = 0, dst_y = 0;                          // texture destination
    D3D12_BOX box{};                                    // texture source (or the region, for buffer sources)
    UINT row_pitch = 0;                                 // buffer footprints
};

struct Program {
    FormatInfo format;
    TextureShape shapes[kTextures];
    std::vector<Copy> copies;
    std::vector<UINT> list_breaks;  // copy indices where a new command list starts
};

UINT mip_extent(UINT size, UINT mip) { return std::max(1u, size >> mip); }

Program generate(std::uint64_t seed) {
    std::mt19937_64 rng(seed * 0x2545f4914f6cdd1dull + 7);
    const auto pick = [&](UINT n) { return UINT(rng() % n); };
    Program p;
    p.format = kFormats[pick(UINT(std::size(kFormats)))];
    const UINT b = p.format.block;
    for (UINT t = 0; t < kTextures; ++t) {
        TextureShape& s = p.shapes[t];
        s.width = b * (1 + pick(40));
        s.height = b * (1 + pick(40));
        s.layers = 1 + pick(3);
        UINT full = 0;
        for (UINT m = std::max(s.width, s.height); m; m >>= 1) ++full;
        s.mips = 1 + pick(full);
    }
    if (pick(2)) p.shapes[1] = p.shapes[2];  // so CopyResource has a pair to copy
    const auto sub_count = [&](UINT t) { return p.shapes[t].mips * p.shapes[t].layers; };
    // A block-aligned box inside a subresource (whole blocks only: mips smaller than a block are skipped).
    const auto box_in = [&](UINT t, UINT sub, D3D12_BOX& box) {
        const TextureShape& s = p.shapes[t];
        const UINT mip = sub % s.mips;
        const UINT w = mip_extent(s.width, mip) / b * b, h = mip_extent(s.height, mip) / b * b;
        if (!w || !h) return false;
        const UINT x0 = pick(w / b) * b, y0 = pick(h / b) * b;
        const UINT x1 = x0 + b * (1 + pick((w - x0) / b)), y1 = y0 + b * (1 + pick((h - y0) / b));
        box = {x0, y0, 0, x1, y1, 1};
        return true;
    };
    const UINT n = 12 + pick(30);
    for (UINT i = 0; i < n; ++i) {
        if (i && pick(5) == 0) p.list_breaks.push_back(i);
        Copy c{};
        switch (pick(5)) {
        case 0:
            c.kind = Copy::buffer_to_buffer;
            c.dst = pick(kBuffers);
            c.src = (c.dst + 1 + pick(kBuffers - 1)) % kBuffers;
            c.bytes = 1 + rng() % 9000;
            c.dst_offset = rng() % (kBufferBytes - c.bytes);
            c.src_offset = rng() % (kBufferBytes - c.bytes);
            break;
        case 1:
        case 3: {
            const bool to_texture = pick(2) == 0;
            c.kind = to_texture ? Copy::buffer_to_texture : Copy::texture_to_buffer;
            const UINT t = pick(kTextures);
            UINT sub = pick(sub_count(t));
            D3D12_BOX box{};
            if (!box_in(t, sub, box) && !box_in(t, sub = 0, box)) continue;
            const UINT row_bytes = (box.right - box.left) / b * p.format.bytes;
            c.row_pitch = (row_bytes + 255) / 256 * 256 + 256 * pick(2);
            const UINT64 footprint = UINT64(c.row_pitch) * ((box.bottom - box.top) / b);
            if (footprint + 512 > kBufferBytes) continue;
            const UINT64 offset = (rng() % ((kBufferBytes - footprint) / 512)) * 512;
            if (to_texture) {
                c.dst = 2 + t; c.dst_sub = sub; c.dst_x = box.left; c.dst_y = box.top;
                c.src = pick(kBuffers); c.src_offset = offset;
                c.box = {0, 0, 0, box.right - box.left, box.bottom - box.top, 1};
            } else {
                c.src = 2 + t; c.src_sub = sub; c.box = box;
                c.dst = pick(kBuffers); c.dst_offset = offset;
            }
            break;
        }
        case 2: {
            c.kind = Copy::texture_to_texture;
            const UINT s = pick(kTextures), d = (s + 1 + pick(kTextures - 1)) % kTextures;
            c.src = 2 + s; c.dst = 2 + d;
            c.src_sub = pick(sub_count(s));
            D3D12_BOX box{};
            if (!box_in(s, c.src_sub, box)) continue;
            // Destination: a subresource with room for the box, at a block-aligned spot.
            c.dst_sub = pick(sub_count(d));
            const TextureShape& ds = p.shapes[d];
            const UINT dmip = c.dst_sub % ds.mips;
            const UINT dw = mip_extent(ds.width, dmip) / b * b, dh = mip_extent(ds.height, dmip) / b * b;
            UINT w = box.right - box.left, h = box.bottom - box.top;
            if (dw < b || dh < b) continue;
            w = std::min(w, dw); h = std::min(h, dh);
            box.right = box.left + w; box.bottom = box.top + h;
            c.box = box;
            c.dst_x = pick((dw - w) / b + 1) * b;
            c.dst_y = pick((dh - h) / b + 1) * b;
            break;
        }
        default:
            if (!(p.shapes[1].width == p.shapes[2].width && p.shapes[1].height == p.shapes[2].height &&
                  p.shapes[1].layers == p.shapes[2].layers && p.shapes[1].mips == p.shapes[2].mips))
                continue;
            c.kind = Copy::whole_resource;
            c.src = 2 + 1 + pick(2);
            c.dst = c.src == 3 ? 4 : 3;
            break;
        }
        p.copies.push_back(c);
    }
    return p;
}

std::string describe(const Program& p, const Copy& c) {
    char text[200];
    switch (c.kind) {
    case Copy::buffer_to_buffer:
        std::snprintf(text, sizeof text, "CopyBufferRegion(b%u+%llu <- b%u+%llu, %llu bytes)", c.dst,
                      static_cast<unsigned long long>(c.dst_offset), c.src,
                      static_cast<unsigned long long>(c.src_offset), static_cast<unsigned long long>(c.bytes));
        break;
    case Copy::buffer_to_texture:
        std::snprintf(text, sizeof text, "CopyTextureRegion(t%u sub %u at %u,%u <- b%u+%llu pitch %u, %ux%u)",
                      c.dst - 2, c.dst_sub, c.dst_x, c.dst_y, c.src, static_cast<unsigned long long>(c.src_offset),
                      c.row_pitch, c.box.right, c.box.bottom);
        break;
    case Copy::texture_to_buffer:
        std::snprintf(text, sizeof text, "CopyTextureRegion(b%u+%llu pitch %u <- t%u sub %u box %u,%u-%u,%u)", c.dst,
                      static_cast<unsigned long long>(c.dst_offset), c.row_pitch, c.src - 2, c.src_sub, c.box.left,
                      c.box.top, c.box.right, c.box.bottom);
        break;
    case Copy::texture_to_texture:
        std::snprintf(text, sizeof text, "CopyTextureRegion(t%u sub %u at %u,%u <- t%u sub %u box %u,%u-%u,%u)",
                      c.dst - 2, c.dst_sub, c.dst_x, c.dst_y, c.src - 2, c.src_sub, c.box.left, c.box.top,
                      c.box.right, c.box.bottom);
        break;
    case Copy::whole_resource:
        std::snprintf(text, sizeof text, "CopyResource(t%u <- t%u)", c.dst - 2, c.src - 2);
        break;
    }
    (void)p;
    return text;
}

D3D12_RESOURCE_DESC buffer_desc(UINT64 bytes) {
    D3D12_RESOURCE_DESC d{};
    d.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    d.Width = bytes;
    d.Height = 1;
    d.DepthOrArraySize = 1;
    d.MipLevels = 1;
    d.SampleDesc = {1, 0};
    d.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    return d;
}

D3D12_RESOURCE_DESC texture_desc(const Program& p, UINT t) {
    D3D12_RESOURCE_DESC d{};
    d.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    d.Width = p.shapes[t].width;
    d.Height = p.shapes[t].height;
    d.DepthOrArraySize = UINT16(p.shapes[t].layers);
    d.MipLevels = UINT16(p.shapes[t].mips);
    d.Format = p.format.format;
    d.SampleDesc = {1, 0};
    return d;
}

std::uint8_t pattern(std::uint64_t seed, UINT resource, UINT64 i) {
    std::uint64_t x = seed * 0x9e3779b97f4a7c15ull + resource * 0xbf58476d1ce4e5b9ull + i * 0x94d049bb133111ebull;
    x ^= x >> 31; x *= 0xd6e8feb86659fd93ull; x ^= x >> 32;
    return std::uint8_t(x);
}

// Runs the program; returns every resource's bytes (buffers whole, textures subresource by subresource, rows without
// padding), or an empty vector and a reason.
std::vector<std::uint8_t> run(ID3D12Device* device, const Program& p, std::uint64_t seed, std::string& error) {
    std::vector<std::uint8_t> out;
    const auto fail = [&](const char* why) { error = why; return std::vector<std::uint8_t>{}; };
    D3D12_HEAP_PROPERTIES def{D3D12_HEAP_TYPE_DEFAULT}, up{D3D12_HEAP_TYPE_UPLOAD}, rb{D3D12_HEAP_TYPE_READBACK};
    ComPtr<ID3D12Resource> res[kBuffers + kTextures], upload, readback;
    D3D12_RESOURCE_STATES state[kBuffers + kTextures];
    for (UINT i = 0; i < kBuffers; ++i) {
        const auto d = buffer_desc(kBufferBytes);
        if (FAILED(device->CreateCommittedResource(&def, D3D12_HEAP_FLAG_NONE, &d, D3D12_RESOURCE_STATE_COMMON,
                                                   nullptr, IID_PPV_ARGS(&res[i]))))
            return fail("buffer");
        state[i] = D3D12_RESOURCE_STATE_COMMON;
    }
    for (UINT t = 0; t < kTextures; ++t) {
        const auto d = texture_desc(p, t);
        if (FAILED(device->CreateCommittedResource(&def, D3D12_HEAP_FLAG_NONE, &d, D3D12_RESOURCE_STATE_COPY_DEST,
                                                   nullptr, IID_PPV_ARGS(&res[kBuffers + t]))))
            return fail("texture");
        state[kBuffers + t] = D3D12_RESOURCE_STATE_COPY_DEST;
    }
    const auto sd = buffer_desc(kStagingBytes);
    if (FAILED(device->CreateCommittedResource(&up, D3D12_HEAP_FLAG_NONE, &sd, D3D12_RESOURCE_STATE_GENERIC_READ,
                                               nullptr, IID_PPV_ARGS(&upload))) ||
        FAILED(device->CreateCommittedResource(&rb, D3D12_HEAP_FLAG_NONE, &sd, D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                               IID_PPV_ARGS(&readback))))
        return fail("staging");
    D3D12_COMMAND_QUEUE_DESC qd{};
    ComPtr<ID3D12CommandQueue> queue;
    ComPtr<ID3D12CommandAllocator> alloc;
    ComPtr<ID3D12GraphicsCommandList> list;
    ComPtr<ID3D12Fence> fence;
    if (FAILED(device->CreateCommandQueue(&qd, IID_PPV_ARGS(&queue))) ||
        FAILED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&alloc))) ||
        FAILED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, alloc.Get(), nullptr, IID_PPV_ARGS(&list))) ||
        FAILED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence))))
        return fail("queue");
    UINT64 fence_value = 0;
    const auto submit = [&]() {
        list->Close();
        ID3D12CommandList* l = list.Get();
        queue->ExecuteCommandLists(1, &l);
        queue->Signal(fence.Get(), ++fence_value);
        HANDLE e = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        fence->SetEventOnCompletion(fence_value, e);
        const bool ok = WaitForSingleObject(e, 20000) == WAIT_OBJECT_0;
        CloseHandle(e);
        alloc->Reset();
        list->Reset(alloc.Get(), nullptr);
        return ok;
    };
    const auto transition = [&](UINT r, D3D12_RESOURCE_STATES to) {
        if (state[r] == to) return;
        D3D12_RESOURCE_BARRIER barrier{};
        barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        barrier.Transition = {res[r].Get(), D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, state[r], to};
        list->ResourceBarrier(1, &barrier);
        state[r] = to;
    };

    // Fill: the upload buffer with the seed's bytes, then every buffer and every texture subresource from it.
    std::uint8_t* mapped = nullptr;
    if (FAILED(upload->Map(0, nullptr, reinterpret_cast<void**>(&mapped)))) return fail("map upload");
    for (UINT64 i = 0; i < kStagingBytes; ++i) mapped[i] = pattern(seed, 99, i);
    upload->Unmap(0, nullptr);
    for (UINT i = 0; i < kBuffers; ++i) {
        transition(i, D3D12_RESOURCE_STATE_COPY_DEST);
        list->CopyBufferRegion(res[i].Get(), 0, upload.Get(), i * kBufferBytes, kBufferBytes);
    }
    UINT64 staging_offset = kBuffers * kBufferBytes;
    for (UINT t = 0; t < kTextures; ++t) {
        const auto d = texture_desc(p, t);
        const UINT subs = p.shapes[t].mips * p.shapes[t].layers;
        std::vector<D3D12_PLACED_SUBRESOURCE_FOOTPRINT> fp(subs);
        UINT64 total = 0;
        device->GetCopyableFootprints(&d, 0, subs, staging_offset, fp.data(), nullptr, nullptr, &total);
        if (staging_offset + total > kStagingBytes) return fail("staging too small");
        for (UINT s = 0; s < subs; ++s) {
            D3D12_TEXTURE_COPY_LOCATION dst{res[kBuffers + t].Get(), D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX};
            dst.SubresourceIndex = s;
            D3D12_TEXTURE_COPY_LOCATION src{upload.Get(), D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT};
            src.PlacedFootprint = fp[s];
            list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
        }
        staging_offset = (staging_offset + total + 511) / 512 * 512;
    }
    if (!submit()) return fail("fill timed out");

    // The program, with transitions only where a resource changes between source and destination.
    size_t next_break = 0;
    for (size_t i = 0; i < p.copies.size(); ++i) {
        if (next_break < p.list_breaks.size() && p.list_breaks[next_break] == i) {
            ++next_break;
            if (!submit()) return fail("program timed out");
        }
        const Copy& c = p.copies[i];
        transition(c.src, D3D12_RESOURCE_STATE_COPY_SOURCE);
        transition(c.dst, D3D12_RESOURCE_STATE_COPY_DEST);
        switch (c.kind) {
        case Copy::buffer_to_buffer:
            list->CopyBufferRegion(res[c.dst].Get(), c.dst_offset, res[c.src].Get(), c.src_offset, c.bytes);
            break;
        case Copy::buffer_to_texture: {
            D3D12_TEXTURE_COPY_LOCATION dst{res[c.dst].Get(), D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX};
            dst.SubresourceIndex = c.dst_sub;
            D3D12_TEXTURE_COPY_LOCATION src{res[c.src].Get(), D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT};
            src.PlacedFootprint = {c.src_offset, {p.format.format, c.box.right, c.box.bottom, 1, c.row_pitch}};
            list->CopyTextureRegion(&dst, c.dst_x, c.dst_y, 0, &src, nullptr);
            break;
        }
        case Copy::texture_to_buffer: {
            D3D12_TEXTURE_COPY_LOCATION dst{res[c.dst].Get(), D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT};
            dst.PlacedFootprint = {c.dst_offset, {p.format.format, c.box.right - c.box.left,
                                                  c.box.bottom - c.box.top, 1, c.row_pitch}};
            D3D12_TEXTURE_COPY_LOCATION src{res[c.src].Get(), D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX};
            src.SubresourceIndex = c.src_sub;
            list->CopyTextureRegion(&dst, 0, 0, 0, &src, &c.box);
            break;
        }
        case Copy::texture_to_texture: {
            D3D12_TEXTURE_COPY_LOCATION dst{res[c.dst].Get(), D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX};
            dst.SubresourceIndex = c.dst_sub;
            D3D12_TEXTURE_COPY_LOCATION src{res[c.src].Get(), D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX};
            src.SubresourceIndex = c.src_sub;
            list->CopyTextureRegion(&dst, c.dst_x, c.dst_y, 0, &src, &c.box);
            break;
        }
        case Copy::whole_resource:
            list->CopyResource(res[c.dst].Get(), res[c.src].Get());
            break;
        }
    }
    if (!submit()) return fail("program timed out");

    // Read every resource back: buffers whole, texture subresources at the device's own footprints.
    UINT64 rb_offset = 0;
    struct Region { UINT64 offset; UINT rows; UINT64 row_bytes; UINT pitch; };
    std::vector<Region> regions;
    for (UINT i = 0; i < kBuffers; ++i) {
        transition(i, D3D12_RESOURCE_STATE_COPY_SOURCE);
        list->CopyBufferRegion(readback.Get(), rb_offset, res[i].Get(), 0, kBufferBytes);
        regions.push_back({rb_offset, 1, kBufferBytes, 0});
        rb_offset += kBufferBytes;
    }
    for (UINT t = 0; t < kTextures; ++t) {
        const UINT r = kBuffers + t;
        transition(r, D3D12_RESOURCE_STATE_COPY_SOURCE);
        const auto d = texture_desc(p, t);
        const UINT subs = p.shapes[t].mips * p.shapes[t].layers;
        std::vector<D3D12_PLACED_SUBRESOURCE_FOOTPRINT> fp(subs);
        std::vector<UINT> rows(subs);
        std::vector<UINT64> row_bytes(subs);
        UINT64 total = 0;
        device->GetCopyableFootprints(&d, 0, subs, rb_offset, fp.data(), rows.data(), row_bytes.data(), &total);
        if (rb_offset + total > kStagingBytes) return fail("readback too small");
        for (UINT s = 0; s < subs; ++s) {
            D3D12_TEXTURE_COPY_LOCATION dst{readback.Get(), D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT};
            dst.PlacedFootprint = fp[s];
            D3D12_TEXTURE_COPY_LOCATION src{res[r].Get(), D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX};
            src.SubresourceIndex = s;
            list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
            regions.push_back({fp[s].Offset, rows[s], row_bytes[s], fp[s].Footprint.RowPitch});
        }
        rb_offset = (rb_offset + total + 511) / 512 * 512;
    }
    if (!submit()) return fail("readback timed out");
    std::uint8_t* back = nullptr;
    const D3D12_RANGE range{0, SIZE_T(kStagingBytes)};
    if (FAILED(readback->Map(0, &range, reinterpret_cast<void**>(&back)))) return fail("map readback");
    for (const Region& rg : regions)
        for (UINT row = 0; row < rg.rows; ++row) {
            const std::uint8_t* from = back + rg.offset + UINT64(row) * rg.pitch;
            out.insert(out.end(), from, from + rg.row_bytes);
        }
    readback->Unmap(0, nullptr);
    return out;
}

void print_program(const Program& p) {
    size_t brk = 0;
    for (size_t i = 0; i < p.copies.size(); ++i) {
        if (brk < p.list_breaks.size() && p.list_breaks[brk] == i) {
            ++brk;
            std::printf("  -- new command list\n");
        }
        std::printf("  %s\n", describe(p, p.copies[i]).c_str());
    }
}

void print_shapes(const Program& p) {
    std::printf("%s, textures %ux%ux%u m%u / %ux%ux%u m%u / %ux%ux%u m%u", p.format.name, p.shapes[0].width,
                p.shapes[0].height, p.shapes[0].layers, p.shapes[0].mips, p.shapes[1].width, p.shapes[1].height,
                p.shapes[1].layers, p.shapes[1].mips, p.shapes[2].width, p.shapes[2].height, p.shapes[2].layers,
                p.shapes[2].mips);
}

}  // namespace

int main(int argc, char** argv) {
    unsigned seeds = 30, start = 1, reduce = 0;
    bool verbose = false, hardware = false, reducing = false;
    for (int i = 1; i < argc; ++i) {
        // A count is decimal digits that fit in 32 bits. Anything else (a missing value, the next option,
        // "-1", "12x") is a usage error: "--seeds --verbose" used to read as 0 seeds and pass.
        bool bad = false;
        const auto number = [&](unsigned& out) {
            const char* s = i + 1 < argc ? argv[i + 1] : "";
            char* end = nullptr;
            const unsigned long long v = std::strtoull(s, &end, 10);
            if (*s < '0' || *s > '9' || *end || v > 0xffffffffull) {
                bad = true;
                return;
            }
            out = unsigned(v);
            ++i;
        };
        if (!std::strcmp(argv[i], "--seeds")) number(seeds);
        else if (!std::strcmp(argv[i], "--start")) number(start);
        else if (!std::strcmp(argv[i], "--reduce")) { number(reduce); reducing = true; }
        else if (!std::strcmp(argv[i], "--hardware")) hardware = true;
        else if (!std::strcmp(argv[i], "--verbose")) verbose = true;
        else {
            std::fprintf(stderr, "unknown option %s\n", argv[i]);
            return 2;
        }
        if (bad) {
            std::fprintf(stderr, "%s needs a decimal count\n", argv[i]);
            return 2;
        }
    }
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    if (GetProcAddress(GetModuleHandleW(L"ntdll.dll"), "wine_get_version")) {
        std::puts("SKIP: Wine's D3D12 is not a reference");
        return 77;
    }
    ComPtr<ID3D12Device> layer;
    if (FAILED(rt::d3d12::create_vulkan_device_standalone(IID_PPV_ARGS(&layer)))) {
        std::puts("SKIP: no Vulkan device");
        return 77;
    }
    std::wstring name;
    ComPtr<ID3D12Device> host = reference_device(hardware, name);
    if (!host) {
        std::puts("SKIP: no reference D3D12 device");
        return 77;
    }
    const auto differs = [&](const Program& q, std::uint64_t seed, std::vector<std::uint8_t>* theirs_out,
                             std::vector<std::uint8_t>* ours_out) {
        std::string he, le;
        auto theirs = run(host.Get(), q, seed, he), ours = run(layer.Get(), q, seed, le);
        const bool result = !theirs.empty() && !ours.empty() && theirs != ours;
        if (theirs_out) *theirs_out = std::move(theirs);
        if (ours_out) *ours_out = std::move(ours);
        return result;
    };

    if (reducing) {
        // Drop copies, then list breaks, one at a time while the layer still differs: the smallest program left.
        Program p = generate(reduce);
        if (!differs(p, reduce, nullptr, nullptr)) {
            std::printf("seed %u does not differ from %ls\n", reduce, name.c_str());
            return 0;
        }
        for (bool progress = true; progress;) {
            progress = false;
            for (size_t i = 0; i < p.copies.size(); ++i) {
                Program q = p;
                q.copies.erase(q.copies.begin() + i);
                q.list_breaks.clear();
                for (UINT b : p.list_breaks)
                    if (b != i) q.list_breaks.push_back(b > i ? b - 1 : b);
                if (differs(q, reduce, nullptr, nullptr)) {
                    p = q;
                    progress = true;
                    --i;
                }
            }
            for (size_t k = 0; k < p.list_breaks.size(); ++k) {
                Program q = p;
                q.list_breaks.erase(q.list_breaks.begin() + k);
                if (differs(q, reduce, nullptr, nullptr)) {
                    p = q;
                    progress = true;
                    --k;
                }
            }
        }
        std::printf("seed %u reduced (against %ls): ", reduce, name.c_str());
        print_shapes(p);
        std::printf("\n");
        print_program(p);
        std::vector<std::uint8_t> theirs, ours;
        differs(p, reduce, &theirs, &ours);
        unsigned shown = 0;
        for (size_t i = 0; i < theirs.size() && shown < 8; ++i)
            if (theirs[i] != ours[i])
                std::printf("  byte %zu: reference %02x layer %02x\n", i, theirs[i], ours[i]), ++shown;
        return 0;
    }

    unsigned failures = 0;
    for (unsigned run_index = 0; run_index < seeds; ++run_index) {  // counted from 0: start + seeds may wrap
        const unsigned seed = start + run_index;
        const Program p = generate(seed);
        std::vector<std::uint8_t> theirs, ours;
        std::string host_error, layer_error;
        theirs = run(host.Get(), p, seed, host_error);
        if (theirs.empty()) {
            std::printf("seed %u: the reference run failed (%s)\n", seed, host_error.c_str());
            ++failures;
            continue;
        }
        ours = run(layer.Get(), p, seed, layer_error);
        if (ours.empty()) {
            std::printf("seed %u: the layer run failed (%s)\n", seed, layer_error.c_str());
            ++failures;
            continue;
        }
        if (theirs == ours) {
            if (verbose) std::printf("seed %u: %s, %zu copies, identical\n", seed, p.format.name, p.copies.size());
            continue;
        }
        ++failures;
        size_t first = 0, count = 0;
        for (size_t i = 0; i < theirs.size() && i < ours.size(); ++i)
            if (theirs[i] != ours[i]) {
                if (!count) first = i;
                ++count;
            }
        std::printf("MISMATCH seed %u: ", seed);
        print_shapes(p);
        std::printf(", %zu of %zu bytes differ (first at %zu); program (--reduce %u shrinks it):\n", count,
                    theirs.size(), first, seed);
        print_program(p);
    }
    std::printf("copy fuzz (layer against %ls): %u seeds, %u differ\n", name.c_str(), seeds, failures);
    return failures ? 1 : 0;
}
