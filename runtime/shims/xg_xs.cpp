// xg_xs.dll: the console's texture layout library (XGCreateTextureComputer, XGSuggestSwizzleMode).
//
// What the retail executable (1.0.2.1) shows, measured from its call sites:
//   * The description it passes is a D3D12_RESOURCE_DESC (0x38 bytes) built at RVA 0x4ecc7f0: Dimension (2 = 1D,
//     3 = 2D, 4 = 3D), Alignment 0, Width, Height, DepthOrArraySize, MipLevels, DXGI format, SampleDesc, Layout (+0x2c,
//     filled by XGSuggestSwizzleMode(desc, 0, &desc.Layout) or set to 0x100 | mode by the title) and Flags (+0x30).
//   * XGCreateTextureComputer(desc, &computer) returns an object whose table the title calls at
//     +0x08 (Release, after every use), +0x10 (GetResourceLayout(layout*), RVA 0x979c9f and 0x5121a84),
//     +0x40 (CopyIntoSubresource(base, plane, subresource, linear data, row pitch, slice pitch), RVA 0x16f71fb) and
//     +0x50 (the arguments of ID3D12Device::GetResourceTiling, RVA 0x344283, only for reserved resources).
//   * The layout it reads: a 0x18-byte header, then up to four planes of 0x6c8 bytes (copied whole at RVA 0x5121aa9),
//     each a 0x38-byte header and fifteen 0x70-byte mip records; the title reads a mip's +0x00 (size), +0x10 (2D
//     slice size) and +0x1c (row pitch) (RVA 0x979ca5).
//
// What this is: a LINEAR layout. Every resource this runtime makes is read back by this runtime's own D3D12 layer, so
// the layout only has to agree with itself; XGSuggestSwizzleMode suggests mode 0 (0x100 | SW_LINEAR) to say so.
// Rows are padded to 256 bytes and subresources to 512 (D3D12's placed-footprint rules), mips follow each other and
// a mip holds all its array slices. ASSUMPTION: the field names of the 0x38-byte plane header and of the mip record
// past +0x1c follow the Xbox One XDK's XG_PLANE_LAYOUT / XG_MIPLEVEL_LAYOUT order; the title is not seen reading them.
// Methods the title is not seen calling log their slot and return E_NOTIMPL.
#include <windows.h>
#include <d3d12.h>

#include <algorithm>
#include <cstddef>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <new>

#include "shims.h"

namespace {

struct FormatShape {
    std::uint32_t block_bytes;  // bytes per element (a pixel, or a 4x4 block)
    std::uint32_t block_dim;    // 1, or 4 for block-compressed formats
};

FormatShape format_shape(std::uint32_t f) {
    if (f >= 1 && f <= 4) return {16, 1};
    if (f >= 5 && f <= 8) return {12, 1};
    if (f >= 9 && f <= 22) return {8, 1};
    if (f >= 23 && f <= 47) return {4, 1};
    if (f >= 48 && f <= 59) return {2, 1};
    if (f >= 60 && f <= 66) return {1, 1};
    if (f >= 67 && f <= 69) return {4, 1};
    if ((f >= 70 && f <= 72) || (f >= 79 && f <= 81)) return {8, 4};             // BC1, BC4
    if ((f >= 73 && f <= 78) || (f >= 82 && f <= 84) || (f >= 94 && f <= 99)) return {16, 4};  // BC2/3/5/6H/7
    if (f == 85 || f == 86 || f == 115) return {2, 1};
    if (f >= 87 && f <= 93) return {4, 1};
    return {4, 1};
}

constexpr std::uint64_t kRowAlign = 256;          // D3D12_TEXTURE_DATA_PITCH_ALIGNMENT
constexpr std::uint64_t kSubresourceAlign = 512;  // D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT
constexpr std::uint64_t kResourceAlign = 64 * 1024;
constexpr std::uint32_t kMaxMips = 15;

std::uint64_t align_up(std::uint64_t v, std::uint64_t a) { return (v + a - 1) / a * a; }

#pragma pack(push, 1)
struct MipRecord {  // 0x70 bytes
    std::uint64_t size_bytes;          // +0x00 (all slices)
    std::uint64_t offset_bytes;        // +0x08
    std::uint64_t slice2d_size_bytes;  // +0x10
    std::uint32_t pitch_pixels;        // +0x18
    std::uint32_t pitch_bytes;         // +0x1c
    std::uint32_t alignment_bytes;     // +0x20
    std::uint32_t padded_width;        // +0x24
    std::uint32_t padded_height;       // +0x28
    std::uint32_t padded_depth;        // +0x2c
    std::uint32_t width_elements;      // +0x30
    std::uint32_t height_elements;     // +0x34
    std::uint32_t depth_or_array;      // +0x38
    std::uint32_t sample_count;        // +0x3c
    std::uint32_t tile_mode;           // +0x40
    std::uint8_t rest[0x70 - 0x44];
};
struct PlaneLayout {  // 0x6c8 bytes
    std::uint32_t usage;                 // +0x00
    std::uint32_t pad0;
    std::uint64_t size_bytes;            // +0x08
    std::uint64_t base_offset_bytes;     // +0x10
    std::uint64_t base_alignment_bytes;  // +0x18
    std::uint32_t bytes_per_element;     // +0x20
    std::uint8_t rest[0x38 - 0x24];
    MipRecord mip[kMaxMips];
};
struct ResourceLayout {  // what GetResourceLayout writes: 0x18 + 4 planes (+ a dimension word)
    std::uint64_t size_bytes;
    std::uint64_t base_alignment_bytes;
    std::uint32_t mip_levels;
    std::uint32_t planes;
    PlaneLayout plane[4];
    std::uint32_t dimension;
};
#pragma pack(pop)
static_assert(sizeof(MipRecord) == 0x70);
static_assert(sizeof(PlaneLayout) == 0x6c8);
static_assert(offsetof(ResourceLayout, plane) == 0x18);

struct Computer;
using Slot = std::uintptr_t (*)(Computer*, std::uintptr_t, std::uintptr_t, std::uintptr_t, std::uintptr_t,
                                std::uintptr_t, std::uintptr_t);

struct Computer {
    void* const* vtable;
    std::atomic<long> refs{1};
    D3D12_RESOURCE_DESC desc{};
    ResourceLayout layout{};
    std::uint32_t array_slices = 1;  // slices that are subresources (1 for 3D)
};

void build_layout(Computer& c) {
    const D3D12_RESOURCE_DESC& d = c.desc;
    const FormatShape shape = format_shape(static_cast<std::uint32_t>(d.Format));
    std::uint32_t mips = d.MipLevels;
    const std::uint64_t largest = std::max<std::uint64_t>({d.Width, d.Height, d.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D ? d.DepthOrArraySize : 1u});
    std::uint32_t full = 1;
    while ((largest >> full) > 0) ++full;
    if (!mips || mips > full) mips = full;
    mips = std::min(mips, kMaxMips);
    const bool is3d = d.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D;
    c.array_slices = is3d ? 1u : std::max<std::uint32_t>(d.DepthOrArraySize, 1);
    ResourceLayout& L = c.layout;
    L = {};
    L.mip_levels = mips;
    L.planes = 1;
    L.base_alignment_bytes = kResourceAlign;
    L.dimension = static_cast<std::uint32_t>(d.Dimension);
    PlaneLayout& P = L.plane[0];
    P.base_alignment_bytes = kResourceAlign;
    P.bytes_per_element = shape.block_bytes;
    const std::uint32_t samples = std::max<std::uint32_t>(d.SampleDesc.Count, 1);
    std::uint64_t offset = 0;
    for (std::uint32_t m = 0; m < mips; ++m) {
        const std::uint64_t w = std::max<std::uint64_t>(d.Width >> m, 1);
        const std::uint32_t h = std::max<std::uint32_t>(d.Height >> m, 1);
        const std::uint32_t z = is3d ? std::max<std::uint32_t>(d.DepthOrArraySize >> m, 1) : c.array_slices;
        const std::uint32_t we = static_cast<std::uint32_t>((w + shape.block_dim - 1) / shape.block_dim);
        const std::uint32_t he = (h + shape.block_dim - 1) / shape.block_dim;
        const std::uint64_t pitch = align_up(std::uint64_t{we} * shape.block_bytes * samples, kRowAlign);
        const std::uint64_t slice = align_up(pitch * he, kSubresourceAlign);
        MipRecord& r = P.mip[m];
        r.offset_bytes = offset;
        r.slice2d_size_bytes = slice;
        r.size_bytes = slice * z;
        r.pitch_bytes = static_cast<std::uint32_t>(pitch);
        r.pitch_pixels = static_cast<std::uint32_t>(pitch / shape.block_bytes / samples * shape.block_dim);
        r.alignment_bytes = static_cast<std::uint32_t>(kSubresourceAlign);
        r.padded_width = static_cast<std::uint32_t>(pitch / shape.block_bytes / samples);
        r.padded_height = he;
        r.padded_depth = z;
        r.width_elements = we;
        r.height_elements = he;
        r.depth_or_array = z;
        r.sample_count = samples;
        r.tile_mode = 0;  // linear
        offset += r.size_bytes;
    }
    P.size_bytes = align_up(offset, kResourceAlign);
    L.size_bytes = P.size_bytes;
}

std::uintptr_t xg_addref(Computer* c, std::uintptr_t, std::uintptr_t, std::uintptr_t, std::uintptr_t, std::uintptr_t,
                         std::uintptr_t) {
    return static_cast<std::uintptr_t>(++c->refs);
}

std::uintptr_t xg_release(Computer* c, std::uintptr_t, std::uintptr_t, std::uintptr_t, std::uintptr_t, std::uintptr_t,
                          std::uintptr_t) {
    const long left = --c->refs;
    if (!left) delete c;
    return static_cast<std::uintptr_t>(left);
}

std::uintptr_t hr(HRESULT h) { return static_cast<std::uintptr_t>(static_cast<std::uint32_t>(h)); }

// +0x10: GetResourceLayout(layout*).
std::uintptr_t xg_get_resource_layout(Computer* c, std::uintptr_t out, std::uintptr_t, std::uintptr_t, std::uintptr_t,
                                      std::uintptr_t, std::uintptr_t) {
    if (!out) return hr(E_POINTER);
    std::memcpy(reinterpret_cast<void*>(out), &c->layout, sizeof(c->layout));
    return hr(S_OK);
}

// Copies one subresource between the layout at `tiled` and linear rows (`row_pitch`, `slice_pitch` apart).
HRESULT copy_subresource(Computer* c, std::uint8_t* tiled, std::uint32_t plane, std::uint32_t subresource,
                         std::uint8_t* linear, std::uint32_t row_pitch, std::uint32_t slice_pitch, bool into) {
    if (!tiled || !linear || plane != 0) return E_INVALIDARG;
    const std::uint32_t mips = c->layout.mip_levels;
    const std::uint32_t mip = subresource % mips;
    const std::uint32_t slice = subresource / mips;
    if (slice >= c->array_slices) return E_INVALIDARG;
    const MipRecord& r = c->layout.plane[0].mip[mip];
    const bool is3d = c->desc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D;
    const std::uint32_t depth = is3d ? r.depth_or_array : 1;
    const std::uint64_t row_bytes = std::uint64_t{r.width_elements} * c->layout.plane[0].bytes_per_element *
                                    std::max<std::uint32_t>(r.sample_count, 1);
    const std::uint64_t copy = std::min<std::uint64_t>(row_bytes, row_pitch ? row_pitch : row_bytes);
    std::uint8_t* base = tiled + r.offset_bytes + (is3d ? 0 : std::uint64_t{slice} * r.slice2d_size_bytes);
    for (std::uint32_t z = 0; z < depth; ++z) {
        for (std::uint32_t y = 0; y < r.height_elements; ++y) {
            std::uint8_t* t = base + std::uint64_t{z} * r.slice2d_size_bytes + std::uint64_t{y} * r.pitch_bytes;
            std::uint8_t* l = linear + std::uint64_t{z} * slice_pitch + std::uint64_t{y} * row_pitch;
            if (into) std::memcpy(t, l, copy);
            else std::memcpy(l, t, copy);
        }
    }
    return S_OK;
}

// +0x40: CopyIntoSubresource(tiled base, plane, subresource, linear data, row pitch, slice pitch).
std::uintptr_t xg_copy_into(Computer* c, std::uintptr_t base, std::uintptr_t plane, std::uintptr_t sub,
                            std::uintptr_t linear, std::uintptr_t row_pitch, std::uintptr_t slice_pitch) {
    return hr(copy_subresource(c, reinterpret_cast<std::uint8_t*>(base), static_cast<std::uint32_t>(plane),
                               static_cast<std::uint32_t>(sub), reinterpret_cast<std::uint8_t*>(linear),
                               static_cast<std::uint32_t>(row_pitch), static_cast<std::uint32_t>(slice_pitch), true));
}

// +0x48: CopyFromSubresource(linear data, plane, subresource, tiled base, row pitch, slice pitch). ASSUMPTION: the
// mirror of +0x40 (not seen called).
std::uintptr_t xg_copy_from(Computer* c, std::uintptr_t linear, std::uintptr_t plane, std::uintptr_t sub,
                            std::uintptr_t base, std::uintptr_t row_pitch, std::uintptr_t slice_pitch) {
    std::fprintf(stderr, "[rt] xg_xs: texture computer slot 9 (CopyFromSubresource?) called\n");
    return hr(copy_subresource(c, reinterpret_cast<std::uint8_t*>(base), static_cast<std::uint32_t>(plane),
                               static_cast<std::uint32_t>(sub), reinterpret_cast<std::uint8_t*>(linear),
                               static_cast<std::uint32_t>(row_pitch), static_cast<std::uint32_t>(slice_pitch), false));
}

// +0x50: the arguments of ID3D12Device::GetResourceTiling: (UINT* tiles, D3D12_PACKED_MIP_INFO*, D3D12_TILE_SHAPE*,
// UINT* subresource count (in/out), UINT first, D3D12_SUBRESOURCE_TILING*). The linear layout has no standard tile
// shape, so every mip is reported packed into the resource's 64 KiB tiles.
std::uintptr_t xg_get_resource_tiling(Computer* c, std::uintptr_t tiles, std::uintptr_t packed, std::uintptr_t shape,
                                      std::uintptr_t count, std::uintptr_t first, std::uintptr_t out) {
    const auto total = static_cast<UINT>(c->layout.size_bytes / kResourceAlign);
    if (tiles) *reinterpret_cast<UINT*>(tiles) = total;
    if (packed) {
        auto* p = reinterpret_cast<D3D12_PACKED_MIP_INFO*>(packed);
        p->NumStandardMips = 0;
        p->NumPackedMips = static_cast<UINT8>(c->layout.mip_levels);
        p->NumTilesForPackedMips = total;
        p->StartTileIndexInOverallResource = 0;
    }
    if (shape) *reinterpret_cast<D3D12_TILE_SHAPE*>(shape) = {0, 0, 0};
    if (count) {
        auto& n = *reinterpret_cast<UINT*>(count);
        const UINT subresources = c->layout.mip_levels * c->array_slices;
        const UINT start = static_cast<UINT>(first);
        const UINT available = start < subresources ? subresources - start : 0;
        n = std::min(n, available);
        auto* tilings = reinterpret_cast<D3D12_SUBRESOURCE_TILING*>(out);
        for (UINT i = 0; tilings && i < n; ++i) tilings[i] = {0, 0, 0, D3D12_PACKED_TILE};
    }
    return hr(S_OK);
}

template <int N>
std::uintptr_t xg_unknown(Computer*, std::uintptr_t, std::uintptr_t, std::uintptr_t, std::uintptr_t, std::uintptr_t,
                          std::uintptr_t) {
    static std::atomic<bool> said{false};
    if (!said.exchange(true))
        std::fprintf(stderr, "[rt] xg_xs: texture computer slot %d (+0x%x) is not implemented; E_NOTIMPL\n", N, N * 8);
    return hr(E_NOTIMPL);
}

void* const kComputerVtable[] = {
    reinterpret_cast<void*>(&xg_addref),
    reinterpret_cast<void*>(&xg_release),
    reinterpret_cast<void*>(&xg_get_resource_layout),
    reinterpret_cast<void*>(&xg_unknown<3>),
    reinterpret_cast<void*>(&xg_unknown<4>),
    reinterpret_cast<void*>(&xg_unknown<5>),
    reinterpret_cast<void*>(&xg_unknown<6>),
    reinterpret_cast<void*>(&xg_unknown<7>),
    reinterpret_cast<void*>(&xg_copy_into),
    reinterpret_cast<void*>(&xg_copy_from),
    reinterpret_cast<void*>(&xg_get_resource_tiling),
    reinterpret_cast<void*>(&xg_unknown<11>),
    reinterpret_cast<void*>(&xg_unknown<12>),
    reinterpret_cast<void*>(&xg_unknown<13>),
    reinterpret_cast<void*>(&xg_unknown<14>),
    reinterpret_cast<void*>(&xg_unknown<15>),
};

HRESULT __stdcall XGCreateTextureComputer(const D3D12_RESOURCE_DESC* desc, void** out) {
    if (!out) return E_POINTER;
    *out = nullptr;
    if (!desc) return E_INVALIDARG;
    if (desc->Dimension < D3D12_RESOURCE_DIMENSION_TEXTURE1D || desc->Dimension > D3D12_RESOURCE_DIMENSION_TEXTURE3D ||
        !desc->Width)
        return E_INVALIDARG;
    auto* c = new (std::nothrow) Computer{kComputerVtable};
    if (!c) return E_OUTOFMEMORY;
    c->desc = *desc;
    build_layout(*c);
    static std::atomic<unsigned> logged{0};
    if (logged.fetch_add(1, std::memory_order_relaxed) < 6)
        std::fprintf(stderr, "[rt] xg_xs: texture computer: dimension %d %llux%ux%u mips %u format %u -> %llu bytes "
                             "(linear)\n",
                     static_cast<int>(desc->Dimension), static_cast<unsigned long long>(desc->Width), desc->Height,
                     static_cast<unsigned>(desc->DepthOrArraySize), c->layout.mip_levels,
                     static_cast<unsigned>(desc->Format), static_cast<unsigned long long>(c->layout.size_bytes));
    *out = c;
    return S_OK;
}

// XGSuggestSwizzleMode(desc, 0, UINT* layout) (RVA 0x4eccca2 passes &desc->Layout). Linear: 0x100 | 0.
HRESULT __stdcall XGSuggestSwizzleMode(const D3D12_RESOURCE_DESC* /*desc*/, std::uintptr_t /*reserved*/,
                                       std::uint32_t* layout) {
    if (!layout) return E_POINTER;
    *layout = 0x100;
    return S_OK;
}

}  // namespace

SHIM_TABLE(xg_xs,
    {"xg_xs.dll", "XGCreateTextureComputer", reinterpret_cast<void*>(&XGCreateTextureComputer)},
    {"xg_xs.dll", "XGSuggestSwizzleMode", reinterpret_cast<void*>(&XGSuggestSwizzleMode)},
);
