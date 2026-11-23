#pragma once

// The xg_xs.dll texture computer's model (shims/xg_xs.cpp), with no Windows dependency: the description the title
// passes, the layout structure it reads back, and the planes of a description laid out by gfx10_layout.h.
//
// Measured in the retail executable (1.0.2.1), read statically:
// - The description is a D3D12_RESOURCE_DESC (0x38 bytes). The title's builder (RVA 0x4ecc940) writes dimension 2/3/4,
//   alignment 0, width, height, array size or depth, mips, the DXGI format, the sample count, Quality 0, and
//   0x100 | mode in Layout: 0x100 | the swizzle mode its texture record holds, 0x10A (SW_64KB_D) for a texture its
//   record marks sparse, 0x102 (SW_256B_D) for one kind of small 2D texture, or XGSuggestSwizzleMode's answer when the
//   record says 0xFF. Flags carries console bits above the PC's.
// - The layout structure is 0x1B38 bytes or more: callers give it 0x1B40 bytes of stack, copy 0x6C8-byte planes from
//   +0x18 and index 0x70-byte mips from a plane's +0x38. What they read: a mip's +0x00 (size), +0x08 (offset of
//   array slice 0), +0x10 (one 2D slice) and +0x1C (row pitch in bytes), and a plane's +0x10, the bytes between
//   array slices (slice s of mip m is at s * plane+0x10 + mip m's +0x08).
// The other fields carry what their names say, in the order of the Xbox One XDK's XG_MIPLEVEL_LAYOUT; nothing read
// them yet.

#include "gfx10_layout.h"
#include "resource_tiling.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace leonida::xg {

struct ResourceDesc {
    std::uint32_t dimension = 0;  // D3D12_RESOURCE_DIMENSION: 1 buffer, 2 1D, 3 2D, 4 3D
    std::uint32_t pad0 = 0;
    std::uint64_t alignment = 0;
    std::uint64_t width = 0;
    std::uint32_t height = 0;
    std::uint16_t depth_or_array_size = 0;
    std::uint16_t mip_levels = 0;
    std::uint32_t format = 0;
    std::uint32_t sample_count = 0;
    std::uint32_t sample_quality = 0;
    std::uint32_t layout = 0;     // 0x100 | swizzle mode on the console
    std::uint32_t flags = 0;
    std::uint32_t pad1 = 0;
};
static_assert(sizeof(ResourceDesc) == 0x38);
static_assert(offsetof(ResourceDesc, format) == 0x20 && offsetof(ResourceDesc, layout) == 0x2c);

struct MipLevelLayout {
    std::uint64_t size_bytes = 0;            // +0x00
    std::uint64_t offset_bytes = 0;          // +0x08
    std::uint64_t slice_2d_size_bytes = 0;   // +0x10
    std::uint32_t pitch_pixels = 0;          // +0x18
    std::uint32_t pitch_bytes = 0;           // +0x1C
    std::uint32_t alignment_bytes = 0;       // +0x20
    std::uint32_t padded_width_elements = 0;
    std::uint32_t padded_height_elements = 0;
    std::uint32_t padded_depth_or_array_size = 0;
    std::uint32_t width_elements = 0;        // +0x30
    std::uint32_t height_elements = 0;
    std::uint32_t depth_or_array_size = 0;
    std::uint32_t sample_count = 0;
    std::uint32_t swizzle_mode = 0;          // +0x40
    std::uint32_t reserved[11] = {};
};
static_assert(sizeof(MipLevelLayout) == 0x70);
static_assert(offsetof(MipLevelLayout, slice_2d_size_bytes) == 0x10 && offsetof(MipLevelLayout, pitch_bytes) == 0x1c);

inline constexpr std::uint32_t kMaxPlanes = 4;
inline constexpr std::uint32_t kMaxMips = 15;

struct PlaneLayout {
    std::uint32_t usage = 0;                 // +0x00: 0 colour, 1 depth, 2 stencil
    std::uint32_t pad0 = 0;
    std::uint64_t size_bytes = 0;            // +0x08
    std::uint64_t slice_bytes = 0;           // +0x10, measured: bytes between array slices
    std::uint64_t base_offset_bytes = 0;     // +0x18
    std::uint64_t base_alignment_bytes = 0;  // +0x20
    std::uint32_t bytes_per_element = 0;     // +0x28
    std::uint32_t pad1 = 0;
    std::uint64_t reserved = 0;
    MipLevelLayout mip[kMaxMips] = {};       // +0x38
};
static_assert(sizeof(PlaneLayout) == 0x6c8);
static_assert(offsetof(PlaneLayout, slice_bytes) == 0x10 && offsetof(PlaneLayout, mip) == 0x38);

struct ResourceLayout {
    std::uint64_t size_bytes = 0;
    std::uint64_t base_alignment_bytes = 0;
    std::uint32_t mip_levels = 0;
    std::uint32_t planes = 0;
    PlaneLayout plane[kMaxPlanes] = {};      // +0x18
};
static_assert(sizeof(ResourceLayout) == 0x1b38);

// D3D12_PACKED_MIP_INFO, D3D12_TILE_SHAPE and D3D12_SUBRESOURCE_TILING, which the title's slot 10 call passes.
struct PackedMipInfo {
    std::uint8_t standard_mips = 0;
    std::uint8_t packed_mips = 0;
    std::uint32_t tiles_for_packed_mips = 0;
    std::uint32_t start_tile = 0;
};
static_assert(sizeof(PackedMipInfo) == 12);
struct TileShape {
    std::uint32_t width = 0, height = 0, depth = 0;
};
struct SubresourceTiling {
    std::uint32_t width_in_tiles = 0;
    std::uint16_t height_in_tiles = 0;
    std::uint16_t depth_in_tiles = 0;
    std::uint32_t start_tile = 0;
};
static_assert(sizeof(SubresourceTiling) == 12);

struct FormatInfo {
    std::uint32_t element_bytes = 0;  // 0: not a format this model lays out
    std::uint32_t block = 1;          // 4 for block-compressed formats
    std::uint32_t stencil = 0;        // 1: a second plane of 1-byte stencil
};

// DXGI_FORMAT numbers from the public dxgiformat.h, including its Xbox entries (116-120, 189, 190). 121-124: the
// title treats them as block-compressed with the BC formats (RVA 0x4ecca85); taken as 16-byte 4x4 blocks, unmeasured.
constexpr FormatInfo format_info(std::uint32_t format) {
    if (format >= 1 && format <= 4) return {16};
    if (format >= 5 && format <= 8) return {12};
    if (format >= 9 && format <= 18) return {8};
    if (format >= 19 && format <= 22) return {4, 1, 1};   // R32G8X24 family: depth plane, stencil plane
    if (format >= 23 && format <= 43) return {4};
    if (format >= 44 && format <= 47) return {4, 1, 1};   // R24G8 family
    if (format >= 48 && format <= 59) return {2};
    if (format >= 60 && format <= 66) return {1};
    if (format >= 67 && format <= 69) return {4};
    if (format >= 70 && format <= 72) return {8, 4};      // BC1
    if (format >= 73 && format <= 78) return {16, 4};     // BC2, BC3
    if (format >= 79 && format <= 81) return {8, 4};      // BC4
    if (format >= 82 && format <= 84) return {16, 4};     // BC5
    if (format == 85 || format == 86) return {2};
    if (format >= 87 && format <= 93) return {4};
    if (format >= 94 && format <= 99) return {16, 4};     // BC6H, BC7
    if (format == 115) return {2};
    if (format == 116 || format == 117) return {4};
    if (format >= 118 && format <= 120) return {2, 1, 1}; // D16_UNORM_S8_UINT family
    if (format >= 121 && format <= 124) return {16, 4};
    if (format == 189) return {4};
    if (format == 190) return {1};
    return {};
}

inline constexpr std::uint32_t kLayoutBase = 0x100;

// The swizzle mode a description's Layout word names: 0x100 | mode on the console; the PC words read as row major
// (1) and the 64 KiB swizzles (2 undefined, 3 standard). UNKNOWN (0) and anything else: no mode (-1).
constexpr int layout_mode(std::uint32_t layout) {
    if (layout >= kLayoutBase && layout < kLayoutBase + 32 && gfx10::block_bytes(layout - kLayoutBase))
        return int(layout - kLayoutBase);
    switch (layout) {
    case 1: return gfx10::sw_linear;
    case 2: return gfx10::sw_64kb_d;
    case 3: return gfx10::sw_64kb_s;
    default: return -1;
    }
}

struct Surface {
    ResourceDesc desc;
    FormatInfo format;
    std::uint32_t mips = 0, layers = 1, planes = 1;
    std::uint64_t size = 0, alignment = 0;
    std::uint64_t plane_base[2] = {};
    gfx10::SurfaceLayout plane[2];
};

// Mips the description asks for: its count, or the full chain for 0.
constexpr std::uint32_t mip_count(const ResourceDesc& d) {
    if (d.mip_levels) return d.mip_levels;
    std::uint64_t largest = d.width;
    if (d.dimension >= 3) largest = std::max<std::uint64_t>(largest, d.height);
    if (d.dimension == 4) largest = std::max<std::uint64_t>(largest, d.depth_or_array_size);
    std::uint32_t mips = 1;
    while (largest > 1) {
        largest >>= 1;
        ++mips;
    }
    return mips;
}

// The swizzle mode suggested for a description: linear for 1D textures and elements the swizzles do not take, else
// SW_64KB_D when a slice of mip 0 fills a 64 KiB block, SW_4KB_S below that. Both are the modes the retail texture
// pools use (docs/textures.md), so the title's data meets the bit order measured for them.
constexpr std::uint32_t suggest_mode(const ResourceDesc& d) {
    const FormatInfo f = format_info(d.format);
    if (d.dimension != 3 && d.dimension != 4) return gfx10::sw_linear;
    if (!f.element_bytes || (f.element_bytes & (f.element_bytes - 1)) || f.element_bytes > 16)
        return gfx10::sw_linear;
    const std::uint64_t w = (d.width + f.block - 1) / f.block, h = (std::uint64_t(d.height) + f.block - 1) / f.block;
    const std::uint64_t samples = d.sample_count ? d.sample_count : 1;
    return w * h * f.element_bytes * samples >= 65536 ? gfx10::sw_64kb_d : gfx10::sw_4kb_s;
}

// Lays the description out: false for a dimension, format, mode or size this model has no answer for.
inline bool surface(const ResourceDesc& d, Surface& out) {
    out = {};
    Surface s;
    s.desc = d;
    s.format = format_info(d.format);
    if (d.dimension < 2 || d.dimension > 4 || !s.format.element_bytes || !d.width) return false;
    int mode = layout_mode(d.layout);
    if (mode < 0) mode = int(suggest_mode(d));
    if (d.dimension == 2) mode = gfx10::sw_linear;
    // A 12-byte element has no swizzle on GFX10.
    if (s.format.element_bytes == 12) mode = gfx10::sw_linear;
    s.mips = mip_count(d);
    if (s.mips > kMaxMips) return false;
    const bool volume = d.dimension == 4;
    s.layers = volume ? 1 : std::max<std::uint32_t>(1, d.depth_or_array_size);
    s.planes = s.format.stencil ? 2 : 1;
    std::uint64_t cursor = 0;
    for (std::uint32_t p = 0; p < s.planes; ++p) {
        gfx10::SurfaceDesc g;
        g.mode = std::uint32_t(mode);
        g.element_bytes = p ? 1 : s.format.element_bytes;
        g.block = p ? 1 : s.format.block;
        g.samples = d.sample_count ? d.sample_count : 1;
        g.width = d.width;
        g.height = d.dimension == 2 ? 1 : std::max<std::uint32_t>(1, d.height);
        g.depth = volume ? std::max<std::uint32_t>(1, d.depth_or_array_size) : 1;
        g.layers = s.layers;
        g.mips = s.mips;
        g.volume = volume;
        if (!gfx10::surface_layout(g, s.plane[p])) return false;
        const std::uint64_t align = s.plane[p].alignment;
        cursor = (cursor + align - 1) / align * align;
        s.plane_base[p] = cursor;
        cursor += s.plane[p].size;
        s.alignment = std::max(s.alignment, align);
    }
    s.size = (cursor + s.alignment - 1) / s.alignment * s.alignment;
    out = s;
    return true;
}

// The structure slot 2 (GetResourceLayout) fills.
inline void resource_layout(const Surface& s, ResourceLayout& out) {
    out = {};
    out.size_bytes = s.size;
    out.base_alignment_bytes = s.alignment;
    out.mip_levels = s.mips;
    out.planes = s.planes;
    const bool depth = s.format.stencil != 0;
    for (std::uint32_t p = 0; p < s.planes; ++p) {
        const gfx10::SurfaceLayout& g = s.plane[p];
        PlaneLayout& plane = out.plane[p];
        plane.usage = depth ? p + 1 : 0;
        plane.size_bytes = g.size;
        plane.slice_bytes = g.slice_size;
        plane.base_offset_bytes = s.plane_base[p];
        plane.base_alignment_bytes = g.alignment;
        plane.bytes_per_element = g.element_bytes;
        const std::uint32_t block = p ? 1 : s.format.block;
        for (std::uint32_t m = 0; m < s.mips && m < kMaxMips; ++m) {
            const gfx10::MipPlacement& mip = g.mip[m];
            MipLevelLayout& out_mip = plane.mip[m];
            out_mip.size_bytes = mip.size;
            out_mip.offset_bytes = s.plane_base[p] + mip.offset;
            out_mip.slice_2d_size_bytes = mip.slice_2d;
            out_mip.pitch_pixels = mip.padded_width * block;
            out_mip.pitch_bytes = mip.pitch_bytes;
            out_mip.alignment_bytes = g.block_bytes;
            out_mip.padded_width_elements = mip.padded_width;
            out_mip.padded_height_elements = mip.padded_height;
            out_mip.padded_depth_or_array_size = g.volume ? mip.padded_depth : g.layers;
            out_mip.width_elements = mip.width;
            out_mip.height_elements = mip.height;
            out_mip.depth_or_array_size = g.volume ? mip.depth : g.layers;
            out_mip.sample_count = s.desc.sample_count ? s.desc.sample_count : 1;
            out_mip.swizzle_mode = g.mode;
        }
    }
}

// Slot 10 (GetResourceTiling): the PC device's 64 KiB tiling of the description (src/resource_tiling.h), the same
// numbers the Vulkan layer's GetResourceTiling gives. False for a description that has none.
inline bool resource_tiling(const Surface& s, leonida::ResourceTiling& out) {
    leonida::TilingDesc t;
    t.kind = s.desc.dimension == 4 ? leonida::TiledKind::texture3d : leonida::TiledKind::texture2d;
    t.width = s.desc.width;
    t.height = s.desc.dimension == 2 ? 1 : std::max<std::uint32_t>(1, s.desc.height);
    t.depth_or_layers = std::max<std::uint32_t>(1, s.desc.depth_or_array_size);
    t.mips = s.mips;
    t.element_bytes = s.format.element_bytes;
    t.block = s.format.block;
    t.samples = s.desc.sample_count ? s.desc.sample_count : 1;
    return leonida::resource_tiling(t, out);
}

}  // namespace leonida::xg
