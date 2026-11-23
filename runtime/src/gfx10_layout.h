#pragma once

// Where every element of a GFX10 (RDNA2, Xbox Series) texture sits: mip placement, the mip tail, array slices and the
// address inside a swizzle block. No Windows or Vulkan dependency. The xg_xs.dll shim answers the title's layout
// questions from it (shims/xg_xs.cpp), so a layout the title computes and the data it copies agree.
//
// What is measured, and where:
// - Mips inside a slice: standalone mips padded to whole blocks and stored smallest first; a 4 KiB or 64 KiB mode
//   puts the mips that fit half a block wide and one block high, among the last 8 (4 KiB) or 12 (64 KiB), into one
//   shared tail block at offset 0, mip m of the tail at AMD's entry offset. Byte-exact against 356,413 allocation
//   spans of the retail pools and pixel-checked (docs/textures.md, "The layout is the AMD mip tail";
//   tools/tex_headers.py compact_surface_layout).
// - Array slices: slice-major, the whole mip chain of slice s at s * slice size. That is what the executable assumes
//   of XG's answer: its lock path and its view path both address slice s of mip m as
//   s * plane[+0x10] + mip[m].OffsetBytes (the executable at RVA 0x5121852 and 0x555eaf5). The pools' streaming
//   payloads are packed mip-major (docs/textures_arrays.md); that is the file layout, which the title copies.
// - The address inside a block: SW_4KB_S and RbPlus SW_64KB_D for 1/2/4/8/16-byte elements, from Mesa addrlib's
//   GFX10 pattern tables and pixel-checked on the pools (gfx10_tiling.h). SW_64KB_S is SW_4KB_S's twelve bits
//   continued, 256 B and 4 KiB D modes are SW_64KB_D's first bits, and 256 B S is SW_4KB_S's first eight: the
//   non-XOR GFX10 patterns of one family share their low bits.
// Not measured, and said so where they are used:
// - Z, R and the XOR (_X, _T) modes. Their block sizes are known, so their layouts are; the bit order inside a block
//   is not, and the D (or S) order of the same block size stands in (`exact_pattern` is false). Everything the
//   runtime writes and reads through this header agrees with itself; data tiled by the console's hardware in those
//   modes does not.
// - Volumes: each depth slice of a mip is tiled as a 2D slice (thin), mips smallest first, no tail.
// - Linear: rows padded to 256 bytes, mips largest first, each at a 256-byte boundary.

#include "gfx10_tiling.h"

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>

namespace leonida::gfx10 {

// AMD's ADDR_SW_MODE numbering, which the console's layout word carries as 0x100 | mode.
enum SwizzleMode : std::uint32_t {
    sw_linear = 0,
    sw_256b_s = 1,
    sw_256b_d = 2,
    sw_256b_r = 3,
    sw_4kb_z = 4,
    sw_4kb_s = 5,
    sw_4kb_d = 6,
    sw_4kb_r = 7,
    sw_64kb_z = 8,
    sw_64kb_s = 9,
    sw_64kb_d = 10,
    sw_64kb_r = 11,
    sw_64kb_z_t = 16,
    sw_64kb_s_t = 17,
    sw_64kb_d_t = 18,
    sw_64kb_r_t = 19,
    sw_4kb_z_x = 20,
    sw_4kb_s_x = 21,
    sw_4kb_d_x = 22,
    sw_4kb_r_x = 23,
    sw_64kb_z_x = 24,
    sw_64kb_s_x = 25,
    sw_64kb_d_x = 26,
    sw_64kb_r_x = 27,
};

// Bytes of one swizzle block, or 0 for a mode number GFX10 does not have (the VAR modes are not on RDNA2).
constexpr std::uint32_t block_bytes(std::uint32_t mode) {
    switch (mode) {
    case sw_linear: return 256;  // the row and mip alignment of a linear surface
    case sw_256b_s: case sw_256b_d: case sw_256b_r: return 256;
    case sw_4kb_z: case sw_4kb_s: case sw_4kb_d: case sw_4kb_r:
    case sw_4kb_z_x: case sw_4kb_s_x: case sw_4kb_d_x: case sw_4kb_r_x: return 4096;
    case sw_64kb_z: case sw_64kb_s: case sw_64kb_d: case sw_64kb_r:
    case sw_64kb_z_t: case sw_64kb_s_t: case sw_64kb_d_t: case sw_64kb_r_t:
    case sw_64kb_z_x: case sw_64kb_s_x: case sw_64kb_d_x: case sw_64kb_r_x: return 65536;
    default: return 0;
    }
}

// Whether the mode's in-block bit order is the measured one (S or D, without XOR).
constexpr bool exact_pattern(std::uint32_t mode) {
    switch (mode) {
    case sw_linear: case sw_256b_s: case sw_256b_d: case sw_4kb_s: case sw_4kb_d: case sw_64kb_s: case sw_64kb_d:
        return true;
    default: return false;
    }
}

// The S family (standard swizzle) for the mode, by bit order; D otherwise (D, R, Z and their XOR forms).
constexpr bool s_family(std::uint32_t mode) {
    return mode == sw_256b_s || mode == sw_4kb_s || mode == sw_64kb_s || mode == sw_64kb_s_t || mode == sw_4kb_s_x ||
           mode == sw_64kb_s_x;
}

struct SurfaceDesc {
    std::uint32_t mode = sw_linear;
    std::uint32_t element_bytes = 0;  // bytes of one texel, or of one compressed block
    std::uint32_t block = 1;          // texels per block side: 4 for BC formats
    std::uint32_t samples = 1;
    std::uint64_t width = 0;          // texels
    std::uint32_t height = 1;
    std::uint32_t depth = 1;          // a volume's depth
    std::uint32_t layers = 1;         // array slices (6 per cube)
    std::uint32_t mips = 1;
    bool volume = false;
};

inline constexpr std::uint32_t kMaxMips = 16;

struct MipPlacement {
    std::uint64_t offset = 0;    // slice 0, from the surface start; a tail mip's includes its entry offset
    std::uint64_t size = 0;      // every depth slice of a volume mip, one array slice otherwise
    std::uint64_t slice_2d = 0;  // one depth slice
    std::uint32_t width = 0, height = 0, depth = 1;  // in elements
    std::uint32_t padded_width = 0, padded_height = 0, padded_depth = 1;
    std::uint32_t pitch_bytes = 0;
    bool in_tail = false;
    std::uint64_t tail_block = 0;  // the tail block's offset in slice 0
    std::uint32_t tail_x = 0, tail_y = 0;  // the mip's origin inside the tail block, in elements
};

struct SurfaceLayout {
    std::uint32_t mode = sw_linear;
    std::uint32_t element_bytes = 0;  // times the sample count
    std::uint32_t block_bytes = 0;
    std::uint32_t block_bits = 0;
    std::uint32_t block_w = 1, block_h = 1;  // elements
    std::uint32_t mips = 0;
    std::uint32_t layers = 1;
    std::uint32_t first_tail = 0;     // == mips without a tail
    std::uint64_t slice_size = 0;     // bytes between array slices
    std::uint64_t size = 0;
    std::uint64_t alignment = 0;
    bool volume = false;
    bool exact = true;                // the in-block bit order is the measured one (or the surface is linear)
    std::array<int, 16> pattern{};    // in-block bits: x bit (v - 1) for v > 0, y bit (-v - 1) for v < 0
    std::array<MipPlacement, kMaxMips> mip{};
};

namespace layout_detail {

constexpr bool checked_mul(std::uint64_t a, std::uint64_t b, std::uint64_t& out) {
    if (a && b > std::numeric_limits<std::uint64_t>::max() / a) return false;
    out = a * b;
    return true;
}

constexpr bool checked_add(std::uint64_t a, std::uint64_t b, std::uint64_t& out) {
    if (b > std::numeric_limits<std::uint64_t>::max() - a) return false;
    out = a + b;
    return true;
}

constexpr std::uint64_t up(std::uint64_t value, std::uint64_t by) { return value ? 1 + (value - 1) / by : 0; }

}  // namespace layout_detail

// The surface's layout, or false for one it cannot describe (a mode GFX10 lacks, a swizzled element size other than
// 1/2/4/8/16 bytes, more than kMaxMips mips, or sizes that overflow).
inline bool surface_layout(const SurfaceDesc& d, SurfaceLayout& out) {
    using namespace layout_detail;
    out = {};
    const std::uint32_t block_size = block_bytes(d.mode);
    if (!block_size || !d.element_bytes || !d.width || !d.height || !d.depth || !d.layers || !d.mips ||
        d.mips > kMaxMips || !d.block || !d.samples || (d.volume && d.layers != 1))
        return false;
    const std::uint64_t element = std::uint64_t(d.element_bytes) * d.samples;
    if (element > 0xffffffffu) return false;
    SurfaceLayout l;
    l.mode = d.mode;
    l.element_bytes = std::uint32_t(element);
    l.block_bytes = block_size;
    l.block_bits = std::uint32_t(std::countr_zero(block_size));
    l.mips = d.mips;
    l.layers = d.layers;
    l.volume = d.volume;
    l.first_tail = d.mips;
    const std::uint64_t ew = up(d.width, d.block), eh = up(d.height, d.block);
    if (ew > 0xffffffffu || eh > 0xffffffffu) return false;

    const auto mip_extent = [&](std::uint32_t i, std::uint32_t& w, std::uint32_t& h, std::uint32_t& z) {
        w = std::uint32_t(std::max<std::uint64_t>(1, up(ew, std::uint64_t(1) << i)));
        h = std::uint32_t(std::max<std::uint64_t>(1, up(eh, std::uint64_t(1) << i)));
        z = d.volume ? std::max(1u, d.depth >> i) : 1u;
    };

    if (d.mode == sw_linear) {
        l.alignment = 256;
        std::uint64_t cursor = 0;
        for (std::uint32_t i = 0; i < d.mips; ++i) {
            MipPlacement& m = l.mip[i];
            mip_extent(i, m.width, m.height, m.depth);
            std::uint64_t row = 0, pitch = 0, slice = 0, size = 0;
            if (!checked_mul(m.width, element, row)) return false;
            pitch = up(row, 256) * 256;
            if (pitch > 0xffffffffu || !checked_mul(pitch, m.height, slice) || !checked_mul(slice, m.depth, size))
                return false;
            cursor = up(cursor, 256) * 256;
            m.offset = cursor;
            m.size = size;
            m.slice_2d = slice;
            m.pitch_bytes = std::uint32_t(pitch);
            m.padded_width = std::uint32_t(pitch / element);
            m.padded_height = m.height;
            m.padded_depth = m.depth;
            if (!checked_add(cursor, size, cursor)) return false;
        }
        l.slice_size = up(cursor, 256) * 256;
    } else {
        if (!std::has_single_bit(element) || element > 16) return false;
        const std::uint32_t log = std::uint32_t(std::countr_zero(element));
        const std::uint32_t elements = block_size >> log;
        const std::uint32_t lb = std::uint32_t(std::countr_zero(elements));
        l.block_w = 1u << ((lb + 1) / 2);
        l.block_h = elements / l.block_w;
        l.alignment = block_size;
        l.exact = exact_pattern(d.mode);
        for (std::uint32_t bit = 0; bit < 16; ++bit) {
            l.pattern[bit] = s_family(d.mode) ? (bit < 12 ? tiling_detail::patterns_4k[log][bit]
                                                          : tiling_detail::patterns_64k[log][bit])
                                              : tiling_detail::patterns_64k[log][bit];
        }
        // The tail: 4 KiB and 64 KiB blocks, 2D surfaces with more than one mip (docs/textures.md).
        const std::uint32_t max_tail = block_size == 65536 ? 12 : block_size == 4096 ? 8 : 0;
        if (max_tail && !d.volume && d.mips > 1) {
            for (std::uint32_t i = 0; i < d.mips; ++i) {
                std::uint32_t w, h, z;
                mip_extent(i, w, h, z);
                if (w <= l.block_w / 2 && h <= l.block_h && d.mips - i <= max_tail) {
                    l.first_tail = i;
                    break;
                }
            }
        }
        std::uint64_t cursor = l.first_tail < d.mips ? block_size : 0;
        for (std::uint32_t k = l.first_tail; k-- > 0;) {
            MipPlacement& m = l.mip[k];
            mip_extent(k, m.width, m.height, m.depth);
            const std::uint64_t pw = up(m.width, l.block_w) * l.block_w, ph = up(m.height, l.block_h) * l.block_h;
            std::uint64_t slice = 0, size = 0;
            if (pw > 0xffffffffu || ph > 0xffffffffu || !checked_mul(pw * ph, element, slice) ||
                !checked_mul(slice, m.depth, size))
                return false;
            m.padded_width = std::uint32_t(pw);
            m.padded_height = std::uint32_t(ph);
            m.padded_depth = m.depth;
            m.pitch_bytes = std::uint32_t(pw * element);
            m.offset = cursor;
            m.slice_2d = slice;
            m.size = size;
            if (!checked_add(cursor, size, cursor)) return false;
        }
        // AMD's tail entry offsets: entry m (0 for the smallest mip that fits) at m * 256 bytes up to entry 6, then
        // 16 << m; the entry's 256-byte micro-block sits at the Morton (x, y) of its offset (tools/tex_headers.py).
        static constexpr std::uint32_t micro[5][2] = {{16, 16}, {16, 8}, {8, 8}, {8, 4}, {4, 4}};
        for (std::uint32_t i = l.first_tail; i < d.mips; ++i) {
            MipPlacement& m = l.mip[i];
            mip_extent(i, m.width, m.height, m.depth);
            const std::uint32_t entry = max_tail - 1 - (i - l.first_tail);
            const std::uint64_t entry_offset = entry > 6 ? std::uint64_t(16) << entry : std::uint64_t(entry) << 8;
            std::uint32_t x = 0, y = 0;
            for (std::uint32_t bit = 0; bit < 6; ++bit) {
                x |= std::uint32_t((entry_offset >> (9 + 2 * bit)) & 1) << bit;
                y |= std::uint32_t((entry_offset >> (8 + 2 * bit)) & 1) << bit;
            }
            m.in_tail = true;
            m.tail_block = 0;
            m.tail_x = x * micro[log][0];
            m.tail_y = y * micro[log][1];
            m.offset = entry_offset;
            m.padded_width = l.block_w;
            m.padded_height = l.block_h;
            m.pitch_bytes = std::uint32_t(l.block_w * element);
            m.slice_2d = std::uint64_t(m.width) * m.height * element;
            m.size = block_size - entry_offset;
        }
        l.slice_size = cursor;
    }
    if (!checked_mul(l.slice_size, d.layers, l.size)) return false;
    out = l;
    return true;
}

// Byte offset from the surface start of element (x, y) of depth slice z of array slice `layer` of mip `mip`; the
// element's bytes follow it. The arguments must be inside the surface.
inline std::uint64_t element_offset(const SurfaceLayout& l, std::uint32_t mip, std::uint32_t layer, std::uint32_t x,
                                    std::uint32_t y, std::uint32_t z = 0) {
    const MipPlacement& m = l.mip[mip];
    const std::uint64_t slice = std::uint64_t(layer) * l.slice_size;
    if (l.mode == sw_linear)
        return slice + m.offset + z * m.slice_2d + std::uint64_t(y) * m.pitch_bytes + std::uint64_t(x) * l.element_bytes;
    std::uint64_t base;
    if (m.in_tail) {
        base = slice + m.tail_block;
        x += m.tail_x;
        y += m.tail_y;
    } else {
        const std::uint64_t blocks_x = m.padded_width / l.block_w;
        base = slice + m.offset + z * m.slice_2d +
               ((y / l.block_h) * blocks_x + x / l.block_w) * std::uint64_t(l.block_bytes);
        x %= l.block_w;
        y %= l.block_h;
    }
    std::uint64_t in_block = 0;
    for (std::uint32_t bit = 0; bit < l.block_bits; ++bit) {
        const int source = l.pattern[bit];
        if (source > 0) in_block |= std::uint64_t((x >> (source - 1)) & 1) << bit;
        else if (source < 0) in_block |= std::uint64_t((y >> (-source - 1)) & 1) << bit;
    }
    return base + in_block;
}

// Copies one subresource (every depth slice of a volume mip) between the tiled surface at `surface` and linear rows
// of elements: `row_pitch` bytes between rows (rows of blocks for BC formats), `slice_pitch` between depth slices.
// The tiled side is `surface_size` bytes from the surface start and must hold the whole surface. A multisampled
// surface's element is all its samples. False, with nothing written, for anything out of range.
inline bool copy_subresource(const SurfaceLayout& l, std::uint32_t mip, std::uint32_t layer, std::byte* surface,
                             std::uint64_t surface_size, std::byte* rows, std::uint64_t row_pitch,
                             std::uint64_t slice_pitch, bool to_surface) {
    if (mip >= l.mips || layer >= l.layers || !surface || !rows || surface_size < l.size) return false;
    const MipPlacement& m = l.mip[mip];
    const std::uint64_t row_bytes = std::uint64_t(m.width) * l.element_bytes;
    if (row_pitch < row_bytes || (m.depth > 1 && slice_pitch < row_pitch * m.height)) return false;
    // The in-block contribution of x and of y separately: every pattern bit is one or the other.
    const bool linear = l.mode == sw_linear;
    std::array<std::uint32_t, 256> x_bits{};
    if (!linear) {
        for (std::uint32_t x = 0; x < l.block_w; ++x)  // at most 256: a 64 KiB block of 1-byte elements
            for (std::uint32_t bit = 0; bit < l.block_bits; ++bit)
                if (l.pattern[bit] > 0) x_bits[x] |= ((x >> (l.pattern[bit] - 1)) & 1u) << bit;
    }
    for (std::uint32_t z = 0; z < m.depth; ++z) {
        for (std::uint32_t y = 0; y < m.height; ++y) {
            std::byte* line = rows + z * slice_pitch + std::uint64_t(y) * row_pitch;
            if (linear) {
                std::byte* at = surface + element_offset(l, mip, layer, 0, y, z);
                if (to_surface) std::memcpy(at, line, row_bytes);
                else std::memcpy(line, at, row_bytes);
                continue;
            }
            const std::uint32_t ty = m.in_tail ? y + m.tail_y : y % l.block_h;
            std::uint32_t y_bits = 0;
            for (std::uint32_t bit = 0; bit < l.block_bits; ++bit)
                if (l.pattern[bit] < 0) y_bits |= ((ty >> (-l.pattern[bit] - 1)) & 1u) << bit;
            const std::uint64_t slice = std::uint64_t(layer) * l.slice_size;
            std::uint64_t row_base;
            if (m.in_tail) {
                row_base = slice + m.tail_block;
            } else {
                const std::uint64_t blocks_x = m.padded_width / l.block_w;
                row_base = slice + m.offset + z * m.slice_2d + (y / l.block_h) * blocks_x * l.block_bytes;
            }
            for (std::uint32_t x = 0; x < m.width; ++x) {
                std::uint64_t at;
                if (m.in_tail) {
                    at = row_base + (x_bits[x + m.tail_x] | y_bits);
                } else {
                    at = row_base + std::uint64_t(x / l.block_w) * l.block_bytes + (x_bits[x % l.block_w] | y_bits);
                }
                std::byte* element = line + std::uint64_t(x) * l.element_bytes;
                if (to_surface) std::memcpy(surface + at, element, l.element_bytes);
                else std::memcpy(element, surface + at, l.element_bytes);
            }
        }
    }
    return true;
}

}  // namespace leonida::gfx10
