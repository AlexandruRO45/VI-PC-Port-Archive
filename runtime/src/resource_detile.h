#pragma once

// Detiling and tiling for AMD GFX10 / RDNA2 console-tiled resources (docs/d3d12_vulkan.md section 4.2).
//
// Resources created through Xbox slot 76 carry layout 0x101 (kXboxTiledLayout).
// The data layout inside each 64 KiB tile is GFX10 SW_64KB_D (Mesa addrlib GFX10_SW_64K_D_RBPLUS_PATINFO):
// 64 KiB macro-blocks are laid out in row-major order across the surface, and intra-tile coordinates (x, y)
// are swizzled into byte offsets via bit-permutation equations matching RDNA2 / Xbox Series hardware.
//
// Independent X and Y coordinate bit mappings permit O(1) intra-tile offset resolution via compile-time
// lookup tables (lut_x[x % tile_w] | lut_y[y % tile_h]). No Windows or Vulkan dependency.

#include "resource_tiling.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>

namespace leonida {

// The Xbox tiled layout word (device slot 76, docs/d3d12_vulkan.md section 4.2).
inline constexpr std::uint32_t xbox_tiled_layout = 0x101;

constexpr bool is_xbox_tiled(std::uint32_t layout) {
    return layout == xbox_tiled_layout;
}

// Intra-tile bit permutation equations for 64 KiB tiles (Mesa addrlib GFX10_SW_64K_D_RBPLUS_PATINFO).
// For each bit 0..15:
//   s > 0: bit is ((x >> (s - 1)) & 1)
//   s < 0: bit is ((y >> (-s - 1)) & 1)
//   s == 0: intra-element byte bit (element_bytes > 1)
template <std::uint32_t ElementBytes>
struct SwizzleLut64K {
    static constexpr std::uint32_t bpe = ElementBytes;
    static constexpr TileShape shape = element_tile_shape(false, bpe);
    static constexpr std::uint32_t tile_w = shape.width;
    static constexpr std::uint32_t tile_h = shape.height;

    std::array<std::uint16_t, tile_w> lut_x{};
    std::array<std::uint16_t, tile_h> lut_y{};

    constexpr SwizzleLut64K() {
        constexpr std::array<int8_t, 16> pat = []() {
            if constexpr (bpe == 1) {
                return std::array<int8_t, 16>{1, 2, 3, -2, -1, -3, 4, -4, -5, 5, -6, 6, -7, 7, -8, 8};
            } else if constexpr (bpe == 2) {
                return std::array<int8_t, 16>{0, 1, 2, 3, -1, -2, -3, 4, -4, 5, -5, 6, -6, 7, -7, 8};
            } else if constexpr (bpe == 4) {
                // RDNA2/RbPlus SW_64KB_D: bits 6..7 swapped vs Navi 1x ((3, -3) instead of (-3, 3))
                return std::array<int8_t, 16>{0, 0, 1, 2, -1, -2, 3, -3, -4, 4, -5, 5, -6, 6, -7, 7};
            } else if constexpr (bpe == 8) {
                return std::array<int8_t, 16>{0, 0, 0, 1, -1, 2, 3, -2, -3, 4, -4, 5, -5, 6, -6, 7};
            } else if constexpr (bpe == 16) {
                return std::array<int8_t, 16>{0, 0, 0, 0, 1, -1, 2, -2, -3, 3, -4, 4, -5, 5, -6, 6};
            } else {
                return std::array<int8_t, 16>{};
            }
        }();

        for (std::uint32_t x = 0; x < tile_w; ++x) {
            std::uint16_t val = 0;
            for (std::uint32_t bit = 0; bit < 16; ++bit) {
                const int8_t s = pat[bit];
                if (s > 0) {
                    val |= static_cast<std::uint16_t>(((x >> (s - 1)) & 1) << bit);
                }
            }
            lut_x[x] = val;
        }

        for (std::uint32_t y = 0; y < tile_h; ++y) {
            std::uint16_t val = 0;
            for (std::uint32_t bit = 0; bit < 16; ++bit) {
                const int8_t s = pat[bit];
                if (s < 0) {
                    val |= static_cast<std::uint16_t>(((y >> (-s - 1)) & 1) << bit);
                }
            }
            lut_y[y] = val;
        }
    }
};

inline constexpr SwizzleLut64K<1> swizzle_lut_64k_1bpe{};
inline constexpr SwizzleLut64K<2> swizzle_lut_64k_2bpe{};
inline constexpr SwizzleLut64K<4> swizzle_lut_64k_4bpe{};
inline constexpr SwizzleLut64K<8> swizzle_lut_64k_8bpe{};
inline constexpr SwizzleLut64K<16> swizzle_lut_64k_16bpe{};

// Compute required tiled byte size for a 2D surface of the given dimensions.
constexpr std::size_t tiled_surface_size_64k(std::uint32_t width, std::uint32_t height,
                                            std::uint32_t element_bytes, std::uint32_t block_dim = 1) {
    if (!width || !height || (block_dim != 1 && block_dim != 4)) return 0;
    const TileShape shape = element_tile_shape(false, element_bytes);
    if (!shape.width || !shape.height) return 0;
    const std::uint32_t blocks_w = (width + block_dim - 1) / block_dim;
    const std::uint32_t blocks_h = (height + block_dim - 1) / block_dim;
    const std::uint32_t tiles_x = (blocks_w + shape.width - 1) / shape.width;
    const std::uint32_t tiles_y = (blocks_h + shape.height - 1) / shape.height;
    return static_cast<std::size_t>(tiles_x) * tiles_y * tile_size_in_bytes;
}

// Compute required linear byte size for a 2D surface of the given dimensions.
constexpr std::size_t linear_surface_size(std::uint32_t width, std::uint32_t height,
                                         std::uint32_t element_bytes, std::uint32_t block_dim = 1) {
    if (!width || !height || (block_dim != 1 && block_dim != 4)) return 0;
    const std::uint32_t blocks_w = (width + block_dim - 1) / block_dim;
    const std::uint32_t blocks_h = (height + block_dim - 1) / block_dim;
    return static_cast<std::size_t>(blocks_w) * blocks_h * element_bytes;
}

namespace detail {

// One element as a load and a store of its own width: Bpe is a compile-time constant, so this is one move.
//
// `wide_load` is set when the reads are the scattered side (detiling). A 16-byte element is then one 128-bit load
// (two 64-bit loads measured slower there). Tiling has the scattered side on the stores, and there two 64-bit
// stores measured faster than one unaligned 128-bit store.
template <std::uint32_t Bpe, bool wide_load>
inline void move_element(std::byte* dst, const std::byte* src) {
    if constexpr (Bpe == 1) {
        *dst = *src;
    } else if constexpr (Bpe == 2) {
        std::uint16_t v;
        std::memcpy(&v, src, sizeof(v));
        std::memcpy(dst, &v, sizeof(v));
    } else if constexpr (Bpe == 4) {
        std::uint32_t v;
        std::memcpy(&v, src, sizeof(v));
        std::memcpy(dst, &v, sizeof(v));
    } else if constexpr (Bpe == 8) {
        std::uint64_t v;
        std::memcpy(&v, src, sizeof(v));
        std::memcpy(dst, &v, sizeof(v));
    } else if constexpr (wide_load) {
        std::memcpy(dst, src, 16);
    } else {
        std::uint64_t lo, hi;
        std::memcpy(&lo, src, sizeof(lo));
        std::memcpy(&hi, src + sizeof(lo), sizeof(hi));
        std::memcpy(dst, &lo, sizeof(lo));
        std::memcpy(dst + sizeof(lo), &hi, sizeof(hi));
    }
}

// `n` elements of one tile row. `from_tile` says which side is scattered: detiling reads the tile at `off[x]`
// and writes the row at `x * Bpe`, tiling reads the row and writes the tile.
//
// Four at a time, so the offset loads and the address arithmetic of one element overlap with the moves of the
// next and the four moves stay in flight together. The offsets come from a table built once per tile row (below)
// instead of a division, a modulo, a table load and an OR per element, which for a 1-byte element was most of the
// loop.
template <std::uint32_t Bpe, bool from_tile>
inline void move_row(std::byte* dst, const std::byte* src, const std::uint32_t* off, std::uint32_t n) {
    std::uint32_t x = 0;
    for (; x + 4 <= n; x += 4) {
        if constexpr (from_tile) {
            move_element<Bpe, true>(dst + static_cast<std::size_t>(x) * Bpe, src + off[x]);
            move_element<Bpe, true>(dst + static_cast<std::size_t>(x + 1) * Bpe, src + off[x + 1]);
            move_element<Bpe, true>(dst + static_cast<std::size_t>(x + 2) * Bpe, src + off[x + 2]);
            move_element<Bpe, true>(dst + static_cast<std::size_t>(x + 3) * Bpe, src + off[x + 3]);
        } else {
            move_element<Bpe, false>(dst + off[x], src + static_cast<std::size_t>(x) * Bpe);
            move_element<Bpe, false>(dst + off[x + 1], src + static_cast<std::size_t>(x + 1) * Bpe);
            move_element<Bpe, false>(dst + off[x + 2], src + static_cast<std::size_t>(x + 2) * Bpe);
            move_element<Bpe, false>(dst + off[x + 3], src + static_cast<std::size_t>(x + 3) * Bpe);
        }
    }
    for (; x < n; ++x) {
        if constexpr (from_tile) {
            move_element<Bpe, true>(dst + static_cast<std::size_t>(x) * Bpe, src + off[x]);
        } else {
            move_element<Bpe, false>(dst + off[x], src + static_cast<std::size_t>(x) * Bpe);
        }
    }
}

// The byte offset of every element of one tile row inside its 64 KiB tile: lut_x[x] | lut_y[y % tile_h]. A tile
// row has the same offsets in every tile of the surface, so this is built once per row and used for every tile
// column in it. Without it each element paid a division, a modulo, a table load and an OR before its move, which
// for a 1-byte element was most of the loop.
template <std::uint32_t Bpe>
using TileRowOffsets = std::array<std::uint32_t, SwizzleLut64K<Bpe>::tile_w>;

template <std::uint32_t Bpe>
void detile_typed(std::span<const std::byte> tiled, std::span<std::byte> linear,
                  std::uint32_t blocks_w, std::uint32_t blocks_h, std::uint32_t tiles_x) {
    constexpr const auto& lut = []() constexpr -> const auto& {
        if constexpr (Bpe == 1) return swizzle_lut_64k_1bpe;
        else if constexpr (Bpe == 2) return swizzle_lut_64k_2bpe;
        else if constexpr (Bpe == 4) return swizzle_lut_64k_4bpe;
        else if constexpr (Bpe == 8) return swizzle_lut_64k_8bpe;
        else if constexpr (Bpe == 16) return swizzle_lut_64k_16bpe;
    }();

    constexpr std::uint32_t tile_w = lut.tile_w;
    constexpr std::uint32_t tile_h = lut.tile_h;
    TileRowOffsets<Bpe> intra{};
    const std::uint32_t full_x = blocks_w / tile_w;
    const std::uint32_t tail_x = blocks_w % tile_w;

    for (std::uint32_t y = 0; y < blocks_h; ++y) {
        const std::uint32_t intra_y = lut.lut_y[y % tile_h];
        const std::size_t macro_y_base = static_cast<std::size_t>(y / tile_h) * tiles_x * tile_size_in_bytes;
        std::byte* const dst_row = linear.data() + static_cast<std::size_t>(y) * blocks_w * Bpe;
        const std::byte* const src_row = tiled.data() + macro_y_base;
        for (std::uint32_t x = 0; x < tile_w; ++x) {
            intra[x] = lut.lut_x[x] | intra_y;
        }
        for (std::uint32_t tx = 0; tx < full_x; ++tx) {
            move_row<Bpe, true>(dst_row + static_cast<std::size_t>(tx) * tile_w * Bpe,
                                src_row + static_cast<std::size_t>(tx) * tile_size_in_bytes, intra.data(), tile_w);
        }
        if (tail_x) {
            move_row<Bpe, true>(dst_row + static_cast<std::size_t>(full_x) * tile_w * Bpe,
                                src_row + static_cast<std::size_t>(full_x) * tile_size_in_bytes, intra.data(), tail_x);
        }
    }
}

template <std::uint32_t Bpe>
void tile_typed(std::span<const std::byte> linear, std::span<std::byte> tiled,
                std::uint32_t blocks_w, std::uint32_t blocks_h, std::uint32_t tiles_x) {
    constexpr const auto& lut = []() constexpr -> const auto& {
        if constexpr (Bpe == 1) return swizzle_lut_64k_1bpe;
        else if constexpr (Bpe == 2) return swizzle_lut_64k_2bpe;
        else if constexpr (Bpe == 4) return swizzle_lut_64k_4bpe;
        else if constexpr (Bpe == 8) return swizzle_lut_64k_8bpe;
        else if constexpr (Bpe == 16) return swizzle_lut_64k_16bpe;
    }();

    constexpr std::uint32_t tile_w = lut.tile_w;
    constexpr std::uint32_t tile_h = lut.tile_h;

    TileRowOffsets<Bpe> intra{};
    const std::uint32_t full_x = blocks_w / tile_w;
    const std::uint32_t tail_x = blocks_w % tile_w;

    for (std::uint32_t y = 0; y < blocks_h; ++y) {
        const std::uint32_t intra_y = lut.lut_y[y % tile_h];
        const std::size_t macro_y_base = static_cast<std::size_t>(y / tile_h) * tiles_x * tile_size_in_bytes;
        const std::byte* const src_row = linear.data() + static_cast<std::size_t>(y) * blocks_w * Bpe;
        std::byte* const dst_row = tiled.data() + macro_y_base;
        for (std::uint32_t x = 0; x < tile_w; ++x) {
            intra[x] = lut.lut_x[x] | intra_y;
        }
        for (std::uint32_t tx = 0; tx < full_x; ++tx) {
            move_row<Bpe, false>(dst_row + static_cast<std::size_t>(tx) * tile_size_in_bytes,
                                 src_row + static_cast<std::size_t>(tx) * tile_w * Bpe, intra.data(), tile_w);
        }
        if (tail_x) {
            move_row<Bpe, false>(dst_row + static_cast<std::size_t>(full_x) * tile_size_in_bytes,
                                 src_row + static_cast<std::size_t>(full_x) * tile_w * Bpe, intra.data(), tail_x);
        }
    }
}

}  // namespace detail

// Detile a 2D surface from console-tiled 64 KiB layout into linear row-major order.
inline bool detile_surface_64k(std::span<const std::byte> tiled, std::span<std::byte> linear,
                               std::uint32_t width, std::uint32_t height,
                               std::uint32_t element_bytes, std::uint32_t block_dim = 1) {
    if (!width || !height || (block_dim != 1 && block_dim != 4)) return false;
    const TileShape shape = element_tile_shape(false, element_bytes);
    if (!shape.width || !shape.height) return false;

    const std::uint32_t blocks_w = (width + block_dim - 1) / block_dim;
    const std::uint32_t blocks_h = (height + block_dim - 1) / block_dim;
    const std::uint32_t tiles_x = (blocks_w + shape.width - 1) / shape.width;
    const std::uint32_t tiles_y = (blocks_h + shape.height - 1) / shape.height;

    const std::size_t expected_linear = static_cast<std::size_t>(blocks_w) * blocks_h * element_bytes;
    const std::size_t expected_tiled = static_cast<std::size_t>(tiles_x) * tiles_y * tile_size_in_bytes;

    if (linear.size() < expected_linear || tiled.size() < expected_tiled) return false;

    switch (element_bytes) {
    case 1:
        detail::detile_typed<1>(tiled, linear, blocks_w, blocks_h, tiles_x);
        return true;
    case 2:
        detail::detile_typed<2>(tiled, linear, blocks_w, blocks_h, tiles_x);
        return true;
    case 4:
        detail::detile_typed<4>(tiled, linear, blocks_w, blocks_h, tiles_x);
        return true;
    case 8:
        detail::detile_typed<8>(tiled, linear, blocks_w, blocks_h, tiles_x);
        return true;
    case 16:
        detail::detile_typed<16>(tiled, linear, blocks_w, blocks_h, tiles_x);
        return true;
    default:
        return false;
    }
}

// Tile a 2D surface from linear row-major order into console-tiled 64 KiB layout.
inline bool tile_surface_64k(std::span<const std::byte> linear, std::span<std::byte> tiled,
                             std::uint32_t width, std::uint32_t height,
                             std::uint32_t element_bytes, std::uint32_t block_dim = 1) {
    if (!width || !height || (block_dim != 1 && block_dim != 4)) return false;
    const TileShape shape = element_tile_shape(false, element_bytes);
    if (!shape.width || !shape.height) return false;

    const std::uint32_t blocks_w = (width + block_dim - 1) / block_dim;
    const std::uint32_t blocks_h = (height + block_dim - 1) / block_dim;
    const std::uint32_t tiles_x = (blocks_w + shape.width - 1) / shape.width;
    const std::uint32_t tiles_y = (blocks_h + shape.height - 1) / shape.height;

    const std::size_t expected_linear = static_cast<std::size_t>(blocks_w) * blocks_h * element_bytes;
    const std::size_t expected_tiled = static_cast<std::size_t>(tiles_x) * tiles_y * tile_size_in_bytes;

    if (linear.size() < expected_linear || tiled.size() < expected_tiled) return false;

    switch (element_bytes) {
    case 1:
        detail::tile_typed<1>(linear, tiled, blocks_w, blocks_h, tiles_x);
        return true;
    case 2:
        detail::tile_typed<2>(linear, tiled, blocks_w, blocks_h, tiles_x);
        return true;
    case 4:
        detail::tile_typed<4>(linear, tiled, blocks_w, blocks_h, tiles_x);
        return true;
    case 8:
        detail::tile_typed<8>(linear, tiled, blocks_w, blocks_h, tiles_x);
        return true;
    case 16:
        detail::tile_typed<16>(linear, tiled, blocks_w, blocks_h, tiles_x);
        return true;
    default:
        return false;
    }
}

}  // namespace leonida
