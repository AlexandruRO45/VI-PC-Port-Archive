// The xg_xs texture computer's model (src/gfx10_layout.h, src/xg_surface.h), without Windows or a GPU.
//
// - Single-slice 2D layouts equal tools/tex_headers.py compact_surface_layout, the rule measured byte-exact on the
//   retail pools (docs/textures.md): size, first tail mip, every standalone mip's offset and padding, every tail
//   mip's origin. The expected rows were printed by that function.
// - The measured bit orders: one mip of SW_4KB_S and SW_64KB_D puts every element where gfx10_tiling.h does.
// - Every mode, with mips, tails and array slices: each element of each subresource has its own bytes inside the
//   surface, and a copy in and out returns the rows.
// - Arrays are slice-major, as the executable addresses them; the XG structure carries that and the planes.
#include "../src/xg_surface.h"

#include <cstdio>
#include <cstring>
#include <vector>

#define CHECK(condition) do {                                      \
    if (!(condition)) {                                            \
        std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); \
        return 1;                                                  \
    }                                                              \
} while (0)

namespace {

using namespace leonida;

struct ExpectedMip {
    std::uint64_t offset;
    std::uint32_t padded_w, padded_h, tail_x, tail_y, in_tail;
};
struct Expected {
    std::uint32_t width, height, mips, element_bytes, block, tile;
    std::uint64_t size;
    std::uint32_t first_tail;
    std::vector<ExpectedMip> mip;
};

// python -I -c "from tex_headers import compact_surface_layout ..." in tools/, one row per case.
const Expected kExpected[] = {
    {4096,4096,13,1,4,65536,1441792,3,{{393216,1024,1024,0,0,0},{131072,512,512,0,0,0},{65536,256,256,0,0,0},{0,256,256,128,0,1},{0,256,256,0,128,1},{0,256,256,64,0,1},{0,256,256,0,64,1},{0,256,256,32,0,1},{0,256,256,16,32,1},{0,256,256,0,48,1},{0,256,256,0,32,1},{0,256,256,16,16,1},{0,256,256,16,0,1}}},
    {1024,512,11,16,4,4096,704512,5,{{180224,256,128,0,0,0},{49152,128,64,0,0,0},{16384,64,32,0,0,0},{8192,32,16,0,0,0},{4096,16,16,0,0,0},{0,16,16,8,0,1},{0,16,16,4,8,1},{0,16,16,0,12,1},{0,16,16,0,8,1},{0,16,16,4,4,1},{0,16,16,4,0,1}}},
    {256,256,9,4,1,65536,393216,2,{{131072,256,256,0,0,0},{65536,128,128,0,0,0},{0,128,128,64,0,1},{0,128,128,0,64,1},{0,128,128,32,0,1},{0,128,128,0,32,1},{0,128,128,16,0,1},{0,128,128,8,16,1},{0,128,128,0,24,1}}},
    {300,200,9,4,1,4096,409600,5,{{122880,320,224,0,0,0},{40960,160,128,0,0,0},{16384,96,64,0,0,0},{8192,64,32,0,0,0},{4096,32,32,0,0,0},{0,32,32,16,0,1},{0,32,32,8,16,1},{0,32,32,0,24,1},{0,32,32,0,16,1}}},
    {64,64,7,8,4,65536,65536,0,{{0,128,64,64,0,1},{0,128,64,0,32,1},{0,128,64,32,0,1},{0,128,64,0,16,1},{0,128,64,16,0,1},{0,128,64,8,8,1},{0,128,64,0,12,1}}},
    {2048,64,12,2,1,65536,1048576,4,{{524288,2048,128,0,0,0},{262144,1024,128,0,0,0},{131072,512,128,0,0,0},{65536,256,128,0,0,0},{0,256,128,128,0,1},{0,256,128,0,64,1},{0,256,128,64,0,1},{0,256,128,0,32,1},{0,256,128,32,0,1},{0,256,128,16,16,1},{0,256,128,0,24,1},{0,256,128,0,16,1}}},
    {8,8,4,1,1,4096,4096,0,{{0,64,64,32,0,1},{0,64,64,16,32,1},{0,64,64,0,48,1},{0,64,64,0,32,1}}},
    {1,1,1,4,1,4096,4096,1,{{0,32,32,0,0,0}}},
    {512,512,10,8,4,4096,176128,3,{{45056,128,128,0,0,0},{12288,64,64,0,0,0},{4096,32,32,0,0,0},{0,32,16,16,0,1},{0,32,16,8,8,1},{0,32,16,0,12,1},{0,32,16,0,8,1},{0,32,16,8,4,1},{0,32,16,8,0,1},{0,32,16,0,4,1}}},
};

int check_measured_layouts() {
    for (const Expected& e : kExpected) {
        gfx10::SurfaceDesc d;
        d.mode = e.tile == 65536 ? gfx10::sw_64kb_d : gfx10::sw_4kb_s;
        d.element_bytes = e.element_bytes;
        d.block = e.block;
        d.width = e.width;
        d.height = e.height;
        d.mips = e.mips;
        gfx10::SurfaceLayout l;
        CHECK(gfx10::surface_layout(d, l));
        CHECK(l.size == e.size && l.slice_size == e.size);
        CHECK(l.first_tail == e.first_tail);
        for (std::uint32_t i = 0; i < e.mips; ++i) {
            const gfx10::MipPlacement& m = l.mip[i];
            const ExpectedMip& x = e.mip[i];
            CHECK(m.in_tail == bool(x.in_tail));
            if (m.in_tail) {
                CHECK(m.tail_x == x.tail_x && m.tail_y == x.tail_y && m.tail_block == 0);
            } else {
                CHECK(m.offset == x.offset && m.padded_width == x.padded_w && m.padded_height == x.padded_h);
            }
        }
    }
    return 0;
}

int check_measured_bit_order() {
    const gfx10::TileMode modes[] = {gfx10::TileMode::sw_4kb_s, gfx10::TileMode::sw_64kb_d};
    for (const gfx10::TileMode mode : modes) {
        for (std::uint32_t bytes : {1u, 2u, 4u, 8u, 16u}) {
            const std::uint32_t w = 300, h = 170;
            gfx10::TiledSurface2D reference;
            CHECK(gfx10::TiledSurface2D::make(mode, bytes, w, h, reference));
            gfx10::SurfaceDesc d;
            d.mode = mode == gfx10::TileMode::sw_4kb_s ? gfx10::sw_4kb_s : gfx10::sw_64kb_d;
            d.element_bytes = bytes;
            d.width = w;
            d.height = h;
            gfx10::SurfaceLayout l;
            CHECK(gfx10::surface_layout(d, l));
            CHECK(l.exact && l.size == reference.tiled_size());
            for (std::uint32_t y = 0; y < h; y += 3)
                for (std::uint32_t x = 0; x < w; x += 7) {
                    std::size_t expected = 0;
                    CHECK(reference.element_offset(x, y, expected));
                    CHECK(gfx10::element_offset(l, 0, 0, x, y) == expected);
                }
        }
    }
    return 0;
}

// Every element of every subresource owns element_bytes bytes no other element touches, inside the surface; a copy
// in and back out returns the rows.
int check_round_trip(const gfx10::SurfaceDesc& d) {
    gfx10::SurfaceLayout l;
    CHECK(gfx10::surface_layout(d, l));
    std::vector<std::uint8_t> owner(l.size, 0);
    std::vector<std::byte> surface(l.size);
    for (std::uint32_t layer = 0; layer < l.layers; ++layer)
        for (std::uint32_t mip = 0; mip < l.mips; ++mip) {
            const gfx10::MipPlacement& m = l.mip[mip];
            for (std::uint32_t z = 0; z < m.depth; ++z)
                for (std::uint32_t y = 0; y < m.height; ++y)
                    for (std::uint32_t x = 0; x < m.width; ++x) {
                        const std::uint64_t at = gfx10::element_offset(l, mip, layer, x, y, z);
                        CHECK(at % l.element_bytes == 0 && at + l.element_bytes <= l.size);
                        for (std::uint32_t b = 0; b < l.element_bytes; ++b) CHECK(owner[at + b]++ == 0);
                    }
            const std::uint64_t row_pitch = std::uint64_t(m.width) * l.element_bytes + 12;
            const std::uint64_t slice_pitch = row_pitch * m.height + 40;
            std::vector<std::byte> rows(slice_pitch * m.depth), back(rows.size());
            for (std::size_t i = 0; i < rows.size(); ++i) rows[i] = std::byte((i * 131 + mip * 7 + layer) & 0xff);
            CHECK(gfx10::copy_subresource(l, mip, layer, surface.data(), surface.size(), rows.data(), row_pitch,
                                          slice_pitch, true));
            CHECK(gfx10::copy_subresource(l, mip, layer, surface.data(), surface.size(), back.data(), row_pitch,
                                          slice_pitch, false));
            for (std::uint32_t z = 0; z < m.depth; ++z)
                for (std::uint32_t y = 0; y < m.height; ++y) {
                    const std::size_t at = z * slice_pitch + y * row_pitch;
                    CHECK(std::memcmp(rows.data() + at, back.data() + at, std::size_t(m.width) * l.element_bytes) == 0);
                }
            // The copy and element_offset agree on one element.
            const std::uint32_t x = m.width - 1, y = m.height - 1;
            CHECK(std::memcmp(surface.data() + gfx10::element_offset(l, mip, layer, x, y),
                              rows.data() + y * row_pitch + std::uint64_t(x) * l.element_bytes, l.element_bytes) == 0);
        }
    return 0;
}

int check_every_mode() {
    const std::uint32_t modes[] = {gfx10::sw_linear, gfx10::sw_256b_s, gfx10::sw_256b_d, gfx10::sw_4kb_s,
                                   gfx10::sw_4kb_d, gfx10::sw_64kb_s, gfx10::sw_64kb_d, gfx10::sw_64kb_r_x,
                                   gfx10::sw_4kb_z_x};
    for (const std::uint32_t mode : modes) {
        for (std::uint32_t bytes : {1u, 4u, 16u}) {
            gfx10::SurfaceDesc d;
            d.mode = mode;
            d.element_bytes = bytes;
            d.width = 77;
            d.height = 45;
            d.mips = 7;
            d.layers = 3;
            if (check_round_trip(d)) return 1;
            gfx10::SurfaceDesc bc = d;  // a BC1 chain whose last mips are one block
            bc.element_bytes = 8;
            bc.block = 4;
            bc.width = 260;
            bc.height = 130;
            bc.mips = 9;
            bc.layers = 1;
            if (check_round_trip(bc)) return 1;
            gfx10::SurfaceDesc volume = d;
            volume.layers = 1;
            volume.volume = true;
            volume.depth = 9;
            volume.mips = 4;
            if (check_round_trip(volume)) return 1;
        }
    }
    return 0;
}

int check_xg() {
    xg::ResourceDesc d;
    d.dimension = 3;
    d.width = 512;
    d.height = 256;
    d.depth_or_array_size = 4;
    d.mip_levels = 10;
    d.format = 71;  // BC1_UNORM
    d.sample_count = 1;
    d.layout = 0x100 | gfx10::sw_4kb_s;
    xg::Surface s;
    CHECK(xg::surface(d, s));
    CHECK(s.planes == 1 && s.layers == 4 && s.plane[0].mode == gfx10::sw_4kb_s);
    xg::ResourceLayout layout;
    xg::resource_layout(s, layout);
    CHECK(layout.size_bytes == s.size && layout.size_bytes == 4 * layout.plane[0].slice_bytes);
    CHECK(layout.mip_levels == 10 && layout.planes == 1 && layout.plane[0].bytes_per_element == 8);
    // How the executable finds slice 2 of mip 1: 2 * plane+0x10 + mip 1's +0x08.
    const std::uint64_t at = 2 * layout.plane[0].slice_bytes + layout.plane[0].mip[1].offset_bytes;
    CHECK(!s.plane[0].mip[1].in_tail && at == gfx10::element_offset(s.plane[0], 1, 2, 0, 0));
    CHECK(layout.plane[0].mip[0].pitch_bytes == 128 * 8);  // 512 / 4 blocks, already a whole 4 KiB block wide
    CHECK(layout.plane[0].mip[0].size_bytes == layout.plane[0].mip[0].slice_2d_size_bytes);

    // The tail: mips past first_tail share the slice's first block, at their entry offsets.
    CHECK(s.plane[0].first_tail < 10 && layout.plane[0].mip[9].offset_bytes < 4096);

    // Depth with stencil: two planes, the stencil after the depth.
    xg::ResourceDesc ds = d;
    ds.format = 20;  // D32_FLOAT_S8X24_UINT
    ds.mip_levels = 1;
    ds.depth_or_array_size = 1;
    ds.layout = 0x100 | gfx10::sw_64kb_d;
    CHECK(xg::surface(ds, s));
    CHECK(s.planes == 2 && s.plane_base[1] >= s.plane[0].size && s.plane[1].element_bytes == 1);
    xg::resource_layout(s, layout);
    CHECK(layout.planes == 2 && layout.plane[0].usage == 1 && layout.plane[1].usage == 2);
    CHECK(layout.plane[1].mip[0].offset_bytes == s.plane_base[1]);

    // Layout words: 0x100 | mode, the PC's row major and 64 KiB swizzles, and none.
    CHECK(xg::layout_mode(0x10a) == gfx10::sw_64kb_d && xg::layout_mode(0x102) == gfx10::sw_256b_d);
    CHECK(xg::layout_mode(1) == gfx10::sw_linear && xg::layout_mode(0) == -1 && xg::layout_mode(0x10c) == -1);

    // Suggestions: what the pools use, linear for 1D and 12-byte elements.
    CHECK(xg::suggest_mode(d) == gfx10::sw_64kb_d);  // 128 x 64 BC1 blocks: a 64 KiB slice
    xg::ResourceDesc small = d;
    small.width = 256;
    CHECK(xg::suggest_mode(small) == gfx10::sw_4kb_s);
    xg::ResourceDesc line = d;
    line.dimension = 2;
    CHECK(xg::suggest_mode(line) == gfx10::sw_linear);
    xg::ResourceDesc wide = d;
    wide.format = 6;  // R32G32B32_FLOAT
    CHECK(xg::suggest_mode(wide) == gfx10::sw_linear);

    // A mip count of 0 is the full chain; an unknown format has no layout.
    xg::ResourceDesc full = d;
    full.mip_levels = 0;
    CHECK(xg::surface(full, s) && s.mips == 10);
    xg::ResourceDesc unknown = d;
    unknown.format = 0;
    CHECK(!xg::surface(unknown, s));

    // Slot 10: the PC's 64 KiB tiling.
    xg::ResourceDesc sparse = d;
    sparse.format = 28;  // R8G8B8A8_UNORM
    sparse.layout = 0x10a;
    sparse.depth_or_array_size = 1;
    CHECK(xg::surface(sparse, s));
    leonida::ResourceTiling t;
    CHECK(xg::resource_tiling(s, t));
    CHECK(t.shape.width == 128 && t.shape.height == 128 && t.standard_mips >= 1);
    return 0;
}

}  // namespace

int main() {
    if (check_measured_layouts()) return 1;
    if (check_measured_bit_order()) return 1;
    if (check_every_mode()) return 1;
    if (check_xg()) return 1;
    std::printf("xg_surface: all checks passed\n");
    return 0;
}
