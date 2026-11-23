// Lanterns stage 6: the textures, made here from integer formulas, mipmapped with a 2x2 box filter and compressed by
// the small encoders below: BC1 (four-colour blocks, endpoints from the block's bounding box, each texel the nearest of
// the four palette colours) and BC7 mode 6 (one subset, 7-bit RGBA endpoints with a p-bit each, 4-bit indices along
// the endpoint line). decode() is the reference decoder: BC7 as the format specifies it bit for bit; BC1's two
// interpolated colours as exact thirds, which hardware may round differently (lanterns_sampling.h's tolerance covers
// one 8-bit step). Nothing here comes from a game; the patterns are a stone mosaic and nested squares.
#pragma once

#include "lanterns_reference.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <vector>

namespace lanterns {

using Rgba8 = std::array<std::uint8_t, 4>;

struct Image8 {
    unsigned width = 0, height = 0;
    std::vector<Rgba8> texels;
    const Rgba8& at(unsigned x, unsigned y) const { return texels[std::size_t(y) * width + x]; }
};

enum class BcFormat { bc1, bc7 };

// One compressed texture: every mip level's 4x4 blocks, row-major (8 bytes a block for BC1, 16 for BC7), and the
// levels decoded back to float (what the sampler reads, as the reference sees it).
struct BcTexture {
    BcFormat format;
    unsigned width, height;
    std::vector<std::vector<std::uint8_t>> blocks;
    std::vector<std::vector<Color>> decoded;
    unsigned block_bytes() const { return format == BcFormat::bc1 ? 8 : 16; }
    unsigned mip_width(unsigned m) const { return std::max(1u, width >> m); }
    unsigned mip_height(unsigned m) const { return std::max(1u, height >> m); }
};

// 256x256: 32-texel stone tiles in six muted colours with a soft diagonal shading, 2-texel dark grout between them.
inline Image8 mosaic_pattern() {
    static constexpr Rgba8 stones[6] = {{{180, 170, 150, 255}}, {{150, 160, 170, 255}}, {{170, 140, 120, 255}},
                                        {{130, 150, 130, 255}}, {{160, 150, 175, 255}}, {{195, 185, 160, 255}}};
    Image8 img{256, 256, {}};
    for (unsigned y = 0; y < 256; ++y)
        for (unsigned x = 0; x < 256; ++x) {
            const unsigned tx = x / 32, ty = y / 32, lx = x % 32, ly = y % 32;
            if (lx < 2 || ly < 2) {
                img.texels.push_back({40, 40, 48, 255});
                continue;
            }
            Rgba8 c = stones[(tx * 7 + ty * 13 + (tx ^ ty)) % 6];
            const int shade = int(lx + ly) / 4 - 7;   // -7..7
            for (int k = 0; k < 3; ++k) c[k] = std::uint8_t(std::clamp(int(c[k]) + 2 * shade, 0, 255));
            img.texels.push_back(c);
        }
    return img;
}

// 64x64: nested squares around the centre, alternating gold and teal bands four texels wide, with a light rim.
inline Image8 squares_pattern() {
    Image8 img{64, 64, {}};
    for (unsigned y = 0; y < 64; ++y)
        for (unsigned x = 0; x < 64; ++x) {
            const unsigned d = std::max(unsigned(std::abs(int(2 * x) - 63)), unsigned(std::abs(int(2 * y) - 63))) / 2;
            const unsigned band = d / 4;
            Rgba8 c = band % 2 ? Rgba8{40, 115, 125, 255} : Rgba8{235, 185, 90, 255};
            if (d % 4 == 3) c = {245, 240, 220, 255};
            img.texels.push_back(c);
        }
    return img;
}

inline Image8 half_size(const Image8& in) {
    Image8 out{std::max(1u, in.width / 2), std::max(1u, in.height / 2), {}};
    for (unsigned y = 0; y < out.height; ++y)
        for (unsigned x = 0; x < out.width; ++x) {
            const unsigned x0 = std::min(2 * x, in.width - 1), x1 = std::min(2 * x + 1, in.width - 1);
            const unsigned y0 = std::min(2 * y, in.height - 1), y1 = std::min(2 * y + 1, in.height - 1);
            Rgba8 c;
            for (int k = 0; k < 4; ++k)
                c[k] = std::uint8_t((in.at(x0, y0)[k] + in.at(x1, y0)[k] + in.at(x0, y1)[k] + in.at(x1, y1)[k] + 2) / 4);
            out.texels.push_back(c);
        }
    return out;
}

namespace bc {

// The block's 16 texels (clamped at the level's edge, for levels smaller than a block).
inline std::array<Rgba8, 16> gather(const Image8& img, unsigned bx, unsigned by) {
    std::array<Rgba8, 16> t;
    for (unsigned i = 0; i < 16; ++i)
        t[i] = img.at(std::min(bx * 4 + i % 4, img.width - 1), std::min(by * 4 + i / 4, img.height - 1));
    return t;
}

inline int distance2(const Rgba8& a, const Rgba8& b, int channels) {
    int d = 0;
    for (int k = 0; k < channels; ++k) d += (int(a[k]) - int(b[k])) * (int(a[k]) - int(b[k]));
    return d;
}

// ---- BC1
inline std::uint16_t pack565(const Rgba8& c) {
    return std::uint16_t(((c[0] * 31 + 127) / 255) << 11 | ((c[1] * 63 + 127) / 255) << 5 | ((c[2] * 31 + 127) / 255));
}
inline Rgba8 unpack565(std::uint16_t v) {
    const unsigned r = v >> 11, g = (v >> 5) & 63, b = v & 31;
    return {std::uint8_t(r << 3 | r >> 2), std::uint8_t(g << 2 | g >> 4), std::uint8_t(b << 3 | b >> 2), 255};
}
// The four palette colours as floats; colours 2 and 3 are the exact thirds.
inline std::array<Color, 4> bc1_palette(std::uint16_t c0, std::uint16_t c1) {
    const Rgba8 a = unpack565(c0), b = unpack565(c1);
    std::array<Color, 4> p;
    for (int k = 0; k < 3; ++k) {
        p[0][k] = a[k] / 255.0f;
        p[1][k] = b[k] / 255.0f;
        p[2][k] = float((2.0 * a[k] + b[k]) / 3.0 / 255.0);
        p[3][k] = float((a[k] + 2.0 * b[k]) / 3.0 / 255.0);
    }
    for (auto& c : p) c[3] = 1.0f;
    return p;
}

inline void encode_bc1(const std::array<Rgba8, 16>& t, std::uint8_t out[8]) {
    Rgba8 lo{255, 255, 255, 255}, hi{0, 0, 0, 255};
    for (const Rgba8& c : t)
        for (int k = 0; k < 3; ++k) lo[k] = std::min(lo[k], c[k]), hi[k] = std::max(hi[k], c[k]);
    std::uint16_t c0 = pack565(hi), c1 = pack565(lo);
    std::uint32_t indices = 0;
    if (c0 == c1) {
        c1 = c0 ? std::uint16_t(c0 - 1) : 0;   // keep four-colour mode (c0 > c1); every texel takes c0
        if (c0 == 0) c0 = 1, c1 = 0;
        if (pack565(hi) == 0) indices = 0x55555555u;   // black: index 1 is the 0 endpoint
    } else {
        Rgba8 pal[4];
        const Rgba8 a = unpack565(c0), b = unpack565(c1);
        for (int k = 0; k < 3; ++k) {
            pal[0][k] = a[k], pal[1][k] = b[k];
            pal[2][k] = std::uint8_t((2 * a[k] + b[k] + 1) / 3), pal[3][k] = std::uint8_t((a[k] + 2 * b[k] + 1) / 3);
        }
        for (unsigned i = 0; i < 16; ++i) {
            unsigned best = 0;
            for (unsigned p = 1; p < 4; ++p)
                if (distance2(t[i], pal[p], 3) < distance2(t[i], pal[best], 3)) best = p;
            indices |= best << (2 * i);
        }
    }
    out[0] = std::uint8_t(c0), out[1] = std::uint8_t(c0 >> 8), out[2] = std::uint8_t(c1), out[3] = std::uint8_t(c1 >> 8);
    for (int k = 0; k < 4; ++k) out[4 + k] = std::uint8_t(indices >> (8 * k));
}

inline void decode_bc1(const std::uint8_t* b, Color out[16]) {
    const std::uint16_t c0 = std::uint16_t(b[0] | b[1] << 8), c1 = std::uint16_t(b[2] | b[3] << 8);
    const auto pal = bc1_palette(c0, c1);   // the encoder always writes c0 > c1: four-colour mode
    const std::uint32_t idx = std::uint32_t(b[4]) | std::uint32_t(b[5]) << 8 | std::uint32_t(b[6]) << 16 |
                              std::uint32_t(b[7]) << 24;
    for (unsigned i = 0; i < 16; ++i) out[i] = pal[(idx >> (2 * i)) & 3];
}

// ---- BC7 mode 6
constexpr int kWeights4[16] = {0, 4, 9, 13, 17, 21, 26, 30, 34, 38, 43, 47, 51, 55, 60, 64};

struct Bits {
    std::uint8_t* out;
    unsigned at = 0;
    void put(unsigned value, unsigned count) {
        for (unsigned i = 0; i < count; ++i, ++at)
            if (value >> i & 1) out[at / 8] |= std::uint8_t(1u << (at % 8));
    }
};

inline unsigned get_bits(const std::uint8_t* in, unsigned& at, unsigned count) {
    unsigned v = 0;
    for (unsigned i = 0; i < count; ++i, ++at) v |= unsigned(in[at / 8] >> (at % 8) & 1) << i;
    return v;
}

// An endpoint as 7 bits per channel and one p-bit: the 8-bit value is (c7 << 1) | p.
inline void quantize_endpoint(const Rgba8& c, std::array<unsigned, 4>& c7, unsigned& p) {
    unsigned odd = 0;
    for (int k = 0; k < 4; ++k) odd += c[k] & 1;
    p = odd >= 2 ? 1 : 0;
    for (int k = 0; k < 4; ++k) c7[k] = unsigned(std::clamp((int(c[k]) - int(p) + 1) / 2, 0, 127));
}

inline void encode_bc7(const std::array<Rgba8, 16>& t, std::uint8_t out[16]) {
    Rgba8 lo{255, 255, 255, 255}, hi{0, 0, 0, 0};
    for (const Rgba8& c : t)
        for (int k = 0; k < 4; ++k) lo[k] = std::min(lo[k], c[k]), hi[k] = std::max(hi[k], c[k]);
    std::array<unsigned, 4> e7[2];
    unsigned pbit[2];
    quantize_endpoint(lo, e7[0], pbit[0]);
    quantize_endpoint(hi, e7[1], pbit[1]);
    int e[2][4];
    for (int s = 0; s < 2; ++s)
        for (int k = 0; k < 4; ++k) e[s][k] = int(e7[s][k] << 1 | pbit[s]);
    unsigned idx[16];
    for (unsigned i = 0; i < 16; ++i) {
        unsigned best = 0;
        int best_d = 1 << 30;
        for (unsigned w = 0; w < 16; ++w) {
            Rgba8 c;
            for (int k = 0; k < 4; ++k)
                c[k] = std::uint8_t(((64 - kWeights4[w]) * e[0][k] + kWeights4[w] * e[1][k] + 32) >> 6);
            const int d = distance2(t[i], c, 4);
            if (d < best_d) best_d = d, best = w;
        }
        idx[i] = best;
    }
    if (idx[0] >= 8) {   // the anchor index has no top bit: swap the endpoints and mirror the indices
        std::swap(e7[0], e7[1]);
        std::swap(pbit[0], pbit[1]);
        for (unsigned& i : idx) i = 15 - i;
    }
    std::fill(out, out + 16, std::uint8_t(0));
    Bits bits{out};
    bits.put(1u << 6, 7);   // mode 6
    for (int k = 0; k < 4; ++k) bits.put(e7[0][k], 7), bits.put(e7[1][k], 7);
    bits.put(pbit[0], 1);
    bits.put(pbit[1], 1);
    for (unsigned i = 0; i < 16; ++i) bits.put(idx[i], i ? 4 : 3);
}

inline void decode_bc7(const std::uint8_t* in, Color out[16]) {
    unsigned at = 7;   // the encoder writes mode 6 only
    int e[2][4];
    for (int k = 0; k < 4; ++k) e[0][k] = int(get_bits(in, at, 7)), e[1][k] = int(get_bits(in, at, 7));
    const unsigned p0 = get_bits(in, at, 1), p1 = get_bits(in, at, 1);
    for (int k = 0; k < 4; ++k) e[0][k] = e[0][k] << 1 | int(p0), e[1][k] = e[1][k] << 1 | int(p1);
    for (unsigned i = 0; i < 16; ++i) {
        const unsigned w = kWeights4[get_bits(in, at, i ? 4 : 3)];
        for (int k = 0; k < 4; ++k)
            out[i][k] = float(((64 - int(w)) * e[0][k] + int(w) * e[1][k] + 32) >> 6) / 255.0f;
    }
}

}  // namespace bc

// The texture of pattern (whose sides are powers of two, at least 4) with every mip level down to 1x1.
inline BcTexture make_texture(const Image8& pattern, BcFormat format) {
    BcTexture tex{format, pattern.width, pattern.height, {}, {}};
    Image8 level = pattern;
    for (;;) {
        const unsigned bw = (level.width + 3) / 4, bh = (level.height + 3) / 4;
        std::vector<std::uint8_t> blocks(std::size_t(bw) * bh * tex.block_bytes());
        std::vector<Color> decoded(std::size_t(level.width) * level.height);
        for (unsigned by = 0; by < bh; ++by)
            for (unsigned bx = 0; bx < bw; ++bx) {
                std::uint8_t* b = blocks.data() + (std::size_t(by) * bw + bx) * tex.block_bytes();
                Color texels[16];
                if (format == BcFormat::bc1) {
                    bc::encode_bc1(bc::gather(level, bx, by), b);
                    bc::decode_bc1(b, texels);
                } else {
                    bc::encode_bc7(bc::gather(level, bx, by), b);
                    bc::decode_bc7(b, texels);
                }
                for (unsigned i = 0; i < 16; ++i) {
                    const unsigned x = bx * 4 + i % 4, y = by * 4 + i / 4;
                    if (x < level.width && y < level.height) decoded[std::size_t(y) * level.width + x] = texels[i];
                }
            }
        tex.blocks.push_back(std::move(blocks));
        tex.decoded.push_back(std::move(decoded));
        if (level.width == 1 && level.height == 1) break;
        level = half_size(level);
    }
    return tex;
}

}  // namespace lanterns
