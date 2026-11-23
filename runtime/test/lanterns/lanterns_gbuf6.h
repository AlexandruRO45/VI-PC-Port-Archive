// Lanterns stage 22: a G-buffer of five or six targets written through packed half-float exports, and its decode
// (ps_lanterns_gbuf6.asm, cs_lanterns_gbuf_decode.asm). The title's deferred pixel shaders write five or six targets
// and pack every colour export as half floats; a lighting pass reads them back. Here every value comes from the pixel's
// integer position, so the reference is exact:
//   - what each target must hold, byte for byte (the sRGB albedo within one code of the encoding, as D3D allows);
//   - what the decode pass must write, 32 bytes a pixel.
// The byte IDs survive the round trip only because their target is RGBA8: an f16 rounded toward zero, stored in a
// binary16 target and decoded by multiplying by 255 and truncating, loses one in 254 of 256 values (a nearest-integer
// decoder would recover them, but the truncating one is what the check uses).
#pragma once

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>

namespace lanterns {

constexpr unsigned kGbufTargets = 6;

// f32 -> f16 bits, rounding toward zero (v_cvt_pkrtz_f16_f32); the values here are finite and inside the f16 range.
inline std::uint16_t half_rtz_bits(float f) {
    std::uint32_t u = 0;
    std::memcpy(&u, &f, 4);
    const std::uint32_t sign = (u >> 16) & 0x8000u;
    const int exp = int((u >> 23) & 0xffu) - 127 + 15;
    const std::uint32_t mant = u & 0x7fffffu;
    if ((u & 0x7fffffffu) == 0) return std::uint16_t(sign);
    if (exp <= 0) {                         // subnormal f16: shift the full mantissa, dropping bits (toward zero)
        if (exp < -10) return std::uint16_t(sign);
        return std::uint16_t(sign | ((mant | 0x800000u) >> (14 - exp)));
    }
    return std::uint16_t(sign | (std::uint32_t(exp) << 10) | (mant >> 13));
}

inline float half_to_float(std::uint16_t h) {
    const int exp = (h >> 10) & 0x1f;
    const float mant = float(h & 0x3ff);
    const float v = exp == 0 ? std::ldexp(mant, -24) : std::ldexp(1.0f + mant / 1024.0f, exp - 15);
    return (h & 0x8000) ? -v : v;
}

inline float half_rtz(float f) { return half_to_float(half_rtz_bits(f)); }

// What the hardware stores in an n-bit UNORM channel for value v (round to nearest).
inline std::uint32_t unorm(float v, unsigned bits) {
    const float scale = float((1u << bits) - 1);
    const float c = std::fmin(std::fmax(v, 0.0f), 1.0f) * scale;
    return std::uint32_t(c + 0.5f);
}

inline std::uint8_t srgb_code(float linear) {
    const double c = std::fmin(std::fmax(double(linear), 0.0), 1.0);
    const double s = c <= 0.0031308 ? 12.92 * c : 1.055 * std::pow(c, 1.0 / 2.4) - 0.055;
    return std::uint8_t(s * 255.0 + 0.5);
}

struct GbufPixel {
    std::array<std::uint8_t, 4> ids{}, ids2{}, albedo{};   // mrt0, mrt5, mrt3 (sRGB codes)
    std::uint32_t codes = 0;                               // mrt1 as stored (RGB10A2)
    std::array<std::uint16_t, 2> motion{};                 // mrt2 halves
    std::array<std::uint16_t, 4> emissive{};               // mrt4 halves
    std::array<std::uint32_t, 8> decoded{};                // what the decode pass writes
};

inline GbufPixel gbuf_expect(unsigned x, unsigned y, bool six) {
    const float inv255 = 1.0f / 255.0f, inv1023 = 1.0f / 1023.0f, inv3 = 1.0f / 3.0f;
    const auto stored8 = [&](unsigned n) { return std::uint8_t(unorm(half_rtz(float(n) * inv255), 8)); };
    GbufPixel p;
    const unsigned id[4] = {x & 255u, y & 255u, (x + 3 * y) & 255u, (x ^ y) & 255u};
    const unsigned id2[4] = {(7 * x) & 255u, (5 * y) & 255u, (11 * (x + y)) & 255u, 255u - (x & 255u)};
    for (unsigned c = 0; c < 4; ++c) p.ids[c] = stored8(id[c]), p.ids2[c] = six ? stored8(id2[c]) : 0;
    const unsigned code[3] = {x & 511u, (2 * y) & 511u, (x + y) & 511u};
    std::uint32_t stored_code[3];
    for (unsigned c = 0; c < 3; ++c) stored_code[c] = unorm(half_rtz(float(code[c]) * inv1023), 10);
    p.codes = stored_code[0] | stored_code[1] << 10 | stored_code[2] << 20 | unorm(half_rtz(float(x & 3u) * inv3), 2) << 30;
    p.motion = {half_rtz_bits((float(int(x) - 192)) * 0.125f), half_rtz_bits(float(128 - int(y)) * 0.0625f)};
    const unsigned alb[3] = {x & 255u, y & 255u, (x * y) & 255u};
    for (unsigned c = 0; c < 3; ++c) p.albedo[c] = srgb_code(half_rtz(float(alb[c]) * inv255));
    p.albedo[3] = 255;
    const float e0 = half_rtz(float((x + y) & 511u) * 0.25f);
    p.emissive = {half_rtz_bits(e0), half_rtz_bits(float(x & 255u) * -0.5f), half_rtz_bits(float(y & 127u) * 2.0f),
                  half_rtz_bits(1.0f)};
    // The decode: what the targets hold, read back as floats (UNORM n / (2^bits - 1), f16 as it is).
    const auto ids_word = [&](const std::array<std::uint8_t, 4>& b) {
        std::uint32_t w = 0;
        for (unsigned c = 0; c < 4; ++c) w |= std::uint32_t(float(b[c]) / 255.0f * 255.0f) << (8 * c);
        return w;
    };
    p.decoded[0] = ids_word(p.ids);
    std::uint32_t dcode[3];
    for (unsigned c = 0; c < 3; ++c) dcode[c] = std::uint32_t(std::fma(float(stored_code[c]) / 1023.0f, 1023.0f, 0.5f));
    p.decoded[1] = dcode[0] | dcode[1] << 10 | dcode[2] << 20;
    const float mx = half_to_float(p.motion[0]), my = half_to_float(p.motion[1]);
    std::memcpy(&p.decoded[2], &mx, 4);
    std::memcpy(&p.decoded[3], &my, 4);
    const float lit = std::fma(std::fma(float(p.decoded[0] & 255u), float(dcode[0]), float(dcode[1])), 0.25f, e0);
    std::memcpy(&p.decoded[4], &lit, 4);
    p.decoded[5] = ids_word(p.ids2);
    return p;
}

}  // namespace lanterns
