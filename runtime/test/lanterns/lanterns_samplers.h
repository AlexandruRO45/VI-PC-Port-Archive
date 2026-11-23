// Lanterns stage 20: samplers a shader builds from immediates (cs_lanterns_samplers.asm), and their CPU reference.
// The console's compiler turns a root signature's static samplers into literal S#s: s_mov_b32 of four immediates into
// SGPRs, then image_sample. The layer has no heap slot or root argument to find them by, so it registers the words it
// finds in the shader when the pipeline is created. Three literals, all clamping, MIP_POINT_PRECLAMP set and
// MIP_FILTER point as the title's literals have them: POINT, LINEAR (bilinear) and a LESS_EQUAL comparison with point
// filtering. One lane per probe; each lane samples:
//   the colour texture through POINT and through LINEAR,
//   the depth texture through the comparison sampler, and through POINT (a plain sample must stay a plain sample).
// Every probe sits a quarter or three quarters of a texel from the texel edges, so POINT never meets a tie and the
// bilinear weights are 1/4 and 3/4, which every filter unit carries exactly; the texels are small integers, so every
// expected value is exact.
#pragma once

#include <array>
#include <cmath>
#include <cstdint>

namespace lanterns {

constexpr unsigned kSamplerSize = 4;      // both textures are 4x4
constexpr unsigned kSamplerProbes = 32;   // one per lane of the one wave32 group

// The colour texture (R32_FLOAT): texel (x, y) holds 1 + x + 4y.
inline float sampler_texel(int x, int y) { return float(1 + x + 4 * y); }

// The depth texture (D32_FLOAT, read through an R32_FLOAT view): one depth per 2x2 quadrant, cleared with rectangles.
constexpr std::array<float, 4> kSamplerDepth = {0.2f, 0.4f, 0.6f, 0.8f};
inline float sampler_depth(int x, int y) { return kSamplerDepth[std::size_t((x >= 2) + 2 * (y >= 2))]; }

// Lane i's probe, as the shader computes it: x and y in texels, and the comparison value.
struct SamplerProbe {
    float x = 0, y = 0, ref = 0;
};
inline SamplerProbe sampler_probe(unsigned lane) {
    SamplerProbe p;
    p.x = std::fma(float(lane & 7u), 0.5f, 0.25f);
    p.y = std::fma(float((lane >> 2) & 1u), 0.5f, float(lane >> 3)) + 0.25f;
    p.ref = std::fma(0.2f, float(lane & 3u), 0.3f);
    return p;
}

// What each sample must return: [POINT, LINEAR, LESS_EQUAL comparison, POINT on the depth texture].
inline std::array<float, 4> sampler_expect(unsigned lane) {
    const SamplerProbe p = sampler_probe(lane);
    const int xi = int(std::floor(p.x)), yi = int(std::floor(p.y));
    const auto clamped = [](int v) { return v < 0 ? 0 : v > int(kSamplerSize) - 1 ? int(kSamplerSize) - 1 : v; };
    const auto t = [&](int x, int y) { return sampler_texel(clamped(x), clamped(y)); };
    const float tx = p.x - 0.5f, ty = p.y - 0.5f;
    const int x0 = int(std::floor(tx)), y0 = int(std::floor(ty));
    const float fx = tx - float(x0), fy = ty - float(y0);
    const float linear = (1 - fy) * ((1 - fx) * t(x0, y0) + fx * t(x0 + 1, y0)) +
                         fy * ((1 - fx) * t(x0, y0 + 1) + fx * t(x0 + 1, y0 + 1));
    const float depth = sampler_depth(xi, yi);
    return {sampler_texel(xi, yi), linear, p.ref <= depth ? 1.0f : 0.0f, depth};
}

}  // namespace lanterns
