// Lanterns stage 5: the post-process and its CPU reference. The scene is drawn into an R16G16B16A16_FLOAT target, as
// a game draws into an HDR target. Then:
//   cs_lanterns_bloom_h  (wave64, 128 lanes: two waves sharing LDS) keeps what is brighter than kBloomThreshold,
//                        quantized to 1/kBloomSteps, and blurs it along rows with the 9-tap binomial filter;
//   cs_lanterns_bloom_v  (the same, along columns) blurs that result;
//   ps_lanterns_composite (fullscreen, SV_Position) adds the blur times kBloomGain to the scene, into the final target.
// The arithmetic is exact by construction: a quantized value is a small integer (q < 64), each blur sums integer
// multiples of it (weights 1..70, total 256), so both sums are integers below 2^20, and the composite adds an
// integer times 2^-21 to an fp16 value. post_reference() computes it in float as the shaders do, checks every value
// against the same computation in double, and says per pixel either "this exact colour" or "between these bounds"
// (where the stage-1 reference is uncertain: the bounds come from its candidates, through the monotone blur).
#pragma once

#include "lanterns_reference.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

namespace lanterns {

constexpr float kBloomThreshold = 0.625f;
constexpr float kBloomSteps = 64.0f;               // quantization of (colour - threshold)
constexpr int kBloomRadius = 4;                    // 9 taps
constexpr float kBloomWeights[2 * kBloomRadius + 1] = {1, 8, 28, 56, 70, 56, 28, 8, 1};
constexpr float kBloomGain = 0x1p-19f;             // 8 / (64 * 256 * 256): a fully bright area adds 8 * (c - 0.625)

// float -> fp16 bits, rounding to nearest even (what a render target write of a float16 format does).
inline std::uint16_t half_bits(float value) {
    std::uint32_t x;
    std::memcpy(&x, &value, 4);
    const std::uint32_t sign = (x >> 16) & 0x8000u, mag = x & 0x7fffffffu;
    if (mag >= 0x7f800000u) return std::uint16_t(sign | 0x7c00u | (mag > 0x7f800000u ? 0x200u : 0u));
    if (mag >= 0x477ff000u) return std::uint16_t(sign | 0x7c00u);               // rounds past 65504
    if (mag < 0x38800000u)                                                     // below 2^-14: subnormal or zero
        return std::uint16_t(sign | std::uint32_t(std::nearbyint(std::fabs(value) * 0x1p24f)));
    return std::uint16_t(sign | ((mag - (112u << 23) + 0xfffu + ((mag >> 13) & 1u)) >> 13));
}

inline float half_value(std::uint16_t h) {
    const std::uint32_t sign = std::uint32_t(h & 0x8000u) << 16, exponent = (h >> 10) & 0x1fu, mantissa = h & 0x3ffu;
    float magnitude;
    if (exponent == 0) magnitude = float(mantissa) * 0x1p-24f;
    else if (exponent == 31) magnitude = mantissa ? NAN : INFINITY;
    else magnitude = std::ldexp(float(mantissa | 0x400u), int(exponent) - 25);
    std::uint32_t bits;
    std::memcpy(&bits, &magnitude, 4);
    bits |= sign;
    float out;
    std::memcpy(&out, &bits, 4);
    return out;
}

inline Color to_half(const Color& c) {
    return {half_value(half_bits(c[0])), half_value(half_bits(c[1])), half_value(half_bits(c[2])),
            half_value(half_bits(c[3]))};
}

// The stage-1 reference of a frame drawn into the fp16 target: every colour and candidate rounded to fp16.
inline Reference half_reference(const Reference& ref) {
    Reference out = ref;
    for (auto& c : out.color) c = to_half(c);
    for (auto& list : out.candidates) {
        std::vector<Color> rounded;
        for (const Color& c : list) detail::add_unique(rounded, to_half(c));
        list = rounded.size() > 1 ? std::move(rounded) : std::vector<Color>{};
    }
    return out;
}

// What cs_lanterns_bloom_h does to one channel before the blur: floor(max(c - threshold, 0) * steps).
inline float bloom_quantize(float c) { return std::floor(std::max(c - kBloomThreshold, 0.0f) * kBloomSteps); }

struct PostReference {
    std::vector<Color> lo, hi;            // bounds of the final colour, per channel; equal where certain
    std::vector<std::uint8_t> certain;
    unsigned bloom_pixels = 0;            // certain pixels the bloom changes: what skipping the blur must change
    unsigned inexact = 0;                 // values where float and double disagree (0, by construction)
};

namespace detail {

// One separable pass over per-channel bounds: out(x, y) = sum_k w_k in(x + (k - r) * step, clamped).
inline void blur(const std::vector<Color>& in, std::vector<Color>& out, bool rows, unsigned& inexact) {
    out.assign(in.size(), Color{});
    for (unsigned y = 0; y < kHeight; ++y)
        for (unsigned x = 0; x < kWidth; ++x)
            for (int ch = 0; ch < 3; ++ch) {
                float acc = 0.0f;
                double exact = 0.0;
                for (int k = 0; k <= 2 * kBloomRadius; ++k) {
                    const int d = k - kBloomRadius;
                    const int sx = rows ? std::clamp(int(x) + d, 0, int(kWidth) - 1) : int(x);
                    const int sy = rows ? int(y) : std::clamp(int(y) + d, 0, int(kHeight) - 1);
                    const float v = in[std::size_t(sy) * kWidth + sx][ch];
                    acc = std::fma(kBloomWeights[k], v, acc);
                    exact += double(kBloomWeights[k]) * v;
                }
                inexact += double(acc) != exact;
                out[std::size_t(y) * kWidth + x][ch] = acc;
            }
}

}  // namespace detail

// The final frame of stage 5 for the scene whose stage-1 reference (depth test on) is ref.
inline PostReference post_reference(const Reference& ref) {
    const Reference scene = half_reference(ref);
    const std::size_t n = scene.color.size();
    std::vector<Color> qlo(n), qhi(n), slo(n), shi(n);
    for (std::size_t i = 0; i < n; ++i) {
        const auto& cand = scene.candidates[i];
        const std::vector<Color> one{scene.color[i]};
        const auto& list = cand.empty() ? one : cand;
        for (int ch = 0; ch < 4; ++ch) {
            float lo = INFINITY, hi = -INFINITY;
            for (const Color& c : list) lo = std::min(lo, c[ch]), hi = std::max(hi, c[ch]);
            slo[i][ch] = lo, shi[i][ch] = hi;
            qlo[i][ch] = ch < 3 ? bloom_quantize(lo) : 0.0f;
            qhi[i][ch] = ch < 3 ? bloom_quantize(hi) : 0.0f;
        }
    }
    PostReference out;
    std::vector<Color> hlo, hhi, vlo, vhi;
    detail::blur(qlo, hlo, true, out.inexact);
    detail::blur(qhi, hhi, true, out.inexact);
    detail::blur(hlo, vlo, false, out.inexact);
    detail::blur(hhi, vhi, false, out.inexact);
    out.lo.resize(n), out.hi.resize(n), out.certain.resize(n);
    for (std::size_t i = 0; i < n; ++i) {
        bool certain = scene.candidates[i].empty(), changes = false;
        for (int ch = 0; ch < 4; ++ch) {
            const float lo = std::fma(vlo[i][ch], kBloomGain, slo[i][ch]);
            const float hi = std::fma(vhi[i][ch], kBloomGain, shi[i][ch]);
            out.inexact += double(lo) != double(vlo[i][ch]) * kBloomGain + slo[i][ch];
            out.inexact += double(hi) != double(vhi[i][ch]) * kBloomGain + shi[i][ch];
            out.lo[i][ch] = lo, out.hi[i][ch] = hi;
            certain = certain && vlo[i][ch] == vhi[i][ch];
            changes = changes || vlo[i][ch] != 0.0f;
        }
        out.certain[i] = certain;
        out.bloom_pixels += certain && changes;
    }
    return out;
}

// Certain pixels must equal the reference bit for bit; the others must lie within their bounds.
inline Comparison compare_post(const std::vector<Color>& gpu, const PostReference& ref) {
    Comparison c;
    bool reported = false;
    for (std::size_t i = 0; i < gpu.size(); ++i) {
        bool ok;
        if (ref.certain[i]) {
            ++c.certain;
            ok = detail::same(gpu[i], ref.lo[i]);
            c.certain_exact += ok;
        } else {
            ++c.uncertain;
            ok = true;
            for (int ch = 0; ch < 4; ++ch) ok = ok && gpu[i][ch] >= ref.lo[i][ch] && gpu[i][ch] <= ref.hi[i][ch];
            c.uncertain_plausible += ok;
        }
        if (!ok && !reported) {
            reported = true;
            c.first_bad_x = unsigned(i % kWidth), c.first_bad_y = unsigned(i / kWidth);
            c.first_bad_gpu = gpu[i], c.first_bad_ref = ref.lo[i];
        }
    }
    return c;
}

// How many certain pixels of a frame differ from the reference (the skipped-blur control must differ at exactly
// ref.bloom_pixels of them).
inline unsigned certain_differences(const std::vector<Color>& frame, const PostReference& ref) {
    unsigned n = 0;
    for (std::size_t i = 0; i < frame.size(); ++i) n += ref.certain[i] && !detail::same(frame[i], ref.lo[i]);
    return n;
}

}  // namespace lanterns
