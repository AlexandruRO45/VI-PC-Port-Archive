// Lanterns stage 10: 4x MSAA. The stage-1 scene drawn into a 4-sample R32G32B32A32_FLOAT target with a 4-sample D32
// depth buffer, resolved with ResolveSubresource into a single-sample target. The CPU rasterizes the scene once per
// sample at D3D12's standard 4x positions (offsets from the pixel centre in 1/16 pixel: (-2,-6), (6,-2), (-6,2),
// (2,6); Vulkan's standard locations are the same), with coverage and depth at the sample and the colour of a pixel
// shader run once per pixel (at the centre). The resolve is the samples' average; the colours are short binary
// fractions, so the sum of four is exact in float in any order. Where a sample is uncertain, the resolve must equal
// one of the averages its candidates can make. Drawn again with the sample mask 1 << s, the resolve isolates sample s.
#pragma once

#include "lanterns_sampling.h"

#include <algorithm>
#include <array>

namespace lanterns {

constexpr double kMsaaOffsets[4][2] = {{-0.125, -0.375}, {0.375, -0.125}, {-0.375, 0.125}, {0.125, 0.375}};

struct MsaaReference {
    TexturedReference resolved;   // bounds of the average; certain where every sample has one colour
    std::vector<std::uint8_t> blended;   // certain pixels whose four samples are not all one colour (the edges)
    // Uncertain pixels: every average the four samples' candidates can make (each sample one of its candidates), so
    // the resolve must equal one of them exactly; empty where certain.
    std::vector<std::vector<Color>> averages;
    std::array<Reference, 4> per_sample;   // the frame rasterized at each standard sample position alone
};

constexpr unsigned kMsaaMaxAverages = 4096;   // the most combinations enumerated per pixel (the scene needs far fewer)

namespace detail {

// Every exact average of one candidate per sample (a certain sample has its one colour). The colours are short binary
// fractions, so each sum of four is exact in float.
inline std::vector<Color> sample_averages(const std::array<Reference, 4>& per, std::size_t i) {
    std::vector<Color> lists[4];
    std::size_t combos = 1;
    for (int s = 0; s < 4; ++s) {
        lists[s] = per[s].candidates[i].empty() ? std::vector<Color>{per[s].color[i]} : per[s].candidates[i];
        combos *= lists[s].size();
    }
    std::vector<Color> out;
    if (combos > kMsaaMaxAverages) return out;
    for (std::size_t c = 0; c < combos; ++c) {
        std::size_t rest = c;
        Color sum{0, 0, 0, 0};
        for (int s = 0; s < 4; ++s) {
            const Color& pick = lists[s][rest % lists[s].size()];
            rest /= lists[s].size();
            for (int k = 0; k < 4; ++k) sum[k] += pick[k];
        }
        for (int k = 0; k < 4; ++k) sum[k] *= 0.25f;
        add_unique(out, sum);
    }
    return out;
}

}  // namespace detail

// The resolve of a frame drawn with the PSO's sample mask 1 << s: sample s as the scene puts it, the other three
// keeping the clear colour, so (colour + 3 clear) / 4; its candidates where sample s is uncertain.
inline Reference masked_resolve(const Reference& sample) {
    const Color clear{kClear[0], kClear[1], kClear[2], kClear[3]};
    const auto mix = [&](const Color& c) {
        Color out;
        for (int k = 0; k < 4; ++k) out[k] = (c[k] + 3.0f * clear[k]) * 0.25f;
        return out;
    };
    Reference out = sample;
    for (auto& c : out.color) c = mix(c);
    for (auto& list : out.candidates)
        for (auto& c : list) c = mix(c);
    return out;
}

inline MsaaReference msaa_reference(const Scene& scene) {
    std::array<Reference, 4> per;
    for (int s = 0; s < 4; ++s) per[s] = reference_frame(scene, true, kMsaaOffsets[s][0], kMsaaOffsets[s][1]);
    const std::size_t n = per[0].color.size();
    MsaaReference out;
    out.resolved.bounds.resize(n);
    out.resolved.certain.resize(n);
    out.blended.resize(n);
    out.averages.resize(n);
    for (std::size_t i = 0; i < n; ++i) {
        double lo[4] = {0, 0, 0, 0}, hi[4] = {0, 0, 0, 0};
        bool certain = true, mixed = false;
        for (int s = 0; s < 4; ++s) {
            const auto& cand = per[s].candidates[i];
            if (cand.empty()) {
                for (int k = 0; k < 4; ++k) lo[k] += per[s].color[i][k], hi[k] += per[s].color[i][k];
                mixed = mixed || !detail::same(per[s].color[i], per[0].color[i]);
                continue;
            }
            certain = false;
            for (int k = 0; k < 4; ++k) {
                double a = cand[0][k], b = cand[0][k];
                for (const Color& c : cand) a = std::min(a, double(c[k])), b = std::max(b, double(c[k]));
                lo[k] += a, hi[k] += b;
            }
        }
        for (int k = 0; k < 4; ++k)
            out.resolved.bounds[i].lo[k] = float(lo[k] / 4), out.resolved.bounds[i].hi[k] = float(hi[k] / 4);
        out.resolved.certain[i] = certain;
        out.blended[i] = certain && mixed;
        if (!certain) out.averages[i] = detail::sample_averages(per, i);
    }
    out.per_sample = std::move(per);
    return out;
}

}  // namespace lanterns
