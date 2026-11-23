// Lanterns stage 8: shadows from a directional light. A depth-only pass draws the solids (the floor receives but does
// not cast) from the light, through an orthographic light matrix, into a 512x512 D32 shadow map, with the PSO's depth
// bias (constant and slope-scaled). The main pass's pixel shader moves its world position into light space and takes
// four comparison samples (LESS_EQUAL, point filtered) half a texel apart: 2x2 PCF, so a pixel is lit by 0, 1/4, 2/4,
// 3/4 or all of the light. Colour = face colour * (3/8 + 5/8 * lit fraction), exact in float.
//
// The reference rasterizes the shadow map too, but a texel's stored depth is only known within bounds: coverage at a
// triangle edge, the depth tolerance, and a bias anywhere in what D3D12 and Vulkan allow (the slope factor m between the
// largest and the sum of |dz/dx| and |dz/dy|, the constant term between 0 and DepthBias * 2^-23). A tap is certainly lit
// or certainly shadowed when the receiver's light-space depth is clear of those bounds, and the texel a tap reads is
// uncertain within 1/64 texel of a texel border. Pixels where every tap is certain have one exact colour.
#pragma once

#include "lanterns_sampling.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

namespace lanterns {

constexpr unsigned kShadowSize = 512;
constexpr int kShadowDepthBias = 1000;
constexpr float kShadowSlopeBias = 3.0f;
constexpr float kShadowAmbient = 0.375f, kShadowStep = 0.15625f;   // shade = 3/8 + 5/32 * lit taps
constexpr double kShadowTexelSlack = 1.0 / 64.0, kShadowDepthSlack = 1e-5;

struct Light {
    std::array<float, 16> clip;   // world -> light clip space (x, y in -1..1, z in 0..1, w = 1), row-major
    std::array<float, 12> rows;   // world -> (u, v, depth): the shadow map coordinates the pixel shader computes
};

inline Light make_light() {
    const double dir_raw[3] = {0.45, -1.0, -0.55}, centre[3] = {0.0, 0.5, 3.0}, half = 10.0, depth = 30.0;
    double d[3], len = 0;
    for (int k = 0; k < 3; ++k) len += dir_raw[k] * dir_raw[k];
    for (int k = 0; k < 3; ++k) d[k] = dir_raw[k] / std::sqrt(len);
    // right = normalize(d x up), up' = right x d
    double r[3] = {d[1] * 0 - d[2] * 1, d[2] * 0 - d[0] * 0, d[0] * 1 - d[1] * 0};
    const double rl = std::sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2]);
    for (double& x : r) x /= rl;
    const double u[3] = {r[1] * d[2] - r[2] * d[1], r[2] * d[0] - r[0] * d[2], r[0] * d[1] - r[1] * d[0]};
    const auto dot = [&](const double* a) { return a[0] * centre[0] + a[1] * centre[1] + a[2] * centre[2]; };
    double clip[4][4] = {{r[0] / half, r[1] / half, r[2] / half, -dot(r) / half},
                         {u[0] / half, u[1] / half, u[2] / half, -dot(u) / half},
                         {d[0] / depth, d[1] / depth, d[2] / depth, -dot(d) / depth + 0.5},
                         {0, 0, 0, 1}};
    Light light;
    for (int i = 0; i < 16; ++i) light.clip[i] = float(clip[i / 4][i % 4]);
    for (int k = 0; k < 4; ++k) {
        light.rows[k] = float(0.5 * light.clip[k] + (k == 3 ? 0.5 : 0.0));        // u = 0.5 x + 0.5
        light.rows[4 + k] = float(-0.5 * light.clip[4 + k] + (k == 3 ? 0.5 : 0.0));  // v = 0.5 - 0.5 y
        light.rows[8 + k] = light.clip[8 + k];                                       // depth = z
    }
    return light;
}

// What a shadow-map texel can hold (depth plus bias): [lo, hi]. A texel no solid certainly covers may hold 1 (clear).
struct ShadowMap {
    std::vector<double> lo, hi;
};

inline ShadowMap shadow_map(const Scene& scene, const Light& light, bool casters) {
    ShadowMap map{std::vector<double>(kShadowSize * kShadowSize, 1.0), std::vector<double>(kShadowSize * kShadowSize, 1.0)};
    if (!casters) return map;
    Scene from_light = scene;
    for (auto& k : from_light.view_projection) k = 0;
    std::copy(light.clip.begin(), light.clip.end(), from_light.view_projection.begin());
    for (Draw& d : from_light.draws) d.hidden = d.hidden || d.floor;
    unsigned degenerate = 0;
    const auto frags = detail::fragments(
        from_light, [](const PixelSite&, detail::Fragment&) {}, degenerate, kShadowSize, kShadowSize);
    for (std::size_t i = 0; i < frags.size(); ++i) {
        double lo = 1.0, hi = 1.0;
        for (const auto& f : frags[i]) {
            const double slope_sum = (f.tol - 1e-6) / kMargin;   // |dz/dx| + |dz/dy| per texel (see fragments())
            const double bias_lo = kShadowSlopeBias * slope_sum / 2;
            const double bias_hi = kShadowSlopeBias * slope_sum + kShadowDepthBias * std::ldexp(1.0, -23);
            lo = std::min(lo, f.depth - f.tol + bias_lo);
            if (f.certain_cover) hi = std::min(hi, f.depth + f.tol + bias_hi);
        }
        map.lo[i] = lo, map.hi[i] = std::max(lo, hi);
    }
    return map;
}

// The lit fraction's bounds (0..4 taps) at a world position.
inline void lit_taps(const ShadowMap& map, const Light& light, const double w[3], int& lo, int& hi) {
    const auto row = [&](int r) {
        return light.rows[r * 4] * w[0] + light.rows[r * 4 + 1] * w[1] + light.rows[r * 4 + 2] * w[2] + light.rows[r * 4 + 3];
    };
    const double u = row(0) * kShadowSize, v = row(1) * kShadowSize, depth = row(2);
    lo = hi = 0;
    for (int tap = 0; tap < 4; ++tap) {
        const double x = u + (tap & 1 ? 0.5 : -0.5), y = v + (tap & 2 ? 0.5 : -0.5);
        bool may_lit = false, may_shadow = false;
        for (double ex : {-kShadowTexelSlack, kShadowTexelSlack})
            for (double ey : {-kShadowTexelSlack, kShadowTexelSlack}) {
                const int tx = std::clamp(int(std::floor(x + ex)), 0, int(kShadowSize) - 1);
                const int ty = std::clamp(int(std::floor(y + ey)), 0, int(kShadowSize) - 1);
                const std::size_t t = std::size_t(ty) * kShadowSize + tx;
                if (depth - kShadowDepthSlack <= map.hi[t]) may_lit = true;     // depth <= stored may hold
                if (depth + kShadowDepthSlack > map.lo[t]) may_shadow = true;   // depth > stored may hold
            }
        lo += may_shadow ? 0 : 1;
        hi += may_lit ? 1 : 0;
    }
}

inline float shadow_shade_factor(int lit) { return std::fma(float(lit), kShadowStep, kShadowAmbient); }

inline TexturedReference shadow_reference(const Scene& scene, const Light& light, const ShadowMap& map) {
    unsigned degenerate = 0;
    const auto shade = [&](const PixelSite& site, detail::Fragment& f) {
        double w[3];
        site.world(site.px, site.py, w);
        int lo, hi;
        lit_taps(map, light, w, lo, hi);
        const float a = shadow_shade_factor(lo), b = shadow_shade_factor(hi);
        for (int k = 0; k < 3; ++k) f.colors[0][k] = site.lit[k] * a, f.colors[1][k] = site.lit[k] * b;
        f.colors[0][3] = f.colors[1][3] = site.lit[3];
        f.count = 2;
    };
    return resolve_bounds(detail::fragments(scene, shade, degenerate));
}

// The stage-8 scene: stage 1's solids on a plain floor (the shadow pixel shader has no checker).
inline Scene stage8() { return stage1(); }

}  // namespace lanterns
