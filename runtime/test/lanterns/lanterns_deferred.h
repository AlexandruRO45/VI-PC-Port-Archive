// Lanterns stage 9: deferred lighting. A G-buffer pass writes three render targets: the albedo (RGBA8_UNORM, the
// draw's tint times the instance colour), the face normal (R10G10B10A2_UNORM, from a normal packed into the vertex's
// shade word as three 8-bit codes) and the view depth (R32_FLOAT, the clip w). A compute pass, one 16x16 workgroup per
// screen tile, rebuilds each pixel's world position from the depth, takes the tile's bounding box with LDS min/max
// atomics, keeps the point lights whose sphere meets the box in an LDS list (ds_add_rtn_u32 hands out the slots), and
// shades each pixel with the kept lights in integers:
//   q = floor(64 * max(0, 1 - d^2 / r^2)) for a light the face looks towards (n . (L - p) > 0), else 0
//   out = min(255, (albedo8 * (384 + sum q * colour)) >> 12) per channel
// The sums are integers, so the order the atomics give the list does not matter: with culling off every light is in
// every list and the image must be the same, bit for bit.
//
// The CPU knows every pixel's world position up to kPosSlack (interpolation and reconstruction in float) and computes
// each light's q as an interval from it; a pixel whose intervals are all one value has one exact colour.
#pragma once

#include "lanterns_sampling.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <vector>

namespace lanterns {

constexpr unsigned kLights = 256, kTile = 16, kTilesX = kWidth / kTile, kTilesY = kHeight / kTile;
constexpr float kLightRadius2 = 1.5625f, kLightInvRadius2 = 0.64f;
constexpr float kCullRadius2 = kLightRadius2 * 1.001f + 0.001f;   // the GPU's box test, a little generous
constexpr std::uint32_t kAmbient = 384, kLightShift = 12;
constexpr std::uint32_t kBackground = 0xff201010u;   // r 16, g 16, b 32, a 255: where the view depth is 0 (cleared)
constexpr double kPosSlack = 5e-4;
// How far vertex snapping may move a face's plane at a pixel centre, in pixels: twice an 8-bit sub-pixel step (the
// coverage margin kMargin is eight times it). Stage 9's probes measure the real error against it (at most 0.24 of it).
constexpr double kSnapSlack = 1.0 / 128.0;
constexpr double kNormalWindow = 0.1;   // the 10-bit normal codes: D3D12's 0.6 ULP conversion tolerance

// One light as cs_lanterns_tiled reads it through a V# of stride 32.
struct PointLight {
    float pos[4];
    std::uint32_t color[4];
};
static_assert(sizeof(PointLight) == 32);

// 256 lights over the floor, a 16 x 16 grid with jitter, heights 0.2 to 1.0, colours 1..15 per channel.
inline std::vector<PointLight> make_lights() {
    std::vector<PointLight> lights(kLights);
    for (unsigned i = 0; i < kLights; ++i) {
        const float jx = float(int((i * 37u) % 9u) - 4) * 0.0625f, jz = float(int((i * 23u) % 7u) - 3) * 0.0625f;
        lights[i].pos[0] = -6.0f + 0.75f * float(i % 16) + 0.375f + jx;
        lights[i].pos[1] = 0.2f + 0.2f * float((i * 7u) % 5u);
        lights[i].pos[2] = -1.5f + 0.625f * float(i / 16) + jz;
        lights[i].pos[3] = 0;
        lights[i].color[0] = 1 + (i * 7u) % 15u;
        lights[i].color[1] = 1 + (i * 11u) % 15u;
        lights[i].color[2] = 1 + (i * 13u) % 15u;
        lights[i].color[3] = 0;
    }
    return lights;
}

// The compute pass's 16 root constants: eye, the ray at the centre of the view (its clip w is 1), the ray's change
// per pixel right and down, 1/r^2, the mode (0 cull, 1 every light in every tile), the box test's r^2, 0.
// The eye and rays are solved from the view-projection rows x, y and w, so they match the float matrix.
inline std::array<float, 16> deferred_constants(const Scene& scene, std::uint32_t mode) {
    const auto& m = scene.view_projection;
    const double a[3][3] = {{m[0], m[1], m[2]}, {m[4], m[5], m[6]}, {m[12], m[13], m[14]}};
    const double det = a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1]) -
                       a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0]) +
                       a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);
    const auto solve = [&](const double b[3], double x[3]) {   // Cramer's rule
        for (int c = 0; c < 3; ++c) {
            double t[3][3];
            for (int r = 0; r < 3; ++r)
                for (int k = 0; k < 3; ++k) t[r][k] = k == c ? b[r] : a[r][k];
            x[c] = (t[0][0] * (t[1][1] * t[2][2] - t[1][2] * t[2][1]) - t[0][1] * (t[1][0] * t[2][2] - t[1][2] * t[2][0]) +
                    t[0][2] * (t[1][0] * t[2][1] - t[1][1] * t[2][0])) / det;
        }
    };
    const double be[3] = {-double(m[3]), -double(m[7]), -double(m[15])}, b0[3] = {0, 0, 1}, bx[3] = {1, 0, 0},
                 by[3] = {0, 1, 0};
    double eye[3], d0[3], dx[3], dy[3];
    solve(be, eye), solve(b0, d0), solve(bx, dx), solve(by, dy);
    std::array<float, 16> c{};
    for (int k = 0; k < 3; ++k) {
        c[k] = float(eye[k]);
        c[3 + k] = float(d0[k]);
        c[6 + k] = float(dx[k] * 2.0 / kWidth);
        c[9 + k] = float(-dy[k] * 2.0 / kHeight);
    }
    c[12] = kLightInvRadius2;
    std::uint32_t bits = mode;
    std::memcpy(&c[13], &bits, 4);
    c[14] = kCullRadius2;
    return c;
}

// A scene for vs_lanterns_gbuf (stage 9: stage 1's): each face's outward normal packed into its vertices' shade word as an integer
// How far, per axis, the world position the compute passes rebuild from the G-buffer's view depth may be from
// site.world at the pixel centre. The depth is the clip w at the centre of a plane the rasterizer may have moved by
// up to kSnapSlack pixels (vertex snapping), so it may be off by its change over that distance (plus 1e-5 relative);
// the rebuilt p = eye + w (p - eye) / w moves along the view ray by that much. kPosSlack covers the float arithmetic.
// Before this, a fixed kPosSlack was used, which steep faces exceed (stage 9 measures how often).
inline std::array<double, 3> position_slack(const Scene& scene, const PixelSite& site) {
    const auto& vp = scene.view_projection;
    const auto clip_w = [&](double px, double py, double p[3]) {
        site.world(px, py, p);
        return vp[12] * p[0] + vp[13] * p[1] + vp[14] * p[2] + vp[15];
    };
    double p[3], q[3];
    const double cw = clip_w(site.px, site.py, p);
    const double dw = (std::fabs(clip_w(site.px + 1, site.py, q) - cw) + std::fabs(clip_w(site.px, site.py + 1, q) - cw)) *
                          kSnapSlack + 1e-5 * cw;
    const auto k = deferred_constants(scene, 0);
    std::array<double, 3> out;
    for (int c = 0; c < 3; ++c) out[c] = kPosSlack + dw * std::fabs(p[c] - double(k[c])) / cw;
    return out;
}

// float: three codes 128 + round(127 n), x in bits 0-7, y 8-15, z 16-23 (below 2^24, so exact).
inline std::uint32_t pack_normal(double nx, double ny, double nz) {
    const double l = std::sqrt(nx * nx + ny * ny + nz * nz);
    const auto code = [&](double v) { return std::uint32_t(128 + std::lround(127.0 * v / l)); };
    return code(nx) | code(ny) << 8 | code(nz) << 16;
}

inline Scene with_packed_normals(Scene s) {
    for (Mesh& mesh : s.meshes)
        for (const Face& face : mesh.faces) {
            double n[3] = {0, 0, 0};
            for (unsigned i = 0; i < face.count; ++i) {   // Newell's method; faces are wound outward
                const Vertex& a = mesh.vertices[face.v[i]];
                const Vertex& b = mesh.vertices[face.v[(i + 1) % face.count]];
                n[0] += (double(a.y) - b.y) * (double(a.z) + b.z);
                n[1] += (double(a.z) - b.z) * (double(a.x) + b.x);
                n[2] += (double(a.x) - b.x) * (double(a.y) + b.y);
            }
            const float word = float(pack_normal(n[0], n[1], n[2]));
            for (unsigned i = 0; i < face.count; ++i) mesh.vertices[face.v[i]].shade = word;
        }
    return s;
}

inline Scene stage9() { return with_packed_normals(stage1()); }

// A float's UNORM code(s): one, or the two neighbours when x * max is within `window` of a rounding tie. D3D12 lets
// FLOAT -> UNORM conversion be off by 0.6 ULP (window 0.1); the albedo target is checked with a near-exact window.
inline void unorm_codes(float x, unsigned max, unsigned& lo, unsigned& hi, double window = 1e-3) {
    const double v = double(x) * max, f = std::floor(v);
    if (std::fabs(v - f - 0.5) < window) lo = unsigned(f), hi = unsigned(f) + 1;
    else lo = hi = unsigned(std::lround(v));
}

namespace detail {

struct Iv {
    double lo, hi;
};
inline Iv iv_mul(Iv a, Iv b) {
    const double p[4] = {a.lo * b.lo, a.lo * b.hi, a.hi * b.lo, a.hi * b.hi};
    return {*std::min_element(p, p + 4), *std::max_element(p, p + 4)};
}
inline Iv iv_sq(Iv a) {
    if (a.lo >= 0) return {a.lo * a.lo, a.hi * a.hi};
    if (a.hi <= 0) return {a.hi * a.hi, a.lo * a.lo};
    return {0, std::max(a.lo * a.lo, a.hi * a.hi)};
}
inline unsigned q_floor(double x) { return x <= 0 ? 0u : unsigned(std::min(64.0, std::floor(x))); }

// The albedo codes the G-buffer holds for a site (tint times instance colour, as vs_lanterns_gbuf multiplies).
inline void albedo_codes(const PixelSite& site, unsigned lo[4], unsigned hi[4]) {
    for (int k = 0; k < 4; ++k) unorm_codes(site.draw->tint[k] * site.inst->color[k], 255, lo[k], hi[k]);
}

}  // namespace detail

// The G-buffer's albedo target (RGBA8): its bounds (codes / 255), cleared to 0.
inline TexturedReference albedo_reference(const Scene& scene) {
    unsigned degenerate = 0;
    const auto shade = [](const PixelSite& site, detail::Fragment& f) {
        unsigned lo[4], hi[4];
        detail::albedo_codes(site, lo, hi);
        for (int k = 0; k < 4; ++k) f.colors[0][k] = float(lo[k]) / 255.0f, f.colors[1][k] = float(hi[k]) / 255.0f;
        f.count = 2;
    };
    return resolve_bounds(detail::fragments(scene, shade, degenerate), Color{0, 0, 0, 0});
}

// The lit image's bounds (codes / 255): every light, as the GPU with culling off.
inline TexturedReference deferred_reference(const Scene& scene, const std::vector<PointLight>& lights) {
    using detail::Iv;
    unsigned degenerate = 0;
    const auto shade = [&](const PixelSite& site, detail::Fragment& f) {
        unsigned alo[4], ahi[4];
        detail::albedo_codes(site, alo, ahi);
        // The normal: 8-bit code -> code / 255 in float -> a 10-bit UNORM code (or two) -> 2 f - 1.
        const auto packed = std::uint32_t(site.shade);
        Iv n[3];
        for (int k = 0; k < 3; ++k) {
            const float f8 = float((packed >> (8 * k)) & 255u) * (1.0f / 255.0f);
            unsigned lo, hi;
            unorm_codes(f8, 1023, lo, hi, kNormalWindow);
            n[k] = {2.0 * double(float(lo / 1023.0)) - 1.0 - 1e-6, 2.0 * double(float(hi / 1023.0)) - 1.0 + 1e-6};
        }
        double w[3];
        site.world(site.px, site.py, w);
        const auto slack = position_slack(scene, site);
        std::uint64_t acc_lo[3] = {0, 0, 0}, acc_hi[3] = {0, 0, 0};
        for (const PointLight& light : lights) {
            Iv lp[3];
            double d2_lo = 0;
            for (int k = 0; k < 3; ++k) {
                lp[k] = {light.pos[k] - (w[k] + slack[k]), light.pos[k] - (w[k] - slack[k])};
                d2_lo += detail::iv_sq(lp[k]).lo;
            }
            if (d2_lo * kLightInvRadius2 > 1.0001) continue;
            double d2_hi = 0, dot_lo = 0, dot_hi = 0;
            for (int k = 0; k < 3; ++k) {
                d2_hi += detail::iv_sq(lp[k]).hi;
                const Iv p = detail::iv_mul(n[k], lp[k]);
                dot_lo += p.lo, dot_hi += p.hi;
            }
            if (dot_hi < -1e-5) continue;
            unsigned q_lo = detail::q_floor((1.0 - d2_hi * 1.000001 * kLightInvRadius2) * 64.0 - 1e-3);
            const unsigned q_hi = detail::q_floor((1.0 - d2_lo * 0.999999 * kLightInvRadius2) * 64.0 + 1e-3);
            if (dot_lo <= 1e-5) q_lo = 0;
            for (int c = 0; c < 3; ++c) acc_lo[c] += q_lo * light.color[c], acc_hi[c] += q_hi * light.color[c];
        }
        for (int c = 0; c < 3; ++c) {
            const auto out = [&](unsigned a, std::uint64_t acc) {
                return float(std::min<std::uint64_t>(255, (a * (kAmbient + acc)) >> kLightShift)) / 255.0f;
            };
            f.colors[0][c] = out(alo[c], acc_lo[c]);
            f.colors[1][c] = out(ahi[c], acc_hi[c]);
        }
        f.colors[0][3] = f.colors[1][3] = 1.0f;
        f.count = 2;
    };
    const Color background{float(kBackground & 255u) / 255.0f, float((kBackground >> 8) & 255u) / 255.0f,
                           float((kBackground >> 16) & 255u) / 255.0f, 1.0f};
    return resolve_bounds(detail::fragments(scene, shade, degenerate), background);
}

// Which lights each tile must keep (1), must not keep (0) or may keep either way (2), from the view depths the GPU
// wrote (read back) and the pass's constants: the tile's box of rebuilt world positions p = eye + t (d0 + cx R + cy U)
// over its pixels with t > 0, then the squared distance from each light to the box against the box test's r^2 (k[14]),
// in double. A light within 2e-3 of the threshold is "either": the GPU forms the same numbers in float. An empty tile
// keeps nothing. Index: tile * kLights + light.
inline std::vector<std::uint8_t> tile_light_states(const std::vector<float>& depth, const std::array<float, 16>& k,
                                                   const std::vector<PointLight>& lights) {
    std::vector<std::uint8_t> out(std::size_t(kTilesX) * kTilesY * kLights, 0);
    for (unsigned ty = 0; ty < kTilesY; ++ty)
        for (unsigned tx = 0; tx < kTilesX; ++tx) {
            double lo[3] = {1e300, 1e300, 1e300}, hi[3] = {-1e300, -1e300, -1e300};
            bool any = false;
            for (unsigned y = ty * 16; y < ty * 16 + 16; ++y)
                for (unsigned x = tx * 16; x < tx * 16 + 16; ++x) {
                    const double t = depth[std::size_t(y) * kWidth + x];
                    if (!(t > 0)) continue;
                    any = true;
                    const double cx = double(x) - 191.5, cy = double(y) - 127.5;
                    for (int c = 0; c < 3; ++c) {
                        const double p = k[c] + t * (double(k[3 + c]) + cx * k[6 + c] + cy * k[9 + c]);
                        lo[c] = std::min(lo[c], p), hi[c] = std::max(hi[c], p);
                    }
                }
            if (!any) continue;
            for (unsigned l = 0; l < kLights; ++l) {
                double d2 = 0;
                for (int c = 0; c < 3; ++c) {
                    const double d = std::max(0.0, std::max(lo[c] - lights[l].pos[c], lights[l].pos[c] - hi[c]));
                    d2 += d * d;
                }
                const double r2 = k[14];
                out[(std::size_t(ty) * kTilesX + tx) * kLights + l] = d2 < r2 - 2e-3 ? 1 : d2 > r2 + 2e-3 ? 0 : 2;
            }
        }
    return out;
}

// The G-buffer's normal codes and view depth at every pixel: per fragment, the three 10-bit codes (lo in colors[0],
// hi in colors[1], from the 8-bit codes as the reference's lighting reads them) and the clip w at the pixel centre,
// widened by its change over kSnapSlack pixels (vertex snapping moves the plane that much) and 1e-5 relative (.w of
// colors[0] and colors[1]).
inline TexturedReference gbuffer_reference(const Scene& scene) {
    unsigned degenerate = 0;
    const auto& vp = scene.view_projection;
    const auto shade = [&](const PixelSite& site, detail::Fragment& f) {
        const auto packed = std::uint32_t(site.shade);
        for (int k = 0; k < 3; ++k) {
            unsigned lo, hi;
            unorm_codes(float((packed >> (8 * k)) & 255u) * (1.0f / 255.0f), 1023, lo, hi, kNormalWindow);
            f.colors[0][k] = float(lo), f.colors[1][k] = float(hi);
        }
        double w[3];
        site.world(site.px, site.py, w);
        const auto clip_w = [&](double px, double py) {
            double p[3];
            site.world(px, py, p);
            return vp[12] * p[0] + vp[13] * p[1] + vp[14] * p[2] + vp[15];
        };
        const double cw = clip_w(site.px, site.py);
        const double slack = (std::fabs(clip_w(site.px + 1, site.py) - cw) + std::fabs(clip_w(site.px, site.py + 1) - cw)) *
                                 kSnapSlack + 1e-5 * cw;
        (void)w;
        f.colors[0][3] = float(cw - slack), f.colors[1][3] = float(cw + slack);
        f.count = 2;
    };
    return resolve_bounds(detail::fragments(scene, shade, degenerate), Color{0, 0, 0, 0});
}

// The compute pass's packed RGBA8 words as colours (code / 255).
inline std::vector<Color> unpack_rgba8(const std::vector<std::uint32_t>& words) {
    std::vector<Color> out(words.size());
    for (std::size_t i = 0; i < words.size(); ++i)
        for (int k = 0; k < 4; ++k) out[i][k] = float((words[i] >> (8 * k)) & 255u) / 255.0f;
    return out;
}

}  // namespace lanterns
