// Lanterns stage 6: the CPU reference of the textured frame. The floor samples the BC1 mosaic with image_sample
// (trilinear, implicit LOD from the 2x2 quad's derivatives); the solids sample the BC7 squares with image_sample_l at
// the LOD their draw's tint.a gives. Sampling is not bit-reproducible across GPUs (bilinear weights and the LOD are
// fixed point, the BC1 thirds may round either way, and Vulkan lets the LOD's scale factor lie anywhere between the
// largest component and the sum of the components), so every fragment gets per-channel bounds instead of one colour:
//   - the solids: the bilinear sample at the pixel centre's (u, v), exactly, then +- kTextureTolerance;
//   - the floor: the trilinear sample over every LOD the rules allow for that pixel (coarse or fine derivatives, any
//     scale factor in the allowed range, +- 1/32), whose extremes lie at the range's ends or the integer LODs inside
//     it, then +- kTextureTolerance.
// kTextureTolerance (4/255) covers 8-bit sub-texel and LOD-fraction weights and one 8-bit step of BC1 rounding. The
// alpha written is 1 everywhere and is compared exactly. Pixels whose stage-1 coverage is uncertain take the union of
// their candidates' bounds.
#pragma once

#include "lanterns_bc.h"
#include "lanterns_reference.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace lanterns {

constexpr float kTextureTolerance = 4.0f / 255.0f;
constexpr double kFloorTexScale = 0.25;   // floor uv = world xz * 0.25 (the mosaic repeats every 4 units)
constexpr double kSolidTexScale = 0.5;    // solid uv = ((x + z) * 0.5, y * 0.5)

inline Color bilinear(const BcTexture& tex, unsigned mip, double u, double v) {
    const unsigned w = tex.mip_width(mip), h = tex.mip_height(mip);
    const double x = u * w - 0.5, y = v * h - 0.5;
    const double fx = std::floor(x), fy = std::floor(y);
    const double ax = x - fx, ay = y - fy;
    const auto wrap = [](long i, unsigned n) { return unsigned(((i % long(n)) + long(n)) % long(n)); };
    const unsigned x0 = wrap(long(fx), w), x1 = wrap(long(fx) + 1, w), y0 = wrap(long(fy), h), y1 = wrap(long(fy) + 1, h);
    const auto& t = tex.decoded[mip];
    Color c;
    for (int k = 0; k < 4; ++k) {
        const double top = t[y0 * w + x0][k] * (1 - ax) + t[y0 * w + x1][k] * ax;
        const double bottom = t[y1 * w + x0][k] * (1 - ax) + t[y1 * w + x1][k] * ax;
        c[k] = float(top * (1 - ay) + bottom * ay);
    }
    return c;
}

inline Color trilinear(const BcTexture& tex, double u, double v, double lod) {
    const double top = double(tex.decoded.size() - 1);
    lod = std::clamp(lod, 0.0, top);
    const unsigned m0 = unsigned(std::floor(lod));
    const double f = lod - m0;
    const Color a = bilinear(tex, m0, u, v);
    if (f == 0.0 || m0 + 1 > unsigned(top)) return a;
    const Color b = bilinear(tex, m0 + 1, u, v);
    Color c;
    for (int k = 0; k < 4; ++k) c[k] = float(a[k] * (1 - f) + b[k] * f);
    return c;
}

struct Bounds {
    Color lo, hi;
};

inline void widen(Bounds& b, const Color& c) {
    for (int k = 0; k < 4; ++k) b.lo[k] = std::min(b.lo[k], c[k]), b.hi[k] = std::max(b.hi[k], c[k]);
}

inline Bounds empty_bounds() {
    return {{INFINITY, INFINITY, INFINITY, INFINITY}, {-INFINITY, -INFINITY, -INFINITY, -INFINITY}};
}

// The textured colour's bounds: texture bounds times the flat face colour, +- the tolerance, alpha 1.
inline Bounds lit_bounds(const Bounds& texel, const Color& lit) {
    Bounds b;
    for (int k = 0; k < 3; ++k) {
        b.lo[k] = texel.lo[k] * lit[k] - kTextureTolerance;
        b.hi[k] = texel.hi[k] * lit[k] + kTextureTolerance;
    }
    b.lo[3] = b.hi[3] = 1.0f;
    return b;
}

// The floor's LOD range at a pixel: log2 of the scale factor, over coarse and fine derivatives and the factor's
// allowed range (max component .. sum of components of each axis's derivative, then the larger axis).
inline void floor_lod_range(const PixelSite& site, const BcTexture& tex, double& lod_lo, double& lod_hi) {
    const auto uv = [&](double px, double py, double out[2]) {
        double w[3];
        site.world(px, py, w);
        out[0] = w[0] * kFloorTexScale, out[1] = w[2] * kFloorTexScale;
    };
    const double qx = std::floor(site.px / 2) * 2 + 0.5, qy = std::floor(site.py / 2) * 2 + 0.5;
    const double size[2] = {double(tex.width), double(tex.height)};
    double rho_lo = INFINITY, rho_hi = 0;
    // Coarse: the quad's top-left differences. Fine: this pixel's row and column within the quad.
    const double rows[2][2] = {{qy, qx}, {site.py, site.px}};
    for (const auto& r : rows) {
        double a[2], bx[2], ay[2], by[2];
        uv(qx, r[0], a), uv(qx + 1, r[0], bx), uv(r[1], qy, ay), uv(r[1], qy + 1, by);
        double lo_axis[2], hi_axis[2];
        for (int axis = 0; axis < 2; ++axis) {
            const double du = std::fabs((axis ? by[0] - ay[0] : bx[0] - a[0]) * size[0]);
            const double dv = std::fabs((axis ? by[1] - ay[1] : bx[1] - a[1]) * size[1]);
            lo_axis[axis] = std::max(du, dv);
            hi_axis[axis] = du + dv;
        }
        rho_lo = std::min(rho_lo, std::max(lo_axis[0], lo_axis[1]));
        rho_hi = std::max(rho_hi, std::max(hi_axis[0], hi_axis[1]));
    }
    lod_lo = std::log2(std::max(rho_lo, 1e-9)) - 1.0 / 32;
    lod_hi = std::log2(std::max(rho_hi, 1e-9)) + 1.0 / 32;
}

struct TexturedScene {
    const BcTexture* floor;
    const BcTexture* solid;
    float lod_offset = 0.0f;   // added to every solid draw's LOD (the control)
};

// What a textured fragment may show: its bounds go in colors[0] (lo) and colors[1] (hi).
inline void textured_shade(const TexturedScene& ts, const PixelSite& site, detail::Fragment& f) {
    Bounds texel = empty_bounds();
    double w[3];
    site.world(site.px, site.py, w);
    if (site.draw->floor) {
        const double u = w[0] * kFloorTexScale, v = w[2] * kFloorTexScale;
        double lo, hi;
        floor_lod_range(site, *ts.floor, lo, hi);
        widen(texel, trilinear(*ts.floor, u, v, lo));
        widen(texel, trilinear(*ts.floor, u, v, hi));
        for (double m = std::ceil(std::max(lo, 0.0)); m < hi; m += 1.0) widen(texel, trilinear(*ts.floor, u, v, m));
    } else {
        const double u = (w[0] + w[2]) * kSolidTexScale, v = w[1] * kSolidTexScale;
        const double lod = std::clamp(double(site.draw->tint[3] + ts.lod_offset), 0.0, double(ts.solid->decoded.size() - 1));
        widen(texel, bilinear(*ts.solid, unsigned(lod), u, v));
    }
    const Bounds b = lit_bounds(texel, site.lit);
    f.colors[0] = b.lo, f.colors[1] = b.hi, f.count = 2;
}

struct TexturedReference {
    std::vector<Bounds> bounds;                 // the nearest certain surface's (or the clear colour's) bounds
    std::vector<std::uint8_t> certain;          // 1: bounds is the only box
    std::vector<std::vector<Bounds>> others;    // else the box of each other surface that may show instead
};

inline TexturedReference textured_reference(const Scene& scene, const TexturedScene& ts) {
    unsigned degenerate = 0;
    const auto frags = detail::fragments(
        scene, [&](const PixelSite& site, detail::Fragment& f) { textured_shade(ts, site, f); }, degenerate);
    TexturedReference ref;
    ref.bounds.resize(frags.size());
    ref.certain.resize(frags.size());
    ref.others.resize(frags.size());
    const Color clear{kClear[0], kClear[1], kClear[2], kClear[3]};
    for (std::size_t i = 0; i < frags.size(); ++i) {
        // The nearest cover (detail::covers): a certain fragment, or two faces of one solid sharing the edge the
        // centre is near. The clear colour only where nothing covers.
        const auto found = detail::covers(frags[i]);
        const detail::Cover* winner = nullptr;
        for (const auto& c : found)
            if (!winner || c.depth < winner->depth) winner = &c;
        const auto box = [](const detail::Fragment& f) {
            Bounds own = empty_bounds();
            widen(own, f.colors[0]), widen(own, f.colors[1]);
            return own;
        };
        Bounds b = empty_bounds();
        if (winner) b = box(*winner->a);
        else widen(b, clear);
        bool certain = !winner || !winner->b;
        if (winner && winner->b) ref.others[i].push_back(box(*winner->b));
        for (const auto& f : frags[i]) {
            if (winner && (&f == winner->a || &f == winner->b)) continue;
            if (winner && f.depth - f.tol > winner->depth + winner->tol) continue;
            // Each surface keeps its own box: a pixel must be within one of them, not within their union, which
            // would also take a mix of two surfaces' channels.
            ref.others[i].push_back(box(f));
            certain = false;
        }
        ref.bounds[i] = b;
        ref.certain[i] = certain;
    }
    return ref;
}

inline bool within(const Color& c, const Bounds& b) {
    for (int k = 0; k < 4; ++k)
        if (!(c[k] >= b.lo[k] && c[k] <= b.hi[k])) return false;
    return true;
}

// Pixel i is within its bounds or within the box of one of the other surfaces that may show there.
inline bool within_any(const TexturedReference& ref, std::size_t i, const Color& c) {
    return within(c, ref.bounds[i]) ||
           std::any_of(ref.others[i].begin(), ref.others[i].end(), [&](const Bounds& b) { return within(c, b); });
}

inline Comparison compare_textured(const std::vector<Color>& gpu, const TexturedReference& ref) {
    Comparison c;
    bool reported = false;
    for (std::size_t i = 0; i < gpu.size(); ++i) {
        const bool ok = within_any(ref, i, gpu[i]);
        if (ref.certain[i]) ++c.certain, c.certain_exact += ok;
        else ++c.uncertain, c.uncertain_plausible += ok;
        if (!ok && !reported) {
            reported = true;
            c.first_bad_x = unsigned(i % kWidth), c.first_bad_y = unsigned(i / kWidth);
            c.first_bad_gpu = gpu[i], c.first_bad_ref = ref.bounds[i].lo;
        }
    }
    return c;
}

// Certain pixels where the two references' bounds do not overlap: there the control frame must leave the main bounds.
inline std::vector<std::uint8_t> disjoint(const TexturedReference& a, const TexturedReference& b) {
    std::vector<std::uint8_t> out(a.bounds.size());
    for (std::size_t i = 0; i < out.size(); ++i) {
        if (!a.certain[i] || !b.certain[i]) continue;
        for (int k = 0; k < 3; ++k)
            out[i] |= a.bounds[i].hi[k] < b.bounds[i].lo[k] || b.bounds[i].hi[k] < a.bounds[i].lo[k];
    }
    return out;
}

}  // namespace lanterns
