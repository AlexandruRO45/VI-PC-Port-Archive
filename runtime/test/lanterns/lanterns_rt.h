// Lanterns stage 11: ray-traced shadows. The scene is ten instances of two solids (seven hexagonal prisms in two
// draws, three square columns in a third) on the floor. Each solid's triangles go into its own BLAS and the ten
// instances into a TLAS, all through BuildRaytracingAccelerationStructure; each instance record names its BLAS, which
// the shader reads per lane. The frame's G-buffer comes from stage 9's shaders; then
// cs_lanterns_rt_shadow traces one shadow ray per pixel through the TLAS and the BLAS with image_bvh_intersect_ray
// (its comment says how) and darkens the occluded pixels: min(255, (albedo8 * (occluded ? 96 : 256)) >> 8).
//
// The CPU traces the same rays against the same triangles (the instance transforms applied in double). A ray is
// certainly blocked when it crosses a triangle more than kRtBaryMargin inside its edges at t > 1e-3, certainly free
// when every triangle is missed by that margin, and undecided otherwise; the world position is known within
// kPosSlack and the decoded normal moves the origin by less than 1e-3.
#pragma once

#include "lanterns_deferred.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

namespace lanterns {

inline float float_bits(std::uint32_t b) {
    float f;
    std::memcpy(&f, &b, 4);
    return f;
}

// To the light (normalised (0.7, 1, 0.4)), as cs_lanterns_rt_shadow.asm's literals.
inline std::array<float, 3> rt_light() { return {float_bits(0x3f0b81cbu), float_bits(0x3f474bb5u), float_bits(0x3e9f6fc4u)}; }
constexpr float kRtBias = 0.02f, kRtMax = 64.0f;
constexpr std::uint32_t kRtLit = 256, kRtShadowed = 96;
constexpr double kRtBaryMargin = 0.01;
// Stage 23's alpha test: a hit counts only where its barycentrics give u + 2v < 1 (u weighs the triangle's second
// vertex, v its third, as D3D12 defines them). Within kRtAlphaMargin of that line the CPU leaves the hit undecided.
constexpr double kRtAlphaMargin = 0.01;
constexpr std::uint32_t kRtStepCap = 2048;              // the traversal loop's bound (cs_lanterns_rt_shadow: s17)
constexpr std::uint32_t kRtUnfinished = 0xffff00ffu;    // magenta: a ray still tracing when the cap ends the loop

inline Scene stage11() {
    Scene s;
    s.meshes.push_back(prism("lantern", 6, 0.45, 0.0, 1.5));   // 0: BLAS 0
    s.meshes.push_back(prism("column", 4, 0.3, 0.0, 2.25));    // 1: BLAS 1, taller and thinner
    s.meshes.push_back(floor_quad(-6.0, 6.0, -3.0, 9.0));      // 2: drawn, not traced
    Draw a{0, false, {}, rotation_y(10), {1.0f, 1.0f, 1.0f, 1.0f}};
    const float ax[4] = {-3.0f, -0.75f, 1.5f, 3.5f}, az[4] = {0.5f, 1.5f, 0.25f, 2.5f};
    for (unsigned i = 0; i < 4; ++i) a.instances.push_back(instance(ax[i], 0.0f, az[i], 1.0f, {7, 5, 1}, i));
    Draw b{0, false, {}, rotation_y(35), {0.875f, 1.0f, 1.0f, 1.0f}};
    const float bx[3] = {-2.0f, 0.5f, 2.75f}, bz[3] = {3.5f, 4.0f, 5.0f}, bs[3] = {0.75f, 1.25f, 0.75f};
    for (unsigned i = 0; i < 3; ++i) b.instances.push_back(instance(bx[i], 0.0f, bz[i], bs[i], {2, 6, 6}, i));
    Draw c{1, false, {}, rotation_y(0), {1.0f, 1.0f, 0.875f, 1.0f}};
    const float cx[3] = {-4.25f, -1.0f, 4.5f}, cz[3] = {5.5f, 6.5f, 6.5f}, cs[3] = {1.5f, 1.0f, 1.25f};
    for (unsigned i = 0; i < 3; ++i) c.instances.push_back(instance(cx[i], 0.0f, cz[i], cs[i], {5, 3, 7}, i));
    Draw floor{2, true, {}, rotation_y(0), {1.0f, 0.875f, 0.75f, 1.0f}};
    floor.instances.push_back({{0.0f, 0.0f, 0.0f}, 1.0f, {1.0f, 1.0f, 1.0f, 1.0f}});
    s.draws = {a, b, c, floor};
    const double eye[3] = {0.5, 3.25, -6.5}, target[3] = {0.0, 0.6, 2.5};
    s.view_projection = camera(eye, target, 52.0, double(kWidth) / double(kHeight), 0.5, 40.0);
    return with_packed_normals(s);
}

constexpr unsigned kRtMeshes = 2;   // scene meshes 0 and 1 are traced, one BLAS each
// The control that swaps BLAS references: these two instances (a prism and a column) trade BLASes in a third TLAS.
constexpr unsigned kRtSwapA = 1, kRtSwapB = 8;

// Object to world for every traced instance, as vs_lanterns places it: world = (M p) * scale + offset.
inline std::vector<std::array<float, 12>> rt_transforms(const Scene& scene) {
    std::vector<std::array<float, 12>> out;
    for (const Draw& d : scene.draws) {
        if (d.floor) continue;
        for (const Instance& inst : d.instances) {
            std::array<float, 12> t{};
            for (int r = 0; r < 3; ++r) {
                for (int c = 0; c < 3; ++c) t[r * 4 + c] = d.model[r * 4 + c] * inst.scale;
                t[r * 4 + 3] = d.model[r * 4 + 3] * inst.scale + inst.offset[r];
            }
            out.push_back(t);
        }
    }
    return out;
}

// The mesh (= BLAS) each traced instance uses, in rt_transforms order; swapped: kRtSwapA and kRtSwapB trade theirs.
inline std::vector<unsigned> rt_instance_meshes(const Scene& scene, bool swapped = false) {
    std::vector<unsigned> out;
    for (const Draw& d : scene.draws)
        if (!d.floor) out.insert(out.end(), d.instances.size(), d.mesh);
    if (swapped) std::swap(out[kRtSwapA], out[kRtSwapB]);
    return out;
}

// The BLAS mesh's triangles (each face fanned), as indices into its vertices.
inline std::vector<std::uint16_t> rt_indices(const Mesh& mesh) {
    std::vector<std::uint16_t> out;
    for (const Face& f : mesh.faces)
        for (unsigned k = 1; k + 1 < f.count; ++k) out.insert(out.end(), {f.v[0], f.v[k], f.v[k + 1]});
    return out;
}

using Triangle = std::array<double, 9>;

// World triangles of the traced instances; skip leaves one instance out (the control's masked instance), swapped
// traces kRtSwapA and kRtSwapB with each other's mesh (the control's swapped BLAS references).
inline std::vector<Triangle> rt_triangles(const Scene& scene, int skip = -1, bool swapped = false) {
    const auto xf = rt_transforms(scene);
    const auto meshes = rt_instance_meshes(scene, swapped);
    std::vector<Triangle> out;
    for (std::size_t i = 0; i < xf.size(); ++i) {
        if (int(i) == skip) continue;
        const Mesh& mesh = scene.meshes[meshes[i]];
        const auto idx = rt_indices(mesh);
        for (std::size_t k = 0; k < idx.size(); k += 3) {
            Triangle t{};
            for (int v = 0; v < 3; ++v) {
                const Vertex& p = mesh.vertices[idx[k + v]];
                for (int r = 0; r < 3; ++r)
                    t[v * 3 + r] = double(xf[i][r * 4]) * p.x + double(xf[i][r * 4 + 1]) * p.y +
                                   double(xf[i][r * 4 + 2]) * p.z + double(xf[i][r * 4 + 3]);
            }
            out.push_back(t);
        }
    }
    return out;
}

// 0 free, 1 blocked, 2 undecided: a ray from o towards the light against every triangle (Moller-Trumbore in double:
// u and v are D3D12's barycentrics of the triangle's vertices in index order). alpha: stage 23's cut-outs apply.
inline int rt_occlusion(const std::vector<Triangle>& tris, const double o[3], const double d[3], bool alpha = false) {
    bool maybe = false;
    for (const Triangle& t : tris) {
        const double e1[3] = {t[3] - t[0], t[4] - t[1], t[5] - t[2]}, e2[3] = {t[6] - t[0], t[7] - t[1], t[8] - t[2]};
        const double p[3] = {d[1] * e2[2] - d[2] * e2[1], d[2] * e2[0] - d[0] * e2[2], d[0] * e2[1] - d[1] * e2[0]};
        const double det = e1[0] * p[0] + e1[1] * p[1] + e1[2] * p[2];
        if (std::fabs(det) < 1e-12) continue;   // parallel: no crossing (its edges are other triangles' too)
        const double s[3] = {o[0] - t[0], o[1] - t[1], o[2] - t[2]};
        const double u = (s[0] * p[0] + s[1] * p[1] + s[2] * p[2]) / det;
        const double q[3] = {s[1] * e1[2] - s[2] * e1[1], s[2] * e1[0] - s[0] * e1[2], s[0] * e1[1] - s[1] * e1[0]};
        const double v = (d[0] * q[0] + d[1] * q[1] + d[2] * q[2]) / det;
        const double tt = (e2[0] * q[0] + e2[1] * q[1] + e2[2] * q[2]) / det;
        const double m = kRtBaryMargin;
        if (u < -m || v < -m || u + v > 1 + m || tt < -1e-3 || tt > kRtMax + 1e-3) continue;
        const bool inside = u > m && v > m && u + v < 1 - m && tt > 1e-3 && tt < kRtMax - 1e-3;
        if (alpha) {
            const double a = u + 2 * v;
            if (a > 1 + kRtAlphaMargin) continue;   // cut out wherever on this triangle the ray crosses
            if (inside && a < 1 - kRtAlphaMargin) return 1;
            maybe = true;
            continue;
        }
        if (inside) return 1;
        maybe = true;
    }
    return maybe ? 2 : 0;
}

inline TexturedReference rt_reference(const Scene& scene, const std::vector<Triangle>& tris, bool alpha = false) {
    const auto light = rt_light();
    const double d[3] = {light[0], light[1], light[2]};
    unsigned degenerate = 0;
    const auto shade = [&](const PixelSite& site, detail::Fragment& f) {
        unsigned alo[4], ahi[4];
        detail::albedo_codes(site, alo, ahi);
        const auto packed = std::uint32_t(site.shade);
        double w[3], o[3];
        site.world(site.px, site.py, w);
        for (int k = 0; k < 3; ++k) {   // the decoded normal, as the shader reads it back
            const float f8 = float((packed >> (8 * k)) & 255u) * (1.0f / 255.0f);
            const double n = 2.0 * double(std::lround(double(f8) * 1023.0)) / 1023.0 - 1.0;
            o[k] = w[k] + double(kRtBias) * n;
        }
        // Undecided if any origin within the slack decides differently: the centre and the box's corners. The slack
        // is the rebuilt position's (position_slack) plus 1e-3 for the decoded normal's offset.
        int state = rt_occlusion(tris, o, d, alpha);
        const auto ps = position_slack(scene, site);
        const double slack[3] = {ps[0] + 1e-3, ps[1] + 1e-3, ps[2] + 1e-3};
        for (int c = 0; c < 8 && state != 2; ++c) {
            const double oc[3] = {o[0] + (c & 1 ? slack[0] : -slack[0]), o[1] + (c & 2 ? slack[1] : -slack[1]),
                                  o[2] + (c & 4 ? slack[2] : -slack[2])};
            if (rt_occlusion(tris, oc, d, alpha) != state) state = 2;
        }
        const std::uint32_t f_lo = state == 0 ? kRtLit : kRtShadowed, f_hi = state == 1 ? kRtShadowed : kRtLit;
        for (int k = 0; k < 3; ++k) {
            f.colors[0][k] = float(std::min<std::uint32_t>(255, (alo[k] * f_lo) >> 8)) / 255.0f;
            f.colors[1][k] = float(std::min<std::uint32_t>(255, (ahi[k] * f_hi) >> 8)) / 255.0f;
        }
        f.colors[0][3] = f.colors[1][3] = 1.0f;
        f.count = 2;
    };
    const Color background{float(kBackground & 255u) / 255.0f, float((kBackground >> 8) & 255u) / 255.0f,
                           float((kBackground >> 16) & 255u) / 255.0f, 1.0f};
    return resolve_bounds(detail::fragments(scene, shade, degenerate), background);
}

}  // namespace lanterns
