// Lanterns: the CPU reference for a stage-1 frame. It rasterizes the scene of lanterns_scene.h in double precision,
// face by face (each face is a convex polygon; the GPU draws it as a triangle fan, which covers the same pixels), and
// says for every pixel either "this exact colour" or "uncertain, one of these colours". A pixel is uncertain when a
// tiny change in where the GPU puts a vertex could change it: its centre lies within kMargin pixels of a face edge,
// two faces with different colours are within their depth tolerance of each other, or (on the floor) the centre is
// within kMargin pixels of a checker border. kMargin is 1/32 pixel, eight times the vertex snapping of an 8-bit
// sub-pixel rasterizer. Background is a candidate only where no surface certainly covers the centre: a centre near
// an edge that two faces of one solid share (and no other edge of either) is covered by one of the two, since both
// get the same vertex positions and the rasterizer leaves no gap between them; a crack there fails. Every other pixel must match bit for bit: its colour is a product of short binary fractions
// (lanterns_scene.h), which float arithmetic computes exactly, and nothing near it is in doubt. The uncertain pixels
// are counted apart, as d3d12_translated_draw_test counts its diagonal, and each must still be one of its candidates.
#pragma once

#include "lanterns_scene.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <vector>

namespace lanterns {

using Color = std::array<float, 4>;
constexpr double kMargin = 1.0 / 32.0;

struct Reference {
    std::vector<Color> color;                    // the expected colour (the most likely one where uncertain)
    std::vector<std::vector<Color>> candidates;  // empty: certain; else every colour the pixel may have
    unsigned degenerate_faces = 0;               // faces seen edge-on, which cover nothing for certain
    unsigned shared_edge_pixels = 0;             // pixels a pair of faces covers (detail::covers)
};

namespace detail {

struct ScreenVertex {
    double x, y, z, q;     // screen position (pixels, y down), depth z/w, q = 1/w
    double wx, wy, wz;     // world position (the floor's checker and stage 6's textures read it)
};

struct Fragment {
    unsigned seq;          // submission order of the face
    double depth, tol;
    bool certain_cover;    // the centre is more than kMargin inside every edge
    unsigned count;        // 1 or 2 colours (2: a floor pixel on a checker border)
    Color colors[2];
    unsigned group = 0;    // the draw instance (one solid) the face belongs to
    bool on_edge = false;  // the centre is within kMargin of exactly one edge of the face, this one:
    std::array<double, 4> edge{};   // its screen end points, ordered, so the face across it has the same four
    int side = 0;          // which side of that edge the face lies on
};

// Whether two fragments of one solid share the one edge each is near, from opposite sides: then the centre is inside
// one of the two faces (both get the same vertex positions, and a rasterizer leaves no gap along a shared edge).
inline bool share_edge(const Fragment& a, const Fragment& b) {
    return a.on_edge && b.on_edge && a.group == b.group && a.edge == b.edge && a.side == -b.side;
}

// What certainly covers a pixel centre: a fragment more than kMargin inside its face, or two faces of one solid that
// share the one edge the centre is near (share_edge), one of which covers it; the pair's depth spans both.
struct Cover {
    const Fragment* a;
    const Fragment* b;   // the pair's other face, or null
    double depth, tol;
    unsigned seq;
};

inline std::vector<Cover> covers(const std::vector<Fragment>& list) {
    std::vector<Cover> out;
    for (const Fragment& f : list)
        if (f.certain_cover) out.push_back({&f, nullptr, f.depth, f.tol, f.seq});
    for (const Fragment& f : list)
        for (const Fragment& g : list)
            if (&f < &g && share_edge(f, g)) {
                const double lo = std::min(f.depth - f.tol, g.depth - g.tol);
                const double hi = std::max(f.depth + f.tol, g.depth + g.tol);
                out.push_back({&f, &g, (lo + hi) / 2, (hi - lo) / 2, std::max(f.seq, g.seq)});
            }
    return out;
}

inline bool same(const Color& a, const Color& b) { return std::memcmp(a.data(), b.data(), sizeof(Color)) == 0; }

inline void add_unique(std::vector<Color>& out, const Color& c) {
    for (const Color& o : out)
        if (same(o, c)) return;
    out.push_back(c);
}

// The vertex shader's arithmetic, in double: local = M * (p, 1), world = local * scale + offset, clip = VP * world.
inline ScreenVertex transform(const Scene& scene, const Draw& draw, const Instance& inst, const Vertex& v) {
    const auto& m = draw.model;
    double local[3], world[3], clip[4];
    for (int r = 0; r < 3; ++r)
        local[r] = double(m[r * 4]) * v.x + double(m[r * 4 + 1]) * v.y + double(m[r * 4 + 2]) * v.z + m[r * 4 + 3];
    for (int r = 0; r < 3; ++r) world[r] = local[r] * inst.scale + inst.offset[r];
    const auto& vp = scene.view_projection;
    for (int r = 0; r < 4; ++r)
        clip[r] = double(vp[r * 4]) * world[0] + double(vp[r * 4 + 1]) * world[1] + double(vp[r * 4 + 2]) * world[2] +
                  vp[r * 4 + 3];
    const double q = 1.0 / clip[3];
    return {(clip[0] * q + 1.0) * 0.5 * kWidth, (1.0 - clip[1] * q) * 0.5 * kHeight, clip[2] * q, q, world[0],
            world[1], world[2]};
}

// The colour vs_lanterns.asm exports for a face, as the float products it forms: (tint * colour) * shade.
inline Color face_color(const Draw& draw, const Instance& inst, float shade) {
    Color c;
    for (int k = 0; k < 3; ++k) c[k] = (draw.tint[k] * inst.color[k]) * shade;
    c[3] = draw.tint[3] * inst.color[3];
    return c;
}

inline Color times(const std::array<float, 4>& a, const Color& b) {
    return {a[0] * b[0], a[1] * b[1], a[2] * b[2], a[3] * b[3]};
}

inline double cross(double ax, double ay, double bx, double by) { return ax * by - ay * bx; }

}  // namespace detail

// Where a face covers a pixel centre: what a shading function gets to colour the fragment. world(px, py) is the
// perspective-correct world position at any screen point of the face's plane (what the PS's v_interp_p1/p2 of the
// VS's param1 reads), so a function can also take differences across the pixel's 2x2 quad.
struct PixelSite {
    const Draw* draw;
    const Instance* inst;
    Color lit;            // the face colour vs_lanterns.asm exports (flat)
    double px, py;        // the pixel centre
    std::function<void(double, double, double[3])> world;
};

namespace detail {

// Every fragment of every face, in submission order, coloured by shade(site, fragment) (which sets colors and count).
template <class Shade>
std::vector<std::vector<Fragment>> fragments(const Scene& scene, Shade shade, unsigned& degenerate_faces) {
    std::vector<std::vector<Fragment>> frags(std::size_t(kWidth) * kHeight);
    unsigned seq = 0, groups = 0;
    for (const Draw& draw : scene.draws) {
        const Mesh& mesh = scene.meshes[draw.mesh];
        for (const Instance& inst : draw.instances) {
            const unsigned group = groups++;
            for (const Face& face : mesh.faces) {
                const unsigned self = seq++;
                ScreenVertex s[6];
                for (unsigned k = 0; k < face.count; ++k) s[k] = transform(scene, draw, inst, mesh.vertices[face.v[k]]);
                double area = 0;
                for (unsigned k = 0; k < face.count; ++k) {
                    const auto& a = s[k];
                    const auto& b = s[(k + 1) % face.count];
                    area += cross(a.x, a.y, b.x, b.y);
                }
                const double a012 = cross(s[1].x - s[0].x, s[1].y - s[0].y, s[2].x - s[0].x, s[2].y - s[0].y);
                if (std::fabs(area) < 1e-6 || std::fabs(a012) < 1e-9) {
                    ++degenerate_faces;
                    continue;
                }
                const double orient = area > 0 ? 1.0 : -1.0;
                double cx = 0, cy = 0;   // the face's centre, to tell which side of an edge it lies on
                for (unsigned k = 0; k < face.count; ++k) cx += s[k].x / face.count, cy += s[k].y / face.count;
                // Barycentrics of the plane through s0, s1, s2 (the whole face lies in it).
                const auto bary = [&](double px, double py, double l[3]) {
                    l[1] = cross(px - s[0].x, py - s[0].y, s[2].x - s[0].x, s[2].y - s[0].y) / a012;
                    l[2] = cross(s[1].x - s[0].x, s[1].y - s[0].y, px - s[0].x, py - s[0].y) / a012;
                    l[0] = 1.0 - l[1] - l[2];
                };
                const auto depth_at = [&](double px, double py) {
                    double l[3];
                    bary(px, py, l);
                    return l[0] * s[0].z + l[1] * s[1].z + l[2] * s[2].z;
                };
                // Perspective-correct world position, as the PS's v_interp_p1/p2 reads it.
                const auto world_at = [&](double px, double py, double w[3]) {
                    double l[3];
                    bary(px, py, l);
                    const double qs = l[0] * s[0].q + l[1] * s[1].q + l[2] * s[2].q;
                    w[0] = (l[0] * s[0].wx * s[0].q + l[1] * s[1].wx * s[1].q + l[2] * s[2].wx * s[2].q) / qs;
                    w[1] = (l[0] * s[0].wy * s[0].q + l[1] * s[1].wy * s[1].q + l[2] * s[2].wy * s[2].q) / qs;
                    w[2] = (l[0] * s[0].wz * s[0].q + l[1] * s[1].wz * s[1].q + l[2] * s[2].wz * s[2].q) / qs;
                };
                const double dzdx = depth_at(1, 0) - depth_at(0, 0), dzdy = depth_at(0, 1) - depth_at(0, 0);
                const double tol = (std::fabs(dzdx) + std::fabs(dzdy)) * kMargin + 1e-6;
                const Color lit = face_color(draw, inst, mesh.vertices[face.v[0]].shade);
                double minx = 1e30, maxx = -1e30, miny = 1e30, maxy = -1e30;
                for (unsigned k = 0; k < face.count; ++k) {
                    minx = std::min(minx, s[k].x), maxx = std::max(maxx, s[k].x);
                    miny = std::min(miny, s[k].y), maxy = std::max(maxy, s[k].y);
                }
                const int x0 = std::max(0, int(std::floor(minx)) - 1), x1 = std::min(int(kWidth) - 1, int(maxx) + 1);
                const int y0 = std::max(0, int(std::floor(miny)) - 1), y1 = std::min(int(kHeight) - 1, int(maxy) + 1);
                for (int y = y0; y <= y1; ++y)
                    for (int x = x0; x <= x1; ++x) {
                        const double px = x + 0.5, py = y + 0.5;
                        double inside = 1e30;
                        unsigned near_count = 0, near_edge = 0;
                        for (unsigned k = 0; k < face.count; ++k) {
                            const auto& a = s[k];
                            const auto& b = s[(k + 1) % face.count];
                            const double ex = b.x - a.x, ey = b.y - a.y;
                            const double d = orient * cross(ex, ey, px - a.x, py - a.y) / std::sqrt(ex * ex + ey * ey);
                            inside = std::min(inside, d);
                            if (d <= kMargin) ++near_count, near_edge = k;
                        }
                        if (inside < -kMargin) continue;
                        Fragment f{self, depth_at(px, py), tol, inside > kMargin, 1, {lit, lit}};
                        f.group = group;
                        if (near_count == 1) {
                            const auto& a = s[near_edge];
                            const auto& b = s[(near_edge + 1) % face.count];
                            const bool swap = b.x < a.x || (b.x == a.x && b.y < a.y);
                            const auto& p = swap ? b : a;
                            const auto& q = swap ? a : b;
                            f.on_edge = true;
                            f.edge = {p.x, p.y, q.x, q.y};
                            f.side = cross(q.x - p.x, q.y - p.y, cx - p.x, cy - p.y) > 0 ? 1 : -1;
                        }
                        shade(PixelSite{&draw, &inst, lit, px, py, world_at}, f);
                        frags[std::size_t(y) * kWidth + x].push_back(f);
                    }
            }
        }
    }

    return frags;
}

// Stage 1's shading: the face colour, and on the floor the checker (both colours near a checker border).
inline void checker_shade(const PixelSite& site, Fragment& f) {
    if (!site.draw->floor) return;
    double w[3], w1[3], w2[3];
    site.world(site.px, site.py, w);
    site.world(site.px + 1, site.py, w1);
    site.world(site.px, site.py + 1, w2);
    const double gx = std::fabs(w1[0] - w[0]) + std::fabs(w2[0] - w[0]);
    const double gz = std::fabs(w1[2] - w[2]) + std::fabs(w2[2] - w[2]);
    const auto near_border = [](double v, double g) { return std::fabs(v - std::round(v)) <= g * kMargin + 1e-5; };
    const long parity = (long(std::floor(w[0])) + long(std::floor(w[2]))) & 1;
    f.colors[0] = times(parity ? kFloorB : kFloorA, site.lit);
    f.colors[1] = times(parity ? kFloorA : kFloorB, site.lit);
    f.count = near_border(w[0], gx) || near_border(w[2], gz) ? 2 : 1;
}

}  // namespace detail

// The frame the stage-1 draws make, with the depth test on (LESS, cleared to 1) or off (submission order wins).
inline Reference reference_frame(const Scene& scene, bool depth_test) {
    using namespace detail;
    Reference ref;
    const auto frags = fragments(scene, checker_shade, ref.degenerate_faces);
    ref.color.assign(frags.size(), Color{kClear[0], kClear[1], kClear[2], kClear[3]});
    ref.candidates.resize(frags.size());
    for (std::size_t i = 0; i < frags.size(); ++i) {
        const auto& list = frags[i];
        const std::vector<Cover> found = covers(list);
        // Without the depth test the order inside a pair decides which face shows, so a pair is no cover there; it
        // still rules out the background.
        const Cover* winner = nullptr;
        bool paired = false;
        for (const Cover& c : found) {
            paired = paired || c.b;
            if (c.b && !depth_test) continue;
            if (!winner || (depth_test ? c.depth < winner->depth : c.seq > winner->seq)) winner = &c;
        }
        std::vector<Color> cand;
        if (winner) {
            ref.color[i] = winner->a->colors[0];
            for (const Fragment* m : {winner->a, winner->b})
                if (m)
                    for (unsigned k = 0; k < m->count; ++k) add_unique(cand, m->colors[k]);
            ref.shared_edge_pixels += winner->b != nullptr;
        } else if (!paired) {
            add_unique(cand, ref.color[i]);
        }
        for (const Fragment& f : list) {
            if (winner && (&f == winner->a || &f == winner->b)) continue;
            // Could f be what the GPU shows instead? With the depth test, if it may be nearer; without, if it comes
            // later. An uncertain fragment counts wherever it might be covered.
            const bool may_win = !winner ? true
                                 : depth_test ? f.depth - f.tol <= winner->depth + winner->tol
                                              : f.seq > winner->seq;
            if (!may_win) continue;
            for (unsigned k = 0; k < f.count; ++k) add_unique(cand, f.colors[k]);
        }
        if (cand.size() > 1) ref.candidates[i] = std::move(cand);
    }
    return ref;
}

struct Comparison {
    unsigned certain = 0, certain_exact = 0, uncertain = 0, uncertain_plausible = 0;
    unsigned first_bad_x = 0, first_bad_y = 0;
    Color first_bad_gpu{}, first_bad_ref{};
    bool pass() const { return certain_exact == certain && uncertain_plausible == uncertain; }
};

inline Comparison compare(const std::vector<Color>& gpu, const Reference& ref) {
    using detail::same;
    Comparison c;
    bool reported = false;
    for (std::size_t i = 0; i < gpu.size(); ++i) {
        bool ok;
        if (ref.candidates[i].empty()) {
            ++c.certain;
            ok = same(gpu[i], ref.color[i]);
            c.certain_exact += ok;
        } else {
            ++c.uncertain;
            ok = std::any_of(ref.candidates[i].begin(), ref.candidates[i].end(),
                             [&](const Color& k) { return same(gpu[i], k); });
            c.uncertain_plausible += ok;
        }
        if (!ok && !reported) {
            reported = true;
            c.first_bad_x = unsigned(i % kWidth), c.first_bad_y = unsigned(i / kWidth);
            c.first_bad_gpu = gpu[i], c.first_bad_ref = ref.color[i];
        }
    }
    return c;
}

}  // namespace lanterns
