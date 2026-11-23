// Lanterns, stage 25 ("scale"): a title-shaped frame, as data both the guest's draws and its CPU reference read. A field
// of a few thousand draws (abstract solids, some instanced, some not) over a checkered floor at 3840x2160, the rising
// points of stage 3 in front, a second camera for a 4x MSAA pass at 1920x1080 and a third, high up, for an RGBA8 pass.
// Each draw names one of four root signatures and one of twelve pipeline variants (cull none, back or front, times four
// flavours that must not change a pixel: plain, blending ONE/ZERO, FrontCounterClockwise with the cull mode swapped,
// LESS_EQUAL with a depth bias of 1). Colours are short binary fractions as in stage 1, so certain pixels are exact.
// The reference (check_sparse) rasterizes the main pass in double, as lanterns_reference.h does, but only at a lattice
// of sample pixels and one band of rows at a time, so a 4K frame with tens of thousands of instances fits in memory.
#pragma once

#include "lanterns_particles.h"
#include "lanterns_reference.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <vector>

namespace lanterns::scale {

constexpr unsigned kW = 3840, kH = 2160;          // main pass: R16G16B16A16_FLOAT + D32_FLOAT
constexpr unsigned kAuxW = 1920, kAuxH = 1080;    // aux pass: 4x MSAA R16G16B16A16_FLOAT, resolved
constexpr unsigned kTopW = 1920, kTopH = 1080;    // top pass: R8G8B8A8_UNORM + D32_FLOAT
constexpr unsigned kVariants = 12, kRootSignatures = 4;
constexpr unsigned kStepsA = 48, kStepsB = 40;    // rise dispatches per frame: async compute (set A), direct (set B)

enum class Cull : unsigned { none = 0, back = 1, front = 2 };
inline Cull variant_cull(unsigned v) { return Cull(v % 3); }
inline unsigned variant_flavour(unsigned v) { return v / 3; }   // 0 plain, 1 blend, 2 FCC swapped, 3 LESS_EQUAL+bias
inline const char* flavour_name(unsigned f) {
    static const char* names[4] = {"plain", "blend ONE/ZERO", "FrontCounterClockwise, cull swapped",
                                   "LESS_EQUAL, depth bias 1"};
    return names[f & 3];
}

// What a draw is beyond lanterns::Draw.
struct Item {
    unsigned rs = 0, variant = 0;
    bool streamed = false;    // its instances live in the copy queue's stream buffer (re-copied every frame)
    bool particles = false;   // its instances are what the async compute pass wrote (set A)
};

struct World {
    Scene scene;                       // the main pass: meshes, draws in submission order, the main camera
    std::vector<Item> items;           // one per draw
    std::array<float, 16> aux_vp{}, top_vp{};
    std::vector<unsigned> aux_draws, top_draws;   // indices into scene.draws (the aux pass adds particle set B)
    unsigned particle_mesh = 0, floor_mesh = 0;
    unsigned instanced_draws = 0, objects = 0;
};

inline std::uint32_t mix(std::uint32_t v) {
    v ^= v >> 16; v *= 0x7feb352dU; v ^= v >> 15; v *= 0x846ca68bU; v ^= v >> 16;
    return v;
}

// draws: the main pass's draws besides the particles and the floor (both drawn last).
inline World make_world(unsigned draws, const std::vector<Particle>& particles_a) {
    World w;
    auto& s = w.scene;
    s.meshes.push_back(prism("lantern", 6, 0.3, 0.0, 1.0));
    s.meshes.push_back(prism("cube", 4, 0.35355339, 0.0, 0.5));
    s.meshes.push_back(octahedron(0.35));
    s.meshes.push_back(pyramid(0.35, 0.7));
    s.meshes.push_back(prism("triangle", 3, 0.35, 0.0, 0.6));
    s.meshes.push_back(prism("pentagon", 5, 0.3, 0.0, 0.8));
    s.meshes.push_back(prism("tall hexagon", 6, 0.25, 0.0, 1.2));   // a Face holds at most 6 vertices
    w.particle_mesh = unsigned(s.meshes.size());
    s.meshes.push_back(light_point());
    w.floor_mesh = unsigned(s.meshes.size());
    s.meshes.push_back(floor_quad(-45.0, 45.0, -3.0, 90.0));
    constexpr unsigned kColumns = 100;
    for (unsigned d = 0; d < draws; ++d) {
        const std::uint32_t h = mix(d * 2654435761u + 12345u);
        const unsigned col = d % kColumns, row = d / kColumns;
        const float cx = -42.5f + 0.85f * float(col) + 0.125f * float((h >> 3) % 3);
        const float cz = 5.0f + 1.5f * float(row) + 0.25f * float((h >> 5) % 3);
        Draw draw{(h >> 7) % 7, false, {}, rotation_y(double((h >> 11) % 360)),
                  {1.0f, float(6 + (h >> 20) % 3) / 8.0f, 1.0f, 1.0f}};
        const std::array<int, 3> base = {int(2 + (h >> 13) % 6), int(2 + (h >> 16) % 6), int(2 + (h >> 23) % 6)};
        Item item;
        item.rs = (d / 37) % kRootSignatures;
        item.variant = (d / 5 + d / 61) % kVariants;
        if (h % 10 < 3) {   // instanced: a little cluster of 2..16
            const unsigned count = 2 + (h >> 25) % 15;
            for (unsigned i = 0; i < count; ++i)
                draw.instances.push_back(instance(cx - 0.3f + 0.2f * float(i % 4), 0.0f, cz - 0.3f + 0.2f * float(i / 4),
                                                  0.375f, base, i));
            item.streamed = true;
            ++w.instanced_draws;
        } else {
            draw.instances.push_back(instance(cx, 0.0f, cz, 0.75f + 0.125f * float((h >> 27) % 3), base, d & 3));
        }
        w.objects += unsigned(draw.instances.size());
        s.draws.push_back(draw);
        w.items.push_back(item);
    }
    Draw points{w.particle_mesh, false, particle_instances(particles_a), rotation_y(0), {1.0f, 1.0f, 1.0f, 1.0f}};
    points.gpu_instances = true;
    s.draws.push_back(points);
    Item pitem;
    pitem.particles = true;
    w.items.push_back(pitem);
    Draw floor{w.floor_mesh, true, {}, rotation_y(0), {1.0f, 0.875f, 0.75f, 1.0f}};
    floor.instances.push_back({{0.0f, 0.0f, 0.0f}, 1.0f, {1.0f, 1.0f, 1.0f, 1.0f}});
    s.draws.push_back(floor);
    w.items.push_back(Item{});
    w.objects += kParticles + 1;
    for (unsigned d = 0; d < draws; ++d) {
        if (d % 4 == 0) w.aux_draws.push_back(d);
        if (d % 8 == 1) w.top_draws.push_back(d);
    }
    const unsigned floor_index = unsigned(s.draws.size() - 1);
    w.aux_draws.push_back(floor_index);
    w.top_draws.push_back(floor_index);
    const double eye[3] = {0.0, 6.5, -9.0}, target[3] = {0.0, 0.0, 30.0};
    s.view_projection = camera(eye, target, 50.0, double(kW) / double(kH), 0.5, 200.0);
    const double aux_eye[3] = {-30.0, 10.0, 10.0}, aux_target[3] = {5.0, 0.0, 40.0};
    w.aux_vp = camera(aux_eye, aux_target, 55.0, double(kAuxW) / double(kAuxH), 0.5, 200.0);
    const double top_eye[3] = {0.0, 80.0, 20.0}, top_target[3] = {0.0, 0.0, 42.0};
    w.top_vp = camera(top_eye, top_target, 60.0, double(kTopW) / double(kTopH), 1.0, 300.0);
    return w;
}

// The pixels a check looks at: (ox + i * sx, oy + j * sy).
struct Lattice {
    unsigned sx = 1, sy = 1, ox = 0, oy = 0;
};

struct SparseCheck {
    unsigned long long certain = 0, exact = 0, uncertain = 0, plausible = 0;
    unsigned culled_faces = 0, unsure_orientation = 0, degenerate_faces = 0;
    unsigned first_bad_x = 0, first_bad_y = 0;
    Color first_bad_gpu{}, first_bad_ref{};
    // One byte per lattice pixel, row-major over the lattice: 0 exact, 1 uncertain and plausible, 2 wrong.
    std::vector<std::uint8_t> marks;
    // Wrong pixels by the pipeline variant of the draw the reference expects there (kVariants: the clear colour).
    std::array<unsigned long long, kVariants + 1> wrong_by_variant{};
    bool pass() const { return exact == certain && plausible == uncertain; }
};

// The main pass at the lattice pixels, compared with gpu(x, y) (a Color). Same rules as reference_frame (depth test
// LESS, uncertain pixels within kMargin of an edge, a checker border or a depth near-tie), plus each draw's cull mode:
// with the default winding a face is front-facing when it is clockwise on the screen (y down), so its shoelace area
// is positive. A face so thin that snapping could flip its orientation may be culled or not: its fragments only count
// as candidates.
template <class Fetch>
SparseCheck check_sparse(const World& w, const Lattice& lat, Fetch gpu, unsigned band_rows = 256) {
    using namespace detail;
    SparseCheck out;
    const Scene& scene = w.scene;
    const unsigned cols = (kW - lat.ox + lat.sx - 1) / lat.sx, rows = (kH - lat.oy + lat.sy - 1) / lat.sy;
    out.marks.assign(std::size_t(cols) * rows, 0);
    bool reported = false;
    std::vector<std::uint8_t> counted;
    for (unsigned band_y0 = 0; band_y0 < rows; band_y0 += band_rows) {
        const unsigned band_n = std::min(band_rows, rows - band_y0);
        const int py0 = int(lat.oy + band_y0 * lat.sy), py1 = int(lat.oy + (band_y0 + band_n - 1) * lat.sy);
        std::vector<std::vector<Fragment>> frags(std::size_t(cols) * band_n);
        unsigned seq = 0;
        std::vector<std::uint32_t> seq_draw;   // face sequence number -> draw
        for (std::size_t d = 0; d < scene.draws.size(); ++d) {
            const Draw& draw = scene.draws[d];
            const Cull cull = variant_cull(w.items[d].variant);
            const Mesh& mesh = scene.meshes[draw.mesh];
            for (const Instance& inst : draw.instances)
                for (const Face& face : mesh.faces) {
                    const unsigned self = seq++;
                    seq_draw.push_back(std::uint32_t(d));
                    ScreenVertex sv[6];
                    double minx = 1e30, maxx = -1e30, miny = 1e30, maxy = -1e30;
                    for (unsigned k = 0; k < face.count; ++k) {
                        sv[k] = transform(scene, draw, inst, mesh.vertices[face.v[k]], kW, kH);
                        minx = std::min(minx, sv[k].x), maxx = std::max(maxx, sv[k].x);
                        miny = std::min(miny, sv[k].y), maxy = std::max(maxy, sv[k].y);
                    }
                    if (maxy < py0 - 1 || miny > py1 + 1 || maxx < -1 || minx > kW + 1) continue;
                    double area = 0, perimeter = 0;
                    for (unsigned k = 0; k < face.count; ++k) {
                        const auto& a = sv[k];
                        const auto& b = sv[(k + 1) % face.count];
                        area += cross(a.x, a.y, b.x, b.y);
                        perimeter += std::hypot(b.x - a.x, b.y - a.y);
                    }
                    const double a012 = cross(sv[1].x - sv[0].x, sv[1].y - sv[0].y, sv[2].x - sv[0].x, sv[2].y - sv[0].y);
                    // Each face is counted once, in the first band it reaches.
                    if (counted.size() <= self) counted.resize(self + 1, 0);
                    const bool first_band = !counted[self];
                    counted[self] = 1;
                    if (std::fabs(area) < 1e-6 || std::fabs(a012) < 1e-9) {
                        if (first_band) ++out.degenerate_faces;
                        continue;
                    }
                    // Twice the area over the perimeter: about the width of the face; under 1/16 pixel its sign is
                    // not certain after snapping.
                    const bool unsure = std::fabs(area) / perimeter < 1.0 / 16.0;
                    const bool front = area > 0;
                    if (!unsure && ((cull == Cull::back && !front) || (cull == Cull::front && front))) {
                        if (first_band) ++out.culled_faces;
                        continue;
                    }
                    if (unsure && cull != Cull::none && first_band) ++out.unsure_orientation;
                    const double orient = area > 0 ? 1.0 : -1.0;
                    const auto bary = [&](double px, double py, double l[3]) {
                        l[1] = cross(px - sv[0].x, py - sv[0].y, sv[2].x - sv[0].x, sv[2].y - sv[0].y) / a012;
                        l[2] = cross(sv[1].x - sv[0].x, sv[1].y - sv[0].y, px - sv[0].x, py - sv[0].y) / a012;
                        l[0] = 1.0 - l[1] - l[2];
                    };
                    const auto depth_at = [&](double px, double py) {
                        double l[3];
                        bary(px, py, l);
                        return l[0] * sv[0].z + l[1] * sv[1].z + l[2] * sv[2].z;
                    };
                    const auto world_at = [&](double px, double py, double wo[3]) {
                        double l[3];
                        bary(px, py, l);
                        const double qs = l[0] * sv[0].q + l[1] * sv[1].q + l[2] * sv[2].q;
                        wo[0] = (l[0] * sv[0].wx * sv[0].q + l[1] * sv[1].wx * sv[1].q + l[2] * sv[2].wx * sv[2].q) / qs;
                        wo[1] = (l[0] * sv[0].wy * sv[0].q + l[1] * sv[1].wy * sv[1].q + l[2] * sv[2].wy * sv[2].q) / qs;
                        wo[2] = (l[0] * sv[0].wz * sv[0].q + l[1] * sv[1].wz * sv[1].q + l[2] * sv[2].wz * sv[2].q) / qs;
                    };
                    const double dzdx = depth_at(1, 0) - depth_at(0, 0), dzdy = depth_at(0, 1) - depth_at(0, 0);
                    const double tol = (std::fabs(dzdx) + std::fabs(dzdy)) * kMargin + 1e-6;
                    const Color lit = face_color(draw, inst, mesh.vertices[face.v[0]].shade);
                    const int x0 = std::max(0, int(std::floor(minx)) - 1), x1 = std::min(int(kW) - 1, int(maxx) + 1);
                    const int y0 = std::max(py0, int(std::floor(miny)) - 1), y1 = std::min(py1, int(maxy) + 1);
                    if (x0 > x1 || y0 > y1) continue;
                    // The first lattice column and row inside the box.
                    const unsigned i0 = x0 <= int(lat.ox) ? 0 : (unsigned(x0) - lat.ox + lat.sx - 1) / lat.sx;
                    const unsigned j0 = (unsigned(y0) - lat.oy + lat.sy - 1) / lat.sy;
                    for (unsigned j = j0; j < band_y0 + band_n; ++j) {
                        const int y = int(lat.oy + j * lat.sy);
                        if (y > y1) break;
                        for (unsigned i = i0; i < cols; ++i) {
                            const int x = int(lat.ox + i * lat.sx);
                            if (x > x1) break;
                            const double px = x + 0.5, py = y + 0.5;
                            double inside = 1e30;
                            for (unsigned k = 0; k < face.count; ++k) {
                                const auto& a = sv[k];
                                const auto& b = sv[(k + 1) % face.count];
                                const double ex = b.x - a.x, ey = b.y - a.y;
                                inside = std::min(inside, orient * cross(ex, ey, px - a.x, py - a.y) /
                                                              std::sqrt(ex * ex + ey * ey));
                            }
                            if (inside < -kMargin) continue;
                            Fragment f{self, depth_at(px, py), tol, inside > kMargin && !(unsure && cull != Cull::none), 1, {lit, lit}};
                            if (draw.floor)
                                checker_shade(PixelSite{&draw, &inst, lit, px, py, world_at, mesh.vertices[face.v[0]].shade}, f);
                            frags[std::size_t(j - band_y0) * cols + i].push_back(f);
                        }
                    }
                }
        }
        for (unsigned j = 0; j < band_n; ++j)
            for (unsigned i = 0; i < cols; ++i) {
                const auto& list = frags[std::size_t(j) * cols + i];
                const Fragment* winner = nullptr;
                for (const Fragment& f : list)
                    if (f.certain_cover && (!winner || f.depth < winner->depth)) winner = &f;
                Color expect{kClear[0], kClear[1], kClear[2], kClear[3]};
                std::vector<Color> cand;
                if (winner) {
                    expect = winner->colors[0];
                    for (unsigned k = 0; k < winner->count; ++k) add_unique(cand, winner->colors[k]);
                } else {
                    add_unique(cand, expect);
                }
                for (const Fragment& f : list) {
                    if (&f == winner) continue;
                    if (winner && f.depth - f.tol > winner->depth + winner->tol) continue;
                    for (unsigned k = 0; k < f.count; ++k) add_unique(cand, f.colors[k]);
                }
                const unsigned x = lat.ox + i * lat.sx, y = lat.oy + (band_y0 + j) * lat.sy;
                const Color got = gpu(x, y);
                bool ok;
                std::uint8_t& mark = out.marks[std::size_t(band_y0 + j) * cols + i];
                if (cand.size() <= 1) {
                    ++out.certain;
                    ok = same(got, expect);
                    out.exact += ok;
                    mark = ok ? 0 : 2;
                } else {
                    ++out.uncertain;
                    ok = std::any_of(cand.begin(), cand.end(), [&](const Color& c) { return same(got, c); });
                    out.plausible += ok;
                    mark = ok ? 1 : 2;
                }
                if (!ok) ++out.wrong_by_variant[winner ? w.items[seq_draw[winner->seq]].variant : kVariants];
                if (!ok && !reported) {
                    reported = true;
                    out.first_bad_x = x, out.first_bad_y = y, out.first_bad_gpu = got, out.first_bad_ref = expect;
                }
            }
    }
    return out;
}

}  // namespace lanterns::scale