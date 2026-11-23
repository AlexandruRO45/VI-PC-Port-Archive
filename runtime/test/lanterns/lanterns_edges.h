// Lanterns stage 1, the shared edges: eight triangles fanned around one vertex, every vertex on a pixel centre, each
// triangle a different flat colour, drawn with additive blending by a wave64 VS and PS that read their root data where
// the title's do (docs/d3d12_vulkan.md 3.5: the lowest table pointer at s8; eight root constants s0..s7 below it).
// The fan's inner edges run horizontally, vertically, at 1:1, 1:2, 3:2 and 1:4, so many pixel centres lie exactly on
// an edge two triangles share; D3D's top-left rule gives each such pixel to exactly one of them. With red = 2^k for
// triangle k, green and alpha 1 per triangle, a pixel drawn twice or not at all shows it. The CPU applies the rule with
// exact integers (coordinates in half pixels).
#pragma once

#include "lanterns_reference.h"

#include <array>
#include <cstdint>
#include <vector>

namespace lanterns {

constexpr unsigned kEdgeTriangles = 8;
constexpr float kEdgeWrongAttr = 1000.0f;   // the two non-provoking vertices' red: wrong if flat used them

// The fan in half pixels (a pixel centre x + 0.5 is 2x + 1): the centre, then the ring in screen order.
constexpr int kEdgeCentre[2] = {385, 257};
constexpr int kEdgeRing[kEdgeTriangles][2] = {{705, 257}, {705, 417}, {545, 497}, {225, 497},
                                              {65, 337},  {65, 97},   {385, 17},  {625, 17}};

struct EdgeVertex {
    float pos[4];    // screen x, y (pixels), z, w
    float attr[4];   // param0: red 2^k, green 1, blue 0, alpha unused
};
static_assert(sizeof(EdgeVertex) == 32);

// Triangle k: the centre (its provoking vertex, carrying the flat colour), ring k, ring k + 1.
inline std::vector<EdgeVertex> edge_vertices() {
    std::vector<EdgeVertex> out;
    for (unsigned k = 0; k < kEdgeTriangles; ++k) {
        const int* p[3] = {kEdgeCentre, kEdgeRing[k], kEdgeRing[(k + 1) % kEdgeTriangles]};
        for (unsigned v = 0; v < 3; ++v) {
            EdgeVertex e{{float(p[v][0]) * 0.5f, float(p[v][1]) * 0.5f, 0.5f, 1.0f}, {}};
            e.attr[0] = v == 0 ? float(1u << k) : kEdgeWrongAttr + float(k);
            e.attr[1] = v == 0 ? 1.0f : kEdgeWrongAttr;
            out.push_back(e);
        }
    }
    return out;
}

namespace detail {

// Whether triangle (a, b, c) in half pixels owns the point p (also in half pixels) under the top-left rule.
inline bool owns(const int* a, const int* b, const int* c, long long px, long long py) {
    const long long area = (long long)(b[0] - a[0]) * (c[1] - a[1]) - (long long)(b[1] - a[1]) * (c[0] - a[0]);
    if (area == 0) return false;
    const int* v[3] = {a, area > 0 ? b : c, area > 0 ? c : b};   // clockwise on screen (y down): inside is E > 0
    for (int e = 0; e < 3; ++e) {
        const int* s = v[e];
        const int* t = v[(e + 1) % 3];
        const long long dx = t[0] - s[0], dy = t[1] - s[1];
        const long long f = dx * (py - s[1]) - dy * (px - s[0]);
        if (f < 0) return false;
        if (f == 0 && !((dy == 0 && dx > 0) || dy < 0)) return false;   // only top and left edges own their points
    }
    return true;
}

}  // namespace detail

struct EdgeReference {
    std::vector<Color> color;   // red: the sum of 2^k over the owners, green and alpha: their count
    unsigned on_shared = 0;     // pixel centres exactly on an inner edge (the centre included)
    unsigned covered = 0;
};

inline EdgeReference edge_reference() {
    EdgeReference out;
    out.color.assign(std::size_t(kWidth) * kHeight, Color{0, 0, 0, 0});
    for (unsigned y = 0; y < kHeight; ++y)
        for (unsigned x = 0; x < kWidth; ++x) {
            const long long px = 2 * x + 1, py = 2 * y + 1;
            Color& c = out.color[std::size_t(y) * kWidth + x];
            for (unsigned k = 0; k < kEdgeTriangles; ++k)
                if (detail::owns(kEdgeCentre, kEdgeRing[k], kEdgeRing[(k + 1) % kEdgeTriangles], px, py))
                    c[0] += float(1u << k), c[1] += 1.0f, c[3] += 1.0f;
            out.covered += c[1] > 0;
            for (unsigned k = 0; k < kEdgeTriangles; ++k) {   // on segment centre -> ring k?
                const long long dx = kEdgeRing[k][0] - kEdgeCentre[0], dy = kEdgeRing[k][1] - kEdgeCentre[1];
                const long long qx = px - kEdgeCentre[0], qy = py - kEdgeCentre[1];
                const long long t = qx * dx + qy * dy;
                if (qx * dy - qy * dx == 0 && t >= 0 && t < dx * dx + dy * dy) {
                    ++out.on_shared;
                    break;
                }
            }
        }
    return out;
}

}  // namespace lanterns