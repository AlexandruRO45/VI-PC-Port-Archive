#pragma once
// CPU reference for triangle facing and face culling in DXR. It uses only the
// standard library so it can be compiled and checked without Direct3D.
#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace dxr_facing {
constexpr std::uint32_t cullBack = 0x10, cullFront = 0x20; // RAY_FLAG_CULL_{BACK,FRONT}_FACING_TRIANGLES
using Vec = std::array<double, 3>;
using Vertex = std::array<float, 3>;
struct Instance {
    std::uint32_t id;
    std::array<float, 3> scale; // Diagonal object-to-world scale; a negative entry mirrors.
    float translateX;
    bool counterclockwise;      // D3D12_RAYTRACING_INSTANCE_FLAG_TRIANGLE_FRONT_COUNTERCLOCKWISE
    bool cullDisable;           // D3D12_RAYTRACING_INSTANCE_FLAG_TRIANGLE_CULL_DISABLE
    const char* facing;         // Hand-derived: primitive 0 then 1, each seen along world +z then -z.
};
struct Ray {
    std::array<float, 3> origin, direction;
    float nearT, farT;
    std::uint32_t flags;
};
struct Hit {
    bool hit = false, front = false;
    std::uint32_t instance = 0, primitive = 0;
    double distance = 0;
};

// Primitive 0 appears clockwise and primitive 1 counter-clockwise when seen
// along object +z in a left-handed frame.
inline std::vector<Vertex> vertices() {
    return {{0,0,1}, {0,1,1}, {1,0,1}, {2,0,1}, {3,0,1}, {2,1,1}};
}
inline std::vector<Instance> instances() {
    return {
        {1, {1, 1, 1}, 0, false, false, "FBBF"},
        {2, {1, 1, 1}, 8, true, false, "BFFB"},
        {3, {1, 1, 1}, 16, false, true, "FBBF"},
        // Mirroring x reverses the world-space winding but not the facing.
        {4, {-2, 1, 4}, 32, false, false, "FBBF"},
        {5, {-2, .5f, 4}, 40, true, false, "BFFB"},
        // Mirroring z keeps the world-space winding but reverses the facing.
        {6, {1.5f, 1, -2}, 48, false, false, "BFFB"},
        {7, {1, -1, 1}, 56, true, true, "BFFB"},
        // The z direction in object space is +/-2/3 for a unit world direction.
        {8, {1, 1, 1.5f}, 64, false, false, "FBBF"},
    };
}
constexpr std::uint32_t rayFlags[]{0, cullBack, cullFront};
constexpr float nearDistance = 2, farDistance = 3; // Hit distances of the +z and -z rays.

// Twelve rays per instance: both primitives, both directions, three flag sets.
inline std::vector<Ray> rays() {
    std::vector<Ray> result;
    for (const auto& instance : instances())
        for (int primitive = 0; primitive < 2; ++primitive)
            for (int reversed = 0; reversed < 2; ++reversed)
                for (const auto flags : rayFlags) {
                    const float x = instance.scale[0] * (primitive ? 2.25f : .25f) + instance.translateX;
                    const float y = instance.scale[1] * .25f, z = instance.scale[2];
                    result.push_back({{x, y, reversed ? z + farDistance : z - nearDistance},
                                      {0, 0, reversed ? -1.f : 1.f}, .25f, 8, flags});
                }
    return result;
}

inline Vec sub(const Vec& a, const Vec& b) { return {a[0]-b[0], a[1]-b[1], a[2]-b[2]}; }
inline Vec cross(const Vec& a, const Vec& b) {
    return {a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]};
}
inline double dot(const Vec& a, const Vec& b) { return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]; }

// Brute force over every instance and triangle in double precision. The ray
// moves into object space, where DXR defines winding, and keeps its parameter.
inline Hit closest(const std::vector<Vertex>& mesh, const std::vector<Instance>& scene, const Ray& ray) {
    Hit best;
    for (const auto& instance : scene) {
        Vec origin, direction;
        for (int axis = 0; axis < 3; ++axis) {
            const double translate = axis == 0 ? instance.translateX : 0;
            const double scale = instance.scale[axis];
            origin[axis] = (ray.origin[axis] - translate) / scale;
            direction[axis] = ray.direction[axis] / scale;
        }
        for (std::uint32_t primitive = 0; primitive < mesh.size() / 3; ++primitive) {
            Vec v[3];
            for (int corner = 0; corner < 3; ++corner)
                for (int axis = 0; axis < 3; ++axis) v[corner][axis] = mesh[3*primitive + corner][axis];
            const Vec e1 = sub(v[1], v[0]), e2 = sub(v[2], v[0]), normal = cross(e1, e2);
            const double facing = dot(direction, normal);
            if (facing == 0) continue;
            const Vec delta = sub(origin, v[0]), q = cross(delta, direction);
            const double t = -dot(delta, normal) / facing;
            const double u = -dot(e2, q) / facing, w = dot(e1, q) / facing;
            // Include both geometric candidates at a shared edge. At equal distances
            // this reference keeps the first primitive; DXR may choose either neighbor.
            if (!(t > ray.nearT && t < ray.farT && u >= 0 && w >= 0 && u + w <= 1)) continue;
            // Clockwise from the ray origin in a left-handed frame is front facing.
            const bool front = (facing < 0) != instance.counterclockwise;
            if (!instance.cullDisable && (ray.flags & (front ? cullFront : cullBack))) continue;
            if (!best.hit || t < best.distance) best = {true, front, instance.id, primitive, t};
        }
    }
    return best;
}

// The outcome each ray must have, taken from the hand-derived facing letters
// without any intersection arithmetic.
inline Hit expected(const std::vector<Instance>& scene, std::size_t ray) {
    const auto& instance = scene[ray / 12];
    const std::uint32_t primitive = static_cast<std::uint32_t>(ray % 12 / 6);
    const bool reversed = ray % 6 >= 3;
    const std::uint32_t flags = rayFlags[ray % 3];
    const bool front = instance.facing[2*primitive + reversed] == 'F';
    if (!instance.cullDisable && (flags & (front ? cullFront : cullBack))) return {};
    return {true, front, instance.id, primitive, reversed ? farDistance : nearDistance};
}

// Returns the first ray where the geometric reference and the hand-derived
// table disagree, or the ray count when they agree everywhere.
inline std::size_t first_reference_mismatch() {
    const auto mesh = vertices();
    const auto scene = instances();
    const auto all = rays();
    for (std::size_t i = 0; i < all.size(); ++i) {
        const Hit a = closest(mesh, scene, all[i]), b = expected(scene, i);
        if (a.hit != b.hit || a.front != b.front || a.instance != b.instance ||
            a.primitive != b.primitive || a.distance != b.distance) return i;
    }
    return all.size();
}
}
