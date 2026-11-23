// Lanterns, stage 1: the scene, as data both the guest's draws and its CPU reference read (lanterns_reference.h).
// Everything here is ours and abstract: a few convex solids (hexagonal prisms, cubes, octahedra, square pyramids)
// standing on a floor plane, seen by a perspective camera. Colours come from short binary fractions (multiples of
// 1/8 for the instance colours, the per-draw tint and the per-face shade; 1/16 for the floor's two checker colours),
// so every product the shaders form is exact in float and the CPU reference can compare interior pixels bit for bit.
// Positions and matrices are ordinary floats: the reference treats the pixels they make uncertain (edges, depth
// near-ties, checker borders) apart. No Windows or D3D12 types: lanterns.cpp turns this into buffers and draws.
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <vector>

namespace lanterns {

constexpr unsigned kWidth = 384, kHeight = 256;   // render target, R32G32B32A32_FLOAT
constexpr unsigned kPitch = kWidth * 16;          // 6144 bytes, a multiple of D3D12's 256
constexpr std::array<float, 4> kClear = {0.0625f, 0.0625f, 0.125f, 1.0f};

// One vertex as vs_lanterns.asm fetches it through the mesh V# (stride 16): position xyz, then the face's shade.
struct Vertex {
    float x, y, z, shade;
};
// One instance as vs_lanterns.asm fetches it through the instance V# (stride 32): offset xyz, uniform scale, colour.
struct Instance {
    float offset[3];
    float scale;
    float color[4];
};
static_assert(sizeof(Vertex) == 16 && sizeof(Instance) == 32);

// A convex planar face: 3 to 6 vertex indices into its mesh, wound consistently; the index buffer fans it.
struct Face {
    unsigned count;
    std::array<std::uint16_t, 6> v;
};

struct Mesh {
    const char* name;
    std::vector<Vertex> vertices;   // each face has its own vertices, so a face's shade is constant across it
    std::vector<Face> faces;
};

// The shade of a face from its local normal and a fixed light: 0.5 + 0.5 * max(0, n.l), rounded to 1/8, clamped to
// [0.25, 1]. Integer eighths keep the colour products exact.
inline float face_shade(double nx, double ny, double nz) {
    const double lx = -0.4, ly = 0.8, lz = -0.45;   // towards the light, roughly from above and in front
    const double len = std::sqrt(nx * nx + ny * ny + nz * nz) * std::sqrt(lx * lx + ly * ly + lz * lz);
    const double d = (nx * lx + ny * ly + nz * lz) / len;
    long eighths = std::lround((0.5 + 0.5 * (d > 0 ? d : 0.0)) * 8.0);
    eighths = eighths < 2 ? 2 : eighths > 8 ? 8 : eighths;
    return float(eighths) / 8.0f;
}

// Adds one face from positions; the shade comes from the face's own normal (Newell's method), turned to point away
// from `inside` (a point inside the convex solid), and the face is wound the same way for every face.
inline void add_face(Mesh& mesh, std::vector<std::array<double, 3>> p, const std::array<double, 3>& inside) {
    double nx = 0, ny = 0, nz = 0, cx = 0, cy = 0, cz = 0;
    for (std::size_t i = 0; i < p.size(); ++i) {
        const auto& a = p[i];
        const auto& b = p[(i + 1) % p.size()];
        nx += (a[1] - b[1]) * (a[2] + b[2]);
        ny += (a[2] - b[2]) * (a[0] + b[0]);
        nz += (a[0] - b[0]) * (a[1] + b[1]);
        cx += a[0], cy += a[1], cz += a[2];
    }
    const double n = double(p.size());
    if (nx * (cx / n - inside[0]) + ny * (cy / n - inside[1]) + nz * (cz / n - inside[2]) < 0) {
        nx = -nx, ny = -ny, nz = -nz;
        std::reverse(p.begin(), p.end());
    }
    const float shade = face_shade(nx, ny, nz);
    Face face{unsigned(p.size()), {}};
    for (std::size_t i = 0; i < p.size(); ++i) {
        face.v[i] = std::uint16_t(mesh.vertices.size());
        mesh.vertices.push_back({float(p[i][0]), float(p[i][1]), float(p[i][2]), shade});
    }
    mesh.faces.push_back(face);
}

// A prism with an n-gon cross-section, axis along y from y0 to y1, circumradius r.
inline Mesh prism(const char* name, unsigned n, double r, double y0, double y1) {
    Mesh m{name, {}, {}};
    const double pi = 3.14159265358979323846;
    const std::array<double, 3> centre{0.0, (y0 + y1) / 2, 0.0};
    std::vector<std::array<double, 3>> ring(n);
    for (unsigned i = 0; i < n; ++i)
        ring[i] = {r * std::cos(2 * pi * i / n), 0.0, r * std::sin(2 * pi * i / n)};
    for (unsigned i = 0; i < n; ++i) {
        const auto& a = ring[i];
        const auto& b = ring[(i + 1) % n];
        add_face(m, {{a[0], y0, a[2]}, {a[0], y1, a[2]}, {b[0], y1, b[2]}, {b[0], y0, b[2]}}, centre);
    }
    std::vector<std::array<double, 3>> top, bottom;
    for (unsigned i = 0; i < n; ++i) {
        top.push_back({ring[i][0], y1, ring[i][2]});
        bottom.push_back({ring[i][0], y0, ring[i][2]});
    }
    add_face(m, top, centre);
    add_face(m, bottom, centre);
    return m;
}

inline Mesh octahedron(double r) {
    Mesh m{"octahedron", {}, {}};
    for (double sy : {1.0, -1.0})
        for (double sx : {1.0, -1.0})
            for (double sz : {1.0, -1.0})
                add_face(m, {{sx * r, 0, 0}, {0, sy * r, 0}, {0, 0, sz * r}}, {0, 0, 0});
    // Lift it so its lowest vertex rests at y = 0 when the instance offset is on the floor.
    for (Vertex& v : m.vertices) v.y += float(r);
    return m;
}

inline Mesh pyramid(double half, double height) {
    Mesh m{"pyramid", {}, {}};
    const std::array<double, 3> c[4] = {{-half, 0, -half}, {half, 0, -half}, {half, 0, half}, {-half, 0, half}};
    const std::array<double, 3> apex{0, height, 0};
    const std::array<double, 3> centre{0, height / 4, 0};
    for (int i = 0; i < 4; ++i) add_face(m, {c[i], apex, c[(i + 1) % 4]}, centre);
    add_face(m, {c[0], c[1], c[2], c[3]}, centre);
    return m;
}

inline Mesh floor_quad(double x0, double x1, double z0, double z1) {
    Mesh m{"floor", {}, {}};
    Face face{4, {0, 1, 2, 3}};
    m.vertices = {{float(x0), 0, float(z0), 1}, {float(x0), 0, float(z1), 1}, {float(x1), 0, float(z1), 1},
                  {float(x1), 0, float(z0), 1}};
    m.faces.push_back(face);
    return m;
}

// The floor pixel shader's two checker colours, the data words ps_lanterns_floor.asm carries in its own FSSM program
// (read through s_getpc_b64, the code base): colour A where floor(x) + floor(z) is even, B where it is odd.
constexpr std::array<float, 4> kFloorA = {0.4375f, 0.4375f, 0.5f, 1.0f};
constexpr std::array<float, 4> kFloorB = {0.125f, 0.1875f, 0.25f, 1.0f};

// One DrawIndexedInstanced: a mesh, its instances, the root constants (a 3x4 model matrix, row-major, then a tint).
struct Draw {
    unsigned mesh;
    bool floor;                     // drawn with ps_lanterns_floor (the checker) instead of ps_lanterns_lit
    std::vector<Instance> instances;
    std::array<float, 12> model;    // rows: local = M * (x, y, z, 1)
    std::array<float, 4> tint;
};

inline std::array<float, 12> rotation_y(double degrees) {
    const double a = degrees * 3.14159265358979323846 / 180.0;
    const float c = float(std::cos(a)), s = float(std::sin(a));
    return {c, 0, s, 0, 0, 1, 0, 0, -s, 0, c, 0};
}

// Instance colour i of a family: the base colour in eighths, red raised by i mod 2 and blue by i / 2 (capped at 8/8).
inline Instance instance(float x, float y, float z, float scale, std::array<int, 3> base_eighths, unsigned i) {
    const int r = std::min(8, base_eighths[0] + int(i % 2)), b = std::min(8, base_eighths[2] + int(i / 2));
    return {{x, y, z}, scale, {float(r) / 8.0f, float(base_eighths[1]) / 8.0f, float(b) / 8.0f, 1.0f}};
}

struct Scene {
    std::vector<Mesh> meshes;
    std::vector<Draw> draws;
    std::array<float, 16> view_projection;   // rows: clip = VP * (world, 1); the root CBV's 64 bytes
};

// Camera: left-handed, y up, looking from the eye at the target; D3D-style depth (0 at near, 1 at far).
inline std::array<float, 16> camera(const double eye[3], const double target[3], double fov_y_deg, double aspect,
                                    double znear, double zfar) {
    double f[3] = {target[0] - eye[0], target[1] - eye[1], target[2] - eye[2]};
    const double fl = std::sqrt(f[0] * f[0] + f[1] * f[1] + f[2] * f[2]);
    for (double& v : f) v /= fl;
    double r[3] = {f[2], 0, -f[0]};   // up (0,1,0) x forward
    const double rl = std::sqrt(r[0] * r[0] + r[2] * r[2]);
    for (double& v : r) v /= rl;
    const double u[3] = {f[1] * r[2] - f[2] * r[1], f[2] * r[0] - f[0] * r[2], f[0] * r[1] - f[1] * r[0]};
    const double ys = 1.0 / std::tan(fov_y_deg * 3.14159265358979323846 / 360.0), xs = ys / aspect;
    const double zf = zfar / (zfar - znear), zn = -znear * zfar / (zfar - znear);
    const auto dot_eye = [&](const double* a) { return -(a[0] * eye[0] + a[1] * eye[1] + a[2] * eye[2]); };
    const double view[3][4] = {{r[0], r[1], r[2], dot_eye(r)}, {u[0], u[1], u[2], dot_eye(u)},
                               {f[0], f[1], f[2], dot_eye(f)}};
    std::array<float, 16> m{};
    for (int c = 0; c < 4; ++c) {
        m[0 + c] = float(xs * view[0][c]);
        m[4 + c] = float(ys * view[1][c]);
        m[8 + c] = float(zf * view[2][c] + (c == 3 ? zn : 0.0));
        m[12 + c] = float(view[2][c]);
    }
    return m;
}

// Stage 1. Draws are submitted nearest family first and the floor last, so with the depth test off the floor paints
// over everything and the far solids over the near ones: the negative control has plenty to change.
inline Scene stage1() {
    Scene s;
    s.meshes.push_back(prism("lantern", 6, 0.45, 0.0, 1.5));   // 0: hexagonal prisms, the "lanterns"
    s.meshes.push_back(prism("cube", 4, 0.70710678, 0.0, 1.0));  // 1: square prism = a unit cube (turned 45 degrees)
    s.meshes.push_back(octahedron(0.6));                       // 2
    s.meshes.push_back(pyramid(0.6, 1.1));                     // 3
    s.meshes.push_back(floor_quad(-6.0, 6.0, -3.0, 9.0));      // 4

    Draw lanterns{0, false, {}, rotation_y(10), {1.0f, 1.0f, 1.0f, 1.0f}};
    for (unsigned i = 0; i < 3; ++i)
        lanterns.instances.push_back(instance(-2.25f + 2.25f * float(i), 0.0f, -0.5f + 0.75f * float(i % 2), 1.0f,
                                              {7, 5, 1}, i));
    Draw cubes{1, false, {}, rotation_y(20), {1.0f, 0.875f, 1.0f, 1.0f}};
    const float cube_x[3] = {-1.25f, 1.125f, 3.375f};
    for (unsigned i = 0; i < 3; ++i)
        cubes.instances.push_back(instance(cube_x[i], 0.0f, 2.0f, 0.875f + 0.125f * float(i % 2), {2, 6, 6}, i));
    Draw octahedra{2, false, {}, rotation_y(30), {1.0f, 1.0f, 1.0f, 1.0f}};
    for (unsigned i = 0; i < 2; ++i)
        octahedra.instances.push_back(instance(-1.5f + 3.0f * float(i), 0.0f, 4.0f, 1.25f, {5, 3, 7}, i));
    Draw pyramids{3, false, {}, rotation_y(-15), {1.0f, 1.0f, 0.875f, 1.0f}};
    for (unsigned i = 0; i < 2; ++i)
        pyramids.instances.push_back(instance(-4.0f + 8.0f * float(i), 0.0f, 5.5f, 1.5f, {7, 3, 2}, i));
    Draw floor{4, true, {}, rotation_y(0), {1.0f, 0.875f, 0.75f, 1.0f}};
    floor.instances.push_back({{0.0f, 0.0f, 0.0f}, 1.0f, {1.0f, 1.0f, 1.0f, 1.0f}});
    s.draws = {lanterns, cubes, octahedra, pyramids, floor};

    const double eye[3] = {0.5, 3.25, -6.5}, target[3] = {0.0, 0.6, 2.5};
    s.view_projection = camera(eye, target, 52.0, double(kWidth) / double(kHeight), 0.5, 40.0);
    return s;
}

}  // namespace lanterns
