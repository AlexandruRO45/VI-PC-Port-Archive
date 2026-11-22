#pragma once

// Scene file (tools/sandbox_scene.py), SPIR-V loading, the sandbox camera and PPM output, shared by vk_sandbox
// and vk_gbuffer. Nothing here is game data.

#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iterator>
#include <vector>

namespace leonida::sandbox {

struct Texture { uint32_t w, h; std::vector<uint8_t> rgba; };
struct Mesh { uint32_t first_vertex, vertex_count, first_index, index_count; };
struct Draw { uint32_t mesh; int32_t texture; float rows[4][3]; };
struct Scene {
    std::vector<Texture> textures;
    std::vector<Mesh> meshes;
    std::vector<Draw> draws;
    std::vector<float> vertices;     // 8 floats each
    std::vector<uint32_t> indices;
    float center[3];
};

inline bool load_scene(const char* path, Scene& s) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    const uint64_t file_size = f ? uint64_t(f.tellg()) : 0;
    f.seekg(0);
    char magic[4];
    uint32_t hdr[4];
    if (!f.read(magic, 4) || std::memcmp(magic, "LSB1", 4) || !f.read(reinterpret_cast<char*>(hdr), 16) ||
        hdr[0] != 1 || !f.read(reinterpret_cast<char*>(s.center), 12))
        return false;
    for (uint32_t i = 0; i < hdr[1]; ++i) {
        Texture t;
        f.read(reinterpret_cast<char*>(&t.w), 4).read(reinterpret_cast<char*>(&t.h), 4);
        if (!f || !t.w || !t.h || t.w > 16384 || t.h > 16384 || uint64_t(t.w) * t.h * 4 > file_size) return false;
        t.rgba.resize(size_t(t.w) * t.h * 4);
        f.read(reinterpret_cast<char*>(t.rgba.data()), std::streamsize(t.rgba.size()));
        s.textures.push_back(std::move(t));
    }
    for (uint32_t i = 0; i < hdr[2]; ++i) {
        uint32_t vc = 0, ic = 0;
        f.read(reinterpret_cast<char*>(&vc), 4).read(reinterpret_cast<char*>(&ic), 4);
        if (!f || uint64_t(vc) * 32 + uint64_t(ic) * 4 > file_size) return false;
        Mesh m{uint32_t(s.vertices.size() / 8), vc, uint32_t(s.indices.size()), ic};
        s.vertices.resize(s.vertices.size() + size_t(vc) * 8);
        s.indices.resize(s.indices.size() + ic);
        f.read(reinterpret_cast<char*>(s.vertices.data() + size_t(m.first_vertex) * 8), std::streamsize(vc) * 32);
        f.read(reinterpret_cast<char*>(s.indices.data() + m.first_index), std::streamsize(ic) * 4);
        if (!f) return false;
        for (uint32_t k = 0; k < ic; ++k)
            if (s.indices[m.first_index + k] >= vc) return false;
        s.meshes.push_back(m);
    }
    for (uint32_t i = 0; i < hdr[3]; ++i) {
        Draw d;
        f.read(reinterpret_cast<char*>(&d), sizeof(d));
        if (!f || d.mesh >= s.meshes.size() || d.texture >= int32_t(s.textures.size())) return false;
        s.draws.push_back(d);
    }
    return bool(f);
}

inline std::vector<uint32_t> read_spv(const char* p) {
    std::ifstream f(p, std::ios::binary);
    std::vector<char> b((std::istreambuf_iterator<char>(f)), {});
    std::vector<uint32_t> w(b.size() / 4);
    std::memcpy(w.data(), b.data(), w.size() * 4);
    return w;
}

struct Camera {
    float pos[3], yaw = 0, pitch = 0, fov = 70, speed = 0;
    int mode = 0;   // 0 fly, 1 walk, 2 drive
    void forward(float f[3]) const {
        f[0] = std::cos(pitch) * std::cos(yaw); f[1] = std::cos(pitch) * std::sin(yaw); f[2] = std::sin(pitch);
    }
};

// clip = wx*row0 + wy*row1 + wz*row2 + row3 (D3D clip space: y up, z 0..1), stored as 16 floats.
inline void view_projection(const Camera& c, float aspect, float out[16]) {
    float f[3]; c.forward(f);
    float r[3] = {f[1], -f[0], 0};
    const float rl = std::sqrt(r[0] * r[0] + r[1] * r[1]);
    if (rl < 1e-6f) { r[0] = std::sin(c.yaw); r[1] = -std::cos(c.yaw); } else { r[0] /= rl; r[1] /= rl; }
    const float u[3] = {r[1] * f[2] - r[2] * f[1], r[2] * f[0] - r[0] * f[2], r[0] * f[1] - r[1] * f[0]};
    const float n = 0.1f, fa = 5000.f, sy = 1.f / std::tan(c.fov * 3.14159265f / 360.f), sx = sy / aspect;
    const float A = fa / (fa - n), B = -n * fa / (fa - n);
    // view = (dot(w - p, r), dot(w - p, u), dot(w - p, f)); clip = (vx*sx, vy*sy, vz*A + B, vz)
    const float* axes[3] = {r, u, f};
    const float scale[4] = {sx, sy, A, 1};
    for (int k = 0; k < 4; ++k) {
        const float* a = axes[k < 3 ? k : 2];
        for (int j = 0; j < 3; ++j) out[j * 4 + k] = a[j] * scale[k];
        out[12 + k] = -(a[0] * c.pos[0] + a[1] * c.pos[1] + a[2] * c.pos[2]) * scale[k] + (k == 2 ? B : 0);
    }
}


inline bool write_ppm(const char* path, const uint8_t* bgra_or_rgba, uint32_t w, uint32_t h, bool bgra) {
    std::ofstream o(path, std::ios::binary);
    o << "P6\n" << w << " " << h << "\n255\n";
    for (size_t i = 0; i < size_t(w) * h; ++i) {
        const uint8_t* p = bgra_or_rgba + i * 4;
        const char px[3] = {char(p[bgra ? 2 : 0]), char(p[1]), char(p[bgra ? 0 : 2])};
        o.write(px, 3);
    }
    return bool(o);
}


}  // namespace leonida::sandbox
