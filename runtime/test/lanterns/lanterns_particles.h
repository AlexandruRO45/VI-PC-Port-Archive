// Lanterns, stage 3: the rising light points. Their state lives on the GPU and a compute shader
// (shaders/cs_lanterns_rise.asm) moves it one step per dispatch; this is the same integer arithmetic on the CPU, so the
// state the GPU reads back can be compared word for word, and the instance records it writes for the draw bit for bit.
// Positions are fixed point in 1/256 units: converting them to float and scaling by 1/256 is exact.
#pragma once

#include "lanterns_scene.h"

#include <array>
#include <cstdint>
#include <vector>

namespace lanterns {

constexpr unsigned kParticles = 256;          // 4 groups of 64 threads
constexpr std::int32_t kTop = 896;            // 3.5 units: a point at or above it ...
constexpr std::int32_t kRange = 832;          // ... drops back by 3.25 units
constexpr unsigned kStepsPerFrame = 24;

// One point as cs_lanterns_rise.asm reads and writes it through its state V# (stride 16).
struct Particle {
    std::int32_t x, y, z;
    std::uint32_t packed;   // bits 0-7: rise per step (1/256 units); bit 8: big; bits 9-10: green; bits 11-12: blue
};
static_assert(sizeof(Particle) == 16);

inline std::vector<Particle> initial_particles() {
    std::vector<Particle> p(kParticles);
    for (unsigned i = 0; i < kParticles; ++i) {
        p[i].x = (std::int32_t((i * 53) % 193) - 96) * 6;      // -2.25 .. 2.25 units
        p[i].z = std::int32_t((i * 29) % 113) * 12 - 256;      // -1 .. 4.3 units
        p[i].y = 64 + std::int32_t((i * 97) % 832);            // 0.25 .. 3.5 units
        p[i].packed = (4 + (i * 7) % 9) | (((i * 13) & 0xff) << 8);
    }
    return p;
}

// One dispatch of cs_lanterns_rise.asm: y += speed; if (y >= top) y -= range.
inline void step_particles(std::vector<Particle>& p) {
    for (Particle& q : p) {
        q.y += std::int32_t(q.packed & 0xff);
        if (q.y >= kTop) q.y -= kRange;
    }
}

// The instance record the same dispatch writes for each point (what the draw's instance V# reads).
inline Instance particle_instance(const Particle& q) {
    const float big = float((q.packed >> 8) & 1), green = float(5 + ((q.packed >> 9) & 3)),
                blue = float(1 + ((q.packed >> 11) & 3));
    return {{float(q.x) * 0.00390625f, float(q.y) * 0.00390625f, float(q.z) * 0.00390625f},
            0.125f + 0.0625f * big,
            {1.0f, green * 0.125f, blue * 0.125f, 1.0f}};
}

inline std::vector<Instance> particle_instances(const std::vector<Particle>& p) {
    std::vector<Instance> out;
    for (const Particle& q : p) out.push_back(particle_instance(q));
    return out;
}

// A light point: an octahedron with every face at full shade, so it reads as one bright diamond.
inline Mesh light_point() {
    Mesh m = octahedron(1.0);
    m.name = "light point";
    for (Vertex& v : m.vertices) v.shade = 1.0f;
    return m;
}

// Stage 3: stage 1's scene plus the points as a sixth draw, whose instances come from the compute shader on the GPU.
// The instances here are what the reference draws: the CPU's simulation of the same steps.
inline Scene stage3(const std::vector<Particle>& state) {
    Scene s = stage1();
    s.meshes.push_back(light_point());
    Draw points{unsigned(s.meshes.size() - 1), false, particle_instances(state), rotation_y(0), {1.0f, 1.0f, 1.0f, 1.0f}};
    points.gpu_instances = true;
    s.draws.push_back(points);
    return s;
}

}  // namespace lanterns
