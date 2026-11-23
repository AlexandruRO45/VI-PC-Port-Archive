// Lanterns: an original stand-in title that runs under gta6rt --run and drives the whole chain a real title would:
// an x64 PE the loader maps, imports the loader's shims answer, the Xbox D3D12 device on Vulkan
// (D3D12XboxCreateDevice from d3d12_xs.dll, LEONIDA_D3D12=vulkan), pipelines from shader containers holding RDNA2
// programs (DXBC { DXIL { XDAT { XBCH + zlib(FSSM) } } }, packed by fssm_pack.py beside this file, translated offline by
// fssm_unpack.py, shader_disasm.py and rdna2spirv.py; the layer finds each by the SHA-256 of its FSSM program), draws,
// compute, and readbacks checked against a CPU reference. Content: abstract solids on a checkered floor and rising
// points of light, colours from short binary fractions; no game data. How to build and run it: lanterns.cmake.
//
//   lanterns.exe --stage 1|3 --shaders <folder fssm_pack.py wrote> --out <folder for the frame dumps>
//
// Stage 1: five DrawIndexedInstanced (hexagonal prisms, cubes, octahedra, pyramids, the floor) into a 384x256
// R32G32B32A32_FLOAT target with a D32_FLOAT depth buffer. Vertex, index and instance buffers live in DEFAULT heaps,
// filled from an upload buffer by copies; the vertex shader reads vertices and instances through V#s in two descriptor
// tables, the model matrix and tint from root constants, the camera from a root CBV. The frame is drawn twice: with
// the depth test (the image) and without it (the negative control, which must change exactly the pixels the reference
// says it changes).
//
// Stage 3: a compute shader on a COMPUTE queue moves 256 light points (integer fixed point) and writes their instance
// records into a DEFAULT UAV buffer; the direct queue waits on the compute fence and draws them as a sixth
// DrawIndexedInstanced whose instance V# points at that buffer. Frame 1: compute submitted, then the frame. Frame 2: the
// frame is submitted first, waiting for a fence value the compute submission after it signals (the layer must hold
// it). Both frames, the state words and the instance records are compared with the CPU's simulation of the same steps;
// between the two frames the points must move exactly where the reference says. Control: frame 3 without the Wait,
// submitted before its compute work; it is reported (old state, new state or a mix), not required, since a layer may
// order the queues on its own.
//
// On success the guest prints one line starting "Lanterns stage N: PASS" and exits 0; run_lanterns.py checks that
// line, the exit code and the device line.
#include "lanterns_edges_gpu.h"
#include "lanterns_gpu.h"
#include "lanterns_particles.h"
#include "lanterns_reference.h"
#include "lanterns_scene.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace {

using lanterns::Color;
using lanterns::kHeight;
using lanterns::kWidth;

bool write_file(const std::filesystem::path& path, const void* data, std::size_t size) {
    std::ofstream out(path, std::ios::binary);
    out.write(static_cast<const char*>(data), std::streamsize(size));
    return bool(out);
}

// Writes <out>/<name>.gpu.rgba32f, .ref.rgba32f and .uncertain.u8 (one byte per pixel, 1 = uncertain), which
// run_lanterns.py turns into PNGs.
bool dump(const std::filesystem::path& out, const char* name, const std::vector<Color>& gpu,
          const lanterns::Reference& ref) {
    std::vector<std::uint8_t> uncertain(gpu.size());
    for (std::size_t i = 0; i < gpu.size(); ++i) uncertain[i] = ref.candidates[i].empty() ? 0 : 1;
    const std::string base = (out / name).string();
    const bool ok = write_file(base + ".gpu.rgba32f", gpu.data(), gpu.size() * sizeof(Color)) &&
                    write_file(base + ".ref.rgba32f", ref.color.data(), ref.color.size() * sizeof(Color)) &&
                    write_file(base + ".uncertain.u8", uncertain.data(), uncertain.size());
    if (!ok) std::fprintf(stderr, "lanterns: cannot write %s.*\n", base.c_str());
    return ok;
}

void report(const char* what, const lanterns::Comparison& c) {
    std::printf("lanterns: %s: %u of %u certain pixels exact, %u of %u uncertain pixels one of their candidates\n",
                what, c.certain_exact, c.certain, c.uncertain_plausible, c.uncertain);
    if (!c.pass())
        std::printf("lanterns: %s: first bad pixel (%u, %u): gpu (%g, %g, %g, %g), reference (%g, %g, %g, %g)\n", what,
                    c.first_bad_x, c.first_bad_y, c.first_bad_gpu[0], c.first_bad_gpu[1], c.first_bad_gpu[2],
                    c.first_bad_gpu[3], c.first_bad_ref[0], c.first_bad_ref[1], c.first_bad_ref[2], c.first_bad_ref[3]);
}

// Pixels certain in both references whose expected colours differ, and how many of those differ on the GPU too.
struct Change {
    unsigned expected = 0, changed = 0;
};
Change changes(const std::vector<Color>& a, const lanterns::Reference& ra, const std::vector<Color>& b,
               const lanterns::Reference& rb) {
    Change c;
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (!ra.candidates[i].empty() || !rb.candidates[i].empty()) continue;
        if (lanterns::detail::same(ra.color[i], rb.color[i])) continue;
        ++c.expected;
        c.changed += !lanterns::detail::same(a[i], b[i]);
    }
    return c;
}

int stage1(const std::filesystem::path& shader_dir, const std::filesystem::path& out_dir) {
    const lanterns::Scene scene = lanterns::stage1();
    lanterns::Shaders shaders;
    lanterns::EdgeShaders edge_shaders;
    if (!lanterns::load_shaders(shader_dir, 1, shaders) || !lanterns::load_edge_shaders(shader_dir, edge_shaders)) return 1;
    std::vector<Color> with_depth, without_depth, edges;
    {
        lanterns::Gpu gpu;
        if (!gpu.create(scene, shaders)) return 1;
        if (!gpu.submit_frame(scene, true) || !gpu.finish_frame(with_depth)) return 1;
        if (!gpu.submit_frame(scene, false) || !gpu.finish_frame(without_depth)) return 1;
        lanterns::EdgeDraw edge_draw;
        if (!edge_draw.create(gpu, edge_shaders) || !edge_draw.run(edges)) return 1;
        if (!gpu.idle()) return 1;
    }
    // The shared edges: every pixel exact (red = which triangles drew it, green and alpha = how many times).
    const lanterns::EdgeReference edge_ref = lanterns::edge_reference();
    unsigned edge_exact = 0, edge_on_ok = 0, edge_twice = 0, edge_gap = 0;
    for (std::size_t i = 0; i < edges.size(); ++i) {
        const bool ok = lanterns::detail::same(edges[i], edge_ref.color[i]);
        edge_exact += ok;
        edge_twice += edges[i][1] > 1.0f;
        edge_gap += edge_ref.color[i][1] == 1.0f && edges[i][1] == 0.0f;
        if (!ok && edges.size() - edge_exact <= 3)
            std::printf("lanterns: shared edges, pixel (%zu, %zu): gpu (%g %g %g %g) want (%g %g %g %g)\n",
                        i % lanterns::kWidth, i / lanterns::kWidth, edges[i][0], edges[i][1], edges[i][2], edges[i][3],
                        edge_ref.color[i][0], edge_ref.color[i][1], edge_ref.color[i][2], edge_ref.color[i][3]);
    }
    for (unsigned y = 0; y < lanterns::kHeight; ++y)
        for (unsigned x = 0; x < lanterns::kWidth; ++x) {
            const long long px = 2 * x + 1, py = 2 * y + 1;
            bool on = false;
            for (unsigned k = 0; k < lanterns::kEdgeTriangles && !on; ++k) {
                const long long dx = lanterns::kEdgeRing[k][0] - lanterns::kEdgeCentre[0],
                                dy = lanterns::kEdgeRing[k][1] - lanterns::kEdgeCentre[1];
                const long long qx = px - lanterns::kEdgeCentre[0], qy = py - lanterns::kEdgeCentre[1];
                const long long t = qx * dx + qy * dy;
                on = qx * dy - qy * dx == 0 && t >= 0 && t < dx * dx + dy * dy;
            }
            const std::size_t i = std::size_t(y) * lanterns::kWidth + x;
            edge_on_ok += on && lanterns::detail::same(edges[i], edge_ref.color[i]);
        }
    {   // The picture: each triangle's own colour from the bits the GPU (and the CPU) summed; white twice, black none.
        const auto shown_of = [](const Color& c) {
            static constexpr float palette[8][3] = {{0.9f, 0.3f, 0.2f}, {0.9f, 0.7f, 0.2f}, {0.5f, 0.8f, 0.3f},
                                                    {0.2f, 0.7f, 0.6f}, {0.2f, 0.5f, 0.9f}, {0.5f, 0.3f, 0.9f},
                                                    {0.8f, 0.3f, 0.7f}, {0.6f, 0.6f, 0.6f}};
            if (c[1] == 0.0f) return Color{0, 0, 0, 1};
            if (c[1] != 1.0f) return Color{1, 1, 1, 1};
            const unsigned k = unsigned(std::log2(std::max(1.0f, c[0]))) & 7u;
            return Color{palette[k][0], palette[k][1], palette[k][2], 1};
        };
        std::vector<Color> shown_gpu(edges.size());
        lanterns::Reference shown_ref;
        shown_ref.color.resize(edges.size());
        shown_ref.candidates.resize(edges.size());
        for (std::size_t i = 0; i < edges.size(); ++i)
            shown_gpu[i] = shown_of(edges[i]), shown_ref.color[i] = shown_of(edge_ref.color[i]);
        if (!dump(out_dir, "stage1-edges", shown_gpu, shown_ref)) return 1;
    }
    std::printf("lanterns: shared edges (wave64 VS and PS, root table at s8): %u of %u pixels exact; %u of %u pixel "
                "centres exactly on an inner edge drawn once by the triangle the top-left rule names; %u pixels drawn "
                "twice, %u gaps\n", edge_exact, unsigned(edges.size()), edge_on_ok, edge_ref.on_shared, edge_twice,
                edge_gap);
    const lanterns::Reference ref_on = lanterns::reference_frame(scene, true);
    const lanterns::Reference ref_off = lanterns::reference_frame(scene, false);
    const auto on = lanterns::compare(with_depth, ref_on), off = lanterns::compare(without_depth, ref_off);
    report("depth test on", on);
    report("depth test off (negative control)", off);
    const Change control = changes(with_depth, ref_on, without_depth, ref_off);
    std::printf("lanterns: the depth test changes %u certain pixels (the reference expects %u); %u faces edge-on\n",
                control.changed, control.expected, ref_on.degenerate_faces);
    std::printf("lanterns: %u pixels lie on an edge two faces of one solid share: one of the two must show there, not "
                "what is behind them\n", ref_on.shared_edge_pixels);
    if (!dump(out_dir, "stage1-depth", with_depth, ref_on) || !dump(out_dir, "stage1-nodepth", without_depth, ref_off))
        return 1;
    if (!on.pass() || !off.pass() || control.expected < 1000 || control.changed != control.expected ||
        edge_exact != edges.size() || edge_on_ok != edge_ref.on_shared || edge_ref.on_shared < 500 ||
        edge_ref.covered < 50000) {
        std::fprintf(stderr, "lanterns: stage 1 FAILED\n");
        return 1;
    }
    // Printed only here, after every check: run_lanterns.py wants this line once and exit code 0.
    std::printf("Lanterns stage 1: PASS, %u of %u pixels exact, %u edge pixels plausible, depth-off control changed "
                "%u pixels; shared edges through a wave64 VS/PS with the title's s8 table: %u of %u pixels exact, %u "
                "centres on shared edges each drawn once\n", on.certain_exact, kWidth * kHeight, on.uncertain_plausible,
                control.changed, edge_exact, unsigned(edges.size()), edge_on_ok);
    return 0;
}

// The GPU's state words and instance records against the CPU's simulation; prints the first difference.
bool same_particles(const char* what, const std::vector<lanterns::Particle>& state,
                    const std::vector<lanterns::Instance>& instances, const std::vector<lanterns::Particle>& want) {
    const auto want_instances = lanterns::particle_instances(want);
    unsigned bad_state = 0, bad_instances = 0;
    for (unsigned i = 0; i < lanterns::kParticles; ++i) {
        bad_state += std::memcmp(&state[i], &want[i], sizeof(lanterns::Particle)) != 0;
        bad_instances += std::memcmp(&instances[i], &want_instances[i], sizeof(lanterns::Instance)) != 0;
    }
    std::printf("lanterns: %s: %u of %u point states and %u of %u instance records equal the CPU's\n", what,
                lanterns::kParticles - bad_state, lanterns::kParticles, lanterns::kParticles - bad_instances,
                lanterns::kParticles);
    return !bad_state && !bad_instances;
}

int stage3(const std::filesystem::path& shader_dir, const std::filesystem::path& out_dir) {
    using lanterns::kStepsPerFrame;
    std::vector<lanterns::Particle> cpu = lanterns::initial_particles();
    std::vector<std::vector<lanterns::Particle>> cpu_after;   // after 1, 2 and 3 frames' worth of steps
    for (int f = 0; f < 3; ++f) {
        for (unsigned k = 0; k < kStepsPerFrame; ++k) lanterns::step_particles(cpu);
        cpu_after.push_back(cpu);
    }
    const lanterns::Scene scene1 = lanterns::stage3(cpu_after[0]), scene2 = lanterns::stage3(cpu_after[1]),
                          scene3 = lanterns::stage3(cpu_after[2]);
    lanterns::Shaders shaders;
    if (!lanterns::load_shaders(shader_dir, 3, shaders)) return 1;
    std::vector<Color> frame1, frame2, frame3;
    lanterns::Readback read1, read2;
    {
        lanterns::Gpu gpu;
        if (!gpu.create(scene1, shaders)) return 1;
        std::uint64_t signalled = 0;
        // Frame 1: the compute work first, then the frame that waits for it.
        if (!gpu.submit_compute(kStepsPerFrame, signalled) || !gpu.submit_frame(scene1, true, signalled) ||
            !gpu.finish_frame(frame1, &read1))
            return 1;
        // Frame 2: the frame first, waiting for the value the compute submission after it will signal.
        const std::uint64_t next = gpu.next_compute_value();
        if (!gpu.submit_frame(scene2, true, next) || !gpu.submit_compute(kStepsPerFrame, signalled) || signalled != next ||
            !gpu.finish_frame(frame2, &read2))
            return 1;
        // Control: the same order without the Wait.
        if (!gpu.submit_frame(scene3, true, 0) || !gpu.submit_compute(kStepsPerFrame, signalled) ||
            !gpu.finish_frame(frame3) || !gpu.idle())
            return 1;
    }
    const bool particles1 =
        same_particles("frame 1 (compute, then the frame)", read1.state, read1.instances, cpu_after[0]);
    const bool particles2 = same_particles("frame 2 (the frame waits for compute submitted after it)", read2.state,
                                           read2.instances, cpu_after[1]);
    const bool particles_ok = particles1 && particles2;
    const lanterns::Reference ref1 = lanterns::reference_frame(scene1, true);
    const lanterns::Reference ref2 = lanterns::reference_frame(scene2, true);
    const lanterns::Reference ref3 = lanterns::reference_frame(scene3, true);
    const auto c1 = lanterns::compare(frame1, ref1), c2 = lanterns::compare(frame2, ref2);
    report("frame 1", c1);
    report("frame 2", c2);
    const Change moved = changes(frame1, ref1, frame2, ref2);
    std::printf("lanterns: between the frames the points changed %u certain pixels (the reference expects %u)\n",
                moved.changed, moved.expected);
    // The control, reported only: did the frame without the Wait show the new points, the old ones, or a mix?
    const auto as_new = lanterns::compare(frame3, ref3), as_old = lanterns::compare(frame3, ref2);
    const char* verdict = as_new.pass() ? "the new state: the layer ordered the queues without the Wait"
                          : as_old.pass() ? "the previous frame's state: the draw ran before the compute work"
                                          : "neither state exactly: the draw and the compute work overlapped";
    std::printf("lanterns: control without the Wait showed %s (%u certain pixels differ from the new state)\n", verdict,
                as_new.certain - as_new.certain_exact);
    if (!dump(out_dir, "stage3-frame1", frame1, ref1) || !dump(out_dir, "stage3-frame2", frame2, ref2) ||
        !dump(out_dir, "stage3-control", frame3, ref3))
        return 1;
    if (!particles_ok || !c1.pass() || !c2.pass() || moved.expected < 50 || moved.changed != moved.expected) {
        std::fprintf(stderr, "lanterns: stage 3 FAILED\n");
        return 1;
    }
    std::printf("Lanterns stage 3: PASS, %u and %u of %u pixels exact, %u points exact after %u and %u compute steps, "
                "points moved %u pixels\n", c1.certain_exact, c2.certain_exact, kWidth * kHeight, lanterns::kParticles,
                kStepsPerFrame, 2 * kStepsPerFrame, moved.changed);
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);  // the marker reaches the launcher even if the process dies after it
    int stage = 0;
    std::filesystem::path shader_dir, out_dir;
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string key = argv[i];
        if (key == "--stage") stage = std::atoi(argv[i + 1]);
        else if (key == "--shaders") shader_dir = argv[i + 1];
        else if (key == "--out") out_dir = argv[i + 1];
    }
    if (shader_dir.empty() || out_dir.empty()) stage = 0;
    std::error_code ec;
    if (stage) std::filesystem::create_directories(out_dir, ec);
    switch (stage) {
    case 1: return stage1(shader_dir, out_dir);
    case 3: return stage3(shader_dir, out_dir);
    default:
        std::fprintf(stderr, "usage: lanterns --stage 1|3 --shaders <folder fssm_pack.py wrote> --out <folder>\n");
        return 2;
    }
}
