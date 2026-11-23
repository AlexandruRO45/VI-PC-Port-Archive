// Lanterns: an original stand-in title that runs under gta6rt --run and drives the whole chain a real title would:
// an x64 PE the loader maps, imports the loader's shims answer, the Xbox D3D12 device on Vulkan
// (D3D12XboxCreateDevice from d3d12_xs.dll, LEONIDA_D3D12=vulkan), pipelines from shader containers holding RDNA2
// programs (DXBC { DXIL { XDAT { XBCH + zlib(FSSM) } } }, packed by fssm_pack.py beside this file, translated offline by
// fssm_unpack.py, shader_disasm.py and rdna2spirv.py; the layer finds each by the SHA-256 of its FSSM program), draws,
// compute, and readbacks checked against a CPU reference. Content: abstract solids on a checkered floor and rising
// points of light, colours from short binary fractions; no game data. How to build and run it: lanterns.cmake.
//
//   lanterns.exe --stage 1|3|5|6|8|9|10|11 --shaders <folder fssm_pack.py wrote> --out <folder for the frame dumps>
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
// Stage 5: the stage-3 frame after two rounds of rise steps (the points drawn directly, their instances written by
// the compute shader) into an R16G16B16A16_FLOAT target, then a
// post-process on the same direct queue (lanterns_post.h): two compute passes (wave32 rows, wave64 columns), each 128
// lanes in several waves sharing LDS across an s_barrier, blur what is brighter than a threshold along rows (reading the scene through a T#
// with image_load) and then columns; a fullscreen pixel shader reads SV_Position, the scene texel and the blur, and
// writes the sum into an R32G32B32A32_FLOAT target. The fp16 scene and the final frame are compared with the CPU
// (exact where certain, within bounds where the scene's edges leave it uncertain). Control: the composite with the
// blur skipped must change exactly the certain pixels the reference says the bloom changes. The row pass without its
// s_barrier is run too and its difference reported (a race: not predicted).
//
// Stage 6: the stage-1 scene textured. A BC1 mosaic (256x256, 9 mips) on the floor, sampled with image_sample
// (trilinear, LOD from derivatives); a BC7 pattern (64x64, 7 mips) on the solids, sampled with image_sample_l at a
// per-draw mip level from a root constant. The textures are made, mipmapped and compressed here (lanterns_bc.h) and
// uploaded with CopyTextureRegion per level; the reference decodes them and bounds every pixel (lanterns_sampling.h).
// Control: the solids two mip levels down must leave the main bounds wherever the two references cannot overlap.
//
// Stage 8: shadows. A depth-only pass (no render target, depth bias from the PSO) draws the solids from a directional
// light into a 512x512 D32 shadow map; the main pass's pixel shader takes four comparison samples (LESS_EQUAL, 2x2 PCF)
// and scales the face colour by the lit fraction (lanterns_shadow.h). Control: the map left cleared to 1 must change
// exactly the pixels the reference says are shadowed.
//
// Stage 9: deferred lighting. A G-buffer pass writes three render targets (RGBA8 albedo, RGB10A2 normal, R32F view
// depth); a compute pass, one 16x16 workgroup per tile, builds each tile's light list in LDS with atomics and shades
// with 256 point lights in integers (lanterns_deferred.h). Controls: culling off must give the same image bit for bit;
// one light moved must change exactly the pixels the reference predicts.
//
// Stage 10: 4x MSAA. The stage-1 scene into a 4-sample target and depth buffer, ResolveSubresource to 1x; the CPU
// rasterizes each standard sample position and averages (lanterns_msaa.h). Control: the 1x render must differ from the
// resolve exactly on the edge pixels the reference predicts.
//
// Stage 11: ray-traced shadows. A BLAS (one prism) and a TLAS (ten instances) built through the D3D12 API; a compute
// shader traces one shadow ray per pixel of a stage-9 G-buffer with image_bvh_intersect_ray, its own stack in LDS
// (lanterns_rt.h). The CPU traces the same rays. Control: a second TLAS with one instance's mask 0 must change exactly
// the pixels the reference predicts.
//
// On success the guest prints one line starting "Lanterns stage N: PASS" and exits 0; run_lanterns.py checks that
// line, the exit code and the device line.
#include "lanterns_edges_gpu.h"
#include "lanterns_gpu.h"
#include "lanterns_particles.h"
#include "lanterns_post.h"
#include "lanterns_post_gpu.h"
#include "lanterns_reference.h"
#include "lanterns_sampling.h"
#include "lanterns_scene.h"
#include "lanterns_shadow.h"
#include "lanterns_deferred_gpu.h"
#include "lanterns_msaa.h"
#include "lanterns_rt_gpu.h"
#include "lanterns_textures_gpu.h"

#include <algorithm>
#include <chrono>
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

// Writes <out>/<name>.gpu.rgba32f, .ref.rgba32f and .uncertain.u8 (one byte per pixel: 0 = compare the two exactly,
// 1 = uncertain, 2 = out of bounds, 3 = within bounds), which run_lanterns.py turns into PNGs. states, when given,
// replaces the bytes the reference's candidates make (0 or 1).
bool dump(const std::filesystem::path& out, const char* name, const std::vector<Color>& gpu,
          const lanterns::Reference& ref, const std::vector<std::uint8_t>* states = nullptr) {
    std::vector<std::uint8_t> uncertain(gpu.size());
    for (std::size_t i = 0; i < gpu.size(); ++i) uncertain[i] = ref.candidates[i].empty() ? 0 : 1;
    if (states) uncertain = *states;
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

// The post reference as a Reference for dump(): the exact colour (or the lower bound) and "uncertain" where bounded.
lanterns::Reference as_reference(const lanterns::PostReference& post) {
    lanterns::Reference ref;
    ref.color = post.lo;
    ref.candidates.resize(post.lo.size());
    for (std::size_t i = 0; i < post.lo.size(); ++i)
        if (!post.certain[i]) ref.candidates[i] = {post.lo[i], post.hi[i]};
    return ref;
}

// Stage 5: the stage-3 frame into an fp16 target, then the bloom post-process (see the top of this file).
int stage5(const std::filesystem::path& shader_dir, const std::filesystem::path& out_dir) {
    using lanterns::kStepsPerFrame;
    std::vector<lanterns::Particle> cpu = lanterns::initial_particles();
    for (unsigned k = 0; k < 2 * kStepsPerFrame; ++k) lanterns::step_particles(cpu);
    const lanterns::Scene scene = lanterns::stage3(cpu);
    lanterns::Shaders shaders;
    lanterns::PostShaders post_shaders;
    if (!lanterns::load_shaders(shader_dir, 5, shaders) || !lanterns::load_post_shaders(shader_dir, post_shaders)) return 1;
    std::vector<Color> scene_pixels, final_pixels, skipped, no_barrier;
    lanterns::Readback read;
    {
        lanterns::Gpu gpu;
        if (!gpu.create(scene, shaders, DXGI_FORMAT_R16G16B16A16_FLOAT)) return 1;
        lanterns::Post post;   // after gpu: released first
        if (!post.create(gpu, post_shaders)) return 1;
        std::uint64_t first = 0, second = 0;
        if (!gpu.submit_compute(kStepsPerFrame, first) || !gpu.submit_compute(kStepsPerFrame, second) ||
            !gpu.submit_frame(scene, true, second) || !gpu.finish_frame(scene_pixels, &read))
            return 1;
        if (!post.run(lanterns::PostMode::full, final_pixels) || !post.run(lanterns::PostMode::skip_blur, skipped) ||
            !post.run(lanterns::PostMode::no_barrier, no_barrier) || !gpu.idle())
            return 1;
    }
    const bool particles_ok = same_particles("the points after two rounds of rise steps", read.state, read.instances, cpu);
    const lanterns::Reference ref = lanterns::reference_frame(scene, true);
    const lanterns::Reference half = lanterns::half_reference(ref);
    const lanterns::Comparison scene_check = lanterns::compare(scene_pixels, half);
    report("the fp16 scene target", scene_check);
    const lanterns::PostReference post = lanterns::post_reference(ref);
    const lanterns::Comparison final_check = lanterns::compare_post(final_pixels, post);
    std::printf("lanterns: the final frame: %u of %u certain pixels exact, %u of %u bounded pixels within their bounds "
                "(the reference's float and double sums differ at %u values)\n", final_check.certain_exact,
                final_check.certain, final_check.uncertain_plausible, final_check.uncertain, post.inexact);
    if (!final_check.pass())
        report("the final frame", final_check);
    const unsigned control = lanterns::certain_differences(skipped, post);
    const unsigned misplaced = lanterns::misplaced_differences(skipped, post);
    const unsigned race = lanterns::certain_differences(no_barrier, post);
    std::printf("lanterns: control, the blur skipped: %u certain pixels differ (the reference predicts %u), %u of them "
                "or of the predicted ones in the wrong place\n", control, post.bloom_pixels, misplaced);
    std::printf("lanterns: measured, the row pass without s_barrier: %u certain pixels differ\n", race);
    if (!dump(out_dir, "stage5-scene", scene_pixels, half) ||
        !dump(out_dir, "stage5-final", final_pixels, as_reference(post)) ||
        !dump(out_dir, "stage5-skipped", skipped, as_reference(post)) ||
        !dump(out_dir, "stage5-nobarrier", no_barrier, as_reference(post)))
        return 1;
    if (!particles_ok || !scene_check.pass() || !final_check.pass() || post.inexact ||
        post.bloom_pixels < 1000 || control != post.bloom_pixels || misplaced != 0) {
        std::fprintf(stderr, "lanterns: stage 5 FAILED\n");
        return 1;
    }
    std::printf("Lanterns stage 5: PASS, fp16 scene %u of %u certain pixels exact, bloom composite %u of %u certain "
                "pixels exact and %u bounded pixels within bounds, skipping the blur changed %u pixels as predicted\n",
                scene_check.certain_exact, scene_check.certain, final_check.certain_exact, final_check.certain,
                final_check.uncertain, control);
    return 0;
}

// Stage 6: the stage-1 scene with compressed, mipmapped textures (see the top of this file).
int stage6(const std::filesystem::path& shader_dir, const std::filesystem::path& out_dir) {
    lanterns::Shaders shaders;
    if (!lanterns::load_shaders(shader_dir, 6, shaders)) return 1;
    const lanterns::BcTexture mosaic = lanterns::make_texture(lanterns::mosaic_pattern(), lanterns::BcFormat::bc1);
    const lanterns::BcTexture squares = lanterns::make_texture(lanterns::squares_pattern(), lanterns::BcFormat::bc7);
    const lanterns::Scene scene = lanterns::stage6();
    lanterns::Scene control = scene;   // every solid two mip levels further down
    for (unsigned d = 0; d < 4; ++d) control.draws[d].tint[3] += 2.0f;
    std::vector<Color> frame, control_frame;
    {
        lanterns::Gpu gpu;
        if (!gpu.create(scene, shaders)) return 1;
        lanterns::Ref<ID3D12Resource> floor_tex, solid_tex;   // after gpu: released first
        if (!lanterns::upload_texture(gpu.device(), gpu.queue(), mosaic, floor_tex) ||
            !lanterns::upload_texture(gpu.device(), gpu.queue(), squares, solid_tex) ||
            !gpu.bind_textures(floor_tex.p, solid_tex.p))
            return 1;
        std::printf("lanterns: textures: BC1 %ux%u with %zu mips, BC7 %ux%u with %zu mips, uploaded by "
                    "CopyTextureRegion per level\n", mosaic.width, mosaic.height, mosaic.blocks.size(), squares.width,
                    squares.height, squares.blocks.size());
        if (!gpu.submit_frame(scene, true, 0) || !gpu.finish_frame(frame) ||
            !gpu.submit_frame(control, true, 0) || !gpu.finish_frame(control_frame) || !gpu.idle())
            return 1;
    }
    const lanterns::TexturedReference ref = lanterns::textured_reference(scene, {&mosaic, &squares, 0.0f});
    const lanterns::TexturedReference ctl = lanterns::textured_reference(scene, {&mosaic, &squares, 2.0f});
    const lanterns::Comparison c = lanterns::compare_textured(frame, ref);
    std::printf("lanterns: textured frame: %u of %u certain pixels within their bounds, %u of %u uncertain pixels "
                "within their candidates' bounds\n", c.certain_exact, c.certain, c.uncertain_plausible, c.uncertain);
    if (!c.pass()) report("textured frame", c);
    // The control: wherever the references for LOD and LOD + 2 cannot overlap, the control frame must leave the
    // main frame's bounds, and it must stay inside its own.
    const auto predicted = lanterns::disjoint(ref, ctl);
    unsigned expected = 0, moved = 0;
    for (std::size_t i = 0; i < predicted.size(); ++i) {
        expected += predicted[i];
        moved += predicted[i] && !lanterns::within(control_frame[i], ref.bounds[i]);
    }
    const lanterns::Comparison own = lanterns::compare_textured(control_frame, ctl);
    std::printf("lanterns: control, the solids two mips down: %u of %u predicted pixels left the bounds; the control "
                "frame is within its own bounds at %u of %u certain pixels\n", moved, expected, own.certain_exact,
                own.certain);
    // The dumps: the reference panel is the middle of each pixel's bounds; the mask is grey within bounds, yellow
    // where coverage is uncertain, red outside the bounds (for the control: where it left them, as predicted).
    lanterns::Reference shown;
    shown.color.resize(ref.bounds.size());
    shown.candidates.resize(ref.bounds.size());
    std::vector<std::uint8_t> states(ref.bounds.size()), control_states(ref.bounds.size());
    for (std::size_t i = 0; i < states.size(); ++i) {
        for (int k = 0; k < 4; ++k) shown.color[i][k] = (ref.bounds[i].lo[k] + ref.bounds[i].hi[k]) / 2;
        states[i] = !ref.certain[i] ? 1 : lanterns::within(frame[i], ref.bounds[i]) ? 3 : 2;
        control_states[i] = !ref.certain[i] ? 1 : lanterns::within(control_frame[i], ref.bounds[i]) ? 3 : 2;
    }
    if (!dump(out_dir, "stage6-frame", frame, shown, &states) ||
        !dump(out_dir, "stage6-control", control_frame, shown, &control_states))
        return 1;
    if (!c.pass() || !own.pass() || expected < 1000 || moved != expected) {
        std::fprintf(stderr, "lanterns: stage 6 FAILED\n");
        return 1;
    }
    std::printf("Lanterns stage 6: PASS, %u of %u certain pixels within the BC1/BC7 sampling bounds (trilinear floor, "
                "image_sample_l solids), control two mips down moved %u of %u predicted pixels\n", c.certain_exact,
                c.certain, moved, expected);
    return 0;
}

// Stage 8: a shadow map from a directional light, sampled with 2x2 PCF (see the top of this file).
int stage8(const std::filesystem::path& shader_dir, const std::filesystem::path& out_dir) {
    lanterns::Shaders shaders;
    if (!lanterns::load_shaders(shader_dir, 8, shaders)) return 1;
    const lanterns::Scene scene = lanterns::stage8();
    const lanterns::Light light = lanterns::make_light();
    std::vector<Color> frame, control;
    {
        lanterns::Gpu gpu;
        if (!gpu.create(scene, shaders) || !gpu.set_light(light.clip, light.rows)) return 1;
        if (!gpu.submit_frame(scene, true, 0) || !gpu.finish_frame(frame)) return 1;
        gpu.set_shadow_casters(false);
        if (!gpu.submit_frame(scene, true, 0) || !gpu.finish_frame(control) || !gpu.idle()) return 1;
    }
    const auto ref = lanterns::shadow_reference(scene, light, lanterns::shadow_map(scene, light, true));
    const auto ref_lit = lanterns::shadow_reference(scene, light, lanterns::shadow_map(scene, light, false));
    const lanterns::Comparison c = lanterns::compare_textured(frame, ref), cl = lanterns::compare_textured(control, ref_lit);
    // Exact pixels: certain, with one possible colour (every tap certain), and equal to it.
    unsigned exact = 0, single = 0, expected = 0, changed = 0;
    for (std::size_t i = 0; i < frame.size(); ++i) {
        const bool one = ref.certain[i] && lanterns::detail::same(ref.bounds[i].lo, ref.bounds[i].hi);
        const bool one_lit = ref_lit.certain[i] && lanterns::detail::same(ref_lit.bounds[i].lo, ref_lit.bounds[i].hi);
        single += one;
        exact += one && lanterns::detail::same(frame[i], ref.bounds[i].lo);
        if (one && one_lit && !lanterns::detail::same(ref.bounds[i].lo, ref_lit.bounds[i].lo)) {
            ++expected;
            changed += !lanterns::detail::same(frame[i], control[i]);
        }
    }
    std::printf("lanterns: shadowed frame: %u of %u certain pixels within their bounds, %u of %u single-colour pixels "
                "exact, %u of %u uncertain within bounds\n", c.certain_exact, c.certain, exact, single,
                c.uncertain_plausible, c.uncertain);
    if (!c.pass()) report("shadowed frame", c);
    std::printf("lanterns: control, the map left cleared to 1: %u of %u certain pixels within the all-lit bounds; %u "
                "pixels changed (the reference predicts %u)\n", cl.certain_exact, cl.certain, changed, expected);
    // Each dump with its own reference: the middle of each pixel's bounds, and per pixel 3 (one colour, matched),
    // 1 (bounded, within) or 2 (outside every box).
    const auto bounds_dump = [&](const char* name, const std::vector<Color>& gpu, const lanterns::TexturedReference& r) {
        lanterns::Reference shown;
        shown.color.resize(gpu.size());
        shown.candidates.resize(gpu.size());
        std::vector<std::uint8_t> states(gpu.size());
        for (std::size_t i = 0; i < gpu.size(); ++i) {
            for (int k = 0; k < 4; ++k) shown.color[i][k] = (r.bounds[i].lo[k] + r.bounds[i].hi[k]) / 2;
            const bool one = r.certain[i] && lanterns::detail::same(r.bounds[i].lo, r.bounds[i].hi);
            states[i] = !lanterns::within_any(r, i, gpu[i]) ? 2 : one ? 3 : 1;
        }
        return dump(out_dir, name, gpu, shown, &states);
    };
    if (!bounds_dump("stage8-frame", frame, ref) || !bounds_dump("stage8-control", control, ref_lit)) return 1;
    if (!c.pass() || !cl.pass() || exact != single || expected < 1000 || changed != expected) {
        std::fprintf(stderr, "lanterns: stage 8 FAILED\n");
        return 1;
    }
    std::printf("Lanterns stage 8: PASS, %u of %u single-colour pixels exact and %u of %u certain pixels within bounds "
                "(D32 shadow map, PSO depth bias, 2x2 PCF with comparison samples), the map cleared to far changed %u "
                "pixels as predicted\n", exact, single, c.certain_exact, c.certain, changed);
    return 0;
}

// Stage 9: deferred lighting with tiled light culling (see the top of this file).
int stage9(const std::filesystem::path& shader_dir, const std::filesystem::path& out_dir) {
    lanterns::Shaders shaders;
    std::vector<std::uint8_t> tiled;
    if (!lanterns::load_shaders(shader_dir, 9, shaders) || !lanterns::load_tiled_shader(shader_dir, tiled)) return 1;
    const lanterns::Scene scene = lanterns::stage9();
    const std::vector<lanterns::PointLight> lights = lanterns::make_lights();
    // The control light: the one nearest (0, 0.5) on the floor, moved 0.875 to the right.
    std::size_t moved_index = 0;
    for (std::size_t i = 1; i < lights.size(); ++i)
        if (std::hypot(lights[i].pos[0], lights[i].pos[2] - 0.5f) < std::hypot(lights[moved_index].pos[0], lights[moved_index].pos[2] - 0.5f))
            moved_index = i;
    std::vector<lanterns::PointLight> moved = lights;
    moved[moved_index].pos[0] += 0.875f;
    std::vector<Color> albedo;
    std::vector<std::uint32_t> culled, all, shifted, counts, counts_all, counts_shifted, lists, lists_all, lists_shifted;
    std::vector<std::uint32_t> normal_words;
    std::vector<float> depth;
    {
        lanterns::Gpu gpu;
        lanterns::Deferred deferred;
        if (!gpu.create(scene, shaders, DXGI_FORMAT_R8G8B8A8_UNORM) || !gpu.submit_frame(scene, true, 0) ||
            !gpu.finish_frame(albedo) || !deferred.create(gpu, tiled) || !deferred.read_gbuffer(normal_words, depth))
            return 1;
        if (!deferred.run(lights, lanterns::deferred_constants(scene, 0), culled, counts, lists) ||
            !deferred.run(lights, lanterns::deferred_constants(scene, 1), all, counts_all, lists_all) ||
            !deferred.run(moved, lanterns::deferred_constants(scene, 0), shifted, counts_shifted, lists_shifted) ||
            !gpu.idle())
            return 1;
    }
    // Poison: every pixel word and tile count of every run, and every list entry below its tile's count, was written.
    unsigned poisoned = 0;
    for (const auto* v : {&culled, &all, &shifted, &counts, &counts_all, &counts_shifted})
        for (std::uint32_t w : *v) poisoned += w == lanterns::kDeferredPoison;
    // The exact light sets: each tile's list holds every light the CPU says it must keep, none it must not, none twice;
    // with culling off, all 256.
    const auto check_lists = [&](const std::vector<std::uint32_t>& cnt, const std::vector<std::uint32_t>& lst,
                                 const std::vector<std::uint8_t>* states, unsigned& tiles_ok, unsigned& either) {
        tiles_ok = either = 0;
        for (std::size_t t = 0; t < cnt.size(); ++t) {
            std::vector<std::uint8_t> seen(lanterns::kLights, 0);
            bool ok = cnt[t] <= lanterns::kLights;
            for (std::uint32_t j = 0; ok && j < cnt[t]; ++j) {
                const std::uint32_t id = lst[t * lanterns::kLights + j];
                ok = id < lanterns::kLights && !seen[id];
                if (ok) seen[id] = 1;
            }
            for (unsigned l = 0; ok && l < lanterns::kLights; ++l) {
                const std::uint8_t want = states ? (*states)[t * lanterns::kLights + l] : 1;
                either += want == 2;
                if (want != 2 && seen[l] != want) ok = false;
            }
            tiles_ok += ok;
        }
    };
    const auto light_states = lanterns::tile_light_states(depth, lanterns::deferred_constants(scene, 0), lights);
    const auto states_moved = lanterns::tile_light_states(depth, lanterns::deferred_constants(scene, 0), moved);
    unsigned sets_ok = 0, sets_either = 0, sets_all_ok = 0, sets_moved_ok = 0, unused = 0;
    check_lists(counts, lists, &light_states, sets_ok, sets_either);
    check_lists(counts_all, lists_all, nullptr, sets_all_ok, unused);
    check_lists(counts_shifted, lists_shifted, &states_moved, sets_moved_ok, unused);
    // G-buffer probes: every certain pixel whose depth bounds are tight (no other surface may be in front), its three
    // normal codes and its view depth within the bounds.
    const auto gref = lanterns::gbuffer_reference(scene);
    unsigned probes = 0, probes_ok = 0, beyond_fixed = 0;
    double worst = 0;
    for (std::size_t i = 0; i < depth.size(); ++i) {
        const auto& b = gref.bounds[i];
        if (!gref.certain[i] || !(b.lo[3] > 0) || b.hi[3] - b.lo[3] > 0.01 * b.lo[3]) continue;
        ++probes;
        bool ok = depth[i] >= b.lo[3] && depth[i] <= b.hi[3];
        // The old fixed slack: a depth error above kPosSlack moves the rebuilt point by more than kPosSlack (the view
        // ray's clip-w-1 direction is at least 1 long).
        const double mid = (double(b.lo[3]) + b.hi[3]) / 2, half = (double(b.hi[3]) - b.lo[3]) / 2;
        beyond_fixed += std::fabs(depth[i] - mid) > lanterns::kPosSlack;
        worst = std::max(worst, std::fabs(depth[i] - mid) / half);
        for (int k = 0; k < 3; ++k) {
            const unsigned code = (normal_words[i] >> (10 * k)) & 1023u;
            ok = ok && code >= unsigned(b.lo[k]) && code <= unsigned(b.hi[k]);
        }
        probes_ok += ok;
        if (!ok && probes - probes_ok <= 5)
            std::printf("lanterns: probe (%zu, %zu): depth %.7g want %.7g..%.7g; codes %u %u %u want [%g..%g] [%g..%g] [%g..%g]\n",
                        i % lanterns::kWidth, i / lanterns::kWidth, double(depth[i]), double(b.lo[3]), double(b.hi[3]),
                        normal_words[i] & 1023u, (normal_words[i] >> 10) & 1023u, (normal_words[i] >> 20) & 1023u,
                        double(b.lo[0]), double(b.hi[0]), double(b.lo[1]), double(b.hi[1]), double(b.lo[2]), double(b.hi[2]));
    }
    const auto albedo_ref = lanterns::albedo_reference(scene);
    const auto ref = lanterns::deferred_reference(scene, lights), ref_moved = lanterns::deferred_reference(scene, moved);
    const std::vector<Color> frame = lanterns::unpack_rgba8(culled), frame_moved = lanterns::unpack_rgba8(shifted);
    const lanterns::Comparison ca = lanterns::compare_textured(albedo, albedo_ref);
    const lanterns::Comparison c = lanterns::compare_textured(frame, ref), cm = lanterns::compare_textured(frame_moved, ref_moved);
    const auto single = [](const lanterns::TexturedReference& r, std::size_t i) {
        return r.certain[i] && lanterns::detail::same(r.bounds[i].lo, r.bounds[i].hi);
    };
    unsigned exact = 0, singles = 0, expected = 0, changed = 0, kept_still = 0, still = 0, same_as_all = 0;
    for (std::size_t i = 0; i < frame.size(); ++i) {
        same_as_all += culled[i] == all[i];
        if (!single(ref, i)) continue;
        ++singles;
        exact += lanterns::detail::same(frame[i], ref.bounds[i].lo);
        if (!single(ref_moved, i)) continue;
        if (!lanterns::detail::same(ref.bounds[i].lo, ref_moved.bounds[i].lo)) {
            ++expected;
            changed += culled[i] != shifted[i];
        } else {
            ++still;
            kept_still += culled[i] == shifted[i];
        }
    }
    std::uint64_t kept_total = 0;
    unsigned kept_max = 0, empty_tiles = 0, all_full = 0;
    for (std::size_t t = 0; t < counts.size(); ++t) {
        kept_total += counts[t];
        kept_max = std::max(kept_max, counts[t]);
        empty_tiles += counts[t] == 0;
        all_full += counts_all[t] == lanterns::kLights;
    }
    std::printf("lanterns: G-buffer albedo (RGBA8 MRT0): %u of %u certain pixels exact, %u of %u uncertain within bounds\n",
                ca.certain_exact, ca.certain, ca.uncertain_plausible, ca.uncertain);
    if (!ca.pass()) report("G-buffer albedo", ca);
    std::printf("lanterns: lit frame: %u of %u single-colour pixels exact, %u of %u certain pixels within bounds, %u of "
                "%u uncertain within bounds\n", exact, singles, c.certain_exact, c.certain, c.uncertain_plausible,
                c.uncertain);
    if (!c.pass()) report("lit frame", c);
    std::printf("lanterns: tiled culling kept %.1f lights per tile on average (max %u of %u, %u empty tiles); culling "
                "off: %u of %zu tiles listed every light, %u of %zu pixels identical to the culled frame\n",
                double(kept_total) / double(counts.size()), kept_max, lanterns::kLights, empty_tiles, all_full,
                counts.size(), same_as_all, frame.size());
    std::printf("lanterns: depth slack: %u probes are off by more than the old fixed kPosSlack (%.0e), the largest error "
                "is %.2f of its gradient slack, which the lighting and ray references now use\n", beyond_fixed,
                lanterns::kPosSlack, worst);
    std::printf("lanterns: G-buffer probes: %u of %u certain pixels with the expected 10-bit normal codes and view depth "
                "(within the edge slack); exact light sets: %u of %zu tiles (%u light-tile pairs within 2e-3 of the box test either way), "
                "culling off %u of %zu tiles with all 256 once, moved light %u of %zu; %u poisoned words left\n",
                probes_ok, probes, sets_ok, counts.size(), sets_either, sets_all_ok, counts.size(), sets_moved_ok,
                counts.size(), poisoned);
    std::printf("lanterns: control, light %zu moved 0.875 right: %u of %u predicted pixels changed, %u of %u pixels "
                "predicted unchanged kept; moved frame %u of %u certain pixels within its bounds\n", moved_index,
                changed, expected, kept_still, still, cm.certain_exact, cm.certain);
    if (!cm.pass()) report("moved-light frame", cm);
    lanterns::Reference shown;
    shown.color.resize(frame.size());
    shown.candidates.resize(frame.size());
    std::vector<std::uint8_t> states(frame.size());
    for (std::size_t i = 0; i < frame.size(); ++i) {
        for (int k = 0; k < 4; ++k) shown.color[i][k] = (ref.bounds[i].lo[k] + ref.bounds[i].hi[k]) / 2;
        states[i] = !lanterns::within(frame[i], ref.bounds[i]) ? 2 : single(ref, i) ? 3 : 1;
    }
    if (!dump(out_dir, "stage9-frame", frame, shown, &states) || !dump(out_dir, "stage9-moved", frame_moved, shown, &states) ||
        !dump(out_dir, "stage9-albedo", albedo, shown))
        return 1;
    if (!ca.pass() || !c.pass() || !cm.pass() || exact != singles || same_as_all != frame.size() ||
        all_full != counts.size() || poisoned || probes < 50000 || worst > 0.5 || probes_ok != probes || sets_ok != counts.size() ||
        sets_either > lanterns::kMaxEitherPairs || sets_all_ok != counts.size() || sets_moved_ok != counts.size() || expected < 500 || changed != expected || kept_still != still ||
        kept_total >= std::uint64_t(lanterns::kLights) * counts.size() / 2) {
        std::fprintf(stderr, "lanterns: stage 9 FAILED\n");
        return 1;
    }
    std::printf("Lanterns stage 9: PASS, G-buffer of 3 MRTs (RGBA8 albedo %u of %u exact, RGB10A2 normal and R32F depth "
                "%u of %u probes), %u of %u single-colour pixels exact through tiled culling with LDS atomics (%.1f of "
                "256 lights per tile, the exact set in %u of %zu tiles, %u light-tile pairs within 2e-3 of the box test left free), "
                "culling off gave the same image bit for bit "
                "from poisoned buffers, one light moved changed %u of %u predicted pixels\n",
                ca.certain_exact, ca.certain, probes_ok, probes, exact, singles,
                double(kept_total) / double(counts.size()), sets_ok, counts.size(), sets_either, changed, expected);
    return 0;
}

// Stage 10: 4x MSAA with ResolveSubresource (see the top of this file).
int stage10(const std::filesystem::path& shader_dir, const std::filesystem::path& out_dir) {
    lanterns::Shaders shaders;
    if (!lanterns::load_shaders(shader_dir, 1, shaders)) return 1;
    const lanterns::Scene scene = lanterns::stage1();
    std::vector<Color> resolved, single;
    std::array<std::vector<Color>, 4> isolated;   // drawn with SampleMask 1 << s
    {
        lanterns::Gpu gpu;
        if (!gpu.create(scene, shaders, DXGI_FORMAT_R32G32B32A32_FLOAT, 4) || !gpu.submit_frame(scene, true, 0) ||
            !gpu.finish_frame(resolved) || !gpu.idle())
            return 1;
    }
    for (unsigned s = 0; s < 4; ++s) {
        lanterns::Gpu gpu;
        if (!gpu.create(scene, shaders, DXGI_FORMAT_R32G32B32A32_FLOAT, 4, 1u << s) || !gpu.submit_frame(scene, true, 0) ||
            !gpu.finish_frame(isolated[s]) || !gpu.idle())
            return 1;
    }
    {
        lanterns::Gpu gpu;
        if (!gpu.create(scene, shaders) || !gpu.submit_frame(scene, true, 0) || !gpu.finish_frame(single) || !gpu.idle())
            return 1;
    }
    const lanterns::MsaaReference ref = lanterns::msaa_reference(scene);
    const lanterns::Reference ref1 = lanterns::reference_frame(scene, true);
    const lanterns::Comparison c = lanterns::compare_textured(resolved, ref.resolved), c1 = lanterns::compare(single, ref1);
    unsigned blended = 0, blended_exact = 0, expected = 0, changed = 0, still = 0, kept = 0;
    for (std::size_t i = 0; i < resolved.size(); ++i) {
        if (ref.blended[i]) ++blended, blended_exact += lanterns::detail::same(resolved[i], ref.resolved.bounds[i].lo);
        if (!ref.resolved.certain[i] || !ref1.candidates[i].empty()) continue;
        if (!lanterns::detail::same(ref.resolved.bounds[i].lo, ref1.color[i])) {
            ++expected;
            changed += !lanterns::detail::same(resolved[i], single[i]);
        } else {
            ++still;
            kept += lanterns::detail::same(resolved[i], single[i]);
        }
    }
    std::printf("lanterns: 4x resolved frame: %u of %u certain pixels exact (%u of %u edge pixels whose samples differ, "
                "averaged exactly), %u of %u uncertain within bounds\n", c.certain_exact, c.certain, blended_exact,
                blended, c.uncertain_plausible, c.uncertain);
    if (!c.pass()) report("4x resolved frame", c);
    std::printf("lanterns: control, the same frame at 1x: %u of %u certain pixels exact; %u of %u predicted edge pixels "
                "differ from the resolve, %u of %u predicted equal pixels equal\n", c1.certain_exact, c1.certain, changed,
                expected, kept, still);
    if (!c1.pass()) report("1x frame", c1);
    // Uncertain pixels: the resolve must be one of the averages their samples' candidates can make, exactly.
    unsigned enumerated = 0, enumerated_ok = 0, too_many = 0;
    for (std::size_t i = 0; i < resolved.size(); ++i) {
        if (ref.resolved.certain[i]) continue;
        if (ref.averages[i].empty()) {
            ++too_many;
            continue;
        }
        ++enumerated;
        for (const Color& a : ref.averages[i])
            if (lanterns::detail::same(resolved[i], a)) {
                ++enumerated_ok;
                break;
            }
    }
    // One sample at a time: certain pixels exact, uncertain ones one of their candidates; "distinct" counts the certain
    // pixels where this sample's prediction differs from every other sample's (what a wrong mask bit to position
    // mapping, or an ignored mask, gets wrong).
    std::array<unsigned, 4> iso_certain{}, iso_exact{}, iso_uncertain{}, iso_plausible{}, distinct{}, distinct_ok{};
    std::array<lanterns::Reference, 4> masked;
    for (unsigned s = 0; s < 4; ++s) masked[s] = lanterns::masked_resolve(ref.per_sample[s]);
    for (unsigned s = 0; s < 4; ++s)
        for (std::size_t i = 0; i < resolved.size(); ++i) {
            const auto& m = masked[s];
            if (!m.candidates[i].empty()) {
                ++iso_uncertain[s];
                for (const Color& cand : m.candidates[i])
                    if (lanterns::detail::same(isolated[s][i], cand)) {
                        ++iso_plausible[s];
                        break;
                    }
                continue;
            }
            ++iso_certain[s];
            const bool ok = lanterns::detail::same(isolated[s][i], m.color[i]);
            iso_exact[s] += ok;
            bool unique = true;
            for (unsigned o = 0; o < 4 && unique; ++o)
                if (o != s) unique = !masked[o].candidates[i].empty() ? false : !lanterns::detail::same(masked[o].color[i], m.color[i]);
            distinct[s] += unique, distinct_ok[s] += unique && ok;
        }
    bool isolated_ok = true;
    unsigned min_distinct = ~0u;
    for (unsigned s = 0; s < 4; ++s) {
        std::printf("lanterns: SampleMask 0x%x: %u of %u certain pixels exact (%u of %u where only this sample's position "
                    "gives that colour), %u of %u uncertain one of their candidates\n", 1u << s, iso_exact[s],
                    iso_certain[s], distinct_ok[s], distinct[s], iso_plausible[s], iso_uncertain[s]);
        isolated_ok = isolated_ok && iso_exact[s] == iso_certain[s] && iso_plausible[s] == iso_uncertain[s] &&
                      distinct_ok[s] == distinct[s];
        min_distinct = std::min(min_distinct, distinct[s]);
    }
    std::printf("lanterns: uncertain pixels: %u of %u equal one of their enumerated 4-sample averages exactly (%u with "
                "too many combinations, bounds only)\n", enumerated_ok, enumerated, too_many);
    lanterns::Reference shown;
    shown.color.resize(resolved.size());
    shown.candidates.resize(resolved.size());
    std::vector<std::uint8_t> states(resolved.size());
    for (std::size_t i = 0; i < resolved.size(); ++i) {
        for (int k = 0; k < 4; ++k)
            shown.color[i][k] = (ref.resolved.bounds[i].lo[k] + ref.resolved.bounds[i].hi[k]) / 2;
        states[i] = !lanterns::within(resolved[i], ref.resolved.bounds[i]) ? 2 : ref.blended[i] ? 1
                    : ref.resolved.certain[i] ? 0 : 3;
    }
    if (!dump(out_dir, "stage10-frame", resolved, shown, &states) || !dump(out_dir, "stage10-single", single, shown, &states))
        return 1;
    if (!c.pass() || !c1.pass() || blended < 500 || blended_exact != blended || expected < 500 || changed != expected ||
        kept != still || enumerated_ok != enumerated || too_many || !isolated_ok || min_distinct < 50) {
        std::fprintf(stderr, "lanterns: stage 10 FAILED\n");
        return 1;
    }
    std::printf("Lanterns stage 10: PASS, 4x MSAA resolved by ResolveSubresource: %u of %u certain pixels exact at the "
                "standard sample positions (%u edge pixels blended exactly), the 1x render differs on %u of %u predicted "
                "edge pixels and nowhere else certain; %u of %u uncertain pixels are one of their enumerated averages; SampleMask "
                "1 << s isolated each sample (at least %u pixels only that sample's position explains)\n", c.certain_exact,
                c.certain, blended, changed, expected, enumerated_ok, enumerated, min_distinct);
    return 0;
}

// Stage 11: ray-traced shadows through BLAS/TLAS and image_bvh_intersect_ray (see the top of this file).
int stage11(const std::filesystem::path& shader_dir, const std::filesystem::path& out_dir) {
    lanterns::Shaders shaders;
    std::vector<std::uint8_t> rt_shader;
    if (!lanterns::load_shaders(shader_dir, 9, shaders) || !lanterns::load_rt_shader(shader_dir, rt_shader)) return 1;
    const lanterns::Scene scene = lanterns::stage11();
    constexpr unsigned kMasked = 1;   // the control TLAS gives this instance mask 0
    std::vector<std::uint32_t> traced, masked, swapped, steps, steps_masked, steps_swapped, stalled, steps_stalled,
        overflowed, steps_overflowed;
    double stall_ms = 0;
    lanterns::RtSizes sizes;
    {
        lanterns::Gpu gpu;
        lanterns::RtShadows rt;
        std::vector<Color> albedo;
        if (!gpu.create(scene, shaders, DXGI_FORMAT_R8G8B8A8_UNORM) || !gpu.submit_frame(scene, true, 0) ||
            !gpu.finish_frame(albedo))
            return 1;
        if (!rt.create(gpu, rt_shader, scene, kMasked)) {
            std::fprintf(stderr, "lanterns: stage 11 BLOCKED before tracing (see the line above)\n");
            return 1;
        }
        sizes = rt.sizes();
        if (!rt.run(scene, lanterns::RtRun::traced, traced, steps) ||
            !rt.run(scene, lanterns::RtRun::masked, masked, steps_masked) ||
            !rt.run(scene, lanterns::RtRun::swapped, swapped, steps_swapped) ||
            !rt.run(scene, lanterns::RtRun::overflow, overflowed, steps_overflowed))
            return 1;
        const auto t0 = std::chrono::steady_clock::now();
        if (!rt.run(scene, lanterns::RtRun::stall, stalled, steps_stalled) || !gpu.idle()) return 1;
        stall_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    }
    const auto ref = lanterns::rt_reference(scene, lanterns::rt_triangles(scene));
    const auto ref_masked = lanterns::rt_reference(scene, lanterns::rt_triangles(scene, int(kMasked)));
    const auto ref_swapped = lanterns::rt_reference(scene, lanterns::rt_triangles(scene, -1, true));
    const std::vector<Color> frame = lanterns::unpack_rgba8(traced), frame_masked = lanterns::unpack_rgba8(masked),
                             frame_swapped = lanterns::unpack_rgba8(swapped);
    const lanterns::Comparison c = lanterns::compare_textured(frame, ref), cm = lanterns::compare_textured(frame_masked, ref_masked),
                               cs = lanterns::compare_textured(frame_swapped, ref_swapped);
    const auto single = [](const lanterns::TexturedReference& r, std::size_t i) {
        return r.certain[i] && lanterns::detail::same(r.bounds[i].lo, r.bounds[i].hi);
    };
    unsigned exact = 0, singles = 0, unfinished = 0, poisoned = 0;
    std::uint64_t step_total = 0, traced_lanes = 0;
    std::uint32_t step_max = 0;
    for (std::size_t i = 0; i < frame.size(); ++i) {
        for (const auto* v : {&traced, &masked, &swapped, &steps, &steps_masked, &steps_swapped, &stalled, &steps_stalled,
                              &overflowed, &steps_overflowed})
            poisoned += (*v)[i] == lanterns::kRtPoison;
        unfinished += traced[i] == lanterns::kRtUnfinished || masked[i] == lanterns::kRtUnfinished ||
                      swapped[i] == lanterns::kRtUnfinished;
        if (steps[i]) step_total += steps[i], ++traced_lanes, step_max = std::max(step_max, steps[i]);
        if (!single(ref, i)) continue;
        ++singles;
        exact += lanterns::detail::same(frame[i], ref.bounds[i].lo);
    }
    // A control changes exactly the single-colour pixels whose expected colour changes, and keeps the others.
    struct Moved { unsigned expected = 0, changed = 0, still = 0, kept = 0; };
    const auto moved = [&](const lanterns::TexturedReference& other, const std::vector<std::uint32_t>& px) {
        Moved mv;
        for (std::size_t i = 0; i < frame.size(); ++i) {
            if (!single(ref, i) || !single(other, i)) continue;
            if (!lanterns::detail::same(ref.bounds[i].lo, other.bounds[i].lo)) ++mv.expected, mv.changed += traced[i] != px[i];
            else ++mv.still, mv.kept += traced[i] == px[i];
        }
        return mv;
    };
    const Moved mask_move = moved(ref_masked, masked), swap_move = moved(ref_swapped, swapped);
    // Which lanes trace at all comes from the CPU, not from the shader: a single-colour pixel that is not the
    // background has geometry (its view depth is above 0, so its lane starts a ray); the background has none.
    const Color background{float(lanterns::kBackground & 255u) / 255.0f, float((lanterns::kBackground >> 8) & 255u) / 255.0f,
                           float((lanterns::kBackground >> 16) & 255u) / 255.0f, 1.0f};
    unsigned rays = 0, capped = 0, empty = 0, idle = 0, overflow_rays = 0, overflow_marked = 0, bad_uncertain = 0;
    for (std::size_t i = 0; i < frame.size(); ++i) {
        const bool cap_ok = steps_stalled[i] == lanterns::kRtStepCap && stalled[i] == lanterns::kRtUnfinished;
        const bool idle_ok = steps_stalled[i] == 0 && stalled[i] == lanterns::kBackground;
        // The overflow fixture: each step pops one entry and pushes four; the eleventh step's third push has no room.
        const bool over_ok = steps_overflowed[i] == (11u | lanterns::kRtOverflow) && overflowed[i] == lanterns::kRtUnfinished;
        const bool over_idle = steps_overflowed[i] == 0 && overflowed[i] == lanterns::kBackground;
        if (!single(ref, i)) {
            bad_uncertain += !((cap_ok && over_ok) || (idle_ok && over_idle));
            continue;
        }
        if (lanterns::detail::same(ref.bounds[i].lo, background)) {
            ++empty, idle += idle_ok && over_idle;
        } else {
            ++rays, capped += cap_ok;
            ++overflow_rays, overflow_marked += over_ok;
        }
    }
    std::printf("lanterns: acceleration structures: 2 BLASes %llu bytes together (a prism of %zu triangles, a column of %zu), "
                "TLAS %llu bytes (%zu instances), scratch %llu bytes\n", static_cast<unsigned long long>(sizes.blas),
                lanterns::rt_indices(scene.meshes[0]).size() / 3, lanterns::rt_indices(scene.meshes[1]).size() / 3,
                static_cast<unsigned long long>(sizes.tlas), lanterns::rt_transforms(scene).size(),
                static_cast<unsigned long long>(sizes.scratch));
    std::printf("lanterns: traced frame: %u of %u single-colour pixels exact, %u of %u certain pixels within bounds, %u of "
                "%u uncertain within bounds; %.1f traversal steps per ray (max %u), %u rays left unfinished, %u "
                "poisoned words left\n", exact, singles, c.certain_exact, c.certain, c.uncertain_plausible, c.uncertain,
                traced_lanes ? double(step_total) / double(traced_lanes) : 0.0, step_max, unfinished, poisoned);
    if (!c.pass()) report("traced frame", c);
    std::printf("lanterns: control, instance %u masked out: %u of %u predicted pixels changed, %u of %u predicted unchanged "
                "kept; %u of %u certain pixels within its bounds\n", kMasked, mask_move.changed, mask_move.expected,
                mask_move.kept, mask_move.still, cm.certain_exact, cm.certain);
    if (!cm.pass()) report("masked frame", cm);
    std::printf("lanterns: control, instances %u and %u swap BLAS references: %u of %u predicted pixels changed, %u of %u "
                "predicted unchanged kept; %u of %u certain pixels within its bounds\n", lanterns::kRtSwapA,
                lanterns::kRtSwapB, swap_move.changed, swap_move.expected, swap_move.kept, swap_move.still,
                cs.certain_exact, cs.certain);
    if (!cs.pass()) report("swapped frame", cs);
    std::printf("lanterns: faked stall (a BVH whose root box is its only child): %u of %u rays stopped at the %u-step cap and "
                "marked, %u of %u background lanes idle (0 steps), the pass returned in %.0f ms\n", capped, rays,
                lanterns::kRtStepCap, idle, empty, stall_ms);
    std::printf("lanterns: stack overflow (a root box that is its own four children): %u of %u rays stopped at step 11 "
                "marked as overflowed; %u uncertain pixels with neither outcome\n", overflow_marked, overflow_rays,
                bad_uncertain);
    lanterns::Reference shown;
    shown.color.resize(frame.size());
    shown.candidates.resize(frame.size());
    std::vector<std::uint8_t> states(frame.size());
    for (std::size_t i = 0; i < frame.size(); ++i) {
        for (int k = 0; k < 4; ++k) shown.color[i][k] = (ref.bounds[i].lo[k] + ref.bounds[i].hi[k]) / 2;
        states[i] = !lanterns::within(frame[i], ref.bounds[i]) ? 2 : single(ref, i) ? 3 : 1;
    }
    if (!dump(out_dir, "stage11-frame", frame, shown, &states) || !dump(out_dir, "stage11-masked", frame_masked, shown, &states) ||
        !dump(out_dir, "stage11-swapped", frame_swapped, shown, &states))
        return 1;
    if (!c.pass() || !cm.pass() || !cs.pass() || exact != singles || unfinished || poisoned || mask_move.expected < 300 ||
        swap_move.expected < 300 || mask_move.changed != mask_move.expected ||
        mask_move.kept != mask_move.still || swap_move.changed != swap_move.expected || swap_move.kept != swap_move.still ||
        rays < 50000 || capped != rays || empty < 5000 || idle != empty || overflow_marked != overflow_rays ||
        bad_uncertain) {
        std::fprintf(stderr, "lanterns: stage 11 FAILED\n");
        return 1;
    }
    std::printf("Lanterns stage 11: PASS, one shadow ray per pixel through a TLAS of %zu instances over 2 BLASes built by "
                "BuildRaytracingAccelerationStructure, traversed with image_bvh_intersect_ray: %u of %u single-colour "
                "pixels exact, masking one instance changed %u of %u predicted pixels, swapping two BLAS references "
                "changed %u of %u; the step cap stopped %u of %u stalled rays and a full stack %u of %u\n",
                lanterns::rt_transforms(scene).size(), exact, singles, mask_move.changed, mask_move.expected,
                swap_move.changed, swap_move.expected, capped, rays, overflow_marked, overflow_rays);
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
    case 5: return stage5(shader_dir, out_dir);
    case 6: return stage6(shader_dir, out_dir);
    case 8: return stage8(shader_dir, out_dir);
    case 9: return stage9(shader_dir, out_dir);
    case 10: return stage10(shader_dir, out_dir);
    case 11: return stage11(shader_dir, out_dir);
    default:
        std::fprintf(stderr, "usage: lanterns --stage 1|3|5|6|8|9|10|11 --shaders <folder fssm_pack.py wrote> --out <folder>\n");
        return 2;
    }
}
