// Lanterns: an original stand-in title that runs under gta6rt --run and drives the whole chain a real title would:
// an x64 PE the loader maps, imports the loader's shims answer, the Xbox D3D12 device on Vulkan
// (D3D12XboxCreateDevice from d3d12_xs.dll, LEONIDA_D3D12=vulkan), pipelines from shader containers holding RDNA2
// programs (DXBC { DXIL { XDAT { XBCH + zlib(FSSM) } } }, packed by fssm_pack.py beside this file, translated offline by
// fssm_unpack.py, shader_disasm.py and rdna2spirv.py; the layer finds each by the SHA-256 of its FSSM program), draws,
// and a readback checked against a CPU reference. Content: abstract solids on a checkered floor, colours from short
// binary fractions; no game data. How to build and run it: lanterns.cmake.
//
//   lanterns.exe --stage 1 --shaders <folder fssm_pack.py wrote> --out <folder for the frame dumps>
//
// Stage 1: five DrawIndexedInstanced (hexagonal prisms, cubes, octahedra, pyramids, the floor) into a 384x256
// R32G32B32A32_FLOAT target with a D32_FLOAT depth buffer. Vertex, index and instance buffers live in DEFAULT heaps,
// filled from an upload buffer by copies; the vertex shader reads vertices and instances through V#s in two descriptor
// tables, the model matrix and tint from root constants, the camera from a root CBV. The frame is drawn twice: with
// the depth test (the image) and without it (the negative control, which must change exactly the pixels the reference
// says it changes).
//
// On success the guest prints one line starting "Lanterns stage N: PASS" and exits 0; run_lanterns.py checks that
// line, the exit code and the device line.
#include "lanterns_gpu.h"
#include "lanterns_reference.h"
#include "lanterns_scene.h"

#include <algorithm>
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
    if (!lanterns::load_shaders(shader_dir, shaders)) return 1;
    std::vector<Color> with_depth, without_depth;
    {
        lanterns::Gpu gpu;
        if (!gpu.create(scene, shaders)) return 1;
        if (!gpu.submit_frame(scene, true) || !gpu.finish_frame(with_depth)) return 1;
        if (!gpu.submit_frame(scene, false) || !gpu.finish_frame(without_depth)) return 1;
        if (!gpu.idle()) return 1;
    }
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
    if (!on.pass() || !off.pass() || control.expected < 1000 || control.changed != control.expected) {
        std::fprintf(stderr, "lanterns: stage 1 FAILED\n");
        return 1;
    }
    // Printed only here, after every check: run_lanterns.py wants this line once and exit code 0.
    std::printf("Lanterns stage 1: PASS, %u of %u pixels exact, %u edge pixels plausible, depth-off control changed "
                "%u pixels\n", on.certain_exact, kWidth * kHeight, on.uncertain_plausible, control.changed);
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
    default:
        std::fprintf(stderr, "usage: lanterns --stage 1 --shaders <folder fssm_pack.py wrote> --out <folder>\n");
        return 2;
    }
}
