// Pure-CPU tests for runtime/src/mesh_compute_vs.h, the vertex shader that draws a --mesh-compute mesh shader's slots:
//   - layouts built as rdna2spirv.mesh_compute_layout builds them: one small, one like the corpus's (128/128 outputs,
//     18 params, 3 clip distances), one with clip and cull distances, one with --f16-interp-fallback halves;
//   - the module's interface: a Vertex entry point reading VertexIndex and the 152-byte push-constant block, Position,
//     the distance arrays and each param location out, then each half's two locations;
//   - refusals: no vertices, the address in the --pc-tail-ubo block, an f16_halves param that is not a layout param,
//     a half location that collides with another output.
// With a folder as the first argument it writes the modules there, for the spirv-val tests CMake adds.
// No GPU or game files required.
#include "../src/mesh_compute_vs.h"
#include "../src/spirv_f16_halves.h"

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <map>
#include <set>
#include <string>
#include <vector>

#define CHECK(cond) do { \
    if (!(cond)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #cond); return 1; } \
} while (0)

namespace {

using namespace leonida;
using namespace leonida::mesh_vs_detail;

// rdna2spirv.mesh_compute_layout, for the default push-constant layout.
MeshComputeLayout layout(std::uint32_t maxv, std::uint32_t maxp, std::uint32_t clip, std::uint32_t cull,
                         const std::vector<std::uint32_t>& params) {
    const std::uint32_t clip_vec4 = (clip + 3) / 4, cull_vec4 = (cull + 3) / 4;
    MeshComputeLayout l;
    l.vertex_stride = 16 * (1 + clip_vec4 + cull_vec4 + std::uint32_t(params.size()));
    l.vertices = 16;
    l.primitives = l.vertices + maxv * l.vertex_stride;
    l.slot_bytes = l.primitives + maxp * 16;
    l.primitive_stride = 16;
    l.max_vertices = maxv;
    l.max_primitives = maxp;
    l.clip_count = clip;
    l.clip_offset = 16;
    l.cull_count = cull;
    l.cull_offset = 16 * (1 + clip_vec4);
    for (std::size_t k = 0; k < params.size(); ++k)
        l.params[params[k]] = 16 * (1 + clip_vec4 + cull_vec4 + std::uint32_t(k));
    l.push_constant_bytes = 4 * 32 + 24;
    return l;
}

struct Summary {
    bool ok = false;
    std::uint32_t model = ~0u, interface_count = 0;
    std::set<std::uint32_t> capabilities, builtins, locations;
    std::uint32_t push_constant_vars = 0;
};

Summary summarize(const std::vector<std::uint32_t>& words) {
    Summary s;
    spirv_detail::Module m;
    if (words.size() < 5 || words[1] != 0x00010500u || !m.parse(words)) return s;
    for (const auto& inst : m.insts) {
        const auto& w = inst.words;
        switch (inst.opcode) {
            case kOpCapability: s.capabilities.insert(w[1]); break;
            case kOpEntryPoint: {
                s.model = w[1];
                std::size_t i = 3;  // past the name's words: its last word holds the terminating nul
                while (i < w.size() && (w[i] >> 24) != 0) ++i;
                s.interface_count = std::uint32_t(w.size() - (i + 1));
                break;
            }
            case kOpDecorate:
                if (w[2] == kDecBuiltIn) s.builtins.insert(w[3]);
                if (w[2] == kDecLocation) s.locations.insert(w[3]);
                break;
            case kOpVariable:
                if (w[3] == kStoragePushConstant) ++s.push_constant_vars;
                break;
            default: break;
        }
    }
    s.ok = true;
    return s;
}

bool write(const std::string& dir, const char* name, const std::vector<std::uint32_t>& words) {
    std::ofstream f(dir + "/" + name + ".spv", std::ios::binary);
    f.write(reinterpret_cast<const char*>(words.data()), std::streamsize(words.size() * 4));
    return bool(f);
}

}  // namespace

int main(int argc, char** argv) {
    // Small: 4 vertices, 2 primitives, two params.
    const auto small = mesh_compute_vs(layout(4, 2, 0, 0, {0, 2}));
    const Summary s = summarize(small);
    CHECK(s.ok && s.model == kExecModelVertex);
    CHECK((s.capabilities == std::set<std::uint32_t>{kCapShader, kCapInt64, kCapPhysicalStorageBufferAddresses}));
    CHECK((s.builtins == std::set<std::uint32_t>{kBuiltInVertexIndex, kBuiltInPosition}));
    CHECK((s.locations == std::set<std::uint32_t>{0, 2}));
    CHECK(s.interface_count == 2 + 1 + 2);  // VertexIndex, the push constants, Position, two params
    CHECK(s.push_constant_vars == 1);

    // Like the corpus: 128 vertices and primitives, 18 params, 3 clip distances.
    std::vector<std::uint32_t> eighteen;
    for (std::uint32_t n = 0; n < 18; ++n) eighteen.push_back(n);
    const auto corpus = mesh_compute_vs(layout(128, 128, 3, 0, eighteen));
    const Summary c = summarize(corpus);
    CHECK(c.ok && c.capabilities.count(kCapClipDistance) && !c.capabilities.count(kCapCullDistance));
    CHECK(c.builtins.count(kBuiltInClipDistance) && c.locations.size() == 18);

    // Clip and cull distances together.
    const auto distances = mesh_compute_vs(layout(8, 4, 2, 5, {1}));
    const Summary d = summarize(distances);
    CHECK(d.ok && d.builtins.count(kBuiltInClipDistance) && d.builtins.count(kBuiltInCullDistance));
    CHECK(d.interface_count == 2 + 1 + 2 + 1);

    // --f16-interp-fallback: param 1's halves go out at 28 and 29 as well.
    MeshComputeLayout halves = layout(4, 2, 0, 0, {0, 1});
    halves.f16_halves[1] = {28, 29};
    const auto f16 = mesh_compute_vs(halves);
    const Summary h = summarize(f16);
    CHECK(h.ok && (h.locations == std::set<std::uint32_t>{0, 1, 28, 29}) && h.interface_count == 2 + 1 + 4);

    // Refusals.
    std::string error;
    CHECK(mesh_compute_vs(MeshComputeLayout{}, &error).empty() && error.find("without vertices") != std::string::npos);
    MeshComputeLayout tail = layout(4, 2, 0, 0, {0});
    tail.address_in_pc_tail = true;
    CHECK(mesh_compute_vs(tail, &error).empty() && error.find("pc-tail-ubo") != std::string::npos);
    MeshComputeLayout stray = layout(4, 2, 0, 0, {0});
    stray.f16_halves[3] = {20, 21};
    CHECK(mesh_compute_vs(stray, &error).empty() && error.find("not in the layout") != std::string::npos);
    MeshComputeLayout collide = layout(4, 2, 0, 0, {0, 1});
    collide.f16_halves[1] = {0, 29};
    CHECK(mesh_compute_vs(collide, &error).empty() && error.find("collides") != std::string::npos);

    if (argc > 1) {
        CHECK(write(argv[1], "small", small));
        CHECK(write(argv[1], "corpus", corpus));
        CHECK(write(argv[1], "distances", distances));
        CHECK(write(argv[1], "f16", f16));
    }
    std::printf("mesh_compute_vs: interfaces of four layouts (small, corpus-like, clip+cull, f16 halves) and the "
                "refusals as expected\n");
    return 0;
}
