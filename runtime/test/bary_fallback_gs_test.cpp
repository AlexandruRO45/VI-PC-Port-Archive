// Pure-CPU tests for runtime/src/bary_fallback_gs.h, the pass-through geometry shader of a --bary-fallback PS:
//   - the module's interface: Geometry capability, triangles in, a 3-vertex strip out, Position and each
//     pass-through location in and out, the barycentric vec3 out at its own location, and the one-hot constants;
//   - ClipDistance and CullDistance arrays forwarded with their capabilities (both at once too);
//   - bary_fallback_gs_for takes the locations from the PS module, not the index (link_f16_halves can move them),
//     and refuses a PS without a barycentric input, one reading PerVertexKHR inputs, and a VS writing Layer;
//   - errors: the barycentric location also a pass-through one, a distance array longer than 8.
// With a folder as the first argument it writes the modules there, for the spirv-val tests CMake adds.
// No GPU or game files required.
#include "../src/bary_fallback_gs.h"

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
using namespace leonida::bary_gs_detail;

struct Summary {
    std::set<std::uint32_t> capabilities, modes, builtins_in, builtins_out;
    std::set<std::uint32_t> locations_in, locations_out;
    std::uint32_t model = ~0u, interface_count = 0, vertex_emits = 0, bary_location = ~0u;
    std::vector<std::vector<std::uint32_t>> composites;  // OpConstantComposite constituents
    bool ok = false;
};

// What the module declares, from its instructions.
Summary summarize(const std::vector<std::uint32_t>& words) {
    Summary s;
    spirv_detail::Module m;
    if (words.size() < 5 || words[1] != 0x00010000u || !m.parse(words)) return s;
    std::map<std::uint32_t, std::uint32_t> storage;  // variable -> storage class
    std::map<std::uint32_t, std::uint32_t> pointee;  // pointer type -> pointee type
    std::map<std::uint32_t, std::uint32_t> vec_size;  // vector type -> component count
    std::map<std::uint32_t, std::uint32_t> var_type;
    std::map<std::uint32_t, std::pair<std::uint32_t, std::uint32_t>> decorations;  // var -> (decoration, value)
    std::map<std::uint32_t, std::uint32_t> constants;
    for (const auto& inst : m.insts) {
        const auto& w = inst.words;
        switch (inst.opcode) {
            case kOpCapability: s.capabilities.insert(w[1]); break;
            case kOpEntryPoint: s.model = w[1]; s.interface_count = std::uint32_t(w.size() - 5); break;
            case kOpExecutionMode: s.modes.insert(w[2]); break;
            case kOpTypeVector: vec_size[w[1]] = w[3]; break;
            case kOpTypePointer: pointee[w[1]] = w[3]; break;
            case kOpVariable: storage[w[2]] = w[3]; var_type[w[2]] = pointee[w[1]]; break;
            case kOpDecorate: decorations[w[1]] = {w[2], w[3]}; break;
            case kOpConstant: constants[w[2]] = w[3]; break;
            case kOpConstantComposite: {
                std::vector<std::uint32_t> c;
                for (std::size_t i = 3; i < w.size(); ++i) c.push_back(constants[w[i]] == 0x3f800000u ? 1 : 0);
                s.composites.push_back(c);
                break;
            }
            case kOpEmitVertex: ++s.vertex_emits; break;
            default: break;
        }
    }
    for (const auto& [var, d] : decorations) {
        const bool in = storage[var] == kStorageInput;
        if (d.first == kDecorationBuiltIn) (in ? s.builtins_in : s.builtins_out).insert(d.second);
        if (d.first != kDecorationLocation) continue;
        if (!in && vec_size[var_type[var]] == 3) s.bary_location = d.second;
        else (in ? s.locations_in : s.locations_out).insert(d.second);
    }
    s.ok = true;
    return s;
}

std::vector<std::uint32_t> header() { return {0x07230203u, 0x00010000u, 0, 100, 0}; }

void inst(std::vector<std::uint32_t>& w, std::uint16_t op, std::initializer_list<std::uint32_t> args) {
    w.push_back((std::uint32_t(1 + args.size()) << 16) | op);
    w.insert(w.end(), args.begin(), args.end());
}

// A PS module reduced to what bary_fallback_gs_for reads: Input variables (id 10 + n) at these locations, an optional
// extra decoration on the first, and with `primitive_id` an Input variable (id 30) decorated BuiltIn PrimitiveId.
std::vector<std::uint32_t> ps_module(const std::vector<std::uint32_t>& locations, std::uint32_t extra = 0,
                                     bool primitive_id = false) {
    auto w = header();
    for (std::size_t i = 0; i < locations.size(); ++i)
        inst(w, kOpDecorate, {std::uint32_t(10 + i), kDecorationLocation, locations[i]});
    if (extra) inst(w, kOpDecorate, {10, extra});
    if (primitive_id) inst(w, kOpDecorate, {30, kDecorationBuiltIn, kBuiltInPrimitiveId});
    inst(w, kOpTypePointer, {5, kStorageInput, 4});
    for (std::size_t i = 0; i < locations.size(); ++i)
        inst(w, kOpVariable, {5, std::uint32_t(10 + i), kStorageInput});
    if (primitive_id) inst(w, kOpVariable, {5, 30, kStorageInput});
    return w;
}

bool write(const std::string& dir, const char* name, const std::vector<std::uint32_t>& words) {
    std::ofstream f(dir + "/" + name + ".spv", std::ios::binary);
    f.write(reinterpret_cast<const char*>(words.data()), std::streamsize(words.size() * 4));
    return bool(f);
}

}  // namespace

int main(int argc, char** argv) {
    // Two pass-through inputs and the barycentric one between them.
    const std::uint32_t plain_locs[] = {2, 0, 2};  // unsorted with a duplicate: sorted and unique in the module
    const auto plain = bary_fallback_gs(plain_locs, 1);
    CHECK(!plain.empty());
    CHECK(plain[3] > 1);  // the id bound
    const Summary s = summarize(plain);
    CHECK(s.ok);
    CHECK(s.capabilities == std::set<std::uint32_t>{kCapabilityGeometry});
    CHECK(s.model == kExecModelGeometry);
    CHECK(s.interface_count == 2 + 2 * 2 + 1);  // Position in/out, two locations in/out, the barycentric output
    CHECK((s.modes == std::set<std::uint32_t>{kModeInvocations, kModeTriangles, kModeOutputVertices,
                                              kModeOutputTriangleStrip}));
    CHECK(s.builtins_in == std::set<std::uint32_t>{kBuiltInPosition});
    CHECK(s.builtins_out == std::set<std::uint32_t>{kBuiltInPosition});
    CHECK((s.locations_in == std::set<std::uint32_t>{0, 2}));
    CHECK((s.locations_out == std::set<std::uint32_t>{0, 2}));
    CHECK(s.bary_location == 1);
    CHECK(s.vertex_emits == 3);
    CHECK((s.composites == std::vector<std::vector<std::uint32_t>>{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}));

    // No pass-through input at all: Position and the barycentrics only.
    const auto bare = bary_fallback_gs({}, 0);
    CHECK(!bare.empty() && summarize(bare).interface_count == 3);

    // Distance arrays, alone and together.
    const std::uint32_t one_loc[] = {0};
    const auto clipped = bary_fallback_gs(one_loc, 1, 2, 0);
    const Summary c = summarize(clipped);
    CHECK(c.ok && c.capabilities.count(kCapabilityClipDistance) && !c.capabilities.count(kCapabilityCullDistance));
    CHECK(c.builtins_in.count(kBuiltInClipDistance) && c.builtins_out.count(kBuiltInClipDistance));
    const auto both = bary_fallback_gs(one_loc, 1, 3, 8);
    const Summary b = summarize(both);
    CHECK(b.ok && b.capabilities.count(kCapabilityClipDistance) && b.capabilities.count(kCapabilityCullDistance));
    CHECK(b.builtins_out.count(kBuiltInClipDistance) && b.builtins_out.count(kBuiltInCullDistance));
    CHECK(b.interface_count == 2 + 2 + 1 + 2 * 2);

    // Errors.
    std::string error;
    const std::uint32_t colliding[] = {0, 1};
    CHECK(bary_fallback_gs(colliding, 1, 0, 0, &error).empty() &&
          error.find("also a pass-through") != std::string::npos);
    CHECK(bary_fallback_gs(one_loc, 1, 9, 0, &error).empty() && error.find("longer than 8") != std::string::npos);

    // bary_fallback_gs_for: the locations come from the PS module. The index says input 0, but the linked module
    // reads location 5 (as after link_f16_halves relocated it) and the barycentrics at 3.
    ShaderEntry vs, ps;
    vs.stage = "vs";
    ps.stage = "ps";
    ps.inputs = {0};
    ps.bary_fallback_location = 3;
    const auto linked = bary_fallback_gs_for(vs, ps, ps_module({5, 3}), &error);
    const Summary l = summarize(linked);
    CHECK(l.ok);
    CHECK(l.locations_in == std::set<std::uint32_t>{5});
    CHECK(l.bary_location == 3);
    vs.clip_distance_count = 4;
    CHECK(summarize(bary_fallback_gs_for(vs, ps, ps_module({5, 3}))).builtins_out.count(kBuiltInClipDistance));
    vs.clip_distance_count = 0;
    // A PS that reads PrimitiveId gets it from the GS (VUID-VkGraphicsPipelineCreateInfo-pStages-00740); one that
    // doesn't, doesn't.
    CHECK(!l.builtins_in.count(kBuiltInPrimitiveId) && !l.builtins_out.count(kBuiltInPrimitiveId));
    const auto with_id = bary_fallback_gs_for(vs, ps, ps_module({5, 3}, 0, true));
    const Summary p = summarize(with_id);
    CHECK(p.ok && p.builtins_in.count(kBuiltInPrimitiveId) && p.builtins_out.count(kBuiltInPrimitiveId));
    CHECK(p.interface_count == l.interface_count + 2);
    CHECK(p.locations_in == std::set<std::uint32_t>{5} && p.bary_location == 3);
    // Refusals.
    CHECK(bary_fallback_gs_for(vs, ps, ps_module({5}), &error).empty() &&
          error.find("no input at") != std::string::npos);
    CHECK(bary_fallback_gs_for(vs, ps, ps_module({5, 3}, kDecorationPerVertexKHR), &error).empty() &&
          error.find("PerVertexKHR") != std::string::npos);
    vs.writes_layer_or_viewport = true;
    CHECK(bary_fallback_gs_for(vs, ps, ps_module({5, 3}), &error).empty() &&
          error.find("Layer") != std::string::npos);
    vs.writes_layer_or_viewport = false;
    ShaderEntry plain_ps = ps;
    plain_ps.bary_fallback_location.reset();
    CHECK(bary_fallback_gs_for(vs, plain_ps, ps_module({5, 3}), &error).empty());
    CHECK(bary_fallback_gs_for(vs, ps, std::vector<std::uint32_t>{1, 2, 3}, &error).empty() &&
          error.find("malformed") != std::string::npos);

    if (argc > 1) {
        CHECK(write(argv[1], "plain", plain));
        CHECK(write(argv[1], "bare", bare));
        CHECK(write(argv[1], "clip", clipped));
        CHECK(write(argv[1], "clip_cull", both));
        CHECK(write(argv[1], "primitive_id", with_id));
    }
    std::printf("bary_fallback_gs: interface, one-hot constants, clip/cull forwarding, PS-module locations and the "
                "refusals all as expected\n");
    return 0;
}
