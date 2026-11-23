#pragma once

// The pass-through geometry shader a `--bary-fallback` pixel shader is drawn with (SHADER-98, #804).
//
// On a device without VK_KHR_fragment_shader_barycentric, `rdna2spirv.py --bary-fallback` declares the raw
// barycentrics a PS reads as an ordinary vec3 input, at the location its index entry gives as "bary_fallback_input".
// Nothing writes that input unless the pipeline has a stage that does: this header builds that stage for a graphics
// PSO, the module tools/bary_fallback_gs.py builds for the GPU tests. The GS copies Position, the PS's vec4 inputs
// and the VS's ClipDistance/CullDistance arrays from each vertex unchanged, and writes (1,0,0), (0,1,0), (0,0,1) at
// the triangle's three vertices. Default (perspective) interpolation of those is the value BaryCoordKHR has.
//
// It is built at PSO creation because only then are both shaders known: the pass-through locations are the linked
// PS's (link_f16_halves may have moved some), and the distance arrays are the VS's.
//
// Pure C++20 stdlib, no Vulkan or Windows headers required.

#include "shader_library.h"
#include "spirv_f16_halves.h"

#include <algorithm>
#include <cstdint>
#include <initializer_list>
#include <span>
#include <string>
#include <vector>

namespace leonida {

namespace bary_gs_detail {

// Opcodes, enums and decorations this module needs beyond spirv_detail's.
constexpr std::uint16_t kOpMemoryModel = 14, kOpEntryPoint = 15, kOpExecutionMode = 16, kOpCapability = 17,
                        kOpTypeVoid = 19, kOpTypeInt = 21, kOpTypeFloat = 22, kOpTypeVector = 23, kOpTypeArray = 28,
                        kOpTypePointer = 32, kOpTypeFunction = 33, kOpConstant = 43, kOpConstantComposite = 44,
                        kOpFunction = 54, kOpFunctionEnd = 56, kOpVariable = 59, kOpLoad = 61, kOpStore = 62,
                        kOpAccessChain = 65, kOpDecorate = 71, kOpEmitVertex = 218, kOpEndPrimitive = 219,
                        kOpLabel = 248, kOpReturn = 253;
constexpr std::uint32_t kCapabilityGeometry = 2, kCapabilityClipDistance = 32, kCapabilityCullDistance = 33;
constexpr std::uint32_t kExecModelGeometry = 3;
constexpr std::uint32_t kModeInvocations = 0, kModeTriangles = 22, kModeOutputVertices = 26,
                        kModeOutputTriangleStrip = 29;
constexpr std::uint32_t kDecorationBuiltIn = 11, kDecorationLocation = 30, kDecorationPerVertexKHR = 5285;
constexpr std::uint32_t kBuiltInPosition = 0, kBuiltInClipDistance = 3, kBuiltInCullDistance = 4;
constexpr std::uint32_t kStorageInput = 1, kStorageOutput = 3;

class Writer {
public:
    std::vector<std::uint32_t> words{0x07230203u, 0x00010000u, 0, 0, 0};  // SPIR-V 1.0; bound set by finish()

    std::uint32_t id() { return next_++; }
    void op(std::uint16_t opcode, std::initializer_list<std::uint32_t> operands) {
        words.push_back((std::uint32_t(1 + operands.size()) << 16) | opcode);
        words.insert(words.end(), operands.begin(), operands.end());
    }
    void op(std::uint16_t opcode, const std::vector<std::uint32_t>& operands) {
        words.push_back((std::uint32_t(1 + operands.size()) << 16) | opcode);
        words.insert(words.end(), operands.begin(), operands.end());
    }
    std::vector<std::uint32_t> finish() {
        words[3] = next_;
        return std::move(words);
    }

private:
    std::uint32_t next_ = 1;
};

}  // namespace bary_gs_detail

// The geometry shader's SPIR-V words: Position and the vec4 inputs at `locations` copied per vertex, the one-hot
// barycentric vec3 at `bary_location`, and ClipDistance/CullDistance arrays of these lengths (0: none) copied too.
// Empty, with *error set, when the bary location is also a pass-through location or a length is over 8.
inline std::vector<std::uint32_t> bary_fallback_gs(std::span<const std::uint32_t> locations,
                                                   std::uint32_t bary_location, std::uint32_t clip_count = 0,
                                                   std::uint32_t cull_count = 0, std::string* error = nullptr) {
    using namespace bary_gs_detail;
    const auto fail = [&](const std::string& why) {
        if (error) *error = why;
        return std::vector<std::uint32_t>{};
    };
    std::vector<std::uint32_t> locs(locations.begin(), locations.end());
    std::sort(locs.begin(), locs.end());
    locs.erase(std::unique(locs.begin(), locs.end()), locs.end());
    if (std::binary_search(locs.begin(), locs.end(), bary_location))
        return fail("--bary-fallback: the barycentric location " + std::to_string(bary_location) +
                    " is also a pass-through input");
    if (clip_count > 8 || cull_count > 8) return fail("--bary-fallback: a clip or cull distance array longer than 8");

    Writer w;
    const std::uint32_t t_void = w.id(), t_fn = w.id(), t_uint = w.id(), t_int = w.id(), t_float = w.id(),
                        t_v3 = w.id(), t_v4 = w.id(), c_uint3 = w.id(), t_arr_v4 = w.id(), p_in_arr_v4 = w.id(),
                        p_in_v4 = w.id(), p_out_v4 = w.id(), p_out_v3 = w.id(), c_f0 = w.id(), c_f1 = w.id();
    std::uint32_t c_bary[3], c_int[3];
    for (std::uint32_t& c : c_bary) c = w.id();
    for (std::uint32_t& c : c_int) c = w.id();

    // A ClipDistance or CullDistance array: float[count] per vertex in, one out.
    struct Distance {
        std::uint32_t count, builtin, capability;
        std::uint32_t c_count = 0, t_arr = 0, t_triangle = 0, p_in_triangle = 0, p_in_arr = 0, p_out_arr = 0,
                      in = 0, out = 0;
    };
    std::vector<Distance> distances;
    if (clip_count) distances.push_back({clip_count, kBuiltInClipDistance, kCapabilityClipDistance});
    if (cull_count) distances.push_back({cull_count, kBuiltInCullDistance, kCapabilityCullDistance});
    for (Distance& d : distances)
        for (std::uint32_t* p : {&d.c_count, &d.t_arr, &d.t_triangle, &d.p_in_triangle, &d.p_in_arr, &d.p_out_arr,
                                 &d.in, &d.out})
            *p = w.id();

    const std::uint32_t in_pos = w.id(), out_pos = w.id();
    std::vector<std::uint32_t> ins, outs;
    for (std::size_t i = 0; i < locs.size(); ++i) {
        ins.push_back(w.id());
        outs.push_back(w.id());
    }
    const std::uint32_t out_bary = w.id(), fn_main = w.id(), entry = w.id();

    w.op(kOpCapability, {kCapabilityGeometry});
    for (const Distance& d : distances) w.op(kOpCapability, {d.capability});
    w.op(kOpMemoryModel, {0 /* Logical */, 1 /* GLSL450 */});
    std::vector<std::uint32_t> entry_point{kExecModelGeometry, fn_main, 0x6e69616du /* "main" */, 0, in_pos, out_pos};
    entry_point.insert(entry_point.end(), ins.begin(), ins.end());
    entry_point.insert(entry_point.end(), outs.begin(), outs.end());
    entry_point.push_back(out_bary);
    for (const Distance& d : distances) {
        entry_point.push_back(d.in);
        entry_point.push_back(d.out);
    }
    w.op(kOpEntryPoint, entry_point);
    w.op(kOpExecutionMode, {fn_main, kModeTriangles});
    w.op(kOpExecutionMode, {fn_main, kModeInvocations, 1});
    w.op(kOpExecutionMode, {fn_main, kModeOutputTriangleStrip});
    w.op(kOpExecutionMode, {fn_main, kModeOutputVertices, 3});

    w.op(kOpDecorate, {in_pos, kDecorationBuiltIn, kBuiltInPosition});
    w.op(kOpDecorate, {out_pos, kDecorationBuiltIn, kBuiltInPosition});
    for (const Distance& d : distances) {
        w.op(kOpDecorate, {d.in, kDecorationBuiltIn, d.builtin});
        w.op(kOpDecorate, {d.out, kDecorationBuiltIn, d.builtin});
    }
    for (std::size_t i = 0; i < locs.size(); ++i) w.op(kOpDecorate, {ins[i], kDecorationLocation, locs[i]});
    for (std::size_t i = 0; i < locs.size(); ++i) w.op(kOpDecorate, {outs[i], kDecorationLocation, locs[i]});
    w.op(kOpDecorate, {out_bary, kDecorationLocation, bary_location});

    w.op(kOpTypeVoid, {t_void});
    w.op(kOpTypeFunction, {t_fn, t_void});
    w.op(kOpTypeInt, {t_uint, 32, 0});
    w.op(kOpTypeInt, {t_int, 32, 1});
    w.op(kOpTypeFloat, {t_float, 32});
    w.op(kOpTypeVector, {t_v3, t_float, 3});
    w.op(kOpTypeVector, {t_v4, t_float, 4});
    w.op(kOpConstant, {t_uint, c_uint3, 3});
    w.op(kOpTypeArray, {t_arr_v4, t_v4, c_uint3});
    w.op(kOpTypePointer, {p_in_arr_v4, kStorageInput, t_arr_v4});
    w.op(kOpTypePointer, {p_in_v4, kStorageInput, t_v4});
    w.op(kOpTypePointer, {p_out_v4, kStorageOutput, t_v4});
    w.op(kOpTypePointer, {p_out_v3, kStorageOutput, t_v3});
    for (const Distance& d : distances) {
        w.op(kOpConstant, {t_uint, d.c_count, d.count});
        w.op(kOpTypeArray, {d.t_arr, t_float, d.c_count});
        w.op(kOpTypeArray, {d.t_triangle, d.t_arr, c_uint3});
        w.op(kOpTypePointer, {d.p_in_triangle, kStorageInput, d.t_triangle});
        w.op(kOpTypePointer, {d.p_in_arr, kStorageInput, d.t_arr});
        w.op(kOpTypePointer, {d.p_out_arr, kStorageOutput, d.t_arr});
    }
    w.op(kOpConstant, {t_float, c_f0, 0x00000000u});
    w.op(kOpConstant, {t_float, c_f1, 0x3f800000u});
    for (std::uint32_t v = 0; v < 3; ++v)
        w.op(kOpConstantComposite, {t_v3, c_bary[v], v == 0 ? c_f1 : c_f0, v == 1 ? c_f1 : c_f0, v == 2 ? c_f1 : c_f0});
    for (std::uint32_t v = 0; v < 3; ++v) w.op(kOpConstant, {t_int, c_int[v], v});

    w.op(kOpVariable, {p_in_arr_v4, in_pos, kStorageInput});
    w.op(kOpVariable, {p_out_v4, out_pos, kStorageOutput});
    for (const std::uint32_t in : ins) w.op(kOpVariable, {p_in_arr_v4, in, kStorageInput});
    for (const std::uint32_t out : outs) w.op(kOpVariable, {p_out_v4, out, kStorageOutput});
    w.op(kOpVariable, {p_out_v3, out_bary, kStorageOutput});
    for (const Distance& d : distances) {
        w.op(kOpVariable, {d.p_in_triangle, d.in, kStorageInput});
        w.op(kOpVariable, {d.p_out_arr, d.out, kStorageOutput});
    }

    w.op(kOpFunction, {t_void, fn_main, 0 /* None */, t_fn});
    w.op(kOpLabel, {entry});
    for (std::uint32_t v = 0; v < 3; ++v) {
        const auto copy = [&](std::uint32_t pointer_type, std::uint32_t value_type, std::uint32_t src,
                              std::uint32_t dst) {
            const std::uint32_t p = w.id(), x = w.id();
            w.op(kOpAccessChain, {pointer_type, p, src, c_int[v]});
            w.op(kOpLoad, {value_type, x, p});
            w.op(kOpStore, {dst, x});
        };
        copy(p_in_v4, t_v4, in_pos, out_pos);
        for (std::size_t i = 0; i < locs.size(); ++i) copy(p_in_v4, t_v4, ins[i], outs[i]);
        for (const Distance& d : distances) copy(d.p_in_arr, d.t_arr, d.in, d.out);
        w.op(kOpStore, {out_bary, c_bary[v]});
        w.op(kOpEmitVertex, {});
    }
    w.op(kOpEndPrimitive, {});
    w.op(kOpReturn, {});
    w.op(kOpFunctionEnd, {});
    return w.finish();
}

// The geometry shader for a PSO's VS and its --bary-fallback PS: the pass-through locations are every located input
// the PS module declares except the barycentric one (so after link_f16_halves, those it links), the distance arrays
// the VS's. Empty, with *error set, when the pair can't be drawn this way: the PS has no barycentric input, or reads
// PerVertexKHR inputs (which only the extension can feed), or the VS writes Layer or ViewportIndex (a geometry stage
// can't read those to forward them).
inline std::vector<std::uint32_t> bary_fallback_gs_for(const ShaderEntry& vs, const ShaderEntry& ps,
                                                       std::span<const std::uint32_t> ps_words,
                                                       std::string* error = nullptr) {
    const auto fail = [&](const std::string& why) {
        if (error) *error = why;
        return std::vector<std::uint32_t>{};
    };
    if (!ps.bary_fallback_location) return fail("--bary-fallback: the PS has no bary_fallback_input");
    if (vs.writes_layer_or_viewport)
        return fail("--bary-fallback: the VS writes Layer or ViewportIndex, which a pass-through GS can't forward");
    spirv_detail::Module module;
    if (!module.parse(ps_words)) return fail("--bary-fallback: malformed PS SPIR-V");
    for (const spirv_detail::Inst& inst : module.insts)
        if (inst.opcode == bary_gs_detail::kOpDecorate && inst.words.size() >= 3 &&
            inst.words[2] == bary_gs_detail::kDecorationPerVertexKHR)
            return fail("--bary-fallback: the PS reads PerVertexKHR inputs");
    const std::uint32_t bary = *ps.bary_fallback_location;
    const auto inputs = spirv_detail::collect_located_vars(module, spirv_detail::kStorageInput);
    if (!inputs.count(bary))
        return fail("--bary-fallback: the PS declares no input at its bary_fallback_input location " +
                    std::to_string(bary));
    std::vector<std::uint32_t> locations;
    for (const auto& [location, var] : inputs)
        if (location != bary) locations.push_back(location);
    return bary_fallback_gs(locations, bary, vs.clip_distance_count, vs.cull_distance_count, error);
}

}  // namespace leonida
