#pragma once

// The vertex shader that draws what a `--mesh-compute` mesh shader wrote (rdna2spirv.py, #1053).
//
// On a device without VK_EXT_mesh_shader, `rdna2spirv.py --mesh-compute` turns each mesh shader into a compute shader
// with the same workgroup size. Each workgroup writes its vertex and primitive counts, its vertices and its
// primitives into a slot of an output buffer whose device address is in push-constant dwords 36..37 (the entry's
// "mesh_compute" layout, MeshComputeLayout). This header builds the vertex shader that draws the slots, the module
// tools/mesh_compute_vs.py writes for the GPU tests, instruction for instruction:
//   - the draw is non-indexed, vkCmdDraw(groups * max_primitives * 3): vertex v is workgroup v / (3 * max_primitives),
//     primitive (v / 3) % max_primitives, corner v % 3;
//   - a primitive past the workgroup's count, or one marked culled, puts the same position outside the clip volume at
//     all three corners, so it is clipped away;
//   - otherwise the corner's vertex index (clamped to max_vertices - 1) selects the vertex whose position, clip and
//     cull distances and params go out, at the mesh shader's locations, so the pixel shader is unchanged; with
//     --f16-interp-fallback each f16_halves param's halves also go out decoded to f32 at their two locations.
//
// Pure C++20 stdlib, no Vulkan or Windows headers required.

#include "shader_library.h"

#include <cstdint>
#include <initializer_list>
#include <map>
#include <set>
#include <string>
#include <string_view>
#include <vector>

namespace leonida {

namespace mesh_vs_detail {

constexpr std::uint16_t kOpExtension = 10, kOpExtInstImport = 11, kOpExtInst = 12, kOpMemoryModel = 14,
                        kOpEntryPoint = 15, kOpCapability = 17, kOpTypeVoid = 19, kOpTypeBool = 20, kOpTypeInt = 21,
                        kOpTypeFloat = 22, kOpTypeVector = 23, kOpTypeArray = 28, kOpTypeRuntimeArray = 29,
                        kOpTypeStruct = 30, kOpTypePointer = 32, kOpTypeFunction = 33, kOpConstant = 43,
                        kOpConstantComposite = 44, kOpFunction = 54, kOpFunctionEnd = 56, kOpVariable = 59,
                        kOpLoad = 61, kOpStore = 62, kOpAccessChain = 65, kOpDecorate = 71, kOpMemberDecorate = 72,
                        kOpCompositeConstruct = 80, kOpCompositeExtract = 81, kOpCopyObject = 83, kOpUConvert = 113,
                        kOpConvertUToPtr = 120, kOpBitcast = 124, kOpIAdd = 128, kOpIMul = 132, kOpUDiv = 134,
                        kOpUMod = 137, kOpLogicalAnd = 167, kOpLogicalNot = 168, kOpSelect = 169, kOpINotEqual = 171,
                        kOpULessThan = 176, kOpShiftLeftLogical = 196, kOpBitwiseOr = 197, kOpLabel = 248,
                        kOpReturn = 253;
constexpr std::uint32_t kCapShader = 1, kCapInt64 = 11, kCapClipDistance = 32, kCapCullDistance = 33,
                        kCapPhysicalStorageBufferAddresses = 5347;
constexpr std::uint32_t kAddressingPhysicalStorageBuffer64 = 5348, kMemoryGlsl450 = 1, kExecModelVertex = 0;
constexpr std::uint32_t kDecBlock = 2, kDecArrayStride = 6, kDecBuiltIn = 11, kDecLocation = 30, kDecOffset = 35;
constexpr std::uint32_t kBuiltInPosition = 0, kBuiltInClipDistance = 3, kBuiltInCullDistance = 4,
                        kBuiltInVertexIndex = 42;
constexpr std::uint32_t kStorageInput = 1, kStorageOutput = 3, kStoragePushConstant = 9,
                        kStoragePhysicalStorageBuffer = 5349;
constexpr std::uint32_t kMemoryAligned = 2;
constexpr std::uint32_t kGlslUMin = 38, kGlslUnpackHalf2x16 = 62;
constexpr std::uint32_t kUserSgprs = 32;  // s0..s31, then heap base, code base and the output address

inline void op(std::vector<std::uint32_t>& w, std::uint16_t opcode, std::initializer_list<std::uint32_t> operands) {
    w.push_back((std::uint32_t(1 + operands.size()) << 16) | opcode);
    w.insert(w.end(), operands.begin(), operands.end());
}

inline void op(std::vector<std::uint32_t>& w, std::uint16_t opcode, const std::vector<std::uint32_t>& operands) {
    w.push_back((std::uint32_t(1 + operands.size()) << 16) | opcode);
    w.insert(w.end(), operands.begin(), operands.end());
}

// A literal string's words: UTF-8, nul-terminated, padded with zeros to a whole word.
inline std::vector<std::uint32_t> literal(std::string_view s) {
    std::vector<std::uint32_t> w(s.size() / 4 + 1, 0);
    for (std::size_t i = 0; i < s.size(); ++i) w[i / 4] |= std::uint32_t(std::uint8_t(s[i])) << (8 * (i % 4));
    return w;
}

}  // namespace mesh_vs_detail

// The vertex shader's SPIR-V words for this layout, or empty with *error set: a layout without vertices or
// primitives, one whose address is in the --pc-tail-ubo block, an f16_halves param that is not a layout param, or
// a half location that collides with another output.
inline std::vector<std::uint32_t> mesh_compute_vs(const MeshComputeLayout& lay, std::string* error = nullptr) {
    using namespace mesh_vs_detail;
    const auto fail = [&](const std::string& why) {
        if (error) *error = why;
        return std::vector<std::uint32_t>{};
    };
    const std::uint32_t maxv = lay.max_vertices, maxp = lay.max_primitives, clip = lay.clip_count,
                        cull = lay.cull_count;
    if (!maxv || !maxp || !lay.slot_bytes) return fail("mesh_compute: a layout without vertices or primitives");
    if (lay.address_in_pc_tail)
        return fail("mesh_compute: the output address is in the --pc-tail-ubo block, which this vertex shader does "
                    "not read");
    if (clip > 8 || cull > 8) return fail("mesh_compute: more than 8 clip or cull distances");
    std::set<std::uint32_t> used(  // output locations: the params, then each half's two
        [&] { std::set<std::uint32_t> s; for (const auto& [loc, off] : lay.params) s.insert(loc); return s; }());
    std::vector<std::uint32_t> half_locations;
    for (const auto& [loc, h] : lay.f16_halves) {
        if (!lay.params.count(loc))
            return fail("mesh_compute: f16_halves param " + std::to_string(loc) + " is not in the layout's params");
        if (h[0] == h[1]) return fail("mesh_compute: f16_halves param " + std::to_string(loc) + " has one location");
        for (const std::uint32_t hloc : h) {
            if (!used.insert(hloc).second)
                return fail("mesh_compute: f16_halves location " + std::to_string(hloc) +
                            " collides with another output");
            half_locations.push_back(hloc);
        }
    }

    std::uint32_t next = 1;
    const auto id = [&] { return next++; };
    // Types, the push-constant block, globals: fixed ids. Constants get theirs as the body asks for them.
    const std::uint32_t glsl = id(), t_void = id(), t_fn = id(), t_bool = id(), t_uint = id(), t_int = id(),
                        t_ulong = id(), t_float = id(), t_v4 = id(), t_v2 = id(), n_ud = id(), arr_ud = id(),
                        t_pc = id(), p_pc = id(), p_pc_uint = id(), rt_uint = id(), p_psb_uint = id(),
                        p_in_int = id(), p_out_v4 = id(), f1 = id(), f2 = id(), outside = id();
    const std::uint32_t arr_clip = clip ? id() : 0, p_out_clip = clip ? id() : 0;
    const std::uint32_t arr_cull = cull ? id() : 0, p_out_cull = cull ? id() : 0;
    const std::uint32_t vertex_index = id(), pc = id(), out_pos = id();
    const std::uint32_t out_clip = clip ? id() : 0, out_cull = cull ? id() : 0;
    std::map<std::uint32_t, std::uint32_t> out_at;  // output location -> variable
    for (const auto& [loc, off] : lay.params) out_at[loc] = id();
    for (const std::uint32_t hloc : half_locations) out_at[hloc] = id();
    const std::uint32_t fn_main = id(), entry = id();

    std::map<std::uint32_t, std::uint32_t> uints, ulongs;  // value -> constant id, emitted sorted by value
    const auto c = [&](std::uint32_t v) {
        auto [it, added] = uints.try_emplace(v, 0);
        if (added) it->second = id();
        return it->second;
    };
    const auto cl = [&](std::uint32_t v) {
        auto [it, added] = ulongs.try_emplace(v, 0);
        if (added) it->second = id();
        return it->second;
    };

    std::vector<std::uint32_t> body;
    const auto value = [&](std::uint16_t opcode, std::uint32_t type, std::initializer_list<std::uint32_t> operands) {
        const std::uint32_t r = id();
        std::vector<std::uint32_t> all{type, r};
        all.insert(all.end(), operands.begin(), operands.end());
        op(body, opcode, all);
        return r;
    };
    // The uint at slot byte offset `off` from the ulong address `base`.
    const auto load_word = [&](std::uint32_t base, std::uint32_t off) {
        const std::uint32_t a = off ? value(kOpIAdd, t_ulong, {base, cl(off)}) : value(kOpCopyObject, t_ulong, {base});
        const std::uint32_t p = value(kOpConvertUToPtr, p_psb_uint, {a});
        return value(kOpLoad, t_uint, {p, kMemoryAligned, 4});
    };
    const auto load_floats = [&](std::uint32_t base, std::uint32_t off, std::uint32_t count) {
        std::vector<std::uint32_t> out;
        for (std::uint32_t i = 0; i < count; ++i)
            out.push_back(value(kOpBitcast, t_float, {load_word(base, off + 4 * i)}));
        return out;
    };
    const auto construct = [&](std::uint32_t type, const std::vector<std::uint32_t>& parts) {
        const std::uint32_t r = id();
        std::vector<std::uint32_t> all{type, r};
        all.insert(all.end(), parts.begin(), parts.end());
        op(body, kOpCompositeConstruct, all);
        return r;
    };

    for (const std::uint32_t v : {0u, 1u, 3u, 5u, 6u}) c(v);  // as the Python module declares them
    if (clip) c(clip);
    if (cull) c(cull);
    const std::uint32_t vid = value(kOpLoad, t_int, {vertex_index});
    const std::uint32_t vidu = value(kOpBitcast, t_uint, {vid});
    const std::uint32_t per = value(kOpCopyObject, t_uint, {c(3 * maxp)});
    const std::uint32_t wg = value(kOpUDiv, t_uint, {vidu, per});
    const std::uint32_t rem = value(kOpUMod, t_uint, {vidu, per});
    const std::uint32_t prim = value(kOpUDiv, t_uint, {rem, c(3)});
    const std::uint32_t corner = value(kOpUMod, t_uint, {rem, c(3)});
    const std::uint32_t alo_p = value(kOpAccessChain, p_pc_uint, {pc, c(5)});
    const std::uint32_t ahi_p = value(kOpAccessChain, p_pc_uint, {pc, c(6)});
    const std::uint32_t alo = value(kOpLoad, t_uint, {alo_p});
    const std::uint32_t ahi = value(kOpLoad, t_uint, {ahi_p});
    const std::uint32_t alo64 = value(kOpUConvert, t_ulong, {alo});
    const std::uint32_t ahi64 = value(kOpUConvert, t_ulong, {ahi});
    const std::uint32_t ahis = value(kOpShiftLeftLogical, t_ulong, {ahi64, cl(32)});
    const std::uint32_t addr = value(kOpBitwiseOr, t_ulong, {ahis, alo64});
    const std::uint32_t wg64 = value(kOpUConvert, t_ulong, {wg});
    const std::uint32_t soff = value(kOpIMul, t_ulong, {wg64, cl(lay.slot_bytes)});
    const std::uint32_t slot = value(kOpIAdd, t_ulong, {addr, soff});
    const std::uint32_t pcount = load_word(slot, 4);
    const std::uint32_t inrange = value(kOpULessThan, t_bool, {prim, pcount});
    const std::uint32_t pofs = value(kOpIMul, t_uint, {prim, c(lay.primitive_stride)});
    const std::uint32_t pofs64 = value(kOpUConvert, t_ulong, {pofs});
    const std::uint32_t pbase0 = value(kOpIAdd, t_ulong, {slot, cl(lay.primitives)});
    const std::uint32_t pbase = value(kOpIAdd, t_ulong, {pbase0, pofs64});
    const std::uint32_t cofs = value(kOpIMul, t_uint, {corner, c(4)});
    const std::uint32_t cofs64 = value(kOpUConvert, t_ulong, {cofs});
    const std::uint32_t iaddr = value(kOpIAdd, t_ulong, {pbase, cofs64});
    const std::uint32_t iptr = value(kOpConvertUToPtr, p_psb_uint, {iaddr});
    const std::uint32_t idx0 = value(kOpLoad, t_uint, {iptr, kMemoryAligned, 4});
    const std::uint32_t idx = value(kOpExtInst, t_uint, {glsl, kGlslUMin, idx0, c(maxv - 1)});
    const std::uint32_t culled_w = load_word(pbase, 12);
    const std::uint32_t culled = value(kOpINotEqual, t_bool, {culled_w, c(0)});
    const std::uint32_t notculled = value(kOpLogicalNot, t_bool, {culled});
    const std::uint32_t valid = value(kOpLogicalAnd, t_bool, {inrange, notculled});
    const std::uint32_t vofs = value(kOpIMul, t_uint, {idx, c(lay.vertex_stride)});
    const std::uint32_t vofs64 = value(kOpUConvert, t_ulong, {vofs});
    const std::uint32_t vbase0 = value(kOpIAdd, t_ulong, {slot, cl(lay.vertices)});
    const std::uint32_t vbase = value(kOpIAdd, t_ulong, {vbase0, vofs64});
    const std::uint32_t pos = construct(t_v4, load_floats(vbase, 0, 4));
    const std::uint32_t pos_out = value(kOpSelect, t_v4, {valid, pos, outside});
    op(body, kOpStore, {out_pos, pos_out});
    if (clip) op(body, kOpStore, {out_clip, construct(arr_clip, load_floats(vbase, lay.clip_offset, clip))});
    if (cull) op(body, kOpStore, {out_cull, construct(arr_cull, load_floats(vbase, lay.cull_offset, cull))});
    for (const auto& [loc, off] : lay.params)
        op(body, kOpStore, {out_at[loc], construct(t_v4, load_floats(vbase, off, 4))});
    for (const auto& [loc, h] : lay.f16_halves) {  // each channel's two f16 halves decoded to f32
        std::vector<std::uint32_t> parts[2];
        for (std::uint32_t i = 0; i < 4; ++i) {
            const std::uint32_t w = load_word(vbase, lay.params.at(loc) + 4 * i);
            const std::uint32_t h2 = value(kOpExtInst, t_v2, {glsl, kGlslUnpackHalf2x16, w});
            for (std::uint32_t half = 0; half < 2; ++half)
                parts[half].push_back(value(kOpCompositeExtract, t_float, {h2, half}));
        }
        for (std::uint32_t half = 0; half < 2; ++half)
            op(body, kOpStore, {out_at[h[half]], construct(t_v4, parts[half])});
    }
    op(body, kOpReturn, {});
    op(body, kOpFunctionEnd, {});

    // The module, in the order the Python one has it.
    std::vector<std::uint32_t> w{0x07230203u, 0x00010500u, 0, 0, 0};  // SPIR-V 1.5; bound set at the end
    for (const std::uint32_t cap : {kCapShader, kCapInt64, kCapPhysicalStorageBufferAddresses})
        op(w, kOpCapability, {cap});
    if (clip) op(w, kOpCapability, {kCapClipDistance});
    if (cull) op(w, kOpCapability, {kCapCullDistance});
    op(w, kOpExtension, literal("SPV_KHR_physical_storage_buffer"));
    {
        std::vector<std::uint32_t> import{glsl};
        const auto name = literal("GLSL.std.450");
        import.insert(import.end(), name.begin(), name.end());
        op(w, kOpExtInstImport, import);
    }
    op(w, kOpMemoryModel, {kAddressingPhysicalStorageBuffer64, kMemoryGlsl450});
    // Output order: position, distances, the params, then the halves' locations in f16_halves order.
    std::vector<std::uint32_t> output_order{out_pos};
    if (clip) output_order.push_back(out_clip);
    if (cull) output_order.push_back(out_cull);
    std::vector<std::uint32_t> located;  // locations in declaration order
    for (const auto& [loc, off] : lay.params) located.push_back(loc);
    located.insert(located.end(), half_locations.begin(), half_locations.end());
    for (const std::uint32_t loc : located) output_order.push_back(out_at[loc]);
    {
        std::vector<std::uint32_t> ep{kExecModelVertex, fn_main};
        const auto name = literal("main");
        ep.insert(ep.end(), name.begin(), name.end());
        ep.push_back(vertex_index);
        ep.push_back(pc);
        ep.insert(ep.end(), output_order.begin(), output_order.end());
        op(w, kOpEntryPoint, ep);
    }
    op(w, kOpDecorate, {vertex_index, kDecBuiltIn, kBuiltInVertexIndex});
    op(w, kOpDecorate, {out_pos, kDecBuiltIn, kBuiltInPosition});
    if (clip) op(w, kOpDecorate, {out_clip, kDecBuiltIn, kBuiltInClipDistance});
    if (cull) op(w, kOpDecorate, {out_cull, kDecBuiltIn, kBuiltInCullDistance});
    for (const std::uint32_t loc : located) op(w, kOpDecorate, {out_at[loc], kDecLocation, loc});
    op(w, kOpDecorate, {arr_ud, kDecArrayStride, 4});
    op(w, kOpDecorate, {t_pc, kDecBlock});
    op(w, kOpMemberDecorate, {t_pc, 0, kDecOffset, 0});
    for (std::uint32_t k = 1; k <= 6; ++k)
        op(w, kOpMemberDecorate, {t_pc, k, kDecOffset, 4 * kUserSgprs + 4 * (k - 1)});
    op(w, kOpDecorate, {rt_uint, kDecArrayStride, 4});
    op(w, kOpTypeVoid, {t_void});
    op(w, kOpTypeFunction, {t_fn, t_void});
    op(w, kOpTypeBool, {t_bool});
    op(w, kOpTypeInt, {t_uint, 32, 0});
    op(w, kOpTypeInt, {t_int, 32, 1});
    op(w, kOpTypeInt, {t_ulong, 64, 0});
    op(w, kOpTypeFloat, {t_float, 32});
    op(w, kOpTypeVector, {t_v4, t_float, 4});
    op(w, kOpTypeVector, {t_v2, t_float, 2});
    for (const auto& [v, cid] : uints) op(w, kOpConstant, {t_uint, cid, v});
    for (const auto& [v, cid] : ulongs) op(w, kOpConstant, {t_ulong, cid, v, 0});  // low word, then high
    op(w, kOpConstant, {t_uint, n_ud, kUserSgprs});
    op(w, kOpTypeArray, {arr_ud, t_uint, n_ud});
    op(w, kOpTypeStruct, {t_pc, arr_ud, t_uint, t_uint, t_uint, t_uint, t_uint, t_uint});
    op(w, kOpTypePointer, {p_pc, kStoragePushConstant, t_pc});
    op(w, kOpTypePointer, {p_pc_uint, kStoragePushConstant, t_uint});
    op(w, kOpTypeRuntimeArray, {rt_uint, t_uint});
    op(w, kOpTypePointer, {p_psb_uint, kStoragePhysicalStorageBuffer, t_uint});
    op(w, kOpTypePointer, {p_in_int, kStorageInput, t_int});
    op(w, kOpTypePointer, {p_out_v4, kStorageOutput, t_v4});
    op(w, kOpConstant, {t_float, f1, 0x3f800000u});
    op(w, kOpConstant, {t_float, f2, 0x40000000u});
    op(w, kOpConstantComposite, {t_v4, outside, f2, f2, f2, f1});
    if (clip) {
        op(w, kOpTypeArray, {arr_clip, t_float, c(clip)});
        op(w, kOpTypePointer, {p_out_clip, kStorageOutput, arr_clip});
    }
    if (cull) {
        op(w, kOpTypeArray, {arr_cull, t_float, c(cull)});
        op(w, kOpTypePointer, {p_out_cull, kStorageOutput, arr_cull});
    }
    op(w, kOpVariable, {p_in_int, vertex_index, kStorageInput});
    op(w, kOpVariable, {p_pc, pc, kStoragePushConstant});
    op(w, kOpVariable, {p_out_v4, out_pos, kStorageOutput});
    if (clip) op(w, kOpVariable, {p_out_clip, out_clip, kStorageOutput});
    if (cull) op(w, kOpVariable, {p_out_cull, out_cull, kStorageOutput});
    for (const std::uint32_t loc : located) op(w, kOpVariable, {p_out_v4, out_at[loc], kStorageOutput});
    op(w, kOpFunction, {t_void, fn_main, 0 /* None */, t_fn});
    op(w, kOpLabel, {entry});
    w.insert(w.end(), body.begin(), body.end());
    w[3] = next;
    return w;
}

}  // namespace leonida
