#pragma once

#include "spirv_f16_halves.h"  // shared SPIR-V opcodes and storage classes

#include <algorithm>
#include <array>
#include <bitset>
#include <cstring>
#include <map>
#include <set>
#include <span>
#include <vector>

namespace leonida {
namespace mesh_input_detail {

constexpr unsigned max_params = 32;
using LocationWords = std::array<std::size_t, max_params>;

// Find direct Location literals on the entry point's ordinary per-vertex float vec4 parameters.
// The mesh has arrays of vec4. A fragment reads a vec4 directly or indexes one per triangle vertex
// (PerVertexKHR). Both address one varying location. Built-ins have no Location.
inline bool locations(std::span<const std::uint32_t> words, bool mesh, LocationWords& result) {
    using namespace spirv_detail;
    constexpr unsigned mesh_ext = 5365, op_type_array = 28, op_constant = 43;
    constexpr unsigned component = 31, index = 32, per_primitive = 5271, per_vertex = 5285;
    if (words.size() < 5 || words[0] != kSpirvMagic) return false;
    const unsigned storage = mesh ? kStorageOutput : kStorageInput;
    std::set<unsigned> interfaces, floats, vectors, arrays, pointers, unsupported;
    std::set<unsigned> threes, triangle_arrays, triangle_pointers, vertex_indexed;
    std::map<unsigned, std::size_t> decorations;
    bool entry = false;
    for (std::size_t at = 5; at < words.size();) {
        const auto size = words[at] >> 16;
        if (!size || size > words.size() - at) return false;
        const auto ins = words.subspan(at, size);
        switch (std::uint16_t(ins[0])) {
        case kOpFunction:
            return entry;  // globals and annotations precede function bodies
        case kOpEntryPoint:
            // The runtime requests "main". A shared module with other entry points needs specialization.
            if (entry || size < 5 || ins[1] != (mesh ? mesh_ext : kExecModelFragment) ||
                ins[3] != 0x6e69616du || ins[4] != 0) return false;
            entry = true;
            interfaces.insert(ins.begin() + 5, ins.end());
            break;
        case kOpDecorate:
            if (size < 3) return false;
            if (ins[2] == kDecorationLocation) {
                if (size != 4 || !decorations.emplace(ins[1], at + 3).second) return false;
            } else if (ins[2] == per_vertex) {
                vertex_indexed.insert(ins[1]);
            } else if (ins[2] == per_primitive || ins[2] == component || ins[2] == index) {
                unsupported.insert(ins[1]);
            }
            break;
        case kOpMemberDecorate:
            if (size < 4 || ins[3] == kDecorationLocation) return false;
            break;
        case kOpDecorationGroup:
            return false;
        case kOpTypeFloat:
            if (size == 3 && ins[2] == 32) floats.insert(ins[1]);
            break;
        case kOpTypeVector:
            if (size == 4 && floats.contains(ins[2]) && ins[3] == 4) vectors.insert(ins[1]);
            break;
        case op_constant:
            if (size == 4 && ins[3] == 3) threes.insert(ins[2]);
            break;
        case op_type_array:
            if (size == 4 && vectors.contains(ins[2])) {
                arrays.insert(ins[1]);
                if (threes.contains(ins[3])) triangle_arrays.insert(ins[1]);
            }
            break;
        case kOpTypePointer:
            if (size == 4 && ins[2] == storage) {
                if ((mesh ? arrays : vectors).contains(ins[3])) pointers.insert(ins[1]);
                if (!mesh && triangle_arrays.contains(ins[3])) triangle_pointers.insert(ins[1]);
            }
            break;
        case kOpVariable:
            if (size < 4) return false;
            if (ins[3] == storage && interfaces.contains(ins[2])) {
                const auto d = decorations.find(ins[2]);
                if (d == decorations.end()) break;
                const auto slot = words[d->second];
                const auto& types = vertex_indexed.contains(ins[2]) ? triangle_pointers : pointers;
                if (slot >= result.size() || result[slot] || !types.contains(ins[1]) ||
                    unsupported.contains(ins[2])) return false;
                result[slot] = d->second;
            }
            break;
        }
        at += size;
    }
    return false;
}

}  // namespace mesh_input_detail

// Link only the fragment words owned by this PSO, after looking up both submitted FSSM programs.
// Missing pixel tables and identity maps require no edits. False leaves the original module intact.
inline bool link_mesh_inputs(std::span<const std::uint8_t> mesh_fssm, std::span<const std::uint8_t> pixel_fssm,
                             std::span<const std::uint32_t> mesh_words, std::vector<std::uint32_t>& pixel_words) {
    using namespace mesh_input_detail;
    constexpr std::size_t pixel_table = 0x138, mesh_table = 0x1dc, mesh_count = 0x29c;
    static_assert(mesh_table + 6 * max_params == mesh_count);
    constexpr unsigned kind = 6;
    if (pixel_fssm.size() <= pixel_table || pixel_fssm[pixel_table] != kind) return true;
    if (std::memcmp(pixel_fssm.data(), "FSSM", 4) != 0) return false;

    std::array<unsigned, max_params> slots{}, keys{};
    std::bitset<max_params> requested;
    unsigned inputs = 0;
    bool changed = false;
    for (std::size_t at = pixel_table; at < pixel_fssm.size() && pixel_fssm[at] == kind; at += 8) {
        if (inputs == max_params || pixel_fssm.size() - at < 8 || pixel_fssm[at + 3] != inputs) return false;
        const unsigned stored = pixel_fssm[at + 6];
        if (!stored || stored > max_params || requested[stored - 1]) return false;
        slots[inputs] = stored - 1;
        keys[inputs] = pixel_fssm[at + 1];
        requested.set(stored - 1);
        changed |= slots[inputs] != inputs;
        ++inputs;
    }
    if (!changed) return true;
    // A dense permutation could already be linked. A sparse map has at least one slot above the dense range.
    if (std::none_of(slots.begin(), slots.begin() + inputs, [&](unsigned slot) { return slot >= inputs; }))
        return false;
    if (mesh_fssm.size() < mesh_count + 4 || std::memcmp(mesh_fssm.data(), "FSSM", 4) != 0) return false;
    std::uint32_t exports;
    std::memcpy(&exports, mesh_fssm.data() + mesh_count, sizeof(exports));
    if (!exports || exports > max_params) return false;
    std::bitset<max_params> produced;
    std::array<unsigned, 256> by_key;
    by_key.fill(max_params);
    for (unsigned i = 0; i < exports; ++i) {
        const auto row = mesh_fssm.subspan(mesh_table + 6 * i, 6);
        const unsigned key = row[1], slot = row[2];
        if (row[0] != kind || slot >= max_params || produced[slot] || by_key[key] != max_params) return false;
        produced.set(slot);
        by_key[key] = slot;
    }
    for (unsigned i = 0; i < inputs; ++i)
        if (by_key[keys[i]] != slots[i]) return false;

    LocationWords ms{}, ps{};
    if (!locations(mesh_words, true, ms) || !locations(pixel_words, false, ps)) return false;
    for (unsigned i = 0; i < max_params; ++i)
        if (bool(ms[i]) != produced[i] || bool(ps[i]) != (i < inputs)) return false;
    for (unsigned i = 0; i < inputs; ++i) pixel_words[ps[i]] = slots[i];
    return true;
}

}  // namespace leonida
