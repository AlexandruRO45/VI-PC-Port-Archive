// Recognize a deliberately small subset of valid fragment SPIR-V for the 8x8 integer fixture.
// One set-0/binding-0 sampled 2D image, direct or array element zero. Every fetch must
// have constant in-bounds coordinates and Lod 0. Unknown instructions fail closed.
#pragma once

#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <unordered_set>
#include <vector>

inline const char* integer_fetch_type(const std::vector<uint32_t>& code) {
    if (code.size() < 5 || code[0] != 0x07230203) return nullptr;
    std::unordered_map<uint32_t, uint32_t> integers, constants, arrays, pointers;
    std::unordered_set<uint32_t> int_vec2, coords, bindings, sets, element_zero, loaded_images;
    uint32_t image = 0, variable = 0, variable_type = 0, entry_points = 0;
    const char* component = nullptr;
    bool has_shader_cap = false, has_memory_model = false, fetched = false;
    auto bounded = [&](uint32_t id, uint32_t limit) {
        auto it = constants.find(id);
        return it != constants.end() && it->second < limit;
    };
    for (std::size_t i = 5; i < code.size();) {
        const auto* w = code.data() + i;
        const uint32_t n = w[0] >> 16;
        if (!n || n > code.size() - i) return nullptr;
        switch (w[0] & 0xffff) {
        case 17: // Capability Shader (no physical pointers or runtime descriptor arrays)
            if (n != 2 || w[1] != 1) return nullptr;
            has_shader_cap = true;
            break;
        case 14: // MemoryModel Logical
            if (n != 3 || w[1] != 0 || has_memory_model) return nullptr;
            has_memory_model = true;
            break;
        case 15: // EntryPoint Fragment
            if (n < 4 || w[1] != 4 || entry_points) return nullptr;
            ++entry_points;
            break;
        case 71: // Decorate
            if (n < 3) return nullptr;
            if (w[2] == 33 || w[2] == 34) {
                if (n != 4 || w[3] != 0) return nullptr;
                (w[2] == 33 ? bindings : sets).insert(w[1]);
            }
            break;
        case 21: // TypeInt
            if (n != 4) return nullptr;
            if (w[2] == 32 && w[3] <= 1) integers[w[1]] = w[3];
            break;
        case 23: // TypeVector
            if (n != 4) return nullptr;
            if (integers.count(w[2]) && w[3] == 2) int_vec2.insert(w[1]);
            break;
        case 25: // TypeImage
            if (n != 9 || image || !integers.count(w[2]) || w[3] != 1 || w[4] ||
                w[5] || w[6] || w[7] != 1 || w[8]) return nullptr;
            image = w[1];
            component = integers.at(w[2]) ? "sint" : "uint";
            break;
        case 43: // Constant
            if (n < 4) return nullptr;
            if (n == 4 && integers.count(w[1])) constants[w[2]] = w[3];
            break;
        case 44: // ConstantComposite: 2-component 32-bit integer vector inside the sole 8x8 mip
            if (n < 3) return nullptr;
            if (n == 5 && int_vec2.count(w[1]) && bounded(w[3], 8) && bounded(w[4], 8)) coords.insert(w[2]);
            break;
        case 46: // ConstantNull: scalar 0 or zero 2D integer coordinate
            if (n != 3) return nullptr;
            if (integers.count(w[1])) constants[w[2]] = 0;
            else if (int_vec2.count(w[1])) coords.insert(w[2]);
            break;
        case 28: // TypeArray
            if (n != 4) return nullptr;
            if (constants.count(w[3]) && constants.at(w[3]) == 1) arrays[w[1]] = w[2];
            break;
        case 32: // TypePointer
            if (n != 4) return nullptr;
            if (w[2] == 0) pointers[w[1]] = w[3];
            break;
        case 59: // Variable
            if (n < 4) return nullptr;
            if (w[3] == 0) {
                if (variable || !pointers.count(w[1])) return nullptr;
                variable = w[2];
                variable_type = pointers.at(w[1]);
            } else if (w[3] != 1 && w[3] != 3 && w[3] != 7) return nullptr;
            break;
        case 65: // AccessChain
        case 66: // InBoundsAccessChain
            if (n != 5 || !pointers.count(w[1]) || pointers.at(w[1]) != image || w[3] != variable ||
                !bounded(w[4], 1) || !arrays.count(variable_type) || arrays.at(variable_type) != image) return nullptr;
            element_zero.insert(w[2]);
            break;
        case 61: // Load the image through the single descriptor or its element-zero pointer
            if (n < 4) return nullptr;
            if (w[1] == image) {
                if (!(w[3] == variable && variable_type == image) && !element_zero.count(w[3])) return nullptr;
                loaded_images.insert(w[2]);
            }
            break;
        case 83: // CopyObject
            if (n != 4) return nullptr;
            if (integers.count(w[1]) && constants.count(w[3])) constants[w[2]] = constants.at(w[3]);
            else if (int_vec2.count(w[1]) && coords.count(w[3])) coords.insert(w[2]);
            else if (pointers.count(w[1]) && pointers.at(w[1]) == image &&
                     (element_zero.count(w[3]) || (w[3] == variable && variable_type == image)))
                element_zero.insert(w[2]);
            else if (w[1] == image) {
                if (!loaded_images.count(w[3])) return nullptr;
                loaded_images.insert(w[2]);
            }
            break;
        case 95: // ImageFetch with only Lod
            if (n != 7 || !loaded_images.count(w[3]) || !coords.count(w[4]) || w[5] != 2 || !bounded(w[6], 1)) return nullptr;
            fetched = true;
            break;
        // Debug + module metadata, plain scalar/function types, straight-line
        // loads, stores and conversions. No other instruction can introduce a resource.
        case 3: case 4: case 5: case 11: case 16: case 19: case 20: case 22:
        case 33: case 54: case 56: case 62: case 111: case 112: case 248: case 253:
            break;
        default: return nullptr;
        }
        i += n;
    }
    if (!has_shader_cap || !has_memory_model || entry_points != 1) return nullptr;
    if (!fetched || !image || !variable || !sets.count(variable) || !bindings.count(variable)) return nullptr;
    if (variable_type == image || (arrays.count(variable_type) && arrays.at(variable_type) == image)) return component;
    return nullptr;
}
