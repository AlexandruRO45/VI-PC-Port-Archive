#pragma once

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace leonida {

// The float widths that the module's OpFmaKHR (4427) results have: OpTypeFloat (22) gives each float type's width,
// OpTypeVector (23) carries its component's, and an OpFmaKHR's result type (word 1) is one of them. A module with
// Float64 but only an f32 OpFmaKHR needs only shaderFmaFloat32, so the width comes from the instruction itself.
inline void spirv_fma_widths(const std::uint32_t* words, std::size_t size, bool& f32, bool& f64) {
    f32 = f64 = false;
    std::vector<std::pair<std::uint32_t, std::uint32_t>> width;   // float type id -> width (a few entries)
    const auto width_of = [&](std::uint32_t id) -> std::uint32_t {
        for (const auto& [t, w] : width) if (t == id) return w;
        return 0;
    };
    for (std::size_t i = 5; i < size && (words[i] >> 16);) {
        const std::uint32_t op = words[i] & 0xffff, count = words[i] >> 16;
        if (i + count > size) break;
        if (op == 22 && count >= 3) width.emplace_back(words[i + 1], words[i + 2]);
        else if (op == 23 && count >= 4 && width_of(words[i + 2])) width.emplace_back(words[i + 1], width_of(words[i + 2]));
        else if (op == 4427 && count >= 3) (width_of(words[i + 1]) == 64 ? f64 : f32) = true;
        i += count;
    }
}

}  // namespace leonida
