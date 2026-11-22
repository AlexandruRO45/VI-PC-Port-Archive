// The draw after a geometry shader that ran as a compute capture pass (tools/spirv_gs_to_compute.py, vk_harness
// VKH_GS_COMPUTE): the emitted strips, in emission order, as a list. CPU-only, so vk_gs_strips_test checks it.
#pragma once

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace leonida::vk_test {

// The GS capture pass wrote, for primitive p, meta[p * (max_verts + 1)] = vertices emitted and
// meta[p * (max_verts + 1) + 1 + n] = 1 where a strip ends after vertex n, and the vertices' output slots at
// rec[(p * max_verts + n) * outputs]. Turns the strips into list primitives in emission order, as the geometry stage
// would (a triangle strip's triangle i is (i, i + 1 + i % 2, i + 2 - i % 2), keeping vertex i the provoking vertex).
// Returns false when a primitive claims more vertices than max_verts.
inline bool gs_strips_to_list(const uint32_t* meta, const float* rec, uint32_t prims, uint32_t max_verts,
                              uint32_t outputs, uint32_t topology, std::vector<float>& vertices) {
    const uint32_t per = topology == 2 ? 3 : topology == 1 ? 2 : 1;
    const std::size_t vec = std::size_t(outputs) * 4;
    for (uint32_t p = 0; p < prims; ++p) {
        const uint32_t* m = meta + std::size_t(p) * (max_verts + 1);
        if (m[0] > max_verts) return false;
        uint32_t start = 0;
        for (uint32_t n = 0; n < m[0]; ++n) {
            if (!m[1 + n] && n + 1 < m[0]) continue;   // the strip goes on
            for (uint32_t i = start; i + per <= n + 1; ++i) {
                uint32_t idx[3] = {i, i + 1, i + 2};
                if (per == 3 && (i - start) % 2) std::swap(idx[1], idx[2]);
                for (uint32_t k = 0; k < per; ++k) {
                    const float* v = rec + (std::size_t(p) * max_verts + idx[k]) * vec;
                    vertices.insert(vertices.end(), v, v + vec);
                }
            }
            start = n + 1;
        }
    }
    return true;
}

}  // namespace leonida::vk_test
