// CPU-only tests for vk_gs_strips.h: the strips a geometry shader emitted through its compute capture pass
// (tools/spirv_gs_to_compute.py) become the list vk_harness draws. No Vulkan instance or GPU is needed.
#include "vk_gs_strips.h"

#include <cstdio>
#include <vector>

namespace {

int failures = 0;

// One output slot per vertex whose x is the vertex's number, so the list reads back as vertex numbers.
std::vector<float> records(uint32_t prims, uint32_t max_verts) {
    std::vector<float> rec;
    for (uint32_t i = 0; i < prims * max_verts; ++i) rec.insert(rec.end(), {float(i), 0, 0, 1});
    return rec;
}

std::vector<int> order(const std::vector<float>& vertices) {
    std::vector<int> out;
    for (std::size_t i = 0; i < vertices.size(); i += 4) out.push_back(int(vertices[i]));
    return out;
}

void expect(const char* what, const std::vector<uint32_t>& meta, uint32_t prims, uint32_t max_verts,
            uint32_t topology, const std::vector<int>& want) {
    std::vector<float> vertices;
    const auto rec = records(prims, max_verts);
    if (!leonida::vk_test::gs_strips_to_list(meta.data(), rec.data(), prims, max_verts, 1, topology, vertices)) {
        std::fprintf(stderr, "FAIL %s: rejected\n", what);
        ++failures;
        return;
    }
    if (order(vertices) != want) {
        std::fprintf(stderr, "FAIL %s: got", what);
        for (int v : order(vertices)) std::fprintf(stderr, " %d", v);
        std::fprintf(stderr, "\n");
        ++failures;
    }
}

}  // namespace

int main() {
    // gs_tri: one primitive, three vertices, then a cut
    expect("one triangle", {3, 0, 0, 1}, 1, 3, 2, {0, 1, 2});
    // a five-vertex strip: triangles (0,1,2) (1,3,2) (2,3,4), the odd one swapped so the winding holds
    expect("strip of five", {5, 0, 0, 0, 0, 0}, 1, 5, 2, {0, 1, 2, 1, 3, 2, 2, 3, 4});
    // a cut after vertex 3 starts a new strip at 4: its first triangle is even again
    expect("two strips", {7, 0, 0, 0, 1, 0, 0, 0}, 1, 7, 2, {0, 1, 2, 1, 3, 2, 4, 5, 6});
    // a strip too short for a triangle draws nothing; a primitive that emitted nothing neither
    expect("short strip", {2, 0, 1, 0, 0, 0, 0}, 2, 3, 2, {});
    // two primitives: the second's vertices start at max_verts
    expect("two primitives", {3, 0, 0, 1, 3, 0, 0, 1}, 2, 3, 2, {0, 1, 2, 3, 4, 5});
    // line strips become segments, points stay points
    expect("line strip", {3, 0, 0, 0}, 1, 3, 1, {0, 1, 1, 2});
    expect("points", {2, 0, 0, 0}, 1, 3, 0, {0, 1});
    // a primitive that claims more vertices than the shader's maximum is refused
    std::vector<float> vertices;
    const std::vector<uint32_t> meta{4, 0, 0, 0};
    const auto rec = records(1, 3);
    if (leonida::vk_test::gs_strips_to_list(meta.data(), rec.data(), 1, 3, 1, 2, vertices)) {
        std::fprintf(stderr, "FAIL overflow: accepted\n");
        ++failures;
    }
    if (failures) return 1;
    std::printf("vk_gs_strips: all cases pass\n");
    return 0;
}
