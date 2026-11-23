// CPU tests for FSSM mesh/fragment linkage. All tables and modules are synthetic.
#include "../src/spirv_mesh_inputs.h"

#include <array>
#include <cstdint>
#include <cstdio>
#include <initializer_list>
#include <vector>

#define CHECK(cond) do { \
    if (!(cond)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #cond); return 1; } \
} while (0)

namespace {
using Bytes = std::vector<std::uint8_t>;
using Words = std::vector<std::uint32_t>;

Bytes mesh_header() {
    Bytes header{'F', 'S', 'S', 'M'};
    header.resize(0x1dc);
    for (std::uint8_t param = 0; param < 3; ++param)
        header.insert(header.end(), {6, std::uint8_t(21 + param), param, 0, 0, 0});
    header.resize(0x29c);
    header.insert(header.end(), {3, 0, 0, 0});
    return header;
}

Bytes pixel_header(std::array<std::uint8_t, 2> params) {
    Bytes header{'F', 'S', 'S', 'M'};
    header.resize(0x138);
    for (std::uint8_t i = 0; i < params.size(); ++i)
        header.insert(header.end(), {6, std::uint8_t(21 + params[i]), 15, i, 0, 0, std::uint8_t(params[i] + 1), 2});
    return header;
}

enum class SecondInput { vector, per_vertex, plain_array };

// Three mesh outputs (arrays of vec4), or two fragment inputs and a colour output at location 1.
// Regenerating the expected fragment module also checks that every other word, including the MRT location, survives.
Words module(bool mesh, std::array<std::uint32_t, 2> locations = {0, 1}, unsigned components = 4,
             SecondInput second = SecondInput::vector) {
    const bool array_input = second != SecondInput::vector;
    Words words{0x07230203, 0x00010500, 0, 17, 0};
    auto emit = [&](std::uint16_t op, std::initializer_list<std::uint32_t> args) {
        words.push_back((std::uint32_t(args.size() + 1) << 16) | op);
        words.insert(words.end(), args);
    };
    emit(17, {1});                                    // Shader
    if (mesh) {
        emit(17, {5283});                             // MeshShadingEXT
        emit(10, {0x5f565053, 0x5f545845, 0x6873656d, 0x6168735f, 0x00726564});  // SPV_EXT_mesh_shader
    } else if (array_input) {
        emit(17, {5284});                             // FragmentBarycentricKHR
        emit(10, {0x5f565053, 0x5f52484b, 0x67617266, 0x746e656d, 0x6168735f,
                  0x5f726564, 0x79726162, 0x746e6563, 0x00636972});  // SPV_KHR_fragment_shader_barycentric
    }
    emit(14, {0, 1});                                 // Logical GLSL450
    emit(15, {mesh ? 5365u : 4u, 14, 0x6e69616d, 0, 11, 12, 13});  // main
    if (mesh) {
        emit(16, {14, 17, 1, 1, 1});                  // LocalSize
        emit(16, {14, 26, 1});                        // OutputVertices
        emit(16, {14, 5270, 1});                      // OutputPrimitivesEXT
        emit(16, {14, 27});                           // OutputPoints
    } else {
        emit(16, {14, 7});                            // OriginUpperLeft
    }
    emit(71, {11, 30, locations[0]});
    emit(71, {12, 30, locations[1]});
    if (second == SecondInput::per_vertex) emit(71, {12, 5285});  // PerVertexKHR
    emit(71, {13, 30, mesh ? 2u : 1u});
    emit(19, {1});                                    // void
    emit(33, {2, 1});                                 // void()
    emit(22, {3, 32});                                // float
    emit(23, {4, 3, components});                     // vec4 for ordinary parameters
    emit(21, {5, 32, 0});                             // uint
    emit(43, {5, 6, array_input ? 3u : 1u});           // three vertex values for the indexed fragment input
    emit(43, {5, 7, 0});                              // zero vertices/primitives
    emit(28, {8, 4, 6});                              // array of vectors
    emit(32, {9, mesh ? 3u : 1u, mesh ? 8u : 4u});   // mesh output or fragment input
    emit(32, {10, 3, 4});                             // fragment output
    if (array_input) emit(32, {16, 1, 8});            // ptr Input vec4[3]
    emit(59, {9, 11, mesh ? 3u : 1u});
    emit(59, {array_input ? 16u : 9u, 12, mesh ? 3u : 1u});
    emit(59, {mesh ? 9u : 10u, 13, 3});
    emit(54, {1, 14, 0, 2});                          // void main()
    emit(248, {15});
    if (mesh) emit(5295, {7, 7});                     // SetMeshOutputsEXT
    emit(253, {});
    emit(56, {});
    return words;
}
}  // namespace

int main() {
    const auto mesh = mesh_header();
    const auto mesh_words = module(true), original = module(false);
    auto verify = [&](const Bytes& ms, const Bytes& ps, bool success, const Words& expected) {
        auto result = original;
        return leonida::link_mesh_inputs(ms, ps, mesh_words, result) == success && result == expected;
    };
    CHECK(verify(mesh, pixel_header({0, 2}), true, module(false, {0, 2})));
    // Move both ordinary input 0 -> 2 and vertex-indexed input 1 -> 0. The fragment MRT stays at 1.
    auto mixed = module(false, {0, 1}, 4, SecondInput::per_vertex);
    CHECK(leonida::link_mesh_inputs(mesh, pixel_header({2, 0}), mesh_words, mixed));
    CHECK(mixed == module(false, {2, 0}, 4, SecondInput::per_vertex));
    const auto plain_array = module(false, {0, 1}, 4, SecondInput::plain_array);
    auto unlinked = plain_array;
    CHECK(!leonida::link_mesh_inputs(mesh, pixel_header({2, 0}), mesh_words, unlinked));
    CHECK(unlinked == plain_array);
    auto already_linked = module(false, {0, 2});
    CHECK(!leonida::link_mesh_inputs(mesh, pixel_header({0, 2}), mesh_words, already_linked));
    CHECK(already_linked == module(false, {0, 2}));
    auto ordinary = original;
    CHECK(!leonida::link_mesh_inputs(mesh, pixel_header({0, 2}), module(true, {0, 1}, 3), ordinary));
    CHECK(ordinary == original);
    CHECK(verify(mesh, pixel_header({1, 0}), false, original));  // dense permutation could already be linked
    CHECK(verify(mesh, pixel_header({0, 1}), true, original));
    const Bytes no_table{'F', 'S', 'S', 'M'};
    CHECK(verify(no_table, no_table, true, original));

    // Both cases request a first relocation. Failure on the second row must leave even that first input untouched.
    auto broken = pixel_header({1, 2});
    broken.pop_back();
    CHECK(verify(mesh, broken, false, original));
    broken = pixel_header({1, 2});
    broken[0x138 + 8 + 1] = 21;  // semantic selects export 0, but the one-based stored slot selects export 2
    CHECK(verify(mesh, broken, false, original));
    auto short_mesh = mesh;
    short_mesh.pop_back();  // incomplete 32-bit export count
    CHECK(verify(short_mesh, pixel_header({1, 2}), false, original));
    return 0;
}
