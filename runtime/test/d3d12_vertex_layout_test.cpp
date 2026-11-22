// CPU-only test of the layout builder used by the Vulkan D3D12 PSO.
#include "../shims/d3d12_vertex_layout.h"
#include "../src/shader_library.h"

#include <algorithm>
#include <cstdio>

#define CHECK(condition) do {                                    \
    if (!(condition)) {                                          \
        std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); \
        return 1;                                                \
    }                                                            \
} while (0)

int main() {
    using rt::d3d12::build_vertex_layout;
    using rt::d3d12::VertexLayout;
    leonida::ShaderEntry vs;
    vs.input_signature = {{"TEXCOORD", 7, 0, 0, "sint", 2}, {"TEXCOORD", 3, 0, 0, "uint", 0},
                          {"POSITION", 0, 1, 0, "float", {}}, {"SV_VertexID", 0, 0, 6, "uint", {}}};
    D3D12_INPUT_ELEMENT_DESC elements[] = {
        {"position", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 4, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
        {"TexCoord", 7, DXGI_FORMAT_R32G32_SINT, 0, D3D12_APPEND_ALIGNED_ELEMENT,
         D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
        {"TEXCOORD", 3, DXGI_FORMAT_R32G32_UINT, 3, 8, D3D12_INPUT_CLASSIFICATION_PER_INSTANCE_DATA, 0},
    };
    VkPhysicalDeviceLimits limits{};
    limits.maxVertexInputAttributes = 3;
    limits.maxVertexInputBindings = 4;
    limits.maxVertexInputAttributeOffset = 32;
    VertexLayout result;
    const auto build = [&](UINT count, bool divisor_supported = false) {
        return build_vertex_layout({elements, count}, vs, limits, divisor_supported, result);
    };
    CHECK(build(3) == S_OK);
    CHECK(result.attributes.size() == 3 && result.bindings.size() == 2 && result.slots == 9);
    CHECK(result.zero_step_slots == 8 && result.bindings[1].inputRate == VK_VERTEX_INPUT_RATE_INSTANCE);
    CHECK(result.attributes[0].location == 1 && result.attributes[0].offset == 4);
    CHECK(result.attributes[1].location == 2 && result.attributes[1].binding == 0 &&
          result.attributes[1].offset == 16 && result.attributes[1].format == VK_FORMAT_R32G32_SINT);
    CHECK(result.attributes[2].location == 0 && result.attributes[2].binding == 3 &&
          result.attributes[2].offset == 8 && result.attributes[2].format == VK_FORMAT_R32G32_UINT);
    CHECK(result.divisors.empty());

    elements[2].InstanceDataStepRate = 2;
    CHECK(build(3) == E_NOTIMPL);
    CHECK(build(3, true) == S_OK);
    CHECK(result.divisors.size() == 1 && result.divisors[0].binding == 3 && result.divisors[0].divisor == 2);
    CHECK(result.zero_step_slots == 0);
    elements[2].InstanceDataStepRate = 0;

    // Reverse the two TEXCOORD elements and interleave them in one buffer instead.
    std::swap(elements[1], elements[2]);
    elements[1].InputSlot = 0;
    elements[1].AlignedByteOffset = 16;
    elements[1].InputSlotClass = D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA;
    CHECK(build(3) == S_OK);
    CHECK(result.attributes.size() == 3 && result.bindings.size() == 1 && result.slots == 1);
    CHECK(result.attributes[1].location == 0 && result.attributes[1].offset == 16);
    CHECK(result.attributes[2].location == 2 && result.attributes[2].offset == 24);

    CHECK(build(2) == E_INVALIDARG);
    elements[2].SemanticIndex = 3;
    CHECK(build(3) == E_INVALIDARG);
    elements[2].SemanticIndex = 7;
    elements[2].SemanticName = "COLOR";
    CHECK(build(3) == E_INVALIDARG);
    elements[2].SemanticName = "TexCoord";
    limits.maxVertexInputAttributes = 2;
    CHECK(build(3) == E_INVALIDARG);
    limits.maxVertexInputAttributes = 3;
    limits.maxVertexInputAttributeOffset = 23;
    CHECK(build(3) == E_INVALIDARG);
    limits.maxVertexInputAttributeOffset = 32;
    vs.input_signature[0].location.reset();
    CHECK(build(3) == E_INVALIDARG);

    // An index without a location field still binds an ordinary input by register.
    vs.input_signature.erase(vs.input_signature.begin(), vs.input_signature.begin() + 2);
    CHECK(build(1) == S_OK);
    CHECK(result.attributes.size() == 1 && result.attributes[0].location == 1);
}
