#version 460
// A pixel shader written to the translator's contract, standing in for a translated one in
// d3d12_indirect_signature_test: it writes root constant s0 + 1, so a drawn pixel shows the constant the draw saw.
layout(location = 0) out vec4 target;
layout(push_constant) uniform UserData {
    uint s[32];
    uvec2 heap_base;
    uvec2 code_base;
} pc;

void main() {
    target = vec4(float(pc.s[0] + 1u), 0.0, 0.0, 1.0);
}
