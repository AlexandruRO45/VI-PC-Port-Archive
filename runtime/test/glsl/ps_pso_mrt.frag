#version 460
// d3d12_pso_fuzz_test's two-target pixel shader, written to the translator's runtime contract: SV_Target0 is the
// triangle's flat colour and SV_Target1 the same colour swizzled (b, g, r, a), so the two targets get different
// values and an independent blend state per target is visible.

layout(location = 0) flat in vec4 color;
layout(location = 0) out vec4 target0;
layout(location = 1) out vec4 target1;

void main() {
    target0 = color;
    target1 = color.zyxw;
}
