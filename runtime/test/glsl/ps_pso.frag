#version 460
// d3d12_pso_fuzz_test's pixel shader, written to the translator's runtime contract: SV_Target0 is the triangle's
// flat colour, so what reaches the target is decided by the PSO's blend, depth-stencil and rasterizer state.

layout(location = 0) flat in vec4 color;
layout(location = 0) out vec4 target;

void main() { target = color; }
