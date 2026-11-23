#version 460
// d3d12_pso_fuzz_test's pixel shader for UINT targets, written to the translator's runtime contract: SV_Target0 is
// the bits of the triangle's flat colour, so logic ops and UINT writes see exact integers on both sides.

layout(location = 0) flat in vec4 color;
layout(location = 0) out uvec4 target;

void main() { target = floatBitsToUint(color); }
