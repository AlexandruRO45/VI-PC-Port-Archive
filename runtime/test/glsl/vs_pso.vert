#version 460
// d3d12_pso_fuzz_test's vertex shader, written to the translator's runtime contract (docs/design.md, "Vertex fetch"):
// POSITION (input register 0) is the clip-space position as given, COLOR (register 1) goes to the pixel shader
// flat, from the first vertex as in D3D12, so a triangle has one colour and no interpolation rounding.

layout(location = 0) in vec4 position;  // POSITION0, input register 0
layout(location = 1) in vec4 color;     // COLOR0, input register 1

layout(location = 0) flat out vec4 out_color;

void main() {
    gl_Position = position;
    out_color = color;
}
