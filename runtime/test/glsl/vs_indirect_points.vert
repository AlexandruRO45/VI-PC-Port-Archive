#version 460
// A vertex shader with no vertex inputs, written to the translator's contract, standing in for a translated one in
// d3d12_indirect_signature_test: vertex v of instance n is a point on the centre of pixel (v, n) of a 16x4 target
// (D3D clip space, y up), so the image read back shows which vertices and instances a draw ran.
layout(push_constant) uniform UserData {
    uint s[32];
    uvec2 heap_base;
    uvec2 code_base;
} pc;

void main() {
    const vec2 pixel = vec2(float(gl_VertexIndex) + 0.5, float(gl_InstanceIndex) + 0.5);
    gl_Position = vec4(pixel.x / 8.0 - 1.0, 1.0 - pixel.y / 2.0, 0.0, 1.0);
    gl_PointSize = 1.0;
}
