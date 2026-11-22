#version 460
// A vertex shader for d3d12_restart_test, written to the translator's runtime contract: positions from SV_VertexID
// alone (no input layout, no vertex buffer). Vertices 0-3 are the corners of a quad from x 8 to 24 and 4-7 those of
// one from x 40 to 56, both from y 8 to 56, in pixels of a 64x64 target (y down). Any other index, such as a cut
// index drawn as a vertex, lands at (32, 32), between the two.

layout(push_constant) uniform UserData {
    uint s[32];
    uvec2 heap_base;
    uvec2 code_base;
} pc;

const vec2 corners[8] = vec2[8](vec2(8, 8), vec2(8, 56), vec2(24, 8), vec2(24, 56),
                                vec2(40, 8), vec2(40, 56), vec2(56, 8), vec2(56, 56));

void main() {
    const uint index = uint(gl_VertexIndex);
    const vec2 pixel = index < 8u ? corners[index] : vec2(32.0, 32.0);
    gl_Position = vec4(pixel.x / 32.0 - 1.0, 1.0 - pixel.y / 32.0, 0.0, 1.0);  // D3D clip space, y up
}
