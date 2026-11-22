#version 460
// A full-screen triangle at a chosen depth, written to the translator's runtime contract (docs/design.md, "User
// data"), for d3d12_depth24_draw_test: s0 is the depth at the centre, s1 how much it changes per unit of clip-space
// x and s4 per unit of clip-space y, so s1 = s4 = 0 gives a flat triangle and anything else a constant slope across
// the target. s3 picks other shapes on the 64x64 target: 1, vertices 0 and 1 are the ends of a horizontal line through
// the centre of row 31 (clip y 1/64), for a LINE pipeline; 2, vertex i is a point at the centre of pixel (i, 31), for
// a POINT pipeline.

layout(push_constant) uniform UserData {
    uint s[32];
    uvec2 heap_base;
    uvec2 code_base;
} pc;

void main() {
    const vec2 corner = vec2(float(gl_VertexIndex & 2), float((gl_VertexIndex << 1) & 2));
    vec2 xy = corner * 2.0 - 1.0;
    if (pc.s[3] == 1u) xy = vec2(gl_VertexIndex == 0 ? -1.0 : 1.0, 0.015625);
    if (pc.s[3] == 2u) xy = vec2((float(gl_VertexIndex) + 0.5) / 32.0 - 1.0, 0.015625);
    gl_PointSize = 1.0;
    gl_Position = vec4(xy, uintBitsToFloat(pc.s[0]) + uintBitsToFloat(pc.s[1]) * xy.x + uintBitsToFloat(pc.s[4]) * xy.y,
                       1.0);
}
