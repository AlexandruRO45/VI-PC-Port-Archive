#version 460
// A vertex shader written to the translator's runtime contract (docs/design.md, "Vertex fetch" and "User data"),
// pairing with gs_double.geom through the geometry fallback (SHADER-1092): the geometry shader consumes position
// and the colour at location 0, while the colour at location 1 -- gs_double does not read it -- is what the fused
// pass's generator stores to its scratch region, so a draw through the fallback proves an unconsumed output cannot
// leak into the capture pass's rows. Inputs sit at location = D3D input register as in vs_draw.vert, and both
// attributes are vec4 so the fallback's fetch (R32G32B32A32 only) reads them whole. s0, s1 (root constants) move
// the vertex in clip space.

layout(location = 0) in vec4 color;     // COLOR0, input register 0
layout(location = 1) in vec4 position;  // POSITION0, input register 1

layout(location = 0) out vec4 out_color;
layout(location = 1) out vec4 out_scratch;

layout(push_constant) uniform UserData {
    uint s[32];
    uvec2 heap_base;
    uvec2 code_base;
} pc;

void main() {
    const vec2 offset = vec2(uintBitsToFloat(pc.s[0]), uintBitsToFloat(pc.s[1]));
    gl_Position = vec4(position.xy + offset, 0.0, 1.0);
    out_color = color;
    out_scratch = vec4(0.5, 0.25, 0.75, 1.0);  // never read by the geometry shader
}
