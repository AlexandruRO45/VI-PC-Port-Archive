#version 460
// A vertex shader written to the translator's runtime contract, for d3d12_ia_bounds_test: position (input register 1)
// goes to clip space as is and COLOR (input register 0) is passed on. No root argument is read, so the user-data block
// is declared and never touched. Compiled with -DPC_TAIL it is a shader translated with --pc-tail-ubo: s0..s31 alone
// fill the 128-byte push-constant block and no set-1 tail is declared, since neither heap base nor code base is read.

layout(location = 0) in vec4 color;     // COLOR0, input register 0
layout(location = 1) in vec4 position;  // POSITION0, input register 1

layout(location = 0) out vec4 out_color;

layout(push_constant) uniform UserData {
    uint s[32];
#ifndef PC_TAIL
    uvec2 heap_base;
    uvec2 code_base;
#endif
} pc;

void main() {
    gl_Position = vec4(position.xy, 0.0, 1.0);
    out_color = color;
}
