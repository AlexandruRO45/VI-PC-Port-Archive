#version 460
// A pixel shader for d3d12_ia_bounds_test: the interpolated vertex colour, written as is (alpha too). No root argument
// is read, so the user-data block is declared and never touched. Compiled with -DPC_TAIL it is a shader translated with
// --pc-tail-ubo: s0..s31 alone fill the 128-byte push-constant block and no set-1 tail is declared, since neither heap
// base nor code base is read.

layout(location = 0) in vec4 color;
layout(location = 0) out vec4 target;

layout(push_constant) uniform UserData {
    uint s[32];
#ifndef PC_TAIL
    uvec2 heap_base;
    uvec2 code_base;
#endif
} pc;

void main() {
    target = color;
}
