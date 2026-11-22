#version 460
// A pixel shader for d3d12_ia_bounds_test: the interpolated vertex colour, written as is (alpha too). Compiled
// with -DPC_TAIL it is a --pc-tail-ubo translation: 128 push-constant bytes (s0..s31) and no set-1 block, for a
// device limited to the Vulkan minimum of maxPushConstantsSize.

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
