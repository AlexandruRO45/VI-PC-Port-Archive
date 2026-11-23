#version 460
// A pixel shader standing in for a translated one in d3d12_blend_format_test: two targets, a float colour for render
// target 0 and integers for render target 1. The host draw's HLSL writes the same constants.
layout(location = 0) out vec4 target0;
layout(location = 1) out uvec4 target1;

void main() {
    target0 = vec4(0.25, 0.5, 0.0, 0.0);
    target1 = uvec4(7u, 300u, 1000u, 3u);
}
