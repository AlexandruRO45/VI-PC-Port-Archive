#version 460
// A pixel shader written to the interface `rdna2spirv.py --bary-fallback` gives a translated one (SHADER-98),
// standing in for it in d3d12_bary_fallback_test: the raw barycentrics are an ordinary perspective-interpolated vec3
// input at the location the index entry names as bary_fallback_input (1 here), next to a VS output at location 0.
// The output is the barycentrics in red, green and blue, and the interpolated colour's green in alpha, which shows
// that the pass-through stage copied location 0 too.

layout(location = 0) in vec4 color;
layout(location = 1) in vec3 bary;
layout(location = 0) out vec4 target;

void main() {
    target = vec4(bary, color.g);
}
