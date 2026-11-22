#version 460
// A pixel shader for d3d12_ia_bounds_test: the interpolated vertex colour, written as is (alpha too).

layout(location = 0) in vec4 color;
layout(location = 0) out vec4 target;

void main() {
    target = color;
}
