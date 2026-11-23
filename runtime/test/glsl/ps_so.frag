#version 460
// Pixel shader for d3d12_stream_output_test when RasterizedStream == 0 runs alongside stream output.

layout(location = 0) in vec4 in_so_pos;
layout(location = 1) in vec4 in_color;
layout(location = 0) out vec4 target;

void main() {
    target = in_color;
}
