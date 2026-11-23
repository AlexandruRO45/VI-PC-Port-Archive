#version 460
// Vertex shader for d3d12_stream_output_test: captures transformed position and colour into SO slot 0
// while still passing clip position and colour down to rasterization when RasterizedStream == 0.

layout(location = 0) in vec4 in_pos;    // POSITION0, register 0
layout(location = 1) in vec4 in_color;  // COLOR0, register 1

layout(location = 0, xfb_buffer = 0, xfb_offset = 0, xfb_stride = 32) out vec4 so_pos;
layout(location = 1, xfb_buffer = 0, xfb_offset = 16, xfb_stride = 32) out vec4 so_color;

layout(push_constant) uniform UserData {
    uint s[32];
    uvec2 heap_base;
    uvec2 code_base;
} pc;

void main() {
    const vec2 offset = vec2(uintBitsToFloat(pc.s[0]), uintBitsToFloat(pc.s[1]));
    const vec4 transformed = vec4(in_pos.xy + offset, in_pos.zw);
    gl_Position = transformed;
    so_pos = transformed;
    so_color = in_color;
}
