#version 460
#extension GL_ARB_shader_viewport_layer_array : require
// Synthetic translated VS contract; selects D3D12 viewport 1 for every vertex.
layout(location = 0) in vec4 color;
layout(location = 1) in vec4 position;
layout(location = 0) out vec4 out_color;
layout(push_constant) uniform UserData {
    uint s[32];
    uvec2 heap_base;
    uvec2 code_base;
} pc;
void main() {
    const vec2 offset = vec2(uintBitsToFloat(pc.s[0]), uintBitsToFloat(pc.s[1]));
    gl_Position = vec4(position.xy + offset, 0.0, 1.0);
    gl_ViewportIndex = 1;
    out_color = color;
}
