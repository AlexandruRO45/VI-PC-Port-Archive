#version 460
// Pixel shader for d3d12_pso_usersig_draw_test. It stands in for a translated PS that reads the 32 user-data
// dwords the layer packs from a mapped root signature: s2 and s3 are the third and fourth dwords of the record's
// 0x401 constants parameter (vs_draw.vert reads s0/s1 as a clip-space offset), so the fragment colour follows
// SetGraphicsRoot32BitConstants. The descriptor table the record also maps sits at s4:s5 and is not read here.
layout(location = 0) in vec4 color;
layout(location = 0) out vec4 target;

layout(push_constant) uniform UserData {
    uint s[32];
    uvec2 heap_base;
    uvec2 code_base;
} pc;

void main() {
    target = vec4(color.r * uintBitsToFloat(pc.s[2]), uintBitsToFloat(pc.s[3]), 0.0, 1.0);
}
