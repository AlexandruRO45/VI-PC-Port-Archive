// Fixture probe for vk_texture_test. Both reads stay within the default 64 KiB heap.
// Expected RGBA: (0,0,0,1) with --zero-root, (256,1,0,1) with the texture pointer in s8.
#version 460
#extension GL_EXT_buffer_reference : require
layout(buffer_reference, std430, buffer_reference_align = 4) readonly buffer Heap { uint words[]; };
layout(push_constant) uniform Inputs { layout(offset=32) uint root8; layout(offset=128) Heap heap; } pc;
layout(location=0) out vec4 color;
void main() {
    color = vec4(float(pc.root8), float(pc.heap.words[64] != 0), 0.0, 1.0);
}
