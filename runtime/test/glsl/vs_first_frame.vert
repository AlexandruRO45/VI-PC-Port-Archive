#version 460
// The first-frame dry-run vertex shader, written to the translator's runtime contract: the 144-byte push-constant
// block is s0..s31 followed by the heap base and the code base. s0/s1 are the clip-space offset the test sets; the
// quad's own vertices are 0.25 to the left of the target so the draw only covers every pixel if those dwords arrive.
layout(location = 0) in vec2 position;  // POSITION0

layout(push_constant) uniform UserData {
    uint s[32];
    uvec2 heap_base;
    uvec2 code_base;
} pc;

void main() {
    const vec2 offset = vec2(uintBitsToFloat(pc.s[0]), uintBitsToFloat(pc.s[1]));
    gl_Position = vec4(position + offset, 0.0, 1.0);
}
