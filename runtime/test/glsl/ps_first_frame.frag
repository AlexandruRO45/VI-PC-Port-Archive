#version 460
// The first-frame dry-run pixel shader: proves every one of the 32 user-SGPR dwords reached the shader (an FNV-1a
// over s0..s31), that the heap-base tail is non-zero, and that this invocation really ran (a fixed marker and alpha).
// The test compares the R32G32B32A32_UINT readback with the same FNV-1a computed on the CPU.
layout(location = 0) out uvec4 out_value;

layout(push_constant) uniform UserData {
    uint s[32];
    uvec2 heap_base;
    uvec2 code_base;
} pc;

void main() {
    uint h = 2166136261u;
    for (int i = 0; i < 32; ++i) {
        h ^= pc.s[i];
        h *= 16777619u;
    }
    const uint heap_ok = (pc.heap_base.x | pc.heap_base.y) != 0u ? 0x48454150u : 0x42414400u;
    out_value = uvec4(h, heap_ok, 0x5EED5EEDu, 0xFFFFFFFFu);
}
