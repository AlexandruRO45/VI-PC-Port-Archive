#version 450

layout(location = 0) flat in int tag;
#ifndef STENCIL_ONLY
layout(location = 0) out vec4 color;
#endif
void main() {
    gl_SampleMask[0] = 1 << (tag % 4);
#ifndef STENCIL_ONLY
    color = vec4(floor(gl_FragCoord.xy), tag + 1, 255) / 255.0;
#endif
}
