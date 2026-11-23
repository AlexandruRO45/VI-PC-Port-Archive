#version 450
layout(location = 0) out uint packed_color;
void main() {
    vec4 rgba = vec4((floor(gl_FragCoord.x) + 0.25) / 15.0, 0.73, 0.15, 0.61);
    uvec4 n = uvec4(roundEven(clamp(rgba, 0.0, 1.0) * 15.0));
    packed_color = (n.a << 12) | (n.r << 8) | (n.g << 4) | n.b;
}
