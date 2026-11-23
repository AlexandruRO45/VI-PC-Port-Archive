#version 460
// A geometry shader for d3d12_volume_draw_test: emits each input triangle twice, to gl_Layer = 0 (red) and
// gl_Layer = 1 (blue), so a layered draw into a 3D volume RTV with WSize >= 2 writes distinct colours to both
// target W slices. Written to the translator's contract (triangles in, triangle_strip out).

layout(triangles) in;
layout(triangle_strip, max_vertices = 6) out;

layout(location = 0) in vec4 in_color[];
layout(location = 0) out vec4 out_color;

void main() {
    const vec4 layer_colors[2] = vec4[2](vec4(1.0, 0.0, 0.0, 1.0), vec4(0.0, 0.0, 1.0, 1.0));
    for (int layer = 0; layer < 2; ++layer) {
        for (int i = 0; i < 3; ++i) {
            gl_Position = gl_in[i].gl_Position;
            gl_Layer = layer;
            out_color = layer_colors[layer];
            EmitVertex();
        }
        EndPrimitive();
    }
}
