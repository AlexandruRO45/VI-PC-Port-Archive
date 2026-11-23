#version 460
// ps_bary_fallback.frag that also reads gl_PrimitiveID, for d3d12_bary_fallback_test: behind the pass-through
// geometry shader, the fragment's PrimitiveId must come from the GS (VUID-VkGraphicsPipelineCreateInfo-pStages-00740).
// The output is the barycentrics in red, green and blue, and alpha 1 for primitive 1, 0 for primitive 0.

layout(location = 0) in vec4 color;
layout(location = 1) in vec3 bary;
layout(location = 0) out vec4 target;

void main() {
    target = vec4(bary, gl_PrimitiveID == 1 ? 1.0 : 0.0);
}
