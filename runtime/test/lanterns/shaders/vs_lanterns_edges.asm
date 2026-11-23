; Lanterns shared-edge vertex shader (stage 1): wave64, with the title's register layout (docs/d3d12_vulkan.md 3.5):
; eight root constants in s0..s7 and the lowest table pointer at s8. Vertex v0 of the draw from the table's first V#
; (stride 32: screen x, y, z, w, then param0); clip x = x * s0 + s1, y = y * s2 + s3. lanterns_edges.h.
; stage: vs
; wave: 64
;
; s0..s3 the screen-to-clip scale and bias (x, then y); s4..s7 constants nothing reads; s[8:9] the table: slot 0 the
; vertices' V#, slot 1 (+0x20) the pixel shader's V#.
s_load_dwordx4 s[12:15], s[8:9], 0x0
s_waitcnt lgkmcnt(0)
buffer_load_dwordx4 v[4:7], v0, s[12:15], 0 idxen
buffer_load_dwordx4 v[8:11], v0, s[12:15], 0 idxen offset:16
s_waitcnt vmcnt(0)
v_fma_f32 v4, v4, s0, s1
v_fma_f32 v5, v5, s2, s3
exp pos0 v4, v5, v6, v7 done
exp param0 v8, v9, v10, v11
s_endpgm