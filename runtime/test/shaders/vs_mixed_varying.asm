; Vertex shader for ps_mixed_f32_varying: the vs_ids full-screen triangle, with param0 = (2i, 2j, 0, 1) as there and
; param1 = 1.0, 2.0 and 3.0 at the three vertices, so vertex 0's param1.x is 1.0 and the interpolated value over the
; triangle is 1 + i + 2j.
v_lshlrev_b32_e32 v1, 1, v0
v_and_b32_e32 v1, 2, v1
v_and_b32_e32 v2, 2, v0
v_cvt_f32_u32_e32 v1, v1
v_cvt_f32_u32_e32 v2, v2
v_fma_f32 v3, v1, 2.0, -1.0
v_fma_f32 v4, v2, 2.0, -1.0
v_mov_b32_e32 v5, 0
v_mov_b32_e32 v6, 1.0
exp pos0 v3, v4, v5, v6 done
exp param0 v1, v2, v5, v6
v_cvt_f32_u32_e32 v7, v0
v_mov_b32_e32 v8, 1.0
v_add_f32 v7, v7, v8
exp param1 v7, v7, v7, v7
s_endpgm
