; Vertex shader for ps_pos_w: the full-screen triangle of vs_ids, with clip w = 2 and no parameters. x and y are the
; vs_ids positions times 2 and z = 0.5, so the position after the divide by w is unchanged in x and y and 0.25 in z.
; With w = 2 the interpolated 1/w is 0.5 everywhere (Vulkan's FragCoord.w), the pixel stage's POS_W_FLOAT is 2.0.
v_lshlrev_b32_e32 v1, 1, v0
v_and_b32_e32 v1, 2, v1
v_and_b32_e32 v2, 2, v0
v_cvt_f32_u32_e32 v1, v1
v_cvt_f32_u32_e32 v2, v2
v_fma_f32 v3, v1, 4.0, -2.0
v_fma_f32 v4, v2, 4.0, -2.0
v_mov_b32_e32 v5, 0.5
v_mov_b32_e32 v6, 2.0
exp pos0 v3, v4, v5, v6 done
s_endpgm
