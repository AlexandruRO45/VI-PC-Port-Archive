; Vertex shader for ps_interp_f16_table: the vs_ids full-screen triangle, with 1.1 (0x3f8ccccd, not an f16 value) in
; x of param0..param3 at all three vertices, so interpolation returns 1.1 exactly at every pixel. y, z and w are 0.
; Run: vk_harness vs_interp_f16.spv ps_interp_f16_table.spv out.raw
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
v_mov_b32_e32 v7, 0x3f8ccccd
exp param0 v7, v5, v5, v5
exp param1 v7, v5, v5, v5
exp param2 v7, v5, v5, v5
exp param3 v7, v5, v5, v5
s_endpgm
