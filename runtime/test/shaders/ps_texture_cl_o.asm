; image_sample_cl_o on vk_texture_test's two-mip texture (--two-mip): same as ps_texture_cl but with a texel
; offset of (1, 0). The min-LOD clamp is 0.0 left of x = 32 and 1.0 from there on, so the left half reads mip 0
; (red) and the right half mip 1 (cyan), same as ps_texture_cl since the offset is within the same quadrant.
; Address v[4:7] = (offset_x, offset_y, u/4, v/4, clamp): offset first, then coords, then clamp.
v_interp_p1_f32_e32 v2, v0, attr0.x
v_interp_p2_f32_e32 v2, v1, attr0.x
v_interp_p1_f32_e32 v3, v0, attr0.y
v_interp_p2_f32_e32 v3, v1, attr0.y
s_load_dwordx8 s[8:15], s[0:1], 0x0
s_waitcnt lgkmcnt(0)
s_mov_b32 s16, 0x92
s_mov_b32 s17, 0xc00000
s_mov_b32 s18, 0x14500000
s_mov_b32 s19, 0
v_cmp_le_f32_e32 vcc_lo, 0.5, v2
; offset packed: x=1 in bits [5:0], y=0 in bits [13:8] -> 0x00000001
v_mov_b32_e32 v4, 1              ; texel offset packed (x=1, y=0)
v_mul_f32_e32 v5, 0.25, v2
v_mul_f32_e32 v6, 0.25, v3
v_cndmask_b32_e64 v7, 0, 1.0, vcc_lo
image_sample_cl_o v[9:12], v[4:7], s[8:15], s[16:19] dmask:0xf dim:SQ_RSRC_IMG_2D
s_waitcnt vmcnt(0)
exp mrt0 v9, v10, v11, v12 done vm
s_endpgm
