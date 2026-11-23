; Depth-compare sampling with a LOD bias and a min-LOD clamp: image_sample_c_b_cl, as ps_sample_c_cl with the bias
; 0.0 in front of the z-compare (address v[4:8] = bias, z-compare, u / 4, v / 4, clamp), so the bias and the clamp
; each have to come from their own VGPR. A clamp read as the bias (1.0) would move the right half to level 1 and
; leave it there; a bias read as the clamp would clamp the LOD to 1 everywhere and turn the left half to 1.0 too.
; mrt0 = (f32 result, widened d16 result, u, v).
; Run: vk_texture_test dref_c_b_cl -> every pixel's first two components are 1.0 right of x = 32, else 0.0
v_interp_p1_f32_e32 v2, v0, attr0.x
v_interp_p2_f32_e32 v2, v1, attr0.x
v_interp_p1_f32_e32 v3, v0, attr0.y
v_interp_p2_f32_e32 v3, v1, attr0.y
v_mov_b32_e32 v4, 0
v_mov_b32_e32 v5, 0.5
v_mul_f32_e32 v6, 0.25, v2
v_mul_f32_e32 v7, 0.25, v3
s_load_dwordx8 s[8:15], s[0:1], 0x0
s_waitcnt lgkmcnt(0)
s_mov_b32 s16, 0x3092
s_mov_b32 s17, 0xc00000
s_mov_b32 s18, 0x14500000
s_mov_b32 s19, 0
v_cmp_le_f32_e32 vcc_lo, 0.5, v2
v_cndmask_b32_e64 v8, 0, 1.0, vcc_lo
image_sample_c_b_cl v9, v[4:8], s[8:15], s[16:19] dmask:0x1 dim:SQ_RSRC_IMG_2D
image_sample_c_b_cl v10, v[4:8], s[8:15], s[16:19] dmask:0x1 dim:SQ_RSRC_IMG_2D d16
s_waitcnt vmcnt(0)
v_cvt_f32_f16_e32 v11, v10
exp mrt0 v9, v11, v2, v3 done vm
s_endpgm
