; Depth-compare sampling with a min-LOD clamp: image_sample_c_cl of a two-mip D32 texture (vk_texture_test dref_c_cl:
; level 0 is 128x128 of depth 0.25 and level 1 64x64 of depth 0.75, sampled through a LESS_OR_EQUAL compare sampler
; at s[16:19]). The coordinates are u / 4, v / 4, so the unclamped LOD is -1 (level 0); the clamp after the
; coordinates is 0.0 left of x = 32 and 1.0 from there on. The left half must read level 0 (0.5 <= 0.25 is false,
; 0.0) and the right half level 1 (0.5 <= 0.75, 1.0). A clamp read from the wrong register, or dropped, leaves the
; right half at 0.0. Address v[4:7] = (z-compare, u / 4, v / 4, clamp); MinLod on the implicit-LOD dref sample
; (OpImageSampleDrefImplicitLod ... MinLod). mrt0 = (f32 result, widened d16 result, u, v).
; Run: vk_texture_test dref_c_cl -> every pixel's first two components are 1.0 right of x = 32, else 0.0
v_interp_p1_f32_e32 v2, v0, attr0.x
v_interp_p2_f32_e32 v2, v1, attr0.x
v_interp_p1_f32_e32 v3, v0, attr0.y
v_interp_p2_f32_e32 v3, v1, attr0.y
v_mov_b32_e32 v4, 0.5
v_mul_f32_e32 v5, 0.25, v2
v_mul_f32_e32 v6, 0.25, v3
s_load_dwordx8 s[8:15], s[0:1], 0x0
s_waitcnt lgkmcnt(0)
s_mov_b32 s16, 0x3092
s_mov_b32 s17, 0xc00000
s_mov_b32 s18, 0x14500000
s_mov_b32 s19, 0
v_cmp_le_f32_e32 vcc_lo, 0.5, v2
v_cndmask_b32_e64 v7, 0, 1.0, vcc_lo
image_sample_c_cl v9, v[4:7], s[8:15], s[16:19] dmask:0x1 dim:SQ_RSRC_IMG_2D
image_sample_c_cl v10, v[4:7], s[8:15], s[16:19] dmask:0x1 dim:SQ_RSRC_IMG_2D d16
s_waitcnt vmcnt(0)
v_cvt_f32_f16_e32 v11, v10
exp mrt0 v9, v11, v2, v3 done vm
s_endpgm
