; d16 image_sample_lz and image_gather4_lz on a 2D_ARRAY (SHADER-308 follow-up, review on #4318): the result is
; narrowed by the T#'s format, as the 2D reads are (UNORM/SNORM: the exact quotient rounded once; 32_FLOAT: toward
; zero). Pixel (px, py) reads texel x = px % 8, y = px / 8 of layer py % 12 of vk_texture_test's 8x8x12 array:
; image_sample_lz at the texel centre, image_gather4_lz at its corner below right (taps (x, y+1), (x+1, y+1),
; (x+1, y), clamped to the edge). Each f16 half is exported as 0x3f80xxxx, a normal float, so RGBA32F keeps its bits:
;   r = sample, g = gather tap 0, b = tap 1, a = tap 2.
; T# at s[8:15] loaded via s[0:1], literal S# (the one vk_texture_test registers; nearest, clamp to edge) at s[16:19].
; Run: vk_texture_test d16_array f32|unorm16|snorm16 [out] -> every half of every pixel is the texel the rule gives
; (768 texels: the 32_FLOAT halfway points and their neighbours, or the UNORM16/SNORM16 values whose f32 is a halfway
; point and others); exit 1 with the count of wrong halves otherwise.
v_interp_p1_f32_e32 v2, v0, attr0.x
v_interp_p2_f32_e32 v2, v1, attr0.x
v_interp_p1_f32_e32 v3, v0, attr0.y
v_interp_p2_f32_e32 v3, v1, attr0.y
v_mul_f32_e32 v4, 0x42800000, v2
v_floor_f32_e32 v4, v4
v_mul_f32_e32 v5, 0x42800000, v3
v_floor_f32_e32 v5, v5
v_add_f32_e32 v6, 0.5, v5
v_mul_f32_e32 v6, 0x3daaaaab, v6
v_floor_f32_e32 v6, v6
v_fma_f32 v6, v6, 0xc1400000, v5
v_add_f32_e32 v7, 0.5, v4
v_mul_f32_e32 v7, 0x3e000000, v7
v_floor_f32_e32 v7, v7
v_fma_f32 v8, v7, 0xc1000000, v4
v_add_f32_e32 v10, 0.5, v8
v_mul_f32_e32 v10, 0x3e000000, v10
v_add_f32_e32 v11, 0.5, v7
v_mul_f32_e32 v11, 0x3e000000, v11
v_mov_b32_e32 v12, v6
v_add_f32_e32 v13, 1.0, v8
v_mul_f32_e32 v13, 0x3e000000, v13
v_add_f32_e32 v14, 1.0, v7
v_mul_f32_e32 v14, 0x3e000000, v14
v_mov_b32_e32 v15, v6
s_load_dwordx8 s[8:15], s[0:1], 0x0
s_waitcnt lgkmcnt(0)
s_mov_b32 s16, 0x92
s_mov_b32 s17, 0xc00000
s_mov_b32 s18, 0x14500000
s_mov_b32 s19, 0
image_sample_lz v20, v[10:12], s[8:15], s[16:19] dmask:0x1 dim:SQ_RSRC_IMG_2D_ARRAY d16
image_gather4_lz v[21:22], v[13:15], s[8:15], s[16:19] dmask:0x1 dim:SQ_RSRC_IMG_2D_ARRAY d16
s_waitcnt vmcnt(0)
s_mov_b32 s20, 0x3f800000
s_mov_b32 s21, 0xffff
v_and_or_b32 v24, v20, s21, s20
v_and_or_b32 v25, v21, s21, s20
v_lshrrev_b32_e32 v26, 16, v21
v_or_b32_e32 v26, s20, v26
v_and_or_b32 v27, v22, s21, s20
exp mrt0 v24, v25, v26, v27 done vm
s_endpgm
