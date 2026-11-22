; 2D image_sample_l (explicit LOD) on R32_UINT and R32_SINT views, 32-bit and D16, with the descriptor-miss float
; fallback as a control. vk_store_test sample-l fills the uint image with 100 + i, the sint image with -2i and the
; rgba8 image with its gradient (r = round(x / 7 * 255)), texel (x, y) at i = 8y + x. Each lane samples its own
; texel centre:
;   sint image <- uint + sint texel = 100 - i, if each D16 sample agrees with its own 32-bit sample, else 0x7fffffff
;   uint image <- round(r * 255) through a T# that is not in the table but names R32_UINT: a miss reads the float
;                 fallback (index 0 = the rgba8 image), so round(x / 7 * 255)
; The rgba8 image is not written, so it still holds its gradient. Before the explicit-LOD form used the T# format
; (#2458), the integer samples went through a float sampled image, which Vulkan answers with an undefined texel; an
; RTX 5070 returned the right 32-bit words but 0 (uint) and -1 (sint) through D16 (#2458's GPU check).
; The unregistered S# selects the harness's nearest, clamp-to-edge sampler; the images have one mip, so LOD 0.
v_and_b32_e32 v1, 7, v0
v_lshrrev_b32_e32 v2, 3, v0
v_cvt_f32_u32_e32 v3, v1
v_cvt_f32_u32_e32 v4, v2
v_add_f32_e32 v8, 0.5, v3
v_add_f32_e32 v9, 0.5, v4
v_mul_f32_e32 v8, 0x3e000000, v8
v_mul_f32_e32 v9, 0x3e000000, v9
v_mov_b32_e32 v10, 0
s_mov_b32 s32, 0
s_mov_b32 s33, 0
s_mov_b32 s34, 0
s_mov_b32 s35, 0
s_mov_b32 s16, 0x2000
s_mov_b32 s17, 0x1400000
s_mov_b32 s18, 0x22
s_mov_b32 s19, 0x22
s_mov_b32 s20, 0x22
s_mov_b32 s21, 0x22
s_mov_b32 s22, 0x22
s_mov_b32 s23, 0x22
image_sample_l v7, v[8:10], s[16:23], s[32:35] dmask:0x1 dim:SQ_RSRC_IMG_2D
image_sample_l v19, v[8:10], s[16:23], s[32:35] dmask:0x1 dim:SQ_RSRC_IMG_2D d16
s_mov_b32 s24, 0x3000
s_mov_b32 s25, 0x1500000
s_mov_b32 s26, 0x33
s_mov_b32 s27, 0x33
s_mov_b32 s28, 0x33
s_mov_b32 s29, 0x33
s_mov_b32 s30, 0x33
s_mov_b32 s31, 0x33
image_sample_l v11, v[8:10], s[24:31], s[32:35] dmask:0x1 dim:SQ_RSRC_IMG_2D
image_sample_l v20, v[8:10], s[24:31], s[32:35] dmask:0x1 dim:SQ_RSRC_IMG_2D d16
s_mov_b32 s40, 0x7000
s_mov_b32 s41, 0x1400000
s_mov_b32 s42, 0x77
s_mov_b32 s43, 0x77
s_mov_b32 s44, 0x77
s_mov_b32 s45, 0x77
s_mov_b32 s46, 0x77
s_mov_b32 s47, 0x77
image_sample_l v15, v[8:10], s[40:47], s[32:35] dmask:0x1 dim:SQ_RSRC_IMG_2D
s_waitcnt vmcnt(0)
v_and_b32_e32 v19, 0xffff, v19
v_bfe_i32 v20, v20, 0, 16
; Compare uint and sint separately: +1 in the uint D16 sample and -1 in the sint D16 sample must not cancel.
v_mov_b32_e32 v21, 1
v_cmp_eq_u32_e32 vcc_lo, v7, v19
v_cndmask_b32_e32 v19, 0, v21, vcc_lo
v_cmp_eq_u32_e32 vcc_lo, v11, v20
v_cndmask_b32_e32 v19, 0, v19, vcc_lo
v_add_nc_u32_e32 v11, v7, v11
v_mov_b32_e32 v21, 0x7fffffff
v_cmp_eq_u32_e32 vcc_lo, 1, v19
v_cndmask_b32_e32 v11, v21, v11, vcc_lo
v_mov_b32_e32 v12, 0
v_mov_b32_e32 v13, 0
v_mov_b32_e32 v14, 0
image_store v[11:14], v[1:2], s[24:31] dmask:0xf dim:SQ_RSRC_IMG_2D unorm
v_mul_f32_e32 v15, 0x437f0000, v15
v_add_f32_e32 v15, 0.5, v15
v_cvt_u32_f32_e32 v15, v15
v_mov_b32_e32 v16, 0
v_mov_b32_e32 v17, 0
v_mov_b32_e32 v18, 0
image_store v[15:18], v[1:2], s[16:23] dmask:0xf dim:SQ_RSRC_IMG_2D unorm
s_endpgm
