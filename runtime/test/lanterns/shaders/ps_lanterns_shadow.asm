; Lanterns shadowed pixel shader (stage 8): the world position (interpolated param1) moved into the shadow map's
; (u, v, depth) by three rows the camera CBV carries after its matrix, four comparison taps half a texel apart (2x2 PCF:
; four image_sample_c, implicit LOD on a one-mip map; the S# compares LESS_EQUAL with point filtering), and the face colour
; times 3/8 + 5/32 * (lit taps). The FSSM input table is ps_lanterns_lit's.
; stage: ps
; wave: 32
; input: flat
; input: interp
;
; s[20:21] root CBV: +0x40 the rows u, v, depth (four floats each)
; s[22:23] root table [4]: slot 0 the shadow map's T# (R32_FLOAT view of the D32 map); s[24:25] table [5]: the S#
s_load_dwordx8 s[28:35], s[22:23], 0x0
s_load_dwordx4 s[36:39], s[24:25], 0x0
s_load_dwordx8 s[40:47], s[20:21], 0x40
s_load_dwordx4 s[48:51], s[20:21], 0x60
v_interp_p1_f32_e32 v4, v0, attr1.x
v_interp_p2_f32_e32 v4, v1, attr1.x
v_interp_p1_f32_e32 v5, v0, attr1.y
v_interp_p2_f32_e32 v5, v1, attr1.y
v_interp_p1_f32_e32 v6, v0, attr1.z
v_interp_p2_f32_e32 v6, v1, attr1.z
s_waitcnt lgkmcnt(0)
v_fma_f32 v8, s40, v4, s43
v_fmac_f32_e32 v8, s41, v5
v_fmac_f32_e32 v8, s42, v6
v_fma_f32 v9, s44, v4, s47
v_fmac_f32_e32 v9, s45, v5
v_fmac_f32_e32 v9, s46, v6
v_fma_f32 v10, s48, v4, s51
v_fmac_f32_e32 v10, s49, v5
v_fmac_f32_e32 v10, s50, v6
v_subrev_f32_e32 v11, 0x3a800000, v8
v_add_f32_e32 v12, 0x3a800000, v8
v_subrev_f32_e32 v13, 0x3a800000, v9
v_add_f32_e32 v14, 0x3a800000, v9
v_mov_b32_e32 v16, v10
v_mov_b32_e32 v17, v11
v_mov_b32_e32 v18, v13
v_mov_b32_e32 v19, v10
v_mov_b32_e32 v20, v12
v_mov_b32_e32 v21, v13
v_mov_b32_e32 v40, v10
v_mov_b32_e32 v41, v11
v_mov_b32_e32 v42, v14
v_mov_b32_e32 v44, v10
v_mov_b32_e32 v45, v12
v_mov_b32_e32 v46, v14
image_sample_c v28, v[16:18], s[28:35], s[36:39] dmask:0x1 dim:SQ_RSRC_IMG_2D
image_sample_c v29, v[19:21], s[28:35], s[36:39] dmask:0x1 dim:SQ_RSRC_IMG_2D
image_sample_c v30, v[40:42], s[28:35], s[36:39] dmask:0x1 dim:SQ_RSRC_IMG_2D
image_sample_c v31, v[44:46], s[28:35], s[36:39] dmask:0x1 dim:SQ_RSRC_IMG_2D
v_interp_mov_f32 v32, p0, attr0.x
v_interp_mov_f32 v33, p0, attr0.y
v_interp_mov_f32 v34, p0, attr0.z
v_interp_mov_f32 v35, p0, attr0.w
s_waitcnt vmcnt(0)
v_add_f32_e32 v28, v28, v29
v_add_f32_e32 v28, v28, v30
v_add_f32_e32 v28, v28, v31
v_mov_b32_e32 v36, 0x3ec00000
v_fmac_f32_e32 v36, 0x3e200000, v28
v_mul_f32_e32 v32, v32, v36
v_mul_f32_e32 v33, v33, v36
v_mul_f32_e32 v34, v34, v36
exp mrt0 v32, v33, v34, v35 done vm
s_endpgm
