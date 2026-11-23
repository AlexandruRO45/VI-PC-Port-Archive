; Lanterns textured solid pixel shader (stage 6): the BC7 squares sampled with image_sample_l at the mip level the
; draw's tint.a gives (root constant s19), at uv = ((x + z) / 2, y / 2) of the interpolated world position, times the
; face colour. The FSSM input table is ps_lanterns_lit's.
; stage: ps
; wave: 32
; input: flat
; input: interp
;
; s19 = tint.a (the LOD); s[22:23] root table [4]: slot 1 (+0x20) the squares' T# (BC7, 7 mips); s[24:25] the S#
s_load_dwordx8 s[28:35], s[22:23], 0x20
s_load_dwordx4 s[36:39], s[24:25], 0x0
v_interp_p1_f32_e32 v4, v0, attr1.x
v_interp_p2_f32_e32 v4, v1, attr1.x
v_interp_p1_f32_e32 v5, v0, attr1.z
v_interp_p2_f32_e32 v5, v1, attr1.z
v_interp_p1_f32_e32 v6, v0, attr1.y
v_interp_p2_f32_e32 v6, v1, attr1.y
v_add_f32_e32 v8, v4, v5
v_mul_f32_e32 v8, 0.5, v8
v_mul_f32_e32 v9, 0.5, v6
v_mov_b32_e32 v10, s19
s_waitcnt lgkmcnt(0)
image_sample_l v[12:15], v[8:10], s[28:35], s[36:39] dmask:0xf dim:SQ_RSRC_IMG_2D
v_interp_mov_f32 v16, p0, attr0.x
v_interp_mov_f32 v17, p0, attr0.y
v_interp_mov_f32 v18, p0, attr0.z
s_waitcnt vmcnt(0)
v_mul_f32_e32 v12, v12, v16
v_mul_f32_e32 v13, v13, v17
v_mul_f32_e32 v14, v14, v18
v_mov_b32_e32 v15, 1.0
exp mrt0 v12, v13, v14, v15 done vm
s_endpgm
