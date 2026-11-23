; Lanterns textured floor pixel shader (stage 6): the BC1 mosaic sampled trilinearly (image_sample: the LOD from the
; 2x2 quad's derivatives) at uv = world xz / 4, times the face colour. The FSSM input table is ps_lanterns_lit's.
; stage: ps
; wave: 32
; input: flat
; input: interp
;
; s[22:23] root table [4]: slot 0 the mosaic's T# (BC1, 9 mips), slot 1 (+0x20) the squares' T# (BC7)
; s[24:25] root table [5]: the S# (linear min/mag/mip, wrap)
s_load_dwordx8 s[28:35], s[22:23], 0x0
s_load_dwordx4 s[36:39], s[24:25], 0x0
v_interp_p1_f32_e32 v4, v0, attr1.x
v_interp_p2_f32_e32 v4, v1, attr1.x
v_interp_p1_f32_e32 v5, v0, attr1.z
v_interp_p2_f32_e32 v5, v1, attr1.z
v_mul_f32_e32 v4, 0.25, v4
v_mul_f32_e32 v5, 0.25, v5
s_waitcnt lgkmcnt(0)
image_sample v[6:9], v[4:5], s[28:35], s[36:39] dmask:0xf dim:SQ_RSRC_IMG_2D
v_interp_mov_f32 v10, p0, attr0.x
v_interp_mov_f32 v11, p0, attr0.y
v_interp_mov_f32 v12, p0, attr0.z
s_waitcnt vmcnt(0)
v_mul_f32_e32 v6, v6, v10
v_mul_f32_e32 v7, v7, v11
v_mul_f32_e32 v8, v8, v12
v_mov_b32_e32 v9, 1.0
exp mrt0 v6, v7, v8, v9 done vm
s_endpgm
