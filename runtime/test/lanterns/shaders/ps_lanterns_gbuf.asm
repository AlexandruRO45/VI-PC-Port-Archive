; Lanterns G-buffer pixel shader (stage 9): three render targets. MRT0 the albedo (RGBA8_UNORM), MRT1 the normal's
; three 8-bit codes as code / 255 (R10G10B10A2_UNORM; alpha 1), MRT2 the view depth (R32_FLOAT): the clip w of the
; interpolated world position, from the view-projection's last row (root CBV + 0x30), in vs_lanterns's fma order.
; stage: ps
; wave: 32
; input: flat
; input: interp
; input: flat
s_load_dwordx4 s[24:27], s[20:21], 0x30
v_interp_p1_f32_e32 v4, v0, attr1.x
v_interp_p2_f32_e32 v4, v1, attr1.x
v_interp_p1_f32_e32 v5, v0, attr1.y
v_interp_p2_f32_e32 v5, v1, attr1.y
v_interp_p1_f32_e32 v6, v0, attr1.z
v_interp_p2_f32_e32 v6, v1, attr1.z
v_interp_mov_f32 v8, p0, attr0.x
v_interp_mov_f32 v9, p0, attr0.y
v_interp_mov_f32 v10, p0, attr0.z
v_interp_mov_f32 v11, p0, attr0.w
v_interp_mov_f32 v12, p0, attr2.x
s_waitcnt lgkmcnt(0)
v_fma_f32 v13, s24, v4, s27
v_fmac_f32_e32 v13, s25, v5
v_fmac_f32_e32 v13, s26, v6
v_cvt_u32_f32_e32 v12, v12
v_bfe_u32 v14, v12, 0, 8
v_bfe_u32 v15, v12, 8, 8
v_bfe_u32 v16, v12, 16, 8
v_cvt_f32_u32_e32 v14, v14
v_cvt_f32_u32_e32 v15, v15
v_cvt_f32_u32_e32 v16, v16
v_mul_f32_e32 v14, 0x3b808081, v14
v_mul_f32_e32 v15, 0x3b808081, v15
v_mul_f32_e32 v16, 0x3b808081, v16
v_mov_b32_e32 v17, 1.0
exp mrt0 v8, v9, v10, v11
exp mrt1 v14, v15, v16, v17
exp mrt2 v13, v13, v13, v13 done vm
s_endpgm
