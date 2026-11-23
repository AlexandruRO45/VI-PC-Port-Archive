; Gather R and G from the 8x8 gradient, where R=32*x/255 and G=32*y/255. The
; four outputs are the linear texel indices x+8*y from the horizontal footprint,
; clamped at either edge. The G gather reads packed coordinates and returns f16s.
v_interp_p1_f32_e32 v2, v0, attr0.x
v_interp_p2_f32_e32 v2, v1, attr0.x
v_interp_p1_f32_e32 v3, v0, attr0.y
v_interp_p2_f32_e32 v3, v1, attr0.y
s_load_dwordx8 s[8:15], s[0:1], 0x0
s_waitcnt lgkmcnt(0)
s_mov_b32 s16, 0x92
s_mov_b32 s17, 0
s_mov_b32 s18, 0
s_mov_b32 s19, 0
v_cvt_pkrtz_f16_f32 v12, v2, v3
; At x % 8 == 3 the red gather is 1/1024 texel below a half-texel boundary.
; Rounding to eight fractional bits moves it onto the boundary.
v_add_f32 v2, 0x3bfc0000, v2
image_gather4h v[4:7], v[2:3], s[8:15], s[16:19] dmask:0x1 dim:SQ_RSRC_IMG_2D
image_gather4h v[8:9], v12, s[8:15], s[16:19] dmask:0x2 dim:SQ_RSRC_IMG_2D a16 d16
s_waitcnt vmcnt(0)
v_lshrrev_b32 v10, 16, v8
v_lshrrev_b32 v11, 16, v9
v_cvt_f32_f16 v8, v8
v_cvt_f32_f16 v10, v10
v_cvt_f32_f16 v9, v9
v_cvt_f32_f16 v11, v11
v_fma_f32 v4, v8, 8.0, v4
v_fma_f32 v5, v10, 8.0, v5
v_fma_f32 v6, v9, 8.0, v6
v_fma_f32 v7, v11, 8.0, v7
v_mul_f32 v4, 0x40ff0000, v4
v_mul_f32 v5, 0x40ff0000, v5
v_mul_f32 v6, 0x40ff0000, v6
v_mul_f32 v7, 0x40ff0000, v7
v_rndne_f32 v4, v4
v_rndne_f32 v5, v5
v_rndne_f32 v6, v6
v_rndne_f32 v7, v7
exp mrt0 v4, v5, v6, v7 done vm
s_endpgm
