; image_gather4_lz (no offset): the 2x2 footprint of one channel at mip 0, in gather order.
; Run: vk_texture_test vs_ids.spv ps_gather4_lz.spv out.raw 0 92 c00000 14500000 0
; The tool binds an 8x8 RGBA8 texture whose texel (x, y) is r = 32x/255, g = 32y/255, b = 128/255, a = 1 and a point
; sampler that clamps to the edge (S# 0x92: clamp x/y/z), and draws 64x64 pixels with u, v = (px + 0.5) / 64,
; (py + 0.5) / 64. Each gathered channel c is turned back into an integer with rndne(c * 255/32) (255/32 = 7.96875
; exactly): red and green give the texel's x and y, blue 128/32 = 4, alpha 255/32 -> 8.
; Gather order (D3D and Vulkan, component k -> v[d + k]): taps (i0, j1), (i1, j1), (i1, j0), (i0, j0), where the
; footprint of coordinate (s, t) in texels is i0 = floor(s - 0.5), i1 = i0 + 1, j0 = floor(t - 0.5), j1 = j0 + 1.
; With bx = px >> 3, by = py >> 3 (0..7), the shader moves u, v to the texel corner (bx + 1) / 8, (by + 1) / 8
; (exact in floats), so the footprint is i0 = bx, j0 = by, and i1 = bx + 1, j1 = by + 1 clamped to 7 by the
; sampler (bx = 7 gives 8 -> 7). Let X0 = bx, X1 = min(bx + 1, 7), Y0 = by, Y1 = min(by + 1, 7). Red taps hold x,
; green taps hold y.
; Expected mrt0 words at pixel (px, py), derived from that and the texture, not from a run (one nibble a tap, tap 0
; lowest):
;   .x  red at the corner:    X0 | X1 << 4 | X1 << 8 | X0 << 12
;   .y  green at the corner:  Y1 | Y1 << 4 | Y0 << 8 | Y0 << 12
;   .z  blue at the corner, 4 in every tap, in the low half, and alpha, 8 in every tap, in the high half: 0x88884444
;   .w  one more pair of gathers at the fixed coordinate (0.34375, 0.84375) = texel coordinate (2.75, 6.75): footprint
;       i0 = 2, i1 = 3, j0 = 6, j1 = 7. Red: 2 | 3 << 4 | 3 << 8 | 2 << 12 = 0x2332 in the low half, green:
;       7 | 7 << 4 | 6 << 8 | 6 << 12 = 0x6677 in the high half: 0x66772332
; Examples: pixel (0, 0) (bx = by = 0):  .x = 0x0110, .y = 0x0011.
;           pixel (63, 63) (bx = by = 7): .x = 0x7777, .y = 0x7777.
;           pixel (20, 50) (bx 2, by 6):  .x = 0x2332, .y = 0x6677.
; A gather of the wrong channel (the dmask) moves a result to another word, swapped taps change the nibble order, and
; u and v swapped put the y footprint in .x.
v_interp_p1_f32_e32 v2, v0, attr0.x
v_interp_p2_f32_e32 v2, v1, attr0.x
v_interp_p1_f32_e32 v3, v0, attr0.y
v_interp_p2_f32_e32 v3, v1, attr0.y
v_mul_f32_e32 v4, 0x41000000, v2
v_floor_f32_e32 v4, v4
v_add_f32_e32 v4, 1.0, v4
v_mul_f32_e32 v4, 0x3e000000, v4
v_mul_f32_e32 v5, 0x41000000, v3
v_floor_f32_e32 v5, v5
v_add_f32_e32 v5, 1.0, v5
v_mul_f32_e32 v5, 0x3e000000, v5
v_mov_b32_e32 v32, 0x3eb00000
v_mov_b32_e32 v33, 0x3f580000
s_load_dwordx8 s[8:15], s[0:1], 0x0
s_waitcnt lgkmcnt(0)
s_mov_b32 s16, 0x92
s_mov_b32 s17, 0xc00000
s_mov_b32 s18, 0x14500000
s_mov_b32 s19, 0
image_gather4_lz v[8:11], v[4:5], s[8:15], s[16:19] dmask:0x1 dim:SQ_RSRC_IMG_2D
image_gather4_lz v[12:15], v[4:5], s[8:15], s[16:19] dmask:0x2 dim:SQ_RSRC_IMG_2D
image_gather4_lz v[16:19], v[4:5], s[8:15], s[16:19] dmask:0x4 dim:SQ_RSRC_IMG_2D
image_gather4_lz v[20:23], v[4:5], s[8:15], s[16:19] dmask:0x8 dim:SQ_RSRC_IMG_2D
image_gather4_lz v[24:27], v[32:33], s[8:15], s[16:19] dmask:0x1 dim:SQ_RSRC_IMG_2D
image_gather4_lz v[28:31], v[32:33], s[8:15], s[16:19] dmask:0x2 dim:SQ_RSRC_IMG_2D
s_waitcnt vmcnt(0)
v_mul_f32_e32 v8, 0x40ff0000, v8
v_rndne_f32_e32 v8, v8
v_cvt_u32_f32_e32 v8, v8
v_mul_f32_e32 v9, 0x40ff0000, v9
v_rndne_f32_e32 v9, v9
v_cvt_u32_f32_e32 v9, v9
v_mul_f32_e32 v10, 0x40ff0000, v10
v_rndne_f32_e32 v10, v10
v_cvt_u32_f32_e32 v10, v10
v_mul_f32_e32 v11, 0x40ff0000, v11
v_rndne_f32_e32 v11, v11
v_cvt_u32_f32_e32 v11, v11
v_mul_f32_e32 v12, 0x40ff0000, v12
v_rndne_f32_e32 v12, v12
v_cvt_u32_f32_e32 v12, v12
v_mul_f32_e32 v13, 0x40ff0000, v13
v_rndne_f32_e32 v13, v13
v_cvt_u32_f32_e32 v13, v13
v_mul_f32_e32 v14, 0x40ff0000, v14
v_rndne_f32_e32 v14, v14
v_cvt_u32_f32_e32 v14, v14
v_mul_f32_e32 v15, 0x40ff0000, v15
v_rndne_f32_e32 v15, v15
v_cvt_u32_f32_e32 v15, v15
v_mul_f32_e32 v16, 0x40ff0000, v16
v_rndne_f32_e32 v16, v16
v_cvt_u32_f32_e32 v16, v16
v_mul_f32_e32 v17, 0x40ff0000, v17
v_rndne_f32_e32 v17, v17
v_cvt_u32_f32_e32 v17, v17
v_mul_f32_e32 v18, 0x40ff0000, v18
v_rndne_f32_e32 v18, v18
v_cvt_u32_f32_e32 v18, v18
v_mul_f32_e32 v19, 0x40ff0000, v19
v_rndne_f32_e32 v19, v19
v_cvt_u32_f32_e32 v19, v19
v_mul_f32_e32 v20, 0x40ff0000, v20
v_rndne_f32_e32 v20, v20
v_cvt_u32_f32_e32 v20, v20
v_mul_f32_e32 v21, 0x40ff0000, v21
v_rndne_f32_e32 v21, v21
v_cvt_u32_f32_e32 v21, v21
v_mul_f32_e32 v22, 0x40ff0000, v22
v_rndne_f32_e32 v22, v22
v_cvt_u32_f32_e32 v22, v22
v_mul_f32_e32 v23, 0x40ff0000, v23
v_rndne_f32_e32 v23, v23
v_cvt_u32_f32_e32 v23, v23
v_mul_f32_e32 v24, 0x40ff0000, v24
v_rndne_f32_e32 v24, v24
v_cvt_u32_f32_e32 v24, v24
v_mul_f32_e32 v25, 0x40ff0000, v25
v_rndne_f32_e32 v25, v25
v_cvt_u32_f32_e32 v25, v25
v_mul_f32_e32 v26, 0x40ff0000, v26
v_rndne_f32_e32 v26, v26
v_cvt_u32_f32_e32 v26, v26
v_mul_f32_e32 v27, 0x40ff0000, v27
v_rndne_f32_e32 v27, v27
v_cvt_u32_f32_e32 v27, v27
v_mul_f32_e32 v28, 0x40ff0000, v28
v_rndne_f32_e32 v28, v28
v_cvt_u32_f32_e32 v28, v28
v_mul_f32_e32 v29, 0x40ff0000, v29
v_rndne_f32_e32 v29, v29
v_cvt_u32_f32_e32 v29, v29
v_mul_f32_e32 v30, 0x40ff0000, v30
v_rndne_f32_e32 v30, v30
v_cvt_u32_f32_e32 v30, v30
v_mul_f32_e32 v31, 0x40ff0000, v31
v_rndne_f32_e32 v31, v31
v_cvt_u32_f32_e32 v31, v31
v_lshl_or_b32 v40, v9, 4, v8
v_lshl_or_b32 v40, v10, 8, v40
v_lshl_or_b32 v40, v11, 12, v40
v_lshl_or_b32 v41, v13, 4, v12
v_lshl_or_b32 v41, v14, 8, v41
v_lshl_or_b32 v41, v15, 12, v41
v_lshl_or_b32 v42, v17, 4, v16
v_lshl_or_b32 v42, v18, 8, v42
v_lshl_or_b32 v42, v19, 12, v42
v_lshl_or_b32 v43, v21, 4, v20
v_lshl_or_b32 v43, v22, 8, v43
v_lshl_or_b32 v43, v23, 12, v43
v_lshl_or_b32 v44, v25, 4, v24
v_lshl_or_b32 v44, v26, 8, v44
v_lshl_or_b32 v44, v27, 12, v44
v_lshl_or_b32 v45, v29, 4, v28
v_lshl_or_b32 v45, v30, 8, v45
v_lshl_or_b32 v45, v31, 12, v45
v_lshl_or_b32 v42, v43, 16, v42
v_lshl_or_b32 v44, v45, 16, v44
exp mrt0 v40, v41, v42, v44 done vm
s_endpgm
