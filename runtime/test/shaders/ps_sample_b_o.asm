; image_sample_b_o: texel offset and LOD bias together (RDNA2 ISA, MIMG address order: offset, bias, x, y). The
; first dword packs signed 6-bit texel offsets, x = [5:0], y = [13:8]; the bias is added to the implicit LOD.
; Run: vk_texture_test vs_ids.spv ps_sample_b_o.spv out.raw 0 92 c00000 14500000 0 --two-mip
;      (VKT_NEED_MAINTENANCE8=1: exit 77 without the extension, as ps_texoff)
; The texture and sampler are those of ps_sample_l.asm (128x128 base level in four quadrants, cyan mip 1, point
; filters, mip filter point, nearest mip). u, v = (x + 0.5) / 64, (y + 0.5) / 64 at pixel (x, y), so the implicit
; LOD is log2(2 texels a pixel) = 1.0, and the texel a sample reads at mip 0 is (2x + 1, 2y + 1) before its offset.
; Offset dword 0x3c06: x = 0x06 = +6, y = 0x3c = -4 (even, so a rounding of u*128 to 2x cannot change the quadrant).
; Bias: -1.0 where u < 0.75 (x < 48), 0.0 where u >= 0.75 (x >= 48).
; Expected mrt0 (r, g, b, a) at pixel (x, y), derived from that, not from a run:
;   x >= 48:  LOD 1.0 + 0.0 -> mip 1, cyan whatever the offset (it stays inside mip 1):    (0, 1, 1, 1)
;   x < 48:   LOD 1.0 - 1.0 = 0 -> mip 0, texel (2x + 7, 2y - 3), clamped to the edge above 0:
;               2x + 7 >= 64 <=> x >= 29 (right half), 2y - 3 >= 64 <=> y >= 34 (lower half)
;             x < 29,  y < 34:   red     (1, 0, 0, 1)
;             x >= 29, y < 34:   green   (0, 1, 0, 1)
;             x < 29,  y >= 34:  blue    (0, 0, 1, 1)
;             x >= 29, y >= 34:  transparent yellow  (1, 1, 0, 0)
; Dropped bias: mip 1 for x < 48. Bias read as an explicit LOD: mip 0 for x >= 48. Dropped offset: the quadrant edges
; sit at x = 32 and y = 32. Offset fields swapped (x -4, y +6): edges at x >= 34 and y >= 29.
v_interp_p1_f32_e32 v2, v0, attr0.x
v_interp_p2_f32_e32 v2, v1, attr0.x
v_interp_p1_f32_e32 v3, v0, attr0.y
v_interp_p2_f32_e32 v3, v1, attr0.y
v_mov_b32_e32 v5, 0
v_cmp_le_f32_e32 vcc_lo, 0x3f400000, v2
v_cndmask_b32_e32 v11, -1.0, v5, vcc_lo
v_mov_b32_e32 v10, 0x3c06
v_mov_b32_e32 v12, v2
v_mov_b32_e32 v13, v3
s_load_dwordx8 s[8:15], s[0:1], 0x0
s_waitcnt lgkmcnt(0)
s_mov_b32 s16, 0x92
s_mov_b32 s17, 0xc00000
s_mov_b32 s18, 0x14500000
s_mov_b32 s19, 0
image_sample_b_o v[6:9], v[10:13], s[8:15], s[16:19] dmask:0xf dim:SQ_RSRC_IMG_2D
s_waitcnt vmcnt(0)
exp mrt0 v6, v7, v8, v9 done vm
s_endpgm
