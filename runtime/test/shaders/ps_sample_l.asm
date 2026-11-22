; image_sample_l: the LOD comes from the VGPR after the coordinates (RDNA2 ISA, MIMG: address = x, y, lod).
; Run: vk_texture_test vs_ids.spv ps_sample_l.spv out.raw 0 92 c00000 14500000 0 --two-mip
; --two-mip: a 128x128 base level with four quadrants (top left red, top right green, bottom left blue, bottom
; right transparent yellow) and a solid cyan 64x64 mip 1; the S# is point filtered with LOD 0..12 and mip filter
; point, so a sample uses the one mip nearest to its LOD (round to nearest: 0.25 -> mip 0, 0.75 -> mip 1).
; vs_ids gives u, v = (x + 0.5) / 64, (y + 0.5) / 64 at pixel (x, y) of the 64x64 target, so an implicit sample
; spans two base texels a pixel and would use LOD 1.0 here.
; The LOD is 0.25 where u < 0.5 (x < 32) and 0.75 where u >= 0.5 (pixel centres never sit on 0.5).
; Expected mrt0 (r, g, b, a) at pixel (x, y), derived from that and the texture, not from a run:
;   x < 32, y < 32:   LOD 0.25 -> mip 0, texel (2x + 1, 2y + 1) in the red quadrant:  (1, 0, 0, 1)
;   x < 32, y >= 32:  mip 0, blue quadrant:                                           (0, 0, 1, 1)
;   x >= 32 (any y):  LOD 0.75 -> mip 1, cyan:                                        (0, 1, 1, 1)
; A LOD operand that is dropped turns the sample implicit (LOD 1.0, so mip 1 everywhere) or LZ (mip 0 everywhere);
; one read as a bias gives 1.25 and 1.75 (mip 1 everywhere): each changes one half of the frame.
; u and v swapped would put green in the bottom left quarter.
v_interp_p1_f32_e32 v2, v0, attr0.x
v_interp_p2_f32_e32 v2, v1, attr0.x
v_interp_p1_f32_e32 v3, v0, attr0.y
v_interp_p2_f32_e32 v3, v1, attr0.y
v_mov_b32_e32 v5, 0x3f400000
v_cmp_le_f32_e32 vcc_lo, 0.5, v2
v_cndmask_b32_e32 v4, 0x3e800000, v5, vcc_lo
s_load_dwordx8 s[8:15], s[0:1], 0x0
s_waitcnt lgkmcnt(0)
s_mov_b32 s16, 0x92
s_mov_b32 s17, 0xc00000
s_mov_b32 s18, 0x14500000
s_mov_b32 s19, 0
image_sample_l v[6:9], v[2:4], s[8:15], s[16:19] dmask:0xf dim:SQ_RSRC_IMG_2D
s_waitcnt vmcnt(0)
exp mrt0 v6, v7, v8, v9 done vm
s_endpgm
