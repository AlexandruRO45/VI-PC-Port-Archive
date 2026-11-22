; image_load_mip: an unfiltered texel fetch at an explicit mip level (RDNA2 ISA, MIMG: address = x, y, mip).
; Run: vk_texture_test vs_ids.spv ps_load_mip.spv out.raw 0 92 c00000 14500000 0 --two-mip
; The texture is the one of ps_sample_l.asm: 128x128 base level (top left red, top right green, bottom left blue,
; bottom right transparent yellow, all RGBA8 UNORM) and a solid cyan 64x64 mip 1. u, v = (x + 0.5) / 64,
; (y + 0.5) / 64 at pixel (x, y) of the 64x64 target.
;   A = image_load_mip at texel (trunc(128u), trunc(128v)) = (2x + 1 or 2x, 2y + 1 or 2y), mip 0: the quadrant colour
;   B = image_load_mip at texel (trunc(64u), trunc(64v)) = (x, y), mip 1: cyan, inside the 64x64 level
;   mrt0 = (A.r, A.g, A.b, B.g)
; Expected mrt0 at pixel (x, y), derived from the texture, not from a run (a texel pair 2x, 2x + 1 never straddles
; the quadrant edge at 64); the last component is B.g = 1 everywhere:
;   x < 32,  y < 32:   red      (1, 0, 0, 1)
;   x >= 32, y < 32:   green    (0, 1, 0, 1)
;   x < 32,  y >= 32:  blue     (0, 0, 1, 1)
;   x >= 32, y >= 32:  yellow   (1, 1, 0, 1)
; B.g is 1 for cyan; a mip operand dropped or read as 0 fetches mip 0 at (x, y) < 64, the red quadrant, with g = 0.
; A mip operand read as 1 for A gives cyan, (0, 1, 1, 1), in the A part: caught by A.b and A.r.
v_interp_p1_f32_e32 v2, v0, attr0.x
v_interp_p2_f32_e32 v2, v1, attr0.x
v_interp_p1_f32_e32 v3, v0, attr0.y
v_interp_p2_f32_e32 v3, v1, attr0.y
v_mul_f32_e32 v4, 0x43000000, v2
v_mul_f32_e32 v5, 0x43000000, v3
v_cvt_u32_f32_e32 v4, v4
v_cvt_u32_f32_e32 v5, v5
v_mov_b32_e32 v6, 0
v_mul_f32_e32 v16, 0x42800000, v2
v_mul_f32_e32 v17, 0x42800000, v3
v_cvt_u32_f32_e32 v16, v16
v_cvt_u32_f32_e32 v17, v17
v_mov_b32_e32 v18, 1
s_load_dwordx8 s[8:15], s[0:1], 0x0
s_waitcnt lgkmcnt(0)
image_load_mip v[8:11], v[4:6], s[8:15] dmask:0xf dim:SQ_RSRC_IMG_2D unorm
image_load_mip v[12:15], v[16:18], s[8:15] dmask:0xf dim:SQ_RSRC_IMG_2D unorm
s_waitcnt vmcnt(0)
exp mrt0 v8, v9, v10, v13 done vm
s_endpgm
