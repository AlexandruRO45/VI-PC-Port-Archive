; image_get_lod (TEST-184): checks clamped (.x) and unclamped (.y) LOD on both 2D and 2D-array images, plus
; partial dmask compaction (dmask:0x1 and dmask:0x2).
; Run: vk_texture_test vs_ids.spv ps_get_lod.spv out.raw 0 92 100000 14500000 0 --two-mip
; Viewport is 64x64 with attr0 UV spanning [0, 1], so du/dx = dv/dy = 1/64 on every pixel.
; S# word 1 = 0x100000 -> minLod = 0.0, maxLod = 1.0.
;   1. 2D 128x128 (2 mips), uv = (4u, 4v): d(texel)/d(px) = 8 -> unclamped = log2(8) = 3.0, clamped = 1.0 (dmask:0x3).
;   2. 2D 128x128 (2 mips), uv = (0.25u, 0.25v): d(texel)/d(px) = 0.5 -> unclamped = -1.0 (dmask:0x2), clamped = 0.0 (dmask:0x1).
;   3. 2D_ARRAY 8x8 (1 mip, 12 layers), uv = (32u, 32v, layer=5): d(texel)/d(px) = 4 -> unclamped = 2.0, clamped = 0.0 (dmask:0x3).
; Exports mrt0 = (clamped_1, unclamped_1, unclamped_2, unclamped_3 + 16*clamped_3 + 64*clamped_2) = (1.0, 3.0, -1.0, 2.0).
v_interp_p1_f32_e32 v2, v0, attr0.x
v_interp_p2_f32_e32 v2, v1, attr0.x
v_interp_p1_f32_e32 v3, v0, attr0.y
v_interp_p2_f32_e32 v3, v1, attr0.y
s_load_dwordx8 s[8:15], s[0:1], 0x0
s_load_dwordx4 s[16:19], s[0:1], 0x20
s_waitcnt lgkmcnt(0)
v_mul_f32_e32 v10, 4.0, v2
v_mul_f32_e32 v11, 4.0, v3
image_get_lod v[4:5], v[10:11], s[8:15], s[16:19] dmask:0x3 dim:SQ_RSRC_IMG_2D
v_mul_f32_e32 v12, 0x3e800000, v2
v_mul_f32_e32 v13, 0x3e800000, v3
image_get_lod v6, v[12:13], s[8:15], s[16:19] dmask:0x2 dim:SQ_RSRC_IMG_2D
image_get_lod v7, v[12:13], s[8:15], s[16:19] dmask:0x1 dim:SQ_RSRC_IMG_2D
v_mul_f32_e32 v14, 0x42000000, v2
v_mul_f32_e32 v15, 0x42000000, v3
v_mov_b32_e32 v16, 0x40a00000
image_get_lod v[8:9], v[14:16], s[8:15], s[16:19] dmask:0x3 dim:SQ_RSRC_IMG_2D_ARRAY
s_waitcnt vmcnt(0)
v_mul_f32_e32 v8, 0x41800000, v8
v_mul_f32_e32 v7, 0x42800000, v7
v_add_f32_e32 v9, v9, v8
v_add_f32_e32 v9, v9, v7
exp mrt0 v4, v5, v6, v9 done vm
s_endpgm
