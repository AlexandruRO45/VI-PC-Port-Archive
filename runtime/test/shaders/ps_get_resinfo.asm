; image_get_resinfo (TEST-184): queries dimensions and mip count of a non-square multi-mip 2D texture across
; mip levels 0, 1 and 2, with full (0xf) and sparse (0x9, 0x6) dmask compaction.
; Run: vk_texture_test vs_ids.spv ps_get_resinfo.spv out.raw 0 92 c00000 14500000 0 --rect-mip
; --rect-mip binds a 16x8 2D texture with 3 mip levels (mip 0: 16x8, mip 1: 8x4, mip 2: 4x2).
;   1. lod 0, dmask:0xf -> (width=16, height=8, depth=1, levels=3)
;   2. lod 1, dmask:0x9 -> (width=8, levels=3)
;   3. lod 2, dmask:0x6 -> (height=2, depth=1)
; Each pair (a, b) is packed as float(a + 256*b) (exact integers in binary32, immune to subnormal FTZ):
;   mrt0 = (16 + 8*256, 1 + 3*256, 8 + 3*256, 2 + 1*256) = (2064.0, 769.0, 776.0, 258.0) on all 4096 pixels.
s_load_dwordx8 s[8:15], s[0:1], 0x0
s_waitcnt lgkmcnt(0)
v_mov_b32_e32 v1, 0
image_get_resinfo v[4:7], v1, s[8:15] dmask:0xf dim:SQ_RSRC_IMG_2D
v_mov_b32_e32 v2, 1
image_get_resinfo v[8:9], v2, s[8:15] dmask:0x9 dim:SQ_RSRC_IMG_2D
v_mov_b32_e32 v3, 2
image_get_resinfo v[10:11], v3, s[8:15] dmask:0x6 dim:SQ_RSRC_IMG_2D
s_waitcnt vmcnt(0)
v_lshl_add_u32 v12, v5, 8, v4
v_lshl_add_u32 v13, v7, 8, v6
v_lshl_add_u32 v14, v9, 8, v8
v_lshl_add_u32 v15, v11, 8, v10
v_cvt_f32_u32_e32 v12, v12
v_cvt_f32_u32_e32 v13, v13
v_cvt_f32_u32_e32 v14, v14
v_cvt_f32_u32_e32 v15, v15
exp mrt0 v12, v13, v14, v15 done vm
s_endpgm
