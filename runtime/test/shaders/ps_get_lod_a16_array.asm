; image_get_lod a16 on a 2D array (dim:SQ_RSRC_IMG_2D_ARRAY): the packed coordinates stay in the first address VGPR
; and the slice is a second address component that takes no part in the LOD. ps_get_lod_array.asm has the plain 2D
; array form and the rules; this fixture packs the coordinates as f16 and puts f16 7.0 in each second address
; register, the slice, which the LOD query does not read, so a slice taken as a coordinate would show up in the result.
; Run: vk_texture_test vs_ids.spv ps_get_lod_a16_array.spv out.raw 0 92 c00000 14500000 0
; The tool's 2D array (set 0 binding 3) is 8x8 with one level, drawn over 64x64 pixels with u and v from 0 to 1: a
; pixel step is 1/8 texel, so the calculated LOD is log2(1/8) = -3. Scaling one axis by 32 makes its step four
; texels, log2(4) = 2, and with a single level the clamped LOD is 0 for every query. The (u, 32v) query is the one
; that fails when v is lost. u and v are odd / 128 and 32u, 32v are odd / 4, all exact in f16.
;   mrt0 = (clamped at (u, v), calculated at (u, v), calculated at (32u, v), calculated at (u, 32v))
;        = 0.0 -3.0 2.0 2.0
; on every pixel, within 1/16 (a device may keep as few as four fraction bits of LOD).
v_interp_p1_f32_e32 v2, v0, attr0.x
v_interp_p2_f32_e32 v2, v1, attr0.x
v_interp_p1_f32_e32 v3, v0, attr0.y
v_interp_p2_f32_e32 v3, v1, attr0.y
v_mul_f32_e32 v10, 32.0, v2
v_mul_f32_e32 v11, 32.0, v3
v_mov_b32_e32 v5, 0x4700
v_mov_b32_e32 v7, 0x4700
v_mov_b32_e32 v9, 0x4700
v_cvt_pkrtz_f16_f32 v4, v2, v3
v_cvt_pkrtz_f16_f32 v6, v10, v3
v_cvt_pkrtz_f16_f32 v8, v2, v11
s_load_dwordx8 s[8:15], s[0:1], 0
s_waitcnt lgkmcnt(0)
s_mov_b32 s16, 0x92
s_mov_b32 s17, 0xc00000
s_mov_b32 s18, 0x14500000
s_mov_b32 s19, 0
image_get_lod v12, v[4:5], s[8:15], s[16:19] dmask:0x1 dim:SQ_RSRC_IMG_2D_ARRAY a16
image_get_lod v13, v[4:5], s[8:15], s[16:19] dmask:0x2 dim:SQ_RSRC_IMG_2D_ARRAY a16
image_get_lod v14, v[6:7], s[8:15], s[16:19] dmask:0x2 dim:SQ_RSRC_IMG_2D_ARRAY a16
image_get_lod v15, v[8:9], s[8:15], s[16:19] dmask:0x2 dim:SQ_RSRC_IMG_2D_ARRAY a16
s_waitcnt vmcnt(0)
exp mrt0 v12, v13, v14, v15 done vm
s_endpgm
