; image_get_lod a16: the two half-precision coordinates share the first address VGPR, u in the low half and v in the
; high one. ps_get_lod.asm has the plain 2D form, the LOD rules and the sampler words; the same four queries run here
; with the coordinates packed as f16 pairs.
; Run: vk_texture_test vs_ids.spv ps_get_lod_a16.spv out.raw 0 92 <S# word 1> 14500000 0 --two-mip, once per word 1.
; --two-mip draws 64x64 pixels over a 128x128 texture that has two levels, with u and v from 0 to 1, so a pixel step
; is two texels each way and the calculated LOD is log2(2) = 1. Scaling one axis by 4 makes its step eight texels,
; and the query keeps the larger of the two steps: log2(8) = 3, which no level of this image reaches, so the clamped
; LOD stops at the last level, 1. The (u, 4v) query is the one that fails when V is lost: with both coordinates taken
; from the low half its step is diagonal and its LOD falls to about 1.5 (log2(2 * sqrt(2)) under the derivative-length
; rule Mesa uses, 1.0 under the larger-component one), where this fixture wants 3.0. Every u, v and 4v here is exactly
; representable in f16 (odd / 128, odd / 32), so the packed coordinates carry no rounding.
;   mrt0 = (clamped at (u, v), calculated at (u, v), calculated at (4u, v), calculated at (u, 4v))
;   S# word 1 = 0xc00000 (LOD 0..12):   1.0 1.0 3.0 3.0
;   S# word 1 = 0x40000  (LOD 0..0.25): 0.0 1.0 3.0 3.0    MAX_LOD limits the clamped LOD to level 0, the calculated
;                                                          LOD is not limited
; Every pixel must hold these within 1/16 (a device may keep as few as four fraction bits of LOD).
; Not shown by this test: u and v taken in the wrong order. The texture is square, so the two axes are
; interchangeable and swapping them leaves every result here unchanged.
v_interp_p1_f32_e32 v2, v0, attr0.x
v_interp_p2_f32_e32 v2, v1, attr0.x
v_interp_p1_f32_e32 v3, v0, attr0.y
v_interp_p2_f32_e32 v3, v1, attr0.y
s_load_dwordx8 s[8:15], s[0:1], 0
s_load_dwordx4 s[16:19], s[0:1], 0x20
s_waitcnt lgkmcnt(0)
v_mul_f32_e32 v6, 4.0, v2
v_mul_f32_e32 v7, 4.0, v3
v_cvt_pkrtz_f16_f32 v4, v2, v3
v_cvt_pkrtz_f16_f32 v5, v6, v3
v_cvt_pkrtz_f16_f32 v8, v2, v7
image_get_lod v10, v4, s[8:15], s[16:19] dmask:0x1 dim:SQ_RSRC_IMG_2D a16
image_get_lod v11, v4, s[8:15], s[16:19] dmask:0x2 dim:SQ_RSRC_IMG_2D a16
image_get_lod v12, v5, s[8:15], s[16:19] dmask:0x2 dim:SQ_RSRC_IMG_2D a16
image_get_lod v13, v8, s[8:15], s[16:19] dmask:0x2 dim:SQ_RSRC_IMG_2D a16
s_waitcnt vmcnt(0)
exp mrt0 v10, v11, v12, v13 done vm
s_endpgm
