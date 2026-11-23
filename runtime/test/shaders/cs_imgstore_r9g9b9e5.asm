; group: 64,1,1
; cs_imgstore_r9g9b9e5: typed image_store to T# FORMAT 132 (GFX10_FORMAT_5_9_9_9_FLOAT / R9G9B9E5_SHAREDEXP) across
; normal floats, RTZ mantissa boundaries, subnormals, negatives, -0.0, +/-Inf, +/-qNaN/+sNaN, overflow >= 65408, sparse
; dmasks (0x5, 0x8), glc a16 d16, inactive-lane EXEC masking (leaving sentinel 0xdeadbeef intact), plus an RGBA8_UNORM
; (FORMAT 56) control store. On GPUs without native E5B9G9R9 storage (RTX 5090 D), --r9g9b9e5-uav-emulate packs into R32_UINT
; on %stv1; without the flag it hits %stv0 and misses img_e5. Run by vk_store_test <spv> r9g9b9e5 (checked in check.py).

	v_and_b32_e32 v1, 7, v0
	v_lshrrev_b32_e32 v2, 3, v0
	s_mov_b32 s8, 0x6000
	s_mov_b32 s9, 0x08400000
	s_mov_b32 s10, 0x66
	s_mov_b32 s11, 0x66
	s_mov_b32 s12, 0x66
	s_mov_b32 s13, 0x66
	s_mov_b32 s14, 0x66
	s_mov_b32 s15, 0x66

	; Default rows 0..3 (tid = 0..31): deterministic varied values + power-of-2 exponent sweep in B.
	v_add_nc_u32_e32 v7, -8, v0
	v_cvt_f32_i32_e32 v3, v7
	v_mul_f32_e32 v3, 0x3ea00000, v3
	v_mul_lo_u32 v7, v0, 3
	v_and_b32_e32 v7, 15, v7
	v_cvt_f32_u32_e32 v4, v7
	v_mul_f32_e32 v4, 0x3e000000, v4
	v_add_f32_e32 v4, 0x3a03126f, v4
	v_add_nc_u32_e32 v7, 102, v0
	v_lshlrev_b32_e32 v5, 23, v7
	v_mov_b32_e32 v6, 0x42c60000

	; Row 4 overrides (tid = 32..39): IEEE edge, NaN, Inf, overflow, subnormal and RTZ boundary triples.
	v_cmp_eq_u32_e32 vcc_lo, 32, v0
	v_mov_b32_e32 v7, 0x3f800000
	v_cndmask_b32_e32 v3, v3, v7, vcc_lo
	v_mov_b32_e32 v7, 0x40000000
	v_cndmask_b32_e32 v4, v4, v7, vcc_lo
	v_mov_b32_e32 v7, 0x40800000
	v_cndmask_b32_e32 v5, v5, v7, vcc_lo

	v_cmp_eq_u32_e32 vcc_lo, 33, v0
	v_mov_b32_e32 v7, 0x3f7f8000
	v_cndmask_b32_e32 v3, v3, v7, vcc_lo
	v_mov_b32_e32 v7, 0x3f7fa000
	v_cndmask_b32_e32 v4, v4, v7, vcc_lo
	v_mov_b32_e32 v7, 0x3f7fc000
	v_cndmask_b32_e32 v5, v5, v7, vcc_lo

	v_cmp_eq_u32_e32 vcc_lo, 34, v0
	v_mov_b32_e32 v7, 0xbf800000
	v_cndmask_b32_e32 v3, v3, v7, vcc_lo
	v_mov_b32_e32 v7, 0x80000000
	v_cndmask_b32_e32 v4, v4, v7, vcc_lo
	v_mov_b32_e32 v7, 0x3f000000
	v_cndmask_b32_e32 v5, v5, v7, vcc_lo

	v_cmp_eq_u32_e32 vcc_lo, 35, v0
	v_mov_b32_e32 v7, 0x7f800000
	v_cndmask_b32_e32 v3, v3, v7, vcc_lo
	v_mov_b32_e32 v7, 0xff800000
	v_cndmask_b32_e32 v4, v4, v7, vcc_lo
	v_mov_b32_e32 v7, 0x7fc00001
	v_cndmask_b32_e32 v5, v5, v7, vcc_lo

	v_cmp_eq_u32_e32 vcc_lo, 36, v0
	v_mov_b32_e32 v7, 0x7f800001
	v_cndmask_b32_e32 v3, v3, v7, vcc_lo
	v_mov_b32_e32 v7, 0x3f800000
	v_cndmask_b32_e32 v4, v4, v7, vcc_lo
	v_mov_b32_e32 v7, 0x40000000
	v_cndmask_b32_e32 v5, v5, v7, vcc_lo

	v_cmp_eq_u32_e32 vcc_lo, 37, v0
	v_mov_b32_e32 v7, 0x477f4000
	v_cndmask_b32_e32 v3, v3, v7, vcc_lo
	v_mov_b32_e32 v7, 0x477f7f00
	v_cndmask_b32_e32 v4, v4, v7, vcc_lo
	v_mov_b32_e32 v7, 0x477f8000
	v_cndmask_b32_e32 v5, v5, v7, vcc_lo

	v_cmp_eq_u32_e32 vcc_lo, 38, v0
	v_mov_b32_e32 v7, 0x477fe000
	v_cndmask_b32_e32 v3, v3, v7, vcc_lo
	v_mov_b32_e32 v7, 0x477f8100
	v_cndmask_b32_e32 v4, v4, v7, vcc_lo
	v_mov_b32_e32 v7, 0x60ad78ec
	v_cndmask_b32_e32 v5, v5, v7, vcc_lo

	v_cmp_eq_u32_e32 vcc_lo, 39, v0
	v_mov_b32_e32 v7, 0x33000000
	v_cndmask_b32_e32 v3, v3, v7, vcc_lo
	v_mov_b32_e32 v7, 0x33c00000
	v_cndmask_b32_e32 v4, v4, v7, vcc_lo
	v_mov_b32_e32 v7, 0x37ff8000
	v_cndmask_b32_e32 v5, v5, v7, vcc_lo

	; Rows 0..4 (tid < 40): full dmask:0xf store to FORMAT 132.
	v_cmp_gt_u32_e32 vcc_lo, 40, v0
	s_and_saveexec_b32 s4, vcc_lo
	image_store v[3:6], v[1:2], s[8:15] dmask:0xf dim:SQ_RSRC_IMG_2D unorm
	s_mov_b32 exec_lo, s4

	; tid 40..41: sparse dmask:0x5 (R=1.0, B=4.0; G=0).
	v_and_b32_e32 v7, 0xfffffffe, v0
	v_cmp_eq_u32_e32 vcc_lo, 40, v7
	s_and_saveexec_b32 s4, vcc_lo
	v_mov_b32_e32 v3, 0x3f800000
	v_mov_b32_e32 v4, 0x40800000
	image_store v[3:4], v[1:2], s[8:15] dmask:0x5 dim:SQ_RSRC_IMG_2D unorm
	s_mov_b32 exec_lo, s4

	; tid 42..43: dmask:0x8 (Alpha only -> R=G=B=0 -> 0x00000000).
	v_cmp_eq_u32_e32 vcc_lo, 42, v7
	s_and_saveexec_b32 s4, vcc_lo
	v_mov_b32_e32 v3, 0x40a00000
	image_store v3, v[1:2], s[8:15] dmask:0x8 dim:SQ_RSRC_IMG_2D unorm
	s_mov_b32 exec_lo, s4

	; tid 44..45: dmask:0x7 glc a16 d16 (f16 1.5, -2.25, 4.0; packed u16 (x, y) in v10).
	v_cmp_eq_u32_e32 vcc_lo, 44, v7
	s_and_saveexec_b32 s4, vcc_lo
	v_lshlrev_b32_e32 v10, 16, v2
	v_or_b32_e32 v10, v10, v1
	v_mov_b32_e32 v3, 0xc0803e00
	v_mov_b32_e32 v4, 0x7fff4400
	image_store v[3:4], v10, s[8:15] dmask:0x7 dim:SQ_RSRC_IMG_2D unorm glc a16 d16
	s_mov_b32 exec_lo, s4

	; tid 46..47: dmask:0xf glc coherent store ((0.25, 0.5, 1.0, 2.0) -> 0x84008040).
	v_cmp_eq_u32_e32 vcc_lo, 46, v7
	s_and_saveexec_b32 s4, vcc_lo
	v_mov_b32_e32 v3, 0x3e800000
	v_mov_b32_e32 v4, 0x3f000000
	v_mov_b32_e32 v5, 0x3f800000
	v_mov_b32_e32 v6, 0x40000000
	image_store v[3:6], v[1:2], s[8:15] dmask:0xf dim:SQ_RSRC_IMG_2D unorm glc
	s_mov_b32 exec_lo, s4

	; tid 48..51 with odd lanes masked off: 48 & 50 store 0x94010040; 49 & 51 keep sentinel 0xdeadbeef.
	v_and_b32_e32 v8, 0xfffffffc, v0
	v_cmp_eq_u32_e32 vcc_lo, 48, v8
	v_and_b32_e32 v9, 1, v0
	v_cmp_eq_u32_e64 s6, 0, v9
	s_and_b32 vcc_lo, vcc_lo, s6
	s_and_saveexec_b32 s4, vcc_lo
	v_mov_b32_e32 v3, 0x3f800000
	v_mov_b32_e32 v4, 0x40000000
	v_mov_b32_e32 v5, 0x40800000
	v_mov_b32_e32 v6, 0
	image_store v[3:6], v[1:2], s[8:15] dmask:0xf dim:SQ_RSRC_IMG_2D unorm
	s_mov_b32 exec_lo, s4

	; tid 0 control store to FORMAT 56 (RGBA8_UNORM, slot 1 = img[0]) at (0, 0).
	s_mov_b32 s16, 0x1000
	s_mov_b32 s17, 0x03800000
	s_mov_b32 s18, 0x11
	s_mov_b32 s19, 0x11
	s_mov_b32 s20, 0x11
	s_mov_b32 s21, 0x11
	s_mov_b32 s22, 0x11
	s_mov_b32 s23, 0x11
	v_cmp_eq_u32_e32 vcc_lo, 0, v0
	s_and_saveexec_b32 s4, vcc_lo
	v_mov_b32_e32 v3, 0x3f000000
	v_mov_b32_e32 v4, 0x3e800000
	v_mov_b32_e32 v5, 0x3f400000
	v_mov_b32_e32 v6, 0x3f800000
	image_store v[3:6], v[1:2], s[16:23] dmask:0xf dim:SQ_RSRC_IMG_2D unorm
	s_mov_b32 exec_lo, s4

	s_endpgm
