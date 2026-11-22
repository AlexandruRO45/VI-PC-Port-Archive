; Wave64 compare masks: v_cmp_* (i32, u32, i64, u64, f32 with NaN and infinity, class) into SGPR pairs and into VCC,
; the 64-bit SALU logic ops on those masks, and the masks used as v_cndmask selectors, over all 64 lanes of one group.
; In a 32-wide wave a compare result is one SGPR; here it is a pair (bit L of the pair is lane L), so a translation that
; keeps only the low word, swaps the words, or compares 64-bit values by their low word shows in one half. The
; per-half patterns below are chosen so each of those gives a different mask. v1 = lane L (0..63) from v_mbcnt_lo/hi,
; one 64-thread group. The output heap is 430 dwords (VKC_OUT_WORDS=430).
;
; RDNA2 ISA pseudocode used (D is the mask: bit L = lane L; every compare reads lane L's S0 and S1):
;   v_cmp_lt_i32 / v_cmp_ge_i32 / v_cmp_gt_i64     D[L] = S0 < S1 / S0 >= S1 / S0 > S1, signed
;   v_cmp_gt_u32 / v_cmp_le_u32 / v_cmp_lt_u64     D[L] = S0 > S1 / S0 <= S1 / S0 < S1, unsigned (a 64-bit compare uses both words)
;   v_cmp_eq_u32 / v_cmp_ne_u32                    D[L] = S0 == S1 / S0 != S1
;   v_cmp_lt_f32 (ordered: false if either is NaN), v_cmp_nlt_f32 = !(S0 < S1) (true for NaN), v_cmp_u_f32 (either is NaN),
;   v_cmp_o_f32 (neither), v_cmp_neq_f32 = !(S0 == S1) (true for NaN), v_cmp_lg_f32 = S0 < S1 || S0 > S1 (ordered)
;   v_cmp_class_f32  D[L] = bit class(S0) of S1, class bits: 0 signaling NaN, 1 quiet NaN, 2 -inf, 3 -normal, 4 -denormal,
;                    5 -0, 6 +0, 7 +denormal, 8 +normal, 9 +inf
;   s_and_b64 / s_or_b64 / s_xor_b64   D = S0 & S1 / S0 | S1 / S0 ^ S1       s_andn2_b64 D = S0 & ~S1   s_orn2_b64 D = S0 | ~S1
;   s_nor_b64 D = ~(S0 | S1)   s_xnor_b64 D = ~(S0 ^ S1)
; Per-lane inputs: V2 = L - 32 (so 32-bit signed: negative below lane 32; unsigned: huge below it, 0 at lane 32);
; H = L * 0x9e3779b1 mod 2^32 (pseudo-random in both halves); L & 31; A64 = (L >> 5) << 32 | ~L, B64 = 0x1_80000000;
; I64 = -(L >> 5) << 32 | L (negative in the upper half), 0x10; F = float(L) except lanes 5 and 37 are a quiet NaN, lanes 1
; and 33 are -infinity and lane 63 is a signaling NaN (0x7f800001); 32.0, 40.0 and the class mask S1 = 0xc (-inf, -normal)
; in the low half and 0x300 (+normal, +inf) in the upper half.
; Masks (Mk, k = 0..22) with the compare, the lanes where it is true, and the mask:
;   0  v_cmp_lt_i32 s[12:13]   V2 < 0: L < 32                                              0x00000000_ffffffff
;   1  v_cmp_gt_u32 s[14:15]   V2 > 0: all lanes but 32 (V2 = 0)                           0xfffffffe_ffffffff
;   2  v_cmp_lt_i32 s[16:17]   H < 0 (bit 31 of H)                                          0xa5ad2d29_696b4b4a
;   3  v_cmp_le_u32 s[18:19]   H <= 0x55555555                                              0x52129094_9484a4a5
;   4  v_cmp_eq_u32 s[20:21]   L == 37                                                      0x00000020_00000000
;   5  v_cmp_ne_u32_e32 vcc    L & 31 != 0: all lanes but 0 and 32                          0xfffffffe_fffffffe
;   6  v_cmp_ge_i32 vcc        V2 >= 5: L >= 37                                             0xffffffe0_00000000
;   7  v_cmp_lt_u64 s[22:23]   A64 < B64: the low half (hi 0 < 1); in the upper half the high words
;                              are equal and ~L > 0x80000000, so no lane (the low words alone: none) 0x00000000_ffffffff
;   8  v_cmp_gt_i64 s[24:25]   I64 > 16: low half L > 16 (lanes 17..31); the upper half is negative  0x00000000_fffe0000
;                              (the low words alone: lanes 17..63)
;   9  v_cmp_lt_f32 s[26:27]   F < 32.0: lanes 0..31 but not 5 (NaN), plus lane 33 (-inf)      0x00000002_ffffffdf
;   10 v_cmp_nlt_f32 s[28:29]  !(F < 32.0): the complement of mask 9 (NaN lanes are in it)      0xfffffffd_00000020
;   11 v_cmp_u_f32 s[30:31]    NaN: lanes 5, 37, 63                                            0x80000020_00000020
;   12 v_cmp_o_f32 s[32:33]    not NaN: the complement of mask 11                              0x7fffffdf_ffffffdf
;   13 v_cmp_neq_f32 s[34:35]  !(F == 40.0): every lane but 40                                  0xfffffeff_ffffffff
;   14 v_cmp_lg_f32 s[36:37]   F < 40.0 or F > 40.0: not 40 and not a NaN lane                  0x7ffffedf_ffffffdf
;   15 v_cmp_class_f32 s[38:39] class(F) & S1: lane 1 (-inf, low half S1 0xc); in the upper half the
;                              +normal lanes 32..62 but not 33 (-inf) and 37 (NaN)               0x7fffffdd_00000002
;   16 s_and_b64 s[40:41]      M1 & M2                                                         0xa5ad2d28_696b4b4a
;   17 s_or_b64 s[42:43]       M0 | M4                                                         0x00000020_ffffffff
;   18 s_xor_b64 s[44:45]      M2 ^ M3                                                         0xf7bfbdbd_fdefefef
;   19 s_andn2_b64 s[46:47]    M1 & ~M3                                                        0xaded6f6a_6b7b5b5a
;   20 s_orn2_b64 s[48:49]     M0 | ~M3                                                        0xaded6f6b_ffffffff
;   21 s_nor_b64 s[50:51]      ~(M7 | M9)                                                      0xfffffffd_00000000
;   22 s_xnor_b64 s[52:53]     ~(M8 ^ M15)                                                     0x80000022_0001fffd
; Output (dwords): out[L] = flag word of lane L, bit k is lane L's bit of Mk (the mask read back per lane with
; v_cndmask_b32_e64 / v_cndmask_b32_e32, so this does not depend on how an SGPR pair reads as data);
; out[64 + 64 j + L], j = 0..4 = v_cndmask(L, L | 0x100, Mk) for k = 0, 7, 15, 20 and for the VCC mask 5 (the VOP2 form):
; L | 0x100 where the mask has lane L, else L; out[384 + k] = s_bcnt1_i32_b64 of Mk, out[407 + k] = s_ff1_i32_b64 of Mk
; (lowest set lane); lane 0 stores them.
;   bcnt1 (k = 0..22): 32 63 32 22 1 62 27 32 15 32 32 3 61 63 60 30 31 33 54 41 54 31 19
;   ff1:               0 0 1 0 37 1 37 0 17 0 5 5 0 0 0 1 1 0 0 1 0 32 0
; The full expected table is W64X_VCMP_WORDS in check.py (the formulas above in plain Python).
; group: 64,1,1
; Run: VKC_OUT_WORDS=430 vk_compute_test cs_w64x_vcmp.spv 1 dump --require-wave64 (check.py / run_gpu_tests.py do this)
; mattr: +wavefrontsize64,-wavefrontsize32
v_mbcnt_lo_u32_b32 v1, -1, 0
v_mbcnt_hi_u32_b32 v1, -1, v1
v_lshlrev_b32_e32 v20, 2, v1
v_mov_b32_e32 v60, 0
v_add_nc_u32_e32 v2, -32, v1
v_mul_lo_u32 v3, 0x9e3779b1, v1
v_and_b32_e32 v4, 31, v1
v_not_b32_e32 v6, v1
v_lshrrev_b32_e32 v7, 5, v1
v_mov_b32_e32 v8, 0x80000000
v_mov_b32_e32 v9, 1
v_mov_b32_e32 v10, v1
v_sub_nc_u32_e32 v11, 0, v7
v_mov_b32_e32 v12, 0x10
v_mov_b32_e32 v13, 0
v_cvt_f32_u32_e32 v14, v1
v_mov_b32_e32 v15, 0x7fc00000
v_mov_b32_e32 v16, 0xff800000
v_mov_b32_e32 v17, 0x7f800001
v_mov_b32_e32 v18, 0x42000000
v_mov_b32_e32 v19, 0x42200000
s_mov_b32 s6, 0x20
s_mov_b32 s7, 0x20
s_mov_b32 s8, 2
s_mov_b32 s9, 2
s_mov_b32 s10, 0
s_mov_b32 s11, 0x80000000
v_cndmask_b32_e64 v14, v14, v15, s[6:7]
v_cndmask_b32_e64 v14, v14, v16, s[8:9]
v_cndmask_b32_e64 v14, v14, v17, s[10:11]
v_mad_u32_u24 v21, v7, 0x2f4, 12
v_or_b32_e32 v72, 0x100, v1
v_cmp_lt_i32_e64 s[12:13], v2, 0
v_cndmask_b32_e64 v62, 0, 1, s[12:13]
v_or_b32_e32 v60, v60, v62
s_bcnt1_i32_b64 s58, s[12:13]
s_ff1_i32_b64 s81, s[12:13]
v_cndmask_b32_e64 v70, v1, v72, s[12:13]
buffer_store_dword v70, v20, s[0:3], 0 offen offset:256
v_cmp_gt_u32_e64 s[14:15], v2, 0
v_cndmask_b32_e64 v62, 0, 2, s[14:15]
v_or_b32_e32 v60, v60, v62
s_bcnt1_i32_b64 s59, s[14:15]
s_ff1_i32_b64 s82, s[14:15]
v_cmp_lt_i32_e64 s[16:17], v3, 0
v_cndmask_b32_e64 v62, 0, 4, s[16:17]
v_or_b32_e32 v60, v60, v62
s_bcnt1_i32_b64 s60, s[16:17]
s_ff1_i32_b64 s83, s[16:17]
v_cmp_le_u32_e64 s[18:19], v3, 0x55555555
v_cndmask_b32_e64 v62, 0, 8, s[18:19]
v_or_b32_e32 v60, v60, v62
s_bcnt1_i32_b64 s61, s[18:19]
s_ff1_i32_b64 s84, s[18:19]
v_cmp_eq_u32_e64 s[20:21], v1, 37
v_cndmask_b32_e64 v62, 0, 16, s[20:21]
v_or_b32_e32 v60, v60, v62
s_bcnt1_i32_b64 s62, s[20:21]
s_ff1_i32_b64 s85, s[20:21]
v_cmp_ne_u32_e32 vcc, 0, v4
v_cndmask_b32_e64 v62, 0, 32, vcc
v_or_b32_e32 v60, v60, v62
s_bcnt1_i32_b64 s63, vcc
s_ff1_i32_b64 s86, vcc
v_cndmask_b32_e32 v74, v1, v72, vcc
buffer_store_dword v74, v20, s[0:3], 0 offen offset:1280
v_cmp_ge_i32_e64 vcc, v2, 5
v_cndmask_b32_e64 v62, 0, 64, vcc
v_or_b32_e32 v60, v60, v62
s_bcnt1_i32_b64 s64, vcc
s_ff1_i32_b64 s87, vcc
v_cmp_lt_u64_e64 s[22:23], v[6:7], v[8:9]
v_cndmask_b32_e64 v62, 0, 128, s[22:23]
v_or_b32_e32 v60, v60, v62
s_bcnt1_i32_b64 s65, s[22:23]
s_ff1_i32_b64 s88, s[22:23]
v_cndmask_b32_e64 v70, v1, v72, s[22:23]
buffer_store_dword v70, v20, s[0:3], 0 offen offset:512
v_cmp_gt_i64_e64 s[24:25], v[10:11], v[12:13]
v_cndmask_b32_e64 v62, 0, 256, s[24:25]
v_or_b32_e32 v60, v60, v62
s_bcnt1_i32_b64 s66, s[24:25]
s_ff1_i32_b64 s89, s[24:25]
v_cmp_lt_f32_e64 s[26:27], v14, v18
v_cndmask_b32_e64 v62, 0, 512, s[26:27]
v_or_b32_e32 v60, v60, v62
s_bcnt1_i32_b64 s67, s[26:27]
s_ff1_i32_b64 s90, s[26:27]
v_cmp_nlt_f32_e64 s[28:29], v14, v18
v_cndmask_b32_e64 v62, 0, 1024, s[28:29]
v_or_b32_e32 v60, v60, v62
s_bcnt1_i32_b64 s68, s[28:29]
s_ff1_i32_b64 s91, s[28:29]
v_cmp_u_f32_e64 s[30:31], v14, v14
v_cndmask_b32_e64 v62, 0, 2048, s[30:31]
v_or_b32_e32 v60, v60, v62
s_bcnt1_i32_b64 s69, s[30:31]
s_ff1_i32_b64 s92, s[30:31]
v_cmp_o_f32_e64 s[32:33], v14, v14
v_cndmask_b32_e64 v62, 0, 4096, s[32:33]
v_or_b32_e32 v60, v60, v62
s_bcnt1_i32_b64 s70, s[32:33]
s_ff1_i32_b64 s93, s[32:33]
v_cmp_neq_f32_e64 s[34:35], v14, v19
v_cndmask_b32_e64 v62, 0, 8192, s[34:35]
v_or_b32_e32 v60, v60, v62
s_bcnt1_i32_b64 s71, s[34:35]
s_ff1_i32_b64 s94, s[34:35]
v_cmp_lg_f32_e64 s[36:37], v14, v19
v_cndmask_b32_e64 v62, 0, 16384, s[36:37]
v_or_b32_e32 v60, v60, v62
s_bcnt1_i32_b64 s72, s[36:37]
s_ff1_i32_b64 s95, s[36:37]
v_cmp_class_f32_e64 s[38:39], v14, v21
v_cndmask_b32_e64 v62, 0, 32768, s[38:39]
v_or_b32_e32 v60, v60, v62
s_bcnt1_i32_b64 s73, s[38:39]
s_ff1_i32_b64 s96, s[38:39]
v_cndmask_b32_e64 v70, v1, v72, s[38:39]
buffer_store_dword v70, v20, s[0:3], 0 offen offset:768
s_and_b64 s[40:41], s[14:15], s[16:17]
v_cndmask_b32_e64 v62, 0, 65536, s[40:41]
v_or_b32_e32 v60, v60, v62
s_bcnt1_i32_b64 s74, s[40:41]
s_ff1_i32_b64 s97, s[40:41]
s_or_b64 s[42:43], s[12:13], s[20:21]
v_cndmask_b32_e64 v62, 0, 131072, s[42:43]
v_or_b32_e32 v60, v60, v62
s_bcnt1_i32_b64 s75, s[42:43]
s_ff1_i32_b64 s98, s[42:43]
s_xor_b64 s[44:45], s[16:17], s[18:19]
v_cndmask_b32_e64 v62, 0, 262144, s[44:45]
v_or_b32_e32 v60, v60, v62
s_bcnt1_i32_b64 s76, s[44:45]
s_ff1_i32_b64 s99, s[44:45]
s_andn2_b64 s[46:47], s[14:15], s[18:19]
v_cndmask_b32_e64 v62, 0, 524288, s[46:47]
v_or_b32_e32 v60, v60, v62
s_bcnt1_i32_b64 s77, s[46:47]
s_ff1_i32_b64 s100, s[46:47]
s_orn2_b64 s[48:49], s[12:13], s[18:19]
v_cndmask_b32_e64 v62, 0, 1048576, s[48:49]
v_or_b32_e32 v60, v60, v62
s_bcnt1_i32_b64 s78, s[48:49]
s_ff1_i32_b64 s101, s[48:49]
v_cndmask_b32_e64 v70, v1, v72, s[48:49]
buffer_store_dword v70, v20, s[0:3], 0 offen offset:1024
s_nor_b64 s[50:51], s[22:23], s[26:27]
v_cndmask_b32_e64 v62, 0, 2097152, s[50:51]
v_or_b32_e32 v60, v60, v62
s_bcnt1_i32_b64 s79, s[50:51]
s_ff1_i32_b64 s102, s[50:51]
s_xnor_b64 s[52:53], s[24:25], s[38:39]
v_cndmask_b32_e64 v62, 0, 4194304, s[52:53]
v_or_b32_e32 v60, v60, v62
s_bcnt1_i32_b64 s80, s[52:53]
s_ff1_i32_b64 s103, s[52:53]
buffer_store_dword v60, v20, s[0:3], 0 offen offset:0
v_cmpx_eq_u32_e32 0, v1
v_mov_b32_e32 v50, s58
buffer_store_dword v50, off, s[0:3], 0 offset:1536
v_mov_b32_e32 v50, s59
buffer_store_dword v50, off, s[0:3], 0 offset:1540
v_mov_b32_e32 v50, s60
buffer_store_dword v50, off, s[0:3], 0 offset:1544
v_mov_b32_e32 v50, s61
buffer_store_dword v50, off, s[0:3], 0 offset:1548
v_mov_b32_e32 v50, s62
buffer_store_dword v50, off, s[0:3], 0 offset:1552
v_mov_b32_e32 v50, s63
buffer_store_dword v50, off, s[0:3], 0 offset:1556
v_mov_b32_e32 v50, s64
buffer_store_dword v50, off, s[0:3], 0 offset:1560
v_mov_b32_e32 v50, s65
buffer_store_dword v50, off, s[0:3], 0 offset:1564
v_mov_b32_e32 v50, s66
buffer_store_dword v50, off, s[0:3], 0 offset:1568
v_mov_b32_e32 v50, s67
buffer_store_dword v50, off, s[0:3], 0 offset:1572
v_mov_b32_e32 v50, s68
buffer_store_dword v50, off, s[0:3], 0 offset:1576
v_mov_b32_e32 v50, s69
buffer_store_dword v50, off, s[0:3], 0 offset:1580
v_mov_b32_e32 v50, s70
buffer_store_dword v50, off, s[0:3], 0 offset:1584
v_mov_b32_e32 v50, s71
buffer_store_dword v50, off, s[0:3], 0 offset:1588
v_mov_b32_e32 v50, s72
buffer_store_dword v50, off, s[0:3], 0 offset:1592
v_mov_b32_e32 v50, s73
buffer_store_dword v50, off, s[0:3], 0 offset:1596
v_mov_b32_e32 v50, s74
buffer_store_dword v50, off, s[0:3], 0 offset:1600
v_mov_b32_e32 v50, s75
buffer_store_dword v50, off, s[0:3], 0 offset:1604
v_mov_b32_e32 v50, s76
buffer_store_dword v50, off, s[0:3], 0 offset:1608
v_mov_b32_e32 v50, s77
buffer_store_dword v50, off, s[0:3], 0 offset:1612
v_mov_b32_e32 v50, s78
buffer_store_dword v50, off, s[0:3], 0 offset:1616
v_mov_b32_e32 v50, s79
buffer_store_dword v50, off, s[0:3], 0 offset:1620
v_mov_b32_e32 v50, s80
buffer_store_dword v50, off, s[0:3], 0 offset:1624
v_mov_b32_e32 v50, s81
buffer_store_dword v50, off, s[0:3], 0 offset:1628
v_mov_b32_e32 v50, s82
buffer_store_dword v50, off, s[0:3], 0 offset:1632
v_mov_b32_e32 v50, s83
buffer_store_dword v50, off, s[0:3], 0 offset:1636
v_mov_b32_e32 v50, s84
buffer_store_dword v50, off, s[0:3], 0 offset:1640
v_mov_b32_e32 v50, s85
buffer_store_dword v50, off, s[0:3], 0 offset:1644
v_mov_b32_e32 v50, s86
buffer_store_dword v50, off, s[0:3], 0 offset:1648
v_mov_b32_e32 v50, s87
buffer_store_dword v50, off, s[0:3], 0 offset:1652
v_mov_b32_e32 v50, s88
buffer_store_dword v50, off, s[0:3], 0 offset:1656
v_mov_b32_e32 v50, s89
buffer_store_dword v50, off, s[0:3], 0 offset:1660
v_mov_b32_e32 v50, s90
buffer_store_dword v50, off, s[0:3], 0 offset:1664
v_mov_b32_e32 v50, s91
buffer_store_dword v50, off, s[0:3], 0 offset:1668
v_mov_b32_e32 v50, s92
buffer_store_dword v50, off, s[0:3], 0 offset:1672
v_mov_b32_e32 v50, s93
buffer_store_dword v50, off, s[0:3], 0 offset:1676
v_mov_b32_e32 v50, s94
buffer_store_dword v50, off, s[0:3], 0 offset:1680
v_mov_b32_e32 v50, s95
buffer_store_dword v50, off, s[0:3], 0 offset:1684
v_mov_b32_e32 v50, s96
buffer_store_dword v50, off, s[0:3], 0 offset:1688
v_mov_b32_e32 v50, s97
buffer_store_dword v50, off, s[0:3], 0 offset:1692
v_mov_b32_e32 v50, s98
buffer_store_dword v50, off, s[0:3], 0 offset:1696
v_mov_b32_e32 v50, s99
buffer_store_dword v50, off, s[0:3], 0 offset:1700
v_mov_b32_e32 v50, s100
buffer_store_dword v50, off, s[0:3], 0 offset:1704
v_mov_b32_e32 v50, s101
buffer_store_dword v50, off, s[0:3], 0 offset:1708
v_mov_b32_e32 v50, s102
buffer_store_dword v50, off, s[0:3], 0 offset:1712
v_mov_b32_e32 v50, s103
buffer_store_dword v50, off, s[0:3], 0 offset:1716
s_endpgm
