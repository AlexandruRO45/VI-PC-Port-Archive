; Wave64 carry chains: v_add_co_u32, v_add_co_ci_u32, v_sub_co_u32, v_subrev_co_u32, v_sub_co_ci_u32 and
; v_subrev_co_ci_u32, with the carry-out in an SGPR pair or in VCC and the carry-in from an SGPR pair or from VCC,
; over all 64 lanes. The wave32 counterpart is cs_w32_dot4_carry.asm. In a 32-wide wave a carry mask is one SGPR; here
; it is a pair (bit L of the pair is lane L), so a translation that keeps only the low word, swaps the two words or
; reads the carry-in of the upper half from the low word shows in one half. v1 = lane L (0..63) from v_mbcnt_lo/hi,
; one 64-thread group. The output heap is 922 dwords (VKC_OUT_WORDS=922).
;
; RDNA2 ISA pseudocode used (VOP3 with the mask pair as sdst; the VOP2 forms name vcc):
;   v_add_co_u32       tmp = S0 + S1;               D = tmp[31:0]; carry-out lane mask = tmp >= 2^32
;   v_add_co_ci_u32    tmp = S0 + S1 + CIN[lane];   D = tmp[31:0]; carry-out lane mask = tmp >= 2^32
;   v_sub_co_u32       D = S0 - S1;                 borrow lane mask = S1 > S0
;   v_subrev_co_u32    D = S1 - S0;                 borrow lane mask = S0 > S1
;   v_sub_co_ci_u32    D = S0 - S1 - CIN[lane];     borrow lane mask = S1 + CIN[lane] > S0
;   v_subrev_co_ci_u32 D = S1 - S0 - CIN[lane];     borrow lane mask = S0 + CIN[lane] > S1
;   CIN[lane] is bit `lane` of the 64-bit carry-in mask. gfx10 has no VOP2 form of v_add/sub/subrev_co_u32; the plain
;   ones are VOP3 and name s[..:..] or vcc as the carry-out.
; The carry-in mask is s[10:11] = 0x0ff0a55b_5a5a5a5a. Low word 0x5a5a5a5a: lanes 1 3 4 6 9 11 12 14 ... have
; carry-in 1, lane 22 too, lane 31 not. High word 0x0ff0a55b: lanes 32, 33, 35, 40 and 47 have carry-in 1, lane 63 not.
; B0 = ((L >> 5) ^ L) & 1 is the lane's low bit in the low half and its complement in the upper half.
; The 13 ops, in order k = 0..12 (the mask each one writes is Mk), with their carry/borrow lanes:
;   0  v_add_co_u32        s[12:13] = 0xffffffe0 + L; carry for L >= 32                    M0  0xffffffff_00000000
;   1  v_add_co_ci_u32     s[14:15] = 0xffffffe9 + L + CIN; carry when L + CIN >= 23:
;                          L >= 23, and L = 22 (its CIN is 1)                              M1  0xffffffff_ffc00000
;   2  v_add_co_u32        s[16:17] = -1 + B0; carry where B0 = 1: odd lanes of the low half, even lanes
;                          of the upper half                                               M2  0x55555555_aaaaaaaa
;   3  v_add_co_ci_u32     s[18:19] = -1 + 0 + M2 (the carry ripples one stage on)        M3  0x55555555_aaaaaaaa
;   4  v_add_co_ci_u32     s[20:21] = 0x7fffffff + 0 + M3; never carries                   M4  0
;   5  v_sub_co_u32        s[22:23] = 48 - L; borrow for L > 48                            M5  0xfffe0000_00000000
;   6  v_subrev_co_u32     s[24:25] = L - 16; borrow for L < 16                            M6  0x00000000_0000ffff
;   7  v_sub_co_ci_u32     s[26:27] = L - 32 - CIN; borrow when L < 32 + CIN:
;                          L <= 31, and L = 32 (CIN 1)                                     M7  0x00000001_ffffffff
;   8  v_subrev_co_ci_u32  s[28:29] = L - 47 - CIN; borrow when L < 47 + CIN:
;                          L <= 46, and L = 47 (CIN 1)                                     M8  0x0000ffff_ffffffff
;   9  v_sub_co_ci_u32_e32 vcc = L - 40 - CIN, VCC loaded with s_mov_b64 vcc, s[10:11]; borrow for
;                          L <= 39, and L = 40 (CIN 1)                                     M9  0x000001ff_ffffffff
;   10 v_add_co_u32        vcc = 0xffffffeb + L; carry for L >= 21                         M10 0xffffffff_ffe00000
;   11 v_add_co_ci_u32_e32 vcc = -1 + 0 + vcc: carries exactly where op 10 did             M11 0xffffffff_ffe00000
;   12 v_add_co_ci_u32_e32 vcc = 0x7fffffff + 0x7fffffff + vcc; never carries              M12 0
; Output (dwords): out[64k + L] = D of op k in lane L (words 0..831); out[832 + L] = flag word, bit k is lane L's bit
; of Mk (the mask read back per lane with v_cndmask_b32_e64, so this does not depend on how an SGPR pair reads as
; data); out[896 + k] = s_bcnt1_i32_b64 of Mk, out[909 + k] = s_ff1_i32_b64 of Mk (lowest set lane, -1 when none); lane 0
; stores those 26 words.
;   bcnt1, k = 0..12: 32 42 32 32 0 15 16 33 48 41 43 43 0      ff1: 32 22 1 1 -1 49 0 0 0 0 21 21 -1
;   A mask cut to its low word loses M0 (bcnt1 0) and shifts the others; M2 with its two words swapped has ff1 0.
; Words by hand, D = tmp mod 2^32:
;   op 0:  L = 0 0xffffffe0, L = 31 0xffffffff, L = 32 0 (carry out), L = 63 0x1f
;   op 1:  L = 0 0xffffffe9, L = 1 0xffffffeb (CIN 1), L = 22 0 (0xffffffe9 + 22 + 1 = 2^32), L = 63 0x28
;   op 2:  L = 0 0xffffffff, L = 1 0 (B0 1), L = 32 0 (B0 = (1 ^ 32) & 1 = 1), L = 33 0xffffffff (B0 0)
;   op 4:  0x7fffffff + carry: L = 1 0x80000000, L = 0 0x7fffffff, L = 32 0x80000000
;   op 5:  L = 0 0x30, L = 48 0, L = 49 0xffffffff, L = 63 0xfffffff1
;   op 6:  L = 0 0xfffffff0, L = 15 0xffffffff, L = 16 0, L = 63 0x2f
;   op 7:  L = 31 0xffffffff, L = 32 0xffffffff (32 - 32 - 1), L = 33 0 (33 - 32 - 1), L = 63 0x1f
;   op 8:  L = 46 0xffffffff, L = 47 0xffffffff (47 - 47 - 1), L = 48 1 (CIN of lane 48 is 0), L = 63 0x10
;   op 12: 0xfffffffe + carry: L = 0 0xfffffffe, L = 22 0xffffffff (carry of op 11 at lane 22)
; Flag words (bit k = Mk): lane 0 0x3c0 (M6 M7 M8 M9), lane 22 0xf82 (M1 and M7..M11),
;   lane 32 0xf8f (M0 M1 M2 M3 and M7..M11), lane 33 0xf03 (M0 M1 and M8..M11), lane 49 0xc23 (M0 M1 M5 M10 M11).
; The full expected table is W64X_CARRY_WORDS in check.py (the formulas above in plain Python).
; group: 64,1,1
; Run: VKC_OUT_WORDS=922 vk_compute_test cs_w64x_carry.spv 1 dump --require-wave64 (check.py / run_gpu_tests.py do this)
; mattr: +wavefrontsize64,-wavefrontsize32
v_mbcnt_lo_u32_b32 v1, -1, 0
v_mbcnt_hi_u32_b32 v1, -1, v1
v_lshlrev_b32_e32 v20, 2, v1
v_mov_b32_e32 v60, 0
s_mov_b32 s10, 0x5a5a5a5a
s_mov_b32 s11, 0x0ff0a55b
v_mov_b32_e32 v2, 0xffffffe0
v_mov_b32_e32 v3, 0xffffffe9
v_lshrrev_b32_e32 v4, 5, v1
v_xor_b32_e32 v4, v4, v1
v_and_b32_e32 v4, 1, v4
v_mov_b32_e32 v5, 0
v_mov_b32_e32 v6, 0x7fffffff
v_mov_b32_e32 v7, 0xffffffeb
v_mov_b32_e32 v8, -1
v_mov_b32_e32 v11, 40
v_add_co_u32_e64 v30, s[12:13], v2, v1
v_cndmask_b32_e64 v62, 0, 1, s[12:13]
v_or_b32_e32 v60, v60, v62
s_bcnt1_i32_b64 s40, s[12:13]
s_ff1_i32_b64 s60, s[12:13]
buffer_store_dword v30, v20, s[0:3], 0 offen offset:0
v_add_co_ci_u32_e64 v31, s[14:15], v3, v1, s[10:11]
v_cndmask_b32_e64 v62, 0, 2, s[14:15]
v_or_b32_e32 v60, v60, v62
s_bcnt1_i32_b64 s41, s[14:15]
s_ff1_i32_b64 s61, s[14:15]
buffer_store_dword v31, v20, s[0:3], 0 offen offset:256
v_add_co_u32_e64 v32, s[16:17], -1, v4
v_cndmask_b32_e64 v62, 0, 4, s[16:17]
v_or_b32_e32 v60, v60, v62
s_bcnt1_i32_b64 s42, s[16:17]
s_ff1_i32_b64 s62, s[16:17]
buffer_store_dword v32, v20, s[0:3], 0 offen offset:512
v_add_co_ci_u32_e64 v33, s[18:19], -1, v5, s[16:17]
v_cndmask_b32_e64 v62, 0, 8, s[18:19]
v_or_b32_e32 v60, v60, v62
s_bcnt1_i32_b64 s43, s[18:19]
s_ff1_i32_b64 s63, s[18:19]
buffer_store_dword v33, v20, s[0:3], 0 offen offset:768
v_add_co_ci_u32_e64 v34, s[20:21], v6, v5, s[18:19]
v_cndmask_b32_e64 v62, 0, 16, s[20:21]
v_or_b32_e32 v60, v60, v62
s_bcnt1_i32_b64 s44, s[20:21]
s_ff1_i32_b64 s64, s[20:21]
buffer_store_dword v34, v20, s[0:3], 0 offen offset:1024
v_sub_co_u32_e64 v35, s[22:23], 48, v1
v_cndmask_b32_e64 v62, 0, 32, s[22:23]
v_or_b32_e32 v60, v60, v62
s_bcnt1_i32_b64 s45, s[22:23]
s_ff1_i32_b64 s65, s[22:23]
buffer_store_dword v35, v20, s[0:3], 0 offen offset:1280
v_subrev_co_u32_e64 v36, s[24:25], 16, v1
v_cndmask_b32_e64 v62, 0, 64, s[24:25]
v_or_b32_e32 v60, v60, v62
s_bcnt1_i32_b64 s46, s[24:25]
s_ff1_i32_b64 s66, s[24:25]
buffer_store_dword v36, v20, s[0:3], 0 offen offset:1536
v_sub_co_ci_u32_e64 v37, s[26:27], v1, 32, s[10:11]
v_cndmask_b32_e64 v62, 0, 128, s[26:27]
v_or_b32_e32 v60, v60, v62
s_bcnt1_i32_b64 s47, s[26:27]
s_ff1_i32_b64 s67, s[26:27]
buffer_store_dword v37, v20, s[0:3], 0 offen offset:1792
v_subrev_co_ci_u32_e64 v38, s[28:29], 47, v1, s[10:11]
v_cndmask_b32_e64 v62, 0, 256, s[28:29]
v_or_b32_e32 v60, v60, v62
s_bcnt1_i32_b64 s48, s[28:29]
s_ff1_i32_b64 s68, s[28:29]
buffer_store_dword v38, v20, s[0:3], 0 offen offset:2048
s_mov_b64 vcc, s[10:11]
v_sub_co_ci_u32_e32 v39, vcc, v1, v11, vcc
v_cndmask_b32_e64 v62, 0, 512, vcc
v_or_b32_e32 v60, v60, v62
s_bcnt1_i32_b64 s49, vcc
s_ff1_i32_b64 s69, vcc
buffer_store_dword v39, v20, s[0:3], 0 offen offset:2304
v_add_co_u32_e64 v40, vcc, v7, v1
v_cndmask_b32_e64 v62, 0, 1024, vcc
v_or_b32_e32 v60, v60, v62
s_bcnt1_i32_b64 s50, vcc
s_ff1_i32_b64 s70, vcc
buffer_store_dword v40, v20, s[0:3], 0 offen offset:2560
v_add_co_ci_u32_e32 v41, vcc, v8, v5, vcc
v_cndmask_b32_e64 v62, 0, 2048, vcc
v_or_b32_e32 v60, v60, v62
s_bcnt1_i32_b64 s51, vcc
s_ff1_i32_b64 s71, vcc
buffer_store_dword v41, v20, s[0:3], 0 offen offset:2816
v_add_co_ci_u32_e32 v42, vcc, v6, v6, vcc
v_cndmask_b32_e64 v62, 0, 4096, vcc
v_or_b32_e32 v60, v60, v62
s_bcnt1_i32_b64 s52, vcc
s_ff1_i32_b64 s72, vcc
buffer_store_dword v42, v20, s[0:3], 0 offen offset:3072
buffer_store_dword v60, v20, s[0:3], 0 offen offset:3328
v_cmpx_eq_u32_e32 0, v1
v_mov_b32_e32 v50, s40
buffer_store_dword v50, off, s[0:3], 0 offset:3584
v_mov_b32_e32 v50, s41
buffer_store_dword v50, off, s[0:3], 0 offset:3588
v_mov_b32_e32 v50, s42
buffer_store_dword v50, off, s[0:3], 0 offset:3592
v_mov_b32_e32 v50, s43
buffer_store_dword v50, off, s[0:3], 0 offset:3596
v_mov_b32_e32 v50, s44
buffer_store_dword v50, off, s[0:3], 0 offset:3600
v_mov_b32_e32 v50, s45
buffer_store_dword v50, off, s[0:3], 0 offset:3604
v_mov_b32_e32 v50, s46
buffer_store_dword v50, off, s[0:3], 0 offset:3608
v_mov_b32_e32 v50, s47
buffer_store_dword v50, off, s[0:3], 0 offset:3612
v_mov_b32_e32 v50, s48
buffer_store_dword v50, off, s[0:3], 0 offset:3616
v_mov_b32_e32 v50, s49
buffer_store_dword v50, off, s[0:3], 0 offset:3620
v_mov_b32_e32 v50, s50
buffer_store_dword v50, off, s[0:3], 0 offset:3624
v_mov_b32_e32 v50, s51
buffer_store_dword v50, off, s[0:3], 0 offset:3628
v_mov_b32_e32 v50, s52
buffer_store_dword v50, off, s[0:3], 0 offset:3632
v_mov_b32_e32 v50, s60
buffer_store_dword v50, off, s[0:3], 0 offset:3636
v_mov_b32_e32 v50, s61
buffer_store_dword v50, off, s[0:3], 0 offset:3640
v_mov_b32_e32 v50, s62
buffer_store_dword v50, off, s[0:3], 0 offset:3644
v_mov_b32_e32 v50, s63
buffer_store_dword v50, off, s[0:3], 0 offset:3648
v_mov_b32_e32 v50, s64
buffer_store_dword v50, off, s[0:3], 0 offset:3652
v_mov_b32_e32 v50, s65
buffer_store_dword v50, off, s[0:3], 0 offset:3656
v_mov_b32_e32 v50, s66
buffer_store_dword v50, off, s[0:3], 0 offset:3660
v_mov_b32_e32 v50, s67
buffer_store_dword v50, off, s[0:3], 0 offset:3664
v_mov_b32_e32 v50, s68
buffer_store_dword v50, off, s[0:3], 0 offset:3668
v_mov_b32_e32 v50, s69
buffer_store_dword v50, off, s[0:3], 0 offset:3672
v_mov_b32_e32 v50, s70
buffer_store_dword v50, off, s[0:3], 0 offset:3676
v_mov_b32_e32 v50, s71
buffer_store_dword v50, off, s[0:3], 0 offset:3680
v_mov_b32_e32 v50, s72
buffer_store_dword v50, off, s[0:3], 0 offset:3684
s_endpgm
