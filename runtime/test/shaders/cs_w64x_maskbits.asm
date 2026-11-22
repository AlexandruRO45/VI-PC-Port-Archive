; Wave64 scalar bit scans on 64-bit masks whose set bits are all (or mostly) in the upper 32 lanes: s_bcnt0_i32_b64,
; s_ff0_i32_b64, s_flbit_i32_b64 and s_flbit_i32_i64. In a 32-wide wave a lane mask is one SGPR; in wave64 it is a pair
; (bit L of the pair is lane L), so a translation that scans only the low word, swaps the words, or counts the zeros of
; one word shows with a mask that has nothing in the low half. s_bcnt1/s_ff1 on masks are already in cs_w64x_vcmp.asm.
; The mask is a v_cmp result (lane masks), a pair filled by s_mov_b32 (plain 64-bit data), or EXEC itself.
;
; RDNA2 ISA pseudocode used (S0 is a 64-bit scalar, bit 0 is the low bit of the low word):
;   s_bcnt0_i32_b64  D = number of 0 bits in S0 (0..64);               SCC = (D != 0)
;   s_ff0_i32_b64    D = index of the lowest 0 bit of S0, -1 if there is none;           SCC not written
;   s_flbit_i32_b64  D = number of 0 bits above the highest 1 bit (63 - its index), -1 if S0 == 0;   SCC not written
;   s_flbit_i32_i64  D = number of bits above the highest bit that differs from bit 63 (S0 sign), -1 if S0 is 0 or -1
; v1 = lane L (0..63) from v_mbcnt_lo/hi, one 64-thread group. The output heap is 64 dwords (VKC_OUT_WORDS=64).
; The translator keeps s_flbit_i32_b64/i64 of a lane mask or EXEC Unsupported (docs/design.md, #210), so those two are run on the
; data pairs only (masks 7..10 and 12); s_bcnt0 and s_ff0 are run on every mask and on EXEC.
; Masks Mk (k = 0..12): the compare or the data, the lanes that are set, the 64-bit value:
;   0  v_cmp_lt_u32 s[10:11]   31 < L: lanes 32..63                         0xffffffff_00000000
;   1  v_cmp_eq_u32 s[12:13]   L == 40                                      0x00000100_00000000
;   2  v_cmp_lt_u32 s[14:15]   61 < L: lanes 62, 63                         0xc0000000_00000000
;   3  v_cmp_gt_u32 s[16:17]   32 > L: lanes 0..31 (low half only)          0x00000000_ffffffff
;   4  v_cmp_eq_u32 s[18:19]   L == L: all lanes                            0xffffffff_ffffffff
;   5  v_cmp_lt_u32 s[20:21]   63 < L: no lane                              0x00000000_00000000
;   6  v_cmp_eq_u32 s[22:23]   L == 32                                      0x00000001_00000000
;   7  s_mov_b32 s24 / s25     data, bit 55                                 0x00800000_00000000
;   8  s_mov_b32 s26 / s27     data, the value of mask 0                    0xffffffff_00000000
;   9  s_mov_b32 s28 / s29     data, all but bit 63                         0x7fffffff_ffffffff
;   10 s_mov_b32 s30 / s31     data, only bit 63                            0x80000000_00000000
;   11 v_cmp_gt_u32 s[32:33]   40 > L: lanes 0..39                          0x000000ff_ffffffff
;   12 s_mov_b32 s34 / s35     data, bit 40 (the value of mask 1)          0x00000100_00000000
; Results, by hand (bcnt0, ff0, SCC of the bcnt0 for every mask; flbit_b64, flbit_i64 for the data masks only):
;   k=0   32  0 1      k=1   63  0 1      k=2   62  0 1      k=3   32 32 1      k=4    0 -1 0      k=5   64  0 1
;   k=6   63  0 1      k=7   63  0 1  flbit  8  8      k=8   32  0 1  flbit  0 32      k=9    1 63 1  flbit  1  1
;   k=10  63  0 1  flbit  0  1      k=11  24 40 1      k=12  63  0 1  flbit 23 23
;   (k=0: ff0 = 0 since bit 0 is 0. k=3: ff0 = 32, the low word is all ones. k=4: no zero bit gives -1 and SCC 0. k=5: no 1 bit,
;    64 zeros. k=7: bit 55 is the top 1, so 63 - 55 = 8 bits above it, and the sign is 0 so flbit_i64 is 8 too. k=8: bit 63 is 1,
;    flbit_b64 = 0; 32 ones sit above bit 31, so flbit_i64 = 32. k=9: bit 63 is the only 0: ff0 = 63, bit 62 is the top 1 so
;    flbit_b64 = 1, flbit_i64 = 1 (sign 0). k=10: flbit_i64 = 1 because bit 62 is the first bit unlike the sign 1.
;    k=11: lanes 0..39 set, so ff0 = 40 and 24 zeros. k=12: bit 40 is the top 1, 63 - 40 = 23.)
; EXEC cases (v_cmpx sets EXEC, bcnt0, ff0 and SCC read EXEC itself, then EXEC is restored): E0 = lanes 32..63 (the values of
; mask 0: 32 0 1), E1 = lanes 62, 63 (mask 2: 62 0 1), E2 = lane 32 (mask 6: 63 0 1), E3 = lanes 0..39 (mask 11: 24 40 1),
; E4 = lanes 0..31 (mask 3: 32 32 1).
; Output (dwords): out[3k + j] = bcnt0, ff0 and the SCC flag (0 or 1) of Mk, k = 0..12 (words 0..38); out[39 + 2i + j] = flbit_b64,
; flbit_i64 of the data masks 7, 8, 9, 10, 12 (i = 0..4, words 39..48); out[49 + 3e + j] = bcnt0, ff0, SCC of EXEC in the five
; EXEC cases e = 0..4 (words 49..63). The expected table is W64X_MASKBITS_WORDS in check.py (the formulas above in plain Python).
; group: 64,1,1
; Run: VKC_OUT_WORDS=64 vk_compute_test cs_w64x_maskbits.spv 1 dump --require-wave64 (check.py / run_gpu_tests.py do this)
; mattr: +wavefrontsize64,-wavefrontsize32
v_mbcnt_lo_u32_b32 v1, -1, 0
v_mbcnt_hi_u32_b32 v1, -1, v1
v_cmp_lt_u32_e64 s[10:11], 31, v1
v_cmp_eq_u32_e64 s[12:13], 40, v1
v_cmp_lt_u32_e64 s[14:15], 61, v1
v_cmp_gt_u32_e64 s[16:17], 32, v1
v_cmp_eq_u32_e64 s[18:19], v1, v1
v_cmp_lt_u32_e64 s[20:21], 63, v1
v_cmp_eq_u32_e64 s[22:23], 32, v1
s_mov_b32 s24, 0x0
s_mov_b32 s25, 0x800000
s_mov_b32 s26, 0x0
s_mov_b32 s27, 0xffffffff
s_mov_b32 s28, 0xffffffff
s_mov_b32 s29, 0x7fffffff
s_mov_b32 s30, 0x0
s_mov_b32 s31, 0x80000000
v_cmp_gt_u32_e64 s[32:33], 40, v1
s_mov_b32 s34, 0x0
s_mov_b32 s35, 0x100
s_bcnt0_i32_b64 s60, s[10:11]
s_cselect_b32 s62, 1, 0
s_ff0_i32_b64 s61, s[10:11]
v_mov_b32_e32 v50, s60
buffer_store_dword v50, off, s[0:3], 0 offset:0
v_mov_b32_e32 v50, s61
buffer_store_dword v50, off, s[0:3], 0 offset:4
v_mov_b32_e32 v50, s62
buffer_store_dword v50, off, s[0:3], 0 offset:8
s_bcnt0_i32_b64 s60, s[12:13]
s_cselect_b32 s62, 1, 0
s_ff0_i32_b64 s61, s[12:13]
v_mov_b32_e32 v50, s60
buffer_store_dword v50, off, s[0:3], 0 offset:12
v_mov_b32_e32 v50, s61
buffer_store_dword v50, off, s[0:3], 0 offset:16
v_mov_b32_e32 v50, s62
buffer_store_dword v50, off, s[0:3], 0 offset:20
s_bcnt0_i32_b64 s60, s[14:15]
s_cselect_b32 s62, 1, 0
s_ff0_i32_b64 s61, s[14:15]
v_mov_b32_e32 v50, s60
buffer_store_dword v50, off, s[0:3], 0 offset:24
v_mov_b32_e32 v50, s61
buffer_store_dword v50, off, s[0:3], 0 offset:28
v_mov_b32_e32 v50, s62
buffer_store_dword v50, off, s[0:3], 0 offset:32
s_bcnt0_i32_b64 s60, s[16:17]
s_cselect_b32 s62, 1, 0
s_ff0_i32_b64 s61, s[16:17]
v_mov_b32_e32 v50, s60
buffer_store_dword v50, off, s[0:3], 0 offset:36
v_mov_b32_e32 v50, s61
buffer_store_dword v50, off, s[0:3], 0 offset:40
v_mov_b32_e32 v50, s62
buffer_store_dword v50, off, s[0:3], 0 offset:44
s_bcnt0_i32_b64 s60, s[18:19]
s_cselect_b32 s62, 1, 0
s_ff0_i32_b64 s61, s[18:19]
v_mov_b32_e32 v50, s60
buffer_store_dword v50, off, s[0:3], 0 offset:48
v_mov_b32_e32 v50, s61
buffer_store_dword v50, off, s[0:3], 0 offset:52
v_mov_b32_e32 v50, s62
buffer_store_dword v50, off, s[0:3], 0 offset:56
s_bcnt0_i32_b64 s60, s[20:21]
s_cselect_b32 s62, 1, 0
s_ff0_i32_b64 s61, s[20:21]
v_mov_b32_e32 v50, s60
buffer_store_dword v50, off, s[0:3], 0 offset:60
v_mov_b32_e32 v50, s61
buffer_store_dword v50, off, s[0:3], 0 offset:64
v_mov_b32_e32 v50, s62
buffer_store_dword v50, off, s[0:3], 0 offset:68
s_bcnt0_i32_b64 s60, s[22:23]
s_cselect_b32 s62, 1, 0
s_ff0_i32_b64 s61, s[22:23]
v_mov_b32_e32 v50, s60
buffer_store_dword v50, off, s[0:3], 0 offset:72
v_mov_b32_e32 v50, s61
buffer_store_dword v50, off, s[0:3], 0 offset:76
v_mov_b32_e32 v50, s62
buffer_store_dword v50, off, s[0:3], 0 offset:80
s_bcnt0_i32_b64 s60, s[24:25]
s_cselect_b32 s62, 1, 0
s_ff0_i32_b64 s61, s[24:25]
v_mov_b32_e32 v50, s60
buffer_store_dword v50, off, s[0:3], 0 offset:84
v_mov_b32_e32 v50, s61
buffer_store_dword v50, off, s[0:3], 0 offset:88
v_mov_b32_e32 v50, s62
buffer_store_dword v50, off, s[0:3], 0 offset:92
s_bcnt0_i32_b64 s60, s[26:27]
s_cselect_b32 s62, 1, 0
s_ff0_i32_b64 s61, s[26:27]
v_mov_b32_e32 v50, s60
buffer_store_dword v50, off, s[0:3], 0 offset:96
v_mov_b32_e32 v50, s61
buffer_store_dword v50, off, s[0:3], 0 offset:100
v_mov_b32_e32 v50, s62
buffer_store_dword v50, off, s[0:3], 0 offset:104
s_bcnt0_i32_b64 s60, s[28:29]
s_cselect_b32 s62, 1, 0
s_ff0_i32_b64 s61, s[28:29]
v_mov_b32_e32 v50, s60
buffer_store_dword v50, off, s[0:3], 0 offset:108
v_mov_b32_e32 v50, s61
buffer_store_dword v50, off, s[0:3], 0 offset:112
v_mov_b32_e32 v50, s62
buffer_store_dword v50, off, s[0:3], 0 offset:116
s_bcnt0_i32_b64 s60, s[30:31]
s_cselect_b32 s62, 1, 0
s_ff0_i32_b64 s61, s[30:31]
v_mov_b32_e32 v50, s60
buffer_store_dword v50, off, s[0:3], 0 offset:120
v_mov_b32_e32 v50, s61
buffer_store_dword v50, off, s[0:3], 0 offset:124
v_mov_b32_e32 v50, s62
buffer_store_dword v50, off, s[0:3], 0 offset:128
s_bcnt0_i32_b64 s60, s[32:33]
s_cselect_b32 s62, 1, 0
s_ff0_i32_b64 s61, s[32:33]
v_mov_b32_e32 v50, s60
buffer_store_dword v50, off, s[0:3], 0 offset:132
v_mov_b32_e32 v50, s61
buffer_store_dword v50, off, s[0:3], 0 offset:136
v_mov_b32_e32 v50, s62
buffer_store_dword v50, off, s[0:3], 0 offset:140
s_bcnt0_i32_b64 s60, s[34:35]
s_cselect_b32 s62, 1, 0
s_ff0_i32_b64 s61, s[34:35]
v_mov_b32_e32 v50, s60
buffer_store_dword v50, off, s[0:3], 0 offset:144
v_mov_b32_e32 v50, s61
buffer_store_dword v50, off, s[0:3], 0 offset:148
v_mov_b32_e32 v50, s62
buffer_store_dword v50, off, s[0:3], 0 offset:152
s_flbit_i32_b64 s60, s[24:25]
s_flbit_i32_i64 s61, s[24:25]
v_mov_b32_e32 v50, s60
buffer_store_dword v50, off, s[0:3], 0 offset:156
v_mov_b32_e32 v50, s61
buffer_store_dword v50, off, s[0:3], 0 offset:160
s_flbit_i32_b64 s60, s[26:27]
s_flbit_i32_i64 s61, s[26:27]
v_mov_b32_e32 v50, s60
buffer_store_dword v50, off, s[0:3], 0 offset:164
v_mov_b32_e32 v50, s61
buffer_store_dword v50, off, s[0:3], 0 offset:168
s_flbit_i32_b64 s60, s[28:29]
s_flbit_i32_i64 s61, s[28:29]
v_mov_b32_e32 v50, s60
buffer_store_dword v50, off, s[0:3], 0 offset:172
v_mov_b32_e32 v50, s61
buffer_store_dword v50, off, s[0:3], 0 offset:176
s_flbit_i32_b64 s60, s[30:31]
s_flbit_i32_i64 s61, s[30:31]
v_mov_b32_e32 v50, s60
buffer_store_dword v50, off, s[0:3], 0 offset:180
v_mov_b32_e32 v50, s61
buffer_store_dword v50, off, s[0:3], 0 offset:184
s_flbit_i32_b64 s60, s[34:35]
s_flbit_i32_i64 s61, s[34:35]
v_mov_b32_e32 v50, s60
buffer_store_dword v50, off, s[0:3], 0 offset:188
v_mov_b32_e32 v50, s61
buffer_store_dword v50, off, s[0:3], 0 offset:192
s_mov_b64 s[6:7], exec
v_cmpx_lt_u32_e32 31, v1
s_bcnt0_i32_b64 s70, exec
s_cselect_b32 s72, 1, 0
s_ff0_i32_b64 s71, exec
s_mov_b64 exec, s[6:7]
v_cmpx_lt_u32_e32 61, v1
s_bcnt0_i32_b64 s73, exec
s_cselect_b32 s75, 1, 0
s_ff0_i32_b64 s74, exec
s_mov_b64 exec, s[6:7]
v_cmpx_eq_u32_e32 32, v1
s_bcnt0_i32_b64 s76, exec
s_cselect_b32 s78, 1, 0
s_ff0_i32_b64 s77, exec
s_mov_b64 exec, s[6:7]
v_cmpx_gt_u32_e32 40, v1
s_bcnt0_i32_b64 s79, exec
s_cselect_b32 s81, 1, 0
s_ff0_i32_b64 s80, exec
s_mov_b64 exec, s[6:7]
v_cmpx_gt_u32_e32 32, v1
s_bcnt0_i32_b64 s82, exec
s_cselect_b32 s84, 1, 0
s_ff0_i32_b64 s83, exec
s_mov_b64 exec, s[6:7]
v_mov_b32_e32 v50, s70
buffer_store_dword v50, off, s[0:3], 0 offset:196
v_mov_b32_e32 v50, s71
buffer_store_dword v50, off, s[0:3], 0 offset:200
v_mov_b32_e32 v50, s72
buffer_store_dword v50, off, s[0:3], 0 offset:204
v_mov_b32_e32 v50, s73
buffer_store_dword v50, off, s[0:3], 0 offset:208
v_mov_b32_e32 v50, s74
buffer_store_dword v50, off, s[0:3], 0 offset:212
v_mov_b32_e32 v50, s75
buffer_store_dword v50, off, s[0:3], 0 offset:216
v_mov_b32_e32 v50, s76
buffer_store_dword v50, off, s[0:3], 0 offset:220
v_mov_b32_e32 v50, s77
buffer_store_dword v50, off, s[0:3], 0 offset:224
v_mov_b32_e32 v50, s78
buffer_store_dword v50, off, s[0:3], 0 offset:228
v_mov_b32_e32 v50, s79
buffer_store_dword v50, off, s[0:3], 0 offset:232
v_mov_b32_e32 v50, s80
buffer_store_dword v50, off, s[0:3], 0 offset:236
v_mov_b32_e32 v50, s81
buffer_store_dword v50, off, s[0:3], 0 offset:240
v_mov_b32_e32 v50, s82
buffer_store_dword v50, off, s[0:3], 0 offset:244
v_mov_b32_e32 v50, s83
buffer_store_dword v50, off, s[0:3], 0 offset:248
v_mov_b32_e32 v50, s84
buffer_store_dword v50, off, s[0:3], 0 offset:252
s_endpgm
