; Scalar integer compares on a GPU: s_cmp_{eq,lg,gt,ge,lt,le}_{i32,u32} with two SGPRs, an inline constant (-1) and a
; literal as the second operand, and s_bitcmp0/s_bitcmp1 in the _b32 and _b64 forms. Each compare's SCC is turned into one
; bit with s_cselect_b32 (1 << k, 0) and OR-ed into an accumulator, so eight dwords carry 246 results (lane 0 stores them;
; the group is one wave32, every lane runs the same scalar code).
; RDNA2 ISA pseudocode used (SOPC; SCC is the only output):
;   s_cmp_eq_i32/u32  SCC = S0 == S1       s_cmp_lg  SCC = S0 != S1      s_cmp_gt  SCC = S0 > S1
;   s_cmp_ge  SCC = S0 >= S1               s_cmp_lt  SCC = S0 < S1       s_cmp_le  SCC = S0 <= S1
;   (the _i32 forms compare as signed 32-bit values, the _u32 forms as unsigned)
;   s_bitcmp0_b32 SCC = S0.u[S1.u[4:0]] == 0      s_bitcmp1_b32 SCC = S0.u[S1.u[4:0]] == 1
;   s_bitcmp0_b64 SCC = S0.u64[S1.u[5:0]] == 0    s_bitcmp1_b64 SCC = S0.u64[S1.u[5:0]] == 1
; Bit n of the output (word n / 32, bit n % 32) is the result of the n-th compare, in this order. In every block of
; twelve the order is eq, lg, gt, ge, lt, le of i32, then the same six of u32.
;   Block A, bits 0..95, S0 = s10, S1 = s11 (a pair per twelve bits), true conditions (signed i32 | unsigned u32):
;     A0  0x00000005 vs 0x00000005: eq,ge,le | eq,ge,le
;     A1  0x00000003 vs 0x00000007: lg,lt,le | lg,lt,le
;     A2  0x00000007 vs 0x00000003: lg,gt,ge | lg,gt,ge
;     A3  0xffffffff vs 0x00000001: lg,lt,le | lg,gt,ge
;     A4  0x00000001 vs 0xffffffff: lg,gt,ge | lg,lt,le
;     A5  0x80000000 vs 0x7fffffff: lg,lt,le | lg,gt,ge
;     A6  0x7fffffff vs 0x80000000: lg,gt,ge | lg,lt,le
;     A7  0x80000000 vs 0x80000000: eq,ge,le | eq,ge,le
;   Block B, bits 96..167, S0 = s10, S1 = the inline constant -1 (signed -1, unsigned 0xffffffff):
;     B0  0x00000000 vs -1: lg,gt,ge | lg,lt,le
;     B1  0x00000001 vs -1: lg,gt,ge | lg,lt,le
;     B2  0x7fffffff vs -1: lg,gt,ge | lg,lt,le
;     B3  0x80000000 vs -1: lg,lt,le | lg,lt,le
;     B4  0xfffffffe vs -1: lg,lt,le | lg,lt,le
;     B5  0xffffffff vs -1: eq,ge,le | eq,ge,le
;   Block C, bits 168..215, S0 = s10, S1 = the literal 0x12345678:
;     C0  0x12345677 vs 0x12345678: lg,lt,le | lg,lt,le
;     C1  0x12345678 vs 0x12345678: eq,ge,le | eq,ge,le
;     C2  0x12345679 vs 0x12345678: lg,gt,ge | lg,gt,ge
;     C3  0xf2345678 vs 0x12345678: lg,lt,le | lg,gt,ge
;   Block D, bits 216..229, S0 = s12 = 0x80000005 (bits 0, 2 and 31 set), S1 = the index [0, 1, 2, 31, 32, 33, 34] (an index of 32 and up wraps
;     to its low five bits: 32 -> bit 0, 33 -> bit 1, 34 -> bit 2); s_bitcmp0_b32 first (n = 216 + i), then s_bitcmp1_b32 (n = 223 + i):
;     bitcmp0 [0, 1, 0, 0, 0, 1, 0]   bitcmp1 [1, 0, 1, 1, 1, 0, 1]
;   Block E, bits 230..245, S0 = s[14:15] = 0x180000004 (bits 2, 31 and 32 set), S1 = the index [0, 2, 31, 32, 33, 63, 64, 66] (64 -> bit 0, 66 -> bit 2);
;     s_bitcmp0_b64 first (n = 230 + i), then s_bitcmp1_b64 (n = 238 + i):
;     bitcmp0 [1, 0, 0, 0, 1, 1, 1, 0]   bitcmp1 [0, 1, 1, 1, 0, 0, 0, 1]
; The words, from the bits above (the bit lists of the blocks, packed 32 to a word, lowest bit first):
;   out[0..7] = 0x8ecb2a69 0x2c8e3b23 0xa69c8e3b 0x8ec8ec8e 0x9cb2cb2c 0xa69cb2a6 0xa23b238e 0x239c6e
; group: 32,1,1
; Run: vk_compute_test cs_w32_scmp.spv 1 dump -> out[0..7] = 2395679337 747518755 2795277883 2395532430 2628963116 2795287206 2721784718 2333806
; mattr: +wavefrontsize32,-wavefrontsize64
s_mov_b32 s40, 0
s_mov_b32 s41, 0
s_mov_b32 s42, 0
s_mov_b32 s43, 0
s_mov_b32 s44, 0
s_mov_b32 s45, 0
s_mov_b32 s46, 0
s_mov_b32 s47, 0
s_mov_b32 s10, 0x5
s_mov_b32 s11, 0x5
s_cmp_eq_i32 s10, s11
s_cselect_b32 s30, 0x1, 0
s_or_b32 s40, s40, s30
s_cmp_lg_i32 s10, s11
s_cselect_b32 s30, 0x2, 0
s_or_b32 s40, s40, s30
s_cmp_gt_i32 s10, s11
s_cselect_b32 s30, 0x4, 0
s_or_b32 s40, s40, s30
s_cmp_ge_i32 s10, s11
s_cselect_b32 s30, 0x8, 0
s_or_b32 s40, s40, s30
s_cmp_lt_i32 s10, s11
s_cselect_b32 s30, 0x10, 0
s_or_b32 s40, s40, s30
s_cmp_le_i32 s10, s11
s_cselect_b32 s30, 0x20, 0
s_or_b32 s40, s40, s30
s_cmp_eq_u32 s10, s11
s_cselect_b32 s30, 0x40, 0
s_or_b32 s40, s40, s30
s_cmp_lg_u32 s10, s11
s_cselect_b32 s30, 0x80, 0
s_or_b32 s40, s40, s30
s_cmp_gt_u32 s10, s11
s_cselect_b32 s30, 0x100, 0
s_or_b32 s40, s40, s30
s_cmp_ge_u32 s10, s11
s_cselect_b32 s30, 0x200, 0
s_or_b32 s40, s40, s30
s_cmp_lt_u32 s10, s11
s_cselect_b32 s30, 0x400, 0
s_or_b32 s40, s40, s30
s_cmp_le_u32 s10, s11
s_cselect_b32 s30, 0x800, 0
s_or_b32 s40, s40, s30
s_mov_b32 s10, 0x3
s_mov_b32 s11, 0x7
s_cmp_eq_i32 s10, s11
s_cselect_b32 s30, 0x1000, 0
s_or_b32 s40, s40, s30
s_cmp_lg_i32 s10, s11
s_cselect_b32 s30, 0x2000, 0
s_or_b32 s40, s40, s30
s_cmp_gt_i32 s10, s11
s_cselect_b32 s30, 0x4000, 0
s_or_b32 s40, s40, s30
s_cmp_ge_i32 s10, s11
s_cselect_b32 s30, 0x8000, 0
s_or_b32 s40, s40, s30
s_cmp_lt_i32 s10, s11
s_cselect_b32 s30, 0x10000, 0
s_or_b32 s40, s40, s30
s_cmp_le_i32 s10, s11
s_cselect_b32 s30, 0x20000, 0
s_or_b32 s40, s40, s30
s_cmp_eq_u32 s10, s11
s_cselect_b32 s30, 0x40000, 0
s_or_b32 s40, s40, s30
s_cmp_lg_u32 s10, s11
s_cselect_b32 s30, 0x80000, 0
s_or_b32 s40, s40, s30
s_cmp_gt_u32 s10, s11
s_cselect_b32 s30, 0x100000, 0
s_or_b32 s40, s40, s30
s_cmp_ge_u32 s10, s11
s_cselect_b32 s30, 0x200000, 0
s_or_b32 s40, s40, s30
s_cmp_lt_u32 s10, s11
s_cselect_b32 s30, 0x400000, 0
s_or_b32 s40, s40, s30
s_cmp_le_u32 s10, s11
s_cselect_b32 s30, 0x800000, 0
s_or_b32 s40, s40, s30
s_mov_b32 s10, 0x7
s_mov_b32 s11, 0x3
s_cmp_eq_i32 s10, s11
s_cselect_b32 s30, 0x1000000, 0
s_or_b32 s40, s40, s30
s_cmp_lg_i32 s10, s11
s_cselect_b32 s30, 0x2000000, 0
s_or_b32 s40, s40, s30
s_cmp_gt_i32 s10, s11
s_cselect_b32 s30, 0x4000000, 0
s_or_b32 s40, s40, s30
s_cmp_ge_i32 s10, s11
s_cselect_b32 s30, 0x8000000, 0
s_or_b32 s40, s40, s30
s_cmp_lt_i32 s10, s11
s_cselect_b32 s30, 0x10000000, 0
s_or_b32 s40, s40, s30
s_cmp_le_i32 s10, s11
s_cselect_b32 s30, 0x20000000, 0
s_or_b32 s40, s40, s30
s_cmp_eq_u32 s10, s11
s_cselect_b32 s30, 0x40000000, 0
s_or_b32 s40, s40, s30
s_cmp_lg_u32 s10, s11
s_cselect_b32 s30, 0x80000000, 0
s_or_b32 s40, s40, s30
s_cmp_gt_u32 s10, s11
s_cselect_b32 s30, 0x1, 0
s_or_b32 s41, s41, s30
s_cmp_ge_u32 s10, s11
s_cselect_b32 s30, 0x2, 0
s_or_b32 s41, s41, s30
s_cmp_lt_u32 s10, s11
s_cselect_b32 s30, 0x4, 0
s_or_b32 s41, s41, s30
s_cmp_le_u32 s10, s11
s_cselect_b32 s30, 0x8, 0
s_or_b32 s41, s41, s30
s_mov_b32 s10, 0xffffffff
s_mov_b32 s11, 0x1
s_cmp_eq_i32 s10, s11
s_cselect_b32 s30, 0x10, 0
s_or_b32 s41, s41, s30
s_cmp_lg_i32 s10, s11
s_cselect_b32 s30, 0x20, 0
s_or_b32 s41, s41, s30
s_cmp_gt_i32 s10, s11
s_cselect_b32 s30, 0x40, 0
s_or_b32 s41, s41, s30
s_cmp_ge_i32 s10, s11
s_cselect_b32 s30, 0x80, 0
s_or_b32 s41, s41, s30
s_cmp_lt_i32 s10, s11
s_cselect_b32 s30, 0x100, 0
s_or_b32 s41, s41, s30
s_cmp_le_i32 s10, s11
s_cselect_b32 s30, 0x200, 0
s_or_b32 s41, s41, s30
s_cmp_eq_u32 s10, s11
s_cselect_b32 s30, 0x400, 0
s_or_b32 s41, s41, s30
s_cmp_lg_u32 s10, s11
s_cselect_b32 s30, 0x800, 0
s_or_b32 s41, s41, s30
s_cmp_gt_u32 s10, s11
s_cselect_b32 s30, 0x1000, 0
s_or_b32 s41, s41, s30
s_cmp_ge_u32 s10, s11
s_cselect_b32 s30, 0x2000, 0
s_or_b32 s41, s41, s30
s_cmp_lt_u32 s10, s11
s_cselect_b32 s30, 0x4000, 0
s_or_b32 s41, s41, s30
s_cmp_le_u32 s10, s11
s_cselect_b32 s30, 0x8000, 0
s_or_b32 s41, s41, s30
s_mov_b32 s10, 0x1
s_mov_b32 s11, 0xffffffff
s_cmp_eq_i32 s10, s11
s_cselect_b32 s30, 0x10000, 0
s_or_b32 s41, s41, s30
s_cmp_lg_i32 s10, s11
s_cselect_b32 s30, 0x20000, 0
s_or_b32 s41, s41, s30
s_cmp_gt_i32 s10, s11
s_cselect_b32 s30, 0x40000, 0
s_or_b32 s41, s41, s30
s_cmp_ge_i32 s10, s11
s_cselect_b32 s30, 0x80000, 0
s_or_b32 s41, s41, s30
s_cmp_lt_i32 s10, s11
s_cselect_b32 s30, 0x100000, 0
s_or_b32 s41, s41, s30
s_cmp_le_i32 s10, s11
s_cselect_b32 s30, 0x200000, 0
s_or_b32 s41, s41, s30
s_cmp_eq_u32 s10, s11
s_cselect_b32 s30, 0x400000, 0
s_or_b32 s41, s41, s30
s_cmp_lg_u32 s10, s11
s_cselect_b32 s30, 0x800000, 0
s_or_b32 s41, s41, s30
s_cmp_gt_u32 s10, s11
s_cselect_b32 s30, 0x1000000, 0
s_or_b32 s41, s41, s30
s_cmp_ge_u32 s10, s11
s_cselect_b32 s30, 0x2000000, 0
s_or_b32 s41, s41, s30
s_cmp_lt_u32 s10, s11
s_cselect_b32 s30, 0x4000000, 0
s_or_b32 s41, s41, s30
s_cmp_le_u32 s10, s11
s_cselect_b32 s30, 0x8000000, 0
s_or_b32 s41, s41, s30
s_mov_b32 s10, 0x80000000
s_mov_b32 s11, 0x7fffffff
s_cmp_eq_i32 s10, s11
s_cselect_b32 s30, 0x10000000, 0
s_or_b32 s41, s41, s30
s_cmp_lg_i32 s10, s11
s_cselect_b32 s30, 0x20000000, 0
s_or_b32 s41, s41, s30
s_cmp_gt_i32 s10, s11
s_cselect_b32 s30, 0x40000000, 0
s_or_b32 s41, s41, s30
s_cmp_ge_i32 s10, s11
s_cselect_b32 s30, 0x80000000, 0
s_or_b32 s41, s41, s30
s_cmp_lt_i32 s10, s11
s_cselect_b32 s30, 0x1, 0
s_or_b32 s42, s42, s30
s_cmp_le_i32 s10, s11
s_cselect_b32 s30, 0x2, 0
s_or_b32 s42, s42, s30
s_cmp_eq_u32 s10, s11
s_cselect_b32 s30, 0x4, 0
s_or_b32 s42, s42, s30
s_cmp_lg_u32 s10, s11
s_cselect_b32 s30, 0x8, 0
s_or_b32 s42, s42, s30
s_cmp_gt_u32 s10, s11
s_cselect_b32 s30, 0x10, 0
s_or_b32 s42, s42, s30
s_cmp_ge_u32 s10, s11
s_cselect_b32 s30, 0x20, 0
s_or_b32 s42, s42, s30
s_cmp_lt_u32 s10, s11
s_cselect_b32 s30, 0x40, 0
s_or_b32 s42, s42, s30
s_cmp_le_u32 s10, s11
s_cselect_b32 s30, 0x80, 0
s_or_b32 s42, s42, s30
s_mov_b32 s10, 0x7fffffff
s_mov_b32 s11, 0x80000000
s_cmp_eq_i32 s10, s11
s_cselect_b32 s30, 0x100, 0
s_or_b32 s42, s42, s30
s_cmp_lg_i32 s10, s11
s_cselect_b32 s30, 0x200, 0
s_or_b32 s42, s42, s30
s_cmp_gt_i32 s10, s11
s_cselect_b32 s30, 0x400, 0
s_or_b32 s42, s42, s30
s_cmp_ge_i32 s10, s11
s_cselect_b32 s30, 0x800, 0
s_or_b32 s42, s42, s30
s_cmp_lt_i32 s10, s11
s_cselect_b32 s30, 0x1000, 0
s_or_b32 s42, s42, s30
s_cmp_le_i32 s10, s11
s_cselect_b32 s30, 0x2000, 0
s_or_b32 s42, s42, s30
s_cmp_eq_u32 s10, s11
s_cselect_b32 s30, 0x4000, 0
s_or_b32 s42, s42, s30
s_cmp_lg_u32 s10, s11
s_cselect_b32 s30, 0x8000, 0
s_or_b32 s42, s42, s30
s_cmp_gt_u32 s10, s11
s_cselect_b32 s30, 0x10000, 0
s_or_b32 s42, s42, s30
s_cmp_ge_u32 s10, s11
s_cselect_b32 s30, 0x20000, 0
s_or_b32 s42, s42, s30
s_cmp_lt_u32 s10, s11
s_cselect_b32 s30, 0x40000, 0
s_or_b32 s42, s42, s30
s_cmp_le_u32 s10, s11
s_cselect_b32 s30, 0x80000, 0
s_or_b32 s42, s42, s30
s_mov_b32 s10, 0x80000000
s_mov_b32 s11, 0x80000000
s_cmp_eq_i32 s10, s11
s_cselect_b32 s30, 0x100000, 0
s_or_b32 s42, s42, s30
s_cmp_lg_i32 s10, s11
s_cselect_b32 s30, 0x200000, 0
s_or_b32 s42, s42, s30
s_cmp_gt_i32 s10, s11
s_cselect_b32 s30, 0x400000, 0
s_or_b32 s42, s42, s30
s_cmp_ge_i32 s10, s11
s_cselect_b32 s30, 0x800000, 0
s_or_b32 s42, s42, s30
s_cmp_lt_i32 s10, s11
s_cselect_b32 s30, 0x1000000, 0
s_or_b32 s42, s42, s30
s_cmp_le_i32 s10, s11
s_cselect_b32 s30, 0x2000000, 0
s_or_b32 s42, s42, s30
s_cmp_eq_u32 s10, s11
s_cselect_b32 s30, 0x4000000, 0
s_or_b32 s42, s42, s30
s_cmp_lg_u32 s10, s11
s_cselect_b32 s30, 0x8000000, 0
s_or_b32 s42, s42, s30
s_cmp_gt_u32 s10, s11
s_cselect_b32 s30, 0x10000000, 0
s_or_b32 s42, s42, s30
s_cmp_ge_u32 s10, s11
s_cselect_b32 s30, 0x20000000, 0
s_or_b32 s42, s42, s30
s_cmp_lt_u32 s10, s11
s_cselect_b32 s30, 0x40000000, 0
s_or_b32 s42, s42, s30
s_cmp_le_u32 s10, s11
s_cselect_b32 s30, 0x80000000, 0
s_or_b32 s42, s42, s30
s_mov_b32 s10, 0x0
s_cmp_eq_i32 s10, -1
s_cselect_b32 s30, 0x1, 0
s_or_b32 s43, s43, s30
s_cmp_lg_i32 s10, -1
s_cselect_b32 s30, 0x2, 0
s_or_b32 s43, s43, s30
s_cmp_gt_i32 s10, -1
s_cselect_b32 s30, 0x4, 0
s_or_b32 s43, s43, s30
s_cmp_ge_i32 s10, -1
s_cselect_b32 s30, 0x8, 0
s_or_b32 s43, s43, s30
s_cmp_lt_i32 s10, -1
s_cselect_b32 s30, 0x10, 0
s_or_b32 s43, s43, s30
s_cmp_le_i32 s10, -1
s_cselect_b32 s30, 0x20, 0
s_or_b32 s43, s43, s30
s_cmp_eq_u32 s10, -1
s_cselect_b32 s30, 0x40, 0
s_or_b32 s43, s43, s30
s_cmp_lg_u32 s10, -1
s_cselect_b32 s30, 0x80, 0
s_or_b32 s43, s43, s30
s_cmp_gt_u32 s10, -1
s_cselect_b32 s30, 0x100, 0
s_or_b32 s43, s43, s30
s_cmp_ge_u32 s10, -1
s_cselect_b32 s30, 0x200, 0
s_or_b32 s43, s43, s30
s_cmp_lt_u32 s10, -1
s_cselect_b32 s30, 0x400, 0
s_or_b32 s43, s43, s30
s_cmp_le_u32 s10, -1
s_cselect_b32 s30, 0x800, 0
s_or_b32 s43, s43, s30
s_mov_b32 s10, 0x1
s_cmp_eq_i32 s10, -1
s_cselect_b32 s30, 0x1000, 0
s_or_b32 s43, s43, s30
s_cmp_lg_i32 s10, -1
s_cselect_b32 s30, 0x2000, 0
s_or_b32 s43, s43, s30
s_cmp_gt_i32 s10, -1
s_cselect_b32 s30, 0x4000, 0
s_or_b32 s43, s43, s30
s_cmp_ge_i32 s10, -1
s_cselect_b32 s30, 0x8000, 0
s_or_b32 s43, s43, s30
s_cmp_lt_i32 s10, -1
s_cselect_b32 s30, 0x10000, 0
s_or_b32 s43, s43, s30
s_cmp_le_i32 s10, -1
s_cselect_b32 s30, 0x20000, 0
s_or_b32 s43, s43, s30
s_cmp_eq_u32 s10, -1
s_cselect_b32 s30, 0x40000, 0
s_or_b32 s43, s43, s30
s_cmp_lg_u32 s10, -1
s_cselect_b32 s30, 0x80000, 0
s_or_b32 s43, s43, s30
s_cmp_gt_u32 s10, -1
s_cselect_b32 s30, 0x100000, 0
s_or_b32 s43, s43, s30
s_cmp_ge_u32 s10, -1
s_cselect_b32 s30, 0x200000, 0
s_or_b32 s43, s43, s30
s_cmp_lt_u32 s10, -1
s_cselect_b32 s30, 0x400000, 0
s_or_b32 s43, s43, s30
s_cmp_le_u32 s10, -1
s_cselect_b32 s30, 0x800000, 0
s_or_b32 s43, s43, s30
s_mov_b32 s10, 0x7fffffff
s_cmp_eq_i32 s10, -1
s_cselect_b32 s30, 0x1000000, 0
s_or_b32 s43, s43, s30
s_cmp_lg_i32 s10, -1
s_cselect_b32 s30, 0x2000000, 0
s_or_b32 s43, s43, s30
s_cmp_gt_i32 s10, -1
s_cselect_b32 s30, 0x4000000, 0
s_or_b32 s43, s43, s30
s_cmp_ge_i32 s10, -1
s_cselect_b32 s30, 0x8000000, 0
s_or_b32 s43, s43, s30
s_cmp_lt_i32 s10, -1
s_cselect_b32 s30, 0x10000000, 0
s_or_b32 s43, s43, s30
s_cmp_le_i32 s10, -1
s_cselect_b32 s30, 0x20000000, 0
s_or_b32 s43, s43, s30
s_cmp_eq_u32 s10, -1
s_cselect_b32 s30, 0x40000000, 0
s_or_b32 s43, s43, s30
s_cmp_lg_u32 s10, -1
s_cselect_b32 s30, 0x80000000, 0
s_or_b32 s43, s43, s30
s_cmp_gt_u32 s10, -1
s_cselect_b32 s30, 0x1, 0
s_or_b32 s44, s44, s30
s_cmp_ge_u32 s10, -1
s_cselect_b32 s30, 0x2, 0
s_or_b32 s44, s44, s30
s_cmp_lt_u32 s10, -1
s_cselect_b32 s30, 0x4, 0
s_or_b32 s44, s44, s30
s_cmp_le_u32 s10, -1
s_cselect_b32 s30, 0x8, 0
s_or_b32 s44, s44, s30
s_mov_b32 s10, 0x80000000
s_cmp_eq_i32 s10, -1
s_cselect_b32 s30, 0x10, 0
s_or_b32 s44, s44, s30
s_cmp_lg_i32 s10, -1
s_cselect_b32 s30, 0x20, 0
s_or_b32 s44, s44, s30
s_cmp_gt_i32 s10, -1
s_cselect_b32 s30, 0x40, 0
s_or_b32 s44, s44, s30
s_cmp_ge_i32 s10, -1
s_cselect_b32 s30, 0x80, 0
s_or_b32 s44, s44, s30
s_cmp_lt_i32 s10, -1
s_cselect_b32 s30, 0x100, 0
s_or_b32 s44, s44, s30
s_cmp_le_i32 s10, -1
s_cselect_b32 s30, 0x200, 0
s_or_b32 s44, s44, s30
s_cmp_eq_u32 s10, -1
s_cselect_b32 s30, 0x400, 0
s_or_b32 s44, s44, s30
s_cmp_lg_u32 s10, -1
s_cselect_b32 s30, 0x800, 0
s_or_b32 s44, s44, s30
s_cmp_gt_u32 s10, -1
s_cselect_b32 s30, 0x1000, 0
s_or_b32 s44, s44, s30
s_cmp_ge_u32 s10, -1
s_cselect_b32 s30, 0x2000, 0
s_or_b32 s44, s44, s30
s_cmp_lt_u32 s10, -1
s_cselect_b32 s30, 0x4000, 0
s_or_b32 s44, s44, s30
s_cmp_le_u32 s10, -1
s_cselect_b32 s30, 0x8000, 0
s_or_b32 s44, s44, s30
s_mov_b32 s10, 0xfffffffe
s_cmp_eq_i32 s10, -1
s_cselect_b32 s30, 0x10000, 0
s_or_b32 s44, s44, s30
s_cmp_lg_i32 s10, -1
s_cselect_b32 s30, 0x20000, 0
s_or_b32 s44, s44, s30
s_cmp_gt_i32 s10, -1
s_cselect_b32 s30, 0x40000, 0
s_or_b32 s44, s44, s30
s_cmp_ge_i32 s10, -1
s_cselect_b32 s30, 0x80000, 0
s_or_b32 s44, s44, s30
s_cmp_lt_i32 s10, -1
s_cselect_b32 s30, 0x100000, 0
s_or_b32 s44, s44, s30
s_cmp_le_i32 s10, -1
s_cselect_b32 s30, 0x200000, 0
s_or_b32 s44, s44, s30
s_cmp_eq_u32 s10, -1
s_cselect_b32 s30, 0x400000, 0
s_or_b32 s44, s44, s30
s_cmp_lg_u32 s10, -1
s_cselect_b32 s30, 0x800000, 0
s_or_b32 s44, s44, s30
s_cmp_gt_u32 s10, -1
s_cselect_b32 s30, 0x1000000, 0
s_or_b32 s44, s44, s30
s_cmp_ge_u32 s10, -1
s_cselect_b32 s30, 0x2000000, 0
s_or_b32 s44, s44, s30
s_cmp_lt_u32 s10, -1
s_cselect_b32 s30, 0x4000000, 0
s_or_b32 s44, s44, s30
s_cmp_le_u32 s10, -1
s_cselect_b32 s30, 0x8000000, 0
s_or_b32 s44, s44, s30
s_mov_b32 s10, 0xffffffff
s_cmp_eq_i32 s10, -1
s_cselect_b32 s30, 0x10000000, 0
s_or_b32 s44, s44, s30
s_cmp_lg_i32 s10, -1
s_cselect_b32 s30, 0x20000000, 0
s_or_b32 s44, s44, s30
s_cmp_gt_i32 s10, -1
s_cselect_b32 s30, 0x40000000, 0
s_or_b32 s44, s44, s30
s_cmp_ge_i32 s10, -1
s_cselect_b32 s30, 0x80000000, 0
s_or_b32 s44, s44, s30
s_cmp_lt_i32 s10, -1
s_cselect_b32 s30, 0x1, 0
s_or_b32 s45, s45, s30
s_cmp_le_i32 s10, -1
s_cselect_b32 s30, 0x2, 0
s_or_b32 s45, s45, s30
s_cmp_eq_u32 s10, -1
s_cselect_b32 s30, 0x4, 0
s_or_b32 s45, s45, s30
s_cmp_lg_u32 s10, -1
s_cselect_b32 s30, 0x8, 0
s_or_b32 s45, s45, s30
s_cmp_gt_u32 s10, -1
s_cselect_b32 s30, 0x10, 0
s_or_b32 s45, s45, s30
s_cmp_ge_u32 s10, -1
s_cselect_b32 s30, 0x20, 0
s_or_b32 s45, s45, s30
s_cmp_lt_u32 s10, -1
s_cselect_b32 s30, 0x40, 0
s_or_b32 s45, s45, s30
s_cmp_le_u32 s10, -1
s_cselect_b32 s30, 0x80, 0
s_or_b32 s45, s45, s30
s_mov_b32 s10, 0x12345677
s_cmp_eq_i32 s10, 0x12345678
s_cselect_b32 s30, 0x100, 0
s_or_b32 s45, s45, s30
s_cmp_lg_i32 s10, 0x12345678
s_cselect_b32 s30, 0x200, 0
s_or_b32 s45, s45, s30
s_cmp_gt_i32 s10, 0x12345678
s_cselect_b32 s30, 0x400, 0
s_or_b32 s45, s45, s30
s_cmp_ge_i32 s10, 0x12345678
s_cselect_b32 s30, 0x800, 0
s_or_b32 s45, s45, s30
s_cmp_lt_i32 s10, 0x12345678
s_cselect_b32 s30, 0x1000, 0
s_or_b32 s45, s45, s30
s_cmp_le_i32 s10, 0x12345678
s_cselect_b32 s30, 0x2000, 0
s_or_b32 s45, s45, s30
s_cmp_eq_u32 s10, 0x12345678
s_cselect_b32 s30, 0x4000, 0
s_or_b32 s45, s45, s30
s_cmp_lg_u32 s10, 0x12345678
s_cselect_b32 s30, 0x8000, 0
s_or_b32 s45, s45, s30
s_cmp_gt_u32 s10, 0x12345678
s_cselect_b32 s30, 0x10000, 0
s_or_b32 s45, s45, s30
s_cmp_ge_u32 s10, 0x12345678
s_cselect_b32 s30, 0x20000, 0
s_or_b32 s45, s45, s30
s_cmp_lt_u32 s10, 0x12345678
s_cselect_b32 s30, 0x40000, 0
s_or_b32 s45, s45, s30
s_cmp_le_u32 s10, 0x12345678
s_cselect_b32 s30, 0x80000, 0
s_or_b32 s45, s45, s30
s_mov_b32 s10, 0x12345678
s_cmp_eq_i32 s10, 0x12345678
s_cselect_b32 s30, 0x100000, 0
s_or_b32 s45, s45, s30
s_cmp_lg_i32 s10, 0x12345678
s_cselect_b32 s30, 0x200000, 0
s_or_b32 s45, s45, s30
s_cmp_gt_i32 s10, 0x12345678
s_cselect_b32 s30, 0x400000, 0
s_or_b32 s45, s45, s30
s_cmp_ge_i32 s10, 0x12345678
s_cselect_b32 s30, 0x800000, 0
s_or_b32 s45, s45, s30
s_cmp_lt_i32 s10, 0x12345678
s_cselect_b32 s30, 0x1000000, 0
s_or_b32 s45, s45, s30
s_cmp_le_i32 s10, 0x12345678
s_cselect_b32 s30, 0x2000000, 0
s_or_b32 s45, s45, s30
s_cmp_eq_u32 s10, 0x12345678
s_cselect_b32 s30, 0x4000000, 0
s_or_b32 s45, s45, s30
s_cmp_lg_u32 s10, 0x12345678
s_cselect_b32 s30, 0x8000000, 0
s_or_b32 s45, s45, s30
s_cmp_gt_u32 s10, 0x12345678
s_cselect_b32 s30, 0x10000000, 0
s_or_b32 s45, s45, s30
s_cmp_ge_u32 s10, 0x12345678
s_cselect_b32 s30, 0x20000000, 0
s_or_b32 s45, s45, s30
s_cmp_lt_u32 s10, 0x12345678
s_cselect_b32 s30, 0x40000000, 0
s_or_b32 s45, s45, s30
s_cmp_le_u32 s10, 0x12345678
s_cselect_b32 s30, 0x80000000, 0
s_or_b32 s45, s45, s30
s_mov_b32 s10, 0x12345679
s_cmp_eq_i32 s10, 0x12345678
s_cselect_b32 s30, 0x1, 0
s_or_b32 s46, s46, s30
s_cmp_lg_i32 s10, 0x12345678
s_cselect_b32 s30, 0x2, 0
s_or_b32 s46, s46, s30
s_cmp_gt_i32 s10, 0x12345678
s_cselect_b32 s30, 0x4, 0
s_or_b32 s46, s46, s30
s_cmp_ge_i32 s10, 0x12345678
s_cselect_b32 s30, 0x8, 0
s_or_b32 s46, s46, s30
s_cmp_lt_i32 s10, 0x12345678
s_cselect_b32 s30, 0x10, 0
s_or_b32 s46, s46, s30
s_cmp_le_i32 s10, 0x12345678
s_cselect_b32 s30, 0x20, 0
s_or_b32 s46, s46, s30
s_cmp_eq_u32 s10, 0x12345678
s_cselect_b32 s30, 0x40, 0
s_or_b32 s46, s46, s30
s_cmp_lg_u32 s10, 0x12345678
s_cselect_b32 s30, 0x80, 0
s_or_b32 s46, s46, s30
s_cmp_gt_u32 s10, 0x12345678
s_cselect_b32 s30, 0x100, 0
s_or_b32 s46, s46, s30
s_cmp_ge_u32 s10, 0x12345678
s_cselect_b32 s30, 0x200, 0
s_or_b32 s46, s46, s30
s_cmp_lt_u32 s10, 0x12345678
s_cselect_b32 s30, 0x400, 0
s_or_b32 s46, s46, s30
s_cmp_le_u32 s10, 0x12345678
s_cselect_b32 s30, 0x800, 0
s_or_b32 s46, s46, s30
s_mov_b32 s10, 0xf2345678
s_cmp_eq_i32 s10, 0x12345678
s_cselect_b32 s30, 0x1000, 0
s_or_b32 s46, s46, s30
s_cmp_lg_i32 s10, 0x12345678
s_cselect_b32 s30, 0x2000, 0
s_or_b32 s46, s46, s30
s_cmp_gt_i32 s10, 0x12345678
s_cselect_b32 s30, 0x4000, 0
s_or_b32 s46, s46, s30
s_cmp_ge_i32 s10, 0x12345678
s_cselect_b32 s30, 0x8000, 0
s_or_b32 s46, s46, s30
s_cmp_lt_i32 s10, 0x12345678
s_cselect_b32 s30, 0x10000, 0
s_or_b32 s46, s46, s30
s_cmp_le_i32 s10, 0x12345678
s_cselect_b32 s30, 0x20000, 0
s_or_b32 s46, s46, s30
s_cmp_eq_u32 s10, 0x12345678
s_cselect_b32 s30, 0x40000, 0
s_or_b32 s46, s46, s30
s_cmp_lg_u32 s10, 0x12345678
s_cselect_b32 s30, 0x80000, 0
s_or_b32 s46, s46, s30
s_cmp_gt_u32 s10, 0x12345678
s_cselect_b32 s30, 0x100000, 0
s_or_b32 s46, s46, s30
s_cmp_ge_u32 s10, 0x12345678
s_cselect_b32 s30, 0x200000, 0
s_or_b32 s46, s46, s30
s_cmp_lt_u32 s10, 0x12345678
s_cselect_b32 s30, 0x400000, 0
s_or_b32 s46, s46, s30
s_cmp_le_u32 s10, 0x12345678
s_cselect_b32 s30, 0x800000, 0
s_or_b32 s46, s46, s30
s_mov_b32 s12, 0x80000005
s_bitcmp0_b32 s12, 0
s_cselect_b32 s30, 0x1000000, 0
s_or_b32 s46, s46, s30
s_bitcmp0_b32 s12, 1
s_cselect_b32 s30, 0x2000000, 0
s_or_b32 s46, s46, s30
s_bitcmp0_b32 s12, 2
s_cselect_b32 s30, 0x4000000, 0
s_or_b32 s46, s46, s30
s_bitcmp0_b32 s12, 31
s_cselect_b32 s30, 0x8000000, 0
s_or_b32 s46, s46, s30
s_bitcmp0_b32 s12, 32
s_cselect_b32 s30, 0x10000000, 0
s_or_b32 s46, s46, s30
s_bitcmp0_b32 s12, 33
s_cselect_b32 s30, 0x20000000, 0
s_or_b32 s46, s46, s30
s_bitcmp0_b32 s12, 34
s_cselect_b32 s30, 0x40000000, 0
s_or_b32 s46, s46, s30
s_bitcmp1_b32 s12, 0
s_cselect_b32 s30, 0x80000000, 0
s_or_b32 s46, s46, s30
s_bitcmp1_b32 s12, 1
s_cselect_b32 s30, 0x1, 0
s_or_b32 s47, s47, s30
s_bitcmp1_b32 s12, 2
s_cselect_b32 s30, 0x2, 0
s_or_b32 s47, s47, s30
s_bitcmp1_b32 s12, 31
s_cselect_b32 s30, 0x4, 0
s_or_b32 s47, s47, s30
s_bitcmp1_b32 s12, 32
s_cselect_b32 s30, 0x8, 0
s_or_b32 s47, s47, s30
s_bitcmp1_b32 s12, 33
s_cselect_b32 s30, 0x10, 0
s_or_b32 s47, s47, s30
s_bitcmp1_b32 s12, 34
s_cselect_b32 s30, 0x20, 0
s_or_b32 s47, s47, s30
s_mov_b32 s14, 0x80000004
s_mov_b32 s15, 0x1
s_bitcmp0_b64 s[14:15], 0
s_cselect_b32 s30, 0x40, 0
s_or_b32 s47, s47, s30
s_bitcmp0_b64 s[14:15], 2
s_cselect_b32 s30, 0x80, 0
s_or_b32 s47, s47, s30
s_bitcmp0_b64 s[14:15], 31
s_cselect_b32 s30, 0x100, 0
s_or_b32 s47, s47, s30
s_bitcmp0_b64 s[14:15], 32
s_cselect_b32 s30, 0x200, 0
s_or_b32 s47, s47, s30
s_bitcmp0_b64 s[14:15], 33
s_cselect_b32 s30, 0x400, 0
s_or_b32 s47, s47, s30
s_bitcmp0_b64 s[14:15], 63
s_cselect_b32 s30, 0x800, 0
s_or_b32 s47, s47, s30
s_bitcmp0_b64 s[14:15], 64
s_cselect_b32 s30, 0x1000, 0
s_or_b32 s47, s47, s30
s_bitcmp0_b64 s[14:15], 66
s_cselect_b32 s30, 0x2000, 0
s_or_b32 s47, s47, s30
s_bitcmp1_b64 s[14:15], 0
s_cselect_b32 s30, 0x4000, 0
s_or_b32 s47, s47, s30
s_bitcmp1_b64 s[14:15], 2
s_cselect_b32 s30, 0x8000, 0
s_or_b32 s47, s47, s30
s_bitcmp1_b64 s[14:15], 31
s_cselect_b32 s30, 0x10000, 0
s_or_b32 s47, s47, s30
s_bitcmp1_b64 s[14:15], 32
s_cselect_b32 s30, 0x20000, 0
s_or_b32 s47, s47, s30
s_bitcmp1_b64 s[14:15], 33
s_cselect_b32 s30, 0x40000, 0
s_or_b32 s47, s47, s30
s_bitcmp1_b64 s[14:15], 63
s_cselect_b32 s30, 0x80000, 0
s_or_b32 s47, s47, s30
s_bitcmp1_b64 s[14:15], 64
s_cselect_b32 s30, 0x100000, 0
s_or_b32 s47, s47, s30
s_bitcmp1_b64 s[14:15], 66
s_cselect_b32 s30, 0x200000, 0
s_or_b32 s47, s47, s30
v_mov_b32_e32 v20, s40
v_mov_b32_e32 v21, 0
buffer_store_dword v20, v21, s[0:3], 0 offen
v_mov_b32_e32 v20, s41
v_mov_b32_e32 v21, 4
buffer_store_dword v20, v21, s[0:3], 0 offen
v_mov_b32_e32 v20, s42
v_mov_b32_e32 v21, 8
buffer_store_dword v20, v21, s[0:3], 0 offen
v_mov_b32_e32 v20, s43
v_mov_b32_e32 v21, 12
buffer_store_dword v20, v21, s[0:3], 0 offen
v_mov_b32_e32 v20, s44
v_mov_b32_e32 v21, 16
buffer_store_dword v20, v21, s[0:3], 0 offen
v_mov_b32_e32 v20, s45
v_mov_b32_e32 v21, 20
buffer_store_dword v20, v21, s[0:3], 0 offen
v_mov_b32_e32 v20, s46
v_mov_b32_e32 v21, 24
buffer_store_dword v20, v21, s[0:3], 0 offen
v_mov_b32_e32 v20, s47
v_mov_b32_e32 v21, 28
buffer_store_dword v20, v21, s[0:3], 0 offen
s_endpgm
