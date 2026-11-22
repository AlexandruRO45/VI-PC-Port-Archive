; 32-bit integer v_cmpx: v_cmpx_{eq,ne,lt,le,gt,ge}_{i32,u32}, in the e32 form and the e64 form, with inline constants
; and literals as operands, on values where signed and unsigned disagree. v_cmpx_f_i32 and v_cmpx_t_u32 are in
; ps_vcmpx_forms, which also holds the f32 and 64-bit v_cmpx forms.
; RDNA2 ISA: _U32 compares S0.u32 with S1.u32, _I32 compares S0.i32 with S1.i32 (two's complement), per lane;
; EQ = S0 == S1, NE = S0 != S1, LT = S0 < S1, LE = S0 <= S1, GT = S0 > S1, GE = S0 >= S1. A v_cmpx writes the result
; to EXEC (EXEC = EXEC & compare, per lane) and to nothing else, so VCC is kept. Each compare runs alone: EXEC is
; copied out and put back after it (s20 holds the EXEC the shader started with, s21 the copy), so a lane that was
; true shows as 1 and every lane has the same operands. Inputs: the V# in s[0:3] points at 0x1000 (num_records 32),
; the test seeds 8 dwords that become v1..v6: 1, 0, 0x80000000, 0x12345678, 3, 5 (then two unused zeros).
; Five pairs (A, B) with A as src0 in e32; A is a constant or VGPR, B is a VGPR:
;   P0: A = literal 0x80000000, B = v1 = 1           unsigned A > B, signed A < B
;   P1: A = inline -1,          B = v2 = 0           unsigned A > B (0xffffffff), signed A < B
;   P2: A = inline 1,           B = v3 = 0x80000000  unsigned A < B, signed A > B
;   P3: A = literal 0x12345678, B = v4 = 0x12345678  equal
;   P4: A = v5 = 3,             B = v6 = 5           A < B both ways
; Result bit 5*k + i of a word is op k (eq, ne, lt, le, gt, ge = 0..5) on pair Pi. The five-bit masks (bit i = Pi):
;   unsigned: eq P3 = 0x08   ne P0 P1 P2 P4 = 0x17   lt P2 P4 = 0x14   le P2 P3 P4 = 0x1c
;             gt P0 P1 = 0x03   ge P0 P1 P3 = 0x0b
;   signed:   eq P3 = 0x08   ne P0 P1 P2 P4 = 0x17   lt P0 P1 P4 = 0x13   le P0 P1 P3 P4 = 0x1b
;             gt P2 = 0x04   ge P2 P3 = 0x0c
; Bits 30 and 31 test the EXEC and VCC effects with one op of the word's form each:
;   bit 30: v_cmpx_ne_u32 v2, v2 is false in every lane and empties EXEC; the op after it is true on (3, 5), but
;           v_cmpx only ANDs into EXEC, so no lane comes back: the bit is 0.
;   bit 31: VCC = 1 in every lane (v_cmp_eq_u32 v2, v2), then an op that is false on (3, 5) empties EXEC; after EXEC
;           is restored VCC must still be 1 (a v_cmpx writes EXEC only): the bit is 1.
; word 0 (i32, e32) = 0x08 | 0x17<<5 | 0x13<<10 | 0x1b<<15 | 0x04<<20 | 0x0c<<25 | 1<<31
;                   = 0x8 + 0x2e0 + 0x4c00 + 0xd8000 + 0x400000 + 0x18000000 + 0x80000000 = 0x984dcee8
; word 1 (u32, e32) = 0x08 | 0x17<<5 | 0x14<<10 | 0x1c<<15 | 0x03<<20 | 0x0b<<25 | 1<<31
;                   = 0x8 + 0x2e0 + 0x5000 + 0xe0000 + 0x300000 + 0x16000000 + 0x80000000 = 0x963e52e8
; words 2 and 3 are the e64 forms with the operands swapped: src0 is the VGPR B, src1 is the constant (or v5 for
; P4), so "op(B, A)" is the mirrored relation of op(A, B): lt <-> gt and le <-> ge, eq and ne are unchanged. The
; masks by position k = eq, ne, lt, le, gt, ge are therefore the masks of eq, ne, gt, ge, lt, le:
; word 2 (i32, e64) = 0x08 | 0x17<<5 | 0x04<<10 | 0x0c<<15 | 0x13<<20 | 0x1b<<25 | 1<<31
;                   = 0x8 + 0x2e0 + 0x1000 + 0x60000 + 0x1300000 + 0x36000000 + 0x80000000 = 0xb73612e8
; word 3 (u32, e64) = 0x08 | 0x17<<5 | 0x03<<10 | 0x0b<<15 | 0x14<<20 | 0x1c<<25 | 1<<31
;                   = 0x8 + 0x2e0 + 0xc00 + 0x58000 + 0x1400000 + 0x38000000 + 0x80000000 = 0xb9458ee8
; A translation that compared one of the _u32 forms signed would clear the P0/P1 ge and gt bits and set the P2 ones;
; one that compared an _i32 form unsigned would do the opposite; a v_cmpx that wrote VCC, or that did not AND with
; the old EXEC, flips bit 31 or bit 30.
; Expected mrt0 bits: 0x984dcee8, 0x963e52e8, 0xb73612e8, 0xb9458ee8
s_buffer_load_dwordx8 s[4:11], s[0:3], 0
s_waitcnt lgkmcnt(0)
v_mov_b32_e32 v1, s4
v_mov_b32_e32 v2, s5
v_mov_b32_e32 v3, s6
v_mov_b32_e32 v4, s7
v_mov_b32_e32 v5, s8
v_mov_b32_e32 v6, s9
s_mov_b32 s20, exec_lo
; word 0: v_cmpx_*_i32_e32, A is src0
v_mov_b32_e32 v40, 0
v_cmpx_eq_i32_e32 0x80000000, v1
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v40, v30, 0, v40
v_cmpx_eq_i32_e32 -1, v2
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v40, v30, 1, v40
v_cmpx_eq_i32_e32 1, v3
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v40, v30, 2, v40
v_cmpx_eq_i32_e32 0x12345678, v4
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v40, v30, 3, v40
v_cmpx_eq_i32_e32 v5, v6
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v40, v30, 4, v40
v_cmpx_ne_i32_e32 0x80000000, v1
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v40, v30, 5, v40
v_cmpx_ne_i32_e32 -1, v2
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v40, v30, 6, v40
v_cmpx_ne_i32_e32 1, v3
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v40, v30, 7, v40
v_cmpx_ne_i32_e32 0x12345678, v4
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v40, v30, 8, v40
v_cmpx_ne_i32_e32 v5, v6
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v40, v30, 9, v40
v_cmpx_lt_i32_e32 0x80000000, v1
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v40, v30, 10, v40
v_cmpx_lt_i32_e32 -1, v2
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v40, v30, 11, v40
v_cmpx_lt_i32_e32 1, v3
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v40, v30, 12, v40
v_cmpx_lt_i32_e32 0x12345678, v4
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v40, v30, 13, v40
v_cmpx_lt_i32_e32 v5, v6
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v40, v30, 14, v40
v_cmpx_le_i32_e32 0x80000000, v1
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v40, v30, 15, v40
v_cmpx_le_i32_e32 -1, v2
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v40, v30, 16, v40
v_cmpx_le_i32_e32 1, v3
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v40, v30, 17, v40
v_cmpx_le_i32_e32 0x12345678, v4
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v40, v30, 18, v40
v_cmpx_le_i32_e32 v5, v6
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v40, v30, 19, v40
v_cmpx_gt_i32_e32 0x80000000, v1
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v40, v30, 20, v40
v_cmpx_gt_i32_e32 -1, v2
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v40, v30, 21, v40
v_cmpx_gt_i32_e32 1, v3
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v40, v30, 22, v40
v_cmpx_gt_i32_e32 0x12345678, v4
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v40, v30, 23, v40
v_cmpx_gt_i32_e32 v5, v6
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v40, v30, 24, v40
v_cmpx_ge_i32_e32 0x80000000, v1
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v40, v30, 25, v40
v_cmpx_ge_i32_e32 -1, v2
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v40, v30, 26, v40
v_cmpx_ge_i32_e32 1, v3
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v40, v30, 27, v40
v_cmpx_ge_i32_e32 0x12345678, v4
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v40, v30, 28, v40
v_cmpx_ge_i32_e32 v5, v6
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v40, v30, 29, v40
v_cmpx_ne_u32_e32 v2, v2
v_cmpx_lt_i32_e32 v5, v6
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v40, v30, 30, v40
v_cmp_eq_u32_e32 vcc_lo, v2, v2
v_cmpx_gt_i32_e32 v5, v6
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, vcc_lo
v_lshl_or_b32 v40, v30, 31, v40
; word 1: v_cmpx_*_u32_e32, A is src0
v_mov_b32_e32 v41, 0
v_cmpx_eq_u32_e32 0x80000000, v1
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v41, v30, 0, v41
v_cmpx_eq_u32_e32 -1, v2
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v41, v30, 1, v41
v_cmpx_eq_u32_e32 1, v3
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v41, v30, 2, v41
v_cmpx_eq_u32_e32 0x12345678, v4
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v41, v30, 3, v41
v_cmpx_eq_u32_e32 v5, v6
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v41, v30, 4, v41
v_cmpx_ne_u32_e32 0x80000000, v1
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v41, v30, 5, v41
v_cmpx_ne_u32_e32 -1, v2
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v41, v30, 6, v41
v_cmpx_ne_u32_e32 1, v3
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v41, v30, 7, v41
v_cmpx_ne_u32_e32 0x12345678, v4
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v41, v30, 8, v41
v_cmpx_ne_u32_e32 v5, v6
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v41, v30, 9, v41
v_cmpx_lt_u32_e32 0x80000000, v1
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v41, v30, 10, v41
v_cmpx_lt_u32_e32 -1, v2
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v41, v30, 11, v41
v_cmpx_lt_u32_e32 1, v3
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v41, v30, 12, v41
v_cmpx_lt_u32_e32 0x12345678, v4
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v41, v30, 13, v41
v_cmpx_lt_u32_e32 v5, v6
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v41, v30, 14, v41
v_cmpx_le_u32_e32 0x80000000, v1
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v41, v30, 15, v41
v_cmpx_le_u32_e32 -1, v2
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v41, v30, 16, v41
v_cmpx_le_u32_e32 1, v3
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v41, v30, 17, v41
v_cmpx_le_u32_e32 0x12345678, v4
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v41, v30, 18, v41
v_cmpx_le_u32_e32 v5, v6
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v41, v30, 19, v41
v_cmpx_gt_u32_e32 0x80000000, v1
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v41, v30, 20, v41
v_cmpx_gt_u32_e32 -1, v2
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v41, v30, 21, v41
v_cmpx_gt_u32_e32 1, v3
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v41, v30, 22, v41
v_cmpx_gt_u32_e32 0x12345678, v4
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v41, v30, 23, v41
v_cmpx_gt_u32_e32 v5, v6
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v41, v30, 24, v41
v_cmpx_ge_u32_e32 0x80000000, v1
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v41, v30, 25, v41
v_cmpx_ge_u32_e32 -1, v2
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v41, v30, 26, v41
v_cmpx_ge_u32_e32 1, v3
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v41, v30, 27, v41
v_cmpx_ge_u32_e32 0x12345678, v4
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v41, v30, 28, v41
v_cmpx_ge_u32_e32 v5, v6
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v41, v30, 29, v41
v_cmpx_ne_u32_e32 v2, v2
v_cmpx_le_u32_e32 v5, v6
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v41, v30, 30, v41
v_cmp_eq_u32_e32 vcc_lo, v2, v2
v_cmpx_ge_u32_e32 v5, v6
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, vcc_lo
v_lshl_or_b32 v41, v30, 31, v41
; word 2: v_cmpx_*_i32_e64, B is src0
v_mov_b32_e32 v42, 0
v_cmpx_eq_i32_e64 v1, 0x80000000
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v42, v30, 0, v42
v_cmpx_eq_i32_e64 v2, -1
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v42, v30, 1, v42
v_cmpx_eq_i32_e64 v3, 1
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v42, v30, 2, v42
v_cmpx_eq_i32_e64 v4, 0x12345678
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v42, v30, 3, v42
v_cmpx_eq_i32_e64 v6, v5
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v42, v30, 4, v42
v_cmpx_ne_i32_e64 v1, 0x80000000
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v42, v30, 5, v42
v_cmpx_ne_i32_e64 v2, -1
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v42, v30, 6, v42
v_cmpx_ne_i32_e64 v3, 1
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v42, v30, 7, v42
v_cmpx_ne_i32_e64 v4, 0x12345678
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v42, v30, 8, v42
v_cmpx_ne_i32_e64 v6, v5
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v42, v30, 9, v42
v_cmpx_lt_i32_e64 v1, 0x80000000
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v42, v30, 10, v42
v_cmpx_lt_i32_e64 v2, -1
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v42, v30, 11, v42
v_cmpx_lt_i32_e64 v3, 1
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v42, v30, 12, v42
v_cmpx_lt_i32_e64 v4, 0x12345678
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v42, v30, 13, v42
v_cmpx_lt_i32_e64 v6, v5
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v42, v30, 14, v42
v_cmpx_le_i32_e64 v1, 0x80000000
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v42, v30, 15, v42
v_cmpx_le_i32_e64 v2, -1
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v42, v30, 16, v42
v_cmpx_le_i32_e64 v3, 1
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v42, v30, 17, v42
v_cmpx_le_i32_e64 v4, 0x12345678
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v42, v30, 18, v42
v_cmpx_le_i32_e64 v6, v5
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v42, v30, 19, v42
v_cmpx_gt_i32_e64 v1, 0x80000000
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v42, v30, 20, v42
v_cmpx_gt_i32_e64 v2, -1
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v42, v30, 21, v42
v_cmpx_gt_i32_e64 v3, 1
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v42, v30, 22, v42
v_cmpx_gt_i32_e64 v4, 0x12345678
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v42, v30, 23, v42
v_cmpx_gt_i32_e64 v6, v5
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v42, v30, 24, v42
v_cmpx_ge_i32_e64 v1, 0x80000000
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v42, v30, 25, v42
v_cmpx_ge_i32_e64 v2, -1
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v42, v30, 26, v42
v_cmpx_ge_i32_e64 v3, 1
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v42, v30, 27, v42
v_cmpx_ge_i32_e64 v4, 0x12345678
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v42, v30, 28, v42
v_cmpx_ge_i32_e64 v6, v5
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v42, v30, 29, v42
v_cmpx_ne_u32_e32 v2, v2
v_cmpx_ne_i32_e64 v5, v6
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v42, v30, 30, v42
v_cmp_eq_u32_e32 vcc_lo, v2, v2
v_cmpx_eq_i32_e64 v5, v6
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, vcc_lo
v_lshl_or_b32 v42, v30, 31, v42
; word 3: v_cmpx_*_u32_e64, B is src0
v_mov_b32_e32 v43, 0
v_cmpx_eq_u32_e64 v1, 0x80000000
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v43, v30, 0, v43
v_cmpx_eq_u32_e64 v2, -1
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v43, v30, 1, v43
v_cmpx_eq_u32_e64 v3, 1
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v43, v30, 2, v43
v_cmpx_eq_u32_e64 v4, 0x12345678
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v43, v30, 3, v43
v_cmpx_eq_u32_e64 v6, v5
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v43, v30, 4, v43
v_cmpx_ne_u32_e64 v1, 0x80000000
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v43, v30, 5, v43
v_cmpx_ne_u32_e64 v2, -1
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v43, v30, 6, v43
v_cmpx_ne_u32_e64 v3, 1
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v43, v30, 7, v43
v_cmpx_ne_u32_e64 v4, 0x12345678
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v43, v30, 8, v43
v_cmpx_ne_u32_e64 v6, v5
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v43, v30, 9, v43
v_cmpx_lt_u32_e64 v1, 0x80000000
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v43, v30, 10, v43
v_cmpx_lt_u32_e64 v2, -1
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v43, v30, 11, v43
v_cmpx_lt_u32_e64 v3, 1
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v43, v30, 12, v43
v_cmpx_lt_u32_e64 v4, 0x12345678
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v43, v30, 13, v43
v_cmpx_lt_u32_e64 v6, v5
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v43, v30, 14, v43
v_cmpx_le_u32_e64 v1, 0x80000000
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v43, v30, 15, v43
v_cmpx_le_u32_e64 v2, -1
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v43, v30, 16, v43
v_cmpx_le_u32_e64 v3, 1
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v43, v30, 17, v43
v_cmpx_le_u32_e64 v4, 0x12345678
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v43, v30, 18, v43
v_cmpx_le_u32_e64 v6, v5
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v43, v30, 19, v43
v_cmpx_gt_u32_e64 v1, 0x80000000
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v43, v30, 20, v43
v_cmpx_gt_u32_e64 v2, -1
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v43, v30, 21, v43
v_cmpx_gt_u32_e64 v3, 1
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v43, v30, 22, v43
v_cmpx_gt_u32_e64 v4, 0x12345678
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v43, v30, 23, v43
v_cmpx_gt_u32_e64 v6, v5
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v43, v30, 24, v43
v_cmpx_ge_u32_e64 v1, 0x80000000
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v43, v30, 25, v43
v_cmpx_ge_u32_e64 v2, -1
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v43, v30, 26, v43
v_cmpx_ge_u32_e64 v3, 1
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v43, v30, 27, v43
v_cmpx_ge_u32_e64 v4, 0x12345678
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v43, v30, 28, v43
v_cmpx_ge_u32_e64 v6, v5
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v43, v30, 29, v43
v_cmpx_ne_u32_e32 v2, v2
v_cmpx_ge_u32_e64 v6, v5
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v43, v30, 30, v43
v_cmp_eq_u32_e32 vcc_lo, v2, v2
v_cmpx_lt_u32_e64 v6, v5
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, vcc_lo
v_lshl_or_b32 v43, v30, 31, v43
exp mrt0 v40, v41, v42, v43 done vm
s_endpgm
