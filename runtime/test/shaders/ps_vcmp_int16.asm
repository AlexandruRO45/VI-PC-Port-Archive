; 16-bit integer VOPC compares: v_cmp_{eq,ne,lt,le,gt,ge}_{i16,u16}, in the e32 form (VCC) and the e64 form (SGPR
; destination), with inline constants and 16-bit literals as operands, on values where signed and unsigned disagree
; and whose high halves hold junk. v_cmp_f/t_{i16,u16} have no gfx10 encoding (llvm-mc rejects them) and are not used.
; RDNA2 ISA: _U16 compares S0.u16 with S1.u16, _I16 compares S0.i16 with S1.i16 (two's complement), per lane, on the
; low 16 bits of the operands only; EQ = S0 == S1, NE = S0 != S1, LT = S0 < S1, LE = S0 <= S1, GT = S0 > S1,
; GE = S0 >= S1. The result goes to VCC (e32) or to the SGPR pair named by VDST (e64). Inputs: the V# in s[0:3]
; points at 0x1000 (num_records 32), the test seeds 8 dwords that become v1..v6:
;   0xaaaa0001 0x55550000 0x12348000 0xabcd1234 0xffff0003 0x00000005 (then two unused zeros)
; Five pairs (A, B) with A as src0 in e32; A is a constant or VGPR, B is a VGPR; only the low halves count:
;   P0: A = literal 0x8000,     B = v1 low 0x0001  unsigned A > B, signed A (-32768) < B
;   P1: A = inline -1 (0xffff), B = v2 low 0x0000  unsigned A > B, signed A < B
;   P2: A = inline 1,           B = v3 low 0x8000  unsigned A < B, signed A > B
;   P3: A = literal 0x1234,     B = v4 low 0x1234  equal (A's high half is 0, B's is 0xabcd)
;   P4: A = v5 low 0x0003,      B = v6 low 0x0005  A < B both ways (the high half of v5 is 0xffff, so a 32-bit
;                                                  compare of the registers would call A greater than B)
; Result bit 5*k + i of a word is op k (eq, ne, lt, le, gt, ge = 0..5) on pair Pi; bits 30 and 31 are 0. The
; five-bit masks (bit i = Pi):
;   unsigned: eq P3 = 0x08   ne P0 P1 P2 P4 = 0x17   lt P2 P4 = 0x14   le P2 P3 P4 = 0x1c
;             gt P0 P1 = 0x03   ge P0 P1 P3 = 0x0b
;   signed:   eq P3 = 0x08   ne P0 P1 P2 P4 = 0x17   lt P0 P1 P4 = 0x13   le P0 P1 P3 P4 = 0x1b
;             gt P2 = 0x04   ge P2 P3 = 0x0c
; word 0 (i16, e32) = 0x08 | 0x17<<5 | 0x13<<10 | 0x1b<<15 | 0x04<<20 | 0x0c<<25
;                   = 0x8 + 0x2e0 + 0x4c00 + 0xd8000 + 0x400000 + 0x18000000 = 0x184dcee8
; word 1 (u16, e32) = 0x08 | 0x17<<5 | 0x14<<10 | 0x1c<<15 | 0x03<<20 | 0x0b<<25
;                   = 0x8 + 0x2e0 + 0x5000 + 0xe0000 + 0x300000 + 0x16000000 = 0x163e52e8
; words 2 and 3 are the e64 forms with the operands swapped: src0 is the VGPR B, src1 is the constant (or v5 for P4),
; so "op(B, A)" is the mirrored relation of op(A, B): lt <-> gt and le <-> ge, eq and ne are unchanged. The masks by
; position k = eq, ne, lt, le, gt, ge are therefore the masks of eq, ne, gt, ge, lt, le:
; word 2 (i16, e64) = 0x08 | 0x17<<5 | 0x04<<10 | 0x0c<<15 | 0x13<<20 | 0x1b<<25
;                   = 0x8 + 0x2e0 + 0x1000 + 0x60000 + 0x1300000 + 0x36000000 = 0x373612e8
; word 3 (u16, e64) = 0x08 | 0x17<<5 | 0x03<<10 | 0x0b<<15 | 0x14<<20 | 0x1c<<25
;                   = 0x8 + 0x2e0 + 0xc00 + 0x58000 + 0x1400000 + 0x38000000 = 0x39458ee8
; A translation that compared an _i16 form unsigned, or a _u16 form signed, moves the P0/P1/P2 bits; one that
; compared the whole 32-bit registers would see P3 as different and P4 as A greater than B.
; Expected mrt0 bits: 0x184dcee8, 0x163e52e8, 0x373612e8, 0x39458ee8
s_buffer_load_dwordx8 s[4:11], s[0:3], 0
s_waitcnt lgkmcnt(0)
v_mov_b32_e32 v1, s4
v_mov_b32_e32 v2, s5
v_mov_b32_e32 v3, s6
v_mov_b32_e32 v4, s7
v_mov_b32_e32 v5, s8
v_mov_b32_e32 v6, s9
; word 0: v_cmp_*_i16_e32, A is src0
v_mov_b32_e32 v40, 0
v_cmp_eq_i16_e32 vcc_lo, 0x8000, v1
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v40, v20, 0, v40
v_cmp_eq_i16_e32 vcc_lo, -1, v2
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v40, v20, 1, v40
v_cmp_eq_i16_e32 vcc_lo, 1, v3
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v40, v20, 2, v40
v_cmp_eq_i16_e32 vcc_lo, 0x1234, v4
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v40, v20, 3, v40
v_cmp_eq_i16_e32 vcc_lo, v5, v6
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v40, v20, 4, v40
v_cmp_ne_i16_e32 vcc_lo, 0x8000, v1
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v40, v20, 5, v40
v_cmp_ne_i16_e32 vcc_lo, -1, v2
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v40, v20, 6, v40
v_cmp_ne_i16_e32 vcc_lo, 1, v3
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v40, v20, 7, v40
v_cmp_ne_i16_e32 vcc_lo, 0x1234, v4
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v40, v20, 8, v40
v_cmp_ne_i16_e32 vcc_lo, v5, v6
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v40, v20, 9, v40
v_cmp_lt_i16_e32 vcc_lo, 0x8000, v1
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v40, v20, 10, v40
v_cmp_lt_i16_e32 vcc_lo, -1, v2
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v40, v20, 11, v40
v_cmp_lt_i16_e32 vcc_lo, 1, v3
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v40, v20, 12, v40
v_cmp_lt_i16_e32 vcc_lo, 0x1234, v4
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v40, v20, 13, v40
v_cmp_lt_i16_e32 vcc_lo, v5, v6
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v40, v20, 14, v40
v_cmp_le_i16_e32 vcc_lo, 0x8000, v1
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v40, v20, 15, v40
v_cmp_le_i16_e32 vcc_lo, -1, v2
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v40, v20, 16, v40
v_cmp_le_i16_e32 vcc_lo, 1, v3
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v40, v20, 17, v40
v_cmp_le_i16_e32 vcc_lo, 0x1234, v4
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v40, v20, 18, v40
v_cmp_le_i16_e32 vcc_lo, v5, v6
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v40, v20, 19, v40
v_cmp_gt_i16_e32 vcc_lo, 0x8000, v1
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v40, v20, 20, v40
v_cmp_gt_i16_e32 vcc_lo, -1, v2
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v40, v20, 21, v40
v_cmp_gt_i16_e32 vcc_lo, 1, v3
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v40, v20, 22, v40
v_cmp_gt_i16_e32 vcc_lo, 0x1234, v4
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v40, v20, 23, v40
v_cmp_gt_i16_e32 vcc_lo, v5, v6
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v40, v20, 24, v40
v_cmp_ge_i16_e32 vcc_lo, 0x8000, v1
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v40, v20, 25, v40
v_cmp_ge_i16_e32 vcc_lo, -1, v2
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v40, v20, 26, v40
v_cmp_ge_i16_e32 vcc_lo, 1, v3
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v40, v20, 27, v40
v_cmp_ge_i16_e32 vcc_lo, 0x1234, v4
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v40, v20, 28, v40
v_cmp_ge_i16_e32 vcc_lo, v5, v6
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v40, v20, 29, v40
; word 1: v_cmp_*_u16_e32, A is src0
v_mov_b32_e32 v41, 0
v_cmp_eq_u16_e32 vcc_lo, 0x8000, v1
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v41, v20, 0, v41
v_cmp_eq_u16_e32 vcc_lo, -1, v2
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v41, v20, 1, v41
v_cmp_eq_u16_e32 vcc_lo, 1, v3
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v41, v20, 2, v41
v_cmp_eq_u16_e32 vcc_lo, 0x1234, v4
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v41, v20, 3, v41
v_cmp_eq_u16_e32 vcc_lo, v5, v6
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v41, v20, 4, v41
v_cmp_ne_u16_e32 vcc_lo, 0x8000, v1
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v41, v20, 5, v41
v_cmp_ne_u16_e32 vcc_lo, -1, v2
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v41, v20, 6, v41
v_cmp_ne_u16_e32 vcc_lo, 1, v3
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v41, v20, 7, v41
v_cmp_ne_u16_e32 vcc_lo, 0x1234, v4
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v41, v20, 8, v41
v_cmp_ne_u16_e32 vcc_lo, v5, v6
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v41, v20, 9, v41
v_cmp_lt_u16_e32 vcc_lo, 0x8000, v1
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v41, v20, 10, v41
v_cmp_lt_u16_e32 vcc_lo, -1, v2
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v41, v20, 11, v41
v_cmp_lt_u16_e32 vcc_lo, 1, v3
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v41, v20, 12, v41
v_cmp_lt_u16_e32 vcc_lo, 0x1234, v4
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v41, v20, 13, v41
v_cmp_lt_u16_e32 vcc_lo, v5, v6
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v41, v20, 14, v41
v_cmp_le_u16_e32 vcc_lo, 0x8000, v1
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v41, v20, 15, v41
v_cmp_le_u16_e32 vcc_lo, -1, v2
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v41, v20, 16, v41
v_cmp_le_u16_e32 vcc_lo, 1, v3
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v41, v20, 17, v41
v_cmp_le_u16_e32 vcc_lo, 0x1234, v4
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v41, v20, 18, v41
v_cmp_le_u16_e32 vcc_lo, v5, v6
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v41, v20, 19, v41
v_cmp_gt_u16_e32 vcc_lo, 0x8000, v1
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v41, v20, 20, v41
v_cmp_gt_u16_e32 vcc_lo, -1, v2
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v41, v20, 21, v41
v_cmp_gt_u16_e32 vcc_lo, 1, v3
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v41, v20, 22, v41
v_cmp_gt_u16_e32 vcc_lo, 0x1234, v4
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v41, v20, 23, v41
v_cmp_gt_u16_e32 vcc_lo, v5, v6
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v41, v20, 24, v41
v_cmp_ge_u16_e32 vcc_lo, 0x8000, v1
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v41, v20, 25, v41
v_cmp_ge_u16_e32 vcc_lo, -1, v2
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v41, v20, 26, v41
v_cmp_ge_u16_e32 vcc_lo, 1, v3
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v41, v20, 27, v41
v_cmp_ge_u16_e32 vcc_lo, 0x1234, v4
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v41, v20, 28, v41
v_cmp_ge_u16_e32 vcc_lo, v5, v6
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v41, v20, 29, v41
; word 2: v_cmp_*_i16_e64, B is src0
v_mov_b32_e32 v42, 0
v_cmp_eq_i16_e64 s22, v1, 0x8000
v_cndmask_b32_e64 v20, 0, 1, s22
v_lshl_or_b32 v42, v20, 0, v42
v_cmp_eq_i16_e64 s22, v2, -1
v_cndmask_b32_e64 v20, 0, 1, s22
v_lshl_or_b32 v42, v20, 1, v42
v_cmp_eq_i16_e64 s22, v3, 1
v_cndmask_b32_e64 v20, 0, 1, s22
v_lshl_or_b32 v42, v20, 2, v42
v_cmp_eq_i16_e64 s22, v4, 0x1234
v_cndmask_b32_e64 v20, 0, 1, s22
v_lshl_or_b32 v42, v20, 3, v42
v_cmp_eq_i16_e64 s22, v6, v5
v_cndmask_b32_e64 v20, 0, 1, s22
v_lshl_or_b32 v42, v20, 4, v42
v_cmp_ne_i16_e64 s22, v1, 0x8000
v_cndmask_b32_e64 v20, 0, 1, s22
v_lshl_or_b32 v42, v20, 5, v42
v_cmp_ne_i16_e64 s22, v2, -1
v_cndmask_b32_e64 v20, 0, 1, s22
v_lshl_or_b32 v42, v20, 6, v42
v_cmp_ne_i16_e64 s22, v3, 1
v_cndmask_b32_e64 v20, 0, 1, s22
v_lshl_or_b32 v42, v20, 7, v42
v_cmp_ne_i16_e64 s22, v4, 0x1234
v_cndmask_b32_e64 v20, 0, 1, s22
v_lshl_or_b32 v42, v20, 8, v42
v_cmp_ne_i16_e64 s22, v6, v5
v_cndmask_b32_e64 v20, 0, 1, s22
v_lshl_or_b32 v42, v20, 9, v42
v_cmp_lt_i16_e64 s22, v1, 0x8000
v_cndmask_b32_e64 v20, 0, 1, s22
v_lshl_or_b32 v42, v20, 10, v42
v_cmp_lt_i16_e64 s22, v2, -1
v_cndmask_b32_e64 v20, 0, 1, s22
v_lshl_or_b32 v42, v20, 11, v42
v_cmp_lt_i16_e64 s22, v3, 1
v_cndmask_b32_e64 v20, 0, 1, s22
v_lshl_or_b32 v42, v20, 12, v42
v_cmp_lt_i16_e64 s22, v4, 0x1234
v_cndmask_b32_e64 v20, 0, 1, s22
v_lshl_or_b32 v42, v20, 13, v42
v_cmp_lt_i16_e64 s22, v6, v5
v_cndmask_b32_e64 v20, 0, 1, s22
v_lshl_or_b32 v42, v20, 14, v42
v_cmp_le_i16_e64 s22, v1, 0x8000
v_cndmask_b32_e64 v20, 0, 1, s22
v_lshl_or_b32 v42, v20, 15, v42
v_cmp_le_i16_e64 s22, v2, -1
v_cndmask_b32_e64 v20, 0, 1, s22
v_lshl_or_b32 v42, v20, 16, v42
v_cmp_le_i16_e64 s22, v3, 1
v_cndmask_b32_e64 v20, 0, 1, s22
v_lshl_or_b32 v42, v20, 17, v42
v_cmp_le_i16_e64 s22, v4, 0x1234
v_cndmask_b32_e64 v20, 0, 1, s22
v_lshl_or_b32 v42, v20, 18, v42
v_cmp_le_i16_e64 s22, v6, v5
v_cndmask_b32_e64 v20, 0, 1, s22
v_lshl_or_b32 v42, v20, 19, v42
v_cmp_gt_i16_e64 s22, v1, 0x8000
v_cndmask_b32_e64 v20, 0, 1, s22
v_lshl_or_b32 v42, v20, 20, v42
v_cmp_gt_i16_e64 s22, v2, -1
v_cndmask_b32_e64 v20, 0, 1, s22
v_lshl_or_b32 v42, v20, 21, v42
v_cmp_gt_i16_e64 s22, v3, 1
v_cndmask_b32_e64 v20, 0, 1, s22
v_lshl_or_b32 v42, v20, 22, v42
v_cmp_gt_i16_e64 s22, v4, 0x1234
v_cndmask_b32_e64 v20, 0, 1, s22
v_lshl_or_b32 v42, v20, 23, v42
v_cmp_gt_i16_e64 s22, v6, v5
v_cndmask_b32_e64 v20, 0, 1, s22
v_lshl_or_b32 v42, v20, 24, v42
v_cmp_ge_i16_e64 s22, v1, 0x8000
v_cndmask_b32_e64 v20, 0, 1, s22
v_lshl_or_b32 v42, v20, 25, v42
v_cmp_ge_i16_e64 s22, v2, -1
v_cndmask_b32_e64 v20, 0, 1, s22
v_lshl_or_b32 v42, v20, 26, v42
v_cmp_ge_i16_e64 s22, v3, 1
v_cndmask_b32_e64 v20, 0, 1, s22
v_lshl_or_b32 v42, v20, 27, v42
v_cmp_ge_i16_e64 s22, v4, 0x1234
v_cndmask_b32_e64 v20, 0, 1, s22
v_lshl_or_b32 v42, v20, 28, v42
v_cmp_ge_i16_e64 s22, v6, v5
v_cndmask_b32_e64 v20, 0, 1, s22
v_lshl_or_b32 v42, v20, 29, v42
; word 3: v_cmp_*_u16_e64, B is src0
v_mov_b32_e32 v43, 0
v_cmp_eq_u16_e64 s22, v1, 0x8000
v_cndmask_b32_e64 v20, 0, 1, s22
v_lshl_or_b32 v43, v20, 0, v43
v_cmp_eq_u16_e64 s22, v2, -1
v_cndmask_b32_e64 v20, 0, 1, s22
v_lshl_or_b32 v43, v20, 1, v43
v_cmp_eq_u16_e64 s22, v3, 1
v_cndmask_b32_e64 v20, 0, 1, s22
v_lshl_or_b32 v43, v20, 2, v43
v_cmp_eq_u16_e64 s22, v4, 0x1234
v_cndmask_b32_e64 v20, 0, 1, s22
v_lshl_or_b32 v43, v20, 3, v43
v_cmp_eq_u16_e64 s22, v6, v5
v_cndmask_b32_e64 v20, 0, 1, s22
v_lshl_or_b32 v43, v20, 4, v43
v_cmp_ne_u16_e64 s22, v1, 0x8000
v_cndmask_b32_e64 v20, 0, 1, s22
v_lshl_or_b32 v43, v20, 5, v43
v_cmp_ne_u16_e64 s22, v2, -1
v_cndmask_b32_e64 v20, 0, 1, s22
v_lshl_or_b32 v43, v20, 6, v43
v_cmp_ne_u16_e64 s22, v3, 1
v_cndmask_b32_e64 v20, 0, 1, s22
v_lshl_or_b32 v43, v20, 7, v43
v_cmp_ne_u16_e64 s22, v4, 0x1234
v_cndmask_b32_e64 v20, 0, 1, s22
v_lshl_or_b32 v43, v20, 8, v43
v_cmp_ne_u16_e64 s22, v6, v5
v_cndmask_b32_e64 v20, 0, 1, s22
v_lshl_or_b32 v43, v20, 9, v43
v_cmp_lt_u16_e64 s22, v1, 0x8000
v_cndmask_b32_e64 v20, 0, 1, s22
v_lshl_or_b32 v43, v20, 10, v43
v_cmp_lt_u16_e64 s22, v2, -1
v_cndmask_b32_e64 v20, 0, 1, s22
v_lshl_or_b32 v43, v20, 11, v43
v_cmp_lt_u16_e64 s22, v3, 1
v_cndmask_b32_e64 v20, 0, 1, s22
v_lshl_or_b32 v43, v20, 12, v43
v_cmp_lt_u16_e64 s22, v4, 0x1234
v_cndmask_b32_e64 v20, 0, 1, s22
v_lshl_or_b32 v43, v20, 13, v43
v_cmp_lt_u16_e64 s22, v6, v5
v_cndmask_b32_e64 v20, 0, 1, s22
v_lshl_or_b32 v43, v20, 14, v43
v_cmp_le_u16_e64 s22, v1, 0x8000
v_cndmask_b32_e64 v20, 0, 1, s22
v_lshl_or_b32 v43, v20, 15, v43
v_cmp_le_u16_e64 s22, v2, -1
v_cndmask_b32_e64 v20, 0, 1, s22
v_lshl_or_b32 v43, v20, 16, v43
v_cmp_le_u16_e64 s22, v3, 1
v_cndmask_b32_e64 v20, 0, 1, s22
v_lshl_or_b32 v43, v20, 17, v43
v_cmp_le_u16_e64 s22, v4, 0x1234
v_cndmask_b32_e64 v20, 0, 1, s22
v_lshl_or_b32 v43, v20, 18, v43
v_cmp_le_u16_e64 s22, v6, v5
v_cndmask_b32_e64 v20, 0, 1, s22
v_lshl_or_b32 v43, v20, 19, v43
v_cmp_gt_u16_e64 s22, v1, 0x8000
v_cndmask_b32_e64 v20, 0, 1, s22
v_lshl_or_b32 v43, v20, 20, v43
v_cmp_gt_u16_e64 s22, v2, -1
v_cndmask_b32_e64 v20, 0, 1, s22
v_lshl_or_b32 v43, v20, 21, v43
v_cmp_gt_u16_e64 s22, v3, 1
v_cndmask_b32_e64 v20, 0, 1, s22
v_lshl_or_b32 v43, v20, 22, v43
v_cmp_gt_u16_e64 s22, v4, 0x1234
v_cndmask_b32_e64 v20, 0, 1, s22
v_lshl_or_b32 v43, v20, 23, v43
v_cmp_gt_u16_e64 s22, v6, v5
v_cndmask_b32_e64 v20, 0, 1, s22
v_lshl_or_b32 v43, v20, 24, v43
v_cmp_ge_u16_e64 s22, v1, 0x8000
v_cndmask_b32_e64 v20, 0, 1, s22
v_lshl_or_b32 v43, v20, 25, v43
v_cmp_ge_u16_e64 s22, v2, -1
v_cndmask_b32_e64 v20, 0, 1, s22
v_lshl_or_b32 v43, v20, 26, v43
v_cmp_ge_u16_e64 s22, v3, 1
v_cndmask_b32_e64 v20, 0, 1, s22
v_lshl_or_b32 v43, v20, 27, v43
v_cmp_ge_u16_e64 s22, v4, 0x1234
v_cndmask_b32_e64 v20, 0, 1, s22
v_lshl_or_b32 v43, v20, 28, v43
v_cmp_ge_u16_e64 s22, v6, v5
v_cndmask_b32_e64 v20, 0, 1, s22
v_lshl_or_b32 v43, v20, 29, v43
exp mrt0 v40, v41, v42, v43 done vm
s_endpgm
