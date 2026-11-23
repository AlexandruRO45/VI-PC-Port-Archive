; Null sources in 64-bit integer VALU compares on a GPU (docs/design.md, "Null sources in 64-bit integer VALU
; compares"; CPU-checked by tests/semantic/test_vcmp64_null_src.py). RDNA2 ISA: the special operand null reads
; zero, so v_cmp(x)_{eq,ne,lt,le,gt,ge}_{u64,i64} with null in either source compare against the 64-bit 0.
; v0 = lane 0..31. Lane L holds value k = L & 3 in v[2:3]:
;   k=0: 0x00000000_00000000   k=1: 0x00000000_00000001   k=2: 0x00000001_00000000   k=3: 0x80000000_00000000
; Every lane runs the 64 compares below and reads each result for itself: v_cndmask on the mask, or for v_cmpx
; a write under the EXEC it leaves. No lane mask is read as data. Bit i of v10 = compare i, bit i of v11 =
; compare 32 + i. Lanes 0..3 store v10 to out[L], lanes 4..7 store v11 to out[L], so out[k] bit i is compare
; i on value k and out[4 + k] bit i is compare 32 + i on value k. e32 takes null only in src0.
; Compare (bit), then its result on values k=0..3:
;    0 v_cmp_eq_u64_e32 null, v[2:3]                              1000
;    1 v_cmp_ne_u64_e32 null, v[2:3]                              0111
;    2 v_cmp_lt_u64_e32 null, v[2:3]                              0111
;    3 v_cmp_le_u64_e32 null, v[2:3]                              1111
;    4 v_cmp_gt_u64_e32 null, v[2:3]                              0000
;    5 v_cmp_ge_u64_e32 null, v[2:3]                              1000
;    6 v_cmp_eq_i64_e32 null, v[2:3]                              1000
;    7 v_cmp_ne_i64_e32 null, v[2:3]                              0111
;    8 v_cmp_lt_i64_e32 null, v[2:3]                              0110
;    9 v_cmp_le_i64_e32 null, v[2:3]                              1110
;   10 v_cmp_gt_i64_e32 null, v[2:3]                              0001
;   11 v_cmp_ge_i64_e32 null, v[2:3]                              1001
;   12 v_cmp_eq_u64_e64 null, v[2:3]                              1000
;   13 v_cmp_ne_u64_e64 null, v[2:3]                              0111
;   14 v_cmp_lt_u64_e64 null, v[2:3]                              0111
;   15 v_cmp_le_u64_e64 null, v[2:3]                              1111
;   16 v_cmp_gt_u64_e64 null, v[2:3]                              0000
;   17 v_cmp_ge_u64_e64 null, v[2:3]                              1000
;   18 v_cmp_eq_i64_e64 null, v[2:3]                              1000
;   19 v_cmp_ne_i64_e64 null, v[2:3]                              0111
;   20 v_cmp_lt_i64_e64 null, v[2:3]                              0110
;   21 v_cmp_le_i64_e64 null, v[2:3]                              1110
;   22 v_cmp_gt_i64_e64 null, v[2:3]                              0001
;   23 v_cmp_ge_i64_e64 null, v[2:3]                              1001
;   24 v_cmp_eq_u64_e64 v[2:3], null                              1000
;   25 v_cmp_ne_u64_e64 v[2:3], null                              0111
;   26 v_cmp_lt_u64_e64 v[2:3], null                              0000
;   27 v_cmp_le_u64_e64 v[2:3], null                              1000
;   28 v_cmp_gt_u64_e64 v[2:3], null                              0111
;   29 v_cmp_ge_u64_e64 v[2:3], null                              1111
;   30 v_cmp_eq_i64_e64 v[2:3], null                              1000
;   31 v_cmp_ne_i64_e64 v[2:3], null                              0111
;   32 v_cmp_lt_i64_e64 v[2:3], null                              0001
;   33 v_cmp_le_i64_e64 v[2:3], null                              1001
;   34 v_cmp_gt_i64_e64 v[2:3], null                              0110
;   35 v_cmp_ge_i64_e64 v[2:3], null                              1110
;   36 v_cmpx_eq_u64_e32 null, v[2:3]                             1000
;   37 v_cmpx_ne_u64_e32 null, v[2:3]                             0111
;   38 v_cmpx_lt_u64_e32 null, v[2:3]                             0111
;   39 v_cmpx_le_u64_e32 null, v[2:3]                             1111
;   40 v_cmpx_gt_u64_e32 null, v[2:3]                             0000
;   41 v_cmpx_ge_u64_e32 null, v[2:3]                             1000
;   42 v_cmpx_eq_i64_e32 null, v[2:3]                             1000
;   43 v_cmpx_ne_i64_e32 null, v[2:3]                             0111
;   44 v_cmpx_lt_i64_e32 null, v[2:3]                             0110
;   45 v_cmpx_le_i64_e32 null, v[2:3]                             1110
;   46 v_cmpx_gt_i64_e32 null, v[2:3]                             0001
;   47 v_cmpx_ge_i64_e32 null, v[2:3]                             1001
;   48 v_cmpx_eq_u64_e64 v[2:3], null                             1000
;   49 v_cmpx_ne_u64_e64 v[2:3], null                             0111
;   50 v_cmpx_lt_u64_e64 v[2:3], null                             0000
;   51 v_cmpx_le_u64_e64 v[2:3], null                             1000
;   52 v_cmpx_gt_u64_e64 v[2:3], null                             0111
;   53 v_cmpx_ge_u64_e64 v[2:3], null                             1111
;   54 v_cmpx_eq_i64_e64 v[2:3], null                             1000
;   55 v_cmpx_ne_i64_e64 v[2:3], null                             0111
;   56 v_cmpx_lt_i64_e64 v[2:3], null                             0001
;   57 v_cmpx_le_i64_e64 v[2:3], null                             1001
;   58 v_cmpx_gt_i64_e64 v[2:3], null                             0110
;   59 v_cmpx_ge_i64_e64 v[2:3], null                             1110
;   60 v_cmp_eq_u64_e64 null, null                                1111
;   61 v_cmp_lt_i64_e64 null, null                                0000
;   62 v_cmp_ge_u64_e64 null, null                                1111
;   63 v_cmp_eq_u64_e64 null, v[2:3]  (EXEC = lanes 0..3)         0000
; The last compare runs with EXEC = lanes 0..3 (s10 bits of lanes 4..7 must be 0, so out[4..7] bit 31 is 0).
; Expected: out[0..7] = 1772526185 2990072718 2990072718 2999512206 1516873370 1555183852 1555183852 1404225763
;   (hex 0x69a69a69 0xb238e38e 0xb238e38e 0xb2c8ec8e 0x5a69a69a 0x5cb238ec 0x5cb238ec 0x53b2c8e3)
; null read as 1 instead of 0 gives 0x8ecb2cb2 0x69a69a69 0xb238e38e 0xb2c8ec8e 0x538ecb23 0x5a69a69a 0x5cb238ec 0x53b2c8e3.
; group: 32,1,1
; Run: VKC_SUBGROUP_SIZE=32 vk_compute_test cs_w32_vcmp64_null.spv 1 dump -> out[0..7] = 1772526185 2990072718 2990072718 2999512206 1516873370 1555183852 1555183852 1404225763
; mattr: +wavefrontsize32,-wavefrontsize64
s_mov_b32 exec_lo, -1
v_and_b32_e32 v4, 3, v0
v_mov_b32_e32 v2, 0x0
v_mov_b32_e32 v3, 0x0
v_mov_b32_e32 v6, 0x1
v_mov_b32_e32 v7, 0x0
v_cmp_eq_u32_e32 vcc_lo, 1, v4
v_cndmask_b32_e32 v2, v2, v6, vcc_lo
v_cndmask_b32_e32 v3, v3, v7, vcc_lo
v_mov_b32_e32 v6, 0x0
v_mov_b32_e32 v7, 0x1
v_cmp_eq_u32_e32 vcc_lo, 2, v4
v_cndmask_b32_e32 v2, v2, v6, vcc_lo
v_cndmask_b32_e32 v3, v3, v7, vcc_lo
v_mov_b32_e32 v6, 0x0
v_mov_b32_e32 v7, 0x80000000
v_cmp_eq_u32_e32 vcc_lo, 3, v4
v_cndmask_b32_e32 v2, v2, v6, vcc_lo
v_cndmask_b32_e32 v3, v3, v7, vcc_lo
v_mov_b32_e32 v10, 0
v_mov_b32_e32 v11, 0
v_cmp_eq_u64_e32 vcc_lo, null, v[2:3]
v_cndmask_b32_e64 v8, 0, 0x1, vcc_lo
v_or_b32_e32 v10, v10, v8
v_cmp_ne_u64_e32 vcc_lo, null, v[2:3]
v_cndmask_b32_e64 v8, 0, 0x2, vcc_lo
v_or_b32_e32 v10, v10, v8
v_cmp_lt_u64_e32 vcc_lo, null, v[2:3]
v_cndmask_b32_e64 v8, 0, 0x4, vcc_lo
v_or_b32_e32 v10, v10, v8
v_cmp_le_u64_e32 vcc_lo, null, v[2:3]
v_cndmask_b32_e64 v8, 0, 0x8, vcc_lo
v_or_b32_e32 v10, v10, v8
v_cmp_gt_u64_e32 vcc_lo, null, v[2:3]
v_cndmask_b32_e64 v8, 0, 0x10, vcc_lo
v_or_b32_e32 v10, v10, v8
v_cmp_ge_u64_e32 vcc_lo, null, v[2:3]
v_cndmask_b32_e64 v8, 0, 0x20, vcc_lo
v_or_b32_e32 v10, v10, v8
v_cmp_eq_i64_e32 vcc_lo, null, v[2:3]
v_cndmask_b32_e64 v8, 0, 0x40, vcc_lo
v_or_b32_e32 v10, v10, v8
v_cmp_ne_i64_e32 vcc_lo, null, v[2:3]
v_cndmask_b32_e64 v8, 0, 0x80, vcc_lo
v_or_b32_e32 v10, v10, v8
v_cmp_lt_i64_e32 vcc_lo, null, v[2:3]
v_cndmask_b32_e64 v8, 0, 0x100, vcc_lo
v_or_b32_e32 v10, v10, v8
v_cmp_le_i64_e32 vcc_lo, null, v[2:3]
v_cndmask_b32_e64 v8, 0, 0x200, vcc_lo
v_or_b32_e32 v10, v10, v8
v_cmp_gt_i64_e32 vcc_lo, null, v[2:3]
v_cndmask_b32_e64 v8, 0, 0x400, vcc_lo
v_or_b32_e32 v10, v10, v8
v_cmp_ge_i64_e32 vcc_lo, null, v[2:3]
v_cndmask_b32_e64 v8, 0, 0x800, vcc_lo
v_or_b32_e32 v10, v10, v8
v_cmp_eq_u64_e64 s10, null, v[2:3]
v_cndmask_b32_e64 v8, 0, 0x1000, s10
v_or_b32_e32 v10, v10, v8
v_cmp_ne_u64_e64 s10, null, v[2:3]
v_cndmask_b32_e64 v8, 0, 0x2000, s10
v_or_b32_e32 v10, v10, v8
v_cmp_lt_u64_e64 s10, null, v[2:3]
v_cndmask_b32_e64 v8, 0, 0x4000, s10
v_or_b32_e32 v10, v10, v8
v_cmp_le_u64_e64 s10, null, v[2:3]
v_cndmask_b32_e64 v8, 0, 0x8000, s10
v_or_b32_e32 v10, v10, v8
v_cmp_gt_u64_e64 s10, null, v[2:3]
v_cndmask_b32_e64 v8, 0, 0x10000, s10
v_or_b32_e32 v10, v10, v8
v_cmp_ge_u64_e64 s10, null, v[2:3]
v_cndmask_b32_e64 v8, 0, 0x20000, s10
v_or_b32_e32 v10, v10, v8
v_cmp_eq_i64_e64 s10, null, v[2:3]
v_cndmask_b32_e64 v8, 0, 0x40000, s10
v_or_b32_e32 v10, v10, v8
v_cmp_ne_i64_e64 s10, null, v[2:3]
v_cndmask_b32_e64 v8, 0, 0x80000, s10
v_or_b32_e32 v10, v10, v8
v_cmp_lt_i64_e64 s10, null, v[2:3]
v_cndmask_b32_e64 v8, 0, 0x100000, s10
v_or_b32_e32 v10, v10, v8
v_cmp_le_i64_e64 s10, null, v[2:3]
v_cndmask_b32_e64 v8, 0, 0x200000, s10
v_or_b32_e32 v10, v10, v8
v_cmp_gt_i64_e64 s10, null, v[2:3]
v_cndmask_b32_e64 v8, 0, 0x400000, s10
v_or_b32_e32 v10, v10, v8
v_cmp_ge_i64_e64 s10, null, v[2:3]
v_cndmask_b32_e64 v8, 0, 0x800000, s10
v_or_b32_e32 v10, v10, v8
v_cmp_eq_u64_e64 s10, v[2:3], null
v_cndmask_b32_e64 v8, 0, 0x1000000, s10
v_or_b32_e32 v10, v10, v8
v_cmp_ne_u64_e64 s10, v[2:3], null
v_cndmask_b32_e64 v8, 0, 0x2000000, s10
v_or_b32_e32 v10, v10, v8
v_cmp_lt_u64_e64 s10, v[2:3], null
v_cndmask_b32_e64 v8, 0, 0x4000000, s10
v_or_b32_e32 v10, v10, v8
v_cmp_le_u64_e64 s10, v[2:3], null
v_cndmask_b32_e64 v8, 0, 0x8000000, s10
v_or_b32_e32 v10, v10, v8
v_cmp_gt_u64_e64 s10, v[2:3], null
v_cndmask_b32_e64 v8, 0, 0x10000000, s10
v_or_b32_e32 v10, v10, v8
v_cmp_ge_u64_e64 s10, v[2:3], null
v_cndmask_b32_e64 v8, 0, 0x20000000, s10
v_or_b32_e32 v10, v10, v8
v_cmp_eq_i64_e64 s10, v[2:3], null
v_cndmask_b32_e64 v8, 0, 0x40000000, s10
v_or_b32_e32 v10, v10, v8
v_cmp_ne_i64_e64 s10, v[2:3], null
v_cndmask_b32_e64 v8, 0, 0x80000000, s10
v_or_b32_e32 v10, v10, v8
v_cmp_lt_i64_e64 s10, v[2:3], null
v_cndmask_b32_e64 v8, 0, 0x1, s10
v_or_b32_e32 v11, v11, v8
v_cmp_le_i64_e64 s10, v[2:3], null
v_cndmask_b32_e64 v8, 0, 0x2, s10
v_or_b32_e32 v11, v11, v8
v_cmp_gt_i64_e64 s10, v[2:3], null
v_cndmask_b32_e64 v8, 0, 0x4, s10
v_or_b32_e32 v11, v11, v8
v_cmp_ge_i64_e64 s10, v[2:3], null
v_cndmask_b32_e64 v8, 0, 0x8, s10
v_or_b32_e32 v11, v11, v8
v_mov_b32_e32 v9, 0
v_cmpx_eq_u64_e32 null, v[2:3]
v_mov_b32_e32 v9, 0x10
s_mov_b32 exec_lo, -1
v_or_b32_e32 v11, v11, v9
v_mov_b32_e32 v9, 0
v_cmpx_ne_u64_e32 null, v[2:3]
v_mov_b32_e32 v9, 0x20
s_mov_b32 exec_lo, -1
v_or_b32_e32 v11, v11, v9
v_mov_b32_e32 v9, 0
v_cmpx_lt_u64_e32 null, v[2:3]
v_mov_b32_e32 v9, 0x40
s_mov_b32 exec_lo, -1
v_or_b32_e32 v11, v11, v9
v_mov_b32_e32 v9, 0
v_cmpx_le_u64_e32 null, v[2:3]
v_mov_b32_e32 v9, 0x80
s_mov_b32 exec_lo, -1
v_or_b32_e32 v11, v11, v9
v_mov_b32_e32 v9, 0
v_cmpx_gt_u64_e32 null, v[2:3]
v_mov_b32_e32 v9, 0x100
s_mov_b32 exec_lo, -1
v_or_b32_e32 v11, v11, v9
v_mov_b32_e32 v9, 0
v_cmpx_ge_u64_e32 null, v[2:3]
v_mov_b32_e32 v9, 0x200
s_mov_b32 exec_lo, -1
v_or_b32_e32 v11, v11, v9
v_mov_b32_e32 v9, 0
v_cmpx_eq_i64_e32 null, v[2:3]
v_mov_b32_e32 v9, 0x400
s_mov_b32 exec_lo, -1
v_or_b32_e32 v11, v11, v9
v_mov_b32_e32 v9, 0
v_cmpx_ne_i64_e32 null, v[2:3]
v_mov_b32_e32 v9, 0x800
s_mov_b32 exec_lo, -1
v_or_b32_e32 v11, v11, v9
v_mov_b32_e32 v9, 0
v_cmpx_lt_i64_e32 null, v[2:3]
v_mov_b32_e32 v9, 0x1000
s_mov_b32 exec_lo, -1
v_or_b32_e32 v11, v11, v9
v_mov_b32_e32 v9, 0
v_cmpx_le_i64_e32 null, v[2:3]
v_mov_b32_e32 v9, 0x2000
s_mov_b32 exec_lo, -1
v_or_b32_e32 v11, v11, v9
v_mov_b32_e32 v9, 0
v_cmpx_gt_i64_e32 null, v[2:3]
v_mov_b32_e32 v9, 0x4000
s_mov_b32 exec_lo, -1
v_or_b32_e32 v11, v11, v9
v_mov_b32_e32 v9, 0
v_cmpx_ge_i64_e32 null, v[2:3]
v_mov_b32_e32 v9, 0x8000
s_mov_b32 exec_lo, -1
v_or_b32_e32 v11, v11, v9
v_mov_b32_e32 v9, 0
v_cmpx_eq_u64_e64 v[2:3], null
v_mov_b32_e32 v9, 0x10000
s_mov_b32 exec_lo, -1
v_or_b32_e32 v11, v11, v9
v_mov_b32_e32 v9, 0
v_cmpx_ne_u64_e64 v[2:3], null
v_mov_b32_e32 v9, 0x20000
s_mov_b32 exec_lo, -1
v_or_b32_e32 v11, v11, v9
v_mov_b32_e32 v9, 0
v_cmpx_lt_u64_e64 v[2:3], null
v_mov_b32_e32 v9, 0x40000
s_mov_b32 exec_lo, -1
v_or_b32_e32 v11, v11, v9
v_mov_b32_e32 v9, 0
v_cmpx_le_u64_e64 v[2:3], null
v_mov_b32_e32 v9, 0x80000
s_mov_b32 exec_lo, -1
v_or_b32_e32 v11, v11, v9
v_mov_b32_e32 v9, 0
v_cmpx_gt_u64_e64 v[2:3], null
v_mov_b32_e32 v9, 0x100000
s_mov_b32 exec_lo, -1
v_or_b32_e32 v11, v11, v9
v_mov_b32_e32 v9, 0
v_cmpx_ge_u64_e64 v[2:3], null
v_mov_b32_e32 v9, 0x200000
s_mov_b32 exec_lo, -1
v_or_b32_e32 v11, v11, v9
v_mov_b32_e32 v9, 0
v_cmpx_eq_i64_e64 v[2:3], null
v_mov_b32_e32 v9, 0x400000
s_mov_b32 exec_lo, -1
v_or_b32_e32 v11, v11, v9
v_mov_b32_e32 v9, 0
v_cmpx_ne_i64_e64 v[2:3], null
v_mov_b32_e32 v9, 0x800000
s_mov_b32 exec_lo, -1
v_or_b32_e32 v11, v11, v9
v_mov_b32_e32 v9, 0
v_cmpx_lt_i64_e64 v[2:3], null
v_mov_b32_e32 v9, 0x1000000
s_mov_b32 exec_lo, -1
v_or_b32_e32 v11, v11, v9
v_mov_b32_e32 v9, 0
v_cmpx_le_i64_e64 v[2:3], null
v_mov_b32_e32 v9, 0x2000000
s_mov_b32 exec_lo, -1
v_or_b32_e32 v11, v11, v9
v_mov_b32_e32 v9, 0
v_cmpx_gt_i64_e64 v[2:3], null
v_mov_b32_e32 v9, 0x4000000
s_mov_b32 exec_lo, -1
v_or_b32_e32 v11, v11, v9
v_mov_b32_e32 v9, 0
v_cmpx_ge_i64_e64 v[2:3], null
v_mov_b32_e32 v9, 0x8000000
s_mov_b32 exec_lo, -1
v_or_b32_e32 v11, v11, v9
v_cmp_eq_u64_e64 s10, null, null
v_cndmask_b32_e64 v8, 0, 0x10000000, s10
v_or_b32_e32 v11, v11, v8
v_cmp_lt_i64_e64 s10, null, null
v_cndmask_b32_e64 v8, 0, 0x20000000, s10
v_or_b32_e32 v11, v11, v8
v_cmp_ge_u64_e64 s10, null, null
v_cndmask_b32_e64 v8, 0, 0x40000000, s10
v_or_b32_e32 v11, v11, v8
s_mov_b32 exec_lo, 0xf
v_cmp_eq_u64_e64 s10, null, v[2:3]
s_mov_b32 exec_lo, -1
v_cndmask_b32_e64 v8, 0, 0x80000000, s10
v_or_b32_e32 v11, v11, v8
v_cmp_gt_u32_e32 vcc_lo, 4, v0
v_cndmask_b32_e32 v12, v11, v10, vcc_lo
v_lshlrev_b32_e32 v13, 2, v0
v_cmpx_gt_u32_e32 8, v0
buffer_store_dword v12, v13, s[0:3], 0 offen
s_endpgm
