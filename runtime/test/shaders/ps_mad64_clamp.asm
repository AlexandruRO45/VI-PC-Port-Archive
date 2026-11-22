; The VOP3 clamp bit on v_mad_u64_u32 / v_mad_i64_i32 (VOP3B, wave32 with a one-SGPR carry-out). RDNA2 saturates
; the 64-bit D when the exact 65-bit sum S0 * S1 + S2 does not fit: to 0xffffffff_ffffffff (u64), to INT64_MIN or
; INT64_MAX by the sum's sign (i64). sdst still gets bit 64 of the exact sum (the carry for u64, the sign for i64),
; as without clamp. Measured on an AMD Radeon iGPU (gfx1036, RDNA2, Windows driver 26.10.1) with raw kernels through
; tools/hw_oracle, wave32 and wave64 alike (EGEN-HW-MAD64-CLAMP). Until then the translator refused clamp here.
;   A: v_mad_u64_u32 v[4:5], s10, v0, v0, v[2:3] clamp, v0 = 0xffffffff, v[2:3] = 0xffffffff_ffffffff:
;      0xffffffff * 0xffffffff = 0xfffffffe_00000001; + 2^64 - 1 = 0x1_fffffffe_00000000, past 64 bits:
;      D = 0xffffffff_ffffffff (wrapped: 0xfffffffe_00000000), s10 lane bit = 1
;   B: v_mad_i64_i32 v[6:7], s11, v8, v9, v[12:13] clamp, v8 = 0x80000000 (-2^31), v9 = 0x7fffffff (2^31 - 1),
;      v[12:13] = 0x80000000_00000000 (-2^63): -2^62 + 2^31 - 2^63 is below INT64_MIN:
;      D = 0x80000000_00000000 (wrapped: 0x40000000_80000000), s11 lane bit = 1 (negative)
;   C: v_mad_i64_i32 v[10:11], s12, v8, v8, v[14:15] clamp, v[14:15] = 0x7fffffff_ffffffff (INT64_MAX):
;      2^62 + 2^63 - 1 is above INT64_MAX: D = 0x7fffffff_ffffffff (wrapped: 0xbfffffff_ffffffff), s12 bit = 0
; Outputs: x = A.hi = 0xffffffff, y = B.hi = 0x80000000, z = C.hi = 0x7fffffff,
;   w = ((A.lo ^ B.lo ^ C.lo) & 0xfffffff0) | s10 bit | s11 bit << 1 | s12 bit << 2
;     = ((0xffffffff ^ 0 ^ 0xffffffff) & 0xfffffff0) | 1 | 2 | 0 = 0x00000003
; Wrapping instead (clamp ignored): 0xfffffffe, 0x40000000, 0xbfffffff, (0 ^ 0x80000000 ^ 0xffffffff) & 0xfffffff0
;   | 3 = 0x7ffffff3.
; Expected mrt0 bits: 0xffffffff, 0x80000000, 0x7fffffff, 0x00000003
v_mov_b32 v0, 0xffffffff
v_mov_b32 v2, 0xffffffff
v_mov_b32 v3, 0xffffffff
v_mov_b32 v8, 0x80000000
v_mov_b32 v9, 0x7fffffff
v_mov_b32 v12, 0
v_mov_b32 v13, 0x80000000
v_mov_b32 v14, 0xffffffff
v_mov_b32 v15, 0x7fffffff
v_mad_u64_u32 v[4:5], s10, v0, v0, v[2:3] clamp
v_mad_i64_i32 v[6:7], s11, v8, v9, v[12:13] clamp
v_mad_i64_i32 v[10:11], s12, v8, v8, v[14:15] clamp
v_xor3_b32 v1, v4, v6, v10
v_and_b32 v1, 0xfffffff0, v1
v_cndmask_b32_e64 v2, 0, 1, s10
v_cndmask_b32_e64 v3, 0, 2, s11
v_cndmask_b32_e64 v12, 0, 4, s12
v_or3_b32 v1, v1, v2, v3
v_or_b32 v1, v1, v12
exp mrt0 v5, v7, v11, v1 done vm
s_endpgm
