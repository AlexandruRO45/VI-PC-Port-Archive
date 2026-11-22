; v_dot4_i32_i8, v_dot4c_i32_i8 and the carry forms v_add/sub/subrev_co_u32, v_add/sub_co_ci_u32 (with and without
; clamp), run on a GPU; until now they had CPU tests only (tests/semantic/test_alu_dot4_i8.py, test_alu_add_co_clamp.py,
; test_carry_chain.py). One wave32 group; v0 = lane L (0..31). The output heap is 456 dwords (VKC_OUT_WORDS=456).
;
; RDNA2 ISA pseudocode used:
;   v_dot4_i32_i8  D = S0.i8[0]*S1.i8[0] + S0.i8[1]*S1.i8[1] + S0.i8[2]*S1.i8[2] + S0.i8[3]*S1.i8[3] + S2.i32
;                  (signed bytes, the sum wraps modulo 2^32; no saturation without clamp)
;   v_dot4c_i32_i8 D = (same dot product of S0, S1) + D (the old destination is the addend)
;   v_add_co_u32     tmp = S0 + S1;            D = tmp[31:0];  carry-out lane mask = tmp >= 2^32
;   v_add_co_ci_u32  tmp = S0 + S1 + CIN[lane]; D = tmp[31:0];  carry-out lane mask = tmp >= 2^32
;   v_sub_co_u32     D = S0 - S1;               borrow lane mask = S1 > S0
;   v_subrev_co_u32  D = S1 - S0;               borrow lane mask = S0 > S1
;   v_sub_co_ci_u32  D = S0 - S1 - CIN[lane];   borrow lane mask = S1 + CIN[lane] > S0
;   clamp (VOP3) saturates D to 0xFFFFFFFF on an add carry and to 0 on a subtract borrow; the carry/borrow mask is the
;   real one (docs/design.md, "Integer clamp").
;   Wave32: a carry lane mask is one SGPR (bit L = lane L); VCC is vcc_lo. gfx10 has no VOP2 form of the *_co_u32 ops,
;   only v_add/sub/subrev_co_ci_u32; the plain ones are VOP3 and name vcc_lo as the carry-out below (K3, K5).
;
; Dot product words (all lanes store the same dword), signed bytes, byte 0 first:
;   out[0] = dot4(0x80808080, 0x80808080, 0)          4 * (-128 * -128) = 65536                       = 0x00010000
;   out[1] = dot4(0x7f7f7f7f, 0x80808080, 0)          4 * (127 * -128) = -65024 = -0xFE00             = 0xFFFF0200
;   out[2] = dot4(0xffffffff, 0x7f7f7f7f, 0x100)      4 * (-1 * 127) + 256 = -252                     = 0xFFFFFF04
;   out[3] = dot4(0x7f7f7f7f, 0x7f7f7f7f, 0x7fffffff) 4 * 16129 = 64516 = 0xFC04; 0x7fffffff + 0xFC04 wraps = 0x8000FC03
;   out[4] = dot4(0x7f80ff01, 0x80ff7f02, 0)          bytes (1,-1,-128,127) . (2,127,-1,-128)
;                                                       = 2 - 127 + 128 - 16256 = -16253              = 0xFFFFC083
;   out[5] = v_dot4c(acc 0x7fffffff, 0x7f7f7f7f, 0x7f7f7f7f)   0x7fffffff + 0xFC04                    = 0x8000FC03
;   out[6] = a second v_dot4c on that register with (0x80808080, 0x7f7f7f7f): -65024, back across 2^31 = 0x7FFFFE03
;   out[7] = dot4(0x80808080 from an SGPR, 0xffffffff, -16 from an SGPR)   4 * 128 - 16 = 496         = 0x000001F0
; Read as unsigned bytes out[1], out[2], out[4] would be 0x0000FE00, 0x1F000 + 0x100 - ..., different (a signed/unsigned
; mix-up fails them); out[0] is the same either way and so is a control.
;
; out[8+L] = v_dot4c_i32_i8 per lane: S0 = L * 0x01010101 (every byte L, L <= 31), S1 = 0x80808080, D starts 0x100:
;            4 * (L * -128) + 256 = 256 - 512 L (mod 2^32)
;
; Carry section. CIN is the SGPR mask s10 = 0x5a5a5a5a: lane L has carry-in 1 when bit L is set, i.e. L mod 8 is
; one of 1, 3, 4, 6, so the carry-in differs lane to lane.
;   A1 = 0xfffffff0, A2 = 0xffffffee (VGPRs), B = L.
;   K1  out[40+L]  v_add_co_u32      (A1 + L)               = 0xfffffff0 + L mod 2^32; carry for L >= 16   mask 0xFFFF0000
;   K2  out[72+L]  v_add_co_ci_u32   (A2 + L + CIN[L])      = 0xffffffee + L + CIN mod 2^32;
;                                                              carry when L + CIN >= 18: L >= 18, and L = 17 (CIN 1)
;                                                              mask 0xFFFE0000 (without the carry-in it would be 0xFFFC0000)
;   K3  96-bit ripple, the carry out of each stage is the carry in of the next (vcc_lo), word0 = L & 1:
;         out[104+L] = -1 + (L&1)               odd lanes 0 (carry 1), even lanes 0xFFFFFFFF   mask 0xAAAAAAAA
;         out[136+L] = -1 + 0 + carry           odd lanes 0 (carry 1), even lanes 0xFFFFFFFF   mask 0xAAAAAAAA
;         out[168+L] = 0x7fffffff + 0 + carry   odd lanes 0x80000000, even lanes 0x7FFFFFFF    mask 0
;   K4  out[200+L] v_sub_co_u32      16 - L mod 2^32; borrow for L > 16                                    mask 0xFFFE0000
;   K5  out[232+L] v_subrev_co_u32   L - 16 mod 2^32 (vcc_lo); borrow for L < 16                           mask 0x0000FFFF
;   K6  out[264+L] v_sub_co_ci_u32   L - 14 - CIN[L] mod 2^32; borrow when L < 14 + CIN: L <= 13, and
;                                                              L = 14 (CIN 1)                              mask 0x00007FFF
;   clamp, the same operands as K1, K2, K4, K6:
;   C1  out[296+L] v_add_co_u32 clamp     0xFFFFFFFF where K1 carried, else K1's word                      mask 0xFFFF0000
;   C2  out[328+L] v_add_co_ci_u32 clamp  0xFFFFFFFF where K2 carried, else K2's word                      mask 0xFFFE0000
;   C3  out[360+L] v_sub_co_u32 clamp     0 where K4 borrowed, else K4's word                              mask 0xFFFE0000
;   C4  out[392+L] v_sub_co_ci_u32 clamp  0 where K6 borrowed, else K6's word                              mask 0x00007FFF
; Carry/borrow flags, one dword per lane at out[424+L] (the carry-out lane mask read back per lane with v_cndmask_b32_e64,
; so the check does not depend on how a mask SGPR reads as data); bit k of lane L is that lane's bit of mask k:
;   bit 0 K1 (L >= 16)   bit 1 K2 (L >= 17)   bit 2 K4 (L >= 17)   bit 3 K5 (L < 16)   bit 4 K6 (L <= 14)
;   bit 5, 6 the first two ripple stages (L odd)   bit 7 the third ripple stage (never)
;   bit 8, 9, 10, 11 the masks of C1, C2, C3, C4 (the same as bits 0, 1, 2, 4)
;   lane 0 = 0x818 (K5, K6, C4)   lane 1 = 0x878 (those and the odd-lane bits 0x60)   lane 15 = 0x068 (K5, odd)
;   lane 16 = 0x101 (K1, C1)   lane 17 = 0x767 (K1 K2 K4 = 7, C1 C2 C3 = 0x700, odd 0x60)   lane 31 = 0x767
; The full expected table is DOT4_CARRY_WORDS in check.py (the formulas above in plain Python).
; group: 32,1,1
; Run: VKC_SUBGROUP_SIZE=32 VKC_OUT_WORDS=456 vk_compute_test cs_w32_dot4_carry.spv 1 dump (check.py / run_gpu_tests.py do this)
; mattr: +wavefrontsize32,-wavefrontsize64,+dot1-insts,+dot6-insts
v_lshlrev_b32_e32 v20, 2, v0
v_mov_b32_e32 v60, 0
s_mov_b32 s10, 0x5a5a5a5a
s_mov_b32 s8, 0x80808080
s_mov_b32 s9, -16
v_mov_b32_e32 v3, 0x80808080
v_mov_b32_e32 v4, 0x7f7f7f7f
v_mov_b32_e32 v6, -1
v_mov_b32_e32 v7, 0x100
v_mov_b32_e32 v8, 0x7fffffff
v_mov_b32_e32 v14, 0x7f80ff01
v_mov_b32_e32 v15, 0x80ff7f02
v_dot4_i32_i8 v10, v3, v3, 0
v_dot4_i32_i8 v11, v4, v3, 0
v_dot4_i32_i8 v12, v6, v4, v7
v_dot4_i32_i8 v13, v4, v4, v8
v_dot4_i32_i8 v16, v14, v15, 0
v_mov_b32_e32 v17, 0x7fffffff
v_dot4c_i32_i8 v17, v4, v4
v_mov_b32_e32 v18, v17
v_dot4c_i32_i8 v17, v3, v4
v_dot4_i32_i8 v19, s8, v6, s9
v_mov_b32_e32 v24, 0x1010101
v_mul_lo_u32 v22, v24, v0
v_mov_b32_e32 v23, 0x100
v_dot4c_i32_i8 v23, v22, v3
v_mov_b32_e32 v1, 0xfffffff0
v_mov_b32_e32 v2, 0xffffffee
v_add_co_u32_e64 v30, s20, v1, v0
v_cndmask_b32_e64 v62, 0, 1, s20
v_or_b32_e32 v60, v60, v62
v_add_co_ci_u32_e64 v31, s21, v2, v0, s10
v_cndmask_b32_e64 v62, 0, 2, s21
v_or_b32_e32 v60, v60, v62
v_and_b32_e32 v32, 1, v0
v_mov_b32_e32 v33, 0
v_add_co_u32_e64 v34, vcc_lo, -1, v32
v_cndmask_b32_e64 v62, 0, 32, vcc_lo
v_or_b32_e32 v60, v60, v62
v_add_co_ci_u32_e32 v35, vcc_lo, -1, v33, vcc_lo
v_cndmask_b32_e64 v62, 0, 64, vcc_lo
v_or_b32_e32 v60, v60, v62
v_add_co_ci_u32_e32 v36, vcc_lo, 0x7fffffff, v33, vcc_lo
v_cndmask_b32_e64 v62, 0, 128, vcc_lo
v_or_b32_e32 v60, v60, v62
v_sub_co_u32_e64 v37, s22, 16, v0
v_cndmask_b32_e64 v62, 0, 4, s22
v_or_b32_e32 v60, v60, v62
v_subrev_co_u32_e64 v38, vcc_lo, 16, v0
v_cndmask_b32_e64 v62, 0, 8, vcc_lo
v_or_b32_e32 v60, v60, v62
v_sub_co_ci_u32_e64 v39, s23, v0, 14, s10
v_cndmask_b32_e64 v62, 0, 16, s23
v_or_b32_e32 v60, v60, v62
v_add_co_u32_e64 v40, s29, v1, v0 clamp
v_cndmask_b32_e64 v62, 0, 256, s29
v_or_b32_e32 v60, v60, v62
v_add_co_ci_u32_e64 v41, s30, v2, v0, s10 clamp
v_cndmask_b32_e64 v62, 0, 512, s30
v_or_b32_e32 v60, v60, v62
v_sub_co_u32_e64 v42, s31, 16, v0 clamp
v_cndmask_b32_e64 v62, 0, 1024, s31
v_or_b32_e32 v60, v60, v62
v_sub_co_ci_u32_e64 v43, s32, v0, 14, s10 clamp
v_cndmask_b32_e64 v62, 0, 2048, s32
v_or_b32_e32 v60, v60, v62
buffer_store_dword v23, v20, s[0:3], 0 offen offset:32
buffer_store_dword v60, v20, s[0:3], 0 offen offset:1696
buffer_store_dword v30, v20, s[0:3], 0 offen offset:160
buffer_store_dword v31, v20, s[0:3], 0 offen offset:288
buffer_store_dword v34, v20, s[0:3], 0 offen offset:416
buffer_store_dword v35, v20, s[0:3], 0 offen offset:544
buffer_store_dword v36, v20, s[0:3], 0 offen offset:672
buffer_store_dword v37, v20, s[0:3], 0 offen offset:800
buffer_store_dword v38, v20, s[0:3], 0 offen offset:928
buffer_store_dword v39, v20, s[0:3], 0 offen offset:1056
buffer_store_dword v40, v20, s[0:3], 0 offen offset:1184
buffer_store_dword v41, v20, s[0:3], 0 offen offset:1312
buffer_store_dword v42, v20, s[0:3], 0 offen offset:1440
buffer_store_dword v43, v20, s[0:3], 0 offen offset:1568
v_cmpx_eq_u32_e32 0, v0
v_mov_b32_e32 v50, v10
buffer_store_dword v50, off, s[0:3], 0 offset:0
v_mov_b32_e32 v50, v11
buffer_store_dword v50, off, s[0:3], 0 offset:4
v_mov_b32_e32 v50, v12
buffer_store_dword v50, off, s[0:3], 0 offset:8
v_mov_b32_e32 v50, v13
buffer_store_dword v50, off, s[0:3], 0 offset:12
v_mov_b32_e32 v50, v16
buffer_store_dword v50, off, s[0:3], 0 offset:16
v_mov_b32_e32 v50, v18
buffer_store_dword v50, off, s[0:3], 0 offset:20
v_mov_b32_e32 v50, v17
buffer_store_dword v50, off, s[0:3], 0 offset:24
v_mov_b32_e32 v50, v19
buffer_store_dword v50, off, s[0:3], 0 offset:28
s_endpgm
