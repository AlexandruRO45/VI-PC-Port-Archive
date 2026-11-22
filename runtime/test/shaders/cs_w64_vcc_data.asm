; Wave64 VCC holding data, read as a lane mask (v_cndmask_b32 selector in e32 and e64 form, VOP2 carry-in,
; s_and_saveexec_b64 source). RDNA2 ISA: VCC is 64 bits, one per lane, and lane L of these instructions reads
; VCC[L] whatever wrote the register: s_mov_b32 vcc_lo, K puts K in VCC[31:0] and s_mov_b32 vcc_hi, K in
; VCC[63:32]. Until 2026-11-22 the translator read such a VCC as the raw vcc_lo word in every lane (Gen.mask_src),
; so a non-zero vcc_lo selected all 64 lanes and vcc_hi was ignored. Wave32 is cs_w32_vcc_data.asm.
; v0 = lane (0..63), v20 = 0, v21 = 1. Each case leaves a mark (non-zero VGPR) in the lanes it selects; with EXEC
; back to all lanes, v_cmp turns the marks into a lane mask and lane 0 stores
; word = (set lanes) | (lowest set lane) << 8, from s_bcnt1_i32_b64 and s_ff1_i32_b64 (no lane: 0xffffff00).
;   out[0] vcc_lo = 0xa, vcc_hi = 0x30000;           v_cndmask_b32_e32 -> lanes 1, 3, 48, 49     4 | 1<<8  = 260
;   out[1] vcc_lo = s5 (0xf0f0), vcc_hi = s6 (0x80000000); v_cndmask_b32_e32
;                                                     -> lanes 4-7, 12-15, 63                    9 | 4<<8  = 1033
;   out[2] vcc_lo = 0x80000001, vcc_hi = 1;          v_add_co_ci_u32_e32 0 + 0 + VCC[L]
;                                                     -> lanes 0, 31, 32                         3 | 0     = 3
;   out[3] vcc_lo = -16 (inline, 0xfffffff0), vcc_hi = 0xffff; v_cndmask_b32_e64 ..., vcc
;                                                     -> lanes 4-31, 32-47                       44 | 4<<8 = 1068
;   out[4] vcc_lo = 0, vcc_hi = 0x100;               v_cndmask_b32_e32 -> lane 40               1 | 40<<8 = 10241
;   out[5] v_cmp_gt_u32_e32 vcc, 40, v0 (a mask, control) -> lanes 0-39                         40 | 0    = 40
;   out[6] vcc_lo = 0xff0000, vcc_hi = 0xf;          s_and_saveexec_b64 s[8:9], vcc
;                                                     -> EXEC = lanes 16-23, 32-35               12 | 16<<8 = 4108
;   out[7] s_mov_b64 vcc, s[4:5] (s4 = 0x12345678, s5 = 0x9abcdef0, a 64-bit data pair); v_cndmask_b32_e32
;          -> the 64 bits 0x9abcdef0_12345678: 13 + 19 = 32 lanes, lowest bit 3          32 | 3<<8 = 800
; (out[7] is a control for VCC written whole: s_mov_b64 already gave each lane its bit.)
; The old translation (the raw vcc_lo word in every lane, any non-zero word selecting the lane) gives
; 64 64 64 32 0xffffff00 40 40 800: in cases 3 and 6 VCC held a mask just before (the carry-out, the compare), and
; s_mov_b32 vcc_lo left lanes 32-63 with that mask's bits (none, lanes 32-39), whatever vcc_hi was set to.
; Run: VKC_SUBGROUP_SIZE=64 vk_compute_test cs_w64_vcc_data.spv 1 dump --require-wave64
;   (or translated with --wave64-emulate, on any subgroup size) -> out[0..7] = 260 1033 3 1068 10241 40 4108 800
; mattr: +wavefrontsize64,-wavefrontsize32
v_mov_b32_e32 v20, 0
v_mov_b32_e32 v21, 1
s_mov_b32 vcc_lo, 0xa
s_mov_b32 vcc_hi, 0x30000
v_cndmask_b32_e32 v10, 0, v21, vcc
s_mov_b32 s5, 0xf0f0
s_mov_b32 s6, 0x80000000
s_mov_b32 vcc_lo, s5
s_mov_b32 vcc_hi, s6
v_cndmask_b32_e32 v11, 0, v21, vcc
s_mov_b32 vcc_lo, 0x80000001
s_mov_b32 vcc_hi, 1
v_add_co_ci_u32_e32 v12, vcc, 0, v20, vcc
s_mov_b32 vcc_lo, -16
s_mov_b32 vcc_hi, 0xffff
v_cndmask_b32_e64 v13, 0, 1, vcc
s_mov_b32 vcc_lo, 0
s_mov_b32 vcc_hi, 0x100
v_cndmask_b32_e32 v14, 0, v21, vcc
v_cmp_gt_u32_e32 vcc, 40, v0
v_cndmask_b32_e32 v15, 0, v21, vcc
v_mov_b32_e32 v16, 0
s_mov_b32 vcc_lo, 0xff0000
s_mov_b32 vcc_hi, 0xf
s_and_saveexec_b64 s[8:9], vcc
v_mov_b32_e32 v16, 1
s_mov_b64 exec, -1
s_mov_b32 s4, 0x12345678
s_mov_b32 s5, 0x9abcdef0
s_mov_b64 vcc, s[4:5]
v_cndmask_b32_e32 v17, 0, v21, vcc
v_cmp_ne_u32_e32 vcc, 0, v10
s_bcnt1_i32_b64 s20, vcc
s_ff1_i32_b64 s21, vcc
s_lshl_b32 s21, s21, 8
s_or_b32 s20, s20, s21
v_cmp_ne_u32_e32 vcc, 0, v11
s_bcnt1_i32_b64 s22, vcc
s_ff1_i32_b64 s23, vcc
s_lshl_b32 s23, s23, 8
s_or_b32 s22, s22, s23
v_cmp_ne_u32_e32 vcc, 0, v12
s_bcnt1_i32_b64 s24, vcc
s_ff1_i32_b64 s25, vcc
s_lshl_b32 s25, s25, 8
s_or_b32 s24, s24, s25
v_cmp_ne_u32_e32 vcc, 0, v13
s_bcnt1_i32_b64 s26, vcc
s_ff1_i32_b64 s27, vcc
s_lshl_b32 s27, s27, 8
s_or_b32 s26, s26, s27
v_cmp_ne_u32_e32 vcc, 0, v14
s_bcnt1_i32_b64 s28, vcc
s_ff1_i32_b64 s29, vcc
s_lshl_b32 s29, s29, 8
s_or_b32 s28, s28, s29
v_cmp_ne_u32_e32 vcc, 0, v15
s_bcnt1_i32_b64 s30, vcc
s_ff1_i32_b64 s31, vcc
s_lshl_b32 s31, s31, 8
s_or_b32 s30, s30, s31
v_cmp_ne_u32_e32 vcc, 0, v16
s_bcnt1_i32_b64 s32, vcc
s_ff1_i32_b64 s33, vcc
s_lshl_b32 s33, s33, 8
s_or_b32 s32, s32, s33
v_cmp_ne_u32_e32 vcc, 0, v17
s_bcnt1_i32_b64 s34, vcc
s_ff1_i32_b64 s35, vcc
s_lshl_b32 s35, s35, 8
s_or_b32 s34, s34, s35
v_cmpx_eq_u32_e32 0, v0
v_mov_b32_e32 v1, s20
v_mov_b32_e32 v2, 0
buffer_store_dword v1, v2, s[0:3], 0 offen
v_mov_b32_e32 v1, s22
v_mov_b32_e32 v2, 4
buffer_store_dword v1, v2, s[0:3], 0 offen
v_mov_b32_e32 v1, s24
v_mov_b32_e32 v2, 8
buffer_store_dword v1, v2, s[0:3], 0 offen
v_mov_b32_e32 v1, s26
v_mov_b32_e32 v2, 12
buffer_store_dword v1, v2, s[0:3], 0 offen
v_mov_b32_e32 v1, s28
v_mov_b32_e32 v2, 16
buffer_store_dword v1, v2, s[0:3], 0 offen
v_mov_b32_e32 v1, s30
v_mov_b32_e32 v2, 20
buffer_store_dword v1, v2, s[0:3], 0 offen
v_mov_b32_e32 v1, s32
v_mov_b32_e32 v2, 24
buffer_store_dword v1, v2, s[0:3], 0 offen
v_mov_b32_e32 v1, s34
v_mov_b32_e32 v2, 28
buffer_store_dword v1, v2, s[0:3], 0 offen
s_endpgm
