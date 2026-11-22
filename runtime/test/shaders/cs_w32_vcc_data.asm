; Wave32 VCC holding data, read as a lane mask (v_cndmask_b32 selector, VOP2 carry-in, s_and_saveexec_b32 source).
; RDNA2 ISA: VCC is one bit per lane, and lane L of these instructions reads VCC[L], whatever wrote the register:
; a scalar move of a constant or of a data SGPR puts that 32-bit value in VCC_LO. Until 2026-11-22 the translator
; read such a VCC raw (Gen.mask_src), so any non-zero value selected every lane.
; v0 = lane (LocalInvocationId.x, 0..31). Each case leaves a mark (non-zero VGPR) in the lanes it selects; with
; EXEC back to all lanes, v_cmp turns the marks into a lane mask and lane 0 stores
; word = (set lanes) | (lowest set lane) << 8, from s_bcnt1 and s_ff1 of that mask (no lane: 0xffffff00).
;   out[0] s_mov_b32 vcc_lo, 0xa;          v_cndmask_b32_e32           -> lanes 1, 3              2 | 1<<8  = 258
;   out[1] s_mov_b32 s5, 0xf0f0; vcc_lo, s5; v_cndmask_b32_e32         -> lanes 4-7, 12-15        8 | 4<<8  = 1032
;   out[2] s_mov_b32 vcc_lo, 0x80000001;   v_add_co_ci_u32_e32 0+0+c   -> lanes 0, 31             2 | 0     = 2
;   out[3] s_mov_b32 vcc_lo, -16 (inline); v_cndmask_b32_e64 ..., vcc_lo -> lanes 4-31 (0xfffffff0) 28 | 4<<8 = 1052
;   out[4] s_mov_b32 vcc_lo, 0;            v_cndmask_b32_e32           -> no lane                 0xffffff00
;   out[5] v_cmp_gt_u32_e32 vcc_lo, 5, v0 (a certified mask, control) -> lanes 0-4               5 | 0     = 5
;   out[6] s_mov_b32 vcc_lo, 0xff0000;     s_and_saveexec_b32 s7, vcc_lo -> EXEC = lanes 16-23   8 | 16<<8 = 4104
;   out[7] s_mov_b32 vcc_lo, 0x12345678;   v_cndmask_b32_e32           -> 13 lanes, lowest 3      13 | 3<<8 = 781
; The old translation (every lane for a non-zero VCC) gives 32 32 32 32 0xffffff00 5 32 32.
; group: 32,1,1
; Run: VKC_SUBGROUP_SIZE=32 vk_compute_test cs_w32_vcc_data.spv 1 dump
;   -> out[0..7] = 258 1032 2 1052 0xffffff00 5 4104 781
; mattr: +wavefrontsize32,-wavefrontsize64
v_mov_b32_e32 v20, 0
v_mov_b32_e32 v21, 1
s_mov_b32 vcc_lo, 0xa
v_cndmask_b32_e32 v10, 0, v21, vcc_lo
s_mov_b32 s5, 0xf0f0
s_mov_b32 vcc_lo, s5
v_cndmask_b32_e32 v11, 0, v21, vcc_lo
s_mov_b32 vcc_lo, 0x80000001
v_add_co_ci_u32_e32 v12, vcc_lo, 0, v20, vcc_lo
s_mov_b32 vcc_lo, -16
v_cndmask_b32_e64 v13, 0, 1, vcc_lo
s_mov_b32 vcc_lo, 0
v_cndmask_b32_e32 v14, 0, v21, vcc_lo
v_cmp_gt_u32_e32 vcc_lo, 5, v0
v_cndmask_b32_e32 v15, 0, v21, vcc_lo
v_mov_b32_e32 v16, 0
s_mov_b32 vcc_lo, 0xff0000
s_and_saveexec_b32 s7, vcc_lo
v_mov_b32_e32 v16, 1
s_mov_b32 exec_lo, -1
s_mov_b32 vcc_lo, 0x12345678
v_cndmask_b32_e32 v17, 0, v21, vcc_lo
v_cmp_ne_u32_e32 vcc_lo, 0, v10
s_bcnt1_i32_b32 s20, vcc_lo
s_ff1_i32_b32 s21, vcc_lo
s_lshl_b32 s21, s21, 8
s_or_b32 s20, s20, s21
v_cmp_ne_u32_e32 vcc_lo, 0, v11
s_bcnt1_i32_b32 s22, vcc_lo
s_ff1_i32_b32 s23, vcc_lo
s_lshl_b32 s23, s23, 8
s_or_b32 s22, s22, s23
v_cmp_ne_u32_e32 vcc_lo, 0, v12
s_bcnt1_i32_b32 s24, vcc_lo
s_ff1_i32_b32 s25, vcc_lo
s_lshl_b32 s25, s25, 8
s_or_b32 s24, s24, s25
v_cmp_ne_u32_e32 vcc_lo, 0, v13
s_bcnt1_i32_b32 s26, vcc_lo
s_ff1_i32_b32 s27, vcc_lo
s_lshl_b32 s27, s27, 8
s_or_b32 s26, s26, s27
v_cmp_ne_u32_e32 vcc_lo, 0, v14
s_bcnt1_i32_b32 s28, vcc_lo
s_ff1_i32_b32 s29, vcc_lo
s_lshl_b32 s29, s29, 8
s_or_b32 s28, s28, s29
v_cmp_ne_u32_e32 vcc_lo, 0, v15
s_bcnt1_i32_b32 s30, vcc_lo
s_ff1_i32_b32 s31, vcc_lo
s_lshl_b32 s31, s31, 8
s_or_b32 s30, s30, s31
v_cmp_ne_u32_e32 vcc_lo, 0, v16
s_bcnt1_i32_b32 s32, vcc_lo
s_ff1_i32_b32 s33, vcc_lo
s_lshl_b32 s33, s33, 8
s_or_b32 s32, s32, s33
v_cmp_ne_u32_e32 vcc_lo, 0, v17
s_bcnt1_i32_b32 s34, vcc_lo
s_ff1_i32_b32 s35, vcc_lo
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
