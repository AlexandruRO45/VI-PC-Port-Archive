; s_cmov_b32 / s_cmovk_i32 into EXEC_LO in wave32: EXEC = SCC ? S0 : EXEC (RDNA2 ISA SOP1 S_CMOV_B32, SOPK S_CMOVK_I32:
; D = SCC ? S0 : D, SCC read and not written; S_CMOVK_I32's S0 is SIMM16 sign-extended). EXEC is one bit per lane, so a
; constant or data SGPR source gives lane L its bit L, and a lane mask from v_cmp is taken as it is. Until this change
; the translator raised Unsupported("s_cmov_b32 destination") for any non-SGPR destination.
; v0 = lane (0..31). Each case sets EXEC, sets SCC with s_cmp, runs the cmov and marks the lanes it leaves on (a VALU
; write under the new EXEC); with EXEC back to all lanes, v_cmp turns the marks into a lane mask and lane 0 stores
; word = (set lanes) | (lowest set lane) << 8, from s_bcnt1 and s_ff1 of that mask (no lane: 0xffffff00).
;   out[0] EXEC=0xff,   SCC=1, s_cmov_b32 exec_lo, -1          -> all 32 lanes                      32 | 0     = 32
;   out[1] EXEC=0xff,   SCC=0, s_cmov_b32 exec_lo, -1          -> EXEC kept, lanes 0-7               8 | 0      = 8
;   out[2] EXEC=all,    SCC=1, s_cmov_b32 exec_lo, s20=0xf0f0  -> lanes 4-7, 12-15                   8 | 4<<8   = 1032
;   out[3] EXEC=all,    SCC=1, s_cmovk_i32 exec_lo, 0x8003     -> 0xffff8003: lanes 0, 1, 15-31      19 | 0     = 19
;   out[4] EXEC=0xff00, SCC=0, s_cmovk_i32 exec_lo, 7          -> EXEC kept, lanes 8-15              8 | 8<<8   = 2056
;   out[5] s20 = v_cmp_gt_u32 6 > lane (lanes 0-5); EXEC=0xff0, SCC=1, s_cmov_b32 exec_lo, s20 -> lanes 0-5   6 | 0 = 6
;   out[6] EXEC=0,      SCC=1, s_cmov_b32 exec_lo, 0x80000002  -> lanes 1, 31                        2 | 1<<8   = 258
;   out[7] EXEC=0xf,    SCC=0, s_cmov_b32 exec_lo, s21=0xf0f0  -> EXEC kept, lanes 0-3               4 | 0      = 4
; group: 32,1,1
; Run: VKC_SUBGROUP_SIZE=32 vk_compute_test cs_w32_cmov_exec.spv 1 dump
;   -> out[0..7] = 32 8 1032 19 2056 6 258 4
; mattr: +wavefrontsize32,-wavefrontsize64
v_mov_b32_e32 v10, 0
v_mov_b32_e32 v11, 0
v_mov_b32_e32 v12, 0
v_mov_b32_e32 v13, 0
v_mov_b32_e32 v14, 0
v_mov_b32_e32 v15, 0
v_mov_b32_e32 v16, 0
v_mov_b32_e32 v17, 0
s_mov_b32 exec_lo, 0xff
s_cmp_eq_u32 s4, s4
s_cmov_b32 exec_lo, -1
v_mov_b32_e32 v10, 1
s_mov_b32 exec_lo, 0xff
s_cmp_lg_u32 s4, s4
s_cmov_b32 exec_lo, -1
v_mov_b32_e32 v11, 1
s_mov_b32 exec_lo, -1
s_mov_b32 s20, 0xf0f0
s_cmp_eq_u32 s4, s4
s_cmov_b32 exec_lo, s20
v_mov_b32_e32 v12, 1
s_mov_b32 exec_lo, -1
s_cmp_eq_u32 s4, s4
s_cmovk_i32 exec_lo, 0x8003
v_mov_b32_e32 v13, 1
s_mov_b32 exec_lo, 0xff00
s_cmp_lg_u32 s4, s4
s_cmovk_i32 exec_lo, 0x7
v_mov_b32_e32 v14, 1
s_mov_b32 exec_lo, -1
v_cmp_gt_u32_e64 s20, 6, v0
s_mov_b32 exec_lo, 0xff0
s_cmp_eq_u32 s4, s4
s_cmov_b32 exec_lo, s20
v_mov_b32_e32 v15, 1
s_mov_b32 exec_lo, 0
s_cmp_eq_u32 s4, s4
s_cmov_b32 exec_lo, 0x80000002
v_mov_b32_e32 v16, 1
s_mov_b32 exec_lo, 0xf
s_mov_b32 s21, 0xf0f0
s_cmp_lg_u32 s4, s4
s_cmov_b32 exec_lo, s21
v_mov_b32_e32 v17, 1
s_mov_b32 exec_lo, -1
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
