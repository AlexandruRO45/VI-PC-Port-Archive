; SGPRs read as lane masks in wave32 (Gen.mask_src on a register that holds plain data, not a lane mask).
; Bit L of an SGPR used as a lane mask is lane L's bit (RDNA2 ISA: EXEC and a mask SGPR are one bit per lane; the
; saveexec ops and v_cndmask's mask operand read it that way). Until SHADER-200 the translator gave every lane the
; raw register value, so any non-zero SGPR turned all lanes on (found by Vonsek on #1218: 0xff00ff left 32 lanes on
; where the ISA gives 16).
; v0 = lane (LocalInvocationId.x, 0..31). Each case sets EXEC, runs the op and marks the lanes it leaves on (a VALU
; write under the new EXEC); with EXEC back to all lanes, v_cmp turns the marks into a lane mask and lane 0 stores
; word = (set lanes) | (lowest set lane) << 8, from s_bcnt1 and s_ff1 of that mask (no lane: 0xffffff00).
;   out[0] EXEC=all,    s4 = 0xff00ff, s_and_saveexec_b32 s7, s4      -> lanes 0-7, 16-23           16 | 0     = 16
;   out[1] EXEC=all,    s4 = 0xf0, s_and_saveexec_b32 s7, s4          -> lanes 4-7                  4 | 4<<8  = 1028
;   out[2] EXEC=0xffff, s4 = 0xff00ff00, s_or_saveexec_b32 s7, s4     -> 0xff00ffff                 24 | 0    = 24
;   out[3] EXEC=all,    s4 = 0, s_and_saveexec_b32 s7, s4             -> no lane                    0xffffff00 = 4294967040
;   out[4] EXEC=all,    s4 = -1, s_and_saveexec_b32 s7, s4            -> all lanes                  32 | 0     = 32
;   out[5] v_cndmask_b32_e64 v15, 0, 1, s4 with s4 = 0xf0f00ff0       -> lanes 4-11, 20-23, 28-31         16 | 4<<8 = 1040
;   out[6] s5 = lanes 0-15 (v_cmp), s4 = 0xff00ff data, s_and_b32 s6, s5, s4, s_and_saveexec_b32 s7, s6
;                                                                     -> 0xffff & 0xff00ff = lanes 0-7   8 | 0 = 8
;   out[7] EXEC=lanes 0-7, s4 = 0xff00ff, s_andn2_saveexec_b32 s7, s4 -> S0 & ~EXEC = 0xff0000 = lanes 16-23  8 | 16<<8 = 4104
; The raw value for every lane (the old translation) made a non-zero register mean all lanes, so out[0..7] were
; 32 32 32 4294967040 32 32 16 2072 before the fix (measured on the GPU).
; group: 32,1,1
; Run: VKC_SUBGROUP_SIZE=32 vk_compute_test cs_w32_mask_sgpr.spv 1 dump
;   -> out[0..7] = 16 1028 24 4294967040 32 1040 8 4104
; mattr: +wavefrontsize32,-wavefrontsize64
v_mov_b32_e32 v10, 0
v_mov_b32_e32 v11, 0
v_mov_b32_e32 v12, 0
v_mov_b32_e32 v13, 0
v_mov_b32_e32 v14, 0
v_mov_b32_e32 v15, 0
v_mov_b32_e32 v16, 0
v_mov_b32_e32 v17, 0
s_mov_b32 exec_lo, -1
s_mov_b32 s4, 0xff00ff
s_and_saveexec_b32 s7, s4
v_mov_b32_e32 v10, 1
s_mov_b32 exec_lo, -1
s_mov_b32 s4, 0xf0
s_and_saveexec_b32 s7, s4
v_mov_b32_e32 v11, 1
s_mov_b32 exec_lo, 0xffff
s_mov_b32 s4, 0xff00ff00
s_or_saveexec_b32 s7, s4
v_mov_b32_e32 v12, 1
s_mov_b32 exec_lo, -1
s_mov_b32 s4, 0
s_and_saveexec_b32 s7, s4
v_mov_b32_e32 v13, 1
s_mov_b32 exec_lo, -1
s_mov_b32 s4, -1
s_and_saveexec_b32 s7, s4
v_mov_b32_e32 v14, 1
s_mov_b32 exec_lo, -1
s_mov_b32 s4, 0xf0f00ff0
v_cndmask_b32_e64 v15, 0, 1, s4
s_mov_b32 exec_lo, -1
v_cmp_gt_u32_e64 s5, 16, v0
s_mov_b32 s4, 0xff00ff
s_and_b32 s6, s5, s4
s_and_saveexec_b32 s7, s6
v_mov_b32_e32 v16, 1
s_mov_b32 exec_lo, -1
v_cmpx_gt_u32_e32 8, v0
s_mov_b32 s4, 0xff00ff
s_andn2_saveexec_b32 s7, s4
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
