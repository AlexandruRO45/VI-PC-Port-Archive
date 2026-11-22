; SGPR pairs read as lane masks in wave64 (Gen.mask_src on a pair that holds plain data, not a lane mask).
; Bit L of s[r:r+1] is lane L's bit: lanes 0-31 read s[r], lanes 32-63 read s[r+1] at bit L-32 (RDNA2 ISA). Until
; SHADER-200 every lane got the raw value of s[r], so any non-zero pair turned all lanes on (#1218).
; v0 = lane (LocalInvocationId.x, 0..63). Each case sets EXEC, runs the op and marks the lanes it leaves on; with EXEC
; back to all lanes, v_cmp turns the marks into a lane mask and lane 0 stores
; word = (set lanes) | (lowest set lane) << 8 (no lane: 0xffffff00).
;   out[0] s[4:5] = 0xf0f0f0f0_00ff00ff, s_and_saveexec_b64 s[8:9], s[4:5]  -> 16 + 16 lanes        32 | 0     = 32
;   out[1] s[4:5] = 1_00000000, s_and_saveexec_b64                          -> lane 32 only        1 | 32<<8 = 8193
;   out[2] s[4:5] = 0, s_and_saveexec_b64                                   -> no lane             4294967040
;   out[3] s[4:5] = -1, s_and_saveexec_b64                                  -> all 64 lanes        64 | 0     = 64
;   out[4] v_cndmask_b32_e64 v, 0, 1, s[4:5] with s[4:5] = 0x0000ff00_f0f00ff0  -> lanes 4-11, 20-27, 40-47   24 | 4<<8 = 1048
;   out[5] EXEC = lanes 0-7, s[4:5] = 0xf0f0f0f0_00ff00ff, s_andn2_saveexec_b64 -> S0 & ~EXEC = 0xf0f0f0f0_00ff0000
;                                                                                            24 | 16<<8 = 4120
;   out[6] s[10:11] = lanes 0-39 (v_cmp), s[4:5] = 0xff_00ff00ff data, s_and_b64 s[12:13], s[10:11], s[4:5],
;          s_and_saveexec_b64 s[8:9], s[12:13]                              -> lanes 0-7, 16-23, 32-39   24 | 0 = 24
;   out[7] EXEC = lanes 0-7, s[4:5] = 1_ff00ff00, s_or_saveexec_b64              -> 0x1_ff00ffff        25 | 0     = 25
; The raw s[r] for every lane (the old translation) made every non-zero pair mean all lanes, so out[0..7] were
; 64 4294967040 4294967040 64 64 2104 40 64 before the fix (measured on the GPU: see design.md).
; group: 64,1,1
; Run: VKC_SUBGROUP_SIZE=64 vk_compute_test cs_w64_mask_sgpr.spv 1 dump --require-wave64
;   (or translated with --wave64-emulate, on any subgroup size) -> out[0..7] = 32 8193 4294967040 64 1048 4120 24 25
; mattr: +wavefrontsize64,-wavefrontsize32
v_mov_b32_e32 v10, 0
v_mov_b32_e32 v11, 0
v_mov_b32_e32 v12, 0
v_mov_b32_e32 v13, 0
v_mov_b32_e32 v14, 0
v_mov_b32_e32 v15, 0
v_mov_b32_e32 v16, 0
v_mov_b32_e32 v17, 0
s_mov_b64 exec, -1
s_mov_b32 s4, 0xff00ff
s_mov_b32 s5, 0xf0f0f0f0
s_and_saveexec_b64 s[8:9], s[4:5]
v_mov_b32_e32 v10, 1
s_mov_b64 exec, -1
s_mov_b32 s4, 0
s_mov_b32 s5, 1
s_and_saveexec_b64 s[8:9], s[4:5]
v_mov_b32_e32 v11, 1
s_mov_b64 exec, -1
s_mov_b32 s4, 0
s_mov_b32 s5, 0
s_and_saveexec_b64 s[8:9], s[4:5]
v_mov_b32_e32 v12, 1
s_mov_b64 exec, -1
s_mov_b32 s4, -1
s_mov_b32 s5, -1
s_and_saveexec_b64 s[8:9], s[4:5]
v_mov_b32_e32 v13, 1
s_mov_b64 exec, -1
s_mov_b32 s4, 0xf0f00ff0
s_mov_b32 s5, 0xff00
v_cndmask_b32_e64 v14, 0, 1, s[4:5]
s_mov_b64 exec, -1
v_cmpx_gt_u32_e32 8, v0
s_mov_b32 s4, 0xff00ff
s_mov_b32 s5, 0xf0f0f0f0
s_andn2_saveexec_b64 s[8:9], s[4:5]
v_mov_b32_e32 v15, 1
s_mov_b64 exec, -1
v_cmp_gt_u32_e64 s[10:11], 40, v0
s_mov_b32 s4, 0xff00ff
s_mov_b32 s5, 0xff
s_and_b64 s[12:13], s[10:11], s[4:5]
s_and_saveexec_b64 s[8:9], s[12:13]
v_mov_b32_e32 v16, 1
s_mov_b64 exec, -1
v_cmpx_gt_u32_e32 8, v0
s_mov_b32 s4, 0xff00ff00
s_mov_b32 s5, 1
s_or_saveexec_b64 s[8:9], s[4:5]
v_mov_b32_e32 v17, 1
s_mov_b64 exec, -1
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
