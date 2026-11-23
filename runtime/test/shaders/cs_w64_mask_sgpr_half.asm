; A wave64 VOP3 mask operand written as one SGPR (`s4`, as the corpus listing prints it) is the pair s[4:5] to the
; hardware, also when s4 is a certified mask half (a copy of exec_lo or vcc_lo). s5 is independent: lanes 32-63 take
; their bit L-32 from s5, here plain data. Review of #2241: the translator left a certified half as one register, so
; lanes 32-63 used s4 as well and s5 was ignored.
; v0 = lane (LocalInvocationId.x, 0..63). Each case writes 0/1 into its register for the lanes it runs, EXEC goes back
; to all lanes, v_cmp turns the marks into a lane mask and lane 0 stores word = (set lanes) | (lowest set lane) << 8.
;   out[0] EXEC = lanes 0-39, s4 = exec_lo, s5 = 0xf0f0f0f0, v_cndmask_b32_e64 v, 0, 1, s4
;          lanes 0-31 (s4 all ones) + 36-39 (bits 4-7 of s5)                       36 | 0 = 36   (old: 0-39 = 40)
;   out[1] the same with v_add_co_ci_u32_e64 v, s6, 0, 0, s4 (carry-in is the result)  36 | 0 = 36   (old: 40)
;   out[2] s4 = vcc_lo of v_cmp 8 > v0 (lanes 0-7), s5 = 0x00ff00ff, v_cndmask_b32_e64 v, 0, 1, s4
;          lanes 0-7 + 32-39 + 48-55                                               24 | 0 = 24   (old: lanes 0-7 = 8)
;   out[3] the same with v_add_co_ci_u32_e64                                         24 | 0 = 24   (old: 8)
; No mattr line: wave64 is inferred from exec_hi, as in cs_w64_mask_sgpr.
; group: 64,1,1
; Run: VKC_SUBGROUP_SIZE=64 vk_compute_test cs_w64_mask_sgpr_half.spv 1 dump --require-wave64
;   (or translated with --wave64-emulate, on any subgroup size) -> out[0..3] = 36 36 24 24
v_mov_b32_e32 v10, 0
v_mov_b32_e32 v11, 0
v_mov_b32_e32 v12, 0
v_mov_b32_e32 v13, 0
s_mov_b64 exec, -1
v_cmpx_gt_u32_e32 40, v0
s_mov_b32 s4, exec_lo
s_mov_b32 s5, 0xf0f0f0f0
v_cndmask_b32_e64 v10, 0, 1, s4
s_mov_b64 exec, -1
v_cmpx_gt_u32_e32 40, v0
s_mov_b32 s4, exec_lo
s_mov_b32 s5, 0xf0f0f0f0
v_add_co_ci_u32_e64 v11, s6, 0, 0, s4
s_mov_b64 exec, -1
v_cmp_gt_u32_e32 vcc_lo, 8, v0
s_mov_b32 s4, vcc_lo
s_mov_b32 s5, 0x00ff00ff
v_cndmask_b32_e64 v12, 0, 1, s4
s_mov_b64 exec, -1
v_cmp_gt_u32_e32 vcc_lo, 8, v0
s_mov_b32 s4, vcc_lo
s_mov_b32 s5, 0x00ff00ff
v_add_co_ci_u32_e64 v13, s6, 0, 0, s4
s_mov_b64 exec, -1
v_cmp_ne_u32_e32 vcc_lo, 0, v10
s_bcnt1_i32_b64 s20, vcc
s_ff1_i32_b64 s21, vcc
s_lshl_b32 s21, s21, 8
s_or_b32 s20, s20, s21
v_cmp_ne_u32_e32 vcc_lo, 0, v11
s_bcnt1_i32_b64 s22, vcc
s_ff1_i32_b64 s23, vcc
s_lshl_b32 s23, s23, 8
s_or_b32 s22, s22, s23
v_cmp_ne_u32_e32 vcc_lo, 0, v12
s_bcnt1_i32_b64 s24, vcc
s_ff1_i32_b64 s25, vcc
s_lshl_b32 s25, s25, 8
s_or_b32 s24, s24, s25
v_cmp_ne_u32_e32 vcc_lo, 0, v13
s_bcnt1_i32_b64 s26, vcc
s_ff1_i32_b64 s27, vcc
s_lshl_b32 s27, s27, 8
s_or_b32 s26, s26, s27
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
s_endpgm
