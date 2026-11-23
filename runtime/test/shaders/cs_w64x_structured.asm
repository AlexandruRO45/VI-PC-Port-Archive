; The structured --wave64-emulate plan (single 64-lane wave, `inline_all`) with a cross-lane exchange inside a
; wave-wide branch. The vote is decided by the upper half only (VCC = lane 40), so the whole wave takes the same
; side and the cross-lane ops after the branch run with all 64 lanes committed to one path.
;   v1 = lane (v_mbcnt_lo/hi_b32 over -1)
; (A) VCC = lane 40 only, s_cbranch_vccnz taken -> `s_mov_b32 s10, 0` runs in NO lane: s10 = 100. A decision taken
;     per half would leave s10 = 0 in the low half (its VCC is 0).
; (B) after the branch, s[6:7] = lanes with (lane & 7) == 7, eight lanes spread over both halves
;     (7, 15, 23, 31, 39, 47, 55, 63); s8 = s_bcnt1_i32_b64 = 8 (an exchange that only saw one half would give 4).
; (C) v2 = v_mbcnt_lo_u32_b32/v_mbcnt_hi_u32_b32 of s[6:7] = the selected lanes below this lane: 7 at lane 63,
;     2 at lane 20.
; (D) EXEC = lanes 32..47 (exec_lo 0, exec_hi 0xffff): s9 = s_bcnt1_i32_b64 exec = 16.
; Stores: out[0] lane 0 s10, out[1] lane 63 s10, out[2] lane 63 v2, out[3] lane 20 v2, out[4] lane 0 s8,
;   out[5] lane 63 s8, out[6] lane 40 s9, out[7] lane 0 s9.
; Run: vk_compute_test cs_w64x_structured.spv 1 dump --require-wave64
;   -> out[0..7] = 100 100 7 2 8 8 16 16
; mattr: +wavefrontsize64,-wavefrontsize32
v_mbcnt_lo_u32_b32 v1, -1, 0
v_mbcnt_hi_u32_b32 v1, -1, v1
s_mov_b32 s10, 100
v_cmp_eq_u32_e32 vcc, 40, v1
s_cbranch_vccnz 1
s_mov_b32 s10, 0
v_and_b32_e32 v3, 7, v1
v_cmp_eq_u32_e64 s[6:7], 7, v3
s_bcnt1_i32_b64 s8, s[6:7]
v_mbcnt_lo_u32_b32 v2, s6, 0
v_mbcnt_hi_u32_b32 v2, s7, v2
s_mov_b32 exec_lo, 0
s_mov_b32 exec_hi, 0xffff
s_bcnt1_i32_b64 s9, exec
s_mov_b64 exec, -1
v_cmpx_eq_u32_e32 0, v1
v_mov_b32_e32 v20, s10
v_mov_b32_e32 v21, 0
buffer_store_dword v20, v21, s[0:3], 0 offen
s_mov_b64 exec, -1
v_cmpx_eq_u32_e32 63, v1
v_mov_b32_e32 v20, s10
v_mov_b32_e32 v21, 4
buffer_store_dword v20, v21, s[0:3], 0 offen
s_mov_b64 exec, -1
v_cmpx_eq_u32_e32 63, v1
v_mov_b32_e32 v20, v2
v_mov_b32_e32 v21, 8
buffer_store_dword v20, v21, s[0:3], 0 offen
s_mov_b64 exec, -1
v_cmpx_eq_u32_e32 20, v1
v_mov_b32_e32 v20, v2
v_mov_b32_e32 v21, 12
buffer_store_dword v20, v21, s[0:3], 0 offen
s_mov_b64 exec, -1
v_cmpx_eq_u32_e32 0, v1
v_mov_b32_e32 v20, s8
v_mov_b32_e32 v21, 16
buffer_store_dword v20, v21, s[0:3], 0 offen
s_mov_b64 exec, -1
v_cmpx_eq_u32_e32 63, v1
v_mov_b32_e32 v20, s8
v_mov_b32_e32 v21, 20
buffer_store_dword v20, v21, s[0:3], 0 offen
s_mov_b64 exec, -1
v_cmpx_eq_u32_e32 40, v1
v_mov_b32_e32 v20, s9
v_mov_b32_e32 v21, 24
buffer_store_dword v20, v21, s[0:3], 0 offen
s_mov_b64 exec, -1
v_cmpx_eq_u32_e32 0, v1
v_mov_b32_e32 v20, s9
v_mov_b32_e32 v21, 28
buffer_store_dword v20, v21, s[0:3], 0 offen
s_endpgm
