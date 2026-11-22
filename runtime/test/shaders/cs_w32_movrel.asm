; v_movrels_b32, v_movreld_b32, v_movrelsd_b32: VGPR indexing by the whole of M0 (RDNA2 ISA, VOP1 opcodes 66, 67, 68):
;   v_movrels_b32:  D = VGPR[S0 + M0]            v_movreld_b32:  VGPR[D + M0] = S0
;   v_movrelsd_b32: VGPR[D + M0] = VGPR[S0 + M0]   (unlike v_movrelsd_2_b32, one M0 offset for both sides)
; Initial: v10..v15 = 0xA0..0xA5, v30 = 0x30, v32 = 0x32, v41 = 0x41, v58 = 0x58, v92 = 0x92, v98 = 0x98, v131 = 0x131.
; 1. M0 = 0:  v_movrels_b32 v20, v10    -> v20 = v10 = 0xA0.
; 2. M0 = 3:  v_movrels_b32 v21, v10    -> v21 = v13 = 0xA3.
; 3. M0 = 82: v_movrels_b32 v22, v10    -> v22 = v92 = 0x92 (a larger index).
; 4. M0 = 2:  v_movreld_b32 v30, v14    -> v32 = v14 = 0xA4; v30 is not written and keeps 0x30.
; 5. M0 = 100: v_movreld_b32 v31, v15   -> v131 = v15 = 0xA5.
; 6. M0 = 1:  v_movrelsd_b32 v40, v10   -> v41 = v11 = 0xA1.
; 7. M0 = 48: v_movrelsd_b32 v50, v10   -> v98 = v58 = 0x58.
; An op that ignored M0 would write v30 (4), v31 (5), v40 (6), v50 (7) instead, and read v10 for 2 and 3.
; Lane 0 stores: out[0..7] = v20 v21 v22 v32 v30 v131 v41 v98 = 0xA0 0xA3 0x92 0xA4 0x30 0xA5 0xA1 0x58.
; group: 32,1,1
; Run: vk_compute_test cs_w32_movrel.spv 1 dump -> out[0..7] = 160 163 146 164 48 165 161 88
; mattr: +wavefrontsize32,-wavefrontsize64
v_mov_b32_e32 v10, 0xa0
v_mov_b32_e32 v11, 0xa1
v_mov_b32_e32 v12, 0xa2
v_mov_b32_e32 v13, 0xa3
v_mov_b32_e32 v14, 0xa4
v_mov_b32_e32 v15, 0xa5
v_mov_b32_e32 v30, 0x30
v_mov_b32_e32 v32, 0x32
v_mov_b32_e32 v41, 0x41
v_mov_b32_e32 v58, 0x58
v_mov_b32_e32 v92, 0x92
v_mov_b32_e32 v98, 0x98
v_mov_b32_e32 v131, 0x131
s_mov_b32 m0, 0
v_movrels_b32 v20, v10
s_mov_b32 m0, 3
v_movrels_b32 v21, v10
s_mov_b32 m0, 82
v_movrels_b32 v22, v10
s_mov_b32 m0, 2
v_movreld_b32 v30, v14
s_mov_b32 m0, 100
v_movreld_b32 v31, v15
s_mov_b32 m0, 1
v_movrelsd_b32 v40, v10
s_mov_b32 m0, 48
v_movrelsd_b32 v50, v10
v_cmpx_eq_u32_e32 0, v0
v_mov_b32_e32 v200, 0
buffer_store_dword v20, v200, s[0:3], 0 offen
v_mov_b32_e32 v200, 4
buffer_store_dword v21, v200, s[0:3], 0 offen
v_mov_b32_e32 v200, 8
buffer_store_dword v22, v200, s[0:3], 0 offen
v_mov_b32_e32 v200, 12
buffer_store_dword v32, v200, s[0:3], 0 offen
v_mov_b32_e32 v200, 16
buffer_store_dword v30, v200, s[0:3], 0 offen
v_mov_b32_e32 v200, 20
buffer_store_dword v131, v200, s[0:3], 0 offen
v_mov_b32_e32 v200, 24
buffer_store_dword v41, v200, s[0:3], 0 offen
v_mov_b32_e32 v200, 28
buffer_store_dword v98, v200, s[0:3], 0 offen
s_endpgm
