; SOPP forms with a plain equivalent. s_cbranch_cdbgsys / cdbguser / cdbgsys_or_user / cdbgsys_and_user branch on
; the debugger-present bits, which are clear when no shader debugger is attached, so they are never taken and the
; v_mov after each runs. s_wait_idle waits for outstanding operations: no register or memory effect.
; s_endpgm_ordered_ps_done ends the program (and signals the end of an ordered pixel-shader section), so the stores
; after it never run. Until this change all six raised Unsupported.
; Lane 0 stores out[0..7] = 0x11 0x22 0x33 0x44 5 6 7 8; had a branch been taken its word would stay 1, 2, 3 or 4, and
; had the code after s_endpgm_ordered_ps_done run, out[0] would be 0xbad.
; group: 32,1,1
; Run: vk_compute_test cs_w32_sopp_misc.spv 1 dump -> out[0..7] = 17 34 51 68 5 6 7 8
; mattr: +wavefrontsize32,-wavefrontsize64
v_mov_b32_e32 v10, 1
v_mov_b32_e32 v11, 2
v_mov_b32_e32 v12, 3
v_mov_b32_e32 v13, 4
v_mov_b32_e32 v14, 5
v_mov_b32_e32 v15, 6
v_mov_b32_e32 v16, 7
v_mov_b32_e32 v17, 8
s_wait_idle
s_cbranch_cdbgsys 1
v_mov_b32_e32 v10, 0x11
s_cbranch_cdbguser 1
v_mov_b32_e32 v11, 0x22
s_cbranch_cdbgsys_or_user 1
v_mov_b32_e32 v12, 0x33
s_cbranch_cdbgsys_and_user 1
v_mov_b32_e32 v13, 0x44
v_cmpx_eq_u32_e32 0, v0
v_mov_b32_e32 v22, 0
buffer_store_dwordx4 v[10:13], v22, s[0:3], 0 offen
v_mov_b32_e32 v22, 16
buffer_store_dwordx4 v[14:17], v22, s[0:3], 0 offen
s_endpgm_ordered_ps_done
v_mov_b32_e32 v10, 0xbad
v_mov_b32_e32 v22, 0
buffer_store_dword v10, v22, s[0:3], 0 offen
s_endpgm
