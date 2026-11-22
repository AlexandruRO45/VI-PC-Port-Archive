; regen_fixtures: this source has a translator fixture in tests/semantic/fixtures/disasm
; v_min3/max3/med3_{i16,u16} and v_min3/max3_f16: three-operand 16-bit min/max/median (#365).
; Lane 0 of a wave32 group computes one result per op and stores eight dwords. The i16 and u16 forms get
; the same operand words 0xFFFE, 3, 1, so the signed pair proves the comparison is signed:
;   out[0] = v_min3_i16(0xFFFE, 3, 1) = 0xDEADFFFE  (-2 is the signed minimum)
;   out[1] = v_max3_i16(0xFFFE, 3, 1) = 0xDEAD0003
;   out[2] = v_med3_i16(0xFFFE, 3, 1) = 0xDEAD0001
;   out[3] = v_min3_u16(0xFFFE, 3, 1) = 0xDEAD0001  (0xFFFE = 65534 is the unsigned maximum)
;   out[4] = v_max3_u16(0xFFFE, 3, 1) = 0xDEADFFFE
;   out[5] = v_med3_u16(0xFFFE, 3, 1) = 0xDEAD0003
;   out[6] = v_min3_f16(-2.0, 1.0, 0.5) = 0xBEEFC000
;   out[7] = v_max3_f16(-2.0, 1.0, 0.5) = 0xBEEF3C00
; Every destination is preloaded (0xDEAD0000, or 0xBEEF0000 for the f16 pair) because these are VOP3
; partial writes on the low half: the high half must come back unchanged. med3_f16 is covered by
; ps_f16_med3_opsel.asm / ps_f16_med3_mods.asm.
; group: 32,1,1
; Run: vk_compute_test cs_w32_minmax3_16.spv 1 dump -> out[0..7] = 3735945214 3735879683 3735879681 3735879681 3735945214 3735879683 3203383296 3203349504
; mattr: +wavefrontsize32,-wavefrontsize64
v_mov_b32_e32 v10, 0xdead0000
v_mov_b32_e32 v11, 0xdead0000
v_mov_b32_e32 v12, 0xdead0000
v_mov_b32_e32 v13, 0xdead0000
v_mov_b32_e32 v14, 0xdead0000
v_mov_b32_e32 v15, 0xdead0000
v_mov_b32_e32 v16, 0xbeef0000
v_mov_b32_e32 v17, 0xbeef0000
v_mov_b32_e32 v4, 0xfffe
v_mov_b32_e32 v5, 3
v_mov_b32_e32 v6, 1
v_min3_i16 v10, v4, v5, v6
v_max3_i16 v11, v4, v5, v6
v_med3_i16 v12, v4, v5, v6
v_min3_u16 v13, v4, v5, v6
v_max3_u16 v14, v4, v5, v6
v_med3_u16 v15, v4, v5, v6
v_mov_b32_e32 v4, 0xc000
v_mov_b32_e32 v5, 0x3c00
v_mov_b32_e32 v6, 0x3800
v_min3_f16 v16, v4, v5, v6
v_max3_f16 v17, v4, v5, v6
v_cmpx_eq_u32_e32 0, v0
v_mov_b32_e32 v24, 0
buffer_store_dword v10, v24, s[0:3], 0 offen
v_mov_b32_e32 v24, 4
buffer_store_dword v11, v24, s[0:3], 0 offen
v_mov_b32_e32 v24, 8
buffer_store_dword v12, v24, s[0:3], 0 offen
v_mov_b32_e32 v24, 12
buffer_store_dword v13, v24, s[0:3], 0 offen
v_mov_b32_e32 v24, 16
buffer_store_dword v14, v24, s[0:3], 0 offen
v_mov_b32_e32 v24, 20
buffer_store_dword v15, v24, s[0:3], 0 offen
v_mov_b32_e32 v24, 24
buffer_store_dword v16, v24, s[0:3], 0 offen
v_mov_b32_e32 v24, 28
buffer_store_dword v17, v24, s[0:3], 0 offen
s_endpgm
