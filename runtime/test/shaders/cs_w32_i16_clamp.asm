; regen_fixtures: this source has a translator fixture in tests/semantic/fixtures/disasm
; v_add_nc_i16 / v_sub_nc_i16: the GFX10 16-bit add/sub forms, their clamp and their op_sel halves (#267).
; Lane 0 of a wave32 group computes one result per form and stores eight dwords. Every destination is
; preloaded with 0xDEAD0000 (or 0x0000DEAD for the dst-high case), because these are VOP3 partial writes
; that keep the half they do not write:
;   out[0] = v_add_nc_i16(0x7FFF, 0x0001) clamp                  = 0xDEAD7FFF  (32768 saturates to 32767)
;   out[1] = v_add_nc_i16(0x8000, 0xFFFF) clamp                  = 0xDEAD8000  (-32768 + -1 saturates to -32768)
;   out[2] = v_sub_nc_i16(0x8000, 0x0001) clamp                  = 0xDEAD8000  (-32769 saturates to -32768)
;   out[3] = v_sub_nc_i16(0x7FFF, 0xFFFF) clamp                  = 0xDEAD7FFF  (32768 saturates to 32767)
;   out[4] = v_add_nc_i16(0x7FFF, 0x0001)                        = 0xDEAD8000  (no clamp: wraps to 0x8000)
;   out[5] = v_sub_nc_i16(0x8000, 0x0001)                        = 0xDEAD7FFF  (no clamp: wraps to 0x7FFF)
;   out[6] = v_add_nc_i16(0x7FFF0000, 0x00010000) op_sel:[1,1,0] clamp = 0xDEAD7FFF  (source high halves)
;   out[7] = v_sub_nc_i16(0x7FFF0000, 0x00010000) op_sel:[1,1,1] clamp = 0x7FFEDEAD  (7FFF - 1 = 7FFE into dst's high half)
; clamp saturates the signed 16-bit sum to [-32768, 32767]; without it the low 16 bits wrap. The unsigned
; forms (v_add_nc_u16 / v_sub_nc_u16) are covered by cs_w32_int_ext.asm's v_add_u16 family.
; group: 32,1,1
; Run: vk_compute_test cs_w32_i16_clamp.spv 1 dump -> out[0..7] = 3735912447 3735912448 3735912448 3735912447 3735912448 3735912447 3735912447 2147417773
; mattr: +wavefrontsize32,-wavefrontsize64
v_mov_b32_e32 v3, 0xdead0000
v_mov_b32_e32 v4, 0xdead0000
v_mov_b32_e32 v5, 0xdead0000
v_mov_b32_e32 v6, 0xdead0000
v_mov_b32_e32 v7, 0xdead0000
v_mov_b32_e32 v8, 0xdead0000
v_mov_b32_e32 v9, 0xdead0000
v_mov_b32_e32 v10, 0xdead
v_mov_b32_e32 v1, 0x7fff
v_mov_b32_e32 v2, 1
v_add_nc_i16 v3, v1, v2 clamp
v_mov_b32_e32 v1, 0x8000
v_mov_b32_e32 v2, 0xffff
v_add_nc_i16 v4, v1, v2 clamp
v_mov_b32_e32 v2, 1
v_sub_nc_i16 v5, v1, v2 clamp
v_mov_b32_e32 v1, 0x7fff
v_mov_b32_e32 v2, 0xffff
v_sub_nc_i16 v6, v1, v2 clamp
v_mov_b32_e32 v2, 1
v_add_nc_i16 v7, v1, v2
v_mov_b32_e32 v1, 0x8000
v_sub_nc_i16 v8, v1, v2
v_mov_b32_e32 v1, 0x7fff0000
v_mov_b32_e32 v2, 0x00010000
v_add_nc_i16 v9, v1, v2 op_sel:[1,1,0] clamp
v_sub_nc_i16 v10, v1, v2 op_sel:[1,1,1] clamp
v_cmpx_eq_u32_e32 0, v0
v_mov_b32_e32 v24, 0
buffer_store_dword v3, v24, s[0:3], 0 offen
v_mov_b32_e32 v24, 4
buffer_store_dword v4, v24, s[0:3], 0 offen
v_mov_b32_e32 v24, 8
buffer_store_dword v5, v24, s[0:3], 0 offen
v_mov_b32_e32 v24, 12
buffer_store_dword v6, v24, s[0:3], 0 offen
v_mov_b32_e32 v24, 16
buffer_store_dword v7, v24, s[0:3], 0 offen
v_mov_b32_e32 v24, 20
buffer_store_dword v8, v24, s[0:3], 0 offen
v_mov_b32_e32 v24, 24
buffer_store_dword v9, v24, s[0:3], 0 offen
v_mov_b32_e32 v24, 28
buffer_store_dword v10, v24, s[0:3], 0 offen
s_endpgm
