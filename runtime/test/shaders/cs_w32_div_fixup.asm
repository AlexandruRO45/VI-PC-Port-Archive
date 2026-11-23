; regen_fixtures: this source has a translator fixture in tests/semantic/fixtures/disasm
; v_div_fixup_f16 / v_div_fixup_f64: the IEEE-754 quotient fixup of division (#382).
; Lane 0 of a wave32 group computes four f16 and two f64 results and stores eight dwords. S0 is the quotient,
; S1 the denominator and S2 the numerator. The f16 destinations are preloaded with 0xBEEF0000 and the f16 forms
; must keep their high half; f64 results are pairs.
;   out[0] = v_div_fixup_f16(0, 0, 0)      = 0xBEEFFE00  (0/0 -> the canonical negative qNaN 0xFE00)
;   out[1] = v_div_fixup_f16(0, 0, +inf)   = 0xBEEF7C00  (x/0 -> +inf with sign_out = +)
;   out[2] = v_div_fixup_f16(1.5, +inf, 1) = 0xBEEF0000  (x/inf -> +0 with sign_out = +)
;   out[3] = v_div_fixup_f16(1.5, 1, 1)    = 0xBEEF3E00  (finite: copysign(S0, sign_out) = 1.5)
;   out[4:5] = v_div_fixup_f64(0, 0, 0)    = 0xFFF80000_00000000  (canonical negative qNaN)
;   out[6:7] = v_div_fixup_f64(1.5, 1, 1)  = 0x3FF8000000000000  (finite)
; The words follow the RDNA2 ISA pseudocode, which the translator models bit-exactly; the fixture checks that the
; hardware agrees on the console's family.
; group: 32,1,1
; Run: vk_compute_test cs_w32_div_fixup.spv 1 dump -> out[0..7] = 3203399168 3203365888 3203334144 3203350016 0 4294443008 0 1073217536
; mattr: +wavefrontsize32,-wavefrontsize64
v_cmpx_eq_u32_e32 0, v0
v_mov_b32_e32 v6, 0xbeef0000
v_mov_b32_e32 v7, 0xbeef0000
v_mov_b32_e32 v8, 0xbeef0000
v_mov_b32_e32 v9, 0xbeef0000
v_mov_b32_e32 v1, 0
v_mov_b32_e32 v2, 0
v_mov_b32_e32 v3, 0
v_div_fixup_f16 v6, v1, v2, v3
v_mov_b32_e32 v3, 0x7c00
v_div_fixup_f16 v7, v1, v2, v3
v_mov_b32_e32 v1, 0x3e00
v_mov_b32_e32 v2, 0x7c00
v_mov_b32_e32 v3, 0x3c00
v_div_fixup_f16 v8, v1, v2, v3
v_mov_b32_e32 v2, 0x3c00
v_div_fixup_f16 v9, v1, v2, v3
v_mov_b32_e32 v11, 0
v_mov_b32_e32 v12, 0
v_mov_b32_e32 v13, 0
v_mov_b32_e32 v14, 0
v_mov_b32_e32 v15, 0
v_mov_b32_e32 v16, 0
v_div_fixup_f64 v[17:18], v[15:16], v[11:12], v[13:14]
v_mov_b32_e32 v12, 0x3ff00000
v_mov_b32_e32 v14, 0x3ff00000
v_mov_b32_e32 v16, 0x3ff80000
v_div_fixup_f64 v[19:20], v[15:16], v[11:12], v[13:14]
v_mov_b32_e32 v24, 0
buffer_store_dword v6, v24, s[0:3], 0 offen
v_mov_b32_e32 v24, 4
buffer_store_dword v7, v24, s[0:3], 0 offen
v_mov_b32_e32 v24, 8
buffer_store_dword v8, v24, s[0:3], 0 offen
v_mov_b32_e32 v24, 12
buffer_store_dword v9, v24, s[0:3], 0 offen
v_mov_b32_e32 v24, 16
buffer_store_dword v17, v24, s[0:3], 0 offen
v_mov_b32_e32 v24, 20
buffer_store_dword v18, v24, s[0:3], 0 offen
v_mov_b32_e32 v24, 24
buffer_store_dword v19, v24, s[0:3], 0 offen
v_mov_b32_e32 v24, 28
buffer_store_dword v20, v24, s[0:3], 0 offen
s_endpgm
