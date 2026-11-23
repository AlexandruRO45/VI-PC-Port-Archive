; Translator contract row s_round_mode (docs/design.md): SIMM16[3:0] writes MODE.FP_ROUND, whose [3:2] field picks
; the f16/f64 rounding (0 nearest even, 3 toward zero; 1 and 2 are the directed roundings and are refused) while
; [1:0] must stay 0 (FP32 nearest even). The write reaches the same per-invocation state as
; s_setreg_imm32_b32 hwreg(HW_REG_MODE, 2, 2).
;
; Every result below is the f16 word the mode selects, on a value where nearest even and toward zero differ. v0 is
; 1 + 2^-10 + 2^-11 (0x3f803000): the f16 tie between 0x3c01 and 0x3c02, so nearest even takes the even one
; (0x3c02) and toward zero truncates to 0x3c01. v1 is that value negated (0xbf803000), where toward zero truncates
; the magnitude to 0xbc01. v4 + v5 is the same tie as f16 operands (1.0 = 0x3c00 plus 1.5 * 2^-10 = 0x1600), so the
; sum follows the same two words, and v6 + v7 is its negative. MODE.FP_ROUND[3:2] = 3 toggles the f16 result of both
; a conversion and an f16 ALU op; no MODE write at all is nearest even.
;
;   out[0]                     v_cvt_f16_f32 v0         nearest    0x00003c02
;   out[1] s_round_mode 12     v_cvt_f16_f32 v0         toward zero 0x00003c01
;   out[2] s_setreg_imm32_b32 hwreg(HW_REG_MODE, 2, 2), 3, v_cvt_f16_f32 v0
;                                                       toward zero 0x00003c01
;   out[3] s_setreg_imm32_b32 hwreg(HW_REG_MODE, 2, 2), 0, v_cvt_f16_f32 v0
;                                                       nearest    0x00003c02
;   out[4] s_round_mode 12     v_cvt_f16_f32 v1         toward zero 0x0000bc01
;   out[5] s_round_mode 12     v_add_f16 v4, v5         toward zero 0x00003c01
;   out[6] s_round_mode 0      v_add_f16 v4, v5         nearest    0x00003c02
;   out[7] s_round_mode 12     v_add_f16 v6, v7         toward zero 0x0000bc01
; Run: vk_compute_test cs_round_mode.spv 1 dump
; out[0..7] = 00003c02 00003c01 00003c01 00003c02 0000bc01 00003c01 00003c02 0000bc01 (hex)
; group: 1,1,1
; rsrc2: 0x8090
v_mov_b32_e32 v0, 0x3f803000
v_mov_b32_e32 v1, 0xbf803000
v_mov_b32_e32 v2, 0
v_cvt_f16_f32_e32 v3, v0
v_and_b32_e32 v3, 0xffff, v3
buffer_store_dword v3, v2, s[0:3], 0 offen
s_round_mode 12
v_cvt_f16_f32_e32 v8, v0
v_and_b32_e32 v8, 0xffff, v8
buffer_store_dword v8, v2, s[0:3], 0 offen offset:4
s_setreg_imm32_b32 hwreg(HW_REG_MODE, 2, 2), 3
v_cvt_f16_f32_e32 v9, v0
v_and_b32_e32 v9, 0xffff, v9
buffer_store_dword v9, v2, s[0:3], 0 offen offset:8
s_setreg_imm32_b32 hwreg(HW_REG_MODE, 2, 2), 0
v_cvt_f16_f32_e32 v10, v0
v_and_b32_e32 v10, 0xffff, v10
buffer_store_dword v10, v2, s[0:3], 0 offen offset:12
s_round_mode 12
v_cvt_f16_f32_e32 v11, v1
v_and_b32_e32 v11, 0xffff, v11
buffer_store_dword v11, v2, s[0:3], 0 offen offset:16
v_mov_b32_e32 v4, 0x3c00
v_mov_b32_e32 v5, 0x1600
v_add_f16_e32 v12, v4, v5
v_and_b32_e32 v12, 0xffff, v12
buffer_store_dword v12, v2, s[0:3], 0 offen offset:20
s_round_mode 0
v_add_f16_e32 v13, v4, v5
v_and_b32_e32 v13, 0xffff, v13
buffer_store_dword v13, v2, s[0:3], 0 offen offset:24
s_round_mode 12
v_mov_b32_e32 v6, 0xbc00
v_mov_b32_e32 v7, 0x9600
v_add_f16_e32 v14, v6, v7
v_and_b32_e32 v14, 0xffff, v14
buffer_store_dword v14, v2, s[0:3], 0 offen offset:28
s_endpgm
