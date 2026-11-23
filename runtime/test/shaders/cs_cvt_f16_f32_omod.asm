; VOP3 output modifiers on v_cvt_f16_f32 under MODE.FP_ROUND[3:2] (docs/design.md "v_cvt_f16_f32 output
; modifiers"): the value is rounded to f16 first, then mul:2 / mul:4 / div:2 scale the f16 bits, then clamp. An omod
; overflow of the scaled value is +-inf in nearest even mode and +-65504 (0x7bff / 0xfbff) under MODE toward zero,
; the clause that row still marked unmeasured. s[0:3] is the output V#. One thread stores eight words: out[0..4]
; under MODE toward zero (FP_ROUND = 3), out[5..7] back under nearest even.
;   out[0] 32768.0  mul:2 toward zero   0x7bff (32768 is f16 0x7800; doubling it overflows)
;   out[1] -32768.0 mul:2 toward zero   0xfbff
;   out[2] 16384.0  mul:4 toward zero   0x7bff
;   out[3] 65520.0  div:2 toward zero   0x77ff (rounds to 65504 first, then halves to 32752)
;   out[4] -65520.0 div:2 toward zero   0xf7ff
;   out[5] 32768.0  mul:2 nearest even 0x7c00
;   out[6] -32768.0 mul:2 nearest even 0xfc00
;   out[7] 65520.0  div:2 nearest even 0x7c00 (rounds to inf first, div:2 passes it through)
; These eight words are the reference tools/cvt_f16_omod_probe.py's self-test and
; tests/semantic/test_cvt_f16_omod.py assert through the translator on the CPU.
; Run: vk_compute_test cs_cvt_f16_f32_omod.spv 1 dump
; out[0..7] = 31743 64511 31743 30719 63487 31744 64512 31744 (hex 00007bff 0000fbff 00007bff 000077ff 0000f7ff
; 00007c00 0000fc00 00007c00)
; group: 1,1,1
; rsrc2: 0x8088
s_setreg_imm32_b32 hwreg(HW_REG_MODE, 2, 2), 3
v_mov_b32_e32 v20, 0
v_mov_b32_e32 v0, 0x47000000
v_cvt_f16_f32_e64 v1, v0 mul:2
v_and_b32_e32 v1, 0xffff, v1
buffer_store_dword v1, v20, s[0:3], 0 offen
v_mov_b32_e32 v0, 0xc7000000
v_cvt_f16_f32_e64 v1, v0 mul:2
v_and_b32_e32 v1, 0xffff, v1
buffer_store_dword v1, v20, s[0:3], 0 offen offset:4
v_mov_b32_e32 v0, 0x46800000
v_cvt_f16_f32_e64 v1, v0 mul:4
v_and_b32_e32 v1, 0xffff, v1
buffer_store_dword v1, v20, s[0:3], 0 offen offset:8
v_mov_b32_e32 v0, 0x477ff000
v_cvt_f16_f32_e64 v1, v0 div:2
v_and_b32_e32 v1, 0xffff, v1
buffer_store_dword v1, v20, s[0:3], 0 offen offset:12
v_mov_b32_e32 v0, 0xc77ff000
v_cvt_f16_f32_e64 v1, v0 div:2
v_and_b32_e32 v1, 0xffff, v1
buffer_store_dword v1, v20, s[0:3], 0 offen offset:16
s_setreg_imm32_b32 hwreg(HW_REG_MODE, 2, 2), 0
v_mov_b32_e32 v0, 0x47000000
v_cvt_f16_f32_e64 v1, v0 mul:2
v_and_b32_e32 v1, 0xffff, v1
buffer_store_dword v1, v20, s[0:3], 0 offen offset:20
v_mov_b32_e32 v0, 0xc7000000
v_cvt_f16_f32_e64 v1, v0 mul:2
v_and_b32_e32 v1, 0xffff, v1
buffer_store_dword v1, v20, s[0:3], 0 offen offset:24
v_mov_b32_e32 v0, 0x477ff000
v_cvt_f16_f32_e64 v1, v0 div:2
v_and_b32_e32 v1, 0xffff, v1
buffer_store_dword v1, v20, s[0:3], 0 offen offset:28
s_endpgm
