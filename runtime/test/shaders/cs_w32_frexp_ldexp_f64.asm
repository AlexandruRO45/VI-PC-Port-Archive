; regen_fixtures: this source has a translator fixture in tests/semantic/fixtures/disasm
; Binary64 frexp (v_frexp_mant_f64, v_frexp_exp_i32_f64) and load-exponent (v_ldexp_f64) under the game's
; inferred MODE (IEEE=0, DX10_CLAMP=1, denorm_mode_16_64=3; docs/design.md).
; Lane 0 of a wave32 workgroup computes and stores eight dwords:
;   out[0] = low dword of v_frexp_mant_f64(|-0x00000001_00000001|) XOR
;            low dword of v_frexp_mant_f64(-3.5) mul:2 (subnormal bit 32 + bit 0 shifts left 20 to
;            0x3FE00000_00100000; 3.5 -> -0.875 = 0xBTEC0000_20000000 with mul:2 ignored under denorm16_64=3) = 0x20100000
;   out[1] = high dword of v_frexp_mant_f64(|-0x00000001_00000001|) (0x3FE00000) XOR
;            high dword of v_frexp_mant_f64(-3.5) mul:2 (0xBFEC0000)                                         = 0x800C0000
;   out[2] = (v_frexp_exp_i32_f64(3.5) << 16) | v_frexp_exp_i32_f64(|-0x00000001_00000001|)
;            (2 << 16) | (-1041 & 0xFFFFFFFF = 0xFFFFFBEF)                                                   = 0xFFFFFBEF
;   out[3] = (low XOR high dword of v_frexp_mant_f64(-sNaN 0x7FF00000_12345678)) OR
;            v_frexp_exp_i32_f64(sNaN) (sNaN payload kept under IEEE=0 -> 0x12345678 ^ 0xFFF00000 | 0)       = 0xEDC45678
;   out[4] = low dword of v_ldexp_f64(0x00080000_00000001, 1024) (subnormal bit 51 + bit 0 normalizes by
;            lsh=1 to significand 0x00100000_00000002, ef=0, e2=1024 -> 0x40000000_00000002)                 = 0x00000002
;   out[5] = high dword of v_ldexp_f64(0x00080000_00000001, 1024)                                            = 0x40000000
;   out[6] = low dword of v_ldexp_f64(0x3FF00001_80000000, -1042) (sh=20 exact half-ULP tie on odd quotient
;            0x1_000017FF -> rounds to even 0x1_00001800) OR low dword of v_ldexp_f64(0xBFFFFFFF_FFFFFFFF,
;            -1023) (sh=1 carry into -min_norm 0x80100000_00000000)                                           = 0x00001800
;   out[7] = (high dword of v_ldexp_f64(0x3FF00001_80000000, -1042) (1) XOR
;            high dword of v_ldexp_f64(0xBFFFFFFF_FFFFFFFF, -1023) (0x80100000)) OR
;            v_ldexp_f64(-3.0, 1) clamp (0) OR low dword of v_ldexp_f64(|-3.0|, 1) clamp (0) OR
;            (high dword of v_ldexp_f64(|-3.0|, 1) clamp (0x3FF00000) XOR
;            high dword of v_ldexp_f64(0x40080000, 1) (32-bit literal in bits [63:32] -> 6.0 = 0x40180000))  = 0xFFF80001
; group: 32,1,1
; Run: vk_compute_test cs_w32_frexp_ldexp_f64.spv 1 dump -> out[0..7] = 537919488 2148270080 4294966255 3989067384 2 1073741824 6144 4294443009
; mattr: +wavefrontsize32,-wavefrontsize64
v_cmpx_eq_u32_e32 0, v0
v_mov_b32_e32 v2, 1
v_mov_b32_e32 v3, 0x80000001
v_frexp_mant_f64_e64 v[4:5], |v[2:3]|
v_frexp_exp_i32_f64_e64 v6, |v[2:3]|
v_mov_b32_e32 v2, 0x20000000
v_mov_b32_e32 v3, 0x400c0000
v_frexp_mant_f64_e64 v[7:8], -v[2:3] mul:2
v_frexp_exp_i32_f64_e32 v9, v[2:3]
v_xor_b32_e32 v4, v4, v7
v_xor_b32_e32 v5, v5, v8
v_lshl_or_b32 v6, v9, 16, v6
v_mov_b32_e32 v2, 0x12345678
v_mov_b32_e32 v3, 0x7ff00000
v_frexp_mant_f64_e64 v[10:11], -v[2:3]
v_frexp_exp_i32_f64_e32 v12, v[2:3]
v_xor_b32_e32 v10, v10, v11
v_or_b32_e32 v10, v10, v12
v_mov_b32_e32 v2, 1
v_mov_b32_e32 v3, 0x80000
v_mov_b32_e32 v13, 0x400
v_ldexp_f64 v[14:15], v[2:3], v13
v_mov_b32_e32 v2, 0x80000000
v_mov_b32_e32 v3, 0x3ff00001
v_mov_b32_e32 v13, 0xfffffbee
v_ldexp_f64 v[16:17], v[2:3], v13
v_mov_b32_e32 v2, -1
v_mov_b32_e32 v3, 0xbfffffff
v_mov_b32_e32 v13, 0xfffffc01
v_ldexp_f64 v[18:19], v[2:3], v13
v_or_b32_e32 v16, v16, v18
v_xor_b32_e32 v17, v17, v19
v_mov_b32_e32 v2, 0
v_mov_b32_e32 v3, 0x40080000
v_ldexp_f64 v[20:21], -v[2:3], 1 clamp
v_ldexp_f64 v[22:23], |v[2:3]|, 1 clamp
v_ldexp_f64 v[24:25], 0x40080000, 1
v_or_b32_e32 v20, v20, v21
v_or_b32_e32 v20, v20, v22
v_or_b32_e32 v20, v20, v24
v_xor_b32_e32 v21, v23, v25
v_or_b32_e32 v20, v20, v21
v_or_b32_e32 v17, v17, v20
v_mov_b32_e32 v26, 0
buffer_store_dword v4, v26, s[0:3], 0 offen
v_mov_b32_e32 v26, 4
buffer_store_dword v5, v26, s[0:3], 0 offen
v_mov_b32_e32 v26, 8
buffer_store_dword v6, v26, s[0:3], 0 offen
v_mov_b32_e32 v26, 12
buffer_store_dword v10, v26, s[0:3], 0 offen
v_mov_b32_e32 v26, 16
buffer_store_dword v14, v26, s[0:3], 0 offen
v_mov_b32_e32 v26, 20
buffer_store_dword v15, v26, s[0:3], 0 offen
v_mov_b32_e32 v26, 24
buffer_store_dword v16, v26, s[0:3], 0 offen
v_mov_b32_e32 v26, 28
buffer_store_dword v17, v26, s[0:3], 0 offen
s_endpgm
