; EGEN-F64-DIV-SCALE: v_div_scale_f64 D, vcc_lo, S0, S1, S2 (S1 denominator, S2 numerator, S0 one of them).
; Lane 0 runs 13 cases; case i stores D.lo, D.hi and vcc as 0/1 (v_cndmask) at out[3i..3i+2]. out[39..63] stay
; 0xeeeeeeee. Needs no shaderFloat64: the translation is 32-bit integer only.
; Rule (docs/design.md, measured on RDNA2 gfx1036 with tools/hw_oracle; the ISA pseudocode differs): e1, e2 are the
; exponent fields of S1, S2 (bits 62:52; 0 for zero/denormal, 2047 for inf/NaN). First match wins:
;   e2 - e1 >= 768 -> vcc=1, D = S0 == S1 ? S0*2^128 : S0     e1 == 0 -> D = S0*2^128
;   e1 >= 2045 and e1 - e2 >= 768 -> vcc=1, D = S0 == S1 ? S0*2^-128 : S0     e1 >= 2045 -> D = S0*2^-128
;   e1 - e2 >= 768 -> vcc=1, D = S0 == S1 ? S0 : S0*2^128     e2 <= 53 -> D = S0*2^128     else D = S0
;   S1 or S2 zero -> D = 0xfff8000000000000 (vcc as above). clamp: NaN -> +0. omod is ignored.
; x2^128 of a normal number adds 0x080 to the exponent field: hi word + 0x08000000, lo word unchanged.
;  0 den: S1 = 0x3ff00000_00000005 (e1 0x3ff), S2 = 0x71f00000_00000000 (e2 0x71f): 0x71f - 0x3ff = 800 >= 768,
;         S0 = S1 -> D = 0x47f00000_00000005, vcc 1                        out[0..2]   = 00000005 47f00000 1
;  1 num: same S1, S2, S0 = S2 != S1 -> D = S2 = 0x71f00000_00000000, vcc 1 out[3..5]   = 00000000 71f00000 1
;  2 num: S1 as 0, S2 = 0x0df00000_00000003 (e2 0x0df): e1 - e2 = 0x3ff - 0x0df = 800 >= 768, e1 < 2045,
;         S0 = S2 != S1 -> D = S2*2^128 = 0x15f00000_00000003, vcc 1      out[6..8]   = 00000003 15f00000 1
;  3 den: same S1, S2, S0 = S1 -> D = S1, vcc 1                            out[9..11]  = 00000005 3ff00000 1
;  4 den: S1 = 2^1023 = 0x7fe00000_0 (e1 0x7fe = 2046), S2 = 1.0 (e2 0x3ff): e1 >= 2045 and e1 - e2 = 1023,
;         S0 = S1 -> D = 2^895: field 0x7fe - 0x080 = 0x77e -> 0x77e00000_0, vcc 1
;                                                                          out[12..14] = 00000000 77e00000 1
;  5 num: S1 = 2^1023, S2 = 0x7f800000_00000009 (e2 0x7f8): e1 >= 2045, e1 - e2 = 6 -> D = S2*2^-128,
;         field 0x7f8 - 0x080 = 0x778 -> 0x77800000_00000009, vcc 0        out[15..17] = 00000009 77800000 0
;  6 den: S1 = 0x00000000_00000007 (denormal, 7 * 2^-1074, e1 0), S2 = 0x25800000_0 (e2 0x258 = 600 < 768):
;         D = S1*2^128 = 7 * 2^-946 = 1.75 * 2^-944: field -944 + 1023 = 79 = 0x04f, mantissa 0.75 = 0xc0000_0
;         -> 0x04fc0000_00000000, vcc 0                                    out[18..20] = 00000000 04fc0000 0
;  7 num: S1 = 2^-1022 = 0x00100000_0 (e1 1), S2 = 0x01400000_00000001 (e2 0x014 = 20 <= 53), no earlier rule
;         (e2 - e1 = 19): D = S2*2^128, field 0x094 -> 0x09400000_00000001, vcc 0
;                                                                          out[21..23] = 00000001 09400000 0
;  8 den: S1 = +0, S2 = 1.0: e2 - e1 = 1023 >= 768 -> vcc 1; S1 zero -> D = 0xfff80000_00000000
;                                                                          out[24..26] = 00000000 fff80000 1
;  9 num: S1 = 1.0, S2 = signalling NaN 0x7ff00000_00000001 (e2 2047): e2 - e1 = 1024 -> vcc 1, S0 = S2 != S1
;         -> D = S2 with its payload, not quieted                          out[27..29] = 00000001 7ff00000 1
; 10 den: S0 = -S1 (neg on src0), S1, S2 as case 0: vcc 1, S0 != S1 -> D = -S1 = 0xbff00000_00000005
;                                                                          out[30..32] = 00000005 bff00000 1
; 11 den clamp: S1 = +0, S2 = 1.0 as case 8: vcc 1, D = NaN, clamp -> +0  out[33..35] = 00000000 00000000 1
; 12 den mul:2: S1 = 0.25 (0x3fd00000_0), S2 = 0.5 (0x3fe00000_0): no rule -> D = S1 = 0.25, vcc 0; the
;         ignored omod leaves it (applied it would be 0.5 = 0x3fe00000)  out[36..38] = 00000000 3fd00000 0
; Run: vk_compute_test cs_div_scale_f64.spv 1 dump with VKC_OUT_WORDS=64 and VKC_DUMP (check.py div_scale_f64_test)
v_cmpx_eq_u32_e32 0, v0
v_mov_b32_e32 v2, 0x00000005
v_mov_b32_e32 v3, 0x3ff00000
v_mov_b32_e32 v4, 0x00000000
v_mov_b32_e32 v5, 0x71f00000
v_div_scale_f64 v[8:9], vcc_lo, v[2:3], v[2:3], v[4:5]
v_cndmask_b32_e64 v10, 0, 1, vcc_lo
buffer_store_dword v8, off, s[0:3], 0 offset:0
buffer_store_dword v9, off, s[0:3], 0 offset:4
buffer_store_dword v10, off, s[0:3], 0 offset:8
v_mov_b32_e32 v2, 0x00000005
v_mov_b32_e32 v3, 0x3ff00000
v_mov_b32_e32 v4, 0x00000000
v_mov_b32_e32 v5, 0x71f00000
v_div_scale_f64 v[8:9], vcc_lo, v[4:5], v[2:3], v[4:5]
v_cndmask_b32_e64 v10, 0, 1, vcc_lo
buffer_store_dword v8, off, s[0:3], 0 offset:12
buffer_store_dword v9, off, s[0:3], 0 offset:16
buffer_store_dword v10, off, s[0:3], 0 offset:20
v_mov_b32_e32 v2, 0x00000005
v_mov_b32_e32 v3, 0x3ff00000
v_mov_b32_e32 v4, 0x00000003
v_mov_b32_e32 v5, 0x0df00000
v_div_scale_f64 v[8:9], vcc_lo, v[4:5], v[2:3], v[4:5]
v_cndmask_b32_e64 v10, 0, 1, vcc_lo
buffer_store_dword v8, off, s[0:3], 0 offset:24
buffer_store_dword v9, off, s[0:3], 0 offset:28
buffer_store_dword v10, off, s[0:3], 0 offset:32
v_mov_b32_e32 v2, 0x00000005
v_mov_b32_e32 v3, 0x3ff00000
v_mov_b32_e32 v4, 0x00000003
v_mov_b32_e32 v5, 0x0df00000
v_div_scale_f64 v[8:9], vcc_lo, v[2:3], v[2:3], v[4:5]
v_cndmask_b32_e64 v10, 0, 1, vcc_lo
buffer_store_dword v8, off, s[0:3], 0 offset:36
buffer_store_dword v9, off, s[0:3], 0 offset:40
buffer_store_dword v10, off, s[0:3], 0 offset:44
v_mov_b32_e32 v2, 0x00000000
v_mov_b32_e32 v3, 0x7fe00000
v_mov_b32_e32 v4, 0x00000000
v_mov_b32_e32 v5, 0x3ff00000
v_div_scale_f64 v[8:9], vcc_lo, v[2:3], v[2:3], v[4:5]
v_cndmask_b32_e64 v10, 0, 1, vcc_lo
buffer_store_dword v8, off, s[0:3], 0 offset:48
buffer_store_dword v9, off, s[0:3], 0 offset:52
buffer_store_dword v10, off, s[0:3], 0 offset:56
v_mov_b32_e32 v2, 0x00000000
v_mov_b32_e32 v3, 0x7fe00000
v_mov_b32_e32 v4, 0x00000009
v_mov_b32_e32 v5, 0x7f800000
v_div_scale_f64 v[8:9], vcc_lo, v[4:5], v[2:3], v[4:5]
v_cndmask_b32_e64 v10, 0, 1, vcc_lo
buffer_store_dword v8, off, s[0:3], 0 offset:60
buffer_store_dword v9, off, s[0:3], 0 offset:64
buffer_store_dword v10, off, s[0:3], 0 offset:68
v_mov_b32_e32 v2, 0x00000007
v_mov_b32_e32 v3, 0x00000000
v_mov_b32_e32 v4, 0x00000000
v_mov_b32_e32 v5, 0x25800000
v_div_scale_f64 v[8:9], vcc_lo, v[2:3], v[2:3], v[4:5]
v_cndmask_b32_e64 v10, 0, 1, vcc_lo
buffer_store_dword v8, off, s[0:3], 0 offset:72
buffer_store_dword v9, off, s[0:3], 0 offset:76
buffer_store_dword v10, off, s[0:3], 0 offset:80
v_mov_b32_e32 v2, 0x00000000
v_mov_b32_e32 v3, 0x00100000
v_mov_b32_e32 v4, 0x00000001
v_mov_b32_e32 v5, 0x01400000
v_div_scale_f64 v[8:9], vcc_lo, v[4:5], v[2:3], v[4:5]
v_cndmask_b32_e64 v10, 0, 1, vcc_lo
buffer_store_dword v8, off, s[0:3], 0 offset:84
buffer_store_dword v9, off, s[0:3], 0 offset:88
buffer_store_dword v10, off, s[0:3], 0 offset:92
v_mov_b32_e32 v2, 0x00000000
v_mov_b32_e32 v3, 0x00000000
v_mov_b32_e32 v4, 0x00000000
v_mov_b32_e32 v5, 0x3ff00000
v_div_scale_f64 v[8:9], vcc_lo, v[2:3], v[2:3], v[4:5]
v_cndmask_b32_e64 v10, 0, 1, vcc_lo
buffer_store_dword v8, off, s[0:3], 0 offset:96
buffer_store_dword v9, off, s[0:3], 0 offset:100
buffer_store_dword v10, off, s[0:3], 0 offset:104
v_mov_b32_e32 v2, 0x00000000
v_mov_b32_e32 v3, 0x3ff00000
v_mov_b32_e32 v4, 0x00000001
v_mov_b32_e32 v5, 0x7ff00000
v_div_scale_f64 v[8:9], vcc_lo, v[4:5], v[2:3], v[4:5]
v_cndmask_b32_e64 v10, 0, 1, vcc_lo
buffer_store_dword v8, off, s[0:3], 0 offset:108
buffer_store_dword v9, off, s[0:3], 0 offset:112
buffer_store_dword v10, off, s[0:3], 0 offset:116
v_mov_b32_e32 v2, 0x00000005
v_mov_b32_e32 v3, 0x3ff00000
v_mov_b32_e32 v4, 0x00000000
v_mov_b32_e32 v5, 0x71f00000
v_div_scale_f64 v[8:9], vcc_lo, -v[2:3], v[2:3], v[4:5]
v_cndmask_b32_e64 v10, 0, 1, vcc_lo
buffer_store_dword v8, off, s[0:3], 0 offset:120
buffer_store_dword v9, off, s[0:3], 0 offset:124
buffer_store_dword v10, off, s[0:3], 0 offset:128
v_mov_b32_e32 v2, 0x00000000
v_mov_b32_e32 v3, 0x00000000
v_mov_b32_e32 v4, 0x00000000
v_mov_b32_e32 v5, 0x3ff00000
v_div_scale_f64 v[8:9], vcc_lo, v[2:3], v[2:3], v[4:5] clamp
v_cndmask_b32_e64 v10, 0, 1, vcc_lo
buffer_store_dword v8, off, s[0:3], 0 offset:132
buffer_store_dword v9, off, s[0:3], 0 offset:136
buffer_store_dword v10, off, s[0:3], 0 offset:140
v_mov_b32_e32 v2, 0x00000000
v_mov_b32_e32 v3, 0x3fd00000
v_mov_b32_e32 v4, 0x00000000
v_mov_b32_e32 v5, 0x3fe00000
v_div_scale_f64 v[8:9], vcc_lo, v[2:3], v[2:3], v[4:5] mul:2
v_cndmask_b32_e64 v10, 0, 1, vcc_lo
buffer_store_dword v8, off, s[0:3], 0 offset:144
buffer_store_dword v9, off, s[0:3], 0 offset:148
buffer_store_dword v10, off, s[0:3], 0 offset:152
s_waitcnt vmcnt(0)
s_endpgm
