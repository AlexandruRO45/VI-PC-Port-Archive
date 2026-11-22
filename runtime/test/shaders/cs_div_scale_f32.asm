; EGEN-F32-DIV-SCALE: v_div_scale_f32 D, vcc_lo, S0, S1, S2 (S1 denominator, S2 numerator, S0 one of them).
; Lane 0 runs 16 cases; case i stores D and vcc as 0/1 (v_cndmask) at out[2i], out[2i+1]. out[32..63] stay 0xeeeeeeee.
; Rule (docs/design.md, measured on RDNA2 gfx1036 with tools/hw_oracle in the game's MODE: IEEE off, f32 denormals flushed;
; the ISA pseudocode differs): a denormal operand reads as a zero of its sign; e1, e2 are the exponent fields
; (bits 30:23) of S1, S2 (0 for zero, 255 for inf/NaN). First match wins:
;   e2 - e1 >= 96 -> vcc=1, D = S0 == S1 ? S0*2^64 : S0     e1 == 0 -> D = S0*2^64
;   e1 >= 253 and e1 - e2 >= 96 -> vcc=1, D = S0 == S1 ? S0*2^-64 : S0     e1 >= 253 -> D = S0*2^-64
;   e1 - e2 >= 96 -> vcc=1, D = S0 == S1 ? S0 : S0*2^64     e2 <= 24 -> D = S0*2^64     else D = S0
;   S1 or S2 zero -> D = 0xffc00000 (vcc as above). x2^-64 gives +-0 at exponent field 64 and +-inf below.
;   omod scales the result (IEEE off), then clamp: NaN -> +0. x2^+-64 of a normal number adds/subtracts 0x40 in the exponent field
;   (0x20000000 in the word).
;  0 den: S1 = 0x3f800005 (e1 127), S2 = 0x71800000 (e2 227): 100 >= 96, S0 = S1 -> 0x5f800005, vcc 1
;  1 num: same, S0 = S2 != S1 -> D = S2 = 0x71800000, vcc 1
;  2 num: S1 as 0, S2 = 0x0d800003 (e2 27): e1 - e2 = 100 >= 96, S0 != S1 -> S2*2^64 = 0x2d800003, vcc 1
;  3 den: same, S0 = S1 -> D = S1 = 0x3f800005, vcc 1
;  4 den: S1 = 0x7f000000 (e1 254), S2 = 1.0 (e2 127): e1 >= 253, e1 - e2 = 127, S0 = S1 -> S1*2^-64:
;         field 254 - 64 = 190 = 0xbe -> 0x5f000000, vcc 1
;  5 num: S1 = 0x7f000000, S2 = 0x7d000009 (e2 250): e1 >= 253, e1 - e2 = 4 -> S2*2^-64 = 0x5d000009, vcc 0
;  6 num: S1 = 1.0, S2 = 0x00000003 (denormal, reads as +0): e1 - e2 = 127 -> vcc 1; S2 zero -> 0xffc00000
;  7 num: S1 = 0x00800000 (e1 1), S2 = 0x0a000001 (e2 20 <= 24, e2 - e1 = 19): S2*2^64, field 84 = 0x54
;         -> 0x2a000001, vcc 0
;  8 den: same, S0 = S1: S1*2^64, field 65 -> 0x20800000, vcc 0
;  9 den: S1 = +0, S2 = 1.0: e2 - e1 = 127 -> vcc 1; S1 zero -> 0xffc00000
; 10 num: S1 = 1.0, S2 = signalling NaN 0x7f800001 (e2 255): e2 - e1 = 128 -> vcc 1, S0 != S1 -> 0x7f800001 kept
; 11 den: S0 = -S1 (neg), S1, S2 as case 0: vcc 1, S0 != S1 -> D = -S1 = 0xbf800005
; 12 den clamp: as case 9: vcc 1, D NaN, clamp -> 0x00000000
; 13 den mul:2: S1 = 0.25 (0x3e800000), S2 = 0.5: no rule -> D = S1, then omod x2 -> 0.5 = 0x3f000000, vcc 0
; 14 S0 = v4 = 0x20000000 (e0 64), S1 = 0x7f000000, S2 = 0x7d000000: e1 >= 253, e1 - e2 = 4 -> S0*2^-64 = 2^-127,
;         a denormal, flushed -> 0x00000000, vcc 0
; 15 as 14 with S0 = 0x1f800000 (e0 63) -> measured +inf 0x7f800000, vcc 0
; Run: vk_compute_test cs_div_scale_f32.spv 1 dump with VKC_OUT_WORDS=64 and VKC_DUMP (check.py div_scale_f32_test)
v_cmpx_eq_u32_e32 0, v0
v_mov_b32_e32 v2, 0x3f800005
v_mov_b32_e32 v3, 0x71800000
v_mov_b32_e32 v4, 0x00000000
v_div_scale_f32 v8, vcc_lo, v2, v2, v3
v_cndmask_b32_e64 v9, 0, 1, vcc_lo
buffer_store_dword v8, off, s[0:3], 0 offset:0
buffer_store_dword v9, off, s[0:3], 0 offset:4
v_mov_b32_e32 v2, 0x3f800005
v_mov_b32_e32 v3, 0x71800000
v_mov_b32_e32 v4, 0x00000000
v_div_scale_f32 v8, vcc_lo, v3, v2, v3
v_cndmask_b32_e64 v9, 0, 1, vcc_lo
buffer_store_dword v8, off, s[0:3], 0 offset:8
buffer_store_dword v9, off, s[0:3], 0 offset:12
v_mov_b32_e32 v2, 0x3f800005
v_mov_b32_e32 v3, 0x0d800003
v_mov_b32_e32 v4, 0x00000000
v_div_scale_f32 v8, vcc_lo, v3, v2, v3
v_cndmask_b32_e64 v9, 0, 1, vcc_lo
buffer_store_dword v8, off, s[0:3], 0 offset:16
buffer_store_dword v9, off, s[0:3], 0 offset:20
v_mov_b32_e32 v2, 0x3f800005
v_mov_b32_e32 v3, 0x0d800003
v_mov_b32_e32 v4, 0x00000000
v_div_scale_f32 v8, vcc_lo, v2, v2, v3
v_cndmask_b32_e64 v9, 0, 1, vcc_lo
buffer_store_dword v8, off, s[0:3], 0 offset:24
buffer_store_dword v9, off, s[0:3], 0 offset:28
v_mov_b32_e32 v2, 0x7f000000
v_mov_b32_e32 v3, 0x3f800000
v_mov_b32_e32 v4, 0x00000000
v_div_scale_f32 v8, vcc_lo, v2, v2, v3
v_cndmask_b32_e64 v9, 0, 1, vcc_lo
buffer_store_dword v8, off, s[0:3], 0 offset:32
buffer_store_dword v9, off, s[0:3], 0 offset:36
v_mov_b32_e32 v2, 0x7f000000
v_mov_b32_e32 v3, 0x7d000009
v_mov_b32_e32 v4, 0x00000000
v_div_scale_f32 v8, vcc_lo, v3, v2, v3
v_cndmask_b32_e64 v9, 0, 1, vcc_lo
buffer_store_dword v8, off, s[0:3], 0 offset:40
buffer_store_dword v9, off, s[0:3], 0 offset:44
v_mov_b32_e32 v2, 0x3f800000
v_mov_b32_e32 v3, 0x00000003
v_mov_b32_e32 v4, 0x00000000
v_div_scale_f32 v8, vcc_lo, v3, v2, v3
v_cndmask_b32_e64 v9, 0, 1, vcc_lo
buffer_store_dword v8, off, s[0:3], 0 offset:48
buffer_store_dword v9, off, s[0:3], 0 offset:52
v_mov_b32_e32 v2, 0x00800000
v_mov_b32_e32 v3, 0x0a000001
v_mov_b32_e32 v4, 0x00000000
v_div_scale_f32 v8, vcc_lo, v3, v2, v3
v_cndmask_b32_e64 v9, 0, 1, vcc_lo
buffer_store_dword v8, off, s[0:3], 0 offset:56
buffer_store_dword v9, off, s[0:3], 0 offset:60
v_mov_b32_e32 v2, 0x00800000
v_mov_b32_e32 v3, 0x0a000001
v_mov_b32_e32 v4, 0x00000000
v_div_scale_f32 v8, vcc_lo, v2, v2, v3
v_cndmask_b32_e64 v9, 0, 1, vcc_lo
buffer_store_dword v8, off, s[0:3], 0 offset:64
buffer_store_dword v9, off, s[0:3], 0 offset:68
v_mov_b32_e32 v2, 0x00000000
v_mov_b32_e32 v3, 0x3f800000
v_mov_b32_e32 v4, 0x00000000
v_div_scale_f32 v8, vcc_lo, v2, v2, v3
v_cndmask_b32_e64 v9, 0, 1, vcc_lo
buffer_store_dword v8, off, s[0:3], 0 offset:72
buffer_store_dword v9, off, s[0:3], 0 offset:76
v_mov_b32_e32 v2, 0x3f800000
v_mov_b32_e32 v3, 0x7f800001
v_mov_b32_e32 v4, 0x00000000
v_div_scale_f32 v8, vcc_lo, v3, v2, v3
v_cndmask_b32_e64 v9, 0, 1, vcc_lo
buffer_store_dword v8, off, s[0:3], 0 offset:80
buffer_store_dword v9, off, s[0:3], 0 offset:84
v_mov_b32_e32 v2, 0x3f800005
v_mov_b32_e32 v3, 0x71800000
v_mov_b32_e32 v4, 0x00000000
v_div_scale_f32 v8, vcc_lo, -v2, v2, v3
v_cndmask_b32_e64 v9, 0, 1, vcc_lo
buffer_store_dword v8, off, s[0:3], 0 offset:88
buffer_store_dword v9, off, s[0:3], 0 offset:92
v_mov_b32_e32 v2, 0x00000000
v_mov_b32_e32 v3, 0x3f800000
v_mov_b32_e32 v4, 0x00000000
v_div_scale_f32 v8, vcc_lo, v2, v2, v3 clamp
v_cndmask_b32_e64 v9, 0, 1, vcc_lo
buffer_store_dword v8, off, s[0:3], 0 offset:96
buffer_store_dword v9, off, s[0:3], 0 offset:100
v_mov_b32_e32 v2, 0x3e800000
v_mov_b32_e32 v3, 0x3f000000
v_mov_b32_e32 v4, 0x00000000
v_div_scale_f32 v8, vcc_lo, v2, v2, v3 mul:2
v_cndmask_b32_e64 v9, 0, 1, vcc_lo
buffer_store_dword v8, off, s[0:3], 0 offset:104
buffer_store_dword v9, off, s[0:3], 0 offset:108
v_mov_b32_e32 v2, 0x7f000000
v_mov_b32_e32 v3, 0x7d000000
v_mov_b32_e32 v4, 0x20000000
v_div_scale_f32 v8, vcc_lo, v4, v2, v3
v_cndmask_b32_e64 v9, 0, 1, vcc_lo
buffer_store_dword v8, off, s[0:3], 0 offset:112
buffer_store_dword v9, off, s[0:3], 0 offset:116
v_mov_b32_e32 v2, 0x7f000000
v_mov_b32_e32 v3, 0x7d000000
v_mov_b32_e32 v4, 0x1f800000
v_div_scale_f32 v8, vcc_lo, v4, v2, v3
v_cndmask_b32_e64 v9, 0, 1, vcc_lo
buffer_store_dword v8, off, s[0:3], 0 offset:120
buffer_store_dword v9, off, s[0:3], 0 offset:124
s_waitcnt vmcnt(0)
s_endpgm
