; EGEN-F32-DIV-FIXUP: v_div_fixup_f32 D, S0, S1, S2 (S0 quotient, S1 denominator, S2 numerator).
; Lane 0 runs 20 cases; case i stores D at out[i]. out[20..63] stay 0xeeeeeeee.
; Rule (docs/design.md, measured on RDNA2 gfx1036 with tools/hw_oracle in the game's MODE: IEEE off, f32 denormals flushed):
; abs/neg act on the sign bits first; a denormal S1 or S2 reads as a zero of its sign, S0 is not flushed.
; s = sign(S1) ^ sign(S2). First match wins: S2 NaN -> S2 unchanged; S1 NaN -> S1 unchanged; S1 and S2 both
; zero or both inf -> 0xffc00000; S1 zero or S2 inf -> inf with sign s; S1 inf or S2 zero -> zero with sign s;
; exp(S2) - exp(S1) < -150 -> zero with sign s; S0 NaN -> inf with sign s; else |S0| with sign s.
; omod scales the result (IEEE off; a zero or denormal result becomes +0), then clamp: negatives and NaN -> +0,
; >= 1.0 -> 1.0.
;  0 S0 1.5, S1 -1.0, S2 2.0: s = 1 -> -1.5 = 0xbfc00000
;  1 S0 -1.5, S1 1.0, S2 3.0: s = 0 -> |S0| = 0x3fc00000
;  2 S2 = signalling NaN 0x7fa00000 -> unchanged 0x7fa00000 (IEEE off)
;  3 S1 = signalling NaN 0xff800001 (S2 1.0) -> unchanged 0xff800001
;  4 S1 +0, S2 -0 -> 0xffc00000
;  5 S1 +inf, S2 -inf -> 0xffc00000
;  6 S1 -0, S2 2.0: s = 1 -> -inf 0xff800000
;  7 S1 1.0, S2 +inf -> +inf 0x7f800000
;  8 S1 -inf, S2 2.0 -> -0 0x80000000
;  9 S1 3.0, S2 +0 -> +0 0x00000000
; 10 S1 -1.0, S2 0x00000003 (denormal, reads as +0) -> -0 0x80000000
; 11 S1 0xff000000 (exp 254), S2 0x32000000 (exp 100): 100 - 254 = -154 < -150 -> -0 0x80000000
; 12 S0 0x00800003, S1 0x7f000000 (254), S2 0x34000000 (104): -150 is not < -150 -> |S0| = 0x00800003
; 13 S0 qNaN 0x7fc00000, S1 1.0, S2 -1.0 -> -inf 0xff800000
; 14 S0 0x80000005 (denormal, not flushed), S1 = S2 = 1.0 -> 0x00000005
; 15 |S2| with S2 = -2.0, S1 1.0, S0 1.5 -> s = 0 -> 0x3fc00000 (without abs 0xbfc00000)
; 16 -S1 with S1 1.0, S2 2.0, S0 1.5 -> s = 1 -> 0xbfc00000
; 17 clamp: S0 0.5, S1 -1.0, S2 1.0 -> -0.5 -> +0 0x00000000
; 18 clamp: S0 1.5, S1 = S2 = 1.0 -> 1.5 -> 1.0 0x3f800000
; 19 mul:2: S0 1.5, S1 = S2 = 1.0 -> 1.5 * 2 = 3.0 = 0x40400000
; Run: vk_compute_test cs_div_fixup_f32.spv 1 dump with VKC_OUT_WORDS=64 and VKC_DUMP (check.py div_fixup_f32_test)
v_cmpx_eq_u32_e32 0, v0
v_mov_b32_e32 v2, 0xbf800000
v_mov_b32_e32 v3, 0x40000000
v_mov_b32_e32 v4, 0x3fc00000
v_div_fixup_f32 v8, v4, v2, v3
buffer_store_dword v8, off, s[0:3], 0 offset:0
v_mov_b32_e32 v2, 0x3f800000
v_mov_b32_e32 v3, 0x40400000
v_mov_b32_e32 v4, 0xbfc00000
v_div_fixup_f32 v8, v4, v2, v3
buffer_store_dword v8, off, s[0:3], 0 offset:4
v_mov_b32_e32 v2, 0x3f800000
v_mov_b32_e32 v3, 0x7fa00000
v_mov_b32_e32 v4, 0x3fc00000
v_div_fixup_f32 v8, v4, v2, v3
buffer_store_dword v8, off, s[0:3], 0 offset:8
v_mov_b32_e32 v2, 0xff800001
v_mov_b32_e32 v3, 0x3f800000
v_mov_b32_e32 v4, 0x3fc00000
v_div_fixup_f32 v8, v4, v2, v3
buffer_store_dword v8, off, s[0:3], 0 offset:12
v_mov_b32_e32 v2, 0x00000000
v_mov_b32_e32 v3, 0x80000000
v_mov_b32_e32 v4, 0x3fc00000
v_div_fixup_f32 v8, v4, v2, v3
buffer_store_dword v8, off, s[0:3], 0 offset:16
v_mov_b32_e32 v2, 0x7f800000
v_mov_b32_e32 v3, 0xff800000
v_mov_b32_e32 v4, 0x3fc00000
v_div_fixup_f32 v8, v4, v2, v3
buffer_store_dword v8, off, s[0:3], 0 offset:20
v_mov_b32_e32 v2, 0x80000000
v_mov_b32_e32 v3, 0x40000000
v_mov_b32_e32 v4, 0x3fc00000
v_div_fixup_f32 v8, v4, v2, v3
buffer_store_dword v8, off, s[0:3], 0 offset:24
v_mov_b32_e32 v2, 0x3f800000
v_mov_b32_e32 v3, 0x7f800000
v_mov_b32_e32 v4, 0x3fc00000
v_div_fixup_f32 v8, v4, v2, v3
buffer_store_dword v8, off, s[0:3], 0 offset:28
v_mov_b32_e32 v2, 0xff800000
v_mov_b32_e32 v3, 0x40000000
v_mov_b32_e32 v4, 0x3fc00000
v_div_fixup_f32 v8, v4, v2, v3
buffer_store_dword v8, off, s[0:3], 0 offset:32
v_mov_b32_e32 v2, 0x40400000
v_mov_b32_e32 v3, 0x00000000
v_mov_b32_e32 v4, 0x3fc00000
v_div_fixup_f32 v8, v4, v2, v3
buffer_store_dword v8, off, s[0:3], 0 offset:36
v_mov_b32_e32 v2, 0xbf800000
v_mov_b32_e32 v3, 0x00000003
v_mov_b32_e32 v4, 0x3fc00000
v_div_fixup_f32 v8, v4, v2, v3
buffer_store_dword v8, off, s[0:3], 0 offset:40
v_mov_b32_e32 v2, 0xff000000
v_mov_b32_e32 v3, 0x32000000
v_mov_b32_e32 v4, 0x3fc00000
v_div_fixup_f32 v8, v4, v2, v3
buffer_store_dword v8, off, s[0:3], 0 offset:44
v_mov_b32_e32 v2, 0x7f000000
v_mov_b32_e32 v3, 0x34000000
v_mov_b32_e32 v4, 0x00800003
v_div_fixup_f32 v8, v4, v2, v3
buffer_store_dword v8, off, s[0:3], 0 offset:48
v_mov_b32_e32 v2, 0x3f800000
v_mov_b32_e32 v3, 0xbf800000
v_mov_b32_e32 v4, 0x7fc00000
v_div_fixup_f32 v8, v4, v2, v3
buffer_store_dword v8, off, s[0:3], 0 offset:52
v_mov_b32_e32 v2, 0x3f800000
v_mov_b32_e32 v3, 0x3f800000
v_mov_b32_e32 v4, 0x80000005
v_div_fixup_f32 v8, v4, v2, v3
buffer_store_dword v8, off, s[0:3], 0 offset:56
v_mov_b32_e32 v2, 0x3f800000
v_mov_b32_e32 v3, 0xc0000000
v_mov_b32_e32 v4, 0x3fc00000
v_div_fixup_f32 v8, v4, v2, |v3|
buffer_store_dword v8, off, s[0:3], 0 offset:60
v_mov_b32_e32 v2, 0x3f800000
v_mov_b32_e32 v3, 0x40000000
v_mov_b32_e32 v4, 0x3fc00000
v_div_fixup_f32 v8, v4, -v2, v3
buffer_store_dword v8, off, s[0:3], 0 offset:64
v_mov_b32_e32 v2, 0xbf800000
v_mov_b32_e32 v3, 0x3f800000
v_mov_b32_e32 v4, 0x3f000000
v_div_fixup_f32 v8, v4, v2, v3 clamp
buffer_store_dword v8, off, s[0:3], 0 offset:68
v_mov_b32_e32 v2, 0x3f800000
v_mov_b32_e32 v3, 0x3f800000
v_mov_b32_e32 v4, 0x3fc00000
v_div_fixup_f32 v8, v4, v2, v3 clamp
buffer_store_dword v8, off, s[0:3], 0 offset:72
v_mov_b32_e32 v2, 0x3f800000
v_mov_b32_e32 v3, 0x3f800000
v_mov_b32_e32 v4, 0x3fc00000
v_div_fixup_f32 v8, v4, v2, v3 mul:2
buffer_store_dword v8, off, s[0:3], 0 offset:76
s_waitcnt vmcnt(0)
s_endpgm
