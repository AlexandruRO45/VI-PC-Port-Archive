; v_cvt_pknorm_i16_f16 / v_cvt_pknorm_u16_f16 (docs/design.md row), run once on a GPU. Lane 0 of a wave32 group runs
; the 29 cases of tools/cvt_pknorm_f16_probe.py (CASES without the raw-omod encodings), the same ones measured on
; gfx1036, for each op, first plain and then with clamp (a hardware no-op), and stores result k at byte 4k:
;   words 0..28 i16, 29..57 u16, then the same two blocks with clamp (116 words in all).
; The cases cover op_sel on both source halves and the destination bit (D = packed << 16, the old D dropped), near-tie
; inputs around 0.5 and 0.25, +-inf, quiet and signalling NaN, +-0 and a denormal, neg/abs, inline f16 and integer
; constants, a 16-bit literal, an SGPR source and a destination that is also a source.
; Expected words: cvt_pknorm_f16_probe.expected_pknorm_f16 (exact rational model), CS_CVT_PKNORM_F16_WORDS in check.py.
; group: 32,1,1
; mattr: +wavefrontsize32,-wavefrontsize64
; Run: vk_compute_test cs_cvt_pknorm_f16.spv 1 dump (VKC_OUT_WORDS=128) -> words 0..115 as in check.py
v_cmpx_eq_u32_e32 0, v0
v_mov_b32_e32 v1, 0x3c003400
v_mov_b32_e32 v2, 0xbc003801
v_mov_b32_e32 v8, 0xdeadbeef
v_cvt_pknorm_i16_f16 v8, v1, v2 op_sel:[0,0,0]
v_mov_b32_e32 v20, 0
buffer_store_dword v8, v20, s[0:3], 0 offen
v_mov_b32_e32 v1, 0x3c003400
v_mov_b32_e32 v2, 0xbc003801
v_mov_b32_e32 v8, 0xdeadbeef
v_cvt_pknorm_i16_f16 v8, v1, v2 op_sel:[0,0,1]
v_mov_b32_e32 v20, 4
buffer_store_dword v8, v20, s[0:3], 0 offen
v_mov_b32_e32 v1, 0x3c003400
v_mov_b32_e32 v2, 0xbc003801
v_mov_b32_e32 v8, 0xdeadbeef
v_cvt_pknorm_i16_f16 v8, v1, v2 op_sel:[0,1,0]
v_mov_b32_e32 v20, 8
buffer_store_dword v8, v20, s[0:3], 0 offen
v_mov_b32_e32 v1, 0x3c003400
v_mov_b32_e32 v2, 0xbc003801
v_mov_b32_e32 v8, 0xdeadbeef
v_cvt_pknorm_i16_f16 v8, v1, v2 op_sel:[0,1,1]
v_mov_b32_e32 v20, 12
buffer_store_dword v8, v20, s[0:3], 0 offen
v_mov_b32_e32 v1, 0x3c003400
v_mov_b32_e32 v2, 0xbc003801
v_mov_b32_e32 v8, 0xdeadbeef
v_cvt_pknorm_i16_f16 v8, v1, v2 op_sel:[1,0,0]
v_mov_b32_e32 v20, 16
buffer_store_dword v8, v20, s[0:3], 0 offen
v_mov_b32_e32 v1, 0x3c003400
v_mov_b32_e32 v2, 0xbc003801
v_mov_b32_e32 v8, 0xdeadbeef
v_cvt_pknorm_i16_f16 v8, v1, v2 op_sel:[1,0,1]
v_mov_b32_e32 v20, 20
buffer_store_dword v8, v20, s[0:3], 0 offen
v_mov_b32_e32 v1, 0x3c003400
v_mov_b32_e32 v2, 0xbc003801
v_mov_b32_e32 v8, 0xdeadbeef
v_cvt_pknorm_i16_f16 v8, v1, v2 op_sel:[1,1,0]
v_mov_b32_e32 v20, 24
buffer_store_dword v8, v20, s[0:3], 0 offen
v_mov_b32_e32 v1, 0x3c003400
v_mov_b32_e32 v2, 0xbc003801
v_mov_b32_e32 v8, 0xdeadbeef
v_cvt_pknorm_i16_f16 v8, v1, v2 op_sel:[1,1,1]
v_mov_b32_e32 v20, 28
buffer_store_dword v8, v20, s[0:3], 0 offen
v_mov_b32_e32 v1, 0x3c003400
v_mov_b32_e32 v2, 0xbc003801
v_mov_b32_e32 v8, 0x12345678
v_cvt_pknorm_i16_f16 v8, v1, v2 op_sel:[0,0,1]
v_mov_b32_e32 v20, 32
buffer_store_dword v8, v20, s[0:3], 0 offen
v_mov_b32_e32 v1, 0x3c003400
v_mov_b32_e32 v2, 0xbc003801
v_mov_b32_e32 v8, 0xffffffff
v_cvt_pknorm_i16_f16 v8, v1, v2 op_sel:[0,0,1]
v_mov_b32_e32 v20, 36
buffer_store_dword v8, v20, s[0:3], 0 offen
v_mov_b32_e32 v1, 0x7e003801
v_mov_b32_e32 v2, 0x3c00b801
v_mov_b32_e32 v8, 0xdeadbeef
v_cvt_pknorm_i16_f16 v8, v1, v2
v_mov_b32_e32 v20, 40
buffer_store_dword v8, v20, s[0:3], 0 offen
v_mov_b32_e32 v1, 0x000137ff
v_mov_b32_e32 v2, 0x8001b7ff
v_mov_b32_e32 v8, 0xdeadbeef
v_cvt_pknorm_i16_f16 v8, v1, v2
v_mov_b32_e32 v20, 44
buffer_store_dword v8, v20, s[0:3], 0 offen
v_mov_b32_e32 v1, 0x38007c00
v_mov_b32_e32 v2, 0x3800fc00
v_mov_b32_e32 v8, 0xdeadbeef
v_cvt_pknorm_i16_f16 v8, v1, v2
v_mov_b32_e32 v20, 48
buffer_store_dword v8, v20, s[0:3], 0 offen
v_mov_b32_e32 v1, 0x3c007e00
v_mov_b32_e32 v2, 0xbc007c01
v_mov_b32_e32 v8, 0xdeadbeef
v_cvt_pknorm_i16_f16 v8, v1, v2
v_mov_b32_e32 v20, 52
buffer_store_dword v8, v20, s[0:3], 0 offen
v_mov_b32_e32 v1, 0x00018000
v_mov_b32_e32 v2, 0x80010000
v_mov_b32_e32 v8, 0xdeadbeef
v_cvt_pknorm_i16_f16 v8, v1, v2
v_mov_b32_e32 v20, 56
buffer_store_dword v8, v20, s[0:3], 0 offen
v_mov_b32_e32 v1, 0x3c003400
v_mov_b32_e32 v2, 0xbc003801
v_mov_b32_e32 v8, 0xdeadbeef
v_cvt_pknorm_i16_f16 v8, |v1|, -v2 op_sel:[1,1,0]
v_mov_b32_e32 v20, 60
buffer_store_dword v8, v20, s[0:3], 0 offen
v_mov_b32_e32 v1, 0x3c003400
v_mov_b32_e32 v2, 0xbc003801
v_mov_b32_e32 v8, 0xdeadbeef
v_cvt_pknorm_i16_f16 v8, -|v1|, |v2| op_sel:[0,1,0]
v_mov_b32_e32 v20, 64
buffer_store_dword v8, v20, s[0:3], 0 offen
v_mov_b32_e32 v1, 0x3c003400
v_mov_b32_e32 v2, 0xbc003801
v_mov_b32_e32 v8, 0xdeadbeef
v_cvt_pknorm_i16_f16 v8, |v1|, -v2 op_sel:[1,1,1]
v_mov_b32_e32 v20, 68
buffer_store_dword v8, v20, s[0:3], 0 offen
v_mov_b32_e32 v8, 0xdeadbeef
v_cvt_pknorm_i16_f16 v8, 0.5, -0.5
v_mov_b32_e32 v20, 72
buffer_store_dword v8, v20, s[0:3], 0 offen
v_mov_b32_e32 v2, 0xbc003801
v_mov_b32_e32 v8, 0xdeadbeef
v_cvt_pknorm_i16_f16 v8, 0.5, v2 op_sel:[1,0,0]
v_mov_b32_e32 v20, 76
buffer_store_dword v8, v20, s[0:3], 0 offen
v_mov_b32_e32 v2, 0xbc003801
v_mov_b32_e32 v8, 0xdeadbeef
v_cvt_pknorm_i16_f16 v8, 1.0, v2 op_sel:[1,0,0]
v_mov_b32_e32 v20, 80
buffer_store_dword v8, v20, s[0:3], 0 offen
v_mov_b32_e32 v2, 0xbc003801
v_mov_b32_e32 v8, 0xdeadbeef
v_cvt_pknorm_i16_f16 v8, -0.5, v2
v_mov_b32_e32 v20, 84
buffer_store_dword v8, v20, s[0:3], 0 offen
v_mov_b32_e32 v8, 0xdeadbeef
v_cvt_pknorm_i16_f16 v8, 1, -1
v_mov_b32_e32 v20, 88
buffer_store_dword v8, v20, s[0:3], 0 offen
v_mov_b32_e32 v2, 0xbc003801
v_mov_b32_e32 v8, 0xdeadbeef
v_cvt_pknorm_i16_f16 v8, 0x3800, v2
v_mov_b32_e32 v20, 92
buffer_store_dword v8, v20, s[0:3], 0 offen
v_mov_b32_e32 v2, 0xbc003801
v_mov_b32_e32 v8, 0xdeadbeef
v_cvt_pknorm_i16_f16 v8, 0x3800, v2 op_sel:[1,0,0]
v_mov_b32_e32 v20, 96
buffer_store_dword v8, v20, s[0:3], 0 offen
s_mov_b32 s4, 0x3c003400
v_mov_b32_e32 v2, 0xbc003801
v_mov_b32_e32 v8, 0xdeadbeef
v_cvt_pknorm_i16_f16 v8, s4, v2 op_sel:[1,0,0]
v_mov_b32_e32 v20, 100
buffer_store_dword v8, v20, s[0:3], 0 offen
s_mov_b32 s4, 0x3c003400
v_mov_b32_e32 v2, 0xbc003801
v_mov_b32_e32 v8, 0xdeadbeef
v_cvt_pknorm_i16_f16 v8, s4, v2 op_sel:[1,0,1]
v_mov_b32_e32 v20, 104
buffer_store_dword v8, v20, s[0:3], 0 offen
v_mov_b32_e32 v8, 0x3c003400
v_mov_b32_e32 v2, 0xbc003801
v_cvt_pknorm_i16_f16 v8, v8, v2 op_sel:[0,0,1]
v_mov_b32_e32 v20, 108
buffer_store_dword v8, v20, s[0:3], 0 offen
v_mov_b32_e32 v1, 0x3c003400
v_mov_b32_e32 v8, 0xbc003801
v_cvt_pknorm_i16_f16 v8, v1, v8 op_sel:[0,0,1]
v_mov_b32_e32 v20, 112
buffer_store_dword v8, v20, s[0:3], 0 offen
v_mov_b32_e32 v1, 0x3c003400
v_mov_b32_e32 v2, 0xbc003801
v_mov_b32_e32 v8, 0xdeadbeef
v_cvt_pknorm_u16_f16 v8, v1, v2 op_sel:[0,0,0]
v_mov_b32_e32 v20, 116
buffer_store_dword v8, v20, s[0:3], 0 offen
v_mov_b32_e32 v1, 0x3c003400
v_mov_b32_e32 v2, 0xbc003801
v_mov_b32_e32 v8, 0xdeadbeef
v_cvt_pknorm_u16_f16 v8, v1, v2 op_sel:[0,0,1]
v_mov_b32_e32 v20, 120
buffer_store_dword v8, v20, s[0:3], 0 offen
v_mov_b32_e32 v1, 0x3c003400
v_mov_b32_e32 v2, 0xbc003801
v_mov_b32_e32 v8, 0xdeadbeef
v_cvt_pknorm_u16_f16 v8, v1, v2 op_sel:[0,1,0]
v_mov_b32_e32 v20, 124
buffer_store_dword v8, v20, s[0:3], 0 offen
v_mov_b32_e32 v1, 0x3c003400
v_mov_b32_e32 v2, 0xbc003801
v_mov_b32_e32 v8, 0xdeadbeef
v_cvt_pknorm_u16_f16 v8, v1, v2 op_sel:[0,1,1]
v_mov_b32_e32 v20, 128
buffer_store_dword v8, v20, s[0:3], 0 offen
v_mov_b32_e32 v1, 0x3c003400
v_mov_b32_e32 v2, 0xbc003801
v_mov_b32_e32 v8, 0xdeadbeef
v_cvt_pknorm_u16_f16 v8, v1, v2 op_sel:[1,0,0]
v_mov_b32_e32 v20, 132
buffer_store_dword v8, v20, s[0:3], 0 offen
v_mov_b32_e32 v1, 0x3c003400
v_mov_b32_e32 v2, 0xbc003801
v_mov_b32_e32 v8, 0xdeadbeef
v_cvt_pknorm_u16_f16 v8, v1, v2 op_sel:[1,0,1]
v_mov_b32_e32 v20, 136
buffer_store_dword v8, v20, s[0:3], 0 offen
v_mov_b32_e32 v1, 0x3c003400
v_mov_b32_e32 v2, 0xbc003801
v_mov_b32_e32 v8, 0xdeadbeef
v_cvt_pknorm_u16_f16 v8, v1, v2 op_sel:[1,1,0]
v_mov_b32_e32 v20, 140
buffer_store_dword v8, v20, s[0:3], 0 offen
v_mov_b32_e32 v1, 0x3c003400
v_mov_b32_e32 v2, 0xbc003801
v_mov_b32_e32 v8, 0xdeadbeef
v_cvt_pknorm_u16_f16 v8, v1, v2 op_sel:[1,1,1]
v_mov_b32_e32 v20, 144
buffer_store_dword v8, v20, s[0:3], 0 offen
v_mov_b32_e32 v1, 0x3c003400
v_mov_b32_e32 v2, 0xbc003801
v_mov_b32_e32 v8, 0x12345678
v_cvt_pknorm_u16_f16 v8, v1, v2 op_sel:[0,0,1]
v_mov_b32_e32 v20, 148
buffer_store_dword v8, v20, s[0:3], 0 offen
v_mov_b32_e32 v1, 0x3c003400
v_mov_b32_e32 v2, 0xbc003801
v_mov_b32_e32 v8, 0xffffffff
v_cvt_pknorm_u16_f16 v8, v1, v2 op_sel:[0,0,1]
v_mov_b32_e32 v20, 152
buffer_store_dword v8, v20, s[0:3], 0 offen
v_mov_b32_e32 v1, 0x7e003801
v_mov_b32_e32 v2, 0x3c00b801
v_mov_b32_e32 v8, 0xdeadbeef
v_cvt_pknorm_u16_f16 v8, v1, v2
v_mov_b32_e32 v20, 156
buffer_store_dword v8, v20, s[0:3], 0 offen
v_mov_b32_e32 v1, 0x000137ff
v_mov_b32_e32 v2, 0x8001b7ff
v_mov_b32_e32 v8, 0xdeadbeef
v_cvt_pknorm_u16_f16 v8, v1, v2
v_mov_b32_e32 v20, 160
buffer_store_dword v8, v20, s[0:3], 0 offen
v_mov_b32_e32 v1, 0x38007c00
v_mov_b32_e32 v2, 0x3800fc00
v_mov_b32_e32 v8, 0xdeadbeef
v_cvt_pknorm_u16_f16 v8, v1, v2
v_mov_b32_e32 v20, 164
buffer_store_dword v8, v20, s[0:3], 0 offen
v_mov_b32_e32 v1, 0x3c007e00
v_mov_b32_e32 v2, 0xbc007c01
v_mov_b32_e32 v8, 0xdeadbeef
v_cvt_pknorm_u16_f16 v8, v1, v2
v_mov_b32_e32 v20, 168
buffer_store_dword v8, v20, s[0:3], 0 offen
v_mov_b32_e32 v1, 0x00018000
v_mov_b32_e32 v2, 0x80010000
v_mov_b32_e32 v8, 0xdeadbeef
v_cvt_pknorm_u16_f16 v8, v1, v2
v_mov_b32_e32 v20, 172
buffer_store_dword v8, v20, s[0:3], 0 offen
v_mov_b32_e32 v1, 0x3c003400
v_mov_b32_e32 v2, 0xbc003801
v_mov_b32_e32 v8, 0xdeadbeef
v_cvt_pknorm_u16_f16 v8, |v1|, -v2 op_sel:[1,1,0]
v_mov_b32_e32 v20, 176
buffer_store_dword v8, v20, s[0:3], 0 offen
v_mov_b32_e32 v1, 0x3c003400
v_mov_b32_e32 v2, 0xbc003801
v_mov_b32_e32 v8, 0xdeadbeef
v_cvt_pknorm_u16_f16 v8, -|v1|, |v2| op_sel:[0,1,0]
v_mov_b32_e32 v20, 180
buffer_store_dword v8, v20, s[0:3], 0 offen
v_mov_b32_e32 v1, 0x3c003400
v_mov_b32_e32 v2, 0xbc003801
v_mov_b32_e32 v8, 0xdeadbeef
v_cvt_pknorm_u16_f16 v8, |v1|, -v2 op_sel:[1,1,1]
v_mov_b32_e32 v20, 184
buffer_store_dword v8, v20, s[0:3], 0 offen
v_mov_b32_e32 v8, 0xdeadbeef
v_cvt_pknorm_u16_f16 v8, 0.5, -0.5
v_mov_b32_e32 v20, 188
buffer_store_dword v8, v20, s[0:3], 0 offen
v_mov_b32_e32 v2, 0xbc003801
v_mov_b32_e32 v8, 0xdeadbeef
v_cvt_pknorm_u16_f16 v8, 0.5, v2 op_sel:[1,0,0]
v_mov_b32_e32 v20, 192
buffer_store_dword v8, v20, s[0:3], 0 offen
v_mov_b32_e32 v2, 0xbc003801
v_mov_b32_e32 v8, 0xdeadbeef
v_cvt_pknorm_u16_f16 v8, 1.0, v2 op_sel:[1,0,0]
v_mov_b32_e32 v20, 196
buffer_store_dword v8, v20, s[0:3], 0 offen
v_mov_b32_e32 v2, 0xbc003801
v_mov_b32_e32 v8, 0xdeadbeef
v_cvt_pknorm_u16_f16 v8, -0.5, v2
v_mov_b32_e32 v20, 200
buffer_store_dword v8, v20, s[0:3], 0 offen
v_mov_b32_e32 v8, 0xdeadbeef
v_cvt_pknorm_u16_f16 v8, 1, -1
v_mov_b32_e32 v20, 204
buffer_store_dword v8, v20, s[0:3], 0 offen
v_mov_b32_e32 v2, 0xbc003801
v_mov_b32_e32 v8, 0xdeadbeef
v_cvt_pknorm_u16_f16 v8, 0x3800, v2
v_mov_b32_e32 v20, 208
buffer_store_dword v8, v20, s[0:3], 0 offen
v_mov_b32_e32 v2, 0xbc003801
v_mov_b32_e32 v8, 0xdeadbeef
v_cvt_pknorm_u16_f16 v8, 0x3800, v2 op_sel:[1,0,0]
v_mov_b32_e32 v20, 212
buffer_store_dword v8, v20, s[0:3], 0 offen
s_mov_b32 s4, 0x3c003400
v_mov_b32_e32 v2, 0xbc003801
v_mov_b32_e32 v8, 0xdeadbeef
v_cvt_pknorm_u16_f16 v8, s4, v2 op_sel:[1,0,0]
v_mov_b32_e32 v20, 216
buffer_store_dword v8, v20, s[0:3], 0 offen
s_mov_b32 s4, 0x3c003400
v_mov_b32_e32 v2, 0xbc003801
v_mov_b32_e32 v8, 0xdeadbeef
v_cvt_pknorm_u16_f16 v8, s4, v2 op_sel:[1,0,1]
v_mov_b32_e32 v20, 220
buffer_store_dword v8, v20, s[0:3], 0 offen
v_mov_b32_e32 v8, 0x3c003400
v_mov_b32_e32 v2, 0xbc003801
v_cvt_pknorm_u16_f16 v8, v8, v2 op_sel:[0,0,1]
v_mov_b32_e32 v20, 224
buffer_store_dword v8, v20, s[0:3], 0 offen
v_mov_b32_e32 v1, 0x3c003400
v_mov_b32_e32 v8, 0xbc003801
v_cvt_pknorm_u16_f16 v8, v1, v8 op_sel:[0,0,1]
v_mov_b32_e32 v20, 228
buffer_store_dword v8, v20, s[0:3], 0 offen
v_mov_b32_e32 v1, 0x3c003400
v_mov_b32_e32 v2, 0xbc003801
v_mov_b32_e32 v8, 0xdeadbeef
v_cvt_pknorm_i16_f16 v8, v1, v2 op_sel:[0,0,0] clamp
v_mov_b32_e32 v20, 232
buffer_store_dword v8, v20, s[0:3], 0 offen
v_mov_b32_e32 v1, 0x3c003400
v_mov_b32_e32 v2, 0xbc003801
v_mov_b32_e32 v8, 0xdeadbeef
v_cvt_pknorm_i16_f16 v8, v1, v2 op_sel:[0,0,1] clamp
v_mov_b32_e32 v20, 236
buffer_store_dword v8, v20, s[0:3], 0 offen
v_mov_b32_e32 v1, 0x3c003400
v_mov_b32_e32 v2, 0xbc003801
v_mov_b32_e32 v8, 0xdeadbeef
v_cvt_pknorm_i16_f16 v8, v1, v2 op_sel:[0,1,0] clamp
v_mov_b32_e32 v20, 240
buffer_store_dword v8, v20, s[0:3], 0 offen
v_mov_b32_e32 v1, 0x3c003400
v_mov_b32_e32 v2, 0xbc003801
v_mov_b32_e32 v8, 0xdeadbeef
v_cvt_pknorm_i16_f16 v8, v1, v2 op_sel:[0,1,1] clamp
v_mov_b32_e32 v20, 244
buffer_store_dword v8, v20, s[0:3], 0 offen
v_mov_b32_e32 v1, 0x3c003400
v_mov_b32_e32 v2, 0xbc003801
v_mov_b32_e32 v8, 0xdeadbeef
v_cvt_pknorm_i16_f16 v8, v1, v2 op_sel:[1,0,0] clamp
v_mov_b32_e32 v20, 248
buffer_store_dword v8, v20, s[0:3], 0 offen
v_mov_b32_e32 v1, 0x3c003400
v_mov_b32_e32 v2, 0xbc003801
v_mov_b32_e32 v8, 0xdeadbeef
v_cvt_pknorm_i16_f16 v8, v1, v2 op_sel:[1,0,1] clamp
v_mov_b32_e32 v20, 252
buffer_store_dword v8, v20, s[0:3], 0 offen
v_mov_b32_e32 v1, 0x3c003400
v_mov_b32_e32 v2, 0xbc003801
v_mov_b32_e32 v8, 0xdeadbeef
v_cvt_pknorm_i16_f16 v8, v1, v2 op_sel:[1,1,0] clamp
v_mov_b32_e32 v20, 256
buffer_store_dword v8, v20, s[0:3], 0 offen
v_mov_b32_e32 v1, 0x3c003400
v_mov_b32_e32 v2, 0xbc003801
v_mov_b32_e32 v8, 0xdeadbeef
v_cvt_pknorm_i16_f16 v8, v1, v2 op_sel:[1,1,1] clamp
v_mov_b32_e32 v20, 260
buffer_store_dword v8, v20, s[0:3], 0 offen
v_mov_b32_e32 v1, 0x3c003400
v_mov_b32_e32 v2, 0xbc003801
v_mov_b32_e32 v8, 0x12345678
v_cvt_pknorm_i16_f16 v8, v1, v2 op_sel:[0,0,1] clamp
v_mov_b32_e32 v20, 264
buffer_store_dword v8, v20, s[0:3], 0 offen
v_mov_b32_e32 v1, 0x3c003400
v_mov_b32_e32 v2, 0xbc003801
v_mov_b32_e32 v8, 0xffffffff
v_cvt_pknorm_i16_f16 v8, v1, v2 op_sel:[0,0,1] clamp
v_mov_b32_e32 v20, 268
buffer_store_dword v8, v20, s[0:3], 0 offen
v_mov_b32_e32 v1, 0x7e003801
v_mov_b32_e32 v2, 0x3c00b801
v_mov_b32_e32 v8, 0xdeadbeef
v_cvt_pknorm_i16_f16 v8, v1, v2 clamp
v_mov_b32_e32 v20, 272
buffer_store_dword v8, v20, s[0:3], 0 offen
v_mov_b32_e32 v1, 0x000137ff
v_mov_b32_e32 v2, 0x8001b7ff
v_mov_b32_e32 v8, 0xdeadbeef
v_cvt_pknorm_i16_f16 v8, v1, v2 clamp
v_mov_b32_e32 v20, 276
buffer_store_dword v8, v20, s[0:3], 0 offen
v_mov_b32_e32 v1, 0x38007c00
v_mov_b32_e32 v2, 0x3800fc00
v_mov_b32_e32 v8, 0xdeadbeef
v_cvt_pknorm_i16_f16 v8, v1, v2 clamp
v_mov_b32_e32 v20, 280
buffer_store_dword v8, v20, s[0:3], 0 offen
v_mov_b32_e32 v1, 0x3c007e00
v_mov_b32_e32 v2, 0xbc007c01
v_mov_b32_e32 v8, 0xdeadbeef
v_cvt_pknorm_i16_f16 v8, v1, v2 clamp
v_mov_b32_e32 v20, 284
buffer_store_dword v8, v20, s[0:3], 0 offen
v_mov_b32_e32 v1, 0x00018000
v_mov_b32_e32 v2, 0x80010000
v_mov_b32_e32 v8, 0xdeadbeef
v_cvt_pknorm_i16_f16 v8, v1, v2 clamp
v_mov_b32_e32 v20, 288
buffer_store_dword v8, v20, s[0:3], 0 offen
v_mov_b32_e32 v1, 0x3c003400
v_mov_b32_e32 v2, 0xbc003801
v_mov_b32_e32 v8, 0xdeadbeef
v_cvt_pknorm_i16_f16 v8, |v1|, -v2 op_sel:[1,1,0] clamp
v_mov_b32_e32 v20, 292
buffer_store_dword v8, v20, s[0:3], 0 offen
v_mov_b32_e32 v1, 0x3c003400
v_mov_b32_e32 v2, 0xbc003801
v_mov_b32_e32 v8, 0xdeadbeef
v_cvt_pknorm_i16_f16 v8, -|v1|, |v2| op_sel:[0,1,0] clamp
v_mov_b32_e32 v20, 296
buffer_store_dword v8, v20, s[0:3], 0 offen
v_mov_b32_e32 v1, 0x3c003400
v_mov_b32_e32 v2, 0xbc003801
v_mov_b32_e32 v8, 0xdeadbeef
v_cvt_pknorm_i16_f16 v8, |v1|, -v2 op_sel:[1,1,1] clamp
v_mov_b32_e32 v20, 300
buffer_store_dword v8, v20, s[0:3], 0 offen
v_mov_b32_e32 v8, 0xdeadbeef
v_cvt_pknorm_i16_f16 v8, 0.5, -0.5 clamp
v_mov_b32_e32 v20, 304
buffer_store_dword v8, v20, s[0:3], 0 offen
v_mov_b32_e32 v2, 0xbc003801
v_mov_b32_e32 v8, 0xdeadbeef
v_cvt_pknorm_i16_f16 v8, 0.5, v2 op_sel:[1,0,0] clamp
v_mov_b32_e32 v20, 308
buffer_store_dword v8, v20, s[0:3], 0 offen
v_mov_b32_e32 v2, 0xbc003801
v_mov_b32_e32 v8, 0xdeadbeef
v_cvt_pknorm_i16_f16 v8, 1.0, v2 op_sel:[1,0,0] clamp
v_mov_b32_e32 v20, 312
buffer_store_dword v8, v20, s[0:3], 0 offen
v_mov_b32_e32 v2, 0xbc003801
v_mov_b32_e32 v8, 0xdeadbeef
v_cvt_pknorm_i16_f16 v8, -0.5, v2 clamp
v_mov_b32_e32 v20, 316
buffer_store_dword v8, v20, s[0:3], 0 offen
v_mov_b32_e32 v8, 0xdeadbeef
v_cvt_pknorm_i16_f16 v8, 1, -1 clamp
v_mov_b32_e32 v20, 320
buffer_store_dword v8, v20, s[0:3], 0 offen
v_mov_b32_e32 v2, 0xbc003801
v_mov_b32_e32 v8, 0xdeadbeef
v_cvt_pknorm_i16_f16 v8, 0x3800, v2 clamp
v_mov_b32_e32 v20, 324
buffer_store_dword v8, v20, s[0:3], 0 offen
v_mov_b32_e32 v2, 0xbc003801
v_mov_b32_e32 v8, 0xdeadbeef
v_cvt_pknorm_i16_f16 v8, 0x3800, v2 op_sel:[1,0,0] clamp
v_mov_b32_e32 v20, 328
buffer_store_dword v8, v20, s[0:3], 0 offen
s_mov_b32 s4, 0x3c003400
v_mov_b32_e32 v2, 0xbc003801
v_mov_b32_e32 v8, 0xdeadbeef
v_cvt_pknorm_i16_f16 v8, s4, v2 op_sel:[1,0,0] clamp
v_mov_b32_e32 v20, 332
buffer_store_dword v8, v20, s[0:3], 0 offen
s_mov_b32 s4, 0x3c003400
v_mov_b32_e32 v2, 0xbc003801
v_mov_b32_e32 v8, 0xdeadbeef
v_cvt_pknorm_i16_f16 v8, s4, v2 op_sel:[1,0,1] clamp
v_mov_b32_e32 v20, 336
buffer_store_dword v8, v20, s[0:3], 0 offen
v_mov_b32_e32 v8, 0x3c003400
v_mov_b32_e32 v2, 0xbc003801
v_cvt_pknorm_i16_f16 v8, v8, v2 op_sel:[0,0,1] clamp
v_mov_b32_e32 v20, 340
buffer_store_dword v8, v20, s[0:3], 0 offen
v_mov_b32_e32 v1, 0x3c003400
v_mov_b32_e32 v8, 0xbc003801
v_cvt_pknorm_i16_f16 v8, v1, v8 op_sel:[0,0,1] clamp
v_mov_b32_e32 v20, 344
buffer_store_dword v8, v20, s[0:3], 0 offen
v_mov_b32_e32 v1, 0x3c003400
v_mov_b32_e32 v2, 0xbc003801
v_mov_b32_e32 v8, 0xdeadbeef
v_cvt_pknorm_u16_f16 v8, v1, v2 op_sel:[0,0,0] clamp
v_mov_b32_e32 v20, 348
buffer_store_dword v8, v20, s[0:3], 0 offen
v_mov_b32_e32 v1, 0x3c003400
v_mov_b32_e32 v2, 0xbc003801
v_mov_b32_e32 v8, 0xdeadbeef
v_cvt_pknorm_u16_f16 v8, v1, v2 op_sel:[0,0,1] clamp
v_mov_b32_e32 v20, 352
buffer_store_dword v8, v20, s[0:3], 0 offen
v_mov_b32_e32 v1, 0x3c003400
v_mov_b32_e32 v2, 0xbc003801
v_mov_b32_e32 v8, 0xdeadbeef
v_cvt_pknorm_u16_f16 v8, v1, v2 op_sel:[0,1,0] clamp
v_mov_b32_e32 v20, 356
buffer_store_dword v8, v20, s[0:3], 0 offen
v_mov_b32_e32 v1, 0x3c003400
v_mov_b32_e32 v2, 0xbc003801
v_mov_b32_e32 v8, 0xdeadbeef
v_cvt_pknorm_u16_f16 v8, v1, v2 op_sel:[0,1,1] clamp
v_mov_b32_e32 v20, 360
buffer_store_dword v8, v20, s[0:3], 0 offen
v_mov_b32_e32 v1, 0x3c003400
v_mov_b32_e32 v2, 0xbc003801
v_mov_b32_e32 v8, 0xdeadbeef
v_cvt_pknorm_u16_f16 v8, v1, v2 op_sel:[1,0,0] clamp
v_mov_b32_e32 v20, 364
buffer_store_dword v8, v20, s[0:3], 0 offen
v_mov_b32_e32 v1, 0x3c003400
v_mov_b32_e32 v2, 0xbc003801
v_mov_b32_e32 v8, 0xdeadbeef
v_cvt_pknorm_u16_f16 v8, v1, v2 op_sel:[1,0,1] clamp
v_mov_b32_e32 v20, 368
buffer_store_dword v8, v20, s[0:3], 0 offen
v_mov_b32_e32 v1, 0x3c003400
v_mov_b32_e32 v2, 0xbc003801
v_mov_b32_e32 v8, 0xdeadbeef
v_cvt_pknorm_u16_f16 v8, v1, v2 op_sel:[1,1,0] clamp
v_mov_b32_e32 v20, 372
buffer_store_dword v8, v20, s[0:3], 0 offen
v_mov_b32_e32 v1, 0x3c003400
v_mov_b32_e32 v2, 0xbc003801
v_mov_b32_e32 v8, 0xdeadbeef
v_cvt_pknorm_u16_f16 v8, v1, v2 op_sel:[1,1,1] clamp
v_mov_b32_e32 v20, 376
buffer_store_dword v8, v20, s[0:3], 0 offen
v_mov_b32_e32 v1, 0x3c003400
v_mov_b32_e32 v2, 0xbc003801
v_mov_b32_e32 v8, 0x12345678
v_cvt_pknorm_u16_f16 v8, v1, v2 op_sel:[0,0,1] clamp
v_mov_b32_e32 v20, 380
buffer_store_dword v8, v20, s[0:3], 0 offen
v_mov_b32_e32 v1, 0x3c003400
v_mov_b32_e32 v2, 0xbc003801
v_mov_b32_e32 v8, 0xffffffff
v_cvt_pknorm_u16_f16 v8, v1, v2 op_sel:[0,0,1] clamp
v_mov_b32_e32 v20, 384
buffer_store_dword v8, v20, s[0:3], 0 offen
v_mov_b32_e32 v1, 0x7e003801
v_mov_b32_e32 v2, 0x3c00b801
v_mov_b32_e32 v8, 0xdeadbeef
v_cvt_pknorm_u16_f16 v8, v1, v2 clamp
v_mov_b32_e32 v20, 388
buffer_store_dword v8, v20, s[0:3], 0 offen
v_mov_b32_e32 v1, 0x000137ff
v_mov_b32_e32 v2, 0x8001b7ff
v_mov_b32_e32 v8, 0xdeadbeef
v_cvt_pknorm_u16_f16 v8, v1, v2 clamp
v_mov_b32_e32 v20, 392
buffer_store_dword v8, v20, s[0:3], 0 offen
v_mov_b32_e32 v1, 0x38007c00
v_mov_b32_e32 v2, 0x3800fc00
v_mov_b32_e32 v8, 0xdeadbeef
v_cvt_pknorm_u16_f16 v8, v1, v2 clamp
v_mov_b32_e32 v20, 396
buffer_store_dword v8, v20, s[0:3], 0 offen
v_mov_b32_e32 v1, 0x3c007e00
v_mov_b32_e32 v2, 0xbc007c01
v_mov_b32_e32 v8, 0xdeadbeef
v_cvt_pknorm_u16_f16 v8, v1, v2 clamp
v_mov_b32_e32 v20, 400
buffer_store_dword v8, v20, s[0:3], 0 offen
v_mov_b32_e32 v1, 0x00018000
v_mov_b32_e32 v2, 0x80010000
v_mov_b32_e32 v8, 0xdeadbeef
v_cvt_pknorm_u16_f16 v8, v1, v2 clamp
v_mov_b32_e32 v20, 404
buffer_store_dword v8, v20, s[0:3], 0 offen
v_mov_b32_e32 v1, 0x3c003400
v_mov_b32_e32 v2, 0xbc003801
v_mov_b32_e32 v8, 0xdeadbeef
v_cvt_pknorm_u16_f16 v8, |v1|, -v2 op_sel:[1,1,0] clamp
v_mov_b32_e32 v20, 408
buffer_store_dword v8, v20, s[0:3], 0 offen
v_mov_b32_e32 v1, 0x3c003400
v_mov_b32_e32 v2, 0xbc003801
v_mov_b32_e32 v8, 0xdeadbeef
v_cvt_pknorm_u16_f16 v8, -|v1|, |v2| op_sel:[0,1,0] clamp
v_mov_b32_e32 v20, 412
buffer_store_dword v8, v20, s[0:3], 0 offen
v_mov_b32_e32 v1, 0x3c003400
v_mov_b32_e32 v2, 0xbc003801
v_mov_b32_e32 v8, 0xdeadbeef
v_cvt_pknorm_u16_f16 v8, |v1|, -v2 op_sel:[1,1,1] clamp
v_mov_b32_e32 v20, 416
buffer_store_dword v8, v20, s[0:3], 0 offen
v_mov_b32_e32 v8, 0xdeadbeef
v_cvt_pknorm_u16_f16 v8, 0.5, -0.5 clamp
v_mov_b32_e32 v20, 420
buffer_store_dword v8, v20, s[0:3], 0 offen
v_mov_b32_e32 v2, 0xbc003801
v_mov_b32_e32 v8, 0xdeadbeef
v_cvt_pknorm_u16_f16 v8, 0.5, v2 op_sel:[1,0,0] clamp
v_mov_b32_e32 v20, 424
buffer_store_dword v8, v20, s[0:3], 0 offen
v_mov_b32_e32 v2, 0xbc003801
v_mov_b32_e32 v8, 0xdeadbeef
v_cvt_pknorm_u16_f16 v8, 1.0, v2 op_sel:[1,0,0] clamp
v_mov_b32_e32 v20, 428
buffer_store_dword v8, v20, s[0:3], 0 offen
v_mov_b32_e32 v2, 0xbc003801
v_mov_b32_e32 v8, 0xdeadbeef
v_cvt_pknorm_u16_f16 v8, -0.5, v2 clamp
v_mov_b32_e32 v20, 432
buffer_store_dword v8, v20, s[0:3], 0 offen
v_mov_b32_e32 v8, 0xdeadbeef
v_cvt_pknorm_u16_f16 v8, 1, -1 clamp
v_mov_b32_e32 v20, 436
buffer_store_dword v8, v20, s[0:3], 0 offen
v_mov_b32_e32 v2, 0xbc003801
v_mov_b32_e32 v8, 0xdeadbeef
v_cvt_pknorm_u16_f16 v8, 0x3800, v2 clamp
v_mov_b32_e32 v20, 440
buffer_store_dword v8, v20, s[0:3], 0 offen
v_mov_b32_e32 v2, 0xbc003801
v_mov_b32_e32 v8, 0xdeadbeef
v_cvt_pknorm_u16_f16 v8, 0x3800, v2 op_sel:[1,0,0] clamp
v_mov_b32_e32 v20, 444
buffer_store_dword v8, v20, s[0:3], 0 offen
s_mov_b32 s4, 0x3c003400
v_mov_b32_e32 v2, 0xbc003801
v_mov_b32_e32 v8, 0xdeadbeef
v_cvt_pknorm_u16_f16 v8, s4, v2 op_sel:[1,0,0] clamp
v_mov_b32_e32 v20, 448
buffer_store_dword v8, v20, s[0:3], 0 offen
s_mov_b32 s4, 0x3c003400
v_mov_b32_e32 v2, 0xbc003801
v_mov_b32_e32 v8, 0xdeadbeef
v_cvt_pknorm_u16_f16 v8, s4, v2 op_sel:[1,0,1] clamp
v_mov_b32_e32 v20, 452
buffer_store_dword v8, v20, s[0:3], 0 offen
v_mov_b32_e32 v8, 0x3c003400
v_mov_b32_e32 v2, 0xbc003801
v_cvt_pknorm_u16_f16 v8, v8, v2 op_sel:[0,0,1] clamp
v_mov_b32_e32 v20, 456
buffer_store_dword v8, v20, s[0:3], 0 offen
v_mov_b32_e32 v1, 0x3c003400
v_mov_b32_e32 v8, 0xbc003801
v_cvt_pknorm_u16_f16 v8, v1, v8 op_sel:[0,0,1] clamp
v_mov_b32_e32 v20, 460
buffer_store_dword v8, v20, s[0:3], 0 offen
s_endpgm
