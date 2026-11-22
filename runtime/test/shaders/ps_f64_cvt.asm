; v_cvt_f32_f64 rounding without denormals, the same words on native doubles and under the translator's --f64-soft.
; w0: 1 + 2^-24 is a tie between two binary32 values, nearest even gives 1.0 (0x3f800000; up would be 0x3f800001).
; w1: 1 + 2^-24 + 2^-52 is just above the tie: 0x3f800001.
; w2: -DBL_MAX overflows to -inf (0xff800000). w3: -|0.75| with the neg/abs modifiers is -0.75 (0xbf400000).
; Expected mrt0 bits: 0x3f800000, 0x3f800001, 0xff800000, 0xbf400000
v_mov_b32 v0, 0x10000000
v_mov_b32 v1, 0x3ff00000
v_cvt_f32_f64 v2, v[0:1]
v_mov_b32 v3, 0x10000001
v_mov_b32 v4, 0x3ff00000
v_cvt_f32_f64 v5, v[3:4]
v_mov_b32 v6, -1
v_mov_b32 v7, 0xffefffff
v_cvt_f32_f64 v8, v[6:7]
v_mov_b32 v9, 0
v_mov_b32 v10, 0x3fe80000
v_cvt_f32_f64 v11, -|v[9:10]|
exp mrt0 v2, v5, v8, v11 done vm
s_endpgm
