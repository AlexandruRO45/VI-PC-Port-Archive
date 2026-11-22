; v_cvt_f32_f64 with the f32 output modifiers: omod first, then clamp to [0, 1]. Only words SPIR-V defines on the
; native path (no NaN, no -0, no f32 denormal), so native doubles and --f64-soft give the same; under --f64-soft the
; omod and clamp are integer code (tools/soft_f64.py scale_f32 / clamp01_f32).
; w0: 0.75 mul:2 is 1.5 (0x3fc00000). w1: 0.75 clamp mul:2 is 1.0, the omod before the clamp (0x3f800000).
; w2: -0.5 clamp is +0 (0x00000000). w3: FLT_MAX mul:4 overflows to +inf (0x7f800000).
; Expected mrt0 bits: 0x3fc00000, 0x3f800000, 0x00000000, 0x7f800000
v_mov_b32 v0, 0
v_mov_b32 v1, 0x3fe80000
v_cvt_f32_f64_e64 v2, v[0:1] mul:2
v_cvt_f32_f64_e64 v3, v[0:1] clamp mul:2
v_mov_b32 v4, 0
v_mov_b32 v5, 0xbfe00000
v_cvt_f32_f64_e64 v6, v[4:5] clamp
v_mov_b32 v7, 0xe0000000
v_mov_b32 v8, 0x47efffff
v_cvt_f32_f64_e64 v9, v[7:8] mul:4
exp mrt0 v2, v3, v6, v9 done vm
s_endpgm
