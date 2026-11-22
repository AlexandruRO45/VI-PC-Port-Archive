; The VOP3 clamp bit on a NaN, a -0 and an infinity (SHADER-186), inputs read from memory at run time (not inline
; constants, which the driver could fold). The V# in s[0:3] points at 0x1000 (num_records 32); the test seeds
; 0x1000..0x1010 with a..e = 0x7fc10001 0x80000000 0x80007e11 0x3c003c00 0x7f800000
; (f32 NaN, f32 -0, f16 pair (lo NaN, hi -0), f16 pair (1.0, 1.0), f32 +inf).
; Expected words, from the RDNA2 measurements in docs/shader_validation.md (gfx1036, `f32_fuzz.py --oracle hw` and
; the f16 / mix-op runs; gfx1150 agrees), in the MODE the translator assumes (IEEE 0, DX10_CLAMP 1):
;   clamp(NaN * 1.0)              = 0x00000000  (a NaN clamps to +0 with DX10_CLAMP 1)
;   clamp((-0 + -0) mul:2)        = 0x00000000  (-0 clamps to +0 in every MODE)
;   v_pk_mul_f16 (NaN, -0) * 1.0  = 0x00000000  (each half on its own)
;   clamp(+inf * 1.0)             = 0x3f800000  (an infinity clamps like any value)
; Expected mrt0 bits: 0x00000000, 0x00000000, 0x00000000, 0x3f800000
; Before SHADER-186 the translator emitted GLSL.std.450 FClamp, whose NaN result is undefined and whose -0 result
; may keep its sign.
s_buffer_load_dwordx8 s[4:11], s[0:3], 0
s_waitcnt lgkmcnt(0)
v_mov_b32_e32 v1, s4
v_mov_b32_e32 v2, s5
v_mov_b32_e32 v3, s6
v_mov_b32_e32 v4, s7
v_mov_b32_e32 v9, s8
v_mul_f32_e64 v5, v1, 1.0 clamp
v_add_f32_e64 v6, v2, v2 clamp mul:2
v_pk_mul_f16 v7, v3, v4 clamp
v_mul_f32_e64 v8, v9, 1.0 clamp
exp mrt0 v5, v6, v7, v8 done vm
s_endpgm
