; v_fma_f64, v_mul_f64, v_max_f64 and v_cvt_i32_f64 (RDNA2 ISA VOP3 / VOP1). Doubles are register pairs, low word first.
; w0: fma(a, a, c) with a = 1 + 2^-30, c = -(1 + 2^-29) is exactly 2^-60 (high word 0x3c300000) with one rounding;
;     rounding a*a first gives 1 + 2^-29 and a result of 0, so the word shows whether the driver fused.
; w1: 1.5 * -2.25 = -3.375 (high word 0xc00b0000).
; w2: v_max_f64(2.0, qNaN) = 2.0: a NaN operand gives the other one (high word 0x40000000).
; w3: v_cvt_i32_f64(3e9) saturates to 0x7fffffff (out of range for i32; NaN would give 0).
; Needs shaderFloat64; vk_harness exits 77 (SKIP) on a device without it.
; Expected mrt0 bits: 0x3c300000, 0xc00b0000, 0x40000000, 0x7fffffff
v_mov_b32 v0, 0x400000
v_mov_b32 v1, 0x3ff00000
v_mov_b32 v2, 0x800000
v_mov_b32 v3, 0xbff00000
v_fma_f64 v[4:5], v[0:1], v[0:1], v[2:3]
v_mov_b32 v0, 0
v_mov_b32 v1, 0x3ff80000
v_mov_b32 v2, 0
v_mov_b32 v3, 0xc0020000
v_mul_f64 v[6:7], v[0:1], v[2:3]
v_mov_b32 v0, 0
v_mov_b32 v1, 0x40000000
v_mov_b32 v2, 0
v_mov_b32 v3, 0x7ff80000
v_max_f64 v[8:9], v[0:1], v[2:3]
v_mov_b32 v0, 0xc0000000
v_mov_b32 v1, 0x41e65a0b
v_cvt_i32_f64_e32 v10, v[0:1]
exp mrt0 v5, v7, v9, v10 done vm
s_endpgm
