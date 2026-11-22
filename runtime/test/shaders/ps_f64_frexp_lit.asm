; v_frexp_mant_f64 / v_frexp_exp_i32_f64 with a 32-bit literal source: llvm-mc encodes it as one literal dword, which
; is bits 63:32 of the binary64 operand (bits 31:0 zero), so 0x3ff80000 is 1.5 and 0xc0080000 is -3.0. Read as the
; low word it would be a denormal: mantissa +0, exponent 0. The mul:2 needs shaderFloat64, or --f64-soft.
; w0, w1: frexp_mant(1.5) = 0.75, mul:2 = 1.5 (0x00000000, 0x3ff80000). w2: frexp_exp(1.5) = 1. w3: frexp_exp(-3.0) = 2.
; Expected mrt0 bits: 0x00000000, 0x3ff80000, 0x00000001, 0x00000002
v_frexp_mant_f64_e64 v[0:1], 0x3ff80000 mul:2
v_frexp_exp_i32_f64_e64 v2, 0x3ff80000
v_frexp_exp_i32_f64_e64 v3, 0xc0080000
exp mrt0 v0, v1, v2, v3 done vm
s_endpgm
