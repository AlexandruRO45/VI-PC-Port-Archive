; f16 denormal operands of v_mul_f16. RDNA2's f16 MODE keeps denormals, so a denormal operand counts at its value
; (mantissa * 2^-24), and every module declares OpExecutionMode DenormPreserve 16 for it (SHADER-526). The operand
; comes from a VGPR, from an SGPR and as a negative denormal; the destinations start at 0, so each word is the low
; half the op writes. v9 holds 1024.0 as an f16 (0x6400), the multiplier.
;   x: v_mul_f16 v2, 1024.0, v1   v1 = 0x00000001 (2^-24, the smallest denormal)       -> 2^-14 = 0x0400
;   y: v_mul_f16 v3, 1024.0, v6   v6 = 0x00008001 (-2^-24)                             -> -2^-14 = 0x8400
;   z: v_mul_f16 v4, s8, v9       s8 = 0x000003ff (1023 * 2^-24, the largest denormal) -> 1023 * 2^-14 = 0x2bfe
;   w: v_mul_f16 v5, 1024.0, v7   v7 = 0x00000400 (2^-14, the smallest normal: the control) -> 2^-4 = 0x2c00
; A widening that flushes the denormal gives 0x0000 in x, y and z and leaves the control at 0x2c00.
; Expected mrt0 bits: 0x00000400, 0x00008400, 0x00002bfe, 0x00002c00 (all four exact in f16)
v_mov_b32 v1, 0x00000001
v_mov_b32 v6, 0x00008001
v_mov_b32 v7, 0x00000400
v_mov_b32 v9, 0x00006400
s_mov_b32 s8, 0x000003ff
v_mov_b32 v2, 0
v_mov_b32 v3, 0
v_mov_b32 v4, 0
v_mov_b32 v5, 0
v_mul_f16 v2, 1024.0, v1
v_mul_f16 v3, 1024.0, v6
v_mul_f16 v4, s8, v9
v_mul_f16 v5, 1024.0, v7
exp mrt0 v2, v3, v4, v5 done vm
s_endpgm
