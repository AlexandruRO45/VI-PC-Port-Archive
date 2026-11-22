; v_cvt_f32_f64 rounding and overflow (EGEN-GPU-CVT16). RDNA 2 ISA: D.f32 = f64_to_f32(S0), IEEE round to nearest even,
; a value too large for f32 gives infinity. Needs shaderFloat64; vk_harness exits 77 (SKIP) on a device without it.
; Doubles are written low word first (v_mov to the pair). An f32 has 23 mantissa bits, so near 1.0 the spacing is 2^-23.
;   x = 1 + 2^-24 (hi 0x3ff00000, lo 0x10000000): the tie between 1.0 (even) and 1 + 2^-23; even -> 1.0 = 0x3f800000
;   y = 1 + 2^-24 + 2^-40 (lo 0x10001000): above the tie -> 1 + 2^-23 = 0x3f800001
;   z = 1 + 3 * 2^-24 (lo 0x30000000): the tie between 1 + 2^-23 (odd) and 1 + 2^-22 (even) -> 1 + 2^-22 = 0x3f800002
;   w = -2^200 (hi 0xcc700000: sign, exponent 1023 + 200 = 0x4c7): far past the f32 range, -> -inf = 0xff800000
; Expected mrt0 bits: 0x3f800000, 0x3f800001, 0x3f800002, 0xff800000
v_mov_b32 v0, 0x10000000
v_mov_b32 v1, 0x3ff00000
v_mov_b32 v2, 0x10001000
v_mov_b32 v3, 0x3ff00000
v_mov_b32 v4, 0x30000000
v_mov_b32 v5, 0x3ff00000
v_mov_b32 v6, 0
v_mov_b32 v7, 0xcc700000
v_cvt_f32_f64_e32 v8, v[0:1]
v_cvt_f32_f64_e32 v9, v[2:3]
v_cvt_f32_f64_e32 v10, v[4:5]
v_cvt_f32_f64_e32 v11, v[6:7]
exp mrt0 v8, v9, v10, v11 done vm
s_endpgm