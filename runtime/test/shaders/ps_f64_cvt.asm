; v_cvt_f64_i32, v_cvt_f64_u32, v_cvt_f64_f32 and v_cvt_i32_f64 (RDNA2 ISA VOP1): every i32, u32 and f32 is exact in
; binary64, so the low words show a conversion that went through f32 by mistake.
; w0: v_cvt_f64_i32(-1) = -1.0 (high word 0xbff00000).
; w1: v_cvt_f64_u32(0xffffffff) = 4294967295.0 = 0x41efffff_ffe00000: the low word (through f32 it would be 0).
; w2: v_cvt_f64_f32(0x3dcccccd, 0.1f) = 0x3fb99999_a0000000: the low word.
; w3: v_cvt_i32_f64(-inf) saturates to 0x80000000.
; Needs shaderFloat64; vk_harness exits 77 (SKIP) on a device without it.
; Expected mrt0 bits: 0xbff00000, 0xffe00000, 0xa0000000, 0x80000000
v_mov_b32 v0, -1
v_cvt_f64_i32_e32 v[2:3], v0
v_mov_b32 v0, 0xffffffff
v_cvt_f64_u32_e32 v[4:5], v0
v_mov_b32 v0, 0x3dcccccd
v_cvt_f64_f32_e32 v[6:7], v0
v_mov_b32 v0, 0
v_mov_b32 v1, 0xfff00000
v_cvt_i32_f64_e32 v8, v[0:1]
exp mrt0 v3, v4, v6, v8 done vm
s_endpgm
