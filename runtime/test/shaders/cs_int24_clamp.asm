; VOP3 clamp on the 24-bit multiplies (EGEN-INT24-CLAMP). RDNA2 ISA: v_mul_u32_u24 D = S0[23:0] * S1[23:0],
; v_mad_u32_u24 D = S0[23:0] * S1[23:0] + S2, the i24 forms sign-extend bit 23. Without clamp D is the low 32 bits.
; With clamp the exact value is saturated instead: u24 to [0, 0xFFFFFFFF], i24 to [0x80000000, 0x7FFFFFFF]
; (measured on a gfx1036 Radeon iGPU with raw wave32 kernels, the same rows plus 512 random lanes). The translator
; ignored the clamp bit on these four ops until this test; the words it gave then are in [brackets].
; Lane 0 of a wave32 group stores eight dwords:
;   out[0] = v_mad_u32_u24 0xffffff, 0xffffff, 0xffffffff clamp: 0xfffffe000001 + 0xffffffff > 2^32 - 1
;            -> 0xffffffff                                                        [0xfe000000, the low 32 bits]
;   out[1] = v_mad_u32_u24 1, 1, 0xffffffff clamp: 1 + 0xffffffff = 2^32, the carry of the add alone
;            -> 0xffffffff                                                        [0x00000000]
;   out[2] = v_mul_u32_u24_e64 0xffffff, 0xffffff clamp: 0xfffffe000001 -> 0xffffffff   [0xfe000001]
;   out[3] = v_mad_i32_i24 0x7fffff, 0x7fffff, 0x7fffffff clamp: 8388607^2 + 2147483647 > 2^31 - 1
;            -> 0x7fffffff                                                        [0x7f000000]
;   out[4] = v_mad_i32_i24 0xffffff, 2, 0x80000000 clamp: (-1) * 2 + (-2^31) < -2^31, the add alone overflows
;            -> 0x80000000                                                        [0x7ffffffe]
;   out[5] = v_mul_i32_i24_e64 0x800000, 0x800000 clamp: (-2^23)^2 = 2^46 -> 0x7fffffff     [0x00000000]
;   out[6] = v_mul_i32_i24_e64 0x800000, 0x7fffff clamp: -2^23 * (2^23 - 1) = -(2^46 - 2^23)
;            -> 0x80000000                                                        [0x00800000]
;   out[7] = v_mad_i32_i24 0xfffffd, 5, 1 clamp: (-3) * 5 + 1 = -14 is in range -> 0xfffffff2 (unchanged)
; group: 32,1,1
; Run: vk_compute_test cs_int24_clamp.spv 1 dump -> out[0..7] = 0xffffffff 0xffffffff 0xffffffff 0x7fffffff
;                                                               0x80000000 0x7fffffff 0x80000000 0xfffffff2
; mattr: +wavefrontsize32,-wavefrontsize64
v_mov_b32_e32 v1, 0xffffff
v_mov_b32_e32 v2, 1
v_mov_b32_e32 v3, -1
v_mov_b32_e32 v4, 0x7fffff
v_mov_b32_e32 v5, 0x7fffffff
v_mov_b32_e32 v6, 2
v_bfrev_b32_e32 v7, 1
v_mov_b32_e32 v8, 0x800000
v_mov_b32_e32 v9, 0xfffffd
v_mov_b32_e32 v10, 5
v_mad_u32_u24 v20, v1, v1, v3 clamp
v_mad_u32_u24 v21, v2, v2, v3 clamp
v_mul_u32_u24_e64 v22, v1, v1 clamp
v_mad_i32_i24 v23, v4, v4, v5 clamp
v_mad_i32_i24 v24, v1, v6, v7 clamp
v_mul_i32_i24_e64 v25, v8, v8 clamp
v_mul_i32_i24_e64 v26, v8, v4 clamp
v_mad_i32_i24 v27, v9, v10, v2 clamp
s_mov_b32 exec_lo, 1
v_mov_b32_e32 v30, 0
v_mov_b32_e32 v31, 16
buffer_store_dwordx4 v[20:23], v30, s[0:3], 0 offen
buffer_store_dwordx4 v[24:27], v31, s[0:3], 0 offen
s_endpgm
