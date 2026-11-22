; Scalar atomics (GFX10.1 SMEM s_atomic_* / s_buffer_atomic_*, gfx1010-gfx1013 only: llvm-mc rejects them for
; gfx1030/gfx1036, so the RDNA2 oracle can't run them). RDNA1 ISA: one atomic per wave (a scalar instruction) at the
; address an s_load / s_buffer_load of the same operands reads; with glc SDATA[0] gets the value before the op.
; cmpswap takes SDATA as {DATA, CMP}; inc wraps to 0 when the old value is >= DATA. The translator has the invocation
; in hardware lane 0 do it and gives every lane the returned value. Until this change they raised Unsupported.
; One wave32 group. Lane 0 first writes the targets out[4..7] = 100, 7, 0xf0, 5 (a vector store, before the atomics in
; the same invocation), then:
;   s_buffer_atomic_add     s20 = 3,                 out[4] (0x10) glc -> s20 = 100,  out[4] = 103
;   s_buffer_atomic_sub     s21 = 2,                 out[5] (0x14) glc -> s21 = 7,    out[5] = 5
;   s_buffer_atomic_cmpswap s[22:23] = {0x0f, 0xf0}, out[6] (0x18) glc -> s22 = 0xf0, out[6] = 0x0f (0xf0 == CMP)
;   s_atomic_inc            s24 = 6, pointer s[8:9], out[7] (0x1c) glc -> s24 = 5,    out[7] = 6 (5 < 6)
; and lane 31 stores s20, s21, s22, s24 to out[0..3] (the broadcast reached the last lane).
; Done once per lane instead of once per wave, out[4] would be 100 + 32 * 3.
;   out[0..7] = 100 7 0xf0 5 103 5 0x0f 6
; group: 32,1,1
; Run: vk_compute_test cs_w32_scalar_atomic.spv 1 dump -> out[0..7] = 100 7 240 5 103 5 15 6
; mattr: +wavefrontsize32,-wavefrontsize64
s_mov_b32 s8, s0
s_and_b32 s9, s1, 0xffff
v_mov_b32_e32 v1, 100
v_mov_b32_e32 v2, 7
v_mov_b32_e32 v3, 0xf0
v_mov_b32_e32 v4, 5
v_mov_b32_e32 v5, 16
v_cmpx_eq_u32_e32 0, v0
buffer_store_dwordx4 v[1:4], v5, s[0:3], 0 offen
s_mov_b32 exec_lo, -1
s_mov_b32 s20, 3
s_mov_b32 s21, 2
s_mov_b32 s22, 0x0f
s_mov_b32 s23, 0xf0
s_mov_b32 s24, 6
s_buffer_atomic_add s20, s[0:3], 0x10 glc
s_buffer_atomic_sub s21, s[0:3], 0x14 glc
s_buffer_atomic_cmpswap s[22:23], s[0:3], 0x18 glc
s_atomic_inc s24, s[8:9], 0x1c glc
s_waitcnt lgkmcnt(0)
v_mov_b32_e32 v1, s20
v_mov_b32_e32 v2, s21
v_mov_b32_e32 v3, s22
v_mov_b32_e32 v4, s24
v_mov_b32_e32 v5, 0
v_cmpx_eq_u32_e32 31, v0
buffer_store_dwordx4 v[1:4], v5, s[0:3], 0 offen
s_endpgm
