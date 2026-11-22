; Scalar atomics in a wave64 (GFX10.1 SMEM s_atomic_* / s_buffer_atomic_*, gfx1010-gfx1013 only, so the RDNA2 oracle
; can't run them). RDNA1 ISA: one atomic per wave; with glc SDATA gets the value before the op in every lane. The
; translator has the invocation in hardware lane 0 do it (one invocation of the 64, also when the wave spans two
; 32-wide subgroups) and sends the returned value to every lane: a subgroup shuffle on a 64-wide subgroup, an exchange
; round under --wave64-emulate. Until this change they raised Unsupported.
; One wave64 group. Lane 0 first writes the targets out[4..7] = 100, 0xf0, 0xf0, 5 (a vector store, before the atomics
; in the same invocation), then:
;   s_buffer_atomic_add     s20 = 3,                 out[4] (0x10) glc    -> s20 = 100,  out[4] = 103
;   s_buffer_atomic_cmpswap s[22:23] = {0x0f, 0xf0}, out[5] (0x14) glc    -> s22 = 0xf0, out[5] = 0x0f
;   s_buffer_atomic_or_x2   s[24:25] = {0xff, 0x100}, out[6..7] (0x18) glc -> s[24:25] = {0xf0, 5},
;                                                                           out[6..7] = {0xff, 0x105}
; Lane 0 stores s20 to out[0]; lane 63 stores s20, s22, s25 to out[1..3] (the value reached the upper half).
; Done once per lane, out[4] would be 100 + 64 * 3; returned only to lanes 0-31, out[1..3] would be 0.
;   out[0..7] = 100 100 0xf0 5 103 0x0f 0xff 0x105
; Run: VKC_SUBGROUP_SIZE=64 vk_compute_test cs_w64_scalar_atomic.spv 1 dump --require-wave64
;   (or translated with --wave64-emulate, on any subgroup size) -> out[0..7] = 100 100 240 5 103 15 255 261
; mattr: +wavefrontsize64,-wavefrontsize32
v_mov_b32_e32 v1, 100
v_mov_b32_e32 v2, 0xf0
v_mov_b32_e32 v3, 0xf0
v_mov_b32_e32 v4, 5
v_mov_b32_e32 v5, 16
v_cmpx_eq_u32_e32 0, v0
buffer_store_dwordx4 v[1:4], v5, s[0:3], 0 offen
s_mov_b64 exec, -1
s_mov_b32 s20, 3
s_mov_b32 s22, 0x0f
s_mov_b32 s23, 0xf0
s_mov_b32 s24, 0xff
s_mov_b32 s25, 0x100
s_buffer_atomic_add s20, s[0:3], 0x10 glc
s_buffer_atomic_cmpswap s[22:23], s[0:3], 0x14 glc
s_buffer_atomic_or_x2 s[24:25], s[0:3], 0x18 glc
s_waitcnt lgkmcnt(0)
v_mov_b32_e32 v1, s20
v_mov_b32_e32 v2, 0
v_cmpx_eq_u32_e32 0, v0
buffer_store_dword v1, v2, s[0:3], 0 offen
s_mov_b64 exec, -1
v_mov_b32_e32 v2, s22
v_mov_b32_e32 v3, s25
v_mov_b32_e32 v5, 4
v_cmpx_eq_u32_e32 63, v0
buffer_store_dwordx3 v[1:3], v5, s[0:3], 0 offen
s_endpgm
