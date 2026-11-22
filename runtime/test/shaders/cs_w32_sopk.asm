; Scalar instructions with 16-bit immediates (AMD RDNA2 ISA, SOPK).
; Lane 0 stores eight words:
;   out[0] = 0xFFFF8000  (s_movk sign-extends -32768)
;   out[1] = 0           (s_cmpk_eq_u32 zero-extends 0x8000 to +32768)
;   out[2] = 0xFFFFFFEB  (7 * -3)
;   out[3] = 0           (s_mulk preserves clear SCC despite a nonzero result)
;   out[4] = 0x00018000  (-32768 * -3)
;   out[5] = 1           (s_cmpk_eq_i32 sign-extends 0x8000 to -32768; s_mulk preserves set SCC)
;   out[6] = 0x80000000  (0x7FFFFFFF + 1 wraps)
;   out[7] = 1           (s_addk sets a previously clear SCC on signed overflow)
; group: 32,1,1
; Run: vk_compute_test cs_w32_sopk.spv 1 dump
; mattr: +wavefrontsize32,-wavefrontsize64
s_movk_i32 s8, -32768
s_mov_b32 s7, s8
s_cmpk_eq_u32 s8, 0x8000
s_cselect_b32 s9, 1, 0
s_movk_i32 s10, 7
s_mulk_i32 s10, -3
s_cselect_b32 s11, 1, 0
s_cmpk_eq_i32 s8, -32768
s_mulk_i32 s8, -3
s_cselect_b32 s12, 1, 0
s_mov_b32 s13, 0x7fffffff
s_cmp_eq_u32 s13, 0
s_addk_i32 s13, 1
s_cselect_b32 s14, 1, 0

v_cmpx_eq_u32_e32 0, v0
v_mov_b32_e32 v1, s7
v_mov_b32_e32 v2, s9
v_mov_b32_e32 v3, s10
v_mov_b32_e32 v4, s11
v_mov_b32_e32 v5, s8
v_mov_b32_e32 v6, s12
v_mov_b32_e32 v7, s13
v_mov_b32_e32 v8, s14
v_mov_b32_e32 v9, 0
buffer_store_dwordx4 v[1:4], v9, s[0:3], 0 offen
buffer_store_dwordx4 v[5:8], v9, s[0:3], 0 offen offset:16
s_endpgm
