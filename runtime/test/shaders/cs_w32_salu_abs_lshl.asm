; SHADER-258/264: s_absdiff_i32, s_mul_hi_i32 and s_lshl1..4_add_u32 (RDNA2 ISA SOP1/SOP2).
; Only lane 0 stores, so the scalar results are written once.
;   absdiff(0x7fffffff, 0x80000000): difference wraps to 0xffffffff, |.| -> 1, SCC = 1
;   absdiff(0x80000000, 0x80000000): difference 0 -> 0
;   mul_hi_i32(0x40000000, 0x40000000): 2^60 -> high word 0x10000000
;   lshl1_add_u32(0x40000000, 0x1000): (S0 << 1) + S1 = 0x80001000, nothing leaves the 64-bit sum -> SCC 0
;   lshl2_add_u32(0x80000000, 0): (S0 << 2) = 0 with bits shifted out -> SCC 1
;   lshl3_add_u32(0x20000000, 1): (S0 << 3) = 0 with bits shifted out, + 1 -> 1, SCC 1
;   lshl4_add_u32(0x0fffffff, 0): 0xfffffff0, SCC 0
; out[0..7] = 1 1 0 0x10000000 0x80001000 0 1 0xfffffff0
; group: 32,1,1
; Run: vk_compute_test cs_w32_salu_abs_lshl.spv 1 dump
; mattr: +wavefrontsize32,-wavefrontsize64
s_mov_b32 s10, 0x7fffffff
s_mov_b32 s11, 0x80000000
s_absdiff_i32 s12, s10, s11
s_cselect_b32 s13, 1, 0
s_absdiff_i32 s14, s11, s11
s_mov_b32 s15, 0x40000000
s_mul_hi_i32 s16, s15, s15
s_mov_b32 s17, 0x1000
s_lshl1_add_u32 s18, s15, s17
s_lshl2_add_u32 s19, s11, 0
s_mov_b32 s20, 0x20000000
s_lshl3_add_u32 s21, s20, 1
s_mov_b32 s22, 0x0fffffff
s_lshl4_add_u32 s23, s22, 0
v_mov_b32_e32 v10, s12
v_mov_b32_e32 v11, s13
v_mov_b32_e32 v12, s14
v_mov_b32_e32 v13, s16
v_mov_b32_e32 v14, s18
v_mov_b32_e32 v15, s19
v_mov_b32_e32 v16, s21
v_mov_b32_e32 v17, s23
v_cmpx_eq_u32_e32 0, v0
v_mov_b32_e32 v21, 0
buffer_store_dword v10, v21, s[0:3], 0 offen
v_mov_b32_e32 v21, 4
buffer_store_dword v11, v21, s[0:3], 0 offen
v_mov_b32_e32 v21, 8
buffer_store_dword v12, v21, s[0:3], 0 offen
v_mov_b32_e32 v21, 12
buffer_store_dword v13, v21, s[0:3], 0 offen
v_mov_b32_e32 v21, 16
buffer_store_dword v14, v21, s[0:3], 0 offen
v_mov_b32_e32 v21, 20
buffer_store_dword v15, v21, s[0:3], 0 offen
v_mov_b32_e32 v21, 24
buffer_store_dword v16, v21, s[0:3], 0 offen
v_mov_b32_e32 v21, 28
buffer_store_dword v17, v21, s[0:3], 0 offen
s_endpgm
