; Hand-written regression: lane mask in s11, scalar 5 in s10.
; Mask lanes are 0, 8, 16, ..., 56: the pair is 0x01010101_00000005.
; Each of the 64 lanes must store (unsigned leading bits, signed leading bits, SCC) = (7, 7, 1).
v_mbcnt_lo_u32_b32 v1, -1, 0
v_mbcnt_hi_u32_b32 v1, -1, v1
v_and_b32_e32 v3, 7, v1
v_cmp_eq_u32_e64 s[10:11], 0, v3
s_mov_b32 s10, 5
s_cmp_eq_u32 s10, 5
s_flbit_i32_b64 s20, s[10:11]
s_flbit_i32_i64 s21, s[10:11]
s_cselect_b32 s22, 1, 0
v_mul_u32_u24_e32 v21, 12, v1
v_mov_b32_e32 v20, s20
buffer_store_dword v20, v21, s[0:3], 0 offen
v_mov_b32_e32 v20, s21
buffer_store_dword v20, v21, s[0:3], 0 offen offset:4
v_mov_b32_e32 v20, s22
buffer_store_dword v20, v21, s[0:3], 0 offen offset:8
s_endpgm
