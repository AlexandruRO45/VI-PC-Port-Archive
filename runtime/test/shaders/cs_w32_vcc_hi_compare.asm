; group: 32,1,1
; mattr: +wavefrontsize32,-wavefrontsize64
; VCC_LO records lane 2. Only lanes 0, 2, 4 and 6 run the second compare,
; and lanes below 6 pass. The resulting VCC_HI word must be 0x15.
; Each lane stores 8 for the exact word, 2 for its VCC_LO bit and 1 for its VCC_HI bit.
; Expected out[0..7] = 9 8 11 8 9 8 8 8.
v_cmp_eq_u32_e64 vcc_lo, v0, 2
s_mov_b32 vcc_hi, -1
s_mov_b32 s6, exec_lo
s_mov_b32 exec_lo, 0x55
v_cmp_lt_u32_e64 vcc_hi, v0, 6
s_mov_b32 exec_lo, s6
s_cmp_eq_u32 vcc_hi, 0x15
s_cselect_b32 s7, 8, 0
v_mov_b32_e32 v1, 1
v_mov_b32_e32 v2, 2
v_cndmask_b32_e64 v3, 0, v1, vcc_hi
v_cndmask_b32_e64 v4, 0, v2, vcc_lo
v_or_b32_e32 v3, v4, v3
v_or_b32_e32 v3, s7, v3
v_lshlrev_b32_e32 v5, 2, v0
buffer_store_dword v3, v5, s[0:3], 0 offen
s_endpgm
