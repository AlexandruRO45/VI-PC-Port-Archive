; group: 1,1,1
; mattr: +wavefrontsize32,-wavefrontsize64
; Packed 16-bit integer ops (v_pk_add/sub_u16/i16, v_pk_mul_lo_u16, v_pk_max_u16)
; Run: vk_compute_test cs_w32_pk_int16.spv 1 dump
; out[0..7] = ffff0000 ffffffff ffff0002 00000000 7fff0002 0001fffe 00010001 8000ffff (hex)
v_mov_b32_e32 v1, 0x7fff0001
v_mov_b32_e32 v2, 0x8000ffff
v_pk_add_u16 v3, v1, v2
v_pk_add_u16 v4, v1, v2 clamp
v_pk_sub_u16 v5, v1, v2
v_pk_sub_u16 v6, v1, v2 clamp
v_pk_add_i16 v7, v1, v1 clamp
v_pk_sub_i16 v8, v2, v1
v_pk_mul_lo_u16 v9, v1, v1
v_pk_max_u16 v10, v1, v2
buffer_store_dword v3, off, s[0:3], 0 offset:0
buffer_store_dword v4, off, s[0:3], 0 offset:4
buffer_store_dword v5, off, s[0:3], 0 offset:8
buffer_store_dword v6, off, s[0:3], 0 offset:12
buffer_store_dword v7, off, s[0:3], 0 offset:16
buffer_store_dword v8, off, s[0:3], 0 offset:20
buffer_store_dword v9, off, s[0:3], 0 offset:24
buffer_store_dword v10, off, s[0:3], 0 offset:28
s_endpgm