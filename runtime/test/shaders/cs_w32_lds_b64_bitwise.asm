; SHADER-313: ds_and_rtn_b64 / ds_or_rtn_b64 / ds_xor_rtn_b64, the 64-bit LDS bitwise ops (RDNA2 ISA DS).
; LDS bytes 0..7 hold A = 0xFFFF0000AAAAAAAA. Only lane 0 runs the atomics, so the order is fixed.
;   start            M = FFFF0000:AAAAAAAA
;   and  B=0F0F0F0F:55555555 -> M = 0F0F0000:00000000, vdst = A
;   or   C=0000FFFF:FFFF0000 -> M = 0F0FFFFF:FFFF0000, vdst = 0F0F0000:00000000
;   xor  D=FFFFFFFF:0000FFFF -> M = F0F00000:FFFFFFFF, vdst = 0F0FFFFF:FFFF0000
;   read back M
; out[0..7] = 0xAAAAAAAA 0xFFFF0000 0x00000000 0x0F0F0000 0xFFFF0000 0x0F0FFFFF 0xFFFFFFFF 0xF0F00000
; group: 32,1,1
; Run: vk_compute_test cs_w32_lds_b64_bitwise.spv 1 dump
; mattr: +wavefrontsize32,-wavefrontsize64
v_mov_b32_e32 v1, 0
v_mov_b32_e32 v2, 0xaaaaaaaa
v_mov_b32_e32 v3, 0xffff0000
ds_write_b64 v1, v[2:3]
s_waitcnt lgkmcnt(0)
v_mov_b32_e32 v4, 0x55555555
v_mov_b32_e32 v5, 0x0f0f0f0f
v_mov_b32_e32 v6, 0xffff0000
v_mov_b32_e32 v7, 0x0000ffff
v_mov_b32_e32 v8, 0x0000ffff
v_mov_b32_e32 v9, 0xffffffff
v_cmpx_eq_u32_e32 0, v0
ds_and_rtn_b64 v[10:11], v1, v[4:5]
ds_or_rtn_b64 v[12:13], v1, v[6:7]
ds_xor_rtn_b64 v[14:15], v1, v[8:9]
s_waitcnt lgkmcnt(0)
ds_read_b64 v[16:17], v1
s_waitcnt lgkmcnt(0)
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
