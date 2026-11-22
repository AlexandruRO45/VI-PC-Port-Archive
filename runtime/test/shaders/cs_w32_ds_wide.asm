; GPU test for ds_write_b128, ds_read_b128, ds_write_b96 and ds_read_b96, which only lds_fuzz.py checks on a GPU.
; RDNA2 ISA pseudocode (DS): ds_write_b96 / ds_write_b128 store VDATA[95:0] / VDATA[127:0] at ADDR, ADDR + 4, ...
; (low dword first), and ds_read_b96 / ds_read_b128 return the same dwords in consecutive VGPRs.
; LDS is 4 KiB (RSRC2 0x40088); lane 0 only. The dwords at 0x5C and 0x6C are seeded 0xAAAA0001 and 0xBBBB0002, next to
; the 96-bit element at 0x60 (16-byte aligned), to show that a b96 access touches exactly three dwords.
;   ds_write_b128 at 0x40 of 0x01020304 0x05060708 0x090A0B0C 0x0D0E0F10, read back with ds_read_b128: out[0..3]
;   ds_write_b96  at 0x60 of 0x11111111 0x22222222 0x33333333, read back with ds_read_b96: out[4..6]
;   out[7] = dword 0x5C ^ dword 0x6C = 0xAAAA0001 ^ 0xBBBB0002 = 0x11110003 (both still the seeds)
; Run: vk_compute_test cs_w32_ds_wide.spv 1 dump
;   -> out[0..7] = 0x01020304 0x05060708 0x090A0B0C 0x0D0E0F10 0x11111111 0x22222222 0x33333333 0x11110003
; rsrc2: 0x40088
; group: 32,1,1
; mattr: +wavefrontsize32,-wavefrontsize64
v_cmpx_eq_u32_e32 0, v0
v_mov_b32_e32 v1, 0
v_mov_b32_e32 v4, 0x01020304
v_mov_b32_e32 v5, 0x05060708
v_mov_b32_e32 v6, 0x090a0b0c
v_mov_b32_e32 v7, 0x0d0e0f10
v_mov_b32_e32 v8, 0x11111111
v_mov_b32_e32 v9, 0x22222222
v_mov_b32_e32 v10, 0x33333333
v_mov_b32_e32 v2, 0xaaaa0001
v_mov_b32_e32 v3, 0xbbbb0002
ds_write_b32 v1, v2 offset:0x5c
ds_write_b32 v1, v3 offset:0x6c
s_waitcnt lgkmcnt(0)
ds_write_b128 v1, v[4:7] offset:0x40
ds_write_b96 v1, v[8:10] offset:0x60
s_waitcnt lgkmcnt(0)
ds_read_b128 v[12:15], v1 offset:0x40
ds_read_b96 v[16:18], v1 offset:0x60
ds_read_b32 v20, v1 offset:0x5c
ds_read_b32 v21, v1 offset:0x6c
s_waitcnt lgkmcnt(0)
v_xor_b32_e32 v20, v20, v21
v_mov_b32_e32 v40, 0
buffer_store_dword v12, v40, s[0:3], 0 offen
v_mov_b32_e32 v40, 4
buffer_store_dword v13, v40, s[0:3], 0 offen
v_mov_b32_e32 v40, 8
buffer_store_dword v14, v40, s[0:3], 0 offen
v_mov_b32_e32 v40, 12
buffer_store_dword v15, v40, s[0:3], 0 offen
v_mov_b32_e32 v40, 16
buffer_store_dword v16, v40, s[0:3], 0 offen
v_mov_b32_e32 v40, 20
buffer_store_dword v17, v40, s[0:3], 0 offen
v_mov_b32_e32 v40, 24
buffer_store_dword v18, v40, s[0:3], 0 offen
v_mov_b32_e32 v40, 28
buffer_store_dword v20, v40, s[0:3], 0 offen
s_endpgm
