; GPU test for ds_read_u8, ds_read_i8, ds_read_u16, ds_read_i16, ds_read_u16_d16, ds_read_u16_d16_hi, ds_mskor_b32,
; ds_mskor_rtn_b32 and ds_add_rtn_u32, which only lds_fuzz.py checks on a GPU (the 8-bit d16 reads are cs_w32_lds_u8_d16).
; RDNA2 ISA pseudocode (DS):
;   ds_read_u8 / i8 / u16 / i16   D = zero or sign extension of the byte or half at ADDR
;   ds_read_u16_d16               D[15:0]  = MEM[ADDR][15:0], D[31:16] kept      _d16_hi: D[31:16] = MEM[ADDR][15:0], D[15:0] kept
;   ds_mskor_b32 (_rtn_)          tmp = MEM[A]; MEM[A] = (tmp & ~DATA) | DATA2; the _rtn_ form returns tmp
;   ds_add_rtn_u32                tmp = MEM[A]; MEM[A] = tmp + DATA; returns tmp
; LDS is 4 KiB (RSRC2 0x40088); lane 0 only. Seeds: dword 0x00 = 0x8081F2F3 (bytes F3 F2 81 80), 0x04 = 0x7FFE8001,
; 0x20 = 0x12345678, 0x24 = 0xFFFFFFFF, 0x28 = 0xFFFFFFF0.
;   out[0]  ds_read_u8  at 1 (0xF2)                                                         0x000000F2
;   out[1]  ds_read_i8  at 1                                                                0xFFFFFFF2
;   out[2]  ds_read_u16 at 2 (0x8081)                                                       0x00008081
;   out[3]  ds_read_i16 at 2                                                                0xFFFF8081
;   out[4]  ds_read_u16_d16 at 4 (0x8001) into 0xAAAAAAAA                                   0xAAAA8001
;   out[5]  ds_read_u16_d16_hi at 6 (0x7FFE) into 0xBBBBBBBB                                0x7FFEBBBB
;   out[6]  ds_mskor_rtn_b32 at 0x20, DATA = 0x00FF00F0, DATA2 = 0xAB00CD05: returns 0x12345678 and stores
;           (0x12345678 & 0xFF00FF0F) | 0xAB00CD05 = 0xBB00DF0D; ds_mskor_b32 at 0x24, DATA = 0x0F0F0F0F, DATA2 = 0x01010101
;           stores (0xFFFFFFFF & 0xF0F0F0F0) | 0x01010101 = 0xF1F1F1F1.
;           out[6] = returned ^ dword 0x20 ^ dword 0x24 = 0x12345678 ^ 0xBB00DF0D ^ 0xF1F1F1F1                0x58C57884
;   out[7]  ds_add_rtn_u32 at 0x28, DATA = 0x20: returns 0xFFFFFFF0, stores 0xFFFFFFF0 + 0x20 = 0x00000010 (wraps);
;           out[7] = returned ^ dword 0x28 = 0xFFFFFFF0 ^ 0x10                                                  0xFFFFFFE0
; Run: vk_compute_test cs_w32_ds_subword_read.spv 1 dump
;   -> out[0..7] = 0x000000F2 0xFFFFFFF2 0x00008081 0xFFFF8081 0xAAAA8001 0x7FFEBBBB 0x58C57884 0xFFFFFFE0
; rsrc2: 0x40088
; group: 32,1,1
; mattr: +wavefrontsize32,-wavefrontsize64
v_cmpx_eq_u32_e32 0, v0
v_mov_b32_e32 v1, 0
v_mov_b32_e32 v2, 0x8081f2f3
v_mov_b32_e32 v3, 0x7ffe8001
v_mov_b32_e32 v4, 0x12345678
v_mov_b32_e32 v5, 0xffffffff
v_mov_b32_e32 v6, 0xfffffff0
ds_write_b32 v1, v2 offset:0x00
ds_write_b32 v1, v3 offset:0x04
ds_write_b32 v1, v4 offset:0x20
ds_write_b32 v1, v5 offset:0x24
ds_write_b32 v1, v6 offset:0x28
s_waitcnt lgkmcnt(0)
v_mov_b32_e32 v14, 0xaaaaaaaa
v_mov_b32_e32 v15, 0xbbbbbbbb
ds_read_u8 v10, v1 offset:1
ds_read_i8 v11, v1 offset:1
ds_read_u16 v12, v1 offset:2
ds_read_i16 v13, v1 offset:2
ds_read_u16_d16 v14, v1 offset:4
ds_read_u16_d16_hi v15, v1 offset:6
v_mov_b32_e32 v20, 0x00ff00f0
v_mov_b32_e32 v21, 0xab00cd05
v_mov_b32_e32 v23, 0x0f0f0f0f
v_mov_b32_e32 v24, 0x01010101
v_mov_b32_e32 v25, 0x20
s_waitcnt lgkmcnt(0)
ds_mskor_rtn_b32 v22, v1, v20, v21 offset:0x20
ds_mskor_b32 v1, v23, v24 offset:0x24
ds_add_rtn_u32 v26, v1, v25 offset:0x28
s_waitcnt lgkmcnt(0)
ds_read_b32 v30, v1 offset:0x20
ds_read_b32 v31, v1 offset:0x24
ds_read_b32 v32, v1 offset:0x28
s_waitcnt lgkmcnt(0)
v_xor_b32_e32 v22, v22, v30
v_xor_b32_e32 v22, v22, v31
v_xor_b32_e32 v26, v26, v32
v_mov_b32_e32 v40, 0
buffer_store_dword v10, v40, s[0:3], 0 offen
v_mov_b32_e32 v40, 4
buffer_store_dword v11, v40, s[0:3], 0 offen
v_mov_b32_e32 v40, 8
buffer_store_dword v12, v40, s[0:3], 0 offen
v_mov_b32_e32 v40, 12
buffer_store_dword v13, v40, s[0:3], 0 offen
v_mov_b32_e32 v40, 16
buffer_store_dword v14, v40, s[0:3], 0 offen
v_mov_b32_e32 v40, 20
buffer_store_dword v15, v40, s[0:3], 0 offen
v_mov_b32_e32 v40, 24
buffer_store_dword v22, v40, s[0:3], 0 offen
v_mov_b32_e32 v40, 28
buffer_store_dword v26, v40, s[0:3], 0 offen
s_endpgm
