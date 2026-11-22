; GPU test for ds_write_b8, ds_write_b16 (32 lanes into shared dwords) and ds_write_b8_d16_hi, ds_write_b16_d16_hi, which
; only lds_fuzz.py checks on a GPU. RDNA2 ISA pseudocode (DS):
;   ds_write_b8          MEM[ADDR][7:0]   = DATA[7:0]       ds_write_b16          MEM[ADDR][15:0]  = DATA[15:0]
;   ds_write_b8_d16_hi   MEM[ADDR][7:0]   = DATA[23:16]     ds_write_b16_d16_hi   MEM[ADDR][15:0]  = DATA[31:16]
; Only the addressed bytes change. LDS is 4 KiB (RSRC2 0x40088). All 32 lanes fill dword L (bytes 0..0x7F) with
; 0xEEEEEEEE and s_barrier; then lane L writes byte L (data 0x12340041 + L: the upper bits must be ignored) with
; ds_write_b8, and the half-word at 0x40 + 2L (data 0xAAAAFF80 + L) with ds_write_b16. Four lanes share each dword and
; two share each half, in some order, with every byte or half written by its own lane.
;   out[0]  dword 0x00 = bytes 0x41 0x42 0x43 0x44 (lanes 0..3, little endian)             0x44434241
;   out[1]  dword 0x1C = lanes 28..31 = 0x5D 0x5E 0x5F 0x60                                0x605F5E5D
;   out[2]  dword 0x20 is past the byte writes and keeps the fill                          0xEEEEEEEE
;   out[3]  dword 0x40 = halves of lanes 0, 1 = 0xFF80, 0xFF81                             0xFF81FF80
;   out[4]  dword 0x7C = halves of lanes 30, 31 = 0xFF9E, 0xFF9F                           0xFF9FFF9E
; Then lane 0 alone (EXEC = 1) seeds the dwords at 0xC0 and 0xD0 with 0xEEEEEEEE and:
;   ds_write_b8_d16_hi data 0x11223344 at 0xC1 writes byte 0xC1 = 0x22: dword 0xC0 =        out[5] 0xEEEE22EE
;   ds_write_b16_d16_hi data 0x5566AABB at 0xC2 writes the upper half = 0x5566: dword 0xC0 = out[6] 0x556622EE
;   ds_write_b16_d16_hi data 0x9ABC1234 at 0xD0 writes the lower half = 0x9ABC: dword 0xD0 = out[7] 0xEEEE9ABC
; Run: vk_compute_test cs_w32_ds_subword_write.spv 1 dump
;   -> out[0..7] = 0x44434241 0x605F5E5D 0xEEEEEEEE 0xFF81FF80 0xFF9FFF9E 0xEEEE22EE 0x556622EE 0xEEEE9ABC
; rsrc2: 0x40088
; group: 32,1,1
; mattr: +wavefrontsize32,-wavefrontsize64
v_lshlrev_b32_e32 v1, 2, v0
v_mov_b32_e32 v2, 0xeeeeeeee
ds_write_b32 v1, v2
s_waitcnt lgkmcnt(0)
s_barrier
v_add_nc_u32_e32 v3, 0x12340041, v0
ds_write_b8 v0, v3
v_lshlrev_b32_e32 v5, 1, v0
v_add_nc_u32_e32 v6, 0xaaaaff80, v0
ds_write_b16 v5, v6 offset:0x40
s_waitcnt lgkmcnt(0)
s_barrier
s_mov_b32 exec_lo, 1
v_mov_b32_e32 v1, 0
ds_read_b32 v10, v1 offset:0x00
ds_read_b32 v11, v1 offset:0x1c
ds_read_b32 v12, v1 offset:0x20
ds_read_b32 v13, v1 offset:0x40
ds_read_b32 v14, v1 offset:0x7c
v_mov_b32_e32 v7, 0xeeeeeeee
ds_write_b32 v1, v7 offset:0xc0
ds_write_b32 v1, v7 offset:0xd0
s_waitcnt lgkmcnt(0)
v_mov_b32_e32 v8, 0x11223344
ds_write_b8_d16_hi v1, v8 offset:0xc1
s_waitcnt lgkmcnt(0)
ds_read_b32 v15, v1 offset:0xc0
s_waitcnt lgkmcnt(0)
v_mov_b32_e32 v9, 0x5566aabb
ds_write_b16_d16_hi v1, v9 offset:0xc2
s_waitcnt lgkmcnt(0)
ds_read_b32 v16, v1 offset:0xc0
v_mov_b32_e32 v18, 0x9abc1234
ds_write_b16_d16_hi v1, v18 offset:0xd0
s_waitcnt lgkmcnt(0)
ds_read_b32 v17, v1 offset:0xd0
s_waitcnt lgkmcnt(0)
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
buffer_store_dword v16, v40, s[0:3], 0 offen
v_mov_b32_e32 v40, 28
buffer_store_dword v17, v40, s[0:3], 0 offen
s_endpgm
