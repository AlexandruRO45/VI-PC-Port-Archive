; global 8/16-bit stores and d16 loads: sub-dword accesses through a global pointer (#243).
; Lane 0 of a group seeds two dwords through the output V# (0x40 = 0x11223344, 0x44 = 0x55667788), then writes
; through a global pointer at Xbox VA 0x40 and reads the results back:
;   global_store_byte       0xAB  at 0x41   -> dword 0x40 becomes 0x1122AB44
;   global_store_byte_d16_hi 0xCD at 0x42   -> 0x11CDAB44  (the high half's low byte of the data VGPR)
;   global_store_short_d16_hi 0x1234 at 0x42 -> 0x1234AB44 (the data VGPR's high half)
;   global_store_short      0xCAFE at 0x43  -> crosses the dword: byte 3 = FE, byte 4 = CA,
;                                              dword 0x40 = 0xFE34AB44, dword 0x44 = 0xBEEF77CA after the next store
;   global_store_short      0xBEEF at 0x46  -> 0xBEEF7788, then byte 0x44 = CA gives 0xBEEF77CA
; Loads (each keeps the half it does not write):
;   out[2] = global_load_ubyte_d16      at 0x41 into 0xDEAD0000 = 0xDEAD00AB  (zero-extended byte)
;   out[3] = global_load_sbyte_d16_hi   at 0x43 into 0x0000BEEF = 0xFFFEBEEF  (0xFE sign-extended to 16 bits)
;   out[4] = global_load_short_d16      at 0x42 into 0xDEAD0000 = 0xDEADFE34  (bytes 0x42/0x43)
;   out[5] = global_load_short_d16_hi   at 0x46 into 0x00007788 = 0xBEEF7788
;   out[6] = global_load_ubyte          at 0x41 = 0x000000AB
;   out[7] = global_load_sbyte          at 0x43 = 0xFFFFFFFE
; out[0..1] are the two dwords read back after all the stores. Every access is one byte at a time, so the store
; across the dword at byte 3 is exact; the byte stores merge with atomic and/or, which keeps the neighbouring bytes.
; group: 64,1,1
; Run: vk_compute_test cs_global_sub_dword.spv 1 dump -> out[0..7] = 4264864580 3203364810 3735879851 4294885103 3735944756 3203364744 171 4294967294
; mattr: +wavefrontsize32,-wavefrontsize64
v_cmpx_eq_u32_e32 0, v0
v_mov_b32 v10, 0x11223344
v_mov_b32 v11, 0x40
buffer_store_dword v10, v11, s[0:3], 0 offen
v_mov_b32 v10, 0x55667788
v_mov_b32 v11, 0x44
buffer_store_dword v10, v11, s[0:3], 0 offen
s_waitcnt vmcnt(0)
s_mov_b32 s20, 0x40
s_mov_b32 s21, 0
v_mov_b32 v2, 0
v_mov_b32 v1, 0xab
global_store_byte v1, v2, s[20:21] offset:1
v_mov_b32 v1, 0xcd0000
global_store_byte_d16_hi v1, v2, s[20:21] offset:2
v_mov_b32 v1, 0x12340000
global_store_short_d16_hi v1, v2, s[20:21] offset:2
v_mov_b32 v1, 0xcafe
global_store_short v1, v2, s[20:21] offset:3
v_mov_b32 v1, 0xbeef
global_store_short v1, v2, s[20:21] offset:6
s_waitcnt vmcnt(0)
s_waitcnt_vscnt null, 0x0
v_mov_b32 v12, 0xdead0000
global_load_ubyte_d16 v12, v2, s[20:21] offset:1
v_mov_b32 v13, 0xbeef
global_load_sbyte_d16_hi v13, v2, s[20:21] offset:3
v_mov_b32 v14, 0xdead0000
global_load_short_d16 v14, v2, s[20:21] offset:2
v_mov_b32 v15, 0x7788
global_load_short_d16_hi v15, v2, s[20:21] offset:6
global_load_ubyte v16, v2, s[20:21] offset:1
global_load_sbyte v17, v2, s[20:21] offset:3
s_waitcnt vmcnt(0)
v_mov_b32 v11, 0x40
buffer_load_dword v18, v11, s[0:3], 0 offen
v_mov_b32 v11, 0x44
buffer_load_dword v19, v11, s[0:3], 0 offen
s_waitcnt vmcnt(0)
v_mov_b32 v24, 0
buffer_store_dword v18, v24, s[0:3], 0 offen
v_mov_b32 v24, 4
buffer_store_dword v19, v24, s[0:3], 0 offen
v_mov_b32 v24, 8
buffer_store_dword v12, v24, s[0:3], 0 offen
v_mov_b32 v24, 12
buffer_store_dword v13, v24, s[0:3], 0 offen
v_mov_b32 v24, 16
buffer_store_dword v14, v24, s[0:3], 0 offen
v_mov_b32 v24, 20
buffer_store_dword v15, v24, s[0:3], 0 offen
v_mov_b32 v24, 24
buffer_store_dword v16, v24, s[0:3], 0 offen
v_mov_b32 v24, 28
buffer_store_dword v17, v24, s[0:3], 0 offen
s_endpgm
