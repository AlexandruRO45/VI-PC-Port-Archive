; flat_store_dword, _byte, _short, _dwordx2, _dwordx3, _dwordx4, _short_d16_hi and _byte_d16_hi at the 64-bit address in a
; VGPR pair plus the 11-bit unsigned offset (RDNA2 ISA, FLAT). The region 0x80..0xbf is zero first (four dwordx4
; MUBUF stores), the flat stores run, then the dwords are read back (MUBUF). A byte or short store writes only its own
; bytes (the rest of the dword stays 0); a _d16_hi store takes bits 31:16 (short) or 23:16 (byte) of the data register.
;   flat_store_dword    v=0x11223344 -> 0x80          out[0] = 0x11223344
;   flat_store_byte     v=0xab       -> 0x84          out[1] = 0x000000ab
;   flat_store_short    v=0x5678cdef -> 0x88          out[2] = 0x0000cdef
;   flat_store_dwordx2  (1, 2)       -> 0x90, 0x94
;   flat_store_dwordx3  (0xa1, 0xa2, 0xa3) -> 0x98, 0x9c, 0xa0
;   flat_store_dwordx4  (0xb1, 0xb2, 0xb3, 0xb4) -> 0xa8, 0xac, 0xb0, 0xb4
;   flat_store_short_d16_hi v=0x5678cdef -> 0xb8       out[6] = 0x00005678
;   flat_store_byte_d16_hi  v=0x5678cdef -> 0xbc       out[7] = 0x00000078
;   out[3] = (dword 0x90 << 8) | dword 0x94                              = 0x0102
;   out[4] = (((0x98 << 8) | 0x9c) << 8) | 0xa0                          = 0x00a1a2a3
;   out[5] = (((((0xa8 << 8) | 0xac) << 8) | 0xb0) << 8) | 0xb4          = 0xb1b2b3b4
; Measured on gfx1036 (Windows) as a raw kernel, wave32 and wave64: the words above (the byte at 0xbc is bits 23:16).
; group: 32,1,1
; Run: vk_compute_test cs_w32_flat_store.spv 1 dump -> out[0..7] = 287454020 171 52719 258 10592931 2981278644 22136 120
; mattr: +wavefrontsize32,-wavefrontsize64
v_cmpx_eq_u32_e32 0, v0
v_mov_b32_e32 v1, 0
v_mov_b32_e32 v2, 0
v_mov_b32_e32 v3, 0
v_mov_b32_e32 v4, 0
v_mov_b32_e32 v5, 0x80
buffer_store_dwordx4 v[1:4], v5, s[0:3], 0 offen
v_mov_b32_e32 v5, 0x90
buffer_store_dwordx4 v[1:4], v5, s[0:3], 0 offen
v_mov_b32_e32 v5, 0xa0
buffer_store_dwordx4 v[1:4], v5, s[0:3], 0 offen
v_mov_b32_e32 v5, 0xb0
buffer_store_dwordx4 v[1:4], v5, s[0:3], 0 offen
s_waitcnt_vscnt null, 0x0
v_mov_b32_e32 v10, 0x80
v_mov_b32_e32 v11, 0
v_mov_b32_e32 v20, 0x11223344
flat_store_dword v[10:11], v20
v_mov_b32_e32 v21, 0xab
flat_store_byte v[10:11], v21 offset:4
v_mov_b32_e32 v22, 0x5678cdef
flat_store_short v[10:11], v22 offset:8
v_mov_b32_e32 v23, 1
v_mov_b32_e32 v24, 2
flat_store_dwordx2 v[10:11], v[23:24] offset:16
v_mov_b32_e32 v25, 0xa1
v_mov_b32_e32 v26, 0xa2
v_mov_b32_e32 v27, 0xa3
flat_store_dwordx3 v[10:11], v[25:27] offset:24
v_mov_b32_e32 v28, 0xb1
v_mov_b32_e32 v29, 0xb2
v_mov_b32_e32 v30, 0xb3
v_mov_b32_e32 v31, 0xb4
flat_store_dwordx4 v[10:11], v[28:31] offset:40
flat_store_short_d16_hi v[10:11], v22 offset:56
flat_store_byte_d16_hi v[10:11], v22 offset:60
s_waitcnt vmcnt(0) lgkmcnt(0)
s_waitcnt_vscnt null, 0x0
v_mov_b32_e32 v5, 0x80
buffer_load_dword v1, v5, s[0:3], 0 offen
v_mov_b32_e32 v5, 0x84
buffer_load_dword v2, v5, s[0:3], 0 offen
v_mov_b32_e32 v5, 0x88
buffer_load_dword v3, v5, s[0:3], 0 offen
v_mov_b32_e32 v5, 0x90
buffer_load_dwordx2 v[6:7], v5, s[0:3], 0 offen
v_mov_b32_e32 v5, 0x98
buffer_load_dwordx3 v[12:14], v5, s[0:3], 0 offen
v_mov_b32_e32 v5, 0xa8
buffer_load_dwordx4 v[15:18], v5, s[0:3], 0 offen
v_mov_b32_e32 v5, 0xb8
buffer_load_dword v8, v5, s[0:3], 0 offen
v_mov_b32_e32 v5, 0xbc
buffer_load_dword v9, v5, s[0:3], 0 offen
s_waitcnt vmcnt(0)
v_lshl_or_b32 v6, v6, 8, v7
v_lshl_or_b32 v12, v12, 8, v13
v_lshl_or_b32 v12, v12, 8, v14
v_lshl_or_b32 v15, v15, 8, v16
v_lshl_or_b32 v15, v15, 8, v17
v_lshl_or_b32 v15, v15, 8, v18
v_mov_b32_e32 v40, 0
buffer_store_dword v1, v40, s[0:3], 0 offen
v_mov_b32_e32 v40, 4
buffer_store_dword v2, v40, s[0:3], 0 offen
v_mov_b32_e32 v40, 8
buffer_store_dword v3, v40, s[0:3], 0 offen
v_mov_b32_e32 v40, 12
buffer_store_dword v6, v40, s[0:3], 0 offen
v_mov_b32_e32 v40, 16
buffer_store_dword v12, v40, s[0:3], 0 offen
v_mov_b32_e32 v40, 20
buffer_store_dword v15, v40, s[0:3], 0 offen
v_mov_b32_e32 v40, 24
buffer_store_dword v8, v40, s[0:3], 0 offen
v_mov_b32_e32 v40, 28
buffer_store_dword v9, v40, s[0:3], 0 offen
s_endpgm
