; flat_load_dword, _ubyte, _sbyte, _ushort, _sshort, _dwordx2, _dwordx3, _dwordx4 at the 64-bit address in a VGPR pair
; plus the 11-bit unsigned instruction offset (RDNA2 ISA, FLAT). The address names the same memory as the global_*
; forms (a global VA). Words, from the memory the seeds below leave (a byte or short load reads the low bytes of its
; dword; the s forms sign-extend from bit 7 / bit 15; a multi-dword load fills v[n], v[n+1], ... from ascending dwords):
;   out[0] flat_load_dword  [0x80]              = 0x11223344
;   out[1] flat_load_ubyte  [0x80+4] = 0xab     = 0x000000ab
;   out[2] flat_load_sbyte  [0x80+4]            = 0xffffffab   (bit 7 set)
;   out[3] flat_load_ushort [0x80+8] = 0xcdef   = 0x0000cdef
;   out[4] flat_load_sshort [0x80+8]            = 0xffffcdef   (bit 15 set)
;   out[5] flat_load_dwordx2 [0x80+16] = (1, 2),            packed (a << 8) | b           = 0x0102
;   out[6] flat_load_dwordx3 [0x80+24] = (0xa1, 0xa2, 0xa3), packed (((a << 8) | b) << 8) | c = 0x00a1a2a3
;   out[7] flat_load_dwordx4 [0x80+40] = (0xb1, 0xb2, 0xb3, 0xb4), packed the same way     = 0xb1b2b3b4
; Measured on an AMD Radeon iGPU (gfx1036, Windows) as a raw kernel in wave32 and wave64 (tools/hw_oracle): every one of
; these loads returned the words above (the hardware gives the global_* forms' words).
; group: 32,1,1
; Run: vk_compute_test cs_w32_flat_load.spv 1 dump -> out[0..7] = 287454020 171 4294967211 52719 4294954479 258 10592931 2981278644
; mattr: +wavefrontsize32,-wavefrontsize64; Seeds (MUBUF stores through the V# in s[0:3]; the heap is the global address space, so byte n is address n):
;   0x80 0x11223344, 0x84 0xab, 0x88 0xcdef, 0x90 1, 0x94 2, 0x98 0xa1, 0x9c 0xa2, 0xa0 0xa3, 0xa8..0xb4 0xb1..0xb4.
; v[10:11] = the 64-bit address 0x80. Lane 0 only (EXEC cut), one dword per out word.
v_cmpx_eq_u32_e32 0, v0
v_mov_b32_e32 v1, 0x11223344
v_mov_b32_e32 v2, 0x80
buffer_store_dword v1, v2, s[0:3], 0 offen
v_mov_b32_e32 v1, 0xab
v_mov_b32_e32 v2, 0x84
buffer_store_dword v1, v2, s[0:3], 0 offen
v_mov_b32_e32 v1, 0xcdef
v_mov_b32_e32 v2, 0x88
buffer_store_dword v1, v2, s[0:3], 0 offen
v_mov_b32_e32 v1, 1
v_mov_b32_e32 v2, 0x90
buffer_store_dword v1, v2, s[0:3], 0 offen
v_mov_b32_e32 v1, 2
v_mov_b32_e32 v2, 0x94
buffer_store_dword v1, v2, s[0:3], 0 offen
v_mov_b32_e32 v1, 0xa1
v_mov_b32_e32 v2, 0x98
buffer_store_dword v1, v2, s[0:3], 0 offen
v_mov_b32_e32 v1, 0xa2
v_mov_b32_e32 v2, 0x9c
buffer_store_dword v1, v2, s[0:3], 0 offen
v_mov_b32_e32 v1, 0xa3
v_mov_b32_e32 v2, 0xa0
buffer_store_dword v1, v2, s[0:3], 0 offen
v_mov_b32_e32 v1, 0xb1
v_mov_b32_e32 v2, 0xa8
buffer_store_dword v1, v2, s[0:3], 0 offen
v_mov_b32_e32 v1, 0xb2
v_mov_b32_e32 v2, 0xac
buffer_store_dword v1, v2, s[0:3], 0 offen
v_mov_b32_e32 v1, 0xb3
v_mov_b32_e32 v2, 0xb0
buffer_store_dword v1, v2, s[0:3], 0 offen
v_mov_b32_e32 v1, 0xb4
v_mov_b32_e32 v2, 0xb4
buffer_store_dword v1, v2, s[0:3], 0 offen
s_waitcnt_vscnt null, 0x0
v_mov_b32_e32 v10, 0x80
v_mov_b32_e32 v11, 0
flat_load_dword v1, v[10:11]
flat_load_ubyte v2, v[10:11] offset:4
flat_load_sbyte v3, v[10:11] offset:4
flat_load_ushort v4, v[10:11] offset:8
flat_load_sshort v5, v[10:11] offset:8
flat_load_dwordx2 v[6:7], v[10:11] offset:16
flat_load_dwordx3 v[12:14], v[10:11] offset:24
flat_load_dwordx4 v[15:18], v[10:11] offset:40
s_waitcnt vmcnt(0) lgkmcnt(0)
v_lshl_or_b32 v6, v6, 8, v7
v_lshl_or_b32 v12, v12, 8, v13
v_lshl_or_b32 v12, v12, 8, v14
v_lshl_or_b32 v15, v15, 8, v16
v_lshl_or_b32 v15, v15, 8, v17
v_lshl_or_b32 v15, v15, 8, v18
v_mov_b32_e32 v25, 0
buffer_store_dword v1, v25, s[0:3], 0 offen
v_mov_b32_e32 v25, 4
buffer_store_dword v2, v25, s[0:3], 0 offen
v_mov_b32_e32 v25, 8
buffer_store_dword v3, v25, s[0:3], 0 offen
v_mov_b32_e32 v25, 12
buffer_store_dword v4, v25, s[0:3], 0 offen
v_mov_b32_e32 v25, 16
buffer_store_dword v5, v25, s[0:3], 0 offen
v_mov_b32_e32 v25, 20
buffer_store_dword v6, v25, s[0:3], 0 offen
v_mov_b32_e32 v25, 24
buffer_store_dword v12, v25, s[0:3], 0 offen
v_mov_b32_e32 v25, 28
buffer_store_dword v15, v25, s[0:3], 0 offen
s_endpgm
