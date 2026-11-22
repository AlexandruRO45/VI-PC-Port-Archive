; GPU test for scratch_load_* / scratch_store_*, which the translator refused before. RDNA2 ISA (FLAT, scratch segment):
; each lane has its OWN private memory; the byte address of an access is vaddr (unless `off`) + saddr (unless `off`) + the
; signed 12-bit instruction offset, and the lanes do not see each other's data. The translator gives every invocation a
; Private array of SCRATCH_DWORDS (256) dwords. Stores and loads of a byte or short touch only those bytes.
;   out[0]  all 32 lanes store 0x1000 + lane at the SAME scratch address 0x40 and load it back: the number of lanes that
;           read their own value. Private memory gives 32; a shared word would leave one lane right.              32
; Then lane 0 alone (EXEC = 1):
;   scratch_store_dword v(0x24), 0x11111111 offset:-4 writes dword 0x20; scratch_store_byte v(0x21) of 0xFFFFFFAB writes
;   only the byte 0xAB (junk upper bits ignored): dword 0x20 = 0x1111AB11                                         out[3]
;   scratch_store_dwordx4 v(0x10), v[4:7] = 0x01020304 0x05060708 0x090A0B0C 0x0D0E0F10; then
;   scratch_load_dwordx2 v[8:9], off, s5(0x10) offset:4 reads dwords 0x14, 0x18: 0x05060708 0x090A0B0C         out[1] out[2]
;   dword 0x30 = 0, scratch_store_short v(0x32), 0xFFFFBEEF writes the high half: dword 0x30 = 0xBEEF0000          out[4]
;   scratch_load_sbyte at 0x21 (0xAB) sign extends:                                                              out[5] 0xFFFFFFAB
;   scratch_load_ushort at 0x32 (0xBEEF):                                                                         out[6] 0x0000BEEF
;   scratch_load_sshort at 0x32:                                                                                  out[7] 0xFFFFBEEF
; Run: vk_compute_test cs_w32_scratch.spv 1 dump
;   -> out[0..7] = 0x00000020 0x05060708 0x090A0B0C 0x1111AB11 0xBEEF0000 0xFFFFFFAB 0x0000BEEF 0xFFFFBEEF
; group: 32,1,1
; mattr: +wavefrontsize32,-wavefrontsize64
v_mov_b32_e32 v1, 0x40
v_add_nc_u32_e32 v2, 0x1000, v0
scratch_store_dword v1, v2, off
s_waitcnt_vscnt null, 0x0
scratch_load_dword v3, v1, off
s_waitcnt vmcnt(0)
v_cmp_eq_u32_e32 vcc_lo, v3, v2
s_bcnt1_i32_b32 s6, vcc_lo
v_mov_b32_e32 v20, s6
s_mov_b32 exec_lo, 1
v_mov_b32_e32 v1, 0x24
v_mov_b32_e32 v11, 0x11111111
scratch_store_dword v1, v11, off offset:-4
v_mov_b32_e32 v1, 0x21
v_mov_b32_e32 v12, 0xffffffab
scratch_store_byte v1, v12, off
v_mov_b32_e32 v1, 0x10
v_mov_b32_e32 v4, 0x01020304
v_mov_b32_e32 v5, 0x05060708
v_mov_b32_e32 v6, 0x090a0b0c
v_mov_b32_e32 v7, 0x0d0e0f10
scratch_store_dwordx4 v1, v[4:7], off
v_mov_b32_e32 v1, 0x30
v_mov_b32_e32 v13, 0
scratch_store_dword v1, v13, off
v_mov_b32_e32 v1, 0x32
v_mov_b32_e32 v14, 0xffffbeef
scratch_store_short v1, v14, off
s_waitcnt_vscnt null, 0x0
s_mov_b32 s5, 0x10
scratch_load_dwordx2 v[8:9], off, s5 offset:4
v_mov_b32_e32 v1, 0x20
scratch_load_dword v21, v1, off
v_mov_b32_e32 v1, 0x30
scratch_load_dword v22, v1, off
v_mov_b32_e32 v1, 0x21
scratch_load_sbyte v23, v1, off
v_mov_b32_e32 v1, 0x32
scratch_load_ushort v24, v1, off
scratch_load_sshort v25, v1, off
s_waitcnt vmcnt(0)
v_mov_b32_e32 v40, 0
buffer_store_dword v20, v40, s[0:3], 0 offen
v_mov_b32_e32 v40, 4
buffer_store_dword v8, v40, s[0:3], 0 offen
v_mov_b32_e32 v40, 8
buffer_store_dword v9, v40, s[0:3], 0 offen
v_mov_b32_e32 v40, 12
buffer_store_dword v21, v40, s[0:3], 0 offen
v_mov_b32_e32 v40, 16
buffer_store_dword v22, v40, s[0:3], 0 offen
v_mov_b32_e32 v40, 20
buffer_store_dword v23, v40, s[0:3], 0 offen
v_mov_b32_e32 v40, 24
buffer_store_dword v24, v40, s[0:3], 0 offen
v_mov_b32_e32 v40, 28
buffer_store_dword v25, v40, s[0:3], 0 offen
s_endpgm
