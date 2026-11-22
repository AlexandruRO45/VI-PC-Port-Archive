; GPU test for scratch_* at byte addresses that are not dword aligned, MEASURED on a Radeon RDNA2 iGPU (gfx1036, AMD driver
; 26.10.1, raw kernel through tools/hw_oracle, wave32 and wave64 alike): a dword access is the four consecutive bytes of the
; lane's private memory, across the dword boundary, not an access aligned down; a short or byte only touches its bytes.
; Lane 0 (EXEC = 1) fills dwords 0..23 of its scratch with byte(k) = 0x80 + k (dword i = 0x83828180 + 0x04040404 * i), then:
;   out[0]  scratch_load_dword at 2: bytes 0x82 0x83 0x84 0x85                                              0x85848382
;   out[1]  scratch_load_dword at 0x11: bytes 0x91 0x92 0x93 0x94                                           0x94939291
;   out[2]  scratch_load_ushort at 3: bytes 0x83 0x84, across the dword boundary                            0x00008483
;   scratch_store_dword at 0x41 of 0xAABBCCDD writes bytes 0x41..0x44 = DD CC BB AA:
;   out[3]  dword 0x40 = 0xC3C2C1C0 with bytes 1..3 replaced                                                0xBBCCDDC0
;   out[4]  dword 0x44 = 0xC7C6C5C4 with byte 0 replaced                                                    0xC7C6C5AA
;   scratch_store_dwordx2 at 0x52 of 0x11111111 0x22222222 writes bytes 0x52..0x59 = 11 11 11 11 22 22 22 22:
;   out[5]  dword 0x50 = 0xD3D2D1D0 with bytes 2, 3 replaced                                                0x1111D1D0
;   out[6]  dword 0x54 = 0xD7D6D5D4 with all four replaced                                                  0x22221111
;   out[7]  dword 0x58 = 0xDBDAD9D8 with bytes 0, 1 replaced                                                0xDBDA2222
; Run: vk_compute_test cs_w32_scratch_unaligned.spv 1 dump
;   -> out[0..7] = 0x85848382 0x94939291 0x00008483 0xBBCCDDC0 0xC7C6C5AA 0x1111D1D0 0x22221111 0xDBDA2222
; group: 32,1,1
; mattr: +wavefrontsize32,-wavefrontsize64
s_mov_b32 exec_lo, 1
v_mov_b32_e32 v1, 0x0
v_mov_b32_e32 v3, 0x83828180
scratch_store_dword v1, v3, off
v_mov_b32_e32 v1, 0x4
v_mov_b32_e32 v3, 0x87868584
scratch_store_dword v1, v3, off
v_mov_b32_e32 v1, 0x8
v_mov_b32_e32 v3, 0x8b8a8988
scratch_store_dword v1, v3, off
v_mov_b32_e32 v1, 0xc
v_mov_b32_e32 v3, 0x8f8e8d8c
scratch_store_dword v1, v3, off
v_mov_b32_e32 v1, 0x10
v_mov_b32_e32 v3, 0x93929190
scratch_store_dword v1, v3, off
v_mov_b32_e32 v1, 0x14
v_mov_b32_e32 v3, 0x97969594
scratch_store_dword v1, v3, off
v_mov_b32_e32 v1, 0x18
v_mov_b32_e32 v3, 0x9b9a9998
scratch_store_dword v1, v3, off
v_mov_b32_e32 v1, 0x1c
v_mov_b32_e32 v3, 0x9f9e9d9c
scratch_store_dword v1, v3, off
v_mov_b32_e32 v1, 0x20
v_mov_b32_e32 v3, 0xa3a2a1a0
scratch_store_dword v1, v3, off
v_mov_b32_e32 v1, 0x24
v_mov_b32_e32 v3, 0xa7a6a5a4
scratch_store_dword v1, v3, off
v_mov_b32_e32 v1, 0x28
v_mov_b32_e32 v3, 0xabaaa9a8
scratch_store_dword v1, v3, off
v_mov_b32_e32 v1, 0x2c
v_mov_b32_e32 v3, 0xafaeadac
scratch_store_dword v1, v3, off
v_mov_b32_e32 v1, 0x30
v_mov_b32_e32 v3, 0xb3b2b1b0
scratch_store_dword v1, v3, off
v_mov_b32_e32 v1, 0x34
v_mov_b32_e32 v3, 0xb7b6b5b4
scratch_store_dword v1, v3, off
v_mov_b32_e32 v1, 0x38
v_mov_b32_e32 v3, 0xbbbab9b8
scratch_store_dword v1, v3, off
v_mov_b32_e32 v1, 0x3c
v_mov_b32_e32 v3, 0xbfbebdbc
scratch_store_dword v1, v3, off
v_mov_b32_e32 v1, 0x40
v_mov_b32_e32 v3, 0xc3c2c1c0
scratch_store_dword v1, v3, off
v_mov_b32_e32 v1, 0x44
v_mov_b32_e32 v3, 0xc7c6c5c4
scratch_store_dword v1, v3, off
v_mov_b32_e32 v1, 0x48
v_mov_b32_e32 v3, 0xcbcac9c8
scratch_store_dword v1, v3, off
v_mov_b32_e32 v1, 0x4c
v_mov_b32_e32 v3, 0xcfcecdcc
scratch_store_dword v1, v3, off
v_mov_b32_e32 v1, 0x50
v_mov_b32_e32 v3, 0xd3d2d1d0
scratch_store_dword v1, v3, off
v_mov_b32_e32 v1, 0x54
v_mov_b32_e32 v3, 0xd7d6d5d4
scratch_store_dword v1, v3, off
v_mov_b32_e32 v1, 0x58
v_mov_b32_e32 v3, 0xdbdad9d8
scratch_store_dword v1, v3, off
v_mov_b32_e32 v1, 0x5c
v_mov_b32_e32 v3, 0xdfdedddc
scratch_store_dword v1, v3, off
s_waitcnt_vscnt null, 0x0
v_mov_b32_e32 v1, 2
scratch_load_dword v20, v1, off
v_mov_b32_e32 v1, 0x11
scratch_load_dword v21, v1, off
v_mov_b32_e32 v1, 3
scratch_load_ushort v22, v1, off
v_mov_b32_e32 v1, 0x41
v_mov_b32_e32 v4, 0xaabbccdd
scratch_store_dword v1, v4, off
v_mov_b32_e32 v1, 0x52
v_mov_b32_e32 v6, 0x11111111
v_mov_b32_e32 v7, 0x22222222
scratch_store_dwordx2 v1, v[6:7], off
s_waitcnt_vscnt null, 0x0
v_mov_b32_e32 v1, 0x40
scratch_load_dword v23, v1, off
v_mov_b32_e32 v1, 0x44
scratch_load_dword v24, v1, off
v_mov_b32_e32 v1, 0x50
scratch_load_dword v25, v1, off
v_mov_b32_e32 v1, 0x54
scratch_load_dword v26, v1, off
v_mov_b32_e32 v1, 0x58
scratch_load_dword v27, v1, off
s_waitcnt vmcnt(0)
v_mov_b32_e32 v40, 0
buffer_store_dword v20, v40, s[0:3], 0 offen
v_mov_b32_e32 v40, 4
buffer_store_dword v21, v40, s[0:3], 0 offen
v_mov_b32_e32 v40, 8
buffer_store_dword v22, v40, s[0:3], 0 offen
v_mov_b32_e32 v40, 12
buffer_store_dword v23, v40, s[0:3], 0 offen
v_mov_b32_e32 v40, 16
buffer_store_dword v24, v40, s[0:3], 0 offen
v_mov_b32_e32 v40, 20
buffer_store_dword v25, v40, s[0:3], 0 offen
v_mov_b32_e32 v40, 24
buffer_store_dword v26, v40, s[0:3], 0 offen
v_mov_b32_e32 v40, 28
buffer_store_dword v27, v40, s[0:3], 0 offen
s_endpgm
