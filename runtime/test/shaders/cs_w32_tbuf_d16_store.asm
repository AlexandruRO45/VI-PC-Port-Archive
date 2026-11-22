; tbuffer_store_format_d16_x and tbuffer_store_format_d16_xyz (the two d16 tbuffer store forms no other test runs).
; One thread. The instruction's format replaces the V#'s: the V# below says 32_FLOAT (a 4-byte element), the stores
; use their own format. d16 data is f16 values, two per VGPR, low half first; d16_x reads only the low half of its
; VGPR, d16_xyz reads v0 (x, y) and the low half of v1 (z). The high half of v1 (0x7777) and of the d16_x
; register (0x4000) is junk the store must not read. A store writes its format's whole element: components the format has but the
; instruction does not give are 0, components the format lacks are dropped (the same rules as cs_tbuffer_store).
; The output buffer is filled with 0xee bytes; each case stores into its own region at 0x40.., then the dwords are read
; back into out[0..7]. f16 bits: 1.0 3c00, 2.0 4000, 3.0 4200, 0.0 0000. UNORM conversion of an f16 v: 1.0 -> all ones, 0.0 -> 0.
;   0x40  d16_x    16_FLOAT, x = 1.0 (lo 3c00, hi junk 4000): the f16 as it is, two bytes     out[0] = 0xeeee3c00
;   0x44  d16_x    8_UNORM,  x = 1.0: 255, one byte                                          out[1] = 0xeeeeeeff
;   0x48  d16_xyz  16_16_16_16_FLOAT, (1.0, 2.0, 3.0), w not given = 0: bytes 00 3c 00 40 00 42 00 00
;                                                                                          out[2] = 0x40003c00 (x, y)
;                                                                                          out[3] = 0x00004200 (z, w = 0)
;   0x50  d16_xyz  8_8_8_8_UNORM, (1.0, 0.0, 1.0), w = 0: bytes ff 00 ff 00               out[4] = 0x00ff00ff
;   0x58  d16_xyz  16_16_FLOAT, (1.0, 2.0, 3.0): the format has x, y only, z is dropped    out[5] = 0x40003c00
;                                               (the next dword, 0x5c, is not written)    out[6] = 0xeeeeeeee
;   0x60  d16_x    16_16_FLOAT, x = 1.0: y is in the format and not given = 0              out[7] = 0x00003c00
; group: 32,1,1
; Run: vk_compute_test cs_w32_tbuf_d16_store.spv 1 dump -> out[0..7] = 4008590336 4008636159 1073757184 16896 16711935 1073757184 4008636142 15360
; mattr: +wavefrontsize32,-wavefrontsize64
v_cmpx_eq_u32_e32 0, v0
s_mov_b32 s4, s0
s_mov_b32 s5, s1
s_mov_b32 s6, 0xffffffff
s_mov_b32 s7, 0x31016fac
v_mov_b32_e32 v20, 0x40003c00
v_mov_b32_e32 v5, 0x40
tbuffer_store_format_d16_x v20, v5, s[4:7], 0 format:[BUF_FMT_16_FLOAT] offen
v_mov_b32_e32 v5, 0x44
tbuffer_store_format_d16_x v20, v5, s[4:7], 0 format:[BUF_FMT_8_UNORM] offen
v_mov_b32_e32 v20, 0x40003c00
v_mov_b32_e32 v21, 0x77774200
v_mov_b32_e32 v5, 0x48
tbuffer_store_format_d16_xyz v[20:21], v5, s[4:7], 0 format:[BUF_FMT_16_16_16_16_FLOAT] offen
v_mov_b32_e32 v20, 0x3c00
v_mov_b32_e32 v21, 0x77773c00
v_mov_b32_e32 v5, 0x50
tbuffer_store_format_d16_xyz v[20:21], v5, s[4:7], 0 format:[BUF_FMT_8_8_8_8_UNORM] offen
v_mov_b32_e32 v20, 0x40003c00
v_mov_b32_e32 v21, 0x77774200
v_mov_b32_e32 v5, 0x58
tbuffer_store_format_d16_xyz v[20:21], v5, s[4:7], 0 format:[BUF_FMT_16_16_FLOAT] offen
v_mov_b32_e32 v20, 0x40003c00
v_mov_b32_e32 v5, 0x60
tbuffer_store_format_d16_x v20, v5, s[4:7], 0 format:[BUF_FMT_16_16_FLOAT] offen
s_waitcnt_vscnt null, 0x0
v_mov_b32_e32 v5, 0x40
buffer_load_dword v10, v5, s[0:3], 0 offen
v_mov_b32_e32 v5, 0x44
buffer_load_dword v11, v5, s[0:3], 0 offen
v_mov_b32_e32 v5, 0x48
buffer_load_dword v12, v5, s[0:3], 0 offen
v_mov_b32_e32 v5, 0x4c
buffer_load_dword v13, v5, s[0:3], 0 offen
v_mov_b32_e32 v5, 0x50
buffer_load_dword v14, v5, s[0:3], 0 offen
v_mov_b32_e32 v5, 0x58
buffer_load_dword v15, v5, s[0:3], 0 offen
v_mov_b32_e32 v5, 0x5c
buffer_load_dword v16, v5, s[0:3], 0 offen
v_mov_b32_e32 v5, 0x60
buffer_load_dword v17, v5, s[0:3], 0 offen
s_waitcnt vmcnt(0)
v_mov_b32_e32 v5, 0
buffer_store_dword v10, v5, s[0:3], 0 offen
v_mov_b32_e32 v5, 4
buffer_store_dword v11, v5, s[0:3], 0 offen
v_mov_b32_e32 v5, 8
buffer_store_dword v12, v5, s[0:3], 0 offen
v_mov_b32_e32 v5, 12
buffer_store_dword v13, v5, s[0:3], 0 offen
v_mov_b32_e32 v5, 16
buffer_store_dword v14, v5, s[0:3], 0 offen
v_mov_b32_e32 v5, 20
buffer_store_dword v15, v5, s[0:3], 0 offen
v_mov_b32_e32 v5, 24
buffer_store_dword v16, v5, s[0:3], 0 offen
v_mov_b32_e32 v5, 28
buffer_store_dword v17, v5, s[0:3], 0 offen
s_endpgm