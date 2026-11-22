; v_sat_pk_u8_i16: each signed 16-bit half of S0 is clamped to [0, 255]; the high half of S0 becomes the high byte of the
; low 16 bits of D, and bits 31:16 of D are zero (gfx1036 measurement: a destination preloaded with 0xdeadbeef came back
; 0x0000hhll, every lane). RDNA2 ISA pseudocode:
;   D.u16[7:0]  = clamp(S0.i16[15:0], 0, 255)      D.u16[15:8] = clamp(S0.i16[31:16], 0, 255)      D[31:16] = 0
; One wave32 group, every lane runs the same code and stores eight dwords. The destinations start as 0xdeadbeef, so a
; result that kept the old high half, or a half read the wrong way round, shows. The VOP3 forms (_e64) take no clamp or
; op_sel (the assembler refuses them), so only plain operands are run: a VGPR, a literal and an SGPR source.
;   out[0] src 0x1234007f (VGPR, e32): high 0x1234 = 4660 -> 0xff, low 0x7f stays      -> 0x0000ff7f
;   out[1] src 0xffff0001 (literal):   high 0xffff = -1 -> 0, low 1                     -> 0x00000001
;   out[2] src 0x80008000 (VGPR, e32): high -32768 -> 0, low -32768 -> 0                -> 0x00000000
;   out[3] src 0x7fff8001 (VGPR, e32): high 32767 -> 0xff, low -32767 -> 0              -> 0x0000ff00
;   out[4] src 0x00ff0100 (VGPR, e64): high 255 stays, low 256 -> 0xff                  -> 0x0000ffff
;   out[5] src 0x010000fe (VGPR, e64): high 256 -> 0xff, low 0xfe stays                 -> 0x0000fffe
;   out[6] src 0xfffe0100 (SGPR):      high -2 -> 0, low 256 -> 0xff                    -> 0x000000ff
;   out[7] src 0x00800080 (VGPR, e32): both 128 stay                                    -> 0x00008080
; group: 32,1,1
; Run: vk_compute_test cs_w32_sat_pk.spv 1 dump -> out[0..7] = 65407 1 0 65280 65535 65534 255 32896
; mattr: +wavefrontsize32,-wavefrontsize64
v_mov_b32_e32 v10, 0x1234007f
v_mov_b32_e32 v11, 0x80008000
v_mov_b32_e32 v12, 0x7fff8001
v_mov_b32_e32 v13, 0x00ff0100
v_mov_b32_e32 v14, 0x010000fe
v_mov_b32_e32 v15, 0x00800080
s_mov_b32 s10, 0xfffe0100
v_mov_b32_e32 v20, 0xdeadbeef
v_mov_b32_e32 v21, 0xdeadbeef
v_mov_b32_e32 v22, 0xdeadbeef
v_mov_b32_e32 v23, 0xdeadbeef
v_mov_b32_e32 v24, 0xdeadbeef
v_mov_b32_e32 v25, 0xdeadbeef
v_mov_b32_e32 v26, 0xdeadbeef
v_mov_b32_e32 v27, 0xdeadbeef
v_sat_pk_u8_i16_e32 v20, v10
v_sat_pk_u8_i16_e32 v21, 0xffff0001
v_sat_pk_u8_i16_e32 v22, v11
v_sat_pk_u8_i16_e32 v23, v12
v_sat_pk_u8_i16_e64 v24, v13
v_sat_pk_u8_i16_e64 v25, v14
v_sat_pk_u8_i16_e32 v26, s10
v_sat_pk_u8_i16_e32 v27, v15
v_mov_b32_e32 v30, 0
buffer_store_dword v20, v30, s[0:3], 0 offen offset:0
buffer_store_dword v21, v30, s[0:3], 0 offen offset:4
buffer_store_dword v22, v30, s[0:3], 0 offen offset:8
buffer_store_dword v23, v30, s[0:3], 0 offen offset:12
buffer_store_dword v24, v30, s[0:3], 0 offen offset:16
buffer_store_dword v25, v30, s[0:3], 0 offen offset:20
buffer_store_dword v26, v30, s[0:3], 0 offen offset:24
buffer_store_dword v27, v30, s[0:3], 0 offen offset:28
s_endpgm
