; The GFX10.3 null operand in the three quad-byte SAD operations, the docs/design.md row "Quad-byte SAD 64-bit
; source operands": null supplies a zero 64-bit input window in all three, and a zero 64-bit addend in the two
; packed forms. Lane 0 of a wave32 group stores eight dwords with S1 = 0x04030200 and the 64-bit window
; x = 0x0807060504030201, whose four windows 0x04030201 0x05040302 0x06050403 0x07060504 differ from S1's bytes
; by 1 5 9 13 (the masked form skips S1's zero byte and gives 0 3 6 9). A null window reads as zero, so its byte
; sum is 0 + 2 + 3 + 4 = 9, and a null addend adds nothing:
;   out[0..1] = v_qsad_pk_u16_u8(D, null, S1, s[4:5])  = 0x000D000C000B000A  (addend u16 fields 1 2 3 4, + 9 each)
;   out[2..3] = v_qsad_pk_u16_u8(D, v[2:3], S1, null)  = 0x000D000900050001  (the null addend leaves the SADs)
;   out[4..5] = v_mqsad_pk_u16_u8(D, null, S1, s[6:7]) = 0x0019001800170016  (fields 13 14 15 16, + 9 each)
;   out[6]    = v_mqsad_pk_u16_u8(D, s[8:9], S1, null) = 0x00030000          (masked SADs 0 3 6 9)
;   out[7]    = v_mqsad_u32_u8(D, null, S1, v[4:7])[3] = 8                   (0xffffffff + 9 wraps)
; group: 32,1,1
; Run: vk_compute_test cs_sad64_null.spv 1 dump -> out[0..7] = 720906 851980 327681 851977 1507350 1638424 196608 8
; mattr: +wavefrontsize32,-wavefrontsize64
s_mov_b32 s4, 0x00020001
s_mov_b32 s5, 0x00040003
s_mov_b32 s6, 0x000e000d
s_mov_b32 s7, 0x0010000f
s_mov_b32 s8, 0x04030201
s_mov_b32 s9, 0x08070605
v_mov_b32_e32 v1, 0x04030200
v_mov_b32_e32 v2, 0x04030201
v_mov_b32_e32 v3, 0x08070605
v_mov_b32_e32 v4, 0
v_mov_b32_e32 v5, 10
v_mov_b32_e32 v6, 20
v_mov_b32_e32 v7, -1
v_qsad_pk_u16_u8 v[8:9], null, v1, s[4:5]
v_qsad_pk_u16_u8 v[10:11], v[2:3], v1, null
v_mqsad_pk_u16_u8 v[12:13], null, v1, s[6:7]
v_mqsad_pk_u16_u8 v[14:15], s[8:9], v1, null
v_mqsad_u32_u8 v[16:19], null, v1, v[4:7]
v_cmpx_eq_u32_e32 0, v0
v_mov_b32_e32 v20, v8
v_mov_b32_e32 v21, 0
buffer_store_dword v20, v21, s[0:3], 0 offen
v_mov_b32_e32 v20, v9
v_mov_b32_e32 v21, 4
buffer_store_dword v20, v21, s[0:3], 0 offen
v_mov_b32_e32 v20, v10
v_mov_b32_e32 v21, 8
buffer_store_dword v20, v21, s[0:3], 0 offen
v_mov_b32_e32 v20, v11
v_mov_b32_e32 v21, 12
buffer_store_dword v20, v21, s[0:3], 0 offen
v_mov_b32_e32 v20, v12
v_mov_b32_e32 v21, 16
buffer_store_dword v20, v21, s[0:3], 0 offen
v_mov_b32_e32 v20, v13
v_mov_b32_e32 v21, 20
buffer_store_dword v20, v21, s[0:3], 0 offen
v_mov_b32_e32 v20, v14
v_mov_b32_e32 v21, 24
buffer_store_dword v20, v21, s[0:3], 0 offen
v_mov_b32_e32 v20, v19
v_mov_b32_e32 v21, 28
buffer_store_dword v20, v21, s[0:3], 0 offen
s_endpgm
