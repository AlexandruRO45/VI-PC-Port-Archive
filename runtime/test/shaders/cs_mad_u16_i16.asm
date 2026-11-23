; Translator contract row `v_mad_u16`, `v_mad_i16` (docs/design.md): D = S0 * S1 + S2 on the low 16-bit halves
; (VOP3 only). The product and the sum are formed in 32 bits, so nothing overflows, and the low 16 bits are the
; result. op_sel bits 0..2 read the high half of a source; bit 3 writes the high half of the destination, and like
; the other opsel-capable 16-bit VOP3 ops the other half of the destination is kept. clamp saturates the full-width
; result to [0, 65535] (v_mad_u16) or [-32768, 32767] (v_mad_i16).
;
; One thread. v1 = 0x00030007 (low half 7, high half 3), v2 = 0x0005000b (11, 5), v3 = 0x0002000d (13, 2). Each
; destination is loaded with a value the instruction cannot touch, so the words below show which half it writes:
;
;   out[0]                    v_mad_u16 v4, v1, v2, v3               7 * 11 + 13 = 90
;   out[1] v5 = 0xaabbccdd    v_mad_u16 v5, v1, v2, v3               the high half of v5 is kept
;   out[2]                    v_mad_u16 v6, v1, v2, v3 op_sel:[1,0,0,0]   high half of S0: 3 * 11 + 13 = 46
;   out[3] v7 = 0xaabbccdd    v_mad_u16 v7, v1, v2, v3 op_sel:[0,0,0,1]   the low half of v7 is kept
;   out[4]                    v_mad_u16 v10, v8, v8, v9 clamp      256 * 256 + 5 = 65541 > 65535, so clamp -> 65535
;   out[5]                    v_mad_i16 v14, v11, v12, v13          -2 * 3 + 9 = 3; the i16 operands sign-extend
;   out[6]                    v_mad_i16 v16, v15, v11, v19 clamp   32767 * -2 = -65534 < -32768, so clamp -> -32768
;   out[7]                    v_mad_i16 v18, v15, v17, v19 clamp   32767 * 2 = 65534 > 32767, so clamp -> 32767
; Run: vk_compute_test cs_mad_u16_i16.spv 1 dump
; out[0..7] = 0000005a aabb005a 0000002e 005accdd 0000ffff 00000003 00008000 00007fff (hex)
; group: 1,1,1
v_mov_b32_e32 v0, 0
v_mov_b32_e32 v1, 0x00030007
v_mov_b32_e32 v2, 0x0005000b
v_mov_b32_e32 v3, 0x0002000d
v_mov_b32_e32 v4, 0
v_mad_u16 v4, v1, v2, v3
buffer_store_dword v4, v0, s[0:3], 0 offen
v_mov_b32_e32 v5, 0xaabbccdd
v_mad_u16 v5, v1, v2, v3
buffer_store_dword v5, v0, s[0:3], 0 offen offset:4
v_mov_b32_e32 v6, 0
v_mad_u16 v6, v1, v2, v3 op_sel:[1,0,0,0]
buffer_store_dword v6, v0, s[0:3], 0 offen offset:8
v_mov_b32_e32 v7, 0xaabbccdd
v_mad_u16 v7, v1, v2, v3 op_sel:[0,0,0,1]
buffer_store_dword v7, v0, s[0:3], 0 offen offset:12
v_mov_b32_e32 v8, 0x100
v_mov_b32_e32 v9, 5
v_mov_b32_e32 v10, 0
v_mad_u16 v10, v8, v8, v9 clamp
buffer_store_dword v10, v0, s[0:3], 0 offen offset:16
v_mov_b32_e32 v11, 0xfffe
v_mov_b32_e32 v12, 3
v_mov_b32_e32 v13, 9
v_mov_b32_e32 v14, 0
v_mad_i16 v14, v11, v12, v13
buffer_store_dword v14, v0, s[0:3], 0 offen offset:20
v_mov_b32_e32 v15, 0x7fff
v_mov_b32_e32 v16, 0
v_mov_b32_e32 v19, 0
v_mad_i16 v16, v15, v11, v19 clamp
buffer_store_dword v16, v0, s[0:3], 0 offen offset:24
v_mov_b32_e32 v17, 2
v_mov_b32_e32 v18, 0
v_mad_i16 v18, v15, v17, v19 clamp
buffer_store_dword v18, v0, s[0:3], 0 offen offset:28
s_endpgm
