; VOP3P neg_lo / neg_hi on v_dot4_i32_i8, GPU-checked against the gfx1036 hardware rule in
; docs/design.md ("Signed-byte dot products"): on S0 and S1, neg_lo flips bit 15 and neg_hi flips bit 31 of the dword;
; on the addend S2, neg_hi clears bit 31 and neg_lo then flips it. The rule itself was measured on an RDNA2 iGPU and
; is pinned by tests/semantic/test_dot4_i8_neg.py; this fixture is what checks that the translation reproduces it on
; a real GPU, not only on the CPU.
;
; One thread. S0 = 0x04030201 (bytes 1, 2, 3, 4), S1 = 0xF8F90605 (5, 6, -7, -8), addend 100, so the plain dot is
; 5 + 12 - 21 - 32 = -36 and out = 64.
;   out[0] plain                                  100 - 36              = 0x00000040
;   out[1] S1 neg_lo (bit 15): second byte 6 -> -122, dot -292        -> 0xFFFFFF40
;   out[2] S1 neg_hi (bit 31): fourth byte -8 -> 120, dot 476         -> 0x00000240
;   out[3] S1 neg_lo and neg_hi: 5 - 244 - 21 + 480 = 220             -> 0x00000140
;   out[4] S0 neg_lo: second byte 2 -> -126, dot -804                 -> 0xFFFFFD40
;   out[5] S0 neg_hi: fourth byte 4 -> -124, and S1's fourth byte is -8, so that product turns positive:
;          dot 5 + 12 - 21 + 992 = 988                                  -> 0x00000440
;   out[6] clamp, S0 = S1 = 0x7F7F7F7F (4 * 127 * 127 = 64516) with the addend 0x7FFFFFFF: the sum overflows and
;          saturates to the largest signed 32-bit value                    -> 0x7FFFFFFF
;   out[7] addend neg_lo (bit 31 of S2): 0x80000064 - 36              -> 0x80000040
; Run: vk_compute_test cs_dot4_neg.spv 1 dump
; out[0..7] = 00000040 ffffff40 00000240 00000140 fffffd40 00000440 7fffffff 80000040 (hex)
; mcpu: gfx1030
; group: 1,1,1
; rsrc2: 0x8090
v_mov_b32_e32 v20, 0
v_mov_b32_e32 v1, 0x04030201
v_mov_b32_e32 v2, 0xf8f90605
v_mov_b32_e32 v3, 0x64

v_dot4_i32_i8 v4, v1, v2, v3
buffer_store_dword v4, v20, s[0:3], 0 offen

v_dot4_i32_i8 v5, v1, v2, v3 neg_lo:[0,1,0]
buffer_store_dword v5, v20, s[0:3], 0 offen offset:4

v_dot4_i32_i8 v6, v1, v2, v3 neg_hi:[0,1,0]
buffer_store_dword v6, v20, s[0:3], 0 offen offset:8

v_dot4_i32_i8 v7, v1, v2, v3 neg_lo:[0,1,0] neg_hi:[0,1,0]
buffer_store_dword v7, v20, s[0:3], 0 offen offset:12

v_dot4_i32_i8 v8, v1, v2, v3 neg_lo:[1,0,0]
buffer_store_dword v8, v20, s[0:3], 0 offen offset:16

v_dot4_i32_i8 v9, v1, v2, v3 neg_hi:[1,0,0]
buffer_store_dword v9, v20, s[0:3], 0 offen offset:20

v_mov_b32_e32 v10, 0x7f7f7f7f
v_mov_b32_e32 v11, 0x7fffffff
v_dot4_i32_i8 v12, v10, v10, v11 clamp
buffer_store_dword v12, v20, s[0:3], 0 offen offset:24

v_dot4_i32_i8 v13, v1, v2, v3 neg_lo:[0,0,1]
buffer_store_dword v13, v20, s[0:3], 0 offen offset:28
s_endpgm
