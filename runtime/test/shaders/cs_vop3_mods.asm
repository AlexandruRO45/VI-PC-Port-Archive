; VOP3 output and source modifiers on the float VALU forms the game's listings use: clamp, the three omods and
; neg/abs on the sources, plus op_sel/op_sel_hi on a packed f16 add. s[0:3] is the output V#. One thread, eight
; words, one modifier class per word; v0 = 2.0, v1 = 0.75, v2 = 0.25, v4 = 0.25 | 3.0 << 16.
;   out[0] v_mul_legacy_f32 v0, v1 clamp      2.0 * 0.75 = 1.5, clamped to 1.0        0x3f800000
;   out[1] v_mul_f32 v0, v1 mul:2             1.5 * 2 = 3.0                          0x40400000
;   out[2] v_mul_f32 v0, v1 mul:4             1.5 * 4 = 6.0                          0x40c00000
;   out[3] v_mul_f32 v0, v1 div:2             1.5 / 2 = 0.75                         0x3f400000
;   out[4] v_add_f32 v1, v2 mul:2 clamp       0.75 + 0.25 = 1.0, * 2 = 2.0, clamp    0x3f800000
;                                             (clamping before the omod would give 1.0 * 2 = 2.0)
;   out[5] v_sub_f32 -v0, -|v1|               -2.0 - (-0.75) = -1.25                 0xbfa00000
;   out[6] v_fma_f32 v0, v1, -v2 mul:2        2.0 * 0.75 - 0.25 = 1.25, * 2 = 2.5     0x40200000
;   out[7] v_pk_add_f16 1.0, v4 op_sel:[1,1] op_sel_hi:[0,0]
;                                             low 0 + 3.0 = 3.0, high 1.0 + 0.25 = 1.25   0x3d004200
; Run: vk_compute_test cs_vop3_mods.spv 1 dump
; out[0..7] = 1065353216 1077936128 1086324736 1061158912 1065353216 3214934016 1075838976 1023427072 (hex 3f800000 40400000 40c00000 3f400000 3f800000 bfa00000 40200000 3d004200)
; group: 1,1,1
; rsrc2: 0x8090
v_mov_b32_e32 v0, 0x40000000
v_mov_b32_e32 v1, 0x3f400000
v_mov_b32_e32 v2, 0x3e800000
v_mov_b32_e32 v20, 0
v_mul_legacy_f32_e64 v5, v0, v1 clamp
buffer_store_dword v5, v20, s[0:3], 0 offen
v_mul_f32_e64 v6, v0, v1 mul:2
buffer_store_dword v6, v20, s[0:3], 0 offen offset:4
v_mul_f32_e64 v7, v0, v1 mul:4
buffer_store_dword v7, v20, s[0:3], 0 offen offset:8
v_mul_f32_e64 v8, v0, v1 div:2
buffer_store_dword v8, v20, s[0:3], 0 offen offset:12
v_add_f32_e64 v9, v1, v2 clamp mul:2
buffer_store_dword v9, v20, s[0:3], 0 offen offset:16
v_sub_f32_e64 v10, -v0, -|v1|
buffer_store_dword v10, v20, s[0:3], 0 offen offset:20
v_fma_f32 v11, v0, v1, -v2 mul:2
buffer_store_dword v11, v20, s[0:3], 0 offen offset:24
v_mov_b32_e32 v12, 0x42003400
v_pk_add_f16 v13, 1.0, v12 op_sel:[1,1] op_sel_hi:[0,0]
buffer_store_dword v13, v20, s[0:3], 0 offen offset:28
s_endpgm
