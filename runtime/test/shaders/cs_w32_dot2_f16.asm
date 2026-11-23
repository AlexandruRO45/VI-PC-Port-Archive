; mcpu: gfx1030
; v_dot2_f32_f16 (VOP3P) and v_dot2c_f32_f16 (VOP2), run on a GPU; until now CPU-checked only
; (tests/semantic/test_alu_dot_complete.py). D = a0*b0 + a1*b1 + acc in f32, each half widened from f16; acc is S2 for
; v_dot2_f32_f16 and the old D for v_dot2c_f32_f16 (docs/design.md, "Packed integer dot products ... v_dot2_f32_f16").
; gfx1013 has neither instruction, so this file is assembled for gfx1030 (the "; mcpu:" line above).
; Every product and sum below is exact in f32, so the words do not depend on how a GPU rounds or fuses; what they
; check is which half each operand reads (op_sel / op_sel_hi), the negations, the accumulator and clamp.
; Halves: 2.0 = 0x4000, 3.0 = 0x4200, 4.0 = 0x4400, 0.5 = 0x3800, 1.5 = 0x3e00, 0.25 = 0x3400, 2^-24 = 0x0001.
;   A = 0x42004000 (lo 2.0, hi 3.0)   B = 0x38004400 (lo 4.0, hi 0.5)   C = 1.0f
;   out[0] v_dot2_f32_f16 A, B, C                                       2*4 + 3*0.5 + 1     = 10.5  0x41280000
;   out[1] op_sel:[1,0,0] op_sel_hi:[0,1,1] (A's halves swapped)        3*4 + 2*0.5 + 1     = 14.0  0x41600000
;   out[2] neg_lo:[1,0,0] neg_hi:[0,1,0] (-a0, -b1)                     -2*4 + 3*-0.5 + 1   = -8.5  0xc1080000
;   out[3] op_sel:[1,0,0] op_sel_hi:[1,1,1] neg_lo:[0,0,1] (both from A.hi, -acc)  3*4 + 3*0.5 - 1 = 12.5  0x41480000
;   out[4] clamp                                                        10.5 -> 1.0               0x3f800000
;   out[5] neg_lo:[1,0,0] neg_hi:[0,1,0] clamp                          -8.5 -> 0                 0x00000000
;   out[6] v_dot2c_f32_f16 A, B with D = 0.25                           8 + 1.5 + 0.25      = 9.75  0x411c0000
;   out[7] v_dot2c_f32_f16 0x3e000001, 0x34004000 with D = 0 (a denormal half: 2^-24 * 2 + 1.5 * 0.25)
;                                                                       2^-23 + 0.375             0x3ec00004
; group: 32,1,1
; Run: vk_compute_test cs_w32_dot2_f16.spv 1 dump -> out[0..7] = 1093140480 1096810496 3238526976 1095237632
; 1065353216 0 1092354048 1052770308
; mattr: +wavefrontsize32,-wavefrontsize64
v_mov_b32_e32 v1, 0x42004000
v_mov_b32_e32 v2, 0x38004400
v_mov_b32_e32 v3, 1.0
v_dot2_f32_f16 v10, v1, v2, v3
v_dot2_f32_f16 v11, v1, v2, v3 op_sel:[1,0,0] op_sel_hi:[0,1,1]
v_dot2_f32_f16 v12, v1, v2, v3 neg_lo:[1,0,0] neg_hi:[0,1,0]
v_dot2_f32_f16 v13, v1, v2, v3 op_sel:[1,0,0] op_sel_hi:[1,1,1] neg_lo:[0,0,1]
v_dot2_f32_f16 v14, v1, v2, v3 clamp
v_dot2_f32_f16 v15, v1, v2, v3 neg_lo:[1,0,0] neg_hi:[0,1,0] clamp
v_mov_b32_e32 v16, 0.25
v_dot2c_f32_f16_e32 v16, v1, v2
v_mov_b32_e32 v4, 0x3e000001
v_mov_b32_e32 v5, 0x34004000
v_mov_b32_e32 v17, 0
v_dot2c_f32_f16_e32 v17, v4, v5
v_cmpx_eq_u32_e32 0, v0
v_mov_b32_e32 v25, 0
buffer_store_dword v10, v25, s[0:3], 0 offen
v_mov_b32_e32 v25, 4
buffer_store_dword v11, v25, s[0:3], 0 offen
v_mov_b32_e32 v25, 8
buffer_store_dword v12, v25, s[0:3], 0 offen
v_mov_b32_e32 v25, 12
buffer_store_dword v13, v25, s[0:3], 0 offen
v_mov_b32_e32 v25, 16
buffer_store_dword v14, v25, s[0:3], 0 offen
v_mov_b32_e32 v25, 20
buffer_store_dword v15, v25, s[0:3], 0 offen
v_mov_b32_e32 v25, 24
buffer_store_dword v16, v25, s[0:3], 0 offen
v_mov_b32_e32 v25, 28
buffer_store_dword v17, v25, s[0:3], 0 offen
s_endpgm
