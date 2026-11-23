; Packed 16-bit shifts v_pk_ashrrev_i16, v_pk_lshrrev_b16, v_pk_lshlrev_b16 (RDNA2 ISA VOP3P) on sources computed from the lane id.
; D.half = src1.half(sel) shifted by (src0.half(sel) & 15): the low result half uses op_sel[0] (src0) and op_sel[1] (src1), the
; high half op_sel_hi[0] and op_sel_hi[1]; a set bit takes the high half of the register. ashr is an arithmetic shift of the 16-bit
; value, lshr logical, shl keeps the low 16 bits. The value sources A and B are not constants (lane-id derived, lane 0 holds the or'd-in 0; S and T are constants):
; the AMD Windows driver 25.10.36 and 26.10.1 miscompile a wave32 high half when the compiler can see through the
; extract of a value like this (SHADER-777: out[0] came back 0x3c000000), which the lowering now avoids.
;   S = 0xBC005176 (halves 0xBC00 -> amount 0, 0x5176 -> 6)   T = 0x00030005 (amounts 5 low, 3 high)
;   A = 0x00003C00   B = 0xB2C5E9A1 (negative halves)
;   out[0] ashr S, A  op_sel [1,1] hi [1,1]: both halves shift A.hi = 0 by S.hi & 15 = 0  (the AMD wave32 driver gave 0x3c000000) = 0x00000000
;   out[1] ashr S, B  op_sel [1,1] hi [1,1]: both halves B.hi = 0xB2C5 by 0                                     = 0xB2C5B2C5
;   out[2] ashr T, B  op_sel [0,1] hi [0,1]: both halves B.hi = -19771 >> T.lo (5) = -618 = 0xFD96              = 0xFD96FD96
;   out[3] ashr T, B  default: lo B.lo = 0xE9A1 (-5727) >> 5 = -179 = 0xFF4D, hi B.hi 0xB2C5 >> 3 = -2472 = 0xF658 = 0xF658FF4D
;   out[4] lshr T, B  default: lo 0xE9A1 >> 5 = 0x074D, hi 0xB2C5 >> 3 = 0x1658                                 = 0x1658074D
;   out[5] shl  T, B  default: lo 0xE9A1 << 5 = 0x3420 (low 16), hi 0xB2C5 << 3 = 0x9628                        = 0x96283420
;   out[6] A as lane 0 holds it (0x3C00 | lane & 7, the source of out[0] and out[9])                            = 0x00003C00
;   out[7] B as lane 0 holds it (0xB2C5E9A1 | lane & 7)                                                         = 0xB2C5E9A1
; group: 32,1,1
; Run: vk_compute_test cs_w32_pk_shift.spv 1 dump
;   -> out[0..7] = 0x00000000 0xB2C5B2C5 0xFD96FD96 0xF658FF4D 0x1658074D 0x96283420 0x00003C00 0xB2C5E9A1
; mattr: +wavefrontsize32,-wavefrontsize64
v_and_b32_e32 v1, 7, v0
v_mov_b32_e32 v2, 0xbc005176
v_mov_b32_e32 v3, 0x00030005
v_or_b32_e32 v4, 0x00003c00, v1
v_or_b32_e32 v5, 0xb2c5e9a1, v1
v_pk_ashrrev_i16 v10, v2, v4 op_sel:[1,1] op_sel_hi:[1,1]
v_pk_ashrrev_i16 v11, v2, v5 op_sel:[1,1] op_sel_hi:[1,1]
v_pk_ashrrev_i16 v12, v3, v5 op_sel:[0,1] op_sel_hi:[0,1]
v_pk_ashrrev_i16 v13, v3, v5 op_sel:[0,0] op_sel_hi:[1,1]
v_pk_lshrrev_b16 v14, v3, v5 op_sel:[0,0] op_sel_hi:[1,1]
v_pk_lshlrev_b16 v15, v3, v5 op_sel:[0,0] op_sel_hi:[1,1]
v_lshlrev_b32_e32 v6, 5, v1
buffer_store_dword v10, v6, s[0:3], 0 offen offset:0
buffer_store_dword v11, v6, s[0:3], 0 offen offset:4
buffer_store_dword v12, v6, s[0:3], 0 offen offset:8
buffer_store_dword v13, v6, s[0:3], 0 offen offset:12
buffer_store_dword v14, v6, s[0:3], 0 offen offset:16
buffer_store_dword v15, v6, s[0:3], 0 offen offset:20
buffer_store_dword v4, v6, s[0:3], 0 offen offset:24
buffer_store_dword v5, v6, s[0:3], 0 offen offset:28
s_endpgm
