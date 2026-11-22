; group: 32,1,1
; DPP with neg/abs on src0 (EGEN-DPP-SRCMOD). The DPP fetch moves the raw register; the op applies the input
; modifiers afterwards (RDNA2 ISA: the DPP fields pick src0's lane, neg/abs are part of the VOP's input; checked on a
; gfx1036 with a raw kernel). So an f16 op negates bit 15 of the fetched word, and a bound_ctrl:1 zero takes the sign
; (-0.0). Until this test the translator applied neg/abs as f32 modifiers before the shuffle: an f16 op saw bit 31
; flipped and its low half unchanged, and the zero fill dropped the sign.
; v0 = lane. v1 = 0x12343c00 (lo 1.0h), v2 = 0x56784000 (lo 2.0h), v3 = 1.0f, v4 = -0.0f in every lane.
; Lanes 0 and 1 store v[10:13] to out[0..3] and out[4..7].
;   v10 = 0xaaaa0000, v_add_f16_dpp -v1, v2 quad_perm:[1,0,3,2]: -1.0h + 2.0h = 1.0h; the f16 op keeps
;         the high half -> 0xaaaa3c00 (lanes 0, 1)                            [before: 1.0h + 2.0h = 0xaaaa4200]
;   v11 = 0xbbbb0000, v_mul_f16_dpp -|v1|, v2 row_shr:1 bound_ctrl:1: lane 0 has no lane below it in its row, so src0
;         is 0 and -|0| = -0.0h; -0.0h * 2.0h = -0.0h -> 0xbbbb8000. Lane 1: -1.0h * 2.0h = -2.0h -> 0xbbbbc000
;                                                                              [before: 0xbbbb0000, 0xbbbb4000]
;   v12 = v_add_f32_dpp -v3, v4 row_shr:1 bound_ctrl:1: lane 0: -0.0 + -0.0 = -0.0 -> 0x80000000;
;         lane 1: -1.0 + -0.0 = -1.0 -> 0xbf800000                            [before: lane 0 +0.0 + -0.0 = 0]
;   v13 = v_cvt_f32_f16_dpp -v1 quad_perm:[0,0,0,0]: -1.0h -> -1.0f = 0xbf800000 (lanes 0, 1) [before: 0x3f800000]
; Run: vk_compute_test cs_dpp_mods.spv 1 dump -> out[0..7] = 0xaaaa3c00 0xbbbb8000 0x80000000 0xbf800000
;                                                             0xaaaa3c00 0xbbbbc000 0xbf800000 0xbf800000
v_mbcnt_lo_u32_b32 v0, -1, 0
v_mov_b32 v1, 0x12343c00
v_mov_b32 v2, 0x56784000
v_mov_b32 v3, 1.0
v_bfrev_b32 v4, 1
v_mov_b32 v10, 0xaaaa0000
v_mov_b32 v11, 0xbbbb0000
v_add_f16_dpp v10, -v1, v2 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf
v_mul_f16_dpp v11, -|v1|, v2 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
v_add_f32_dpp v12, -v3, v4 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
v_cvt_f32_f16_dpp v13, -v1 quad_perm:[0,0,0,0] row_mask:0xf bank_mask:0xf
v_lshlrev_b32 v5, 4, v0
s_mov_b32 exec_lo, 3
buffer_store_dwordx4 v[10:13], v5, s[0:3], 0 offen
s_endpgm
