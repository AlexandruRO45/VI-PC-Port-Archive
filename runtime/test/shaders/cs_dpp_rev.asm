; DPP on the reversed integer VOP2 ops: the lane select (and the bound_ctrl zero) applies to the SECOND source
; operand, the one the op reads as its left-hand value (D = S1 - S0, D = S1 shifted by S0). Measured on RDNA2
; (gfx1036, raw wave32 and wave64 kernels, row_shr / quad_perm / row_mirror, with and without bound_ctrl):
; v_subrev_nc_u32, v_lshlrev_b32, v_lshrrev_b32 and v_ashrrev_i32 all select S1; v_sub_nc_u32 and the float
; v_subrev_f32 / v_subrev_f16 select S0 like every other op. Until this test the translator selected S0 here too.
;
; One wave32, l = lane. Every op uses row_shr:1, so lane l reads lane l-1 and lane 0 has no source.
;   A(l) = l            S(l) = l & 3          B(l) = 40 + 3l        B2(l) = l + 1
;   C(l) = 200 - 8l     D(l) = -(64 + 8l)
;   d1 = v_subrev_nc_u32_dpp A, B              = B(l-1) - A(l) = 37 + 2l;      lane 0 keeps 0xEE
;   d2 = v_lshlrev_b32_dpp S, B2 bound_ctrl:1  = B2(l-1) << S(l) = l << (l&3); lane 0: 0 << 0 = 0
;   d3 = v_lshrrev_b32_dpp S, C                = C(l-1) >> S(l);               lane 0 keeps 0xDD
;   d4 = v_ashrrev_i32_dpp S, D bound_ctrl:1   = D(l-1) >> S(l) (arithmetic);  lane 0: 0 >> 0 = 0
; out[l] = d1 | d2 << 8 | d3 << 16 | d4 << 24 for lanes 0-7:
;   l   d1         d2          d3              d4
;   0   0xEE       0x00        0xDD            0x00              -> 0x00DD00EE
;   1   39 = 0x27  1<<1 = 0x02 200>>1 = 0x64   -64>>1 = -32  E0  -> 0xE0640227
;   2   41 = 0x29  2<<2 = 0x08 192>>2 = 0x30   -72>>2 = -18  EE  -> 0xEE300829
;   3   43 = 0x2B  3<<3 = 0x18 184>>3 = 0x17   -80>>3 = -10  F6  -> 0xF617182B
;   4   45 = 0x2D  4<<0 = 0x04 176>>0 = 0xB0   -88>>0 = -88  A8  -> 0xA8B0042D
;   5   47 = 0x2F  5<<1 = 0x0A 168>>1 = 0x54   -96>>1 = -48  D0  -> 0xD0540A2F
;   6   49 = 0x31  6<<2 = 0x18 160>>2 = 0x28   -104>>2 = -26 E6  -> 0xE6281831
;   7   51 = 0x33  7<<3 = 0x38 152>>3 = 0x13   -112>>3 = -14 F2  -> 0xF2133833
; (Selecting the first operand instead gives d1 = B(l) - A(l-1) = 41 + 2l, 0x2B in lane 1.)
; Run: vk_compute_test cs_dpp_rev.spv 1 dump -> out[0..7] =
;   14483694 3764650535 3996125225 4128708651 2830107693 3495168559 3861387313 4061345843
v_mbcnt_lo_u32_b32 v0, -1, 0
v_and_b32 v3, 3, v0
v_mad_u32_u24 v2, v0, 3, 40
v_add_nc_u32 v4, 1, v0
v_lshlrev_b32 v9, 3, v0
v_sub_nc_u32 v5, 0xc8, v9
v_sub_nc_u32 v6, 0xffffffc0, v9
v_mov_b32 v10, 0xee
v_mov_b32 v11, 0x55
v_mov_b32 v12, 0xdd
v_mov_b32 v13, 0x55
v_subrev_nc_u32_dpp v10, v0, v2 row_shr:1 row_mask:0xf bank_mask:0xf
v_lshlrev_b32_dpp v11, v3, v4 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
v_lshrrev_b32_dpp v12, v3, v5 row_shr:1 row_mask:0xf bank_mask:0xf
v_ashrrev_i32_dpp v13, v3, v6 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
v_and_b32 v11, 0xff, v11
v_lshlrev_b32 v11, 8, v11
v_or_b32 v10, v10, v11
v_lshlrev_b32 v12, 16, v12
v_or_b32 v10, v10, v12
v_lshlrev_b32 v13, 24, v13
v_or_b32 v10, v10, v13
v_lshlrev_b32 v8, 2, v0
buffer_store_dword v10, v8, s[0:3], 0 offen
s_endpgm
