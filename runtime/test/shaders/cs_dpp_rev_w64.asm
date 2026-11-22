; Wave64 form of cs_dpp_rev: DPP on the reversed integer VOP2 ops selects the SECOND source operand (measured on
; RDNA2, gfx1036, raw wave64 kernels), and the float v_subrev_f32 selects the first, like every other op. The
; lanes checked are 32-39, row 2 of the wave, so the test needs 64 lanes: a 64-wide subgroup, or
; LEONIDA_W64_EMULATE=1 on any subgroup.
;
; L = lane (0-63). Every op uses row_shr:1: lane L reads lane L-1, and lane 32, the first of its row, has no source.
;   A(L) = L    S(L) = L & 3    B(L) = 40 + 3L    B2(L) = L + 1    D(L) = -(64 + 8L)    X(L) = L.0    Y(L) = 2L + 10.0
;   d1 = v_subrev_nc_u32_dpp A, B             = B(L-1) - A(L) = 37 + 2L;          lane 32 keeps 0xEE
;   d2 = v_lshlrev_b32_dpp S, B2 bound_ctrl:1 = B2(L-1) << S(L) = L << (L&3);     lane 32: 0 << 0 = 0
;   f  = v_subrev_f32_dpp X, Y                = Y(L) - X(L-1) = L + 11 (first operand selected); lane 32 keeps 5.0
;   d4 = v_ashrrev_i32_dpp S, D bound_ctrl:1  = D(L-1) >> S(L) (arithmetic);      lane 32: 0 >> 0 = 0
; out[L - 32] = d1 | (d2 & 0xff) << 8 | uint(f) << 16 | d4 << 24 for lanes 32-39:
;   L    d1          d2                 f          d4
;   32   0xEE        0x00               5  = 0x05  0x00                     -> 0x000500EE
;   33   103 = 0x67  33<<1 = 66   0x42  44 = 0x2C  -320>>1 = -160  0x60     -> 0x602C4267
;   34   105 = 0x69  34<<2 = 136  0x88  45 = 0x2D  -328>>2 = -82   0xAE     -> 0xAE2D8869
;   35   107 = 0x6B  35<<3 = 280  0x18  46 = 0x2E  -336>>3 = -42   0xD6     -> 0xD62E186B
;   36   109 = 0x6D  36<<0 = 36   0x24  47 = 0x2F  -344>>0 = -344  0xA8     -> 0xA82F246D
;   37   111 = 0x6F  37<<1 = 74   0x4A  48 = 0x30  -352>>1 = -176  0x50     -> 0x50304A6F
;   38   113 = 0x71  38<<2 = 152  0x98  49 = 0x31  -360>>2 = -90   0xA6     -> 0xA6319871
;   39   115 = 0x73  39<<3 = 312  0x38  50 = 0x32  -368>>3 = -46   0xD2     -> 0xD2323873
; Run: vk_compute_test cs_dpp_rev_w64.spv 1 dump -> out[0..7] =
;   327918 1613513319 2922219625 3593345131 2821661805 1345342063 2788268145 3526506611
v_mbcnt_lo_u32_b32 v0, -1, 0
v_mbcnt_hi_u32_b32 v0, -1, v0
v_and_b32 v3, 3, v0
v_mad_u32_u24 v2, v0, 3, 40
v_add_nc_u32 v4, 1, v0
v_lshlrev_b32 v9, 3, v0
v_sub_nc_u32 v6, 0xffffffc0, v9
v_cvt_f32_u32 v14, v0
v_mad_u32_u24 v15, v0, 2, 10
v_cvt_f32_u32 v15, v15
v_mov_b32 v10, 0xee
v_mov_b32 v11, 0x55
v_mov_b32 v16, 0x40a00000
v_mov_b32 v13, 0x55
v_subrev_nc_u32_dpp v10, v0, v2 row_shr:1 row_mask:0xf bank_mask:0xf
v_lshlrev_b32_dpp v11, v3, v4 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
v_subrev_f32_dpp v16, v14, v15 row_shr:1 row_mask:0xf bank_mask:0xf
v_ashrrev_i32_dpp v13, v3, v6 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
v_cvt_u32_f32 v16, v16
v_and_b32 v11, 0xff, v11
v_lshlrev_b32 v11, 8, v11
v_or_b32 v10, v10, v11
v_lshlrev_b32 v16, 16, v16
v_or_b32 v10, v10, v16
v_lshlrev_b32 v13, 24, v13
v_or_b32 v10, v10, v13
v_subrev_nc_u32 v7, 32, v0
v_cmpx_gt_u32 8, v7
v_lshlrev_b32 v8, 2, v7
buffer_store_dword v10, v8, s[0:3], 0 offen
s_endpgm
