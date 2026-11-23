; Carry-in/carry-out VOP2 SDWA and DPP instructions: v_add_co_ci_u32, v_sub_co_ci_u32,
; v_subrev_co_ci_u32 (5-operand canonical forms).
;   out[0] = v_add_co_ci_u32_sdwa (WORD_1, sext(BYTE_1), cin=1 -> WORD_1 UNUSED_PRESERVE) = 0x0FFF1234
;   out[1] = v_sub_co_ci_u32_sdwa consuming carry from out[0] (sext(WORD_0), WORD_0, clamp=no-borrow) = 0xFFFE8000
;   out[2] = v_subrev_co_ci_u32_sdwa with clamp=1 + borrow=1 -> (clamped=0 | borrow_tag=0x2000) = 0x00002000
;   out[3] = v_add_co_ci_u32_dpp row_shr:1 (BC=0) lane 0 (keeps 0x7777, clears VCC=0) = 0x00007777
;   out[4] = v_add_co_ci_u32_dpp row_shr:1 (BC=0) lane 1 (0xFFFFFFFF+10+1 -> 10, VCC=1) = 0x0001000A
;   out[5] = v_sub_co_ci_u32_dpp row_shr:1 bound_ctrl:1 lane 0 (0-5-1=0xFFFFFFFA, borrow=1) = 0x0001FFFA
;   out[6] = v_subrev_co_ci_u32_dpp quad_perm:[1,0,3,2] lane 2 (32-12-1=19, borrow=0) = 0x00000013
;   out[7] = v_add_co_ci_u32_dpp bank_mask:0x1 lane 4 (bank 1 off: keeps 0xABCD, clears VCC=0) = 0x0000ABCD
; group: 32,1,1
; Run: vk_compute_test cs_co_ci_sdwa_dpp.spv 1 dump -> out[0..7] = 268374580 4294868992 8192 30583 65546 131066 19 43981
; mattr: +wavefrontsize32,-wavefrontsize64
v_mbcnt_lo_u32_b32 v0, -1, 0

; ---- 1-3: SDWA carry-in / carry-out, source selects, sext, dst_sel, clamp ----
v_mov_b32 v1, 0xCAFE1234
v_mov_b32 v2, 0x10009999
v_mov_b32 v3, 0x0000FE00
s_mov_b32 vcc_lo, -1
v_add_co_ci_u32_sdwa v1, vcc_lo, v2, sext(v3), vcc_lo dst_sel:WORD_1 dst_unused:UNUSED_PRESERVE src0_sel:WORD_1 src1_sel:BYTE_1

v_mov_b32 v4, 0x00008000
v_mov_b32 v5, 0xDEADFFFF
v_sub_co_ci_u32_sdwa v6, vcc_lo, sext(v4), v5, vcc_lo clamp dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:WORD_0

v_mov_b32 v7, 0x00050000
v_mov_b32 v8, 0x00000005
s_mov_b32 vcc_lo, -1
v_subrev_co_ci_u32_sdwa v9, vcc_lo, v7, v8, vcc_lo clamp dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:BYTE_0
v_cndmask_b32_e64 v10, 1, 0x2000, vcc_lo
v_or_b32 v9, v9, v10

; ---- 4-5: v_add_co_ci_u32_dpp row_shr:1 BC=0 (omitting bound_ctrl gives BC=0) ----
v_add_nc_u32 v12, 10, v0
v_cmp_eq_u32_e32 vcc_lo, 1, v0
v_cndmask_b32_e64 v13, 100, -1, vcc_lo
v_mov_b32 v14, 0x7777
v_cmp_gt_u32_e32 vcc_lo, 2, v0
v_add_co_ci_u32_dpp v14, vcc_lo, v12, v13, vcc_lo row_shr:1 row_mask:0xf bank_mask:0xf
v_cndmask_b32_e64 v15, 0, 1, vcc_lo
v_and_b32 v14, 0xFFFF, v14
v_lshlrev_b32 v15, 16, v15
v_or_b32 v14, v14, v15

; ---- 6: v_sub_co_ci_u32_dpp row_shr:1 bound_ctrl:1 (lane 0 reads src0=0 -> 0-5-1) ----
v_mov_b32 v16, 5
v_mov_b32 v17, 0x7777
v_cmp_eq_u32_e32 vcc_lo, 0, v0
v_sub_co_ci_u32_dpp v17, vcc_lo, v12, v16, vcc_lo row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
v_cndmask_b32_e64 v18, 0, 1, vcc_lo
v_and_b32 v17, 0xFFFF, v17
v_lshlrev_b32 v18, 16, v18
v_or_b32 v17, v17, v18

; ---- 7: v_subrev_co_ci_u32_dpp quad_perm:[1,0,3,2] (lane 2 reads src1 from lane 3 = 32; 32 - 12 - 1 = 19) ----
v_lshlrev_b32 v19, 2, v0
v_add_nc_u32 v19, 20, v19
v_cmp_eq_u32_e32 vcc_lo, 2, v0
v_subrev_co_ci_u32_dpp v20, vcc_lo, v12, v19, vcc_lo quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf
v_cndmask_b32_e64 v21, 0, 1, vcc_lo
v_and_b32 v20, 0xFFFF, v20
v_lshlrev_b32 v21, 16, v21
v_or_b32 v20, v20, v21

; ---- 8: v_add_co_ci_u32_dpp bank_mask:0x1 (lane 4 disabled: keeps v22, clears VCC) ----
v_mov_b32 v22, 0xABCD
v_mov_b32 v25, 1
s_mov_b32 vcc_lo, -1
v_add_co_ci_u32_dpp v22, vcc_lo, v12, v25, vcc_lo quad_perm:[0,0,0,0] row_mask:0xf bank_mask:0x1
v_cndmask_b32_e64 v23, 0, 1, vcc_lo
v_and_b32 v22, 0xFFFF, v22
v_lshlrev_b32 v23, 16, v23
v_or_b32 v22, v22, v23

; ---- Store out[0..7] from lanes 0, 1, 2, 4 ----
s_mov_b32 s8, exec_lo
v_cmpx_eq_u32_e32 0, v0
v_mov_b32 v24, 0
buffer_store_dword v1, v24, s[0:3], 0 offen
v_mov_b32 v24, 4
buffer_store_dword v6, v24, s[0:3], 0 offen
v_mov_b32 v24, 8
buffer_store_dword v9, v24, s[0:3], 0 offen
v_mov_b32 v24, 12
buffer_store_dword v14, v24, s[0:3], 0 offen
v_mov_b32 v24, 20
buffer_store_dword v17, v24, s[0:3], 0 offen

s_mov_b32 exec_lo, s8
v_cmpx_eq_u32_e32 1, v0
v_mov_b32 v24, 16
buffer_store_dword v14, v24, s[0:3], 0 offen

s_mov_b32 exec_lo, s8
v_cmpx_eq_u32_e32 2, v0
v_mov_b32 v24, 24
buffer_store_dword v20, v24, s[0:3], 0 offen

s_mov_b32 exec_lo, s8
v_cmpx_eq_u32_e32 4, v0
v_mov_b32 v24, 28
buffer_store_dword v22, v24, s[0:3], 0 offen

s_endpgm
