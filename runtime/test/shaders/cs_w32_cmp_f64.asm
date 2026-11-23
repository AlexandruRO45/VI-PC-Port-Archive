; v_cmp_*_f64 / v_cmpx_*_f64 (docs/design.md row "v_cmp_*_f64, v_cmpx_*_f64"), run on a GPU.
; Every lane computes the same values; EXEC is cut to lane 0 before the stores, so the 32 lanes do not race on one address (as in cs_w32_new_ops). Predicate order k = f lt eq le gt lg ge o u nge nlg ngt nle neq nlt tru.
;   out[0] bit k = v_cmp_<k>_f64(1.0, 2.0), bit 16+k = v_cmp_<k>_f64(2.0, 2.0) = 0xCCCCAAAA
;   out[1] bit k = v_cmp_<k>_f64(-0.0, 0.0), bit 16+k = v_cmp_<k>_f64(1.0, NaN) = 0xFF00CCCC
;   out[2] bit k = v_cmp_<k>_f64(-inf, inf), bit 16+k = v_cmp_<k>_f64(2.0, 1.0) = 0xF0F0AAAA
;   out[3] = modifiers and sources, one bit each (-x, |x|, -|x|, SGPR pair, inline 1.0, inline 0.5 vs -0 = 0, -NaN, -1 > -2) = 0xDF
;   out[4] = 0x55: v_cmpx_lt_f64(2, 1) is false, so EXEC is off and v_mov 1 does not run (EXEC restored after)
;   out[5] = 2: v_cmpx_gt_f64(2, 1) is true, so EXEC stays on and v_mov 2 runs
; group: 32,1,1
; Run: vk_compute_test cs_w32_cmp_f64.spv 1 dump -> out[0..5] = 3435965098 4278242508 4042304170 223 85 2
; mattr: +wavefrontsize32,-wavefrontsize64
v_mov_b32_e32 v40, 1
v_mov_b32_e32 v20, 0
v_mov_b32_e32 v10, 0
v_mov_b32_e32 v11, 0x3ff00000
v_mov_b32_e32 v12, 0
v_mov_b32_e32 v13, 0x40000000
v_cmp_f_f64_e32 vcc_lo, v[10:11], v[12:13]
v_cndmask_b32_e32 v6, 0, v40, vcc_lo
v_or_b32_e32 v20, v20, v6
v_cmp_lt_f64_e32 vcc_lo, v[10:11], v[12:13]
v_cndmask_b32_e32 v6, 0, v40, vcc_lo
v_lshlrev_b32_e32 v6, 1, v6
v_or_b32_e32 v20, v20, v6
v_cmp_eq_f64_e32 vcc_lo, v[10:11], v[12:13]
v_cndmask_b32_e32 v6, 0, v40, vcc_lo
v_lshlrev_b32_e32 v6, 2, v6
v_or_b32_e32 v20, v20, v6
v_cmp_le_f64_e32 vcc_lo, v[10:11], v[12:13]
v_cndmask_b32_e32 v6, 0, v40, vcc_lo
v_lshlrev_b32_e32 v6, 3, v6
v_or_b32_e32 v20, v20, v6
v_cmp_gt_f64_e32 vcc_lo, v[10:11], v[12:13]
v_cndmask_b32_e32 v6, 0, v40, vcc_lo
v_lshlrev_b32_e32 v6, 4, v6
v_or_b32_e32 v20, v20, v6
v_cmp_lg_f64_e32 vcc_lo, v[10:11], v[12:13]
v_cndmask_b32_e32 v6, 0, v40, vcc_lo
v_lshlrev_b32_e32 v6, 5, v6
v_or_b32_e32 v20, v20, v6
v_cmp_ge_f64_e32 vcc_lo, v[10:11], v[12:13]
v_cndmask_b32_e32 v6, 0, v40, vcc_lo
v_lshlrev_b32_e32 v6, 6, v6
v_or_b32_e32 v20, v20, v6
v_cmp_o_f64_e32 vcc_lo, v[10:11], v[12:13]
v_cndmask_b32_e32 v6, 0, v40, vcc_lo
v_lshlrev_b32_e32 v6, 7, v6
v_or_b32_e32 v20, v20, v6
v_cmp_u_f64_e32 vcc_lo, v[10:11], v[12:13]
v_cndmask_b32_e32 v6, 0, v40, vcc_lo
v_lshlrev_b32_e32 v6, 8, v6
v_or_b32_e32 v20, v20, v6
v_cmp_nge_f64_e32 vcc_lo, v[10:11], v[12:13]
v_cndmask_b32_e32 v6, 0, v40, vcc_lo
v_lshlrev_b32_e32 v6, 9, v6
v_or_b32_e32 v20, v20, v6
v_cmp_nlg_f64_e32 vcc_lo, v[10:11], v[12:13]
v_cndmask_b32_e32 v6, 0, v40, vcc_lo
v_lshlrev_b32_e32 v6, 10, v6
v_or_b32_e32 v20, v20, v6
v_cmp_ngt_f64_e32 vcc_lo, v[10:11], v[12:13]
v_cndmask_b32_e32 v6, 0, v40, vcc_lo
v_lshlrev_b32_e32 v6, 11, v6
v_or_b32_e32 v20, v20, v6
v_cmp_nle_f64_e32 vcc_lo, v[10:11], v[12:13]
v_cndmask_b32_e32 v6, 0, v40, vcc_lo
v_lshlrev_b32_e32 v6, 12, v6
v_or_b32_e32 v20, v20, v6
v_cmp_neq_f64_e32 vcc_lo, v[10:11], v[12:13]
v_cndmask_b32_e32 v6, 0, v40, vcc_lo
v_lshlrev_b32_e32 v6, 13, v6
v_or_b32_e32 v20, v20, v6
v_cmp_nlt_f64_e32 vcc_lo, v[10:11], v[12:13]
v_cndmask_b32_e32 v6, 0, v40, vcc_lo
v_lshlrev_b32_e32 v6, 14, v6
v_or_b32_e32 v20, v20, v6
v_cmp_tru_f64_e32 vcc_lo, v[10:11], v[12:13]
v_cndmask_b32_e32 v6, 0, v40, vcc_lo
v_lshlrev_b32_e32 v6, 15, v6
v_or_b32_e32 v20, v20, v6
v_mov_b32_e32 v10, 0
v_mov_b32_e32 v11, 0x40000000
v_mov_b32_e32 v12, 0
v_mov_b32_e32 v13, 0x40000000
v_cmp_f_f64_e32 vcc_lo, v[10:11], v[12:13]
v_cndmask_b32_e32 v6, 0, v40, vcc_lo
v_lshlrev_b32_e32 v6, 16, v6
v_or_b32_e32 v20, v20, v6
v_cmp_lt_f64_e32 vcc_lo, v[10:11], v[12:13]
v_cndmask_b32_e32 v6, 0, v40, vcc_lo
v_lshlrev_b32_e32 v6, 17, v6
v_or_b32_e32 v20, v20, v6
v_cmp_eq_f64_e32 vcc_lo, v[10:11], v[12:13]
v_cndmask_b32_e32 v6, 0, v40, vcc_lo
v_lshlrev_b32_e32 v6, 18, v6
v_or_b32_e32 v20, v20, v6
v_cmp_le_f64_e32 vcc_lo, v[10:11], v[12:13]
v_cndmask_b32_e32 v6, 0, v40, vcc_lo
v_lshlrev_b32_e32 v6, 19, v6
v_or_b32_e32 v20, v20, v6
v_cmp_gt_f64_e32 vcc_lo, v[10:11], v[12:13]
v_cndmask_b32_e32 v6, 0, v40, vcc_lo
v_lshlrev_b32_e32 v6, 20, v6
v_or_b32_e32 v20, v20, v6
v_cmp_lg_f64_e32 vcc_lo, v[10:11], v[12:13]
v_cndmask_b32_e32 v6, 0, v40, vcc_lo
v_lshlrev_b32_e32 v6, 21, v6
v_or_b32_e32 v20, v20, v6
v_cmp_ge_f64_e32 vcc_lo, v[10:11], v[12:13]
v_cndmask_b32_e32 v6, 0, v40, vcc_lo
v_lshlrev_b32_e32 v6, 22, v6
v_or_b32_e32 v20, v20, v6
v_cmp_o_f64_e32 vcc_lo, v[10:11], v[12:13]
v_cndmask_b32_e32 v6, 0, v40, vcc_lo
v_lshlrev_b32_e32 v6, 23, v6
v_or_b32_e32 v20, v20, v6
v_cmp_u_f64_e32 vcc_lo, v[10:11], v[12:13]
v_cndmask_b32_e32 v6, 0, v40, vcc_lo
v_lshlrev_b32_e32 v6, 24, v6
v_or_b32_e32 v20, v20, v6
v_cmp_nge_f64_e32 vcc_lo, v[10:11], v[12:13]
v_cndmask_b32_e32 v6, 0, v40, vcc_lo
v_lshlrev_b32_e32 v6, 25, v6
v_or_b32_e32 v20, v20, v6
v_cmp_nlg_f64_e32 vcc_lo, v[10:11], v[12:13]
v_cndmask_b32_e32 v6, 0, v40, vcc_lo
v_lshlrev_b32_e32 v6, 26, v6
v_or_b32_e32 v20, v20, v6
v_cmp_ngt_f64_e32 vcc_lo, v[10:11], v[12:13]
v_cndmask_b32_e32 v6, 0, v40, vcc_lo
v_lshlrev_b32_e32 v6, 27, v6
v_or_b32_e32 v20, v20, v6
v_cmp_nle_f64_e32 vcc_lo, v[10:11], v[12:13]
v_cndmask_b32_e32 v6, 0, v40, vcc_lo
v_lshlrev_b32_e32 v6, 28, v6
v_or_b32_e32 v20, v20, v6
v_cmp_neq_f64_e32 vcc_lo, v[10:11], v[12:13]
v_cndmask_b32_e32 v6, 0, v40, vcc_lo
v_lshlrev_b32_e32 v6, 29, v6
v_or_b32_e32 v20, v20, v6
v_cmp_nlt_f64_e32 vcc_lo, v[10:11], v[12:13]
v_cndmask_b32_e32 v6, 0, v40, vcc_lo
v_lshlrev_b32_e32 v6, 30, v6
v_or_b32_e32 v20, v20, v6
v_cmp_tru_f64_e32 vcc_lo, v[10:11], v[12:13]
v_cndmask_b32_e32 v6, 0, v40, vcc_lo
v_lshlrev_b32_e32 v6, 31, v6
v_or_b32_e32 v20, v20, v6
v_mov_b32_e32 v21, 0
v_mov_b32_e32 v10, 0
v_mov_b32_e32 v11, 0x80000000
v_mov_b32_e32 v12, 0
v_mov_b32_e32 v13, 0x0
v_cmp_f_f64_e32 vcc_lo, v[10:11], v[12:13]
v_cndmask_b32_e32 v6, 0, v40, vcc_lo
v_or_b32_e32 v21, v21, v6
v_cmp_lt_f64_e32 vcc_lo, v[10:11], v[12:13]
v_cndmask_b32_e32 v6, 0, v40, vcc_lo
v_lshlrev_b32_e32 v6, 1, v6
v_or_b32_e32 v21, v21, v6
v_cmp_eq_f64_e32 vcc_lo, v[10:11], v[12:13]
v_cndmask_b32_e32 v6, 0, v40, vcc_lo
v_lshlrev_b32_e32 v6, 2, v6
v_or_b32_e32 v21, v21, v6
v_cmp_le_f64_e32 vcc_lo, v[10:11], v[12:13]
v_cndmask_b32_e32 v6, 0, v40, vcc_lo
v_lshlrev_b32_e32 v6, 3, v6
v_or_b32_e32 v21, v21, v6
v_cmp_gt_f64_e32 vcc_lo, v[10:11], v[12:13]
v_cndmask_b32_e32 v6, 0, v40, vcc_lo
v_lshlrev_b32_e32 v6, 4, v6
v_or_b32_e32 v21, v21, v6
v_cmp_lg_f64_e32 vcc_lo, v[10:11], v[12:13]
v_cndmask_b32_e32 v6, 0, v40, vcc_lo
v_lshlrev_b32_e32 v6, 5, v6
v_or_b32_e32 v21, v21, v6
v_cmp_ge_f64_e32 vcc_lo, v[10:11], v[12:13]
v_cndmask_b32_e32 v6, 0, v40, vcc_lo
v_lshlrev_b32_e32 v6, 6, v6
v_or_b32_e32 v21, v21, v6
v_cmp_o_f64_e32 vcc_lo, v[10:11], v[12:13]
v_cndmask_b32_e32 v6, 0, v40, vcc_lo
v_lshlrev_b32_e32 v6, 7, v6
v_or_b32_e32 v21, v21, v6
v_cmp_u_f64_e32 vcc_lo, v[10:11], v[12:13]
v_cndmask_b32_e32 v6, 0, v40, vcc_lo
v_lshlrev_b32_e32 v6, 8, v6
v_or_b32_e32 v21, v21, v6
v_cmp_nge_f64_e32 vcc_lo, v[10:11], v[12:13]
v_cndmask_b32_e32 v6, 0, v40, vcc_lo
v_lshlrev_b32_e32 v6, 9, v6
v_or_b32_e32 v21, v21, v6
v_cmp_nlg_f64_e32 vcc_lo, v[10:11], v[12:13]
v_cndmask_b32_e32 v6, 0, v40, vcc_lo
v_lshlrev_b32_e32 v6, 10, v6
v_or_b32_e32 v21, v21, v6
v_cmp_ngt_f64_e32 vcc_lo, v[10:11], v[12:13]
v_cndmask_b32_e32 v6, 0, v40, vcc_lo
v_lshlrev_b32_e32 v6, 11, v6
v_or_b32_e32 v21, v21, v6
v_cmp_nle_f64_e32 vcc_lo, v[10:11], v[12:13]
v_cndmask_b32_e32 v6, 0, v40, vcc_lo
v_lshlrev_b32_e32 v6, 12, v6
v_or_b32_e32 v21, v21, v6
v_cmp_neq_f64_e32 vcc_lo, v[10:11], v[12:13]
v_cndmask_b32_e32 v6, 0, v40, vcc_lo
v_lshlrev_b32_e32 v6, 13, v6
v_or_b32_e32 v21, v21, v6
v_cmp_nlt_f64_e32 vcc_lo, v[10:11], v[12:13]
v_cndmask_b32_e32 v6, 0, v40, vcc_lo
v_lshlrev_b32_e32 v6, 14, v6
v_or_b32_e32 v21, v21, v6
v_cmp_tru_f64_e32 vcc_lo, v[10:11], v[12:13]
v_cndmask_b32_e32 v6, 0, v40, vcc_lo
v_lshlrev_b32_e32 v6, 15, v6
v_or_b32_e32 v21, v21, v6
v_mov_b32_e32 v10, 0
v_mov_b32_e32 v11, 0x3ff00000
v_mov_b32_e32 v12, 0
v_mov_b32_e32 v13, 0x7ff80000
v_cmp_f_f64_e32 vcc_lo, v[10:11], v[12:13]
v_cndmask_b32_e32 v6, 0, v40, vcc_lo
v_lshlrev_b32_e32 v6, 16, v6
v_or_b32_e32 v21, v21, v6
v_cmp_lt_f64_e32 vcc_lo, v[10:11], v[12:13]
v_cndmask_b32_e32 v6, 0, v40, vcc_lo
v_lshlrev_b32_e32 v6, 17, v6
v_or_b32_e32 v21, v21, v6
v_cmp_eq_f64_e32 vcc_lo, v[10:11], v[12:13]
v_cndmask_b32_e32 v6, 0, v40, vcc_lo
v_lshlrev_b32_e32 v6, 18, v6
v_or_b32_e32 v21, v21, v6
v_cmp_le_f64_e32 vcc_lo, v[10:11], v[12:13]
v_cndmask_b32_e32 v6, 0, v40, vcc_lo
v_lshlrev_b32_e32 v6, 19, v6
v_or_b32_e32 v21, v21, v6
v_cmp_gt_f64_e32 vcc_lo, v[10:11], v[12:13]
v_cndmask_b32_e32 v6, 0, v40, vcc_lo
v_lshlrev_b32_e32 v6, 20, v6
v_or_b32_e32 v21, v21, v6
v_cmp_lg_f64_e32 vcc_lo, v[10:11], v[12:13]
v_cndmask_b32_e32 v6, 0, v40, vcc_lo
v_lshlrev_b32_e32 v6, 21, v6
v_or_b32_e32 v21, v21, v6
v_cmp_ge_f64_e32 vcc_lo, v[10:11], v[12:13]
v_cndmask_b32_e32 v6, 0, v40, vcc_lo
v_lshlrev_b32_e32 v6, 22, v6
v_or_b32_e32 v21, v21, v6
v_cmp_o_f64_e32 vcc_lo, v[10:11], v[12:13]
v_cndmask_b32_e32 v6, 0, v40, vcc_lo
v_lshlrev_b32_e32 v6, 23, v6
v_or_b32_e32 v21, v21, v6
v_cmp_u_f64_e32 vcc_lo, v[10:11], v[12:13]
v_cndmask_b32_e32 v6, 0, v40, vcc_lo
v_lshlrev_b32_e32 v6, 24, v6
v_or_b32_e32 v21, v21, v6
v_cmp_nge_f64_e32 vcc_lo, v[10:11], v[12:13]
v_cndmask_b32_e32 v6, 0, v40, vcc_lo
v_lshlrev_b32_e32 v6, 25, v6
v_or_b32_e32 v21, v21, v6
v_cmp_nlg_f64_e32 vcc_lo, v[10:11], v[12:13]
v_cndmask_b32_e32 v6, 0, v40, vcc_lo
v_lshlrev_b32_e32 v6, 26, v6
v_or_b32_e32 v21, v21, v6
v_cmp_ngt_f64_e32 vcc_lo, v[10:11], v[12:13]
v_cndmask_b32_e32 v6, 0, v40, vcc_lo
v_lshlrev_b32_e32 v6, 27, v6
v_or_b32_e32 v21, v21, v6
v_cmp_nle_f64_e32 vcc_lo, v[10:11], v[12:13]
v_cndmask_b32_e32 v6, 0, v40, vcc_lo
v_lshlrev_b32_e32 v6, 28, v6
v_or_b32_e32 v21, v21, v6
v_cmp_neq_f64_e32 vcc_lo, v[10:11], v[12:13]
v_cndmask_b32_e32 v6, 0, v40, vcc_lo
v_lshlrev_b32_e32 v6, 29, v6
v_or_b32_e32 v21, v21, v6
v_cmp_nlt_f64_e32 vcc_lo, v[10:11], v[12:13]
v_cndmask_b32_e32 v6, 0, v40, vcc_lo
v_lshlrev_b32_e32 v6, 30, v6
v_or_b32_e32 v21, v21, v6
v_cmp_tru_f64_e32 vcc_lo, v[10:11], v[12:13]
v_cndmask_b32_e32 v6, 0, v40, vcc_lo
v_lshlrev_b32_e32 v6, 31, v6
v_or_b32_e32 v21, v21, v6
v_mov_b32_e32 v22, 0
v_mov_b32_e32 v10, 0
v_mov_b32_e32 v11, 0xfff00000
v_mov_b32_e32 v12, 0
v_mov_b32_e32 v13, 0x7ff00000
v_cmp_f_f64_e32 vcc_lo, v[10:11], v[12:13]
v_cndmask_b32_e32 v6, 0, v40, vcc_lo
v_or_b32_e32 v22, v22, v6
v_cmp_lt_f64_e32 vcc_lo, v[10:11], v[12:13]
v_cndmask_b32_e32 v6, 0, v40, vcc_lo
v_lshlrev_b32_e32 v6, 1, v6
v_or_b32_e32 v22, v22, v6
v_cmp_eq_f64_e32 vcc_lo, v[10:11], v[12:13]
v_cndmask_b32_e32 v6, 0, v40, vcc_lo
v_lshlrev_b32_e32 v6, 2, v6
v_or_b32_e32 v22, v22, v6
v_cmp_le_f64_e32 vcc_lo, v[10:11], v[12:13]
v_cndmask_b32_e32 v6, 0, v40, vcc_lo
v_lshlrev_b32_e32 v6, 3, v6
v_or_b32_e32 v22, v22, v6
v_cmp_gt_f64_e32 vcc_lo, v[10:11], v[12:13]
v_cndmask_b32_e32 v6, 0, v40, vcc_lo
v_lshlrev_b32_e32 v6, 4, v6
v_or_b32_e32 v22, v22, v6
v_cmp_lg_f64_e32 vcc_lo, v[10:11], v[12:13]
v_cndmask_b32_e32 v6, 0, v40, vcc_lo
v_lshlrev_b32_e32 v6, 5, v6
v_or_b32_e32 v22, v22, v6
v_cmp_ge_f64_e32 vcc_lo, v[10:11], v[12:13]
v_cndmask_b32_e32 v6, 0, v40, vcc_lo
v_lshlrev_b32_e32 v6, 6, v6
v_or_b32_e32 v22, v22, v6
v_cmp_o_f64_e32 vcc_lo, v[10:11], v[12:13]
v_cndmask_b32_e32 v6, 0, v40, vcc_lo
v_lshlrev_b32_e32 v6, 7, v6
v_or_b32_e32 v22, v22, v6
v_cmp_u_f64_e32 vcc_lo, v[10:11], v[12:13]
v_cndmask_b32_e32 v6, 0, v40, vcc_lo
v_lshlrev_b32_e32 v6, 8, v6
v_or_b32_e32 v22, v22, v6
v_cmp_nge_f64_e32 vcc_lo, v[10:11], v[12:13]
v_cndmask_b32_e32 v6, 0, v40, vcc_lo
v_lshlrev_b32_e32 v6, 9, v6
v_or_b32_e32 v22, v22, v6
v_cmp_nlg_f64_e32 vcc_lo, v[10:11], v[12:13]
v_cndmask_b32_e32 v6, 0, v40, vcc_lo
v_lshlrev_b32_e32 v6, 10, v6
v_or_b32_e32 v22, v22, v6
v_cmp_ngt_f64_e32 vcc_lo, v[10:11], v[12:13]
v_cndmask_b32_e32 v6, 0, v40, vcc_lo
v_lshlrev_b32_e32 v6, 11, v6
v_or_b32_e32 v22, v22, v6
v_cmp_nle_f64_e32 vcc_lo, v[10:11], v[12:13]
v_cndmask_b32_e32 v6, 0, v40, vcc_lo
v_lshlrev_b32_e32 v6, 12, v6
v_or_b32_e32 v22, v22, v6
v_cmp_neq_f64_e32 vcc_lo, v[10:11], v[12:13]
v_cndmask_b32_e32 v6, 0, v40, vcc_lo
v_lshlrev_b32_e32 v6, 13, v6
v_or_b32_e32 v22, v22, v6
v_cmp_nlt_f64_e32 vcc_lo, v[10:11], v[12:13]
v_cndmask_b32_e32 v6, 0, v40, vcc_lo
v_lshlrev_b32_e32 v6, 14, v6
v_or_b32_e32 v22, v22, v6
v_cmp_tru_f64_e32 vcc_lo, v[10:11], v[12:13]
v_cndmask_b32_e32 v6, 0, v40, vcc_lo
v_lshlrev_b32_e32 v6, 15, v6
v_or_b32_e32 v22, v22, v6
v_mov_b32_e32 v10, 0
v_mov_b32_e32 v11, 0x40000000
v_mov_b32_e32 v12, 0
v_mov_b32_e32 v13, 0x3ff00000
v_cmp_f_f64_e32 vcc_lo, v[10:11], v[12:13]
v_cndmask_b32_e32 v6, 0, v40, vcc_lo
v_lshlrev_b32_e32 v6, 16, v6
v_or_b32_e32 v22, v22, v6
v_cmp_lt_f64_e32 vcc_lo, v[10:11], v[12:13]
v_cndmask_b32_e32 v6, 0, v40, vcc_lo
v_lshlrev_b32_e32 v6, 17, v6
v_or_b32_e32 v22, v22, v6
v_cmp_eq_f64_e32 vcc_lo, v[10:11], v[12:13]
v_cndmask_b32_e32 v6, 0, v40, vcc_lo
v_lshlrev_b32_e32 v6, 18, v6
v_or_b32_e32 v22, v22, v6
v_cmp_le_f64_e32 vcc_lo, v[10:11], v[12:13]
v_cndmask_b32_e32 v6, 0, v40, vcc_lo
v_lshlrev_b32_e32 v6, 19, v6
v_or_b32_e32 v22, v22, v6
v_cmp_gt_f64_e32 vcc_lo, v[10:11], v[12:13]
v_cndmask_b32_e32 v6, 0, v40, vcc_lo
v_lshlrev_b32_e32 v6, 20, v6
v_or_b32_e32 v22, v22, v6
v_cmp_lg_f64_e32 vcc_lo, v[10:11], v[12:13]
v_cndmask_b32_e32 v6, 0, v40, vcc_lo
v_lshlrev_b32_e32 v6, 21, v6
v_or_b32_e32 v22, v22, v6
v_cmp_ge_f64_e32 vcc_lo, v[10:11], v[12:13]
v_cndmask_b32_e32 v6, 0, v40, vcc_lo
v_lshlrev_b32_e32 v6, 22, v6
v_or_b32_e32 v22, v22, v6
v_cmp_o_f64_e32 vcc_lo, v[10:11], v[12:13]
v_cndmask_b32_e32 v6, 0, v40, vcc_lo
v_lshlrev_b32_e32 v6, 23, v6
v_or_b32_e32 v22, v22, v6
v_cmp_u_f64_e32 vcc_lo, v[10:11], v[12:13]
v_cndmask_b32_e32 v6, 0, v40, vcc_lo
v_lshlrev_b32_e32 v6, 24, v6
v_or_b32_e32 v22, v22, v6
v_cmp_nge_f64_e32 vcc_lo, v[10:11], v[12:13]
v_cndmask_b32_e32 v6, 0, v40, vcc_lo
v_lshlrev_b32_e32 v6, 25, v6
v_or_b32_e32 v22, v22, v6
v_cmp_nlg_f64_e32 vcc_lo, v[10:11], v[12:13]
v_cndmask_b32_e32 v6, 0, v40, vcc_lo
v_lshlrev_b32_e32 v6, 26, v6
v_or_b32_e32 v22, v22, v6
v_cmp_ngt_f64_e32 vcc_lo, v[10:11], v[12:13]
v_cndmask_b32_e32 v6, 0, v40, vcc_lo
v_lshlrev_b32_e32 v6, 27, v6
v_or_b32_e32 v22, v22, v6
v_cmp_nle_f64_e32 vcc_lo, v[10:11], v[12:13]
v_cndmask_b32_e32 v6, 0, v40, vcc_lo
v_lshlrev_b32_e32 v6, 28, v6
v_or_b32_e32 v22, v22, v6
v_cmp_neq_f64_e32 vcc_lo, v[10:11], v[12:13]
v_cndmask_b32_e32 v6, 0, v40, vcc_lo
v_lshlrev_b32_e32 v6, 29, v6
v_or_b32_e32 v22, v22, v6
v_cmp_nlt_f64_e32 vcc_lo, v[10:11], v[12:13]
v_cndmask_b32_e32 v6, 0, v40, vcc_lo
v_lshlrev_b32_e32 v6, 30, v6
v_or_b32_e32 v22, v22, v6
v_cmp_tru_f64_e32 vcc_lo, v[10:11], v[12:13]
v_cndmask_b32_e32 v6, 0, v40, vcc_lo
v_lshlrev_b32_e32 v6, 31, v6
v_or_b32_e32 v22, v22, v6
v_mov_b32_e32 v23, 0
v_mov_b32_e32 v10, 0
v_mov_b32_e32 v11, 0x40000000
v_mov_b32_e32 v12, 0
v_mov_b32_e32 v13, 0x3ff00000
v_mov_b32_e32 v14, 0
v_mov_b32_e32 v15, 0xc0000000
v_mov_b32_e32 v16, 0
v_mov_b32_e32 v17, 0x7ff80000
v_mov_b32_e32 v18, 0
v_mov_b32_e32 v19, 0x80000000
s_mov_b32 s10, 0
s_mov_b32 s11, 0x3ff00000
v_cmp_lt_f64_e64 s8, -v[10:11], v[12:13]
v_cndmask_b32_e64 v6, 0, v40, s8
v_or_b32_e32 v23, v23, v6
v_cmp_eq_f64_e64 s8, |v[14:15]|, v[10:11]
v_cndmask_b32_e64 v6, 0, v40, s8
v_lshlrev_b32_e32 v6, 1, v6
v_or_b32_e32 v23, v23, v6
v_cmp_eq_f64_e64 s8, -|v[10:11]|, v[14:15]
v_cndmask_b32_e64 v6, 0, v40, s8
v_lshlrev_b32_e32 v6, 2, v6
v_or_b32_e32 v23, v23, v6
v_cmp_lt_f64_e32 vcc_lo, s[10:11], v[10:11]
v_cndmask_b32_e32 v6, 0, v40, vcc_lo
v_lshlrev_b32_e32 v6, 3, v6
v_or_b32_e32 v23, v23, v6
v_cmp_eq_f64_e32 vcc_lo, 1.0, v[12:13]
v_cndmask_b32_e32 v6, 0, v40, vcc_lo
v_lshlrev_b32_e32 v6, 4, v6
v_or_b32_e32 v23, v23, v6
v_cmp_lt_f64_e32 vcc_lo, 0.5, v[18:19]
v_cndmask_b32_e32 v6, 0, v40, vcc_lo
v_lshlrev_b32_e32 v6, 5, v6
v_or_b32_e32 v23, v23, v6
v_cmp_u_f64_e64 s8, -v[16:17], v[12:13]
v_cndmask_b32_e64 v6, 0, v40, s8
v_lshlrev_b32_e32 v6, 6, v6
v_or_b32_e32 v23, v23, v6
v_cmp_gt_f64_e64 s8, -v[12:13], v[14:15]
v_cndmask_b32_e64 v6, 0, v40, s8
v_lshlrev_b32_e32 v6, 7, v6
v_or_b32_e32 v23, v23, v6
s_mov_b32 s20, exec_lo
v_mov_b32_e32 v24, 0x55
v_cmpx_lt_f64_e32 v[10:11], v[12:13]
v_mov_b32_e32 v24, 1
s_mov_b32 exec_lo, s20
v_mov_b32_e32 v25, 0x66
v_cmpx_gt_f64_e32 v[10:11], v[12:13]
v_mov_b32_e32 v25, 2
s_mov_b32 exec_lo, s20
v_cmpx_eq_u32_e32 0, v0
v_mov_b32_e32 v30, 0
buffer_store_dword v20, v30, s[0:3], 0 offen
v_mov_b32_e32 v30, 4
buffer_store_dword v21, v30, s[0:3], 0 offen
v_mov_b32_e32 v30, 8
buffer_store_dword v22, v30, s[0:3], 0 offen
v_mov_b32_e32 v30, 12
buffer_store_dword v23, v30, s[0:3], 0 offen
v_mov_b32_e32 v30, 16
buffer_store_dword v24, v30, s[0:3], 0 offen
v_mov_b32_e32 v30, 20
buffer_store_dword v25, v30, s[0:3], 0 offen
s_endpgm
