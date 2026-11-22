; v_add_f64 rounding without denormals, the same words on native doubles and under the translator's --f64-soft.
; w0: 1.0 + 1.5 ulp is a tie, nearest even gives 1.0 + 2 ulp (low word 2; 1 ulp would give 1).
; w1: -0 + -0 is -0 (high word 0x80000000; +0 would be 0).
; Then MODE.FP_ROUND[3:2] = 3 (toward zero, s_setreg): w2: (1.0 + 2 ulp) + (1 ulp - 2^-105) truncates to 1.0 + 2 ulp
; (low word 2; nearest would give 3). w3: DBL_MAX + DBL_MAX is DBL_MAX toward zero (high word 0x7fefffff; inf would be
; 0x7ff00000), exported shifted right by 8 (0x007fefff): both high words read as f32 NaNs, and a driver may rewrite a NaN
; it stores to a float colour attachment (KosmicKrisp writes 0x7fc00000 for every NaN pattern).
; Expected mrt0 bits: 0x00000002, 0x80000000, 0x00000002, 0x007fefff
v_mov_b32 v0, 0
v_mov_b32 v1, 0x3ff00000
v_mov_b32 v2, 0
v_mov_b32 v3, 0x3cb80000
v_add_f64 v[4:5], v[0:1], v[2:3]
v_mov_b32 v6, 0
v_mov_b32 v7, 0x80000000
v_add_f64 v[8:9], v[6:7], v[6:7]
s_setreg_imm32_b32 hwreg(HW_REG_MODE, 2, 2), 3
v_mov_b32 v10, 2
v_mov_b32 v11, 0x3ff00000
v_mov_b32 v12, -1
v_mov_b32 v13, 0x3cafffff
v_add_f64 v[14:15], v[10:11], v[12:13]
v_mov_b32 v16, -1
v_mov_b32 v17, 0x7fefffff
v_add_f64 v[18:19], v[16:17], v[16:17]
v_lshrrev_b32 v20, 8, v19
exp mrt0 v4, v9, v14, v20 done vm
s_endpgm
