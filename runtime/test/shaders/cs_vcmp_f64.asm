; cs_vcmp_f64.asm
; group: 8,1,1
; mattr: +wavefrontsize32,-wavefrontsize64
; Tests 64-bit vector floating-point compares (v_cmp_*_f64 and v_cmpx_*_f64) across 8 representative pairs:
; Lane 0: (-inf, -1.0)              -> lt=1, eq=0, gt=0, o=1, u=0, neq=1, neg_ge=1 -> 105 (+100 = 205 / 0xCD)
; Lane 1: (-0.0, +0.0)              -> lt=0, eq=1, gt=0, o=1, u=0, neq=0, neg_ge=1 -> 74  (+100 = 174 / 0xAE)
; Lane 2: (+1.0, +1.0)              -> lt=0, eq=1, gt=0, o=1, u=0, neq=0, neg_ge=0 -> 10  (+100 = 110 / 0x6E)
; Lane 3: (+2.0, +1.0)              -> lt=0, eq=0, gt=1, o=1, u=0, neq=1, neg_ge=0 -> 44  (+100 = 144 / 0x90)
; Lane 4: (+1.0, +inf)              -> lt=1, eq=0, gt=0, o=1, u=0, neq=1, neg_ge=0 -> 41  (+100 = 141 / 0x8D)
; Lane 5: (qNaN, +1.0)              -> unordered -> disabled by v_cmpx_o_f64 -> 0xEEEEEEEE
; Lane 6: (sNaN, qNaN)              -> unordered -> disabled by v_cmpx_o_f64 -> 0xEEEEEEEE
; Lane 7: (+1.0, +1.0 + 1 ulp)      -> lt=1, eq=0, gt=0, o=1, u=0, neq=1, neg_ge=0 -> 41  (+100 = 141 / 0x8D)
;
; Expected out[0..7] = 205, 174, 110, 144, 141, 0xEEEEEEEE, 0xEEEEEEEE, 141
	v_lshlrev_b32_e32 v1, 2, v0
	v_mov_b32_e32 v2, 0
	v_mov_b32_e32 v3, 0xfff00000
	v_mov_b32_e32 v4, 0
	v_mov_b32_e32 v5, 0xbff00000
	v_cmp_eq_u32_e32 vcc_lo, 1, v0
	v_cndmask_b32_e64 v3, v3, 0x80000000, vcc_lo
	v_cndmask_b32_e64 v5, v5, 0, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, 2, v0
	v_cndmask_b32_e64 v3, v3, 0x3ff00000, vcc_lo
	v_cndmask_b32_e64 v5, v5, 0x3ff00000, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, 3, v0
	v_cndmask_b32_e64 v3, v3, 0x40000000, vcc_lo
	v_cndmask_b32_e64 v5, v5, 0x3ff00000, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, 4, v0
	v_cndmask_b32_e64 v3, v3, 0x3ff00000, vcc_lo
	v_cndmask_b32_e64 v5, v5, 0x7ff00000, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, 5, v0
	v_cndmask_b32_e64 v3, v3, 0x7ff80000, vcc_lo
	v_cndmask_b32_e64 v5, v5, 0x3ff00000, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, 6, v0
	v_cndmask_b32_e64 v2, v2, 1, vcc_lo
	v_cndmask_b32_e64 v3, v3, 0x7ff00000, vcc_lo
	v_cndmask_b32_e64 v5, v5, 0x7ff80000, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, 7, v0
	v_cndmask_b32_e64 v2, v2, 0, vcc_lo
	v_cndmask_b32_e64 v3, v3, 0x3ff00000, vcc_lo
	v_cndmask_b32_e64 v4, v4, 1, vcc_lo
	v_cndmask_b32_e64 v5, v5, 0x3ff00000, vcc_lo
	v_mov_b32_e32 v10, 0
	v_cmp_lt_f64_e32 vcc_lo, v[2:3], v[4:5]
	v_cndmask_b32_e64 v6, 0, 1, vcc_lo
	v_or_b32_e32 v10, v10, v6
	v_cmp_eq_f64_e64 s4, v[2:3], v[4:5]
	v_cndmask_b32_e64 v6, 0, 2, s4
	v_or_b32_e32 v10, v10, v6
	v_cmp_gt_f64_e64 s4, v[2:3], v[4:5]
	v_cndmask_b32_e64 v6, 0, 4, s4
	v_or_b32_e32 v10, v10, v6
	v_cmp_o_f64_e64 s4, v[2:3], v[4:5]
	v_cndmask_b32_e64 v6, 0, 8, s4
	v_or_b32_e32 v10, v10, v6
	v_cmp_u_f64_e64 s4, v[2:3], v[4:5]
	v_cndmask_b32_e64 v6, 0, 16, s4
	v_or_b32_e32 v10, v10, v6
	v_cmp_neq_f64_e64 s4, v[2:3], v[4:5]
	v_cndmask_b32_e64 v6, 0, 32, s4
	v_or_b32_e32 v10, v10, v6
	v_cmp_ge_f64_e64 s4, -v[2:3], v[4:5]
	v_cndmask_b32_e64 v6, 0, 64, s4
	v_or_b32_e32 v10, v10, v6
	v_cmpx_o_f64_e64 v[2:3], v[4:5]
	v_add_nc_u32_e32 v10, 100, v10
	buffer_store_dword v10, v1, s[0:3], 0 offen
	s_endpgm
