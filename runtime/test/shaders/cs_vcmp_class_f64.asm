; cs_vcmp_class_f64.asm
; group: 8,1,1
; mattr: +wavefrontsize32,-wavefrontsize64
; Tests v_cmp_class_f64 and v_cmpx_class_f64 across 8 distinct binary64 values:
; Lane 0: sNaN (lo=1, hi=0x7ff00000) -> class bit 0
; Lane 1: qNaN (lo=0, hi=0x7ff80000) -> class bit 1
; Lane 2: -inf (lo=0, hi=0xfff00000) -> class bit 2
; Lane 3: -1.5 (lo=0, hi=0xbff80000) -> class bit 3 (-normal)
; Lane 4: -0.0 (lo=0, hi=0x80000000) -> class bit 5 (-0.0)
; Lane 5: +0.0 (lo=0, hi=0x00000000) -> class bit 6 (+0.0)
; Lane 6: +subnormal (lo=1, hi=0)     -> class bit 7 (+subnormal)
; Lane 7: +2.0 (lo=0, hi=0x40000000) -> class bit 8 (+normal)
;
; Compares tested:
; 1. v_cmp_class_f64_e32 vcc_lo, v[2:3], v4  (mask 0x007: bits 0,1,2 -> lanes 0,1,2 hit, bit 0 in v10)
; 2. v_cmp_class_f64_e64 s4, v[2:3], 0x060   (mask 0x060: bits 5,6 -> lanes 4,5 hit, bit 1 in v10)
; 3. v_cmp_class_f64_e64 s4, v[2:3], 0x108   (mask 0x108: bits 3,8 -> lanes 3,7 hit, bit 2 in v10)
; 4. v_cmp_class_f64_e64 s4, -v[2:3], 0x100  (neg modifier: -(-1.5) = +1.5 is +normal -> lane 3 hits, bit 3 in v10)
; 5. v_cmp_class_f64_e64 s4, |v[2:3]|, 0x008 (abs modifier: |x| is never -normal -> all lanes 0)
; 6. v_cmp_class_f64_e64 s4, v[2:3], 0x380   (mask 0x380: bits 7,8,9 -> lanes 6,7 hit, bit 5 in v10)
; 7. v_cmpx_class_f64_e64 v[2:3], 0x3fe      (narrows EXEC: excludes sNaN bit 0 -> lane 0 disabled)
; Active lanes add 100 to v10 and store. Lane 0 is disabled, leaving untouched 0xEEEEEEEE.
; Expected out[0..7] = 0xEEEEEEEE, 101, 101, 112, 102, 102, 132, 136 (hex: 0xEEEEEEEE, 0x65, 0x65, 0x70, 0x66, 0x66, 0x84, 0x88)
	v_lshlrev_b32_e32 v1, 2, v0
	v_mov_b32_e32 v2, 0
	v_mov_b32_e32 v3, 0
	v_cmp_eq_u32_e32 vcc_lo, 0, v0
	v_cndmask_b32_e64 v2, v2, 1, vcc_lo
	v_cndmask_b32_e64 v3, v3, 0x7ff00000, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, 1, v0
	v_cndmask_b32_e64 v3, v3, 0x7ff80000, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, 2, v0
	v_cndmask_b32_e64 v3, v3, 0xfff00000, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, 3, v0
	v_cndmask_b32_e64 v3, v3, 0xbff80000, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, 4, v0
	v_cndmask_b32_e64 v3, v3, 0x80000000, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, 6, v0
	v_cndmask_b32_e64 v2, v2, 1, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, 7, v0
	v_cndmask_b32_e64 v3, v3, 0x40000000, vcc_lo
	v_mov_b32_e32 v10, 0
	v_mov_b32_e32 v4, 0x007
	v_cmp_class_f64_e32 vcc_lo, v[2:3], v4
	v_cndmask_b32_e64 v5, 0, 1, vcc_lo
	v_or_b32_e32 v10, v10, v5
	v_cmp_class_f64_e64 s4, v[2:3], 0x060
	v_cndmask_b32_e64 v5, 0, 2, s4
	v_or_b32_e32 v10, v10, v5
	v_cmp_class_f64_e64 s4, v[2:3], 0x108
	v_cndmask_b32_e64 v5, 0, 4, s4
	v_or_b32_e32 v10, v10, v5
	v_cmp_class_f64_e64 s4, -v[2:3], 0x100
	v_cndmask_b32_e64 v5, 0, 8, s4
	v_or_b32_e32 v10, v10, v5
	v_cmp_class_f64_e64 s4, |v[2:3]|, 0x008
	v_cndmask_b32_e64 v5, 0, 16, s4
	v_or_b32_e32 v10, v10, v5
	v_cmp_class_f64_e64 s4, v[2:3], 0x380
	v_cndmask_b32_e64 v5, 0, 32, s4
	v_or_b32_e32 v10, v10, v5
	v_cmpx_class_f64_e64 v[2:3], 0x3fe
	v_add_nc_u32_e32 v10, 100, v10
	buffer_store_dword v10, v1, s[0:3], 0 offen
	s_endpgm
