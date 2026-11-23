; cs_cvt_pknorm_f32.asm
; group: 8,1,1
; mattr: +wavefrontsize32,-wavefrontsize64
; Tests v_cvt_pknorm_i16_f32 (lanes 0..3) and v_cvt_pknorm_u16_f32 (lanes 4..7):
; Lane 0: snorm(0.0, 1.0)        -> 0x7FFF0000
; Lane 1: snorm(-1.0, -0.0)      -> 0x00008001
; Lane 2: snorm(0.5, -0.5)       -> 0xC0004000
; Lane 3: snorm(2.0, -2.0)       -> 0x80017FFF
; Lane 4: unorm(0.0, 1.0)        -> 0xFFFF0000
; Lane 5: unorm(0.5, -0.5)       -> 0x00008000
; Lane 6: unorm(2.0, -2.0)       -> 0x0000FFFF
; Lane 7: unorm(qNaN, 0.75)      -> 0xBFFF0000
;
; Expected out[0..7] = 0x7FFF0000, 0x00008001, 0xC0004000, 0x80017FFF, 0xFFFF0000, 0x00008000, 0x0000FFFF, 0xBFFF0000
	v_lshlrev_b32_e32 v1, 2, v0
	v_mov_b32_e32 v2, 0
	v_mov_b32_e32 v3, 0x3f800000
	v_cmp_eq_u32_e32 vcc_lo, 1, v0
	v_cndmask_b32_e64 v2, v2, 0xbf800000, vcc_lo
	v_cndmask_b32_e64 v3, v3, 0x80000000, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, 2, v0
	v_cndmask_b32_e64 v2, v2, 0x3f000000, vcc_lo
	v_cndmask_b32_e64 v3, v3, 0xbf000000, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, 3, v0
	v_cndmask_b32_e64 v2, v2, 0x40000000, vcc_lo
	v_cndmask_b32_e64 v3, v3, 0xc0000000, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, 4, v0
	v_cndmask_b32_e64 v2, v2, 0, vcc_lo
	v_cndmask_b32_e64 v3, v3, 0x3f800000, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, 5, v0
	v_cndmask_b32_e64 v2, v2, 0x3f000000, vcc_lo
	v_cndmask_b32_e64 v3, v3, 0xbf000000, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, 6, v0
	v_cndmask_b32_e64 v2, v2, 0x40000000, vcc_lo
	v_cndmask_b32_e64 v3, v3, 0xc0000000, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, 7, v0
	v_cndmask_b32_e64 v2, v2, 0x7fc00000, vcc_lo
	v_cndmask_b32_e64 v3, v3, 0x3f400000, vcc_lo
	v_cvt_pknorm_i16_f32 v4, v2, v3
	v_cvt_pknorm_u16_f32 v5, v2, v3
	v_cmp_le_u32_e32 vcc_lo, 4, v0
	v_cndmask_b32_e32 v4, v4, v5, vcc_lo
	buffer_store_dword v4, v1, s[0:3], 0 offen
	s_endpgm
