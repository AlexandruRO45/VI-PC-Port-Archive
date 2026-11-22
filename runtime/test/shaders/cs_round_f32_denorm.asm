; regen_fixtures: this source has a translator fixture in tests/semantic/fixtures/disasm
; group: 8,1,1
; mattr: +wavefrontsize32,-wavefrontsize64
; v_floor_f32, v_ceil_f32, v_fract_f32 and v_cvt_flr_i32_f32 with an f32 denormal, an infinity or a signalling NaN.
; Lane i takes one operand and keeps one of the four results:
;   0: floor(-1e-45 = 0x80000001)  1: ceil(+1e-45 = 0x00000001)  2: fract(0x80000001)  3: fract(0x00000001)
;   4: fract(+inf)  5: fract(-inf)  6: cvt_flr_i32(0x80000001)  7: floor(signalling NaN 0x7fa00000)
; Measured natively on RDNA2 (gfx1036, Ryzen 9 9950X3D iGPU, AMD driver 32.0.21045.16010, tools/hw_oracle) in the MODE
; inferred for the game (IEEE 0, DX10_CLAMP 1, f32 denormals flushed); the rows of
; tests/semantic/test_alu_floor_ceil_fract_hw.py. A denormal operand is a zero of its sign (using the denormal gives
; -1.0, +1.0, 0x3f7fffff, 0x00000001 and -1), fract(+-inf) is 0xffc00000 and a NaN passes through unchanged.
; Run: vk_compute_test cs_round_f32_denorm.spv 1 dump
; out[0..7] = 80000000 00000000 00000000 00000000 FFC00000 FFC00000 00000000 7FA00000 (hex)
v_mov_b32_e32 v1, 0x80000001
v_mov_b32_e32 v2, 0x00000001
v_cmp_eq_u32_e32 vcc_lo, 1, v0
v_cndmask_b32_e32 v1, v1, v2, vcc_lo
v_cmp_eq_u32_e32 vcc_lo, 3, v0
v_cndmask_b32_e32 v1, v1, v2, vcc_lo
v_mov_b32_e32 v2, 0x7f800000
v_cmp_eq_u32_e32 vcc_lo, 4, v0
v_cndmask_b32_e32 v1, v1, v2, vcc_lo
v_mov_b32_e32 v2, 0xff800000
v_cmp_eq_u32_e32 vcc_lo, 5, v0
v_cndmask_b32_e32 v1, v1, v2, vcc_lo
v_mov_b32_e32 v2, 0x7fa00000
v_cmp_eq_u32_e32 vcc_lo, 7, v0
v_cndmask_b32_e32 v1, v1, v2, vcc_lo
v_floor_f32_e32 v3, v1
v_ceil_f32_e32 v4, v1
v_fract_f32_e32 v5, v1
v_cvt_flr_i32_f32_e32 v6, v1
v_cmp_eq_u32_e32 vcc_lo, 1, v0
v_cndmask_b32_e32 v3, v3, v4, vcc_lo
v_cmp_eq_u32_e32 vcc_lo, 2, v0
v_cndmask_b32_e32 v3, v3, v5, vcc_lo
v_cmp_eq_u32_e32 vcc_lo, 3, v0
v_cndmask_b32_e32 v3, v3, v5, vcc_lo
v_cmp_eq_u32_e32 vcc_lo, 4, v0
v_cndmask_b32_e32 v3, v3, v5, vcc_lo
v_cmp_eq_u32_e32 vcc_lo, 5, v0
v_cndmask_b32_e32 v3, v3, v5, vcc_lo
v_cmp_eq_u32_e32 vcc_lo, 6, v0
v_cndmask_b32_e32 v3, v3, v6, vcc_lo
v_lshlrev_b32_e32 v7, 2, v0
buffer_store_dword v3, v7, s[0:3], 0 offen
s_endpgm
