; regen_fixtures: this source has a translator fixture in tests/semantic/fixtures/disasm
; v_div_fmas_f32 / v_div_fmas_f64: the VCC-scaled FMA used by IEEE division (#268).
; Lane 0 of a wave32 group computes six results and stores eight dwords (the two f64 results are pairs):
;   out[0] = v_div_fmas_f32(2^-75, 2^-75, 0) with VCC=1 = 0                      (S2 = 0 takes the |S2| < 2.0
;                                                                                 post-scale: fma = 2^-150 flushes,
;                                                                                 then * 2^-64 stays zero)
;   out[1] = v_div_fmas_f32(3.0, 5.0, 7.0) with VCC=0 = 22.0                     (plain fma)
;   out[2] = v_div_fmas_f32(2.0, 3.0, 4.0) mul:2 with VCC=0 = 20.0               (output modifier)
;   out[3] = v_div_fmas_f32(-1.0, 1.0, -2.0) clamp with VCC=0 = 0                (clamp to [0, 1])
;   out[4:5] = v_div_fmas_f64(2^-540, 2^-540, 0) with VCC=1 = 0                 (same post-scale: fma = 2^-1080,
;                                                                                 then * 2^-128 stays zero)
;   out[6:7] = v_div_fmas_f64(3.0, 5.0, 7.0) with VCC=0 = 22.0
; With VCC[lane] = 1 the result is scaled by 2^+64 (f32) or 2^+128 (f64) when |S2| >= 2.0 and by 2^-64 / 2^-128
; otherwise; the scale-down pre-scales S0/S1 only when their product could overflow, so a tiny product like the two
; cases above underflows after the post-scale. VCC is set per case with a scalar move, only lane 0 is active.
; Every word below was measured on RDNA2 (gfx1036, wave32) through the DRM oracle; this fixture is the hardware
; check on the console's family.
; group: 32,1,1
; Run: vk_compute_test cs_w32_div_fmas.spv 1 dump -> out[0..7] = 0 1102053376 1101004800 0 0 0 0 1077280768
; mattr: +wavefrontsize32,-wavefrontsize64
v_cmpx_eq_u32_e32 0, v0
s_mov_b32 vcc_lo, 1
v_mov_b32_e32 v1, 0x1a000000
v_mov_b32_e32 v2, 0
v_div_fmas_f32 v10, v1, v1, v2
s_mov_b32 vcc_lo, 0
v_mov_b32_e32 v1, 0x40400000
v_mov_b32_e32 v2, 0x40a00000
v_mov_b32_e32 v3, 0x40e00000
v_div_fmas_f32 v11, v1, v2, v3
v_mov_b32_e32 v1, 0x40000000
v_mov_b32_e32 v2, 0x40400000
v_mov_b32_e32 v3, 0x40800000
v_div_fmas_f32 v12, v1, v2, v3 mul:2
v_mov_b32_e32 v1, 0xbf800000
v_mov_b32_e32 v2, 0x3f800000
v_mov_b32_e32 v3, 0xc0000000
v_div_fmas_f32 v13, v1, v2, v3 clamp
s_mov_b32 vcc_lo, 1
v_mov_b32_e32 v1, 0
v_mov_b32_e32 v2, 0x1e300000
v_mov_b32_e32 v3, 0
v_mov_b32_e32 v4, 0
v_div_fmas_f64 v[14:15], v[1:2], v[1:2], v[3:4]
s_mov_b32 vcc_lo, 0
v_mov_b32_e32 v1, 0
v_mov_b32_e32 v2, 0x40080000
v_mov_b32_e32 v3, 0
v_mov_b32_e32 v4, 0x40140000
v_mov_b32_e32 v5, 0
v_mov_b32_e32 v6, 0x401c0000
v_div_fmas_f64 v[16:17], v[1:2], v[3:4], v[5:6]
v_mov_b32_e32 v24, 0
buffer_store_dword v10, v24, s[0:3], 0 offen
v_mov_b32_e32 v24, 4
buffer_store_dword v11, v24, s[0:3], 0 offen
v_mov_b32_e32 v24, 8
buffer_store_dword v12, v24, s[0:3], 0 offen
v_mov_b32_e32 v24, 12
buffer_store_dword v13, v24, s[0:3], 0 offen
v_mov_b32_e32 v24, 16
buffer_store_dword v14, v24, s[0:3], 0 offen
v_mov_b32_e32 v24, 20
buffer_store_dword v15, v24, s[0:3], 0 offen
v_mov_b32_e32 v24, 24
buffer_store_dword v16, v24, s[0:3], 0 offen
v_mov_b32_e32 v24, 28
buffer_store_dword v17, v24, s[0:3], 0 offen
s_endpgm
