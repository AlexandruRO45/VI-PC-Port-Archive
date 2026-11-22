; regen_fixtures: this source has a translator fixture in tests/semantic/fixtures/disasm
; v_cvt_flr_i32_f32 and v_cvt_rpi_i32_f32: floor and round-half-up to i32, with saturation (#366).
; Lane 0 of a wave32 group converts eight inputs and stores one dword each:
;   out[0] = v_cvt_flr_i32_f32(2.75)                 = 2           (floor)
;   out[1] = v_cvt_flr_i32_f32(-2.25)                = 0xFFFFFFFD  (-3)
;   out[2] = v_cvt_flr_i32_f32(0x4F000000 = 2^31)    = 0x7FFFFFFF  (out of i32 range: saturates by the sign)
;   out[3] = v_cvt_flr_i32_f32(0x80000001 = -1.4e-45) = 0          (denormal flushed to zero, not floor(-1e-45) = -1)
;   out[4] = v_cvt_rpi_i32_f32(2.5)                  = 3           (floor(x + 0.5): the tie goes up)
;   out[5] = v_cvt_rpi_i32_f32(-2.5)                 = 0xFFFFFFFE  (-2, so not round-half-away-from-zero)
;   out[6] = v_cvt_rpi_i32_f32(0x3EFFFFFF = just below 0.5) = 0   (the tie needs frac >= 0.5 exactly)
;   out[7] = v_cvt_rpi_i32_f32(-0.5)                 = 0           (floor(0.0))
; group: 32,1,1
; Run: vk_compute_test cs_w32_cvt_flr_rpi.spv 1 dump -> out[0..7] = 2 4294967293 2147483647 0 3 4294967294 0 0
; mattr: +wavefrontsize32,-wavefrontsize64
v_mov_b32_e32 v1, 2.75
v_cvt_flr_i32_f32 v3, v1
v_mov_b32_e32 v1, -2.25
v_cvt_flr_i32_f32 v4, v1
v_mov_b32_e32 v1, 0x4f000000
v_cvt_flr_i32_f32 v5, v1
v_mov_b32_e32 v1, 0x80000001
v_cvt_flr_i32_f32 v6, v1
v_mov_b32_e32 v1, 2.5
v_cvt_rpi_i32_f32 v7, v1
v_mov_b32_e32 v1, -2.5
v_cvt_rpi_i32_f32 v8, v1
v_mov_b32_e32 v1, 0x3effffff
v_cvt_rpi_i32_f32 v9, v1
v_mov_b32_e32 v1, -0.5
v_cvt_rpi_i32_f32 v10, v1
v_cmpx_eq_u32_e32 0, v0
v_mov_b32_e32 v24, 0
buffer_store_dword v3, v24, s[0:3], 0 offen
v_mov_b32_e32 v24, 4
buffer_store_dword v4, v24, s[0:3], 0 offen
v_mov_b32_e32 v24, 8
buffer_store_dword v5, v24, s[0:3], 0 offen
v_mov_b32_e32 v24, 12
buffer_store_dword v6, v24, s[0:3], 0 offen
v_mov_b32_e32 v24, 16
buffer_store_dword v7, v24, s[0:3], 0 offen
v_mov_b32_e32 v24, 20
buffer_store_dword v8, v24, s[0:3], 0 offen
v_mov_b32_e32 v24, 24
buffer_store_dword v9, v24, s[0:3], 0 offen
v_mov_b32_e32 v24, 28
buffer_store_dword v10, v24, s[0:3], 0 offen
s_endpgm
