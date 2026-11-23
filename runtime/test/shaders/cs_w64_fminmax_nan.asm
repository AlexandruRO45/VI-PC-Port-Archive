; regen_fixtures: this source has a translator fixture in tests/semantic/fixtures/disasm
; SHADER-985: v_min/max/min3/max3/med3_f32 with a quiet NaN source in a wave64 compute shader (compute runs with
; IEEE mode on). The min/max rule returns the other operand when one source is a quiet NaN, in wave32 and in wave64;
; a driver that returned the NaN in wave64 only (reported on AMD Windows 26.10.1, gfx1036) fails this test.
; v2 = qNaN (0x7fc00000), v3 = 1.0, v4 = -inf, v5 = +inf. Lane 5 stores eight results:
;   out[0] = min(qNaN, 1.0)
;   out[1] = max(1.0, qNaN)
;   out[2] = min3(qNaN, 1.0, 2.0)
;   out[3] = max3(qNaN, 1.0, 2.0)
;   out[4] = med3(qNaN, 1.0, 2.0)
;   out[5] = min(1.0, qNaN)
;   out[6] = max(qNaN, -inf)
;   out[7] = min(+inf, qNaN)
; Run: vk_compute_test cs_w64_fminmax_nan.spv 1 dump --require-wave64
;   -> out[0..7] = 1065353216 1065353216 1065353216 1073741824 1065353216 1065353216 4286578688 2139095040
;      (1.0 1.0 1.0 2.0 1.0 1.0 -inf +inf; measured on AMD gfx1036, Windows driver 32.0.21045.16010, native wave64,
;      and on an RTX 5080 with --wave64-emulate)
; mattr: +wavefrontsize64,-wavefrontsize32
v_mbcnt_lo_u32_b32 v1, -1, 0
v_mbcnt_hi_u32_b32 v1, -1, v1
v_mov_b32_e32 v2, 0x7fc00000
v_mov_b32_e32 v3, 1.0
v_mov_b32_e32 v4, 0xff800000
v_mov_b32_e32 v5, 0x7f800000
v_min_f32_e32 v10, v2, v3
v_max_f32_e32 v11, v3, v2
v_min3_f32 v12, v2, v3, 2.0
v_max3_f32 v13, v2, v3, 2.0
v_med3_f32 v14, v2, v3, 2.0
v_min_f32_e32 v15, v3, v2
v_max_f32_e32 v16, v2, v4
v_min_f32_e32 v17, v5, v2
s_mov_b64 exec, -1
v_cmpx_eq_u32_e32 5, v1
v_mov_b32_e32 v20, v10
v_mov_b32_e32 v21, 0
buffer_store_dword v20, v21, s[0:3], 0 offen
v_mov_b32_e32 v20, v11
v_mov_b32_e32 v21, 4
buffer_store_dword v20, v21, s[0:3], 0 offen
v_mov_b32_e32 v20, v12
v_mov_b32_e32 v21, 8
buffer_store_dword v20, v21, s[0:3], 0 offen
v_mov_b32_e32 v20, v13
v_mov_b32_e32 v21, 12
buffer_store_dword v20, v21, s[0:3], 0 offen
v_mov_b32_e32 v20, v14
v_mov_b32_e32 v21, 16
buffer_store_dword v20, v21, s[0:3], 0 offen
v_mov_b32_e32 v20, v15
v_mov_b32_e32 v21, 20
buffer_store_dword v20, v21, s[0:3], 0 offen
v_mov_b32_e32 v20, v16
v_mov_b32_e32 v21, 24
buffer_store_dword v20, v21, s[0:3], 0 offen
v_mov_b32_e32 v20, v17
v_mov_b32_e32 v21, 28
buffer_store_dword v20, v21, s[0:3], 0 offen
s_endpgm
