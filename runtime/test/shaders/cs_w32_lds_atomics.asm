; regen_fixtures: this source has a translator fixture in tests/semantic/fixtures/disasm
; ds_inc/dec/rsub_u32 and ds_min/max_f32 (and their _rtn_ forms): the 32-bit LDS atomics (#322).
; Lane 0 of a wave32 group seeds five LDS dwords, applies the five _rtn_ forms, then reads three of the
; resulting dwords back. The other five ops are the same code path without a returned value.
;   LDS[0]  = 5         ds_inc_rtn_u32  DATA=6   -> M = (5 >= 6) ? 0 : 6 = 6,        out[0] = 5        (old)
;   LDS[4]  = 5         ds_dec_rtn_u32  DATA=4   -> M = (5 > 4) ? 4 : 4 = 4,         out[1] = 5        (old)
;   LDS[8]  = 7         ds_rsub_rtn_u32 DATA=10  -> M = 10 - 7 = 3,                  out[2] = 7        (old)
;   LDS[12] = 2.5       ds_min_rtn_f32  DATA=1.5 -> M = 1.5,                         out[3] = 0x40200000 (old 2.5)
;   LDS[16] = -0.0      ds_max_rtn_f32  DATA=+0  -> M = +0.0 (-0 < +0),               out[4] = 0x80000000 (old -0.0)
;   out[5] = LDS[0]  after the increment = 6
;   out[6] = LDS[12] after the min       = 0x3FC00000 (1.5)
;   out[7] = LDS[16] after the max       = 0x00000000 (+0.0)
; EXEC is cut to lane 0 before the LDS ops: lanes sharing an address would race, and the CAS loop would
; still leave one of the values, but which one is not the point of this fixture.
; group: 32,1,1
; Run: vk_compute_test cs_w32_lds_atomics.spv 1 dump -> out[0..7] = 5 5 7 1075838976 2147483648 6 1069547520 0
; mattr: +wavefrontsize32,-wavefrontsize64
v_cmpx_eq_u32_e32 0, v0
v_mov_b32_e32 v10, 0
v_mov_b32_e32 v11, 5
ds_write_b32 v10, v11
v_mov_b32_e32 v10, 4
ds_write_b32 v10, v11
v_mov_b32_e32 v10, 8
v_mov_b32_e32 v11, 7
ds_write_b32 v10, v11
v_mov_b32_e32 v10, 12
v_mov_b32_e32 v11, 2.5
ds_write_b32 v10, v11
v_mov_b32_e32 v10, 16
v_mov_b32_e32 v11, 0x80000000
ds_write_b32 v10, v11
s_waitcnt lgkmcnt(0)
v_mov_b32_e32 v10, 0
v_mov_b32_e32 v11, 6
ds_inc_rtn_u32 v1, v10, v11
v_mov_b32_e32 v10, 4
v_mov_b32_e32 v11, 4
ds_dec_rtn_u32 v2, v10, v11
v_mov_b32_e32 v10, 8
v_mov_b32_e32 v11, 10
ds_rsub_rtn_u32 v3, v10, v11
v_mov_b32_e32 v10, 12
v_mov_b32_e32 v11, 1.5
ds_min_rtn_f32 v4, v10, v11
v_mov_b32_e32 v10, 16
v_mov_b32_e32 v11, 0
ds_max_rtn_f32 v5, v10, v11
s_waitcnt lgkmcnt(0)
v_mov_b32_e32 v10, 0
ds_read_b32 v6, v10
v_mov_b32_e32 v10, 12
ds_read_b32 v7, v10
v_mov_b32_e32 v10, 16
ds_read_b32 v8, v10
s_waitcnt lgkmcnt(0)
v_mov_b32_e32 v24, 0
buffer_store_dword v1, v24, s[0:3], 0 offen
v_mov_b32_e32 v24, 4
buffer_store_dword v2, v24, s[0:3], 0 offen
v_mov_b32_e32 v24, 8
buffer_store_dword v3, v24, s[0:3], 0 offen
v_mov_b32_e32 v24, 12
buffer_store_dword v4, v24, s[0:3], 0 offen
v_mov_b32_e32 v24, 16
buffer_store_dword v5, v24, s[0:3], 0 offen
v_mov_b32_e32 v24, 20
buffer_store_dword v6, v24, s[0:3], 0 offen
v_mov_b32_e32 v24, 24
buffer_store_dword v7, v24, s[0:3], 0 offen
v_mov_b32_e32 v24, 28
buffer_store_dword v8, v24, s[0:3], 0 offen
s_endpgm
