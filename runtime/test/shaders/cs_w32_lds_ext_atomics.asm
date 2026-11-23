; GPU test for 5 RDNA2 LDS atomic instructions:
;   ds_wrap_rtn_b32, ds_cmpst_f32, ds_cmpst_rtn_f32, ds_mskor_b64, ds_mskor_rtn_b64.
;
; LDS 4 KiB (RSRC2 0x40088), group 32x1x1, only lane 0 runs. Output words:
;
;   out[0]  ds_wrap_rtn_b32 wrap (old=10 >= d0=7 -> 10-7=3, offset:4): (old<<16)|new = 0x000A0003
;   out[1]  ds_wrap_rtn_b32 no-wrap (old=4 < d0=7 -> 4+5=9): (old<<16)|new = 0x00040009
;   out[2]  ds_cmpst_rtn_f32 (-0.0 == +0.0 -> 2.0): old(0x80000000) ^ new(0x40000000) = 0xC0000000
;   out[3]  ds_cmpst_rtn_f32 (qNaN 0x7FC01234 != qNaN -> unchanged): old & mem = 0x7FC01234
;   out[4]  ds_cmpst_f32 hit (1.5 -> -3.0 = 0xC0400000) ^ miss (qNaN 0x7FC00000): 0xBF800000
;   out[5]  ds_mskor_b64 (offset:8) new_lo(0x11AA33BB) ^ new_hi(0xCC66DD88): 0xDDCCEE33
;   out[6]  ds_mskor_rtn_b64 old_lo(0x12345678) ^ old_hi(0x9ABCDEF0): 0x88888888
;   out[7]  ds_mskor_rtn_b64 new_lo(0x1234BEEF) ^ new_hi(0xCAFEDEF0): 0xD8CA601F
;
; Run: vk_compute_test cs_w32_lds_ext_atomics.spv 1 dump
;   -> out[0..7] = 0x000A0003 0x00040009 0xC0000000 0x7FC01234
;                  0xBF800000 0xDDCCEE33 0x88888888 0xD8CA601F
; rsrc2: 0x40088
; group: 32,1,1
; mattr: +wavefrontsize32,-wavefrontsize64
v_cmpx_eq_u32_e32 0, v0

; --- out[0]: ds_wrap_rtn_b32 wrap case with offset:4 ---
v_mov_b32_e32 v1, 4
v_mov_b32_e32 v2, 10
ds_write_b32 v1, v2
s_waitcnt lgkmcnt(0)
v_mov_b32_e32 v1, 0
v_mov_b32_e32 v3, 7
v_mov_b32_e32 v4, 3
ds_wrap_rtn_b32 v5, v1, v3, v4 offset:4
s_waitcnt lgkmcnt(0)
v_mov_b32_e32 v1, 4
ds_read_b32 v6, v1
s_waitcnt lgkmcnt(0)
v_lshlrev_b32_e32 v5, 16, v5
v_or_b32_e32 v10, v5, v6
v_mov_b32_e32 v21, 0
buffer_store_dword v10, v21, s[0:3], 0 offen

; --- out[1]: ds_wrap_rtn_b32 no-wrap case ---
v_mov_b32_e32 v1, 8
v_mov_b32_e32 v2, 4
ds_write_b32 v1, v2
s_waitcnt lgkmcnt(0)
v_mov_b32_e32 v3, 7
v_mov_b32_e32 v4, 5
ds_wrap_rtn_b32 v5, v1, v3, v4
s_waitcnt lgkmcnt(0)
ds_read_b32 v6, v1
s_waitcnt lgkmcnt(0)
v_lshlrev_b32_e32 v5, 16, v5
v_or_b32_e32 v10, v5, v6
v_mov_b32_e32 v21, 4
buffer_store_dword v10, v21, s[0:3], 0 offen

; --- out[2]: ds_cmpst_rtn_f32 (-0.0 == +0.0 -> 2.0) ---
v_mov_b32_e32 v1, 0x10
v_mov_b32_e32 v2, 0x80000000
ds_write_b32 v1, v2
s_waitcnt lgkmcnt(0)
v_mov_b32_e32 v3, 0x00000000
v_mov_b32_e32 v4, 0x40000000
ds_cmpst_rtn_f32 v5, v1, v3, v4
s_waitcnt lgkmcnt(0)
ds_read_b32 v6, v1
s_waitcnt lgkmcnt(0)
v_xor_b32_e32 v10, v5, v6
v_mov_b32_e32 v21, 8
buffer_store_dword v10, v21, s[0:3], 0 offen

; --- out[3]: ds_cmpst_rtn_f32 (qNaN != qNaN -> unchanged) ---
v_mov_b32_e32 v1, 0x14
v_mov_b32_e32 v2, 0x7FC01234
ds_write_b32 v1, v2
s_waitcnt lgkmcnt(0)
v_mov_b32_e32 v3, 0x7FC01234
v_mov_b32_e32 v4, 0x3F800000
ds_cmpst_rtn_f32 v5, v1, v3, v4
s_waitcnt lgkmcnt(0)
ds_read_b32 v6, v1
s_waitcnt lgkmcnt(0)
v_and_b32_e32 v10, v5, v6
v_mov_b32_e32 v21, 12
buffer_store_dword v10, v21, s[0:3], 0 offen

; --- out[4]: ds_cmpst_f32 hit (0x20: 1.5 -> -3.0) ^ miss (0x24: qNaN != qNaN) ---
v_mov_b32_e32 v1, 0x20
v_mov_b32_e32 v2, 0x3FC00000
ds_write_b32 v1, v2
v_mov_b32_e32 v1, 0x24
v_mov_b32_e32 v2, 0x7FC00000
ds_write_b32 v1, v2
s_waitcnt lgkmcnt(0)
v_mov_b32_e32 v1, 0x20
v_mov_b32_e32 v3, 0x3FC00000
v_mov_b32_e32 v4, 0xC0400000
ds_cmpst_f32 v1, v3, v4
v_mov_b32_e32 v1, 0x24
v_mov_b32_e32 v3, 0x7FC00000
v_mov_b32_e32 v4, 0x3F800000
ds_cmpst_f32 v1, v3, v4
s_waitcnt lgkmcnt(0)
v_mov_b32_e32 v1, 0x20
ds_read_b32 v5, v1
v_mov_b32_e32 v1, 0x24
ds_read_b32 v6, v1
s_waitcnt lgkmcnt(0)
v_xor_b32_e32 v10, v5, v6
v_mov_b32_e32 v21, 16
buffer_store_dword v10, v21, s[0:3], 0 offen

; --- out[5]: ds_mskor_b64 with offset:8 ---
v_mov_b32_e32 v1, 0x48
v_mov_b32_e32 v2, 0x11223344
v_mov_b32_e32 v3, 0x55667788
ds_write_b64 v1, v[2:3]
s_waitcnt lgkmcnt(0)
v_mov_b32_e32 v1, 0x40
v_mov_b32_e32 v2, 0x00FF00FF
v_mov_b32_e32 v3, 0xFF00FF00
v_mov_b32_e32 v4, 0x00AA00BB
v_mov_b32_e32 v5, 0xCC00DD00
ds_mskor_b64 v1, v[2:3], v[4:5] offset:8
s_waitcnt lgkmcnt(0)
v_mov_b32_e32 v1, 0x48
ds_read_b64 v[6:7], v1
s_waitcnt lgkmcnt(0)
v_xor_b32_e32 v10, v6, v7
v_mov_b32_e32 v21, 20
buffer_store_dword v10, v21, s[0:3], 0 offen

; --- out[6..7]: ds_mskor_rtn_b64 (old XOR and new XOR) ---
v_mov_b32_e32 v1, 0x60
v_mov_b32_e32 v2, 0x12345678
v_mov_b32_e32 v3, 0x9ABCDEF0
ds_write_b64 v1, v[2:3]
s_waitcnt lgkmcnt(0)
v_mov_b32_e32 v2, 0x0000FFFF
v_mov_b32_e32 v3, 0xFFFF0000
v_mov_b32_e32 v4, 0x0000BEEF
v_mov_b32_e32 v5, 0xCAFE0000
ds_mskor_rtn_b64 v[8:9], v1, v[2:3], v[4:5]
s_waitcnt lgkmcnt(0)
ds_read_b64 v[6:7], v1
s_waitcnt lgkmcnt(0)
v_xor_b32_e32 v10, v8, v9
v_mov_b32_e32 v21, 24
buffer_store_dword v10, v21, s[0:3], 0 offen
v_xor_b32_e32 v11, v6, v7
v_mov_b32_e32 v21, 28
buffer_store_dword v11, v21, s[0:3], 0 offen
s_endpgm
