; Two lanes replace the low two bits of the same LDS word with 1 and 2.
; Repeat on a second word with the non-returning instruction. Both start at B=0xabcd1200.
; out[group*4..+3] = old lane 0, old lane 1, final returning, final non-returning.
; Legal first three words: (B, B+1, B+2) or (B+2, B, B+1).
; The independent fourth word must be B+1 or B+2. The lane order can differ on each word.
; Run: vk_compute_test cs_lds_mskor_contended.spv 64 mskor
; group: 2,1,1
; mattr: +wavefrontsize32,-wavefrontsize64
v_lshlrev_b32_e32 v1, 2, v0
v_mov_b32_e32 v2, 0xabcd1200
ds_write_b32 v1, v2
s_waitcnt lgkmcnt(0)
s_barrier

v_mov_b32_e32 v2, 0
v_mov_b32_e32 v3, 3
v_add_nc_u32_e32 v4, 1, v0
ds_mskor_rtn_b32 v5, v2, v3, v4
ds_mskor_b32 v2, v3, v4 offset:4
s_waitcnt lgkmcnt(0)
s_barrier

ds_read_b32 v6, v1
s_waitcnt lgkmcnt(0)
v_lshl_add_u32 v7, s4, 2, v0
v_lshlrev_b32_e32 v7, 2, v7
buffer_store_dword v5, v7, s[0:3], 0 offen
buffer_store_dword v6, v7, s[0:3], 0 offen offset:8
s_endpgm
