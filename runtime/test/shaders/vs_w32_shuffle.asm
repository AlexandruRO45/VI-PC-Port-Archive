; Wave32 vertex shader on a 64-wide vertex subgroup (w32_vs_halves.py): reads from other lanes of the wave.
; Record A at heap 0x1000 + 16 * v, record B at 0x5000 + 16 * v (v0 = vertex index):
;   A = (lane, readlane5, key, v)   readlane5 = v_readlane_b32 of v0 at lane 5, key = v_readfirstlane of v0
;   B = (xor1, next, readlane31, 0) xor1 = ds_bpermute_b32 of v0 from lane ^ 1, next = from (lane + 1) & 31 (each 0
;                                   when that lane is not in the wave), readlane31 = v_readlane_b32 of v0 at lane 31
; Checked per wave (the vertices with one key): each value is the v of the member on that lane.
; Run: as vs_w32_lane.asm.
; mattr: +wavefrontsize32,-wavefrontsize64
v_mbcnt_lo_u32_b32 v6, -1, 0
v_readlane_b32 s6, v0, 5
v_mov_b32_e32 v7, s6
v_readfirstlane_b32 s5, v0
v_mov_b32_e32 v8, s5
v_mov_b32_e32 v9, v0
v_lshlrev_b32_e32 v12, 4, v0
v_add_nc_u32_e32 v13, 0x1000, v12
buffer_store_dwordx4 v[6:9], v13, s[0:3], 0 offen
v_xor_b32_e32 v14, 1, v6
v_lshlrev_b32_e32 v14, 2, v14
v_add_nc_u32_e32 v15, 1, v6
v_and_b32_e32 v15, 31, v15
v_lshlrev_b32_e32 v15, 2, v15
ds_bpermute_b32 v16, v14, v0
ds_bpermute_b32 v17, v15, v0
s_waitcnt lgkmcnt(0)
v_readlane_b32 s7, v0, 31
v_mov_b32_e32 v6, v16
v_mov_b32_e32 v7, v17
v_mov_b32_e32 v8, s7
v_mov_b32_e32 v9, 0
v_add_nc_u32_e32 v13, 0x5000, v12
buffer_store_dwordx4 v[6:9], v13, s[0:3], 0 offen
v_mov_b32_e32 v5, 0
exp pos0 v5, v5, v5, v5 done
exp param0 v5, v5, v5, v5
s_endpgm
