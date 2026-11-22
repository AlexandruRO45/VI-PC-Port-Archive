; Each lane stores three distinct values at dynamic byte addresses in the first
; two stripes of the loan and the top stripe of guest LDS (64 threads).
s_mov_b64 exec, -1
v_lshlrev_b32_e32 v1, 2, v0
v_add_nc_u32_e32 v2, 100, v0
ds_write_b32 v1, v2
v_add_nc_u32_e32 v3, 200, v0
ds_write_b32 v1, v3 offset:256
v_add_nc_u32_e32 v4, 300, v0
ds_write_b32 v1, v4 offset:65280
; First active is high-half lane 40. Use its value to select another lane.
v_cmp_ge_u32_e32 vcc, v0, 40
s_and_saveexec_b64 s[12:13], vcc
v_readfirstlane_b32 s8, v0
s_mov_b64 exec, s[12:13]
v_readlane_b32 s9, v2, s8
; Target lane (31 - lane%32) in this half. Two sequential exchange rounds.
v_xor_b32_e32 v5, 31, v0
v_lshlrev_b32_e32 v5, 2, v5
ds_permute_b32 v6, v5, v2
ds_read_b32 v7, v1
ds_read_b32 v8, v1 offset:256
ds_read_b32 v9, v1 offset:65280
s_waitcnt lgkmcnt(0)
v_add_nc_u32_e32 v6, s9, v6
v_add_nc_u32_e32 v6, v7, v6
v_add_nc_u32_e32 v6, v8, v6
v_add_nc_u32_e32 v6, v9, v6
buffer_store_dword v6, v1, s[0:3], 0 offen
s_endpgm
