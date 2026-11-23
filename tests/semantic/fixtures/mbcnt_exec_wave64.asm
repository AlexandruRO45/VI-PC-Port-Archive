; Wave64 compute fixture (#6214): v_mbcnt_lo_u32_b32 and v_mbcnt_hi_u32_b32 reading 64-bit SGPR / VCC masks and
; 32-bit half masks (vcc_hi, s7, vcc_lo) while EXEC has lanes 0..15 and 32..47 turned off.
; group: 64,1,1
; mattr: +wavefrontsize64,-wavefrontsize32
v_mbcnt_lo_u32_b32 v0, -1, 0
v_mbcnt_hi_u32_b32 v0, -1, v0
v_add_nc_u32_e32 v10, 1000, v0
v_add_nc_u32_e32 v11, 1000, v0
v_add_nc_u32_e32 v12, 1000, v0
v_add_nc_u32_e32 v13, 1000, v0
v_add_nc_u32_e32 v14, 1000, v0
v_add_nc_u32_e32 v15, 1000, v0
v_cmp_gt_u32_e64 s[4:5], 44, v0
v_and_b32_e32 v1, 1, v0
v_cmp_eq_u32_e64 vcc, 0, v1
s_mov_b32 s7, s5
s_mov_b32 exec_lo, 0xffff0000
s_mov_b32 exec_hi, 0xffff0000
v_mbcnt_lo_u32_b32 v10, s[4:5], 100
v_mbcnt_hi_u32_b32 v11, s[4:5], 200
v_mbcnt_lo_u32_b32 v12, vcc, 300
v_mbcnt_hi_u32_b32 v12, vcc, v12
v_mbcnt_lo_u32_b32 v13, vcc_hi, 400
v_mbcnt_hi_u32_b32 v14, s7, 500
v_mbcnt_hi_u32_b32 v15, vcc_lo, 600
s_mov_b64 exec, -1
v_readlane_b32 s8, v10, 5
v_readlane_b32 s9, v10, 20
v_readlane_b32 s10, v10, 50
v_readlane_b32 s11, v11, 35
v_readlane_b32 s12, v11, 20
v_readlane_b32 s13, v11, 50
v_readlane_b32 s14, v12, 53
v_readlane_b32 s15, v13, 21
v_readlane_b32 s16, v14, 50
v_readlane_b32 s17, v15, 53
v_cmpx_eq_u32_e32 0, v0
v_mov_b32_e32 v20, s8
v_mov_b32_e32 v21, 0
buffer_store_dword v20, v21, s[0:3], 0 offen
v_mov_b32_e32 v20, s9
v_mov_b32_e32 v21, 4
buffer_store_dword v20, v21, s[0:3], 0 offen
v_mov_b32_e32 v20, s10
v_mov_b32_e32 v21, 8
buffer_store_dword v20, v21, s[0:3], 0 offen
v_mov_b32_e32 v20, s11
v_mov_b32_e32 v21, 12
buffer_store_dword v20, v21, s[0:3], 0 offen
v_mov_b32_e32 v20, s12
v_mov_b32_e32 v21, 16
buffer_store_dword v20, v21, s[0:3], 0 offen
v_mov_b32_e32 v20, s13
v_mov_b32_e32 v21, 20
buffer_store_dword v20, v21, s[0:3], 0 offen
v_mov_b32_e32 v20, s14
v_mov_b32_e32 v21, 24
buffer_store_dword v20, v21, s[0:3], 0 offen
v_mov_b32_e32 v20, s15
v_mov_b32_e32 v21, 28
buffer_store_dword v20, v21, s[0:3], 0 offen
v_mov_b32_e32 v20, s16
v_mov_b32_e32 v21, 32
buffer_store_dword v20, v21, s[0:3], 0 offen
v_mov_b32_e32 v20, s17
v_mov_b32_e32 v21, 36
buffer_store_dword v20, v21, s[0:3], 0 offen
s_endpgm
