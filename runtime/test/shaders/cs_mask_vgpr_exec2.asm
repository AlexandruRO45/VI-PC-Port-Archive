; group: 32,1,1
; cs_mask_vgpr_exec with a mask that differs between lanes: vcc_lo = lanes 0-4 (0x1f), copied with v_mov_b32 while
; EXEC is lanes 0-3, then stored by every lane. Hardware: lanes 0-3 get the wave's bits 0x1f, lanes 4-31 keep
; 0xdeadbeef. Taking each lane's own bit gives 0xffffffff in lanes 0-3; gathering at the store instead of at the
; copy would also read lanes 4-31's old 0xdeadbeef as set bits (#210, #2290).
v_mbcnt_lo_u32_b32 v1, -1, 0
v_mov_b32_e32 v2, 0xdeadbeef
v_cmp_gt_u32_e64 vcc_lo, 5, v1
s_mov_b32 exec_lo, 0x0000000f
v_mov_b32_e32 v2, vcc_lo
s_mov_b32 exec_lo, -1
v_lshlrev_b32_e32 v3, 2, v1
buffer_store_dword v2, v3, s[0:3], 0 offen
s_endpgm
