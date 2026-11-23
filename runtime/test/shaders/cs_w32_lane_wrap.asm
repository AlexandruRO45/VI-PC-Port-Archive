; SHADER-52: wave32 v_readlane_b32 / v_writelane_b32 mask the lane select with 31 (S1[4:0]), not 63.
; v0 = LocalInvocationId.x (0..31). v2 = 3 * lane + 1.
; readlane(v2, 40) must equal lane (40 & 31) = 8 -> 3*8+1 = 25. With & 63 the shuffle asks for lane 40
; (outside a 32-wide subgroup) and fails.
; writelane 777 into lane 40 lands on lane 8; readlane of 8 and of 40 both return 777.
; TEST-1044 / SHADER-540: a saved odd-lane predicate must survive a read through the other selector, and an ordinary
; write of zero through that alias must clear it. Count the lanes with the expected value for each case.
; group: 32,1,1
; Run: VKC_SUBGROUP_SIZE=32 vk_compute_test cs_w32_lane_wrap.spv 1 dump -> out[0..5] = 25 16 777 777 32 32
; mattr: +wavefrontsize32,-wavefrontsize64
v_mad_u32_u24 v2, v0, 3, 1
v_readlane_b32 s6, v2, 40
v_readlane_b32 s7, v2, 5
v_mov_b32_e32 v10, 0
s_mov_b32 s8, 777
v_writelane_b32 v10, s8, 40
v_readlane_b32 s9, v10, 8
v_readlane_b32 s10, v10, 40

v_and_b32_e32 v12, 1, v0
v_cmp_ne_u32_e64 s12, 0, v12
v_mov_b32_e32 v13, 0
v_writelane_b32 v13, s12, 40
v_readlane_b32 s13, v13, 8
v_cndmask_b32_e64 v14, 0, 1, s13
v_cmp_eq_u32_e32 vcc_lo, v12, v14
s_bcnt1_i32_b32 s14, vcc_lo

v_writelane_b32 v13, s12, 8
v_writelane_b32 v13, 0, 40
v_readlane_b32 s13, v13, 8
v_cndmask_b32_e64 v14, 0, 1, s13
v_cmp_eq_u32_e32 vcc_lo, 0, v14
s_bcnt1_i32_b32 s15, vcc_lo

v_cmpx_eq_u32_e32 0, v0
v_mov_b32_e32 v20, s6
v_mov_b32_e32 v21, 0
buffer_store_dword v20, v21, s[0:3], 0 offen
v_mov_b32_e32 v20, s7
v_mov_b32_e32 v21, 4
buffer_store_dword v20, v21, s[0:3], 0 offen
v_mov_b32_e32 v20, s9
v_mov_b32_e32 v21, 8
buffer_store_dword v20, v21, s[0:3], 0 offen
v_mov_b32_e32 v20, s10
v_mov_b32_e32 v21, 12
buffer_store_dword v20, v21, s[0:3], 0 offen
v_mov_b32_e32 v20, s14
v_mov_b32_e32 v21, 16
buffer_store_dword v20, v21, s[0:3], 0 offen
v_mov_b32_e32 v20, s15
v_mov_b32_e32 v21, 20
buffer_store_dword v20, v21, s[0:3], 0 offen
s_endpgm
