; A 97-thread group is one full wave64 followed by 33 live lanes (threads 64..96). The last wave has only lane 32
; in its upper half. This combines the partial-wave case with multi-wave scratch and scalar-state isolation.
; v0 = thread id, v1 = lane id, v3 = 3 * thread id + 1. Comparing each lane with itself gives 64 or 33 live bits.
; Narrowing EXEC to lane >= 32 leaves 32 lanes in wave 0 and one in wave 1. readfirstlane(v3) then reads thread
; 32 (97) or thread 96 (289). After restoring the original EXEC, lower-half threads must see those same scalars.
; readlane(v3, 32) also gives 97 or 289; its source exists in both waves, so no absent-lane value is assumed.
; Stores: out[0..1] live counts = 64, 33; out[2..3] upper-only readfirstlane = 97, 289;
; out[4..5] upper live counts = 32, 1; out[6..7] readlane(32) = 97, 289.
; Run natively with VKC_SUBGROUP_SIZE=64 (without --require-wave64's full-subgroup requirement), or translate
; with --wave64-emulate and run at subgroup 32: vk_compute_test cs_w64x_partial_second.spv 1 dump
;   -> out[0..7] = 64 33 97 289 32 1 97 289
; mattr: +wavefrontsize64,-wavefrontsize32
; group: 97,1,1
v_mbcnt_lo_u32_b32 v1, -1, 0
v_mbcnt_hi_u32_b32 v1, -1, v1
v_mad_u32_u24 v3, v0, 3, 1
s_mov_b64 s[8:9], exec
v_cmp_eq_u32_e64 s[4:5], v1, v1
s_bcnt1_i32_b64 s6, s[4:5]
v_cmp_le_u32_e64 s[10:11], 32, v1
s_and_saveexec_b64 s[12:13], s[10:11]
s_bcnt1_i32_b64 s7, exec
v_readfirstlane_b32 s14, v3
s_mov_b64 exec, s[8:9]
v_readlane_b32 s15, v3, 32
v_cmpx_eq_u32_e32 0, v0
v_mov_b32_e32 v20, s6
v_mov_b32_e32 v21, 0
buffer_store_dword v20, v21, s[0:3], 0 offen
s_mov_b64 exec, s[8:9]
v_cmpx_eq_u32_e32 64, v0
v_mov_b32_e32 v20, s6
v_mov_b32_e32 v21, 4
buffer_store_dword v20, v21, s[0:3], 0 offen
s_mov_b64 exec, s[8:9]
v_cmpx_eq_u32_e32 0, v0
v_mov_b32_e32 v20, s14
v_mov_b32_e32 v21, 8
buffer_store_dword v20, v21, s[0:3], 0 offen
s_mov_b64 exec, s[8:9]
v_cmpx_eq_u32_e32 64, v0
v_mov_b32_e32 v20, s14
v_mov_b32_e32 v21, 12
buffer_store_dword v20, v21, s[0:3], 0 offen
s_mov_b64 exec, s[8:9]
v_cmpx_eq_u32_e32 32, v0
v_mov_b32_e32 v20, s7
v_mov_b32_e32 v21, 16
buffer_store_dword v20, v21, s[0:3], 0 offen
s_mov_b64 exec, s[8:9]
v_cmpx_eq_u32_e32 96, v0
v_mov_b32_e32 v20, s7
v_mov_b32_e32 v21, 20
buffer_store_dword v20, v21, s[0:3], 0 offen
s_mov_b64 exec, s[8:9]
v_cmpx_eq_u32_e32 32, v0
v_mov_b32_e32 v20, s15
v_mov_b32_e32 v21, 24
buffer_store_dword v20, v21, s[0:3], 0 offen
s_mov_b64 exec, s[8:9]
v_cmpx_eq_u32_e32 96, v0
v_mov_b32_e32 v20, s15
v_mov_b32_e32 v21, 28
buffer_store_dword v20, v21, s[0:3], 0 offen
s_endpgm
