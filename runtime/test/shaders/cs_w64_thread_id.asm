; Wave size (docs/design.md): a wave64 compute shader's thread id is wave_id * 64 + lane, so the translator maps
; lane = LocalInvocationIndex & 63 and wave = LocalInvocationIndex >> 6. 128 threads = two wave64s.
; v0 = LocalInvocationId.x (0..127); v_mbcnt_lo/hi(-1) give this invocation's lane id (a wave64 code marker),
; v0 >> 6 its wave id.
; s6 = s_bcnt1_i32_b64(64 * wave + lane == tid) = 64: every lane of the wave carries the mapping (a 32-lane
; wave gives 32). s10 = s_bcnt1_i32_b64(lane >= 40) = 24: lanes 40..63 exist only in a 64-lane wave (0 in a
; 32-lane one). s7 = v_readlane_b32(v0, 40) = the thread id of hardware lane 40: 40 in wave 0, 104 (64 + 40) in
; wave 1, and the read is undefined when the wave is only 32 lanes wide.
; Thread 0 stores out[0] = lane, out[1] = wave, out[2] = s6, out[3] = s10; thread 127 stores out[4] = lane,
; out[5] = wave, out[6] = s6, out[7] = s7.
; Run: vk_compute_test cs_w64_thread_id.spv 1 dump --require-wave64
;   -> out[0..7] = 0 0 64 24 63 1 64 104
; mattr: +wavefrontsize64,-wavefrontsize32
; group: 128,1,1
v_mbcnt_lo_u32_b32 v1, -1, 0
v_mbcnt_hi_u32_b32 v1, -1, v1
v_lshrrev_b32_e32 v2, 6, v0
v_lshl_add_u32 v3, v2, 6, v1
v_cmp_eq_u32_e64 s[8:9], v3, v0
s_bcnt1_i32_b64 s6, s[8:9]
v_cmp_ge_u32_e64 s[4:5], v1, 40
s_bcnt1_i32_b64 s10, s[4:5]
v_readlane_b32 s7, v0, 40
s_mov_b64 exec, -1
v_cmpx_eq_u32_e32 0, v0
v_mov_b32_e32 v10, 0
v_mov_b32_e32 v11, v1
buffer_store_dword v11, v10, s[0:3], 0 offen
v_mov_b32_e32 v10, 4
v_mov_b32_e32 v11, v2
buffer_store_dword v11, v10, s[0:3], 0 offen
v_mov_b32_e32 v10, 8
v_mov_b32_e32 v11, s6
buffer_store_dword v11, v10, s[0:3], 0 offen
v_mov_b32_e32 v10, 12
v_mov_b32_e32 v11, s10
buffer_store_dword v11, v10, s[0:3], 0 offen
s_mov_b64 exec, -1
v_cmpx_eq_u32_e32 127, v0
v_mov_b32_e32 v10, 16
v_mov_b32_e32 v11, v1
buffer_store_dword v11, v10, s[0:3], 0 offen
v_mov_b32_e32 v10, 20
v_mov_b32_e32 v11, v2
buffer_store_dword v11, v10, s[0:3], 0 offen
v_mov_b32_e32 v10, 24
v_mov_b32_e32 v11, s6
buffer_store_dword v11, v10, s[0:3], 0 offen
v_mov_b32_e32 v10, 28
v_mov_b32_e32 v11, s7
buffer_store_dword v11, v10, s[0:3], 0 offen
s_endpgm
