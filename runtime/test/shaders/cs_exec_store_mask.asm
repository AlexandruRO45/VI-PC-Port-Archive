; EXEC: buffer_store_dword under partial EXEC (0x55555555, every other lane) and scalar write under
; EXEC=0. If the translator gates buffer_store with EXEC, inactive lanes leave their slot at 0;
; otherwise they overwrite every slot with 0xDEADBEEF. If scalar writes are EXEC-masked, s_mov_b32
; under EXEC=0 silently fails and out[0] stays 0xDEADBEEF instead of 0xCAFE.
; group: 32,1,1
; Run: vk_compute_test cs_exec_store_mask.spv 1 dump -> even lanes 0xDEADBEEF (out[0] overwritten),
; odd lanes 0, out[0]=0xCAFE
; mattr: +wavefrontsize32,-wavefrontsize64
s_mov_b32 exec_lo, 0x55555555
v_mov_b32_e32 v1, 0xDEADBEEF
v_lshlrev_b32_e32 v3, 2, v0
buffer_store_dword v1, v3, s[0:3], 0 offen
s_mov_b32 exec_lo, 0
s_mov_b32 s4, 0xCAFE
s_mov_b32 exec_lo, -1
v_cmpx_eq_u32_e32 0, v0
v_mov_b32_e32 v1, s4
v_mov_b32_e32 v3, 0
buffer_store_dword v1, v3, s[0:3], 0 offen
s_endpgm
