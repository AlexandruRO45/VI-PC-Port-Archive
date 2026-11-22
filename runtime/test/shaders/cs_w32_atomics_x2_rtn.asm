; GPU test for the single-lane glc forms of global_atomic_swap_x2, global_atomic_and_x2 and buffer_atomic_xor_x2 (64-bit, glc returns the old pair).
; RDNA2 ISA pseudocode: tmp = MEM[A]; MEM[A] = op(tmp, DATA); glc returns tmp in vdst (buffer: in vdata). Memory is read back with an atomic OR x2 of 0 (glc).
;   out[0..1] swap_x2 seed (0x11111111, 0x22222222), DATA (0x33333333, 0x44444444): returns the seed
;   out[2..3] swap_x2 memory afterwards: (0x33333333, 0x44444444)
;   out[4..5] and_x2 seed (0xFF00FF00, 0x0FF00FF0), DATA (0x0F0F0F0F, 0xFFFF0000): memory (0x0F000F00, 0x0FF00000); return ^ memory per dword:
;             (0xFF00FF00 ^ 0x0F000F00, 0x0FF00FF0 ^ 0x0FF00000) = (0xF000F000, 0x00000FF0)  (the return is the seed, so a wrong return or memory shows here)
;   out[6]    xor_x2 seed (0x12345678, 0x9ABCDEF0), DATA (0xFFFF0000, 0x0000FFFF): the returned pair XORed together = 0x12345678 ^ 0x9ABCDEF0 = 0x88888888
;   out[7]    the memory pair XORed together = (0x12345678 ^ 0xFFFF0000) ^ (0x9ABCDEF0 ^ 0x0000FFFF) = 0xEDCB5678 ^ 0x9ABC210F = 0x77777777
; Run: vk_compute_test cs_w32_atomics_x2_rtn.spv 1 dump
;   -> out[0..7] = 0x11111111 0x22222222 0x33333333 0x44444444 0xF000F000 0x00000FF0 0x88888888 0x77777777
; group: 32,1,1
; mattr: +wavefrontsize32,-wavefrontsize64
v_mov_b32_e32 v9, 0
v_mov_b32_e32 v1, 0
v_mov_b32_e32 v10, 0
v_mov_b32_e32 v11, 0
v_add_nc_u32_e32 v2, 0x7ffffff0, v0
v_mov_b32_e32 v4, 1
v_lshlrev_b32_e32 v3, v0, v4
v_not_b32_e32 v6, v3
v_or_b32_e32 v6, 0xf0000000, v6
v_and_b32_e32 v7, 0x55555555, v3
v_not_b32_e32 v8, v7
v_mov_b32_e32 v12, v3
v_mov_b32_e32 v13, v7
v_mov_b32_e32 v14, v6
v_mov_b32_e32 v15, v8
v_mov_b32_e32 v16, 0
v_mov_b32_e32 v17, 0
s_mov_b32 exec_lo, 1
v_mov_b32_e32 v44, 0x11111111
v_mov_b32_e32 v45, 0x22222222
buffer_store_dwordx2 v[44:45], off, s[0:3], 0 offset:0x80
v_mov_b32_e32 v44, 0xff00ff00
v_mov_b32_e32 v45, 0xff00ff0
buffer_store_dwordx2 v[44:45], off, s[0:3], 0 offset:0x88
v_mov_b32_e32 v44, 0x12345678
v_mov_b32_e32 v45, 0x9abcdef0
buffer_store_dwordx2 v[44:45], off, s[0:3], 0 offset:0x90
s_waitcnt_vscnt null, 0x0
v_mov_b32_e32 v30, 0x33333333
v_mov_b32_e32 v31, 0x44444444
v_mov_b32_e32 v32, 0x0f0f0f0f
v_mov_b32_e32 v33, 0xffff0000
v_mov_b32_e32 v34, 0xffff0000
v_mov_b32_e32 v35, 0x0000ffff
global_atomic_swap_x2 v[36:37], v[10:11], v[30:31], off offset:0x80 glc
global_atomic_and_x2 v[38:39], v[10:11], v[32:33], off offset:0x88 glc
buffer_atomic_xor_x2 v[34:35], off, s[0:3], 0 offset:0x90 glc
s_waitcnt vmcnt(0)
s_waitcnt_vscnt null, 0x0
v_mov_b32_e32 v20, 0
v_mov_b32_e32 v21, 0
global_atomic_or_x2 v[20:21], v[10:11], v[16:17], off offset:0x80 glc
v_mov_b32_e32 v22, 0
v_mov_b32_e32 v23, 0
global_atomic_or_x2 v[22:23], v[10:11], v[16:17], off offset:0x88 glc
v_mov_b32_e32 v24, 0
v_mov_b32_e32 v25, 0
buffer_atomic_or_x2 v[24:25], off, s[0:3], 0 offset:0x90 glc
s_waitcnt vmcnt(0)
v_xor_b32_e32 v26, v38, v22
v_xor_b32_e32 v27, v39, v23
v_xor_b32_e32 v28, v34, v35
v_xor_b32_e32 v29, v24, v25
v_mov_b32_e32 v40, 0
buffer_store_dword v36, v40, s[0:3], 0 offen
v_mov_b32_e32 v40, 4
buffer_store_dword v37, v40, s[0:3], 0 offen
v_mov_b32_e32 v40, 8
buffer_store_dword v20, v40, s[0:3], 0 offen
v_mov_b32_e32 v40, 12
buffer_store_dword v21, v40, s[0:3], 0 offen
v_mov_b32_e32 v40, 16
buffer_store_dword v26, v40, s[0:3], 0 offen
v_mov_b32_e32 v40, 20
buffer_store_dword v27, v40, s[0:3], 0 offen
v_mov_b32_e32 v40, 24
buffer_store_dword v28, v40, s[0:3], 0 offen
v_mov_b32_e32 v40, 28
buffer_store_dword v29, v40, s[0:3], 0 offen
s_endpgm
