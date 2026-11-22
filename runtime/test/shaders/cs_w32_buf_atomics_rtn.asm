; GPU test for buffer_atomic_xor (32-lane contention) and the single-lane glc forms of buffer_atomic_swap, _cmpswap (hit and miss),
; _add and _sub. RDNA2 ISA pseudocode: tmp = MEM[A]; MEM[A] = op(tmp, DATA); glc returns tmp. cmpswap takes the pair {DATA, CMP}:
; MEM[A] = (tmp == CMP) ? DATA : tmp. Words are read back through an atomic OR of 0 with glc.
;   out[0] xor: 32 lanes, lane L xors 1 << L into seed 0x0F0F0F0F (xor of all bits = 0xFFFFFFFF)  0x0F0F0F0F ^ 0xFFFFFFFF = 0xF0F0F0F0
; Then lane 0 alone:
;   out[1] swap  seed 0x11111111, DATA 0x22222222: returns 0x11111111
;   out[2] swap  memory afterwards: 0x22222222
;   out[3] cmpswap hit  seed 0x33333333, DATA 0x44444444, CMP 0x33333333: returns 0x33333333
;   out[4] cmpswap hit  memory afterwards: 0x44444444
;   out[5] cmpswap miss seed 0x55555555, DATA 0x66666666, CMP 0x77777777: returns 0x55555555
;   out[6] cmpswap miss memory (0x55555555, unchanged) ^ add glc return: add seed 1, DATA 0xFFFFFFFF returns 1 (memory wraps to 0): 0x55555555 ^ 1 = 0x55555554
;   out[7] add memory (0) ^ sub glc return ^ sub memory: sub seed 0x80000000, DATA 1 returns 0x80000000, memory 0x7FFFFFFF: 0 ^ 0x80000000 ^ 0x7FFFFFFF = 0xFFFFFFFF
; Run: vk_compute_test cs_w32_buf_atomics_rtn.spv 1 dump
;   -> out[0..7] = 0xF0F0F0F0 0x11111111 0x22222222 0x33333333 0x44444444 0x55555555 0x55555554 0xFFFFFFFF
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
s_mov_b32 exec_lo, 1
v_mov_b32_e32 v5, 0xf0f0f0f
buffer_store_dword v5, off, s[0:3], 0 offset:0x80
s_waitcnt_vscnt null, 0x0
s_mov_b32 exec_lo, -1
s_barrier
buffer_atomic_xor v3, off, s[0:3], 0 offset:0x80
s_waitcnt vmcnt(0)
s_waitcnt_vscnt null, 0x0
s_barrier
s_mov_b32 exec_lo, 1
s_mov_b32 exec_lo, -1
s_mov_b32 exec_lo, 1
v_mov_b32_e32 v5, 0x11111111
buffer_store_dword v5, off, s[0:3], 0 offset:0x90
v_mov_b32_e32 v5, 0x33333333
buffer_store_dword v5, off, s[0:3], 0 offset:0x94
v_mov_b32_e32 v5, 0x55555555
buffer_store_dword v5, off, s[0:3], 0 offset:0x98
v_mov_b32_e32 v5, 0x1
buffer_store_dword v5, off, s[0:3], 0 offset:0x9c
v_mov_b32_e32 v5, 0x80000000
buffer_store_dword v5, off, s[0:3], 0 offset:0xa0
s_waitcnt_vscnt null, 0x0
v_mov_b32_e32 v12, 0x22222222
v_mov_b32_e32 v14, 0x44444444
v_mov_b32_e32 v15, 0x33333333
v_mov_b32_e32 v16, 0x66666666
v_mov_b32_e32 v17, 0x77777777
v_mov_b32_e32 v18, 0xffffffff
v_mov_b32_e32 v19, 1
buffer_atomic_swap v12, off, s[0:3], 0 offset:0x90 glc
buffer_atomic_cmpswap v[14:15], off, s[0:3], 0 offset:0x94 glc
buffer_atomic_cmpswap v[16:17], off, s[0:3], 0 offset:0x98 glc
buffer_atomic_add v18, off, s[0:3], 0 offset:0x9c glc
buffer_atomic_sub v19, off, s[0:3], 0 offset:0xa0 glc
s_waitcnt vmcnt(0)
s_waitcnt_vscnt null, 0x0
v_mov_b32_e32 v30, 0
buffer_atomic_or v30, off, s[0:3], 0 offset:0x90 glc
v_mov_b32_e32 v31, 0
buffer_atomic_or v31, off, s[0:3], 0 offset:0x94 glc
v_mov_b32_e32 v32, 0
buffer_atomic_or v32, off, s[0:3], 0 offset:0x98 glc
v_mov_b32_e32 v33, 0
buffer_atomic_or v33, off, s[0:3], 0 offset:0x9c glc
v_mov_b32_e32 v34, 0
buffer_atomic_or v34, off, s[0:3], 0 offset:0xa0 glc
s_waitcnt vmcnt(0)
v_xor_b32_e32 v36, v32, v18
v_xor_b32_e32 v37, v33, v19
v_xor_b32_e32 v37, v37, v34
v_mov_b32_e32 v38, 0
buffer_atomic_or v38, off, s[0:3], 0 offset:0x80 glc
s_waitcnt vmcnt(0)
v_mov_b32_e32 v40, 0
buffer_store_dword v38, v40, s[0:3], 0 offen
v_mov_b32_e32 v40, 4
buffer_store_dword v12, v40, s[0:3], 0 offen
v_mov_b32_e32 v40, 8
buffer_store_dword v30, v40, s[0:3], 0 offen
v_mov_b32_e32 v40, 12
buffer_store_dword v14, v40, s[0:3], 0 offen
v_mov_b32_e32 v40, 16
buffer_store_dword v31, v40, s[0:3], 0 offen
v_mov_b32_e32 v40, 20
buffer_store_dword v16, v40, s[0:3], 0 offen
v_mov_b32_e32 v40, 24
buffer_store_dword v36, v40, s[0:3], 0 offen
v_mov_b32_e32 v40, 28
buffer_store_dword v37, v40, s[0:3], 0 offen
s_endpgm
