; GPU test for the 64-bit global_atomic_{or,and,xor}_x2 and buffer_atomic_xor_x2 (buffer or/and/swap_x2 are cs_buf_atomic_x2). M = op(M, DATA) on the
; 64-bit element (two dwords, low first), done by the translator as one 32-bit atomic per dword: the memory is exact for a bitwise op in any lane order.
; 32 lanes hit one element each; lane 0 seeds them, then reads each back with an atomic OR x2 of a zero pair (glc, so the read is coherent).
; Lane L supplies B = 1 << L, H = B & 0x55555555 and N = ~B | 0xF0000000.
;   out[0..1] global or_x2  seed (0, 0), DATA (B, H): (0xFFFFFFFF, 0x55555555)
;   out[2..3] global and_x2 seed (~0, ~0), DATA (N, ~H): N clears bit L for L < 28, ~H clears the odd-numbered dwords' 0x55555555 bits: (0xF0000000, 0xAAAAAAAA)
;   out[4..5] global xor_x2 seed (0x0F0F0F0F, 0xF0F0F0F0), DATA (B, H): (0x0F0F0F0F ^ 0xFFFFFFFF, 0xF0F0F0F0 ^ 0x55555555) = (0xF0F0F0F0, 0xA5A5A5A5)
;   out[6..7] buffer xor_x2 with the same seed and data: (0xF0F0F0F0, 0xA5A5A5A5)
; Run: vk_compute_test cs_w32_atomics_x2_ops.spv 1 dump
;   -> out[0..7] = 0xFFFFFFFF 0x55555555 0xF0000000 0xAAAAAAAA 0xF0F0F0F0 0xA5A5A5A5 0xF0F0F0F0 0xA5A5A5A5
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
v_mov_b32_e32 v44, 0x0
v_mov_b32_e32 v45, 0x0
buffer_store_dwordx2 v[44:45], off, s[0:3], 0 offset:0x80
v_mov_b32_e32 v44, 0xffffffff
v_mov_b32_e32 v45, 0xffffffff
buffer_store_dwordx2 v[44:45], off, s[0:3], 0 offset:0x88
v_mov_b32_e32 v44, 0xf0f0f0f
v_mov_b32_e32 v45, 0xf0f0f0f0
buffer_store_dwordx2 v[44:45], off, s[0:3], 0 offset:0x90
v_mov_b32_e32 v44, 0xf0f0f0f
v_mov_b32_e32 v45, 0xf0f0f0f0
buffer_store_dwordx2 v[44:45], off, s[0:3], 0 offset:0x98
s_waitcnt_vscnt null, 0x0
s_mov_b32 exec_lo, -1
s_barrier
global_atomic_or_x2 v[10:11], v[12:13], off offset:0x80
global_atomic_and_x2 v[10:11], v[14:15], off offset:0x88
global_atomic_xor_x2 v[10:11], v[12:13], off offset:0x90
buffer_atomic_xor_x2 v[12:13], off, s[0:3], 0 offset:0x98
s_waitcnt vmcnt(0)
s_waitcnt_vscnt null, 0x0
s_barrier
s_mov_b32 exec_lo, 1
v_mov_b32_e32 v20, 0
v_mov_b32_e32 v21, 0
global_atomic_or_x2 v[20:21], v[10:11], v[16:17], off offset:0x80 glc
v_mov_b32_e32 v22, 0
v_mov_b32_e32 v23, 0
global_atomic_or_x2 v[22:23], v[10:11], v[16:17], off offset:0x88 glc
v_mov_b32_e32 v24, 0
v_mov_b32_e32 v25, 0
global_atomic_or_x2 v[24:25], v[10:11], v[16:17], off offset:0x90 glc
v_mov_b32_e32 v26, 0
v_mov_b32_e32 v27, 0
buffer_atomic_or_x2 v[26:27], off, s[0:3], 0 offset:0x98 glc
s_waitcnt vmcnt(0)
v_mov_b32_e32 v40, 0
buffer_store_dword v20, v40, s[0:3], 0 offen
v_mov_b32_e32 v40, 4
buffer_store_dword v21, v40, s[0:3], 0 offen
v_mov_b32_e32 v40, 8
buffer_store_dword v22, v40, s[0:3], 0 offen
v_mov_b32_e32 v40, 12
buffer_store_dword v23, v40, s[0:3], 0 offen
v_mov_b32_e32 v40, 16
buffer_store_dword v24, v40, s[0:3], 0 offen
v_mov_b32_e32 v40, 20
buffer_store_dword v25, v40, s[0:3], 0 offen
v_mov_b32_e32 v40, 24
buffer_store_dword v26, v40, s[0:3], 0 offen
v_mov_b32_e32 v40, 28
buffer_store_dword v27, v40, s[0:3], 0 offen
s_endpgm
