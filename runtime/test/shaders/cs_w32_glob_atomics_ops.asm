; GPU test for the 32-bit global_atomic_{add,sub,smin,umin,smax,umax,and,or} with 32 lanes on ONE dword each. RDNA2 ISA
; pseudocode (MUBUF/FLAT atomics): tmp = MEM[A]; MEM[A] = op(tmp, DATA); glc returns tmp. add/sub wrap mod 2^32; smin/smax compare signed,
; umin/umax unsigned. The result of these ops does not depend on the order the lanes arrive in, so the final word is exact.
; Lane 0 seeds the dwords at heap byte 0x80 + 4k, all 32 lanes run one atomic each on its dword, then lane 0 reads each dword
; with an atomic OR of 0 (glc, so the read is coherent) and stores it. Lane L supplies
;   D = 0x7FFFFFF0 + L (lanes 0..15 = 0x7FFFFFF0..0x7FFFFFFF, lanes 16..31 = 0x80000000..0x8000000F, straddling the signed boundary)
;   N = ~(1 << L) | 0xF0000000 (and),  B = 1 << L (or).
;   out[0] add  seed 0x10: 0x10 + sum(D) = 0x10 + (32 * 0x7FFFFFF0 + 496) mod 2^32            = 0x00000000
;   out[1] sub  seed 0x5:  0x5 - sum(D) mod 2^32                                               = 0x00000015
;   out[2] smin seed 0x7FFFFFFF: smallest signed D is lane 16 = 0x80000000                      = 0x80000000
;   out[3] umin seed 0xFFFFFFFF: smallest unsigned D is lane 0                                   = 0x7FFFFFF0
;   out[4] smax seed 0x80000000: largest signed D is lane 15                                     = 0x7FFFFFFF
;   out[5] umax seed 0: largest unsigned D is lane 31                                            = 0x8000000F
;   out[6] and  seed 0xFFFFFFFF: N clears bit L for L < 28 (lanes 28..31 give all ones)           = 0xF0000000
;   out[7] or   seed 0: every bit L set by lane L                                                = 0xFFFFFFFF
; Run: vk_compute_test cs_w32_glob_atomics_ops.spv 1 dump
;   -> out[0..7] = 0x00000000 0x00000015 0x80000000 0x7FFFFFF0 0x7FFFFFFF 0x8000000F 0xF0000000 0xFFFFFFFF
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
v_mov_b32_e32 v5, 0x10
buffer_store_dword v5, off, s[0:3], 0 offset:0x80
v_mov_b32_e32 v5, 0x5
buffer_store_dword v5, off, s[0:3], 0 offset:0x84
v_mov_b32_e32 v5, 0x7fffffff
buffer_store_dword v5, off, s[0:3], 0 offset:0x88
v_mov_b32_e32 v5, 0xffffffff
buffer_store_dword v5, off, s[0:3], 0 offset:0x8c
v_mov_b32_e32 v5, 0x80000000
buffer_store_dword v5, off, s[0:3], 0 offset:0x90
v_mov_b32_e32 v5, 0x0
buffer_store_dword v5, off, s[0:3], 0 offset:0x94
v_mov_b32_e32 v5, 0xffffffff
buffer_store_dword v5, off, s[0:3], 0 offset:0x98
v_mov_b32_e32 v5, 0x0
buffer_store_dword v5, off, s[0:3], 0 offset:0x9c
s_waitcnt_vscnt null, 0x0
s_mov_b32 exec_lo, -1
s_barrier
global_atomic_add v[10:11], v2, off offset:0x80
global_atomic_sub v[10:11], v2, off offset:0x84
global_atomic_smin v[10:11], v2, off offset:0x88
global_atomic_umin v[10:11], v2, off offset:0x8c
global_atomic_smax v[10:11], v2, off offset:0x90
global_atomic_umax v[10:11], v2, off offset:0x94
global_atomic_and v[10:11], v6, off offset:0x98
global_atomic_or v[10:11], v3, off offset:0x9c
s_waitcnt vmcnt(0)
s_waitcnt_vscnt null, 0x0
s_barrier
s_mov_b32 exec_lo, 1
v_mov_b32_e32 v20, 0
global_atomic_or v20, v[10:11], v9, off offset:0x80 glc
v_mov_b32_e32 v21, 0
global_atomic_or v21, v[10:11], v9, off offset:0x84 glc
v_mov_b32_e32 v22, 0
global_atomic_or v22, v[10:11], v9, off offset:0x88 glc
v_mov_b32_e32 v23, 0
global_atomic_or v23, v[10:11], v9, off offset:0x8c glc
v_mov_b32_e32 v24, 0
global_atomic_or v24, v[10:11], v9, off offset:0x90 glc
v_mov_b32_e32 v25, 0
global_atomic_or v25, v[10:11], v9, off offset:0x94 glc
v_mov_b32_e32 v26, 0
global_atomic_or v26, v[10:11], v9, off offset:0x98 glc
v_mov_b32_e32 v27, 0
global_atomic_or v27, v[10:11], v9, off offset:0x9c glc
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
