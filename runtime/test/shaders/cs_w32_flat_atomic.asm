; flat_atomic_{add,swap,cmpswap,and,or,xor,smin,umax} (RDNA2 ISA, FLAT) on a 64-bit address in a VGPR pair: the global_atomic_*
; operations, 32-bit. With glc the old value goes to vdst; flat_atomic_add without glc has no vdst. cmpswap takes data as a pair
; {DATA, CMP}: M = (M == CMP) ? DATA : M. smin/umax discriminate the signed from the unsigned order.
; Seeds (MUBUF): 0x80 5, 0x84 0x1111, 0x88 0x4444, 0x8c 0x5555, 0x90 0xffff, 0x94 0xf0f0, 0x98 0xff, 0x9c 0xfffffffb,
; 0xa0 0xfffffff0, 0xa4 5; v[10:11] = 0x80. Each word is read back after the atomics (MUBUF).
;   out[0] add 7 (glc)            old 5 mem 0xc              = (old << 16) | mem          = 0x0005000c
;   out[1] swap 0x2222 (glc)      old 0x1111 mem 0x2222                                   = 0x11112222
;   out[2] cmpswap {0x3333, 0x4444} (glc), M = 0x4444: match   old 0x4444 mem 0x3333      = 0x44443333
;   out[3] cmpswap {0x3333, 0x4444} (glc), M = 0x5555: no match old 0x5555 mem 0x5555     = 0x55555555
;   out[4] and 0x0f0f on 0xffff -> 0x0f0f; or 0x0f0f on 0xf0f0 -> 0xffff       = (and << 16) | or = 0x0f0fffff
;   out[5] xor 0xff0 on 0xff -> 0xf0f; add 9 without glc on 5 -> 0xe          = (xor << 16) | add = 0x0f0f000e
;   out[6] smin 0x10 on 0xfffffffb (-5): stays -5 (umin would give 0x10)       = 0xfffffffb
;   out[7] umax 0x200 on 0xfffffff0: stays (smax would give 0x200)             = 0xfffffff0
; Measured on gfx1036 (Windows) as a raw kernel, wave32 and wave64: the old values and memory words above.
; group: 32,1,1
; Run: vk_compute_test cs_w32_flat_atomic.spv 1 dump -> out[0..7] = 327692 286335522 1145320243 1431655765 252706815 252641294 4294967291 4294967280
; mattr: +wavefrontsize32,-wavefrontsize64
v_cmpx_eq_u32_e32 0, v0
v_mov_b32_e32 v5, 0x80
v_mov_b32_e32 v1, 5
buffer_store_dword v1, v5, s[0:3], 0 offen
v_mov_b32_e32 v5, 0x84
v_mov_b32_e32 v1, 0x1111
buffer_store_dword v1, v5, s[0:3], 0 offen
v_mov_b32_e32 v5, 0x88
v_mov_b32_e32 v1, 0x4444
buffer_store_dword v1, v5, s[0:3], 0 offen
v_mov_b32_e32 v5, 0x8c
v_mov_b32_e32 v1, 0x5555
buffer_store_dword v1, v5, s[0:3], 0 offen
v_mov_b32_e32 v5, 0x90
v_mov_b32_e32 v1, 0xffff
buffer_store_dword v1, v5, s[0:3], 0 offen
v_mov_b32_e32 v5, 0x94
v_mov_b32_e32 v1, 0xf0f0
buffer_store_dword v1, v5, s[0:3], 0 offen
v_mov_b32_e32 v5, 0x98
v_mov_b32_e32 v1, 0xff
buffer_store_dword v1, v5, s[0:3], 0 offen
v_mov_b32_e32 v5, 0x9c
v_mov_b32_e32 v1, 0xfffffffb
buffer_store_dword v1, v5, s[0:3], 0 offen
v_mov_b32_e32 v5, 0xa0
v_mov_b32_e32 v1, 0xfffffff0
buffer_store_dword v1, v5, s[0:3], 0 offen
v_mov_b32_e32 v5, 0xa4
v_mov_b32_e32 v1, 5
buffer_store_dword v1, v5, s[0:3], 0 offen
s_waitcnt_vscnt null, 0x0
v_mov_b32_e32 v10, 0x80
v_mov_b32_e32 v11, 0
v_mov_b32_e32 v20, 7
flat_atomic_add v1, v[10:11], v20 glc
v_mov_b32_e32 v20, 0x2222
flat_atomic_swap v2, v[10:11], v20 offset:4 glc
v_mov_b32_e32 v20, 0x3333
v_mov_b32_e32 v21, 0x4444
flat_atomic_cmpswap v3, v[10:11], v[20:21] offset:8 glc
flat_atomic_cmpswap v4, v[10:11], v[20:21] offset:12 glc
v_mov_b32_e32 v22, 0xf0f
flat_atomic_and v5, v[10:11], v22 offset:16 glc
flat_atomic_or v6, v[10:11], v22 offset:20 glc
v_mov_b32_e32 v22, 0xff0
flat_atomic_xor v7, v[10:11], v22 offset:24 glc
v_mov_b32_e32 v22, 0x10
flat_atomic_smin v8, v[10:11], v22 offset:28 glc
v_mov_b32_e32 v22, 0x200
flat_atomic_umax v9, v[10:11], v22 offset:32 glc
v_mov_b32_e32 v22, 9
flat_atomic_add v[10:11], v22 offset:36
s_waitcnt vmcnt(0) lgkmcnt(0)
s_waitcnt_vscnt null, 0x0
v_mov_b32_e32 v5, 0x80
buffer_load_dword v30, v5, s[0:3], 0 offen
v_mov_b32_e32 v5, 0x84
buffer_load_dword v31, v5, s[0:3], 0 offen
v_mov_b32_e32 v5, 0x88
buffer_load_dword v32, v5, s[0:3], 0 offen
v_mov_b32_e32 v5, 0x8c
buffer_load_dword v33, v5, s[0:3], 0 offen
v_mov_b32_e32 v5, 0x90
buffer_load_dword v34, v5, s[0:3], 0 offen
v_mov_b32_e32 v5, 0x94
buffer_load_dword v35, v5, s[0:3], 0 offen
v_mov_b32_e32 v5, 0x98
buffer_load_dword v36, v5, s[0:3], 0 offen
v_mov_b32_e32 v5, 0x9c
buffer_load_dword v37, v5, s[0:3], 0 offen
v_mov_b32_e32 v5, 0xa0
buffer_load_dword v38, v5, s[0:3], 0 offen
v_mov_b32_e32 v5, 0xa4
buffer_load_dword v39, v5, s[0:3], 0 offen
s_waitcnt vmcnt(0)
v_lshl_or_b32 v50, v1, 16, v30
v_lshl_or_b32 v51, v2, 16, v31
v_lshl_or_b32 v52, v3, 16, v32
v_lshl_or_b32 v53, v4, 16, v33
v_lshl_or_b32 v54, v34, 16, v35
v_lshl_or_b32 v55, v36, 16, v39
v_mov_b32_e32 v40, 0
buffer_store_dword v50, v40, s[0:3], 0 offen
v_mov_b32_e32 v40, 4
buffer_store_dword v51, v40, s[0:3], 0 offen
v_mov_b32_e32 v40, 8
buffer_store_dword v52, v40, s[0:3], 0 offen
v_mov_b32_e32 v40, 12
buffer_store_dword v53, v40, s[0:3], 0 offen
v_mov_b32_e32 v40, 16
buffer_store_dword v54, v40, s[0:3], 0 offen
v_mov_b32_e32 v40, 20
buffer_store_dword v55, v40, s[0:3], 0 offen
v_mov_b32_e32 v40, 24
buffer_store_dword v37, v40, s[0:3], 0 offen
v_mov_b32_e32 v40, 28
buffer_store_dword v38, v40, s[0:3], 0 offen
s_endpgm
