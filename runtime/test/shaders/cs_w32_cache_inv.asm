; Cache control: buffer_gl0_inv, buffer_gl1_inv, s_dcache_inv, s_gl1_inv, s_dcache_wb (RDNA2 ISA, MUBUF and SMEM). They
; invalidate or write back a data cache and change no register or memory word, so a program that runs them gives the
; same words as without them; the translator turns each into a device-scope memory barrier on buffer-class memory
; (docs/design.md). One lane; each case stores a value, runs the cache op, and reads it back (vector load for the
; vector caches, s_buffer_load_dword through the same V# for the scalar ones).
;   out[0] store 0x11 at 0x80, buffer_gl0_inv, buffer_load_dword                    = 0x11
;   out[1] store 0x22 at 0x84, buffer_gl1_inv, buffer_load_dword                    = 0x22
;   out[2] store 0x33 at 0x88, s_dcache_inv, s_buffer_load_dword                    = 0x33
;   out[3] store 0x44 at 0x8c, s_gl1_inv, s_buffer_load_dword                       = 0x44
;   out[4] store 0x55 at 0x90, s_dcache_wb, buffer_load_dword                        = 0x55
;   out[5] store 0x66 at 0x94, buffer_gl0_inv, load (0x66), store 0x77 there, buffer_gl1_inv, load = 0x77
;   out[6] store 0x88 at 0x98, all five ops back to back, buffer_load_dword          = 0x88
;   out[7] a VGPR set before the ops and read after them                            = 0x1234
; The words are what the program computes without the cache ops: the hardware ops are architecturally invisible. Measured
; on gfx1036 (wave32 and wave64, tools/hw_oracle): a store, the four ops that exist there, a load gives the stored word.
; s_dcache_wb is not an RDNA2 desktop instruction (llvm-mc accepts it for gfx1013, the console part, and rejects it for
; gfx1030/gfx1036), so it has no native run here.
; group: 32,1,1
; Run: vk_compute_test cs_w32_cache_inv.spv 1 dump -> out[0..7] = 17 34 51 68 85 119 136 4660
; mattr: +wavefrontsize32,-wavefrontsize64
v_cmpx_eq_u32_e32 0, v0
v_mov_b32_e32 v7, 0x1234
v_mov_b32_e32 v2, 0x80
v_mov_b32_e32 v1, 0x11
buffer_store_dword v1, v2, s[0:3], 0 offen
s_waitcnt_vscnt null, 0x0
buffer_gl0_inv
buffer_load_dword v10, v2, s[0:3], 0 offen
v_mov_b32_e32 v2, 0x84
v_mov_b32_e32 v1, 0x22
buffer_store_dword v1, v2, s[0:3], 0 offen
s_waitcnt_vscnt null, 0x0
buffer_gl1_inv
buffer_load_dword v11, v2, s[0:3], 0 offen
v_mov_b32_e32 v2, 0x88
v_mov_b32_e32 v1, 0x33
buffer_store_dword v1, v2, s[0:3], 0 offen
s_waitcnt_vscnt null, 0x0
s_dcache_inv
s_buffer_load_dword s8, s[0:3], 0x88
v_mov_b32_e32 v2, 0x8c
v_mov_b32_e32 v1, 0x44
buffer_store_dword v1, v2, s[0:3], 0 offen
s_waitcnt_vscnt null, 0x0
s_gl1_inv
s_buffer_load_dword s9, s[0:3], 0x8c
v_mov_b32_e32 v2, 0x90
v_mov_b32_e32 v1, 0x55
buffer_store_dword v1, v2, s[0:3], 0 offen
s_waitcnt_vscnt null, 0x0
s_dcache_wb
buffer_load_dword v14, v2, s[0:3], 0 offen
v_mov_b32_e32 v2, 0x94
v_mov_b32_e32 v1, 0x66
buffer_store_dword v1, v2, s[0:3], 0 offen
s_waitcnt_vscnt null, 0x0
buffer_gl0_inv
buffer_load_dword v15, v2, s[0:3], 0 offen
s_waitcnt vmcnt(0)
v_mov_b32_e32 v1, 0x77
buffer_store_dword v1, v2, s[0:3], 0 offen
s_waitcnt_vscnt null, 0x0
buffer_gl1_inv
buffer_load_dword v15, v2, s[0:3], 0 offen
v_mov_b32_e32 v2, 0x98
v_mov_b32_e32 v1, 0x88
buffer_store_dword v1, v2, s[0:3], 0 offen
s_waitcnt_vscnt null, 0x0
buffer_gl0_inv
buffer_gl1_inv
s_dcache_inv
s_gl1_inv
s_dcache_wb
buffer_load_dword v16, v2, s[0:3], 0 offen
s_waitcnt vmcnt(0) lgkmcnt(0)
v_mov_b32_e32 v12, s8
v_mov_b32_e32 v13, s9
v_mov_b32_e32 v25, 0
buffer_store_dword v10, v25, s[0:3], 0 offen
v_mov_b32_e32 v25, 4
buffer_store_dword v11, v25, s[0:3], 0 offen
v_mov_b32_e32 v25, 8
buffer_store_dword v12, v25, s[0:3], 0 offen
v_mov_b32_e32 v25, 12
buffer_store_dword v13, v25, s[0:3], 0 offen
v_mov_b32_e32 v25, 16
buffer_store_dword v14, v25, s[0:3], 0 offen
v_mov_b32_e32 v25, 20
buffer_store_dword v15, v25, s[0:3], 0 offen
v_mov_b32_e32 v25, 24
buffer_store_dword v16, v25, s[0:3], 0 offen
v_mov_b32_e32 v25, 28
buffer_store_dword v7, v25, s[0:3], 0 offen
s_endpgm
