; GPU test for the 32-bit LDS atomic min/max ops ds_max_u32, ds_min_i32, ds_max_i32, ds_min_u32 and their _rtn_ forms,
; which no GPU shader on main names (CPU-checked only). RDNA2 ISA pseudocode (DS atomics, 32-bit):
;   ds_max_u32  tmp = MEM[A]; MEM[A] = (DATA > tmp) ? DATA : tmp     (unsigned compare)
;   ds_min_i32  tmp = MEM[A]; MEM[A] = (DATA < tmp) ? DATA : tmp     (signed compare)
;   ds_max_i32  as ds_max_u32 with a signed compare;  ds_min_u32  as ds_min_i32 with an unsigned compare
;   the _rtn_ form also returns tmp (the old value) in vdst.
; LDS is 4 KiB (RSRC2 0x40088). Lane 0 seeds the words, then all 32 lanes run each op on the SAME address with lane L's
; data 0x7FFFFFF0 + L (lanes 0..15 are 0x7FFFFFF0..0x7FFFFFFF, lanes 16..31 are 0x80000000..0x8000000F, so the values
; straddle the signed boundary). Min and max do not depend on the order the lanes arrive in, so the final word is exact.
;   out[0]  ds_max_u32 at 0x00, seed 0:           unsigned max is lane 31 = 0x8000000F (a signed max gives 0x7FFFFFFF)
;   out[1]  ds_min_i32 at 0x04, seed 0x7FFFFFFF:  signed min is lane 16 = 0x80000000 (an unsigned min gives 0x7FFFFFF0)
;   out[2]  ds_max_i32 at 0x08, seed 0x80000000:  signed max is lane 15 = 0x7FFFFFFF (an unsigned max gives 0x8000000F)
;   out[3]  ds_min_u32 at 0x0C, seed 0xFFFFFFFF:  unsigned min is lane 0 = 0x7FFFFFF0 (a signed min gives 0x80000000)
; Then lane 0 alone (EXEC = 1) runs the _rtn_ forms; out[4..7] = returned old value XOR the word left in memory, so both
; the return and the store are checked, and a wrong compare signedness gives 0 or another word:
;   out[4]  ds_max_rtn_u32 at 0x40, seed 0x70000000, data 0x80000000: returns 0x70000000, stores 0x80000000 -> 0xF0000000
;   out[5]  ds_min_rtn_i32 at 0x44, seed 0x00000001, data 0xFFFFFFFF: returns 1, stores 0xFFFFFFFF            -> 0xFFFFFFFE
;   out[6]  ds_min_rtn_u32 at 0x48, seed 0x90000000, data 0x7FFFFFFF: returns 0x90000000, stores 0x7FFFFFFF  -> 0xEFFFFFFF
;   out[7]  ds_max_rtn_i32 at 0x4C, seed 0xFFFFFF00, data 0x00000001: returns 0xFFFFFF00, stores 1            -> 0xFFFFFF01
; Run: vk_compute_test cs_w32_ds_minmax.spv 1 dump
;   -> out[0..7] = 0x8000000F 0x80000000 0x7FFFFFFF 0x7FFFFFF0 0xF0000000 0xFFFFFFFE 0xEFFFFFFF 0xFFFFFF01
; rsrc2: 0x40088
; group: 32,1,1
; mattr: +wavefrontsize32,-wavefrontsize64
v_mov_b32_e32 v1, 0
v_add_nc_u32_e32 v2, 0x7ffffff0, v0
s_mov_b32 exec_lo, 1
v_mov_b32_e32 v4, 0
v_mov_b32_e32 v5, 0x7fffffff
v_mov_b32_e32 v6, 0x80000000
v_mov_b32_e32 v7, 0xffffffff
v_mov_b32_e32 v8, 0x70000000
v_mov_b32_e32 v9, 1
v_mov_b32_e32 v10, 0x90000000
v_mov_b32_e32 v11, 0xffffff00
ds_write_b32 v1, v4 offset:0x00
ds_write_b32 v1, v5 offset:0x04
ds_write_b32 v1, v6 offset:0x08
ds_write_b32 v1, v7 offset:0x0c
ds_write_b32 v1, v8 offset:0x40
ds_write_b32 v1, v9 offset:0x44
ds_write_b32 v1, v10 offset:0x48
ds_write_b32 v1, v11 offset:0x4c
s_waitcnt lgkmcnt(0)
s_mov_b32 exec_lo, -1
s_barrier
ds_max_u32 v1, v2 offset:0x00
ds_min_i32 v1, v2 offset:0x04
ds_max_i32 v1, v2 offset:0x08
ds_min_u32 v1, v2 offset:0x0c
s_waitcnt lgkmcnt(0)
s_barrier
s_mov_b32 exec_lo, 1
v_mov_b32_e32 v12, 0x80000000
v_mov_b32_e32 v13, 0xffffffff
v_mov_b32_e32 v14, 0x7fffffff
v_mov_b32_e32 v15, 1
ds_max_rtn_u32 v20, v1, v12 offset:0x40
ds_min_rtn_i32 v21, v1, v13 offset:0x44
ds_min_rtn_u32 v22, v1, v14 offset:0x48
ds_max_rtn_i32 v23, v1, v15 offset:0x4c
s_waitcnt lgkmcnt(0)
ds_read_b32 v24, v1 offset:0x00
ds_read_b32 v25, v1 offset:0x04
ds_read_b32 v26, v1 offset:0x08
ds_read_b32 v27, v1 offset:0x0c
ds_read_b32 v28, v1 offset:0x40
ds_read_b32 v29, v1 offset:0x44
ds_read_b32 v30, v1 offset:0x48
ds_read_b32 v31, v1 offset:0x4c
s_waitcnt lgkmcnt(0)
v_xor_b32_e32 v28, v28, v20
v_xor_b32_e32 v29, v29, v21
v_xor_b32_e32 v30, v30, v22
v_xor_b32_e32 v31, v31, v23
v_mov_b32_e32 v40, 0
buffer_store_dword v24, v40, s[0:3], 0 offen
v_mov_b32_e32 v40, 4
buffer_store_dword v25, v40, s[0:3], 0 offen
v_mov_b32_e32 v40, 8
buffer_store_dword v26, v40, s[0:3], 0 offen
v_mov_b32_e32 v40, 12
buffer_store_dword v27, v40, s[0:3], 0 offen
v_mov_b32_e32 v40, 16
buffer_store_dword v28, v40, s[0:3], 0 offen
v_mov_b32_e32 v40, 20
buffer_store_dword v29, v40, s[0:3], 0 offen
v_mov_b32_e32 v40, 24
buffer_store_dword v30, v40, s[0:3], 0 offen
v_mov_b32_e32 v40, 28
buffer_store_dword v31, v40, s[0:3], 0 offen
s_endpgm
