; global_atomic_fcmpswap (#3153, Gen.float_cmpswap_loop): vdata is {DATA, CMP}, the compare is a float compare (-0
; matches +0, a NaN never matches, not even its own bits), and the CAS loop stays exact when lanes contend.
; Global addresses are heap offsets (s[20:21] = P = 0x80, v30 = 0). Seeds (MUBUF stores through the V# in s[0:3]):
;   P+0 0x80000000 (-0)   P+4 0x7FC00000 (NaN)   P+8 0x3F800000 (1.0)   P+16 0x00000000 (+0)
; Lane 0 alone:
;   out[0] global_atomic_fcmpswap glc at P+0 {DATA 2.0, CMP +0}: old value                    = 0x80000000
;   out[1] the dword at P+0 after it: -0 == +0, so DATA is stored                            = 0x40000000
;   out[2] global_atomic_fcmpswap glc at P+4 {DATA 1.0, CMP the same NaN}: old value         = 0x7FC00000
;   out[3] the dword at P+4 after it: NaN != NaN, unchanged (a bit compare would store 1.0)  = 0x7FC00000
;   out[4] the dword at P+8 after global_atomic_fcmpswap (no glc) {DATA 3.0, CMP 0x3F800001} = 0x3F800000
; All 32 lanes at P+16 (+0), glc, lane i: DATA 0x3F800000 + i, CMP -0. One lane matches and stores its DATA; every
; later compare sees a nonzero DATA and fails, returning the stored word. Which lane wins is up to the hardware, so
; the counts are stored:
;   out[5] lanes whose old value is 0                                         = 1
;   out[6] lanes whose old value is the final word at P+16                    = 31
;   out[7] lanes whose old value is 0 and whose DATA is the final word        = 1
; group: 32,1,1
; Run: vk_compute_test cs_w32_global_fcmpswap.spv 1 dump -> out[0..7] = 2147483648 1073741824 2143289344 2143289344
; 1065353216 1 31 1
; mattr: +wavefrontsize32,-wavefrontsize64
s_mov_b32 s10, exec_lo
v_cmpx_eq_u32_e32 0, v0
v_mov_b32_e32 v1, 0x80000000
v_mov_b32_e32 v2, 0x80
buffer_store_dword v1, v2, s[0:3], 0 offen
v_mov_b32_e32 v1, 0x7fc00000
v_mov_b32_e32 v2, 0x84
buffer_store_dword v1, v2, s[0:3], 0 offen
v_mov_b32_e32 v1, 1.0
v_mov_b32_e32 v2, 0x88
buffer_store_dword v1, v2, s[0:3], 0 offen
v_mov_b32_e32 v1, 0
v_mov_b32_e32 v2, 0x90
buffer_store_dword v1, v2, s[0:3], 0 offen
s_waitcnt_vscnt null, 0
s_mov_b32 s20, 0x80
s_mov_b32 s21, 0
v_mov_b32_e32 v30, 0
v_mov_b32_e32 v2, 2.0
v_mov_b32_e32 v3, 0
global_atomic_fcmpswap v10, v30, v[2:3], s[20:21] glc
v_mov_b32_e32 v4, 1.0
v_mov_b32_e32 v5, 0x7fc00000
global_atomic_fcmpswap v11, v30, v[4:5], s[20:21] offset:4 glc
v_mov_b32_e32 v6, 0x40400000
v_mov_b32_e32 v7, 0x3f800001
global_atomic_fcmpswap v30, v[6:7], s[20:21] offset:8
s_waitcnt vmcnt(0)
global_load_dword v15, v30, s[20:21] glc
global_load_dword v16, v30, s[20:21] offset:4 glc
global_load_dword v17, v30, s[20:21] offset:8 glc
s_waitcnt vmcnt(0)
s_mov_b32 exec_lo, s10
v_mov_b32_e32 v30, 0
v_add_nc_u32_e32 v8, 0x3f800000, v0
v_mov_b32_e32 v9, 0x80000000
global_atomic_fcmpswap v12, v30, v[8:9], s[20:21] offset:16 glc
s_waitcnt vmcnt(0)
global_load_dword v13, v30, s[20:21] offset:16 glc
s_waitcnt vmcnt(0)
v_cmp_eq_u32_e64 s11, 0, v12
v_cmp_eq_u32_e64 s12, v12, v13
v_cmp_eq_u32_e64 s13, v8, v13
s_and_b32 s13, s11, s13
s_bcnt1_i32_b32 s11, s11
s_bcnt1_i32_b32 s12, s12
s_bcnt1_i32_b32 s13, s13
v_mov_b32_e32 v18, s11
v_mov_b32_e32 v19, s12
v_mov_b32_e32 v21, s13
v_cmpx_eq_u32_e32 0, v0
v_mov_b32_e32 v20, 0
buffer_store_dword v10, v20, s[0:3], 0 offen
v_mov_b32_e32 v20, 4
buffer_store_dword v15, v20, s[0:3], 0 offen
v_mov_b32_e32 v20, 8
buffer_store_dword v11, v20, s[0:3], 0 offen
v_mov_b32_e32 v20, 12
buffer_store_dword v16, v20, s[0:3], 0 offen
v_mov_b32_e32 v20, 16
buffer_store_dword v17, v20, s[0:3], 0 offen
v_mov_b32_e32 v20, 20
buffer_store_dword v18, v20, s[0:3], 0 offen
v_mov_b32_e32 v20, 24
buffer_store_dword v19, v20, s[0:3], 0 offen
v_mov_b32_e32 v20, 28
buffer_store_dword v21, v20, s[0:3], 0 offen
s_endpgm
