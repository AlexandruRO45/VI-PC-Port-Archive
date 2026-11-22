; ds_wrap_rtn_b32 (LDS): tmp = M; M = (tmp >= DATA) ? tmp - DATA : tmp + DATA2 (unsigned, 32-bit wrap); vD = tmp.
; Measured natively on RDNA2 gfx1036 (tools/hw_oracle, 16 single-lane cases; the last six words here are three of
; them). Until this change it raised Unsupported.
; 32 lanes run ds_wrap_rtn_b32 on one word M = 0 with DATA = 10, DATA2 = 1: every lane applies the same map
; f(m) = m >= 10 ? m - 10 : m + 1, so in any order M ends at f^32(0) = 32 mod 11 = 10, and the 32 returned values are
; f^k(0) = k mod 11 for k = 0..31, whose sum (added to a second word with ds_add_u32) is 55 + 55 + 45 = 155.
; Then lane 0 alone (words from the gfx1036 table):
;   M = 4, DATA = 5, DATA2 = 0x64:                     ret 4,          M = 0x68
;   M = 0xfffffff0, DATA = 0x20, DATA2 = 0x30:         ret 0xfffffff0, M = 0xffffffd0
;   M = 0x80000000, DATA = 0x7fffffff, DATA2 = 1:      ret 0x80000000, M = 1
;   out[0..7] = 10 155 4 0x68 0xfffffff0 0xffffffd0 0x80000000 1
; rsrc2: 0x40088
; group: 32,1,1
; Run: vk_compute_test cs_w32_ds_wrap.spv 1 dump -> out[0..7] = 10 155 4 104 4294967280 4294967248 2147483648 1
; mattr: +wavefrontsize32,-wavefrontsize64
v_mov_b32_e32 v1, 0
v_mov_b32_e32 v2, 0
ds_write_b32 v1, v2
ds_write_b32 v1, v2 offset:4
s_waitcnt lgkmcnt(0)
s_barrier
v_mov_b32_e32 v3, 10
v_mov_b32_e32 v4, 1
ds_wrap_rtn_b32 v5, v1, v3, v4
s_waitcnt lgkmcnt(0)
ds_add_u32 v1, v5 offset:4
s_waitcnt lgkmcnt(0)
s_barrier
ds_read_b32 v10, v1
ds_read_b32 v11, v1 offset:4
s_waitcnt lgkmcnt(0)
s_barrier
v_cmpx_eq_u32_e32 0, v0
v_mov_b32_e32 v2, 4
ds_write_b32 v1, v2 offset:8
v_mov_b32_e32 v3, 5
v_mov_b32_e32 v4, 0x64
ds_wrap_rtn_b32 v12, v1, v3, v4 offset:8
ds_read_b32 v13, v1 offset:8
v_mov_b32_e32 v2, 0xfffffff0
ds_write_b32 v1, v2 offset:12
v_mov_b32_e32 v3, 0x20
v_mov_b32_e32 v4, 0x30
ds_wrap_rtn_b32 v14, v1, v3, v4 offset:12
ds_read_b32 v15, v1 offset:12
v_mov_b32_e32 v2, 0x80000000
ds_write_b32 v1, v2 offset:16
v_mov_b32_e32 v3, 0x7fffffff
v_mov_b32_e32 v4, 1
ds_wrap_rtn_b32 v16, v1, v3, v4 offset:16
ds_read_b32 v17, v1 offset:16
s_waitcnt lgkmcnt(0)
v_mov_b32_e32 v22, 0
buffer_store_dwordx4 v[10:13], v22, s[0:3], 0 offen
v_mov_b32_e32 v22, 16
buffer_store_dwordx4 v[14:17], v22, s[0:3], 0 offen
s_endpgm
