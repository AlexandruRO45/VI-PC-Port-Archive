; group: 32,1,1
; mattr: +wavefrontsize32,-wavefrontsize64
; EGEN-BUF-ATOMIC64: 64-bit compare/carry buffer and global atomics (smin/umin/smax/umax/inc/dec/cmpswap _x2), one
; real 64-bit atomic each (Int64Atomics, shaderBufferInt64Atomics). Measured natively on RDNA2 (gfx1036,
; tools/hw_oracle raw kernels, wave32 and wave64: this exact program gives these words, and 3,234 single-lane cases
; match the ISA rules below). Each element is two dwords, low first; the heap and the V# in s[0:3] start at 0, so a
; global address is the byte offset (s[20:21] = 0). RDNA2 ISA, unsigned unless noted:
;   smin/smax: signed 64-bit min/max; umin/umax: unsigned; inc: M = M >= D ? 0 : M + 1; dec: M = (M == 0 || M > D) ? D
;   : M - 1; cmpswap: vdata = {D, CMP}, M = M == CMP ? D : M. With glc the old 64 bits are returned (global: vdst;
;   buffer: vdata[0:1]).
; All 32 lanes seed the elements (same words), then:
;   0x80 global_atomic_umax_x2 from every lane, D = lane << 32 | 3*lane, M = 0: lane 31 wins -> 0x1f_0000005d
;   0x88 buffer_atomic_inc_x2 from every lane, D = ~0, M = 0x1_fffffff0: 32 increments carry into the high dword
;        -> 0x2_00000010
;   0x90 global_atomic_smin_x2 from every lane, D = -lane (sign-extended), M = 0: -31 -> 0xffffffff_ffffffe1
; Then lane 0 alone, with glc (old values stored from 0xc0):
;   0x98 global cmpswap, M = 0x12345678_9abcdef0, CMP = M, D = 0xaaaabbbb_ccccdddd -> D; old M at 0xc0
;   0xa0 buffer cmpswap, M = 0x1_00000000, CMP = 0 (low dwords equal, high not), D = 0x5_00000005 -> M kept;
;        old at 0xc8
;   0xa8 buffer dec, M = 0, D = 7 -> wraps to D = 7; old 0 at 0xd0
;   0xb0 global umin, M = 0x1_00000000, D = 0xffffffff -> 0xffffffff; old at 0xd8
;   0xb8 buffer smax, M = 0x80000000_00000000 (most negative), D = 1 -> 1; old at 0xe0
; out[0..31] and out[58..63] stay 0xeeeeeeee; out[32..57] (check.py ATOMIC64_X2_WORDS) =
;   5d 1f | 10 2 | ffffffe1 ffffffff | ccccdddd aaaabbbb | 0 1 | 7 0 | ffffffff 0 | 1 0 |
;   9abcdef0 12345678 | 0 1 | 0 0 | 0 1 | 0 80000000
; Run: vk_compute_test cs_atomic64_x2.spv 1 dump with VKC_OUT_WORDS=64 and VKC_DUMP (check.py atomic64_x2_test)
s_mov_b32 s20, 0
s_mov_b32 s21, 0
v_mov_b32 v2, 0
v_mov_b32 v3, 0
buffer_store_dwordx2 v[2:3], off, s[0:3], 0 offset:128
buffer_store_dwordx2 v[2:3], off, s[0:3], 0 offset:144
buffer_store_dwordx2 v[2:3], off, s[0:3], 0 offset:168
v_mov_b32 v2, 0xfffffff0
v_mov_b32 v3, 1
buffer_store_dwordx2 v[2:3], off, s[0:3], 0 offset:136
v_mov_b32 v2, 0x9abcdef0
v_mov_b32 v3, 0x12345678
buffer_store_dwordx2 v[2:3], off, s[0:3], 0 offset:152
v_mov_b32 v2, 0
v_mov_b32 v3, 1
buffer_store_dwordx2 v[2:3], off, s[0:3], 0 offset:160
buffer_store_dwordx2 v[2:3], off, s[0:3], 0 offset:176
v_mov_b32 v3, 0x80000000
buffer_store_dwordx2 v[2:3], off, s[0:3], 0 offset:184
s_waitcnt vmcnt(0)
v_mul_u32_u24_e32 v4, 3, v0
v_mov_b32 v5, v0
v_mov_b32 v10, 0x80
global_atomic_umax_x2 v10, v[4:5], s[20:21]
v_mov_b32 v4, -1
v_mov_b32 v5, -1
v_mov_b32 v6, 0x88
buffer_atomic_inc_x2 v[4:5], v6, s[0:3], 0 offen
v_sub_nc_u32_e32 v4, 0, v0
v_ashrrev_i32_e32 v5, 31, v4
v_mov_b32 v10, 0x90
global_atomic_smin_x2 v10, v[4:5], s[20:21]
s_waitcnt vmcnt(0)
v_cmpx_eq_u32_e32 0, v0
v_mov_b32 v4, 0xccccdddd
v_mov_b32 v5, 0xaaaabbbb
v_mov_b32 v6, 0x9abcdef0
v_mov_b32 v7, 0x12345678
v_mov_b32 v10, 0x98
global_atomic_cmpswap_x2 v[12:13], v10, v[4:7], s[20:21] glc
v_mov_b32 v4, 5
v_mov_b32 v5, 5
v_mov_b32 v6, 0
v_mov_b32 v7, 0
v_mov_b32 v8, 0xa0
buffer_atomic_cmpswap_x2 v[4:7], v8, s[0:3], 0 offen glc
v_mov_b32 v14, 7
v_mov_b32 v15, 0
v_mov_b32 v8, 0xa8
buffer_atomic_dec_x2 v[14:15], v8, s[0:3], 0 offen glc
v_mov_b32 v18, -1
v_mov_b32 v19, 0
v_mov_b32 v10, 0xb0
global_atomic_umin_x2 v[16:17], v10, v[18:19], s[20:21] glc
v_mov_b32 v20, 1
v_mov_b32 v21, 0
v_mov_b32 v8, 0xb8
buffer_atomic_smax_x2 v[20:21], v8, s[0:3], 0 offen glc
s_waitcnt vmcnt(0)
buffer_store_dwordx2 v[12:13], off, s[0:3], 0 offset:192
buffer_store_dwordx2 v[4:5], off, s[0:3], 0 offset:200
buffer_store_dwordx2 v[14:15], off, s[0:3], 0 offset:208
buffer_store_dwordx2 v[16:17], off, s[0:3], 0 offset:216
buffer_store_dwordx2 v[20:21], off, s[0:3], 0 offset:224
s_waitcnt vmcnt(0)
s_endpgm
