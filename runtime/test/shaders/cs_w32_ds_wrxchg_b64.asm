; TEST-175: GPU check for 64-bit LDS atomic exchanges (ds_wrxchg_rtn_b64,
; ds_wrxchg2_rtn_b64, ds_wrxchg2st64_rtn_b64) and ds_wrxchg2st64_rtn_b32,
; covering offset:, vdst aliasing data, 8/256/512-byte strides, and
; same-address ordering.
; Fold: mix(lo, hi) = (lo ^ (3 * hi)) & 0xffffffff
;       mix2(a, b, c, d) = mix(mix(a, b), mix(c, d))
; Lane 0 computes eight output words:
;   out[0] = mix(old_lo, old_hi) from ds_wrxchg_rtn_b64 v[4:5], v1, v[4:5] offset:0x18              = 0x811155DC
;   out[1] = mix(new_lo, new_hi) read back from M[0x38]                                             = 0x006646CF
;   out[2] = mix2(old0, old1) from ds_wrxchg2_rtn_b64 (offset0:1 -> 0x48, offset1:3 -> 0x58)        = 0xDA8AFA7D
;   out[3] = mix2(new0, new1) read back from M[0x48] and M[0x58]                                    = 0x4F5B06C8
;   out[4] = mix2(old0, old1) from ds_wrxchg2st64_rtn_b64 (offset0:1 -> 0x220, offset1:2 -> 0x420)  = 0x0F839BB8
;   out[5] = mix2(new0, new1) read back from M[0x220] and M[0x420]                                  = 0xD411551A
;   out[6] = mix2(old0, old1, new0, new1) from ds_wrxchg2st64_rtn_b32 (0x140, 0x340)                = 0xE27FF99F
;   out[7] = mix(mix2(old0, old1), mix(new0, new1)) from ds_wrxchg2_rtn_b64 same addr 0x90, aliased = 0x1113777E
; group: 32,1,1
; rsrc2: 0x20088
; Run: vk_compute_test cs_w32_ds_wrxchg_b64.spv 1 dump -> out[0..7] = 2165396956 6702799 3666541181 1331365576 260283320 3557905690 3800037791 286488446
; mattr: +wavefrontsize32,-wavefrontsize64
v_mov_b32_e32 v4, 0xbad00001
v_mov_b32_e32 v5, 0xbad00002
v_cmpx_eq_u32_e32 0, v0
v_mov_b32_e32 v10, 0x38
v_mov_b32_e32 v2, 0x81223344
v_mov_b32_e32 v3, 0x55667788
ds_write_b64 v10, v[2:3]
s_waitcnt lgkmcnt(0)
v_mov_b32_e32 v1, 0x20
v_mov_b32_e32 v4, 0x99aabbcc
v_mov_b32_e32 v5, 0xddeeff01
ds_wrxchg_rtn_b64 v[4:5], v1, v[4:5] offset:0x18
s_waitcnt lgkmcnt(0)
ds_read_b64 v[6:7], v10
s_waitcnt lgkmcnt(0)
v_add_nc_u32_e32 v28, v5, v5
v_add_nc_u32_e32 v28, v28, v5
v_xor_b32_e32 v26, v4, v28
v_mov_b32_e32 v30, 0
buffer_store_dword v26, v30, s[0:3], 0 offen
v_add_nc_u32_e32 v28, v7, v7
v_add_nc_u32_e32 v28, v28, v7
v_xor_b32_e32 v26, v6, v28
v_mov_b32_e32 v30, 4
buffer_store_dword v26, v30, s[0:3], 0 offen
v_mov_b32_e32 v10, 0x48
v_mov_b32_e32 v2, 0x10203040
v_mov_b32_e32 v3, 0x90a0b0c0
ds_write_b64 v10, v[2:3]
v_mov_b32_e32 v11, 0x58
v_mov_b32_e32 v2, 0x13579bdf
v_mov_b32_e32 v3, 0x2468ace0
ds_write_b64 v11, v[2:3]
s_waitcnt lgkmcnt(0)
v_mov_b32_e32 v1, 0x40
v_mov_b32_e32 v12, 0xcafebabe
v_mov_b32_e32 v13, 0xdeadbeef
v_mov_b32_e32 v14, 0xfeedface
v_mov_b32_e32 v15, 0x8badf00d
ds_wrxchg2_rtn_b64 v[16:19], v1, v[12:13], v[14:15] offset0:1 offset1:3
s_waitcnt lgkmcnt(0)
ds_read_b64 v[20:21], v10
ds_read_b64 v[22:23], v11
s_waitcnt lgkmcnt(0)
v_add_nc_u32_e32 v28, v17, v17
v_add_nc_u32_e32 v28, v28, v17
v_xor_b32_e32 v24, v16, v28
v_add_nc_u32_e32 v28, v19, v19
v_add_nc_u32_e32 v28, v28, v19
v_xor_b32_e32 v25, v18, v28
v_add_nc_u32_e32 v28, v25, v25
v_add_nc_u32_e32 v28, v28, v25
v_xor_b32_e32 v26, v24, v28
v_mov_b32_e32 v30, 8
buffer_store_dword v26, v30, s[0:3], 0 offen
v_add_nc_u32_e32 v28, v21, v21
v_add_nc_u32_e32 v28, v28, v21
v_xor_b32_e32 v24, v20, v28
v_add_nc_u32_e32 v28, v23, v23
v_add_nc_u32_e32 v28, v28, v23
v_xor_b32_e32 v25, v22, v28
v_add_nc_u32_e32 v28, v25, v25
v_add_nc_u32_e32 v28, v28, v25
v_xor_b32_e32 v26, v24, v28
v_mov_b32_e32 v30, 12
buffer_store_dword v26, v30, s[0:3], 0 offen
v_mov_b32_e32 v10, 0x220
v_mov_b32_e32 v2, 0x01020304
v_mov_b32_e32 v3, 0x85060708
ds_write_b64 v10, v[2:3]
v_mov_b32_e32 v11, 0x420
v_mov_b32_e32 v2, 0xa1a2a3a4
v_mov_b32_e32 v3, 0xb5b6b7b8
ds_write_b64 v11, v[2:3]
s_waitcnt lgkmcnt(0)
v_mov_b32_e32 v1, 0x20
v_mov_b32_e32 v12, 0x31415926
v_mov_b32_e32 v13, 0x53589793
v_mov_b32_e32 v14, 0x27182818
v_mov_b32_e32 v15, 0x28459045
ds_wrxchg2st64_rtn_b64 v[16:19], v1, v[12:13], v[14:15] offset0:1 offset1:2
s_waitcnt lgkmcnt(0)
ds_read_b64 v[20:21], v10
ds_read_b64 v[22:23], v11
s_waitcnt lgkmcnt(0)
v_add_nc_u32_e32 v28, v17, v17
v_add_nc_u32_e32 v28, v28, v17
v_xor_b32_e32 v24, v16, v28
v_add_nc_u32_e32 v28, v19, v19
v_add_nc_u32_e32 v28, v28, v19
v_xor_b32_e32 v25, v18, v28
v_add_nc_u32_e32 v28, v25, v25
v_add_nc_u32_e32 v28, v28, v25
v_xor_b32_e32 v26, v24, v28
v_mov_b32_e32 v30, 16
buffer_store_dword v26, v30, s[0:3], 0 offen
v_add_nc_u32_e32 v28, v21, v21
v_add_nc_u32_e32 v28, v28, v21
v_xor_b32_e32 v24, v20, v28
v_add_nc_u32_e32 v28, v23, v23
v_add_nc_u32_e32 v28, v28, v23
v_xor_b32_e32 v25, v22, v28
v_add_nc_u32_e32 v28, v25, v25
v_add_nc_u32_e32 v28, v28, v25
v_xor_b32_e32 v26, v24, v28
v_mov_b32_e32 v30, 20
buffer_store_dword v26, v30, s[0:3], 0 offen
v_mov_b32_e32 v10, 0x140
v_mov_b32_e32 v2, 0x12345678
ds_write_b32 v10, v2
v_mov_b32_e32 v11, 0x340
v_mov_b32_e32 v2, 0x87654321
ds_write_b32 v11, v2
s_waitcnt lgkmcnt(0)
v_mov_b32_e32 v1, 0x40
v_mov_b32_e32 v12, 0xaaaabbbb
v_mov_b32_e32 v13, 0xccccdddd
ds_wrxchg2st64_rtn_b32 v[16:17], v1, v12, v13 offset0:1 offset1:3
s_waitcnt lgkmcnt(0)
ds_read_b32 v20, v10
ds_read_b32 v21, v11
s_waitcnt lgkmcnt(0)
v_add_nc_u32_e32 v28, v17, v17
v_add_nc_u32_e32 v28, v28, v17
v_xor_b32_e32 v24, v16, v28
v_add_nc_u32_e32 v28, v21, v21
v_add_nc_u32_e32 v28, v28, v21
v_xor_b32_e32 v25, v20, v28
v_add_nc_u32_e32 v28, v25, v25
v_add_nc_u32_e32 v28, v28, v25
v_xor_b32_e32 v26, v24, v28
v_mov_b32_e32 v30, 24
buffer_store_dword v26, v30, s[0:3], 0 offen
v_mov_b32_e32 v10, 0x90
v_mov_b32_e32 v2, 0x11112222
v_mov_b32_e32 v3, 0x33334444
ds_write_b64 v10, v[2:3]
s_waitcnt lgkmcnt(0)
v_mov_b32_e32 v1, 0x80
v_mov_b32_e32 v12, 0x55556666
v_mov_b32_e32 v13, 0x77778888
v_mov_b32_e32 v14, 0x9999aaaa
v_mov_b32_e32 v15, 0xbbbbcccc
ds_wrxchg2_rtn_b64 v[12:15], v1, v[12:13], v[14:15] offset0:2 offset1:2
s_waitcnt lgkmcnt(0)
ds_read_b64 v[20:21], v10
s_waitcnt lgkmcnt(0)
v_add_nc_u32_e32 v28, v13, v13
v_add_nc_u32_e32 v28, v28, v13
v_xor_b32_e32 v24, v12, v28
v_add_nc_u32_e32 v28, v15, v15
v_add_nc_u32_e32 v28, v28, v15
v_xor_b32_e32 v25, v14, v28
v_add_nc_u32_e32 v28, v25, v25
v_add_nc_u32_e32 v28, v28, v25
v_xor_b32_e32 v27, v24, v28
v_add_nc_u32_e32 v28, v21, v21
v_add_nc_u32_e32 v28, v28, v21
v_xor_b32_e32 v25, v20, v28
v_add_nc_u32_e32 v28, v25, v25
v_add_nc_u32_e32 v28, v28, v25
v_xor_b32_e32 v26, v27, v28
v_mov_b32_e32 v30, 28
buffer_store_dword v26, v30, s[0:3], 0 offen
s_endpgm
