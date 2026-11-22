; cs_w32_f64_ops.asm: f64 VOP1/VOP3 ops (v_fma/ldexp/ceil/fract_f64, v_cvt_i32/u32_f64), lane 0 stores eight dwords.
; group: 32,1,1
; Run: vk_compute_test cs_w32_f64_ops.spv 1 dump -> out[0..7] = 1202590843 1070889697 1 0 2147483647 0 2147483648 1072693247
; mattr: +wavefrontsize32,-wavefrontsize64
v_cmpx_eq_u32_e32 0, v0
v_mov_b32_e32 v5, 0
v_mov_b32 v10, 0x9999999a
v_mov_b32 v11, 0x3fb99999
v_mov_b32 v12, 0x9999999a
v_mov_b32 v13, 0x3fc99999
v_mov_b32 v14, 0x33333333
v_mov_b32 v15, 0x3fd33333
v_fma_f64 v[20:21], v[10:11], v[12:13], v[14:15]
v_mov_b32 v10, 0x0
v_mov_b32 v11, 0x3ff00000
v_mov_b32 v12, 0xfffffbce
v_ldexp_f64 v[22:23], v[10:11], v12
v_mov_b32 v10, 0x78b58c40
v_mov_b32 v11, 0x4415af1d
v_cvt_i32_f64 v24, v[10:11]
v_mov_b32 v10, 0x0
v_mov_b32 v11, 0xc0140000
v_cvt_u32_f64 v25, v[10:11]
v_mov_b32 v10, 0x0
v_mov_b32 v11, 0xbfe80000
v_ceil_f64 v[30:31], v[10:11]
v_mov_b32 v26, v31
v_mov_b32 v10, 0x8b70e62b
v_mov_b32 v11, 0x80001268
v_fract_f64 v[32:33], v[10:11]
v_mov_b32 v27, v33
buffer_store_dword v20, v5, s[0:3], 0 offen offset:0
buffer_store_dword v21, v5, s[0:3], 0 offen offset:4
buffer_store_dword v22, v5, s[0:3], 0 offen offset:8
buffer_store_dword v23, v5, s[0:3], 0 offen offset:12
buffer_store_dword v24, v5, s[0:3], 0 offen offset:16
buffer_store_dword v25, v5, s[0:3], 0 offen offset:20
buffer_store_dword v26, v5, s[0:3], 0 offen offset:24
buffer_store_dword v27, v5, s[0:3], 0 offen offset:28
s_endpgm
