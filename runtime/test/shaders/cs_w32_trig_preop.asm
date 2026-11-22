; cs_w32_trig_preop.asm: v_trig_preop_f64 segments of 2/pi (seg 0, exponent >= 1968, denormal result, funnel shift), lane 0 stores eight dwords.
; group: 32,1,1
; Run: vk_compute_test cs_w32_trig_preop.spv 1 dump -> out[0..7] = 1841940610 1071931184 1381846399 107934026 132000074 0 501947569 682761470
; mattr: +wavefrontsize32,-wavefrontsize64
v_cmpx_eq_u32_e32 0, v0
v_mov_b32_e32 v5, 0
v_mov_b32 v10, 0x0
v_mov_b32 v11, 0x3ff00000
v_mov_b32 v12, 0
v_trig_preop_f64 v[20:21], v[10:11], v12
v_mov_b32 v10, 0x8800759c
v_mov_b32 v11, 0x7e37e43c
v_mov_b32 v12, 2
v_trig_preop_f64 v[22:23], v[10:11], v12
v_mov_b32 v10, 0x456789ab
v_mov_b32 v11, 0x45d00123
v_mov_b32 v12, 19
v_trig_preop_f64 v[24:25], v[10:11], v12
v_mov_b32 v10, 0x8abcdef1
v_mov_b32 v11, 0xc1234567
v_mov_b32 v12, 7
v_trig_preop_f64 v[26:27], v[10:11], v12
buffer_store_dword v20, v5, s[0:3], 0 offen offset:0
buffer_store_dword v21, v5, s[0:3], 0 offen offset:4
buffer_store_dword v22, v5, s[0:3], 0 offen offset:8
buffer_store_dword v23, v5, s[0:3], 0 offen offset:12
buffer_store_dword v24, v5, s[0:3], 0 offen offset:16
buffer_store_dword v25, v5, s[0:3], 0 offen offset:20
buffer_store_dword v26, v5, s[0:3], 0 offen offset:24
buffer_store_dword v27, v5, s[0:3], 0 offen offset:28
s_endpgm
