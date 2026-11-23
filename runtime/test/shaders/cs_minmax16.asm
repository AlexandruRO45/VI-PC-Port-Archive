; v_min3/v_max3/v_med3 on f16, i16 and u16 (docs/design.md row: three-operand 16-bit min/max/median),
; run once on a GPU. Each of the eight ops gets two operand triples taken from the RDNA2-measured words in
; tests/semantic/test_alu_min3_max3_med3_16_hw.py, each tried with op_sel reading the low or the high half of
; every source and writing the low or the high half of the destination (the other half of v4 is kept).
;   out[k] = v4 after the op for case k; operands' unused half is 0xabcd (low read) or 0x1234 (high read).
; group: 32,1,1
; mattr: +wavefrontsize32,-wavefrontsize64
; Run: vk_compute_test cs_minmax16.spv 1 dump -> out[0..31] = <see check.py MINMAX16_WORDS>
v_mov_b32 v10, 0xabcdbc00
v_mov_b32 v11, 0xabcd4000
v_mov_b32 v12, 0xabcd0001
v_mov_b32 v13, 0xdeadbeef
v_min3_f16 v13, v10, v11, v12
v_mov_b32_e32 v20, 0
buffer_store_dword v13, v20, s[0:3], 0 offen
v_mov_b32 v10, 0xbc001234
v_mov_b32 v11, 0x40001234
v_mov_b32 v12, 0x11234
v_mov_b32 v13, 0xdeadbeef
v_min3_f16 v13, v10, v11, v12 op_sel:[1,1,1,0]
v_mov_b32_e32 v20, 4
buffer_store_dword v13, v20, s[0:3], 0 offen
v_mov_b32 v10, 0xabcd0001
v_mov_b32 v11, 0xabcd4000
v_mov_b32 v12, 0xabcd8000
v_mov_b32 v13, 0xdeadbeef
v_min3_f16 v13, v10, v11, v12 op_sel:[0,0,0,1]
v_mov_b32_e32 v20, 8
buffer_store_dword v13, v20, s[0:3], 0 offen
v_mov_b32 v10, 0x11234
v_mov_b32 v11, 0x40001234
v_mov_b32 v12, 0x80001234
v_mov_b32 v13, 0xdeadbeef
v_min3_f16 v13, v10, v11, v12 op_sel:[1,1,1,1]
v_mov_b32_e32 v20, 12
buffer_store_dword v13, v20, s[0:3], 0 offen
v_mov_b32 v10, 0xabcd7c00
v_mov_b32 v11, 0xabcd4000
v_mov_b32 v12, 0xabcd0001
v_mov_b32 v13, 0xdeadbeef
v_max3_f16 v13, v10, v11, v12
v_mov_b32_e32 v20, 16
buffer_store_dword v13, v20, s[0:3], 0 offen
v_mov_b32 v10, 0x7c001234
v_mov_b32 v11, 0x40001234
v_mov_b32 v12, 0x11234
v_mov_b32 v13, 0xdeadbeef
v_max3_f16 v13, v10, v11, v12 op_sel:[1,1,1,0]
v_mov_b32_e32 v20, 20
buffer_store_dword v13, v20, s[0:3], 0 offen
v_mov_b32 v10, 0xabcd0000
v_mov_b32 v11, 0xabcdbc00
v_mov_b32 v12, 0xabcd7c00
v_mov_b32 v13, 0xdeadbeef
v_max3_f16 v13, v10, v11, v12 op_sel:[0,0,0,1]
v_mov_b32_e32 v20, 24
buffer_store_dword v13, v20, s[0:3], 0 offen
v_mov_b32 v10, 0x1234
v_mov_b32 v11, 0xbc001234
v_mov_b32 v12, 0x7c001234
v_mov_b32 v13, 0xdeadbeef
v_max3_f16 v13, v10, v11, v12 op_sel:[1,1,1,1]
v_mov_b32_e32 v20, 28
buffer_store_dword v13, v20, s[0:3], 0 offen
v_mov_b32 v10, 0xabcd8001
v_mov_b32 v11, 0xabcdfffe
v_mov_b32 v12, 0xabcdffff
v_mov_b32 v13, 0xdeadbeef
v_min3_i16 v13, v10, v11, v12
v_mov_b32_e32 v20, 32
buffer_store_dword v13, v20, s[0:3], 0 offen
v_mov_b32 v10, 0x80011234
v_mov_b32 v11, 0xfffe1234
v_mov_b32 v12, 0xffff1234
v_mov_b32 v13, 0xdeadbeef
v_min3_i16 v13, v10, v11, v12 op_sel:[1,1,1,0]
v_mov_b32_e32 v20, 36
buffer_store_dword v13, v20, s[0:3], 0 offen
v_mov_b32 v10, 0xabcd8001
v_mov_b32 v11, 0xabcd8000
v_mov_b32 v12, 0xabcd0001
v_mov_b32 v13, 0xdeadbeef
v_min3_i16 v13, v10, v11, v12 op_sel:[0,0,0,1]
v_mov_b32_e32 v20, 40
buffer_store_dword v13, v20, s[0:3], 0 offen
v_mov_b32 v10, 0x80011234
v_mov_b32 v11, 0x80001234
v_mov_b32 v12, 0x11234
v_mov_b32 v13, 0xdeadbeef
v_min3_i16 v13, v10, v11, v12 op_sel:[1,1,1,1]
v_mov_b32_e32 v20, 44
buffer_store_dword v13, v20, s[0:3], 0 offen
v_mov_b32 v10, 0xabcd0001
v_mov_b32 v11, 0xabcd7fff
v_mov_b32 v12, 0xabcdffff
v_mov_b32 v13, 0xdeadbeef
v_min3_u16 v13, v10, v11, v12
v_mov_b32_e32 v20, 48
buffer_store_dword v13, v20, s[0:3], 0 offen
v_mov_b32 v10, 0x11234
v_mov_b32 v11, 0x7fff1234
v_mov_b32 v12, 0xffff1234
v_mov_b32 v13, 0xdeadbeef
v_min3_u16 v13, v10, v11, v12 op_sel:[1,1,1,0]
v_mov_b32_e32 v20, 52
buffer_store_dword v13, v20, s[0:3], 0 offen
v_mov_b32 v10, 0xabcd8001
v_mov_b32 v11, 0xabcd1234
v_mov_b32 v12, 0xabcdffff
v_mov_b32 v13, 0xdeadbeef
v_min3_u16 v13, v10, v11, v12 op_sel:[0,0,0,1]
v_mov_b32_e32 v20, 56
buffer_store_dword v13, v20, s[0:3], 0 offen
v_mov_b32 v10, 0x80011234
v_mov_b32 v11, 0x12341234
v_mov_b32 v12, 0xffff1234
v_mov_b32 v13, 0xdeadbeef
v_min3_u16 v13, v10, v11, v12 op_sel:[1,1,1,1]
v_mov_b32_e32 v20, 60
buffer_store_dword v13, v20, s[0:3], 0 offen
v_mov_b32 v10, 0xabcd0000
v_mov_b32 v11, 0xabcdfffe
v_mov_b32 v12, 0xabcd8000
v_mov_b32 v13, 0xdeadbeef
v_max3_i16 v13, v10, v11, v12
v_mov_b32_e32 v20, 64
buffer_store_dword v13, v20, s[0:3], 0 offen
v_mov_b32 v10, 0x1234
v_mov_b32 v11, 0xfffe1234
v_mov_b32 v12, 0x80001234
v_mov_b32 v13, 0xdeadbeef
v_max3_i16 v13, v10, v11, v12 op_sel:[1,1,1,0]
v_mov_b32_e32 v20, 68
buffer_store_dword v13, v20, s[0:3], 0 offen
v_mov_b32 v10, 0xabcd0000
v_mov_b32 v11, 0xabcd8000
v_mov_b32 v12, 0xabcdffff
v_mov_b32 v13, 0xdeadbeef
v_max3_i16 v13, v10, v11, v12 op_sel:[0,0,0,1]
v_mov_b32_e32 v20, 72
buffer_store_dword v13, v20, s[0:3], 0 offen
v_mov_b32 v10, 0x1234
v_mov_b32 v11, 0x80001234
v_mov_b32 v12, 0xffff1234
v_mov_b32 v13, 0xdeadbeef
v_max3_i16 v13, v10, v11, v12 op_sel:[1,1,1,1]
v_mov_b32_e32 v20, 76
buffer_store_dword v13, v20, s[0:3], 0 offen
v_mov_b32 v10, 0xabcd8000
v_mov_b32 v11, 0xabcd0001
v_mov_b32 v12, 0xabcd0000
v_mov_b32 v13, 0xdeadbeef
v_max3_u16 v13, v10, v11, v12
v_mov_b32_e32 v20, 80
buffer_store_dword v13, v20, s[0:3], 0 offen
v_mov_b32 v10, 0x80001234
v_mov_b32 v11, 0x11234
v_mov_b32 v12, 0x1234
v_mov_b32 v13, 0xdeadbeef
v_max3_u16 v13, v10, v11, v12 op_sel:[1,1,1,0]
v_mov_b32_e32 v20, 84
buffer_store_dword v13, v20, s[0:3], 0 offen
v_mov_b32 v10, 0xabcd1234
v_mov_b32 v11, 0xabcd7fff
v_mov_b32 v12, 0xabcdfffe
v_mov_b32 v13, 0xdeadbeef
v_max3_u16 v13, v10, v11, v12 op_sel:[0,0,0,1]
v_mov_b32_e32 v20, 88
buffer_store_dword v13, v20, s[0:3], 0 offen
v_mov_b32 v10, 0x12341234
v_mov_b32 v11, 0x7fff1234
v_mov_b32 v12, 0xfffe1234
v_mov_b32 v13, 0xdeadbeef
v_max3_u16 v13, v10, v11, v12 op_sel:[1,1,1,1]
v_mov_b32_e32 v20, 92
buffer_store_dword v13, v20, s[0:3], 0 offen
v_mov_b32 v10, 0xabcd0000
v_mov_b32 v11, 0xabcd1234
v_mov_b32 v12, 0xabcdffff
v_mov_b32 v13, 0xdeadbeef
v_med3_i16 v13, v10, v11, v12
v_mov_b32_e32 v20, 96
buffer_store_dword v13, v20, s[0:3], 0 offen
v_mov_b32 v10, 0x1234
v_mov_b32 v11, 0x12341234
v_mov_b32 v12, 0xffff1234
v_mov_b32 v13, 0xdeadbeef
v_med3_i16 v13, v10, v11, v12 op_sel:[1,1,1,0]
v_mov_b32_e32 v20, 100
buffer_store_dword v13, v20, s[0:3], 0 offen
v_mov_b32 v10, 0xabcd8001
v_mov_b32 v11, 0xabcd8000
v_mov_b32 v12, 0xabcdfffe
v_mov_b32 v13, 0xdeadbeef
v_med3_i16 v13, v10, v11, v12 op_sel:[0,0,0,1]
v_mov_b32_e32 v20, 104
buffer_store_dword v13, v20, s[0:3], 0 offen
v_mov_b32 v10, 0x80011234
v_mov_b32 v11, 0x80001234
v_mov_b32 v12, 0xfffe1234
v_mov_b32 v13, 0xdeadbeef
v_med3_i16 v13, v10, v11, v12 op_sel:[1,1,1,1]
v_mov_b32_e32 v20, 108
buffer_store_dword v13, v20, s[0:3], 0 offen
v_mov_b32 v10, 0xabcdffff
v_mov_b32 v11, 0xabcd0000
v_mov_b32 v12, 0xabcdfffe
v_mov_b32 v13, 0xdeadbeef
v_med3_u16 v13, v10, v11, v12
v_mov_b32_e32 v20, 112
buffer_store_dword v13, v20, s[0:3], 0 offen
v_mov_b32 v10, 0xffff1234
v_mov_b32 v11, 0x1234
v_mov_b32 v12, 0xfffe1234
v_mov_b32 v13, 0xdeadbeef
v_med3_u16 v13, v10, v11, v12 op_sel:[1,1,1,0]
v_mov_b32_e32 v20, 116
buffer_store_dword v13, v20, s[0:3], 0 offen
v_mov_b32 v10, 0xabcd1234
v_mov_b32 v11, 0xabcd0001
v_mov_b32 v12, 0xabcd8000
v_mov_b32 v13, 0xdeadbeef
v_med3_u16 v13, v10, v11, v12 op_sel:[0,0,0,1]
v_mov_b32_e32 v20, 120
buffer_store_dword v13, v20, s[0:3], 0 offen
v_mov_b32 v10, 0x12341234
v_mov_b32 v11, 0x11234
v_mov_b32 v12, 0x80001234
v_mov_b32 v13, 0xdeadbeef
v_med3_u16 v13, v10, v11, v12 op_sel:[1,1,1,1]
v_mov_b32_e32 v20, 124
buffer_store_dword v13, v20, s[0:3], 0 offen
s_endpgm
