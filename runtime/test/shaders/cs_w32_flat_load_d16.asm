; flat_load_{ubyte,sbyte,short}_d16 and _d16_hi: one 16-bit half of the destination takes the (zero- or sign-extended)
; byte or short, the other half is kept. Also an instruction offset of 2044 on an address 2044 below the target (the
; 64-bit address 0xfffffffffffff884), and a load with EXEC = 0, which leaves the register as it was.
;   out[0] ubyte_d16   [0x80+4]=0xab into v=0xdead1234 low half   = 0xdead00ab
;   out[1] sbyte_d16   same byte, sign-extended to 16 bits        = 0xdeadffab
;   out[2] short_d16   [0x80+8]=0xcdef into low half              = 0xdeadcdef
;   out[3] ubyte_d16_hi 0xab into the high half of 0x1234abcd     = 0x00ababcd
;   out[4] sbyte_d16_hi sign-extended                             = 0xffababcd
;   out[5] short_d16_hi 0xcdef                                    = 0xcdefabcd
;   out[6] flat_load_dword v[30:31]=0x80-2044 offset:2044         = 0x11223344
;   out[7] flat_load_dword with EXEC = 0 into v=0x5a5a5a5a        = 0x5a5a5a5a
; Measured on gfx1036 (Windows) as a raw kernel, wave32 and wave64: the words above.
; group: 32,1,1
; Run: vk_compute_test cs_w32_flat_load_d16.spv 1 dump -> out[0..7] = 3735879851 3735945131 3735932399 11250637 4289440717 3455036365 287454020 1515870810
; mattr: +wavefrontsize32,-wavefrontsize64; Seeds (MUBUF stores through the V# in s[0:3]; the heap is the global address space, so byte n is address n):
;   0x80 0x11223344, 0x84 0xab, 0x88 0xcdef, 0x90 1, 0x94 2, 0x98 0xa1, 0x9c 0xa2, 0xa0 0xa3, 0xa8..0xb4 0xb1..0xb4.
; v[10:11] = the 64-bit address 0x80. Lane 0 only (EXEC cut), one dword per out word.
v_cmpx_eq_u32_e32 0, v0
v_mov_b32_e32 v1, 0x11223344
v_mov_b32_e32 v2, 0x80
buffer_store_dword v1, v2, s[0:3], 0 offen
v_mov_b32_e32 v1, 0xab
v_mov_b32_e32 v2, 0x84
buffer_store_dword v1, v2, s[0:3], 0 offen
v_mov_b32_e32 v1, 0xcdef
v_mov_b32_e32 v2, 0x88
buffer_store_dword v1, v2, s[0:3], 0 offen
v_mov_b32_e32 v1, 1
v_mov_b32_e32 v2, 0x90
buffer_store_dword v1, v2, s[0:3], 0 offen
v_mov_b32_e32 v1, 2
v_mov_b32_e32 v2, 0x94
buffer_store_dword v1, v2, s[0:3], 0 offen
v_mov_b32_e32 v1, 0xa1
v_mov_b32_e32 v2, 0x98
buffer_store_dword v1, v2, s[0:3], 0 offen
v_mov_b32_e32 v1, 0xa2
v_mov_b32_e32 v2, 0x9c
buffer_store_dword v1, v2, s[0:3], 0 offen
v_mov_b32_e32 v1, 0xa3
v_mov_b32_e32 v2, 0xa0
buffer_store_dword v1, v2, s[0:3], 0 offen
v_mov_b32_e32 v1, 0xb1
v_mov_b32_e32 v2, 0xa8
buffer_store_dword v1, v2, s[0:3], 0 offen
v_mov_b32_e32 v1, 0xb2
v_mov_b32_e32 v2, 0xac
buffer_store_dword v1, v2, s[0:3], 0 offen
v_mov_b32_e32 v1, 0xb3
v_mov_b32_e32 v2, 0xb0
buffer_store_dword v1, v2, s[0:3], 0 offen
v_mov_b32_e32 v1, 0xb4
v_mov_b32_e32 v2, 0xb4
buffer_store_dword v1, v2, s[0:3], 0 offen
s_waitcnt_vscnt null, 0x0
v_mov_b32_e32 v10, 0x80
v_mov_b32_e32 v11, 0
v_mov_b32_e32 v20, 0xdead1234
flat_load_ubyte_d16 v20, v[10:11] offset:4
v_mov_b32_e32 v21, 0xdead1234
flat_load_sbyte_d16 v21, v[10:11] offset:4
v_mov_b32_e32 v22, 0xdead1234
flat_load_short_d16 v22, v[10:11] offset:8
v_mov_b32_e32 v23, 0x1234abcd
flat_load_ubyte_d16_hi v23, v[10:11] offset:4
v_mov_b32_e32 v24, 0x1234abcd
flat_load_sbyte_d16_hi v24, v[10:11] offset:4
v_mov_b32_e32 v25, 0x1234abcd
flat_load_short_d16_hi v25, v[10:11] offset:8
v_add_co_u32 v30, vcc_lo, 0xfffff804, v10
v_add_co_ci_u32 v31, vcc_lo, -1, v11, vcc_lo
flat_load_dword v26, v[30:31] offset:2044
v_mov_b32_e32 v27, 0x5a5a5a5a
s_mov_b32 s20, exec_lo
s_mov_b32 exec_lo, 0
flat_load_dword v27, v[10:11]
s_mov_b32 exec_lo, s20
s_waitcnt vmcnt(0) lgkmcnt(0)
v_mov_b32_e32 v40, 0
buffer_store_dword v20, v40, s[0:3], 0 offen
v_mov_b32_e32 v40, 4
buffer_store_dword v21, v40, s[0:3], 0 offen
v_mov_b32_e32 v40, 8
buffer_store_dword v22, v40, s[0:3], 0 offen
v_mov_b32_e32 v40, 12
buffer_store_dword v23, v40, s[0:3], 0 offen
v_mov_b32_e32 v40, 16
buffer_store_dword v24, v40, s[0:3], 0 offen
v_mov_b32_e32 v40, 20
buffer_store_dword v25, v40, s[0:3], 0 offen
v_mov_b32_e32 v40, 24
buffer_store_dword v26, v40, s[0:3], 0 offen
v_mov_b32_e32 v40, 28
buffer_store_dword v27, v40, s[0:3], 0 offen
s_endpgm
