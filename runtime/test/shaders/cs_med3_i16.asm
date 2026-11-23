; group: 32,1,1
; v_med3_i16 / v_med3_u16 with a constant operand (an SGPR constant, an inline constant, a literal), the other two
; operands computed at run time from the output buffer's 0xEEEEEEEE fill (so no compiler can fold them). The median
; of three 16-bit values read through op_sel (RDNA2 ISA, V_MED3_I16 / V_MED3_U16: the middle value; the other half of
; D is kept, op_sel[3] writes the high half). Operands (S0, S1, S2) and results, D starting as given:
;   0: v_med3_i16 v7, s4, v12, v3 op_sel:[1,0,0,0]   s4.hi = 0x0011 = 17, v12 = 0x8000 = -32768, v3 = 0x1d = 29
;        median 17; D = 0xe78a751d keeps its high half and takes 0x0011                          -> 0xE78A0011
;   1: v_med3_i16 v8, 24, v12, v3                    24, -32768, 29: median 24; D = 0x1234abcd   -> 0x12340018
;   2: v_med3_u16 v9, s5, v12, v3                    s5 = 0xfffe0005: low half 5, v12 = 0x8000, v3 = 0x1d: median 0x1d
;        (unsigned: 5, 29, 32768); D = 0x1234abcd                                              -> 0x1234001D
;   3: v_med3_i16 v10, v12, 0xc000, v3 op_sel:[0,0,0,1]   -32768, 0xc000 = -16384, 29: median -16384, written to the
;        high half of D = 0x1234abcd                                                            -> 0xC000ABCD
; Measured natively on RDNA2 (gfx1036, Ryzen 9 9950X3D iGPU, driver 26.10.1) through tools/hw_oracle (raw HIP kernel).
; On the AMD Windows driver the translated min/max/clamp medians of case 0 came out as -32768 (the lower bound); the
; translator now forms the median as the sum minus the smallest and the largest.
; Run: vk_compute_test cs_med3_i16.spv 1 dump
; out[0..7] = E78A0011 12340018 1234001D C000ABCD E78A0011 12340018 1234001D C000ABCD (hex)
s_mov_b32 exec_lo, -1
v_mov_b32_e32 v20, v0
v_lshlrev_b32_e32 v21, 4, v20
buffer_load_dword v22, v21, s[0:3], 0 offen
s_waitcnt vmcnt(0)
v_xor_b32_e32 v12, 0xeeee6eee, v22
v_xor_b32_e32 v3, 0xeeeeeef3, v22
v_xor_b32_e32 v13, 0x1110eeeb, v22
s_mov_b32 s4, 0x110000
s_mov_b32 s5, 0xfffe0005
v_mov_b32_e32 v7, 0xe78a751d
v_mov_b32_e32 v8, 0x1234abcd
v_mov_b32_e32 v9, 0x1234abcd
v_mov_b32_e32 v10, 0x1234abcd
v_med3_i16 v7, s4, v12, v3 op_sel:[1,0,0,0]
v_med3_i16 v8, 24, v12, v3
v_med3_u16 v9, s5, v12, v3
v_med3_i16 v10, v12, 0xc000, v3 op_sel:[0,0,0,1]
buffer_store_dword v7, v21, s[0:3], 0 offen
buffer_store_dword v8, v21, s[0:3], 0 offen offset:4
buffer_store_dword v9, v21, s[0:3], 0 offen offset:8
buffer_store_dword v10, v21, s[0:3], 0 offen offset:12
s_endpgm
