; Single-rounded binary16 FMAC, with the destination high half preserved in every MRT word.
; x: (3/2)(1025/1024) - 2^-24 is just below the midpoint between 0x3e01 and 0x3e02.
;    Rounding the product to f16 first, or rounding the sum to f32 first, gives 0x3e02.
; y: (1025/1024)(1023/1024) - 1 = -2^-20 (0x8010). Rounding the product first gives +0.
; z and w negate the respective exact sums by negating the first factor and the accumulator.
; Expected mrt0 bits: 0xdead3e01, 0xa55a8010, 0xdeadbe01, 0xa55a0010.
v_mov_b32_e32 v0, 0x3e00
v_mov_b32_e32 v1, 0x3c01
v_mov_b32_e32 v2, 0xdead8001
v_fmac_f16_e32 v2, v0, v1
v_mov_b32_e32 v6, 0x3bfe
v_mov_b32_e32 v3, 0xa55abc00
v_fmac_f16_e32 v3, v1, v6
v_mov_b32_e32 v0, 0xbe00
v_mov_b32_e32 v4, 0xdead0001
v_fmac_f16_e32 v4, v0, v1
v_mov_b32_e32 v1, 0xbc01
v_mov_b32_e32 v5, 0xa55a3c00
v_fmac_f16_e32 v5, v1, v6
exp mrt0 v2, v3, v4, v5 done vm
s_endpgm
