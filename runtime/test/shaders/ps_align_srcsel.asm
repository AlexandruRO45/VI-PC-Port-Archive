; The VOP3 source op_sel bits on v_alignbit_b32 / v_alignbyte_b32 (SHADER-447). On RDNA2 bits 0 and 1 together
; pick the byte of S2 that is the shift amount: k = op_sel[0] + 2 * op_sel[1], the op reads S2 >> 8k (then & 31
; for alignbit, & 3 bytes for alignbyte); bit 2 changes nothing, and S0 / S1 are read whole. A constant S2 is its
; 32-bit value (inline 1.0 is 0x3f800000). Measured on an AMD Radeon iGPU (gfx1036, Windows driver 26.10.1) with
; raw kernels through tools/hw_oracle, wave32 and wave64 alike: a single set bit 8..12 of S2 under
; op_sel:[1,0,0,0] shifts alignbit by 1, 2, 4, 8, 16, every other bit by 0. Until then the translator refused it.
; v0 = 0x12345678 (S0), v1 = 0x9abcdef0 (S1), so {S0, S1} = 0x123456789abcdef0.
; v5 = 0x1d0f0703: its bytes 0..3 are 3, 7, 15, 29. v6 = 0x03020100: its bytes 0..3 are 0, 1, 2, 3.
;   x: v_alignbit_b32 v2, v0, v1, v5 op_sel:[1,0,0,0]: k = 1, shift = 0x07 & 31 = 7:
;      (S1 >> 7) | (S0 << 25) = 0x013579bd | (0x78 << 25 = 0xf0000000) = 0xf13579bd
;   y: v_alignbit_b32 v3, v0, v1, 1.0 op_sel:[1,1,0,0]: k = 3, S2 = 0x3f800000, shift = 0x3f & 31 = 31:
;      (S1 >> 31) | (S0 << 1) = 0x00000001 | 0x2468acf0 = 0x2468acf1
;   z: v_alignbyte_b32 v4, v0, v1, v6 op_sel:[0,1,1,0]: k = 2 (bit 2 ignored), 0x02 & 3 = 2 bytes = 16 bits:
;      (S1 >> 16) | (S0 << 16) = 0x00009abc | 0x56780000 = 0x56789abc
;   w: v_alignbit_b32 v7, v0, v1, v5 op_sel:[0,1,0,1]: k = 2, shift = 0x0f & 31 = 15:
;      (S1 >> 15) | (S0 << 17) = 0x00013579 | (0x5678 << 17 = 0xacf00000) = 0xacf13579; the destination bit then
;      writes it shifted left by 16 (EGEN-HW-OPSEL-DST32): 0x35790000
; Reading S2's low byte instead (op_sel ignored): 0x13579bde, 0x9abcdef0, 0x9abcdef0, 0x9bde0000.
; Expected mrt0 bits: 0xf13579bd, 0x2468acf1, 0x56789abc, 0x35790000
v_mov_b32 v0, 0x12345678
v_mov_b32 v1, 0x9abcdef0
v_mov_b32 v5, 0x1d0f0703
v_mov_b32 v6, 0x03020100
v_alignbit_b32 v2, v0, v1, v5 op_sel:[1,0,0,0]
v_alignbit_b32 v3, v0, v1, 1.0 op_sel:[1,1,0,0]
v_alignbyte_b32 v4, v0, v1, v6 op_sel:[0,1,1,0]
v_alignbit_b32 v7, v0, v1, v5 op_sel:[0,1,0,1]
exp mrt0 v2, v3, v4, v7 done vm
s_endpgm
