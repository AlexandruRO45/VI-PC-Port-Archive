; The VOP3 clamp bit on 16-bit integer ops that have no saturating form: v_mul_lo_u16 and the 16-bit shifts (and
; v_max/min_u16/i16, whose result is always in range). RDNA2 ignores the bit there: the result wraps to 16 bits
; exactly as without it, op_sel and the kept other half work as usual. Measured on an AMD Radeon iGPU (gfx1036,
; RDNA2, Windows driver 26.10.1) with raw kernels through tools/hw_oracle, wave32 and wave64 alike
; (EGEN-HW-CLAMP16-NOSAT). Until then the translator refused these forms (Unsupported: <op> clamp).
; v0 = 0x8003fffe, v1 = 0x7fff0005, v5 = 0xaaaa8001, v6 = 0x5555000f. The VOP3 forms keep the half they do not
; write (the destinations start as 0x11112222, 0x33334444, 0x55556666, 0x77778888).
;   x: v_mul_lo_u16 v2, v0, v1 clamp: 0xfffe * 0x0005 = 0x4fff6, low 16 bits 0xfff6 (not 0xffff):
;      0x1111fff6
;   y: v_mul_lo_u16 v3, v5, v0 op_sel:[1,1,1] clamp: 0xaaaa * 0x8003 = 0x55550000 + 0xaaaa * 3 (0x1fffe)
;      = 0x5556fffe, low 16 bits 0xfffe into the high half, the low half 0x4444 kept: 0xfffe4444
;   z: v_lshlrev_b16 v4, v6, v0 clamp: 0xfffe << (0x000f & 15 = 15) = 0x7fff0000, low 16 bits 0: 0x55550000
;   w: v_ashrrev_i16 v7, 1, v5 op_sel:[0,1,0] clamp: S1's high half 0xaaaa (-21846) >> 1 = -10923 = 0xd555:
;      0x7777d555
; Saturating to [0, 0xffff] instead would give 0x1111ffff, 0xffff4444, 0x5555ffff, 0x7777d555.
; Expected mrt0 bits: 0x1111fff6, 0xfffe4444, 0x55550000, 0x7777d555
v_mov_b32 v0, 0x8003fffe
v_mov_b32 v1, 0x7fff0005
v_mov_b32 v5, 0xaaaa8001
v_mov_b32 v6, 0x5555000f
v_mov_b32 v2, 0x11112222
v_mov_b32 v3, 0x33334444
v_mov_b32 v4, 0x55556666
v_mov_b32 v7, 0x77778888
v_mul_lo_u16_e64 v2, v0, v1 clamp
v_mul_lo_u16_e64 v3, v5, v0 op_sel:[1,1,1] clamp
v_lshlrev_b16_e64 v4, v6, v0 clamp
v_ashrrev_i16_e64 v7, 1, v5 op_sel:[0,1,0] clamp
exp mrt0 v2, v3, v4, v7 done vm
s_endpgm
