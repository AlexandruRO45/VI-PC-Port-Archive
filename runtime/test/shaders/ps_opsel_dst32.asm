; The VOP3 op_sel destination bit on an op whose result is a whole dword (v_pack_b32_f16, v_alignbit_b32,
; v_alignbyte_b32; llvm-mc gfx1013 accepts op_sel on these three). RDNA2 writes the result shifted left by 16: the
; result's low half goes to D's high half and D's low half becomes 0; nothing of D's old value (0xaaaa5555 here) is
; kept. Measured on an AMD Radeon iGPU (gfx1036, RDNA2, Windows driver 26.10.1) with raw kernels through
; tools/hw_oracle, wave32 and wave64 alike (EGEN-HW-OPSEL-DST32). Until then the translator ignored the bit and
; wrote the plain result.
; v0 = 0x12345678, v1 = 0x9abcdef0, v5 = 6 (alignbyte reads v5 & 3 = 2 bytes, alignbit v5 & 31 = 6 bits).
;   x: v_pack_b32_f16 v2, v0, v1 op_sel:[1,0,1]: packed = v0.hi | v1.lo << 16 = 0x1234 | 0xdef0 << 16 = 0xdef01234
;      (both halves are normal f16 values, nothing to quiet); << 16 = 0x12340000 (the S1 half drops out)
;   y: v_alignbit_b32 v3, v0, v1, 4 op_sel:[0,0,0,1]: {v0, v1} >> 4 = 0x89abcdef; << 16 = 0xcdef0000
;   z: v_alignbyte_b32 v4, v0, v1, v5 op_sel:[0,0,0,1]: {v0, v1} >> 16 = 0x56789abc; << 16 = 0x9abc0000
;   w: v_alignbit_b32 v6, v0, v1, v5 op_sel:[0,0,0,1]: {v0, v1} >> 6 = (0x9abcdef0 >> 6) | (0x12345678 << 26)
;      = 0x026af37b | 0xe0000000 = 0xe26af37b; << 16 = 0xf37b0000
; Without the rule: 0xdef01234, 0x89abcdef, 0x56789abc, 0xe26af37b.
; Expected mrt0 bits: 0x12340000, 0xcdef0000, 0x9abc0000, 0xf37b0000
v_mov_b32 v0, 0x12345678
v_mov_b32 v1, 0x9abcdef0
v_mov_b32 v5, 6
v_mov_b32 v2, 0xaaaa5555
v_mov_b32 v3, 0xaaaa5555
v_mov_b32 v4, 0xaaaa5555
v_mov_b32 v6, 0xaaaa5555
v_pack_b32_f16 v2, v0, v1 op_sel:[1,0,1]
v_alignbit_b32 v3, v0, v1, 4 op_sel:[0,0,0,1]
v_alignbyte_b32 v4, v0, v1, v5 op_sel:[0,0,0,1]
v_alignbit_b32 v6, v0, v1, v5 op_sel:[0,0,0,1]
exp mrt0 v2, v3, v4, v6 done vm
s_endpgm
