; SDWA source select on the class mask of v_cmp_class_f16. Assemble with llvm-mc -mcpu=gfx1013 -show-encoding,
; translate, run with vk_harness. The mask is the second operand of an f16 op, so a float inline constant there is
; the f16 value in bits 15:0 over a zero high half (docs/design.md, "f16 inline constants"), and src1_sel narrows
; those 32 bits like a register's, zero-extended. Only bits 9:0 of the mask are classes (RDNA2 ISA, VOPC
; v_cmp_class: bit 0 sNaN, 1 qNaN, 2 -inf, 3 -normal, 4 -denormal, 5 -0, 6 +0, 7 +denormal, 8 +normal, 9 +inf).
; v1 = 0x0000001f: WORD_0 = 0x001f is an f16 +denormal (class bit 7), WORD_1 = 0x0000 is +0 (class bit 6).
; v7 = 0x0000c000 is a mask held in a register. v12 = 0x600d600d is the word for a set VCC bit, v13 = 0xbadbadba
; for a clear one (v_cndmask_b32_e64 v, v13, v12, vcc_lo).
;   x: v_cmp_class_f16_sdwa vcc_lo, v1, -2.0 src0_sel:WORD_0 src1_sel:BYTE_1
;        -2.0 = 0x0000c000, BYTE_1 = 0xc0 = bits 6 and 7; the +denormal is class bit 7 -> set -> 0x600d600d
;        (the f32 bits 0xc0000000 would give BYTE_1 = 0x00 and no class)
;   y: v_cmp_class_f16_sdwa vcc_lo, v1, 4.0 src0_sel:WORD_1 src1_sel:BYTE_1
;        4.0 = 0x00004400, BYTE_1 = 0x44 = bits 2 and 6; the +0 is class bit 6 -> set -> 0x600d600d
;   z: v_cmp_class_f16_sdwa vcc_lo, v1, 1.0 src0_sel:WORD_0 src1_sel:WORD_1
;        1.0 = 0x00003c00, WORD_1 = 0x0000 = no class -> clear -> 0xbadbadba
;        (the f32 bits 0x3f800000 would give WORD_1 = 0x3f80, whose bits 9:0 include class 7 -> set)
;   w: v_cmp_class_f16_sdwa vcc_lo, v1, v7 src0_sel:WORD_0 src1_sel:BYTE_1
;        v7 BYTE_1 = 0xc0 = bits 6 and 7; the +denormal is class bit 7 -> set -> 0x600d600d
; Expected mrt0 bits: 0x600d600d, 0x600d600d, 0xbadbadba, 0x600d600d
; Measured: the same four words on an AMD Radeon iGPU (gfx1036, Ryzen 9 9950X3D, driver 26.10.1) through
; tools/hw_oracle (raw HIP kernel, wave32, game MODE), the same instructions writing v8..v11.
v_mov_b32 v1, 0x0000001f
v_mov_b32 v7, 0x0000c000
v_mov_b32 v12, 0x600d600d
v_mov_b32 v13, 0xbadbadba
v_cmp_class_f16_sdwa vcc_lo, v1, -2.0 src0_sel:WORD_0 src1_sel:BYTE_1
v_cndmask_b32_e64 v8, v13, v12, vcc_lo
v_cmp_class_f16_sdwa vcc_lo, v1, 4.0 src0_sel:WORD_1 src1_sel:BYTE_1
v_cndmask_b32_e64 v9, v13, v12, vcc_lo
v_cmp_class_f16_sdwa vcc_lo, v1, 1.0 src0_sel:WORD_0 src1_sel:WORD_1
v_cndmask_b32_e64 v10, v13, v12, vcc_lo
v_cmp_class_f16_sdwa vcc_lo, v1, v7 src0_sel:WORD_0 src1_sel:BYTE_1
v_cndmask_b32_e64 v11, v13, v12, vcc_lo
exp mrt0 v8, v9, v10, v11 done vm
s_endpgm
