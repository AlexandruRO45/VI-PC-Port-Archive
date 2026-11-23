; Hand-written short assembler forms (docs/design.md): llvm-mc takes the three-operand VOP2 carry form, whose carry
; out and in are VCC, and a modifier list shorter than the operands, whose missing entries are 0. It prints both in
; their long form (v_add_co_ci_u32_e32 vD, vcc_lo, src0, src1, vcc_lo and [0,0,0]), which is the spelling the GPU
; runner's llvm-mc -show-encoding pass hands the translator; this checks the words those forms stand for on a device.
; Lane 0 of a wave32 group computes one result per form and stores eight dwords:
;   out[0] = v_add_co_ci_u32(0xfffffff0, 0x20) with VCC carry in 1 = 0x11        (0x1_0000_0011, carry out 1)
;   out[1] = carry out of that add                                 = 1
;   out[2] = v_sub_co_ci_u32(0xfffffff0, 0x20) with VCC carry in 1 = 0xffffffcf  (no borrow)
;   out[3] = carry out of that subtract                            = 0
;   out[4] = v_subrev_co_ci_u32(0xfffffff0, 0x20) carry in 1       = 0x2f        (0x20 - 0xfffffff0 - 1, borrow)
;   out[5] = carry out of that reverse subtract                    = 1
;   out[6] = v_pk_fma_f16(1.0/2.0, 3.0/-4.0, 2.0/1.0) op_sel_hi:[0,0] = 0x45004500 (the short list is [0,0,0])
;   out[7] = v_pk_fma_f16(1.0/2.0, 3.0/-4.0, 2.0/1.0) neg_lo:[1,0]    = 0xc700bc00
; Each carry op overwrites VCC, so the carry in is set again before the next one; v_cmp writes VCC for the whole
; wave with the lanes EXEC turned off cleared, which is why lane 0 is isolated first.
; group: 32,1,1
; Run: vk_compute_test cs_w32_short_asm_forms.spv 1 dump -> out[0..7] = 17 1 4294967247 0 47 1 1157645568 3338714112
; mattr: +wavefrontsize32,-wavefrontsize64
v_mov_b32_e32 v1, 0xfffffff0
v_mov_b32_e32 v2, 0x20
v_cmpx_eq_u32_e32 0, v0
v_cmp_eq_u32 vcc_lo, v1, v1
v_add_co_ci_u32 v10, v1, v2
v_cndmask_b32 v11, 0, 1, vcc_lo
v_cmp_eq_u32 vcc_lo, v1, v1
v_sub_co_ci_u32 v12, v1, v2
v_cndmask_b32 v13, 0, 1, vcc_lo
v_cmp_eq_u32 vcc_lo, v1, v1
v_subrev_co_ci_u32 v14, v1, v2
v_cndmask_b32 v15, 0, 1, vcc_lo
v_mov_b32_e32 v3, 0x40003c00
v_mov_b32_e32 v4, 0xc4004200
v_mov_b32_e32 v5, 0x3c004000
v_pk_fma_f16 v16, v3, v4, v5 op_sel_hi:[0,0]
v_pk_fma_f16 v17, v3, v4, v5 neg_lo:[1,0]
v_mov_b32_e32 v25, 0
buffer_store_dword v10, v25, s[0:3], 0 offen
v_mov_b32_e32 v25, 4
buffer_store_dword v11, v25, s[0:3], 0 offen
v_mov_b32_e32 v25, 8
buffer_store_dword v12, v25, s[0:3], 0 offen
v_mov_b32_e32 v25, 12
buffer_store_dword v13, v25, s[0:3], 0 offen
v_mov_b32_e32 v25, 16
buffer_store_dword v14, v25, s[0:3], 0 offen
v_mov_b32_e32 v25, 20
buffer_store_dword v15, v25, s[0:3], 0 offen
v_mov_b32_e32 v25, 24
buffer_store_dword v16, v25, s[0:3], 0 offen
v_mov_b32_e32 v25, 28
buffer_store_dword v17, v25, s[0:3], 0 offen
s_endpgm
