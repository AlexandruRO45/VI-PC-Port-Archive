; s_setreg_b32 hwreg(HW_REG_MODE, ...) from an SGPR on MODE.FP_ROUND: the save/restore pair (s_getreg_b32 saves the field,
; s_setreg_b32 puts it back) and an SGPR holding a constant. RDNA2 ISA SOPK S_SETREG_B32: the field (offset, size) of the
; hardware register gets S0 & ((1 << size) - 1); SCC not written. Measured on RDNA2 gfx1036 (tools/hw_oracle raw
; kernels): after FP_ROUND[3:2] = 3 is saved with s_getreg_b32 (2,2) or (0,4), set to 0 and restored with s_setreg_b32
; of the same field, s_getreg_b32 reads 3 / 0xc again; s_setreg_b32 (2,2) of 7 gives MODE[3:0] = 0xc; SCC is kept.
; Until this change every s_setreg_b32 from a register raised Unsupported. The translator restores a value that an
; s_getreg_b32 of the same field saved earlier in the block, or applies a constant; other register sources stay
; Unsupported. The rounding is observed through v_cvt_f16_f32 of 1 + 0.75 * 2^-10 (0x3f801800): nearest even gives
; 1 + 2^-10 = 0x3c01, toward zero gives 1.0 = 0x3c00 (MODE.FP_ROUND[3:2] = 3).
; Lane 0 of a wave32 group stores:
;   out[0] cvt after FP_ROUND[3:2] = 0 (the saved value was 3)              = 0x3c01 = 15361
;   out[1] cvt after s_setreg_b32 (2,2) from the (2,2) save: toward zero     = 0x3c00 = 15360
;   out[2] s_getreg_b32 (2,2) after that restore                             = 3
;   out[3] cvt after FP_ROUND[3:2] = 0, then s_setreg_b32 (0,4) from the (0,4) save = 0x3c00 = 15360
;   out[4] s_getreg_b32 (0,4) after that restore                             = 0xc = 12
;   out[5] cvt after s_setreg_b32 (2,2) from s5 = 0 (constant): nearest even = 0x3c01 = 15361
;   out[6] s_getreg_b32 (2,2) after that                                     = 0
;   out[7] SCC set by s_cmp_eq_u32 before an s_setreg_b32, read by s_cselect_b32 5, 6 after it = 5
; group: 32,1,1
; Run: vk_compute_test cs_w32_setreg_sgpr.spv 1 dump -> out[0..7] = 15361 15360 3 15360 12 15361 0 5
; mattr: +wavefrontsize32,-wavefrontsize64
v_mov_b32_e32 v1, 0x3f801800
s_setreg_imm32_b32 hwreg(HW_REG_MODE, 2, 2), 3
s_getreg_b32 s6, hwreg(HW_REG_MODE, 2, 2)
s_getreg_b32 s7, hwreg(HW_REG_MODE, 0, 4)
s_setreg_imm32_b32 hwreg(HW_REG_MODE, 2, 2), 0
v_cvt_f16_f32_e32 v10, v1
s_setreg_b32 hwreg(HW_REG_MODE, 2, 2), s6
v_cvt_f16_f32_e32 v11, v1
s_getreg_b32 s8, hwreg(HW_REG_MODE, 2, 2)
s_setreg_imm32_b32 hwreg(HW_REG_MODE, 2, 2), 0
s_setreg_b32 hwreg(HW_REG_MODE, 0, 4), s7
v_cvt_f16_f32_e32 v12, v1
s_getreg_b32 s9, hwreg(HW_REG_MODE, 0, 4)
s_mov_b32 s5, 0
s_setreg_b32 hwreg(HW_REG_MODE, 2, 2), s5
v_cvt_f16_f32_e32 v13, v1
s_getreg_b32 s10, hwreg(HW_REG_MODE, 2, 2)
s_cmp_eq_u32 s5, 0
s_setreg_b32 hwreg(HW_REG_MODE, 2, 2), s6
s_cselect_b32 s11, 5, 6
v_and_b32_e32 v10, 0xffff, v10
v_and_b32_e32 v11, 0xffff, v11
v_and_b32_e32 v12, 0xffff, v12
v_and_b32_e32 v13, 0xffff, v13
v_cmpx_eq_u32_e32 0, v0
v_mov_b32_e32 v21, 0
buffer_store_dword v10, v21, s[0:3], 0 offen
v_mov_b32_e32 v21, 4
buffer_store_dword v11, v21, s[0:3], 0 offen
v_mov_b32_e32 v20, s8
v_mov_b32_e32 v21, 8
buffer_store_dword v20, v21, s[0:3], 0 offen
v_mov_b32_e32 v21, 12
buffer_store_dword v12, v21, s[0:3], 0 offen
v_mov_b32_e32 v20, s9
v_mov_b32_e32 v21, 16
buffer_store_dword v20, v21, s[0:3], 0 offen
v_mov_b32_e32 v21, 20
buffer_store_dword v13, v21, s[0:3], 0 offen
v_mov_b32_e32 v20, s10
v_mov_b32_e32 v21, 24
buffer_store_dword v20, v21, s[0:3], 0 offen
v_mov_b32_e32 v20, s11
v_mov_b32_e32 v21, 28
buffer_store_dword v20, v21, s[0:3], 0 offen
s_endpgm
