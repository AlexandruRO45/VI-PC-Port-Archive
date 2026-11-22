; s_getreg_b32 of HW_REG_MODE.FP_ROUND (MODE[3:0]: [1:0] f32 rounding, [3:2] f16/f64 rounding).
; RDNA2 ISA SOPK S_GETREG_B32: D = hardware register field (offset, size) right-justified, SCC not written. Measured on
; RDNA2 gfx1036 (tools/hw_oracle raw kernels): MODE[3:0] reads 0 at the start with round-to-nearest descriptors; after
; s_setreg_imm32_b32 hwreg(HW_REG_MODE, 2, 2), 3 the fields read (2,2) = 3, (0,4) = 0xc, (3,1) = 1, (1,3) = 6, and after
; writing 0 back (2,2) = 0; SCC set before the read is still set after it. The translator models MODE[1:0] as 0 (it
; always rounds f32 to nearest even) and MODE[3:2] as the f16/f64 rounding it tracks through s_setreg (%mode_r16,
; 0 at the start). Until this change s_getreg_b32 raised Unsupported.
; Lane 0 of a wave32 group stores:
;   out[0] (0,4) at the start                         = 0
;   out[1] (2,2) at the start                         = 0
;   out[2] (2,2) after setting FP_ROUND[3:2] = 3      = 3
;   out[3] (0,4)                                      = 0b1100 = 12
;   out[4] (3,1)                                      = bit 3 = 1
;   out[5] (1,3)                                      = bits 3:1 = 0b110 = 6
;   out[6] (2,2) after setting FP_ROUND[3:2] = 0      = 0
;   out[7] SCC set by s_cmp_eq_u32 before an s_getreg_b32, read by s_cselect_b32 5, 6 after it = 5
; group: 32,1,1
; Run: vk_compute_test cs_w32_getreg_mode.spv 1 dump -> out[0..7] = 0 0 3 12 1 6 0 5
; mattr: +wavefrontsize32,-wavefrontsize64
s_getreg_b32 s4, hwreg(HW_REG_MODE, 0, 4)
s_getreg_b32 s5, hwreg(HW_REG_MODE, 2, 2)
s_setreg_imm32_b32 hwreg(HW_REG_MODE, 2, 2), 3
s_getreg_b32 s6, hwreg(HW_REG_MODE, 2, 2)
s_getreg_b32 s7, hwreg(HW_REG_MODE, 0, 4)
s_getreg_b32 s8, hwreg(HW_REG_MODE, 3, 1)
s_getreg_b32 s9, hwreg(HW_REG_MODE, 1, 3)
s_setreg_imm32_b32 hwreg(HW_REG_MODE, 2, 2), 0
s_getreg_b32 s10, hwreg(HW_REG_MODE, 2, 2)
s_mov_b32 s12, 7
s_cmp_eq_u32 s12, 7
s_getreg_b32 s12, hwreg(HW_REG_MODE, 2, 2)
s_cselect_b32 s11, 5, 6
v_cmpx_eq_u32_e32 0, v0
v_mov_b32_e32 v20, s4
v_mov_b32_e32 v21, 0
buffer_store_dword v20, v21, s[0:3], 0 offen
v_mov_b32_e32 v20, s5
v_mov_b32_e32 v21, 4
buffer_store_dword v20, v21, s[0:3], 0 offen
v_mov_b32_e32 v20, s6
v_mov_b32_e32 v21, 8
buffer_store_dword v20, v21, s[0:3], 0 offen
v_mov_b32_e32 v20, s7
v_mov_b32_e32 v21, 12
buffer_store_dword v20, v21, s[0:3], 0 offen
v_mov_b32_e32 v20, s8
v_mov_b32_e32 v21, 16
buffer_store_dword v20, v21, s[0:3], 0 offen
v_mov_b32_e32 v20, s9
v_mov_b32_e32 v21, 20
buffer_store_dword v20, v21, s[0:3], 0 offen
v_mov_b32_e32 v20, s10
v_mov_b32_e32 v21, 24
buffer_store_dword v20, v21, s[0:3], 0 offen
v_mov_b32_e32 v20, s11
v_mov_b32_e32 v21, 28
buffer_store_dword v20, v21, s[0:3], 0 offen
s_endpgm
