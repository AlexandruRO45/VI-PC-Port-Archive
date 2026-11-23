; Partial wave32: SALU ops that invert a lane mask set the bits of the lanes that were never launched.
; A 20x1x1 group is one wave32 with lanes 0-19 launched and lanes 20-31 never launched.
; vcc_lo = v_cmp_eq(v1, v1) = EXEC = 0x000fffff, so ~vcc_lo = 0xfff00000 (12 bits, SCC 1).
; Lane 0 stores:
;   out[0] SCC of s_not_b32 s12, vcc_lo                  = 1
;   out[1] s_bcnt1_i32_b32 of s12                        = 12
;   out[2] SCC of s_nand_b32 s13, vcc_lo, exec_lo        = 1
;   out[3] s_bcnt1_i32_b32 of s13                        = 12
;   out[4] s_cmp_lg_u32 s12, 0                           = 1
;   out[5] s_bcnt1_i32_b32 of s12 & exec_lo              = 0
;   out[6] vcc_hi after s_mov_b32 vcc_hi, 0; s_not_b32   = 0xffffffff  (vcc_hi is plain data in wave32)
;   out[7] s_bcnt1_i32_b32 of s_xnor_b32 vcc_lo, vcc_lo  = 32
; The values follow the RDNA2 ISA SALU definitions with the never-launched lanes 0 after a VALU compare and in EXEC
; (measured by cs_w32_partial_scmp). Until the partial-wave shadow: out = 0 0 0 0 0 0 0xffffffff 20.
; Run: VKC_SUBGROUP_SIZE=32 vk_compute_test cs_w32_partial_not.spv 1 dump
;   -> out[0..7] = 1 12 1 12 1 0 4294967295 32
; group: 20,1,1
v_mbcnt_lo_u32_b32 v1, -1, 0
s_mov_b32 s6, exec_lo
v_cmp_eq_u32_e32 vcc_lo, v1, v1
s_not_b32 s12, vcc_lo
s_cselect_b32 s20, 1, 0
s_bcnt1_i32_b32 s21, s12
s_nand_b32 s13, vcc_lo, exec_lo
s_cselect_b32 s22, 1, 0
s_bcnt1_i32_b32 s23, s13
s_cmp_lg_u32 s12, 0
s_cselect_b32 s24, 1, 0
s_and_b32 s14, s12, exec_lo
s_bcnt1_i32_b32 s25, s14
s_mov_b32 vcc_hi, 0
s_not_b32 vcc_hi, vcc_hi
s_mov_b32 s26, vcc_hi
s_xnor_b32 s15, vcc_lo, vcc_lo
s_bcnt1_i32_b32 s27, s15
v_cmpx_eq_u32_e32 0, v1
v_mov_b32_e32 v20, s20
v_mov_b32_e32 v21, 0
buffer_store_dword v20, v21, s[0:3], 0 offen
v_mov_b32_e32 v20, s21
v_mov_b32_e32 v21, 4
buffer_store_dword v20, v21, s[0:3], 0 offen
v_mov_b32_e32 v20, s22
v_mov_b32_e32 v21, 8
buffer_store_dword v20, v21, s[0:3], 0 offen
v_mov_b32_e32 v20, s23
v_mov_b32_e32 v21, 12
buffer_store_dword v20, v21, s[0:3], 0 offen
v_mov_b32_e32 v20, s24
v_mov_b32_e32 v21, 16
buffer_store_dword v20, v21, s[0:3], 0 offen
v_mov_b32_e32 v20, s25
v_mov_b32_e32 v21, 20
buffer_store_dword v20, v21, s[0:3], 0 offen
v_mov_b32_e32 v20, s26
v_mov_b32_e32 v21, 24
buffer_store_dword v20, v21, s[0:3], 0 offen
v_mov_b32_e32 v20, s27
v_mov_b32_e32 v21, 28
buffer_store_dword v20, v21, s[0:3], 0 offen
s_mov_b32 exec_lo, s6
s_endpgm
