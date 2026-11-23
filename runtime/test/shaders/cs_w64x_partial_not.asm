; Partial wave64: SALU ops that invert a lane mask set the bits of the lanes that were never launched.
; A 40x1x1 group is one wave with lanes 0-39 launched and lanes 40-63 never launched (as cs_w64x_partial).
; vcc = v_cmp_eq(v1, v1) has a bit per launched lane only: vcc = EXEC = 0x000000ff_ffffffff. A SALU op computes
; all 64 bits, so ~vcc = 0xffffff00_00000000 (24 bits, SCC 1), although no launched lane has its bit set.
; Lane 0 stores:
;   out[0] SCC of s_not_b64 s[12:13], vcc          = 1
;   out[1] s_bcnt1_i32_b64 of s[12:13]              = 24
;   out[2] SCC of s_orn2_b64 s[14:15], 0, vcc       = 1   (0 | ~vcc)
;   out[3] s_bcnt1_i32_b64 of s_nand_b64 vcc, exec  = 24
;   out[4] s_cmp_lg_u64 s[12:13], 0                 = 1
;   out[5] s_bcnt1_i32_b64 of s[12:13] & exec       = 0   (the never-launched bits are cleared again)
;   out[6] s_bcnt1_i32_b32 of s_not_b32 vcc_hi      = 24  (~0xff)
;   out[7] s_bcnt1_i32_b64 of s_xnor_b64 vcc, vcc   = 64
; The values follow the RDNA2 ISA SALU definitions (bitwise on all 64 bits, SCC = D != 0) with the never-launched
; lanes 0 after a VALU compare and in EXEC (measured by cs_w64x_partial_scmp). Until the partial-wave shadow, a
; launched lane's own bit was all the translator kept: out = 0 0 0 0 0 0 0 40.
; The group has 40 threads, so --require-wave64 (full subgroups) cannot be used; check.py asks for subgroup size 64
; with VKC_SUBGROUP_SIZE=64 instead (or LEONIDA_W64_EMULATE=1 on a 32-wide GPU).
; Run: VKC_SUBGROUP_SIZE=64 vk_compute_test cs_w64x_partial_not.spv 1 dump
;   -> out[0..7] = 1 24 1 24 1 0 24 64
; mattr: +wavefrontsize64,-wavefrontsize32
; group: 40,1,1
v_mbcnt_lo_u32_b32 v1, -1, 0
v_mbcnt_hi_u32_b32 v1, -1, v1
s_mov_b64 s[6:7], exec
v_cmp_eq_u32_e32 vcc, v1, v1
s_not_b64 s[12:13], vcc
s_cselect_b32 s20, 1, 0
s_bcnt1_i32_b64 s21, s[12:13]
s_orn2_b64 s[14:15], 0, vcc
s_cselect_b32 s22, 1, 0
s_nand_b64 s[16:17], vcc, exec
s_bcnt1_i32_b64 s23, s[16:17]
s_cmp_lg_u64 s[12:13], 0
s_cselect_b32 s24, 1, 0
s_and_b64 s[18:19], s[12:13], exec
s_bcnt1_i32_b64 s25, s[18:19]
s_not_b32 s28, vcc_hi
s_bcnt1_i32_b32 s26, s28
s_xnor_b64 s[30:31], vcc, vcc
s_bcnt1_i32_b64 s27, s[30:31]
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
s_mov_b64 exec, s[6:7]
s_endpgm
