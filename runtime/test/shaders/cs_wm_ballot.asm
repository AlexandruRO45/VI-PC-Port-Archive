; Wave64 mask ballot: a split SGPR pair whose two dwords hold separately written mask halves, read as a 64-bit
; count and per lane, read again after its high dword is replaced by an EITHER constant (the --wave64-emulate
; x_mcache key), and v_mbcnt_lo/hi half selection on VCC.
; Assemble (as scripts/run_gpu_tests.py / build.py do): llvm-mc -triple=amdgcn--amdpal -mcpu=gfx1013 -show-encoding (wave64 is inferred by the translator from exec_hi/_b64 EXEC use)
; Run: VKC_SUBGROUP_SIZE=64 vk_compute_test cs_wm_ballot.spv 1 dump
;   (or translated with --wave64-emulate, on any subgroup size)
; The two halves are written from different comparisons, so s6 = lanes 0-2 (low dword 0x00000007) and
; s7 = lanes 62-63 (high dword 0xc0000000); the pair s[6:7] = 0xc000000000000007.
;   out[0] s_bcnt1_i32_b64 s[6:7]                       = 3 + 2 = 5
;   out[1] v_cndmask_b32_e32 with vcc = s[6:7], lane 1  = bit 1 of the low dword = 1
;   out[2] the same, lane 8                             = bit 8 of the low dword = 0
;   out[3] the same, lane 62                            = bit 30 of the high dword = 1
;   out[4] s7 replaced by 0: s_bcnt1_i32_b64            = 3 + 0 = 3
;   out[5] v_mbcnt_lo_u32_b32 vcc_lo, lane 8            = 3 bits below lane 8 in 0x7
;   out[6] v_mbcnt_hi_u32_b32 vcc_hi, lane 63           = 1 bit below lane 63 in 0xc0000000
;   out[7] v_mbcnt_hi_u32_b32 vcc_lo, lane 63           = 3: vcc_lo is the known low half, so 0x7 is read
; A pair read from one dword for both halves gives out[3] = 0, out[4] = 3 from the stale cached exchange, and
; out[6] = 0 or out[7] = 1 from the wrong dword.
; mattr: +wavefrontsize64,-wavefrontsize32
v_cmp_lt_u32_e64 vcc, v0, 3
s_mov_b32 s6, vcc_lo
v_cmp_gt_u32_e64 vcc, v0, 61
s_mov_b32 s7, vcc_hi
s_mov_b64 exec, -1
s_bcnt1_i32_b64 s8, s[6:7]
s_mov_b64 vcc, s[6:7]
v_mov_b32_e32 v18, 1
v_cndmask_b32_e32 v20, 0, v18, vcc
s_mov_b32 exec_lo, 2
s_mov_b32 exec_hi, 0
v_mov_b32_e32 v1, v20
v_mov_b32_e32 v2, 4
buffer_store_dword v1, v2, s[0:3], 0 offen
s_mov_b32 exec_lo, 0x100
s_mov_b32 exec_hi, 0
v_mov_b32_e32 v1, v20
v_mov_b32_e32 v2, 8
buffer_store_dword v1, v2, s[0:3], 0 offen
s_mov_b32 exec_lo, 0
s_mov_b32 exec_hi, 0x40000000
v_mov_b32_e32 v1, v20
v_mov_b32_e32 v2, 12
buffer_store_dword v1, v2, s[0:3], 0 offen
s_mov_b64 exec, -1
s_mov_b32 s7, 0
s_bcnt1_i32_b64 s10, s[6:7]
v_mbcnt_lo_u32_b32 v6, vcc_lo, 0
v_mbcnt_hi_u32_b32 v7, vcc_hi, 0
v_mbcnt_hi_u32_b32 v8, vcc_lo, 0
s_mov_b32 exec_lo, 1
s_mov_b32 exec_hi, 0
v_mov_b32_e32 v1, s8
v_mov_b32_e32 v2, 0
buffer_store_dword v1, v2, s[0:3], 0 offen
v_mov_b32_e32 v1, s10
v_mov_b32_e32 v2, 16
buffer_store_dword v1, v2, s[0:3], 0 offen
s_mov_b32 exec_lo, 0x100
s_mov_b32 exec_hi, 0
v_mov_b32_e32 v1, v6
v_mov_b32_e32 v2, 20
buffer_store_dword v1, v2, s[0:3], 0 offen
s_mov_b32 exec_lo, 0
s_mov_b32 exec_hi, 0x80000000
v_mov_b32_e32 v1, v7
v_mov_b32_e32 v2, 24
buffer_store_dword v1, v2, s[0:3], 0 offen
v_mov_b32_e32 v1, v8
v_mov_b32_e32 v2, 28
buffer_store_dword v1, v2, s[0:3], 0 offen
s_endpgm
