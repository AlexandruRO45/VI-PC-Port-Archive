; s_cmov_b64 into EXEC in wave64: EXEC = SCC ? S0 : EXEC (RDNA2 ISA SOP1 S_CMOV_B64: D = SCC ? S0 : D, SCC read and not
; written). EXEC is one bit per lane: an inline constant gives lane L bit L of its sign-extended 64-bit value, a data
; SGPR pair gives lanes 0-31 their bit of the low SGPR and lanes 32-63 of the high one, a v_cmp lane mask is taken as
; it is. Cases 0-1 are LLVM's llvm.amdgcn.init.exec.from.input prologue (s_bfm_b64 exec, n, 0; s_cmp_eq_u32 n, 64;
; s_cmov_b64 exec, -1: n = 64 wraps s_bfm_b64 to no lanes, so the cmov turns all 64 on). Until this change the
; translator raised Unsupported("s_cmov_b64 destination") for EXEC.
; Each case sets EXEC and SCC, runs the cmov and marks the lanes it leaves on; with EXEC back to all lanes, v_cmp turns
; the marks into a lane mask and lane 0 stores word = (set lanes) | (lowest set lane) << 8 (no lane: 0xffffff00).
;   out[0] s4=40: s_bfm_b64 exec, s4, 0 (lanes 0-39); SCC = (40 == 64) = 0; s_cmov_b64 exec, -1 -> lanes 0-39   40 | 0 = 40
;   out[1] s4=64: s_bfm_b64 exec, s4, 0 (size 64 & 63 = 0: none); SCC = 1; s_cmov_b64 exec, -1 -> all 64      64 | 0 = 64
;   out[2] EXEC=0xff, SCC=1, s_cmov_b64 exec, s[20:21] = (0xf, 0xf0) -> lanes 0-3, 36-39                      8 | 0 = 8
;   out[3] s[20:21] = v_cmp_lt_u32 20 < lane (lanes 21-63); EXEC=0xf, SCC=1, s_cmov_b64 exec, s[20:21]      43 | 21<<8 = 5419
;   out[4] EXEC=0xff00, SCC=0, s_cmov_b64 exec, -1 -> EXEC kept, lanes 8-15                                   8 | 8<<8 = 2056
;   out[5] EXEC=0, SCC=1, s_cmov_b64 exec, -16 -> lanes 4-63                                                  60 | 4<<8 = 1084
;   out[6] EXEC=all, SCC=0, s_cmov_b64 exec, 0 -> EXEC kept, all 64                                           64 | 0 = 64
;   out[7] EXEC=0xffff, SCC=1, s_cmov_b64 exec, s[20:21] = (0, 0x80000001) -> lanes 32, 63                     2 | 32<<8 = 8194
; Run: VKC_SUBGROUP_SIZE=64 vk_compute_test cs_w64_cmov_exec.spv 1 dump --require-wave64
;   (or translated with --wave64-emulate, on any subgroup size) -> out[0..7] = 40 64 8 5419 2056 1084 64 8194
; mattr: +wavefrontsize64,-wavefrontsize32
v_mov_b32_e32 v10, 0
v_mov_b32_e32 v11, 0
v_mov_b32_e32 v12, 0
v_mov_b32_e32 v13, 0
v_mov_b32_e32 v14, 0
v_mov_b32_e32 v15, 0
v_mov_b32_e32 v16, 0
v_mov_b32_e32 v17, 0
s_mov_b32 s4, 40
s_bfm_b64 exec, s4, 0
s_cmp_eq_u32 s4, 64
s_cmov_b64 exec, -1
v_mov_b32_e32 v10, 1
s_mov_b32 s4, 64
s_bfm_b64 exec, s4, 0
s_cmp_eq_u32 s4, 64
s_cmov_b64 exec, -1
v_mov_b32_e32 v11, 1
s_mov_b64 exec, 0xff
s_mov_b32 s20, 0xf
s_mov_b32 s21, 0xf0
s_cmp_eq_u32 s4, s4
s_cmov_b64 exec, s[20:21]
v_mov_b32_e32 v12, 1
s_mov_b64 exec, -1
v_cmp_lt_u32_e64 s[20:21], 20, v0
s_mov_b64 exec, 0xf
s_cmp_eq_u32 s4, s4
s_cmov_b64 exec, s[20:21]
v_mov_b32_e32 v13, 1
s_mov_b64 exec, 0xff00
s_cmp_lg_u32 s4, s4
s_cmov_b64 exec, -1
v_mov_b32_e32 v14, 1
s_mov_b64 exec, 0
s_cmp_eq_u32 s4, s4
s_cmov_b64 exec, -16
v_mov_b32_e32 v15, 1
s_mov_b64 exec, -1
s_cmp_lg_u32 s4, s4
s_cmov_b64 exec, 0
v_mov_b32_e32 v16, 1
s_mov_b64 exec, 0xffff
s_mov_b32 s20, 0
s_mov_b32 s21, 0x80000001
s_cmp_eq_u32 s4, s4
s_cmov_b64 exec, s[20:21]
v_mov_b32_e32 v17, 1
s_mov_b64 exec, -1
v_cmp_ne_u32_e32 vcc, 0, v10
s_bcnt1_i32_b64 s20, vcc
s_ff1_i32_b64 s21, vcc
s_lshl_b32 s21, s21, 8
s_or_b32 s20, s20, s21
v_cmp_ne_u32_e32 vcc, 0, v11
s_bcnt1_i32_b64 s22, vcc
s_ff1_i32_b64 s23, vcc
s_lshl_b32 s23, s23, 8
s_or_b32 s22, s22, s23
v_cmp_ne_u32_e32 vcc, 0, v12
s_bcnt1_i32_b64 s24, vcc
s_ff1_i32_b64 s25, vcc
s_lshl_b32 s25, s25, 8
s_or_b32 s24, s24, s25
v_cmp_ne_u32_e32 vcc, 0, v13
s_bcnt1_i32_b64 s26, vcc
s_ff1_i32_b64 s27, vcc
s_lshl_b32 s27, s27, 8
s_or_b32 s26, s26, s27
v_cmp_ne_u32_e32 vcc, 0, v14
s_bcnt1_i32_b64 s28, vcc
s_ff1_i32_b64 s29, vcc
s_lshl_b32 s29, s29, 8
s_or_b32 s28, s28, s29
v_cmp_ne_u32_e32 vcc, 0, v15
s_bcnt1_i32_b64 s30, vcc
s_ff1_i32_b64 s31, vcc
s_lshl_b32 s31, s31, 8
s_or_b32 s30, s30, s31
v_cmp_ne_u32_e32 vcc, 0, v16
s_bcnt1_i32_b64 s32, vcc
s_ff1_i32_b64 s33, vcc
s_lshl_b32 s33, s33, 8
s_or_b32 s32, s32, s33
v_cmp_ne_u32_e32 vcc, 0, v17
s_bcnt1_i32_b64 s34, vcc
s_ff1_i32_b64 s35, vcc
s_lshl_b32 s35, s35, 8
s_or_b32 s34, s34, s35
v_cmpx_eq_u32_e32 0, v0
v_mov_b32_e32 v1, s20
v_mov_b32_e32 v2, 0
buffer_store_dword v1, v2, s[0:3], 0 offen
v_mov_b32_e32 v1, s22
v_mov_b32_e32 v2, 4
buffer_store_dword v1, v2, s[0:3], 0 offen
v_mov_b32_e32 v1, s24
v_mov_b32_e32 v2, 8
buffer_store_dword v1, v2, s[0:3], 0 offen
v_mov_b32_e32 v1, s26
v_mov_b32_e32 v2, 12
buffer_store_dword v1, v2, s[0:3], 0 offen
v_mov_b32_e32 v1, s28
v_mov_b32_e32 v2, 16
buffer_store_dword v1, v2, s[0:3], 0 offen
v_mov_b32_e32 v1, s30
v_mov_b32_e32 v2, 20
buffer_store_dword v1, v2, s[0:3], 0 offen
v_mov_b32_e32 v1, s32
v_mov_b32_e32 v2, 24
buffer_store_dword v1, v2, s[0:3], 0 offen
v_mov_b32_e32 v1, s34
v_mov_b32_e32 v2, 28
buffer_store_dword v1, v2, s[0:3], 0 offen
s_endpgm
