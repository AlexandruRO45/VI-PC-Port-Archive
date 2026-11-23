; Wave64: a scalar branch from inside a lane-local s_cbranch_execz region straight to its join (docs/design.md,
; "Wave64 emulation: scalar exits from a folded EXECZ region"). Under --wave64-emulate the region is folded inline,
; and the s_cmp + s_cbranch_scc0/scc1 to the join must skip the rest of the region for the whole wave; before the
; fix the exit edge was dropped and the adds after the branch ran in every active lane.
; Lanes with (5 * lane) & 15 > 12 enter each region (lanes 3 and 35 do: 15; lane 40 does not: 200 & 15 = 8).
; Each region starts its VGPR at 1 and adds two inline constants after the scalar branch:
;   (A) s11 = 5,  s_cmp_ge_u32 s11, 33 -> SCC = 0, s_cbranch_scc0 taken     -> v4 = 1
;   (B) s11 = 40, s_cmp_lt_u32 s11, 33 -> SCC = 0, s_cbranch_scc1 not taken -> v5 = 1 + 2 + 3   = 6 inside
;   (C) s11 = 5,  s_cmp_lt_u32 s11, 33 -> SCC = 1, s_cbranch_scc1 taken     -> v6 = 1
;   (D) s11 = 40, s_cmp_ge_u32 s11, 33 -> SCC = 1, s_cbranch_scc0 not taken -> v7 = 1 + 30 + 33 = 64 inside
; A lane outside the region keeps 1 whatever the branch does.
; Stores: out[0] lane 3 v4, out[1] lane 35 v4, out[2] lane 3 v5, out[3] lane 35 v5, out[4] lane 35 v6,
;   out[5] lane 3 v7, out[6] lane 35 v7, out[7] lane 40 v7.
; Every instruction inside a region is one dword (inline constants only), so the branch offsets count instructions:
; s_cbranch_execz 4 and s_cbranch_scc* 2 land on the s_or_b64 that restores EXEC.
; Run: vk_compute_test cs_w64x_fold_scc_exit.spv 1 dump --require-wave64
;   -> out[0..7] = 1 1 6 6 1 64 64 1
; mattr: +wavefrontsize64,-wavefrontsize32
v_mul_lo_u32 v3, v0, 5
v_and_b32_e32 v3, 15, v3
v_mov_b32_e32 v4, 1
s_mov_b32 s11, 5
v_cmp_lt_u32_e32 vcc, 12, v3
s_and_saveexec_b64 s[8:9], vcc
s_cbranch_execz 4
s_cmp_ge_u32 s11, 33
s_cbranch_scc0 2
v_add_nc_u32_e32 v4, 21, v4
v_add_nc_u32_e32 v4, 60, v4
s_or_b64 exec, exec, s[8:9]
v_mov_b32_e32 v5, 1
s_mov_b32 s11, 40
v_cmp_lt_u32_e32 vcc, 12, v3
s_and_saveexec_b64 s[8:9], vcc
s_cbranch_execz 4
s_cmp_lt_u32 s11, 33
s_cbranch_scc1 2
v_add_nc_u32_e32 v5, 2, v5
v_add_nc_u32_e32 v5, 3, v5
s_or_b64 exec, exec, s[8:9]
v_mov_b32_e32 v6, 1
s_mov_b32 s11, 5
v_cmp_lt_u32_e32 vcc, 12, v3
s_and_saveexec_b64 s[8:9], vcc
s_cbranch_execz 4
s_cmp_lt_u32 s11, 33
s_cbranch_scc1 2
v_add_nc_u32_e32 v6, 7, v6
v_add_nc_u32_e32 v6, 8, v6
s_or_b64 exec, exec, s[8:9]
v_mov_b32_e32 v7, 1
s_mov_b32 s11, 40
v_cmp_lt_u32_e32 vcc, 12, v3
s_and_saveexec_b64 s[8:9], vcc
s_cbranch_execz 4
s_cmp_ge_u32 s11, 33
s_cbranch_scc0 2
v_add_nc_u32_e32 v7, 30, v7
v_add_nc_u32_e32 v7, 33, v7
s_or_b64 exec, exec, s[8:9]
s_mov_b64 exec, -1
v_cmpx_eq_u32_e32 3, v0
v_mov_b32_e32 v21, 0
buffer_store_dword v4, v21, s[0:3], 0 offen
s_mov_b64 exec, -1
v_cmpx_eq_u32_e32 35, v0
v_mov_b32_e32 v21, 4
buffer_store_dword v4, v21, s[0:3], 0 offen
s_mov_b64 exec, -1
v_cmpx_eq_u32_e32 3, v0
v_mov_b32_e32 v21, 8
buffer_store_dword v5, v21, s[0:3], 0 offen
s_mov_b64 exec, -1
v_cmpx_eq_u32_e32 35, v0
v_mov_b32_e32 v21, 12
buffer_store_dword v5, v21, s[0:3], 0 offen
s_mov_b64 exec, -1
v_cmpx_eq_u32_e32 35, v0
v_mov_b32_e32 v21, 16
buffer_store_dword v6, v21, s[0:3], 0 offen
s_mov_b64 exec, -1
v_cmpx_eq_u32_e32 3, v0
v_mov_b32_e32 v21, 20
buffer_store_dword v7, v21, s[0:3], 0 offen
s_mov_b64 exec, -1
v_cmpx_eq_u32_e32 35, v0
v_mov_b32_e32 v21, 24
buffer_store_dword v7, v21, s[0:3], 0 offen
s_mov_b64 exec, -1
v_cmpx_eq_u32_e32 40, v0
v_mov_b32_e32 v21, 28
buffer_store_dword v7, v21, s[0:3], 0 offen
s_endpgm
