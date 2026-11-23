; Wave64 emulation (--wave64-emulate): a scalar branch from inside a folded lane-local EXECZ region straight to
; the join must skip the rest of that region for the whole wave. Mirrors the four CPU cases in
; tests/semantic/test_wave64_emulate_opt.py::test_folded_region_scalar_branch_to_join_skips_the_rest.
;
; v1 = lane (0..63 via v_mbcnt), v3 = (5 * lane) & 15; lanes 3, 6, 9, 19, 22, 25, 35, 38, 41, 51, 54, 57
; (v3 > 12) are the region's active lanes. Each case starts with v4 = 1 and SCC from s11, then:
;   s_and_saveexec_b64 s[8:9], vcc / s_cbranch_execz 5 -> join / s_cmp / s_cbranch_scc? 3 -> join /
;   v4 += 21 / v4 += 100 / s_or_b64 exec, exec, s[8:9] (join).
; Cases 1 and 3 take the scalar exit for every wave (s11 = 5): v4 stays 1. Cases 2 and 4 fall through
; (s11 = 40): the active lanes get 1 + 21 + 100 = 122. The other half of the wave keeps v4 = 1.
; The store is outside the fold; lane 0 (inactive) and lane 3 (active) are written per case.
;
; Stores: out[0] case1 lane0, out[1] case1 lane3, out[2] case2 lane0, out[3] case2 lane3,
;   out[4] case3 lane0, out[5] case3 lane3, out[6] case4 lane0, out[7] case4 lane3.
; Run: vk_compute_test cs_w64x_fold_scc_exit.spv 1 dump --require-wave64
;   -> out[0..7] = 1 1 1 122 1 1 1 122
; mattr: +wavefrontsize64,-wavefrontsize32
v_mbcnt_lo_u32_b32 v1, -1, 0
v_mbcnt_hi_u32_b32 v1, -1, v1
v_mul_lo_u32 v3, v1, 5
v_and_b32_e32 v3, 15, v3

; Case 1: s_cmp_ge_u32 s11, 33 with s11 = 5 -> SCC = 0 -> s_cbranch_scc0 taken
s_mov_b64 exec, -1
v_mov_b32_e32 v4, 1
s_mov_b32 s11, 5
v_cmp_lt_u32_e32 vcc, 12, v3
s_and_saveexec_b64 s[8:9], vcc
s_cbranch_execz 5
s_cmp_ge_u32 s11, 33
s_cbranch_scc0 3
v_add_nc_u32_e32 v4, 21, v4
v_add_nc_u32_e32 v4, 100, v4
s_or_b64 exec, exec, s[8:9]
v_cmpx_eq_u32_e32 0, v1
v_mov_b32_e32 v20, v4
v_mov_b32_e32 v21, 0
buffer_store_dword v20, v21, s[0:3], 0 offen
s_mov_b64 exec, -1
v_cmpx_eq_u32_e32 3, v1
v_mov_b32_e32 v20, v4
v_mov_b32_e32 v21, 4
buffer_store_dword v20, v21, s[0:3], 0 offen

; Case 2: s_cmp_ge_u32 s11, 33 with s11 = 40 -> SCC = 1 -> s_cbranch_scc0 not taken
s_mov_b64 exec, -1
v_mov_b32_e32 v4, 1
s_mov_b32 s11, 40
v_cmp_lt_u32_e32 vcc, 12, v3
s_and_saveexec_b64 s[8:9], vcc
s_cbranch_execz 5
s_cmp_ge_u32 s11, 33
s_cbranch_scc0 3
v_add_nc_u32_e32 v4, 21, v4
v_add_nc_u32_e32 v4, 100, v4
s_or_b64 exec, exec, s[8:9]
v_cmpx_eq_u32_e32 0, v1
v_mov_b32_e32 v20, v4
v_mov_b32_e32 v21, 8
buffer_store_dword v20, v21, s[0:3], 0 offen
s_mov_b64 exec, -1
v_cmpx_eq_u32_e32 3, v1
v_mov_b32_e32 v20, v4
v_mov_b32_e32 v21, 12
buffer_store_dword v20, v21, s[0:3], 0 offen

; Case 3: s_cmp_lt_u32 s11, 33 with s11 = 5 -> SCC = 1 -> s_cbranch_scc1 taken
s_mov_b64 exec, -1
v_mov_b32_e32 v4, 1
s_mov_b32 s11, 5
v_cmp_lt_u32_e32 vcc, 12, v3
s_and_saveexec_b64 s[8:9], vcc
s_cbranch_execz 5
s_cmp_lt_u32 s11, 33
s_cbranch_scc1 3
v_add_nc_u32_e32 v4, 21, v4
v_add_nc_u32_e32 v4, 100, v4
s_or_b64 exec, exec, s[8:9]
v_cmpx_eq_u32_e32 0, v1
v_mov_b32_e32 v20, v4
v_mov_b32_e32 v21, 16
buffer_store_dword v20, v21, s[0:3], 0 offen
s_mov_b64 exec, -1
v_cmpx_eq_u32_e32 3, v1
v_mov_b32_e32 v20, v4
v_mov_b32_e32 v21, 20
buffer_store_dword v20, v21, s[0:3], 0 offen

; Case 4: s_cmp_lt_u32 s11, 33 with s11 = 40 -> SCC = 0 -> s_cbranch_scc1 not taken
s_mov_b64 exec, -1
v_mov_b32_e32 v4, 1
s_mov_b32 s11, 40
v_cmp_lt_u32_e32 vcc, 12, v3
s_and_saveexec_b64 s[8:9], vcc
s_cbranch_execz 5
s_cmp_lt_u32 s11, 33
s_cbranch_scc1 3
v_add_nc_u32_e32 v4, 21, v4
v_add_nc_u32_e32 v4, 100, v4
s_or_b64 exec, exec, s[8:9]
v_cmpx_eq_u32_e32 0, v1
v_mov_b32_e32 v20, v4
v_mov_b32_e32 v21, 24
buffer_store_dword v20, v21, s[0:3], 0 offen
s_mov_b64 exec, -1
v_cmpx_eq_u32_e32 3, v1
v_mov_b32_e32 v20, v4
v_mov_b32_e32 v21, 28
buffer_store_dword v20, v21, s[0:3], 0 offen
s_endpgm
