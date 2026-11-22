; TEST-593: the src_scc operand (SCC read as a 0/1 source, as FidelityFX FSR4's FSPX programs use it) on the GPU
; (docs/design.md, "Scalar/buffer memory"; CPU-checked by tests/semantic/test_fspx_rules.py). Words, in out[0..3]:
;   0 v_add_nc_u32 src_scc + 20 after s_cmp_eq_u32 5, 5 (SCC = 1): 21
;   1 the same after s_add_u32 s7, 10, src_scc, whose carry clears SCC: 20
;   2 s_mov_b32 s6, src_scc with SCC = 1 (s_mov does not touch SCC): 1
;   3 s_add_u32 s7, 10, src_scc with SCC = 1: 11
; One thread, no wave64 requirement.
; Run: vk_compute_test cs_fspx_src_scc.spv 1 dump -> out[0..3] = 21 20 1 11.
v_cmpx_eq_u32_e32 0, v0
s_cmp_eq_u32 5, 5
v_mov_b32 v1, 20
v_add_nc_u32_e32 v2, src_scc, v1
s_mov_b32 s6, src_scc
s_add_u32 s7, 10, src_scc
v_add_nc_u32_e32 v3, src_scc, v1
v_mov_b32 v4, s6
v_mov_b32 v5, s7
buffer_store_dwordx4 v[2:5], off, s[0:3], 0
s_endpgm
