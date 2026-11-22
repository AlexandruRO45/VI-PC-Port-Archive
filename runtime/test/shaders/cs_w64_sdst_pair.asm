; A wave64 VALU lane mask whose SDST is printed as one SGPR (v_cmp_gt_u32_e64 s10, ...) writes the pair s[10:11].
; The corpus is disassembled for gfx1013 without +wavefrontsize64, so every wave64 v_cmp_*_e64 / carry-out SDST reads
; like this; no mattr line here on purpose (wave64 is inferred from exec_hi). s[10:11] = lanes 0-39,
; s[12:13] = lanes 21-63, s[14:15] = their AND = lanes 21-39 (19 lanes, so SCC = 1 for the whole wave).
; Each lane computes 7 * SCC + (s_bcnt1_i32_b64 s[14:15] << 8) + (its own bit of s[14:15] << 16); lanes 0, 8, ... 56
; store it. Taking s10 as one register made the AND 64-bit data: SCC became each lane's own bit and the count
; counted data bits (out = 0 / 73735 on main before this test).
; Assemble (as scripts/run_gpu_tests.py / build.py do): llvm-mc -triple=amdgcn--amdpal -mcpu=gfx1013 -show-encoding
; Run: VKC_SUBGROUP_SIZE=64 vk_compute_test cs_w64_sdst_pair.spv 1 dump (or translated with --wave64-emulate)
; Expect: out[0..7] = 4871 4871 4871 70407 70407 4871 4871 4871
; group: 64,1,1
s_mov_b32 exec_hi, -1
v_mov_b32_e32 v5, 0
v_cmp_gt_u32_e64 s10, 40, v0
v_cmp_lt_u32_e64 s12, 20, v0
s_and_b64 s[14:15], s[10:11], s[12:13]
s_cbranch_scc0 1
v_mov_b32_e32 v5, 7
s_bcnt1_i32_b64 s6, s[14:15]
v_mov_b32_e32 v8, s6
v_lshlrev_b32_e32 v8, 8, v8
v_add_nc_u32_e32 v5, v5, v8
v_cndmask_b32_e64 v9, 0, 1, s14
v_lshlrev_b32_e32 v9, 16, v9
v_add_nc_u32_e32 v5, v5, v9
v_and_b32_e32 v6, 7, v0
v_lshrrev_b32_e32 v7, 3, v0
v_lshlrev_b32_e32 v2, 2, v7
v_cmpx_eq_u32_e32 0, v6
buffer_store_dword v5, v2, s[0:3], 0 offen
s_endpgm
