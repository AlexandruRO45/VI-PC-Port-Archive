; A wave64 lane mask whose two halves meet in one VGPR, computed under a partial EXEC (#210, SHADER-402).
; s[20:21] = lanes 0-35: s20 = 0xffffffff, s21 = 0xf. While EXEC is lanes 0-3, v_xor_b32 v10, s20, s21 gives the
; word 0xfffffff0 in lanes 0-3; then every lane stores. Lanes 4-63 keep 0xdeadbeef.
; The s_mov_b32 exec_hi, -1 marks the shader as wave64 (the translator infers the wave size from it).
; Assemble (as scripts/run_gpu_tests.py / build.py do): llvm-mc -triple=amdgcn--amdpal -mcpu=gfx1013 -mattr=+wavefrontsize64 -show-encoding
; mattr: +wavefrontsize64,-wavefrontsize32
; Run: VKC_SUBGROUP_SIZE=64 vk_compute_test cs_wm_vgpr_halves.spv 1 dump (or translated with --wave64-emulate)
; Expect: out[0..7] = 0xfffffff0 x4, 0xdeadbeef x4
s_mov_b32 exec_hi, -1
v_mov_b32_e32 v10, 0xdeadbeef
v_cmp_gt_u32_e64 s[20:21], 36, v0
s_mov_b64 exec, 15
v_xor_b32_e64 v10, s20, s21
s_mov_b64 exec, -1
v_lshlrev_b32_e32 v3, 2, v0
buffer_store_dword v10, v3, s[0:3], 0 offen
s_endpgm
