; SHADER-10 tier A: votes on a compare that is uniform in the active lanes of a narrowed EXEC (docs/design.md, "Wave64
; graphics: partition invariance"). s4 is a user SGPR (the test sets it to 9 with VKH_SGPRS=0,0,0,0,9), copied to a
; VGPR. EXEC is narrowed to the pixels with x < 20, so a 32-lane half can have no active lane where the other half of
; the wave has some. Such a half has an empty vote mask and takes the no side of each branch, whatever the wave does,
; with all its lanes off; the two sides meet again and no side makes EXEC wider, so no lane work can differ.
; x = floor(u * 64) on the 64x64 target of vs_ids.
;  (1) s_cbranch_vccz if/else on 5 < s4 (yes): the yes side adds 7 to v6 = x, the no side 5. r0 = x < 20 ? x + 7 : x.
;  (2) the LLVM form v_cmp_e64 s[14:15] + s_and_b64 vcc, exec, s[14:15] + s_cbranch_vccnz on 50 < s4 (no): the no
;      side has an if of its own (odd x: v7 = 3), the yes side sets v7 = 4. r1 = x < 20 and x odd ? 3 : 0.
;  (3) s_cbranch_vccz on 5 > s4 (no): each side sets s20 and uses it in the side only (40 on the yes side, 50 on the
;      no side). r2 = x < 20 ? 50 : 0.
;  r3 = x + 1, written after EXEC is restored.
; Run: VKH_SGPRS=0,0,0,0,9 vk_harness vs_ids.spv ps_w64_a_uniform_vote_narrowed.spv out.raw with VKH_SUBGROUP_SIZE=64
; and 32: equal, and equal to the expected words.
; mattr: +wavefrontsize64,-wavefrontsize32
v_interp_p1_f32_e32 v2, v0, attr0.x
v_interp_p2_f32_e32 v2, v1, attr0.x
v_mul_f32_e32 v2, 0x42800000, v2
v_cvt_u32_f32_e32 v2, v2
s_mov_b64 s[10:11], exec
v_mov_b32_e32 v5, s4
v_mov_b32_e32 v6, v2
v_mov_b32_e32 v7, 0
v_mov_b32_e32 v8, 0
v_cmp_gt_u32_e32 vcc, 20, v2
s_and_saveexec_b64 s[12:13], vcc
v_cmp_lt_u32_e32 vcc, 5, v5
s_cbranch_vccz 2
v_add_nc_u32_e32 v6, 7, v6
s_branch 1
v_add_nc_u32_e32 v6, 5, v6
v_cmp_lt_u32_e64 s[14:15], 50, v5
s_and_b64 vcc, exec, s[14:15]
s_cbranch_vccnz 6
v_and_b32_e32 v4, 1, v2
v_cmp_eq_u32_e32 vcc, 1, v4
s_and_saveexec_b64 s[16:17], vcc
v_mov_b32_e32 v7, 3
s_or_b64 exec, exec, s[16:17]
s_branch 1
v_mov_b32_e32 v7, 4
v_cmp_gt_u32_e32 vcc, 5, v5
s_cbranch_vccz 3
s_mov_b32 s20, 40
v_add_nc_u32_e32 v8, s20, v8
s_branch 2
s_mov_b32 s20, 50
v_add_nc_u32_e32 v8, s20, v8
s_or_b64 exec, exec, s[12:13]
v_add_nc_u32_e32 v9, 1, v2
exp mrt0 v6, v7, v8, v9 done vm
s_endpgm
