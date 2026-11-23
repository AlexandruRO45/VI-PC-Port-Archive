; SHADER-10 tier A: votes on a uniform compare made under a narrowed EXEC (docs/design.md, "Wave64 graphics: partition
; invariance", the narrowed uniform vote). s4 is a user SGPR (the test sets it to 9 with VKH_SGPRS=0,0,0,0,9), copied to
; a VGPR after the narrowing, so every active lane holds 9 and a v_cmp on it gives every active lane or none. A 32-lane
; half with an active lane votes as the whole wave does; a half with none votes no and runs the no side with every lane
; off, which writes nothing. The narrowings take whole 4-pixel rows / columns, so wherever the hardware packs the
; pixels, some halves lose every lane while the other half of their wave keeps some. x = floor(u * 64), y = floor(v * 64)
; on the 64x64 target of vs_ids.
;  (1) EXEC narrowed to (y & 4) == 0, then s_cbranch_vccz if/else on 5 < s4 (yes): the yes side sets v6 = x + 7, the
;      no side v6 = x + 5; EXEC restored. r0 = x + 7 where (y & 4) == 0, else 0.
;  (2) EXEC narrowed to (x & 4) != 0, then the LLVM form v_cmp_e64 s[14:15] + s_and_b64 vcc, exec, s[14:15] +
;      s_cbranch_vccnz on 50 < s4 (no): the no side sets v7 = 3, the yes side has a nested if (y < 32, saveexec and
;      restore of the mask it saved) that sets v7 = 4; EXEC restored. r1 = 3 where (x & 4) != 0, else 0.
;  r2 = s4 = 9, r3 = 0.
; Run: VKH_SGPRS=0,0,0,0,9 vk_harness vs_ids.spv ps_w64_a_uniform_vote_narrowed.spv out.raw with VKH_SUBGROUP_SIZE=64
; and 32: equal, and equal to the expected words.
; mattr: +wavefrontsize64,-wavefrontsize32
v_interp_p1_f32_e32 v2, v0, attr0.x
v_interp_p2_f32_e32 v2, v1, attr0.x
v_interp_p1_f32_e32 v3, v0, attr0.y
v_interp_p2_f32_e32 v3, v1, attr0.y
v_mul_f32_e32 v2, 0x42800000, v2
v_mul_f32_e32 v3, 0x42800000, v3
v_cvt_u32_f32_e32 v2, v2
v_cvt_u32_f32_e32 v3, v3
s_mov_b64 s[10:11], exec
v_mov_b32_e32 v6, 0
v_mov_b32_e32 v7, 0
v_mov_b32_e32 v9, 0
v_and_b32_e32 v4, 4, v3
v_cmp_eq_u32_e32 vcc, 0, v4
s_and_saveexec_b64 s[12:13], vcc
v_mov_b32_e32 v5, s4
v_cmp_lt_u32_e32 vcc, 5, v5
s_cbranch_vccz 2
v_add_nc_u32_e32 v6, 7, v2
s_branch 1
v_add_nc_u32_e32 v6, 5, v2
s_mov_b64 exec, s[12:13]
v_and_b32_e32 v4, 4, v2
v_cmp_ne_u32_e32 vcc, 0, v4
s_and_saveexec_b64 s[12:13], vcc
v_mov_b32_e32 v5, s4
v_cmp_lt_u32_e64 s[14:15], 50, v5
s_and_b64 vcc, exec, s[14:15]
s_cbranch_vccnz 2
v_mov_b32_e32 v7, 3
s_branch 4
v_cmp_gt_u32_e32 vcc, 32, v3
s_and_saveexec_b64 s[16:17], vcc
v_mov_b32_e32 v7, 4
s_or_b64 exec, exec, s[16:17]
s_mov_b64 exec, s[12:13]
v_mov_b32_e32 v8, s4
exp mrt0 v6, v7, v8, v9 done vm
s_endpgm
