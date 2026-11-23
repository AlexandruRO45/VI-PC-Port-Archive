; SHADER-10 tier A: EXECZ/EXECNZ diamonds with local scalar arithmetic and unchanged EXEC in both sides.
; Only pixel (5,7) is active in each diamond. An empty half can take the other side, but writes no VGPR.
; Both sides join before EXEC is restored; no scalar destination is read outside the sides.
; r0 = x+5 for (5,7), otherwise 77; r1 = y+7 for (5,7), otherwise 88; r2=r3=0.
; Check with w64_graphics.py at required subgroup size 32 and, where available, 64.
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
v_xor_b32_e32 v4, 5, v2
v_xor_b32_e32 v5, 7, v3
v_or_b32_e32 v4, v4, v5
v_mov_b32_e32 v6, 77
v_mov_b32_e32 v7, 88
v_cmp_eq_u32_e32 vcc, 0, v4
s_and_saveexec_b64 s[12:13], vcc
s_cbranch_execz 4
s_mov_b32 s5, 3
s_add_u32 s5, s5, 2
v_add_nc_u32_e32 v6, s5, v2
s_branch 3
s_mov_b32 s5, 9
s_mul_i32 s5, s5, 2
v_add_nc_u32_e32 v6, s5, v2
s_mov_b64 exec, s[10:11]
v_cmp_eq_u32_e32 vcc, 0, v4
s_and_saveexec_b64 s[12:13], vcc
s_cbranch_execnz 4
s_mov_b32 s6, 9
s_mul_i32 s6, s6, 2
v_add_nc_u32_e32 v7, s6, v3
s_branch 3
s_mov_b32 s6, 5
s_add_u32 s6, s6, 2
v_add_nc_u32_e32 v7, s6, v3
s_mov_b64 exec, s[10:11]
v_mov_b32_e32 v8, 0
exp mrt0 v6, v7, v8, v8 done vm
s_endpgm
