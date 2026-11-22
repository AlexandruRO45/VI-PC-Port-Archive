; SHADER-10 tier A, TEST-363: which lanes a wave64 pixel shader's EXEC holds at entry, read through whole-quad mode.
; On RDNA2 a pixel wave starts with EXEC = the covered pixels; helper lanes of partly covered 2x2 quads only join at
; s_wqm_b64 (the LLVM prologue saves the entry EXEC as the live mask first and restores it before exports and stores).
; Here every lane takes its bit of the entry EXEC (1 in a covered pixel, 0 in a helper), whole-quad mode lets the
; helpers run, and two DPP quad_perm adds sum the bit over the 2x2 quad. A quad never crosses lane 32, so the result is
; the same on one 64-lane wave and on two 32-lane halves.
; x = floor(u * 64), y = floor(v * 64) as in the other ps_w64_* fixtures. r0 = the number of covered pixels in the
; pixel's 2x2 quad, r1 = x, r2 = y, r3 = 0. Drawn with vs_ids (every quad covered) r0 is 4; drawn with vs_tri_partial,
; r0 is 1, 2 or 3 in the quads its long edge cuts. An entry EXEC that also holds the helper lanes gives 4 there.
; Run: vk_harness vs_ids.spv ps_w64_a_quad_live.spv out.raw, or vs_tri_partial.spv; w64_graphics.py checks both.
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
s_wqm_b64 exec, exec
v_cndmask_b32_e64 v4, 0, 1, s[10:11]
v_add_nc_u32_dpp v5, v4, v4 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf
v_add_nc_u32_dpp v6, v5, v5 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf
s_mov_b64 exec, s[10:11]
v_mov_b32_e32 v7, 0
exp mrt0 v6, v2, v3, v7 done vm
s_endpgm
