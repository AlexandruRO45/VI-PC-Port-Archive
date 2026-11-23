; Lanterns pixel shader for the solids, wave64 (stage 1): ps_lanterns_lit.asm's code for 64-lane waves (10,974 of
; the title's 12,101 pixel shaders are wave64), drawn after vs_lanterns_w64.
; stage: ps
; wave: 64
; input: flat
; input: interp
v_interp_mov_f32 v2, p0, attr0.x
v_interp_mov_f32 v3, p0, attr0.y
v_interp_mov_f32 v4, p0, attr0.z
v_interp_mov_f32 v5, p0, attr0.w
exp mrt0 v2, v3, v4, v5 done vm
s_endpgm
