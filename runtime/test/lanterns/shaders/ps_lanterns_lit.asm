; Lanterns pixel shader for the solids (stage 1): the face colour, read flat (v_interp_mov_f32), straight to mrt0.
; Packed by ../fssm_pack.py; the FSSM input table declares both of vs_lanterns.asm's parameters, as the title's
; tables do: position 0 read flat (read method 1), position 1 interpolated (read method 2) but unread here.
; stage: ps
; wave: 32
; input: flat
; input: interp
v_interp_mov_f32 v2, p0, attr0.x
v_interp_mov_f32 v3, p0, attr0.y
v_interp_mov_f32 v4, p0, attr0.z
v_interp_mov_f32 v5, p0, attr0.w
exp mrt0 v2, v3, v4, v5 done vm
s_endpgm
