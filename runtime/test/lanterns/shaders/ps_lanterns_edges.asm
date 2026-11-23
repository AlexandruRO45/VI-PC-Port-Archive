; Lanterns shared-edge pixel shader (stage 1): wave64, the title's layout as vs_lanterns_edges: the table at s[8:9],
; whose slot 1 (+0x20) holds a V# over one float the pixel shader writes as alpha (1.0). Red and green are param0's,
; read flat (the provoking vertex's), blue 0.
; stage: ps
; wave: 64
; input: flat
s_load_dwordx4 s[12:15], s[8:9], 0x20
v_interp_mov_f32 v2, p0, attr0.x
v_interp_mov_f32 v3, p0, attr0.y
v_mov_b32_e32 v4, 0
s_waitcnt lgkmcnt(0)
s_buffer_load_dword s16, s[12:15], 0x0
s_waitcnt lgkmcnt(0)
v_mov_b32_e32 v5, s16
exp mrt0 v2, v3, v4, v5 done vm
s_endpgm