; --vs-compute GPU fixture (issue #94; d3d12_vs_compute_test): mrt0 = the provoking vertex's param0.x in all four
; channels (v_interp_mov, so the value is exactly what the vertex exported).
; fssm-at: 0x10
; fssm-data: 0x70637376 0x00000001
v_interp_mov_f32_e32 v2, p0, attr0.x
exp mrt0 v2, v2, v2, v2 done vm
s_endpgm
