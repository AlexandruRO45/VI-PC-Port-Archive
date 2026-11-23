; mrt0 words: 3f800000 3f000000 3f800000 3e800000
v_mov_b32 v0, 1e0
v_mov_b32 v1, .5
v_mov_b32 v2, 1.E+0
v_mov_b32 v3, 2.5E-1
exp mrt0 v0, v1, v2, v3 done vm
s_endpgm
