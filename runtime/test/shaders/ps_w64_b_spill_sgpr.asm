s_mov_b64 s[10:11], exec
s_mov_b32 s4, 123
s_mov_b32 s5, 77
s_mov_b32 s8, 40
v_writelane_b32 v30, s4, s8
s_movk_i32 s8, -1
v_writelane_b32 v30, s5, s8
s_mov_b32 s8, 104
v_readlane_b32 s21, v30, s8
s_mov_b32 s8, 63
v_readlane_b32 s22, v30, s8
s_add_u32 s23, s21, s22
v_mov_b32_e32 v6, s21
v_mov_b32_e32 v7, s22
v_mov_b32_e32 v8, s23
v_mov_b32_e32 v9, 0
exp mrt0 v6, v7, v8, v9 done vm
s_endpgm
