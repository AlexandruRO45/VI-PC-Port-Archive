v_mov_b32_e32 v4, s0
v_mov_b32_e32 v5, s1
v_mov_b32_e32 v6, s2
v_mullit_f32 v0, v4, v5, v6
; Exercise source sign modifiers and an inline light-off gate.
v_mullit_f32 v1, -v4, |v5|, v6
v_mullit_f32 v2, 0.5, v4, 0
; Turn NaNs into finite floats before export so their exact result bits are observable.
v_xor_b32_e32 v3, 0x40000000, v0
exp mrt0 v0, v1, v2, v3 done vm
s_endpgm
