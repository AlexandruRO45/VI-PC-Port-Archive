; v_fract_f16 of a run-time half in s0 (a NaN, an infinity or a number). The other destination half must survive.
v_mov_b32_e32 v6, s0
v_mov_b32_e32 v0, 0x13570000
v_fract_f16_e32 v0, v6
v_mov_b32_e32 v1, 0x24680000
v_fract_f16_e64 v1, -v6
v_mov_b32_e32 v2, 0x369a0000
v_fract_f16_e64 v2, |v6|
v_mov_b32_e32 v3, 0x1234ffff
v_fract_f16_e64 v3, -v6 clamp
exp mrt0 v0, v1, v2, v3 done vm
s_endpgm
