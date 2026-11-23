; v_fract_f16 of a NaN with an output modifier or |x| clamp (run-time half in s0), and of constant NaNs (the clamp
; form was refused before: FClamp of a NaN is undefined). The other destination half must survive.
v_mov_b32_e32 v6, s0
v_mov_b32_e32 v7, 0x7c01
v_mov_b32_e32 v8, 0xfe11
v_mov_b32_e32 v0, 0x13570000
v_fract_f16_e64 v0, v6 mul:2
v_mov_b32_e32 v1, 0x24680000
v_fract_f16_e64 v1, |v6| clamp
v_mov_b32_e32 v2, 0x369a0000
v_fract_f16_e64 v2, v7 clamp
v_mov_b32_e32 v3, 0x12340000
v_fract_f16_e64 v3, v8
exp mrt0 v0, v1, v2, v3 done vm
s_endpgm
