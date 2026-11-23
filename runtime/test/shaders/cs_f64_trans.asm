; v_rcp_f64 / v_rsq_f64 / v_sqrt_f64 on binary64 operands: rcp(4.0) = 0.25 and rsq(4.0) = 0.5 exactly.
; sqrt((2^26 + 1)^2) = 2^26 + 1 = 67108865 exactly, and the operand 2^52 + 2^27 + 1 needs 53 significand bits,
; so it is not a binary32 value: a path that rounded the operand to f32 first would not give 67108865.
; The last case carries the VOP3 neg input modifier (the operand is -4.0, so the double fed to sqrt is +4.0)
; and the div:2 output modifier: sqrt(4.0) / 2 = 1.0.
; Each result is stored low word first. Expected out[0..7] = 0 1070596096 0 1071644672 67108864 1099956224 0 1072693248.
; group: 1,1,1
v_mov_b32 v0, 0
v_mov_b32 v1, 0x40100000
v_mov_b32 v2, 0
v_mov_b32 v3, 0x40100000
v_mov_b32 v4, 0x08000001
v_mov_b32 v5, 0x43300000
v_mov_b32 v6, 0
v_mov_b32 v7, 0xc0100000
v_rcp_f64_e32 v[8:9], v[0:1]
v_rsq_f64_e32 v[10:11], v[2:3]
v_sqrt_f64_e32 v[12:13], v[4:5]
v_sqrt_f64_e64 v[14:15], -v[6:7] div:2
v_mov_b32 v16, 0
buffer_store_dwordx2 v[8:9], v16, s[0:3], 0 offen
v_mov_b32 v16, 8
buffer_store_dwordx2 v[10:11], v16, s[0:3], 0 offen
v_mov_b32 v16, 16
buffer_store_dwordx2 v[12:13], v16, s[0:3], 0 offen
v_mov_b32 v16, 24
buffer_store_dwordx2 v[14:15], v16, s[0:3], 0 offen
s_endpgm
