; v_cvt_pkrtz_f16_f32_e64 with VOP3 output modifiers: each f16 half is narrowed toward zero, then scaled by the
; omod, then clamped to [0, 1], as for v_cvt_f16_f32 (docs/design.md, "v_cvt_pkrtz_f16_f32 output modifiers").
; Every word changes when its modifier is dropped, and v3's low half changes when clamp runs before mul:4.
	v_mov_b32 v4, 1.5
	v_mov_b32 v5, -2.0
	v_mov_b32 v6, 0x40400000
	v_mov_b32 v7, 0x3f400000
	v_mov_b32 v8, -1.0
	v_mov_b32 v9, 0x3ec00000
	v_mov_b32 v10, 0x3e000000
	v_cvt_pkrtz_f16_f32_e64 v0, v4, v5 clamp
	v_cvt_pkrtz_f16_f32_e64 v1, v6, v7 mul:2
	v_cvt_pkrtz_f16_f32_e64 v2, v6, v8 div:2
	v_cvt_pkrtz_f16_f32_e64 v3, v9, v10 clamp mul:4
	exp mrt0 v0, v1, v2, v3 done vm
	s_endpgm
