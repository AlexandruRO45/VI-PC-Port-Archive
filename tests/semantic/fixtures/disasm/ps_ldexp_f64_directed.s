; ps_ldexp_f64_directed  hand-written synthetic fixture
	v_mov_b32 v0, -1
	v_mov_b32 v1, 0x7fefffff
	v_mov_b32 v2, 1
	s_round_mode 0x4
	v_ldexp_f64 v[8:9], -v[0:1], v2
	s_round_mode 0x8
	v_ldexp_f64 v[10:11], -v[0:1], v2
	exp mrt0 v8, v9, v10, v11 done vm
	s_endpgm
