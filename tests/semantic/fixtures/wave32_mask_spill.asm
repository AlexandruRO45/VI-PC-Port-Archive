; The X position of odd-numbered vertices is 1.0, even vertices 0.0.
; Y is 0.0 for every vertex after the mask slot is overwritten with zero.
	v_and_b32 v4, 1, v0
	v_cmp_ne_u32_e64 s8, 0, v4
	v_mov_b32 v6, 0
	v_writelane_b32 v6, s8, 40
	v_readlane_b32 s9, v6, 8
	v_cndmask_b32_e64 v7, 0, 1.0, s9
	v_writelane_b32 v6, s8, 8
	v_writelane_b32 v6, 0, 40
	v_readlane_b32 s10, v6, 8
	v_cndmask_b32_e64 v8, 0, 1.0, s10
	v_mov_b32 v9, 0
	v_mov_b32 v10, 1.0
	exp pos0 v7, v8, v9, v10 done
	s_endpgm
