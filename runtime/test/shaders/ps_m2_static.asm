; M2 static sampler: s[0:1] holds the SRV table, s[2:5] the four S# words by value.
; No sampler table is bound for this draw.
; fssm-at: 0x10
; fssm-data: 0x324d5350 0x53544154
s_load_dwordx8 s[8:15], s[0:1], 0x0
s_mov_b64 s[16:17], s[2:3]
s_mov_b64 s[18:19], s[4:5]
s_waitcnt lgkmcnt(0)
v_interp_p1_f32_e32 v2, v0, attr0.x
v_interp_p2_f32_e32 v2, v1, attr0.x
v_interp_p1_f32_e32 v3, v0, attr0.y
v_interp_p2_f32_e32 v3, v1, attr0.y
image_sample v[4:7], v[2:3], s[8:15], s[16:19] dmask:0xf dim:SQ_RSRC_IMG_2D
s_waitcnt vmcnt(0)
exp mrt0 v4, v5, v6, v7 done vm
s_endpgm
