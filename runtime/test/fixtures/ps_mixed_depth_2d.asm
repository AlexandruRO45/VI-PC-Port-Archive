; vk_bindless_test --separate-depth-2d: ordinary RGB comes from a 1x1 orange image;
; alpha compares s5 against a D32 depth. s[0:1] points to two T#s at offsets 0 and 32.
; The fixture registers the two literal S#s as ordinary and comparison samplers.
s_load_dwordx8 s[8:15], s[0:1], 0x0
s_waitcnt lgkmcnt(0)
s_mov_b32 s16, 0x92
s_mov_b32 s17, 0xc00000
s_mov_b32 s18, 0x14500000
s_mov_b32 s19, 0
v_mov_b32 v1, 0.5
v_mov_b32 v2, 0.5
v_mov_b32 v10, 0
v_mov_b32 v11, 0
v_mov_b32 v12, 0
v_mov_b32 v13, 0
v_mov_b32 v14, 0.5
v_mov_b32 v15, 0.5
image_sample_d v[4:7], v[10:15], s[8:15], s[16:19] dmask:0xf dim:SQ_RSRC_IMG_2D
s_waitcnt vmcnt(0)
s_mov_b32 s16, 0x3092
s_load_dwordx8 s[8:15], s[0:1], 0x20
s_waitcnt lgkmcnt(0)
v_mov_b32 v0, s5
image_sample_c_lz v7, v[0:2], s[8:15], s[16:19] dmask:0x1 dim:SQ_RSRC_IMG_2D
s_waitcnt vmcnt(0)
exp mrt0 v4, v5, v6, v7 done vm
s_endpgm
