; image_sample_l at an explicit LOD from a run-time T# (user SGPRs) and a literal S#.
; The address is v4, v5, v6: the two coordinates and then the LOD, which is the word the
; sample must take its Lod operand from (GFX10 MIMG address order).
s_mov_b32 s8, 0x92
s_mov_b32 s9, 0xc00000
s_mov_b32 s10, 0x14500000
s_mov_b32 s11, 0
v_mov_b32_e32 v4, 0.5
v_mov_b32_e32 v5, 0.5
v_mov_b32_e32 v6, 2.0
image_sample_l v[0:3], v[4:6], s[0:7], s[8:11] dmask:0x1 dim:SQ_RSRC_IMG_2D
exp mrt0 v0, v1, v2, v3 done vm
s_endpgm
