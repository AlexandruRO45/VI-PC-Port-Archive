; Sample a Texture1DArray SRV at view-relative slice 1 and explicit mip 1. Run d3d12_m2_formats_test.
; PS input attr0.x is u. The heap slots are the T# and S# descriptors used by ps_m2.asm.
; fssm-at: 0x10
; fssm-data: 0x324d5350 0x41314431
s_load_dwordx8 s[8:15], s[0:1], 0x0
s_load_dwordx4 s[16:19], s[2:3], 0x0
s_waitcnt lgkmcnt(0)
v_interp_p1_f32_e32 v2, v0, attr0.x
v_interp_p2_f32_e32 v2, v1, attr0.x
v_mov_b32 v3, 1.0
v_mov_b32 v4, 1.0
image_sample_l v[5:8], v[2:4], s[8:15], s[16:19] dmask:0xf dim:SQ_RSRC_IMG_1D_ARRAY
s_waitcnt vmcnt(0)
exp mrt0 v5, v6, v7, v8 done vm
s_endpgm
