; Packed d16 image stores. Same descriptors, coordinates and expected texels as cs_imgstore.
v_and_b32_e32 v1, 7, v0
v_lshrrev_b32_e32 v2, 3, v0
v_cvt_f32_u32_e32 v3, v1
v_cvt_f32_u32_e32 v4, v2
v_mul_f32_e32 v3, 0x3e124925, v3
v_mul_f32_e32 v4, 0x3e124925, v4
v_mov_b32_e32 v5, 0.5
v_mov_b32_e32 v6, 1.0
s_mov_b32 s8, 0x1000
s_mov_b32 s9, 0x3800000
s_mov_b32 s10, 0x11
s_mov_b32 s11, 0x11
s_mov_b32 s12, 0x11
s_mov_b32 s13, 0x11
s_mov_b32 s14, 0x11
s_mov_b32 s15, 0x11
v_cvt_pkrtz_f16_f32_e32 v3, v3, v4
v_cvt_pkrtz_f16_f32_e32 v4, v5, v6
image_store v[3:4], v[1:2], s[8:15] dmask:0xf dim:SQ_RSRC_IMG_2D unorm d16
v_lshl_add_u32 v7, v2, 3, v1
v_add_nc_u32_e32 v7, 100, v7
v_mov_b32_e32 v8, 0
s_mov_b32 s16, 0x2000
s_mov_b32 s17, 0x1400000
s_mov_b32 s18, 0x22
s_mov_b32 s19, 0x22
s_mov_b32 s20, 0x22
s_mov_b32 s21, 0x22
s_mov_b32 s22, 0x22
s_mov_b32 s23, 0x22
image_store v[7:8], v[1:2], s[16:23] dmask:0xf dim:SQ_RSRC_IMG_2D unorm d16
v_sub_nc_u32_e32 v11, 0, v0
v_mov_b32_e32 v12, 0
s_mov_b32 s24, 0x3000
s_mov_b32 s25, 0x1500000
s_mov_b32 s26, 0x33
s_mov_b32 s27, 0x33
s_mov_b32 s28, 0x33
s_mov_b32 s29, 0x33
s_mov_b32 s30, 0x33
s_mov_b32 s31, 0x33
v_and_b32_e32 v11, 0xffff, v11
image_store v[11:12], v[1:2], s[24:31] dmask:0xf dim:SQ_RSRC_IMG_2D unorm d16
s_endpgm
