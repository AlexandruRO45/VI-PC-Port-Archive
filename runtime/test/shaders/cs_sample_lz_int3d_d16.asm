; Sample the 4x4x4 R32_UINT volume at two exact texel centres on different nonzero Z slices.
; Run: vk_store_test cs_sample_lz_int3d_d16.spv words
; The harness fills volume texel i = 16z + 4y + x with 200 + i and binds its 3D sampled view at binding 6.
; The T# is the same 0x5000 descriptor as cs_imgatomic3d. Each D16 result is narrowed to its low 16 bits and
; stored in sint texels 0 and 1: (1,2,1) -> 225, and (3,1,2) -> 239. All other words in all three images keep
; their fill. The unregistered S# selects the harness's default nearest sampler.
v_cmpx_eq_u32_e32 0, v0
s_mov_b32 s16, 0x5000
s_mov_b32 s17, 0x1400000
s_mov_b32 s18, 0x55
s_mov_b32 s19, 0x55
s_mov_b32 s20, 0x55
s_mov_b32 s21, 0x55
s_mov_b32 s22, 0x55
s_mov_b32 s23, 0x55
s_mov_b32 s24, 0x3000
s_mov_b32 s25, 0x1500000
s_mov_b32 s26, 0x33
s_mov_b32 s27, 0x33
s_mov_b32 s28, 0x33
s_mov_b32 s29, 0x33
s_mov_b32 s30, 0x33
s_mov_b32 s31, 0x33
s_mov_b32 s32, 0
s_mov_b32 s33, 0
s_mov_b32 s34, 0
s_mov_b32 s35, 0
v_mov_b32_e32 v1, 0x3ec00000
v_mov_b32_e32 v2, 0x3f200000
v_mov_b32_e32 v3, 0x3ec00000
image_sample_lz v4, v[1:3], s[16:23], s[32:35] dmask:0x1 dim:SQ_RSRC_IMG_3D d16
s_waitcnt vmcnt(0)
v_and_b32_e32 v4, 0xffff, v4
v_mov_b32_e32 v1, 0
v_mov_b32_e32 v2, 0
image_store v4, v[1:2], s[24:31] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm
v_mov_b32_e32 v1, 0x3f600000
v_mov_b32_e32 v2, 0x3ec00000
v_mov_b32_e32 v3, 0x3f200000
image_sample_lz v4, v[1:3], s[16:23], s[32:35] dmask:0x1 dim:SQ_RSRC_IMG_3D d16
s_waitcnt vmcnt(0)
v_and_b32_e32 v4, 0xffff, v4
v_mov_b32_e32 v1, 1
v_mov_b32_e32 v2, 0
image_store v4, v[1:2], s[24:31] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm
s_endpgm
