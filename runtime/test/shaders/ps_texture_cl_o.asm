; image_sample_cl_o on vk_texture_test's two-mip texture (--two-mip: 128x128 quadrants, red top-left, green top-right,
; solid cyan mip 1) drawn on 64x64. u is the centre of mip-0 texel 63 (63.5 / 128 = 0x3efe0000), the last red texel
; before the red/green boundary, and v is v / 4, so the unclamped LOD is -1. The texel offset (1, 0) moves the sample
; one texel right, onto texel 64 (green). The min-LOD clamp after the coordinates is 0.0 left of x = 32 and 1.0 from
; there on: the left half must read mip 0 green, the right half mip 1 cyan. Dropping the offset turns the left half red;
; a clamp read from the wrong register, or ignored, turns the right half green.
; Address v[4:7] = (offset, u, v / 4, clamp): offset first, then the coordinates, then the clamp. The offset dword
; packs x in bits [5:0] and y in bits [13:8], so (1, 0) is 1.
v_interp_p1_f32_e32 v2, v0, attr0.x
v_interp_p2_f32_e32 v2, v1, attr0.x
v_interp_p1_f32_e32 v3, v0, attr0.y
v_interp_p2_f32_e32 v3, v1, attr0.y
s_load_dwordx8 s[8:15], s[0:1], 0x0
s_waitcnt lgkmcnt(0)
s_mov_b32 s16, 0x92
s_mov_b32 s17, 0xc00000
s_mov_b32 s18, 0x14500000
s_mov_b32 s19, 0
v_cmp_le_f32_e32 vcc_lo, 0.5, v2
v_mov_b32_e32 v4, 1
v_mov_b32_e32 v5, 0x3efe0000
v_mul_f32_e32 v6, 0.25, v3
v_cndmask_b32_e64 v7, 0, 1.0, vcc_lo
image_sample_cl_o v[8:11], v[4:7], s[8:15], s[16:19] dmask:0xf dim:SQ_RSRC_IMG_2D
s_waitcnt vmcnt(0)
exp mrt0 v8, v9, v10, v11 done vm
s_endpgm
