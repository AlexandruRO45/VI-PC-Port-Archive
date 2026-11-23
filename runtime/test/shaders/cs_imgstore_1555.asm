; SHADER-1202: T# FORMAT 134 selects R16_UINT storage for B5G5R5A1_UNORM.
; Even rows store RGBA = (x/8, 0.25, 2, x/8), yielding 011f 111f 211f 311f 411f cd1f dd1f ed1f.
; Odd rows store D16 (R = 0.5, A = 1) with dmask 0x9, yielding c000 in every texel.
; Run with vk_store_test cs_imgstore_1555.spv pack-1555.
v_and_b32_e32 v1, 7, v0
v_lshrrev_b32_e32 v2, 3, v0
v_cvt_f32_u32_e32 v3, v1
v_mul_f32_e32 v3, 0x3e000000, v3
v_mov_b32_e32 v4, 0.25
v_mov_b32_e32 v5, 2.0
v_mov_b32_e32 v6, v3
s_mov_b32 s8, 0x2000
s_mov_b32 s9, 0x8600000
s_mov_b32 s10, 0x22
s_mov_b32 s11, 0x22
s_mov_b32 s12, 0x22
s_mov_b32 s13, 0x22
s_mov_b32 s14, 0x22
s_mov_b32 s15, 0x22
v_and_b32_e32 v7, 1, v2
v_cmp_eq_u32_e32 vcc_lo, 0, v7
s_and_saveexec_b32 s0, vcc_lo
image_store v[3:6], v[1:2], s[8:15] dmask:0xf dim:SQ_RSRC_IMG_2D unorm
s_xor_b32 exec_lo, exec_lo, s0
v_mov_b32_e32 v3, 0x3c003800
image_store v3, v[1:2], s[8:15] dmask:0x9 dim:SQ_RSRC_IMG_2D unorm d16
s_endpgm
