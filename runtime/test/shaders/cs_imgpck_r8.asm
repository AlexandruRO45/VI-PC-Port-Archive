; image_load_pck and image_load_pck_sgn on an 8_UINT texel, measured on gfx1036 (tools/hw_oracle): a texel under 32 bits is
; one raw dword with the byte in the low bits; _pck zero extends it and _pck_sgn sign extends it from bit 7 (a 16 bit texel
; from bit 15).
; Thread i reads texel i, whose byte is (0x7c + 0x0d i) & 0xFF: 0x7c, 0x89, 0x96, ... (the sign bit is set from i = 1):
;   uint image <- image_load_pck, dmask 1: that byte, e.g. 0x7c for i = 0 and 0x89 for i = 1
;   sint image <- image_load_pck_sgn, dmask 1: the same, with bits 31:8 set when bit 7 is, e.g. 0x7c and 0xffffff89
; vk_store_test fills, in "words" mode before the dispatch (texel i = 8y + x, byte n of the image's memory in order):
;   rgba8 image (T# 0x1000, FORMAT 56 UNORM; the same texels as T# 0x6000, FORMAT 60 UINT): byte n = (0x81 + 0x0b n) & 0xFF
;   rgba16 image (T# 0x7000, FORMAT 69 16_16_16_16_UINT): byte n = (0x35 + 0x13 n) & 0xFF
;   r8 image (T# 0x8000, FORMAT 5 8_UINT): byte = (0x7c + 0x0d i) & 0xFF, which crosses 0x80 at i = 1
;   uint image (T# 0x2000) = 100 + i, sint image (T# 0x3000) = -2 i; the shader overwrites word i of both
; Run: vk_store_test cs_imgpck_r8.spv words
v_and_b32_e32 v1, 7, v0
v_lshrrev_b32_e32 v2, 3, v0
v_mov_b32_e32 v9, 0
s_mov_b32 s8, 0x1000
s_mov_b32 s9, 0x3800000
s_mov_b32 s10, 0x11
s_mov_b32 s11, 0x11
s_mov_b32 s12, 0x11
s_mov_b32 s13, 0x11
s_mov_b32 s14, 0x11
s_mov_b32 s15, 0x11
s_mov_b32 s16, 0x2000
s_mov_b32 s17, 0x1400000
s_mov_b32 s18, 0x22
s_mov_b32 s19, 0x22
s_mov_b32 s20, 0x22
s_mov_b32 s21, 0x22
s_mov_b32 s22, 0x22
s_mov_b32 s23, 0x22
s_mov_b32 s24, 0x3000
s_mov_b32 s25, 0x1500000
s_mov_b32 s26, 0x33
s_mov_b32 s27, 0x33
s_mov_b32 s28, 0x33
s_mov_b32 s29, 0x33
s_mov_b32 s30, 0x33
s_mov_b32 s31, 0x33
s_mov_b32 s32, 0x6000
s_mov_b32 s33, 0x3c00000
s_mov_b32 s34, 0x66
s_mov_b32 s35, 0x66
s_mov_b32 s36, 0x66
s_mov_b32 s37, 0x66
s_mov_b32 s38, 0x66
s_mov_b32 s39, 0x66
s_mov_b32 s40, 0x7000
s_mov_b32 s41, 0x4500000
s_mov_b32 s42, 0x77
s_mov_b32 s43, 0x77
s_mov_b32 s44, 0x77
s_mov_b32 s45, 0x77
s_mov_b32 s46, 0x77
s_mov_b32 s47, 0x77
s_mov_b32 s48, 0x8000
s_mov_b32 s49, 0x500000
s_mov_b32 s50, 0x88
s_mov_b32 s51, 0x88
s_mov_b32 s52, 0x88
s_mov_b32 s53, 0x88
s_mov_b32 s54, 0x88
s_mov_b32 s55, 0x88
image_load_pck v3, v[1:2], s[48:55] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm
image_load_pck_sgn v4, v[1:2], s[48:55] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm
s_waitcnt vmcnt(0)
image_store v3, v[1:2], s[16:23] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm
image_store v4, v[1:2], s[24:31] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm
s_endpgm
