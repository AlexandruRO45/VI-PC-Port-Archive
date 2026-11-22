; image_load_pck / image_load_mip_pck on an 8_8_8_8 texel, measured on a Radeon RDNA2 iGPU (gfx1036, tools/hw_oracle).
; RDNA2 (MIMG): the _pck forms return the texel's memory bytes as dwords, with no format conversion: the four bytes of an
; 8_8_8_8 texel are ONE dword, byte 0 lowest, whether the FORMAT is UNORM or UINT (image_load would give four floats or
; four integers); dmask bit k selects dword k (a 4 byte texel has only dword 0) and the result goes to consecutive VGPRs.
; Texel 0 of the rgba8 image is bytes 0x81 0x8c 0x97 0xa2 -> 0xa2978c81; texel 1 is 0xad 0xb8 0xc3 0xce -> 0xcec3b8ad.
; Thread i reads texel i twice and stores into texel i:
;   uint image <- image_load_pck, dmask 1, T# 0x1000 (FORMAT 56, UNORM: the float view, bytes put back as round(f * 255))
;   sint image <- image_load_mip_pck, dmask 1, mip 0, T# 0x6000 (FORMAT 60, UINT: the uint view, components packed)
; Both are the raw dword: 0xa2978c81, 0xcec3b8ad, 0xfaefe4d9, ... = sum over k of ((0x81 + 0x0b (4 i + k)) & 0xFF) << 8 k.
; vk_store_test fills, in "words" mode before the dispatch (texel i = 8y + x, byte n of the image's memory in order):
;   rgba8 image (T# 0x1000, FORMAT 56 UNORM; the same texels as T# 0x6000, FORMAT 60 UINT): byte n = (0x81 + 0x0b n) & 0xFF
;   rgba16 image (T# 0x7000, FORMAT 69 16_16_16_16_UINT): byte n = (0x35 + 0x13 n) & 0xFF
;   r8 image (T# 0x8000, FORMAT 5 8_UINT): byte = (0x7c + 0x0d i) & 0xFF, which crosses 0x80 at i = 1
;   uint image (T# 0x2000) = 100 + i, sint image (T# 0x3000) = -2 i; the shader overwrites word i of both
; Run: vk_store_test cs_imgpck_rgba8.spv words
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
image_load_pck v3, v[1:2], s[8:15] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm
v_mov_b32_e32 v10, v1
v_mov_b32_e32 v11, v2
v_mov_b32_e32 v12, 0
image_load_mip_pck v4, v[10:12], s[32:39] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm
s_waitcnt vmcnt(0)
image_store v3, v[1:2], s[16:23] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm
image_store v4, v[1:2], s[24:31] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm
s_endpgm
