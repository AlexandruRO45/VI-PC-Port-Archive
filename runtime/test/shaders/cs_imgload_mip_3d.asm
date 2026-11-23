; 3D explicit-mip loads (docs/design.md): image_load_mip reads x, y, z and the mip from four address VGPRs and
; fetches through the 3D sampled view of the dimension's binding (6), the class (float, UINT or SINT) chosen by the
; T# FORMAT; d16 narrows the selected components and packs them low half first.
; Run: vk_store_test cs_imgload_mip_3d.spv words
; vk_store_test binds a 4x4x4 image with three levels (4x4x4, 2x2x2, 1x1x1) at index 1, 2 and 3 of the 3D sampled
; array, in the same formats as the 2D images: rgba8 unorm (T# 0x1000...), r32 uint (0x2000...) and r32 sint
; (0x3000...). Level L of the uint image holds 0x10000 L + 16z + 4y + x at texel (x, y, z), the sint image the same
; negated, and the unorm image one colour per level: red at level 0, green at level 1, blue at level 2. Lane 0
; stores every result into its own texel of the 2D sint image (T# 0x3000..., index 3 of the 2D storage array),
; texel i = 8y + x:
;   (x, y, z)  mip  T#     dmask  result                        word  texel  value
;   (1, 2, 3)   0   uint   0x1    16*3 + 4*2 + 1 = 57            0    (0, 0)  0x00000039
;   (1, 2, 3)   0   sint   0x1    -57                           1    (1, 0)  0xffffffc7
;   (1, 2, 3)   0   unorm  0xf    red, (1, 0, 0, 1)              2    (2, 0)  0x3f800000
;                                                               3    (3, 0)  0x00000000
;                                                               4    (4, 0)  0x00000000
;                                                               5    (5, 0)  0x3f800000
;   (1, 0, 1)   1   uint   0x1    0x10000 + 17 = 0x10011         6    (6, 0)  0x00010011
;   (1, 0, 1)   1   sint   0x1    -0x10011                      7    (7, 0)  0xfffeffef
;   (0, 0, 0)   2   unorm  0x7    blue, (0, 0, 1)                8    (0, 1)  0x00000000
;                                                               9    (1, 1)  0x00000000
;                                                              10    (2, 1)  0x3f800000
;   (1, 0, 1)   1   uint   0x3 d16  saturate(0x10011), 0        11    (3, 1)  0x0000ffff
;   (1, 2, 3)   0   uint   0x3 d16  57, 0                       12    (4, 1)  0x00000039
;   (1, 0, 1)   1   sint   0x3 d16  saturate(-0x10011), 0       13    (5, 1)  0x00008000
;   (1, 2, 3)   0   unorm  0x3 d16  f16(1.0), f16(0.0)          14    (6, 1)  0x00003c00
; Every other word keeps the tool's fill: sint word i = -2i, uint word i = 100 + i, volume word i = 200 + i.
v_cmpx_eq_u32_e32 0, v0
s_mov_b32 s8, 0x2000
s_mov_b32 s9, 0x1400000
s_mov_b32 s10, 0x22
s_mov_b32 s11, 0x22
s_mov_b32 s12, 0x22
s_mov_b32 s13, 0x22
s_mov_b32 s14, 0x22
s_mov_b32 s15, 0x22
s_mov_b32 s16, 0x3000
s_mov_b32 s17, 0x1500000
s_mov_b32 s18, 0x33
s_mov_b32 s19, 0x33
s_mov_b32 s20, 0x33
s_mov_b32 s21, 0x33
s_mov_b32 s22, 0x33
s_mov_b32 s23, 0x33
s_mov_b32 s24, 0x1000
s_mov_b32 s25, 0x3800000
s_mov_b32 s26, 0x11
s_mov_b32 s27, 0x11
s_mov_b32 s28, 0x11
s_mov_b32 s29, 0x11
s_mov_b32 s30, 0x11
s_mov_b32 s31, 0x11
v_mov_b32_e32 v1, 1
v_mov_b32_e32 v2, 2
v_mov_b32_e32 v3, 3
v_mov_b32_e32 v4, 0
image_load_mip v5, v[1:4], s[8:15] dmask:0x1 dim:SQ_RSRC_IMG_3D
s_waitcnt vmcnt(0)
v_mov_b32_e32 v20, 0
v_mov_b32_e32 v21, 0
image_store v5, v[20:21], s[16:23] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm
image_load_mip v5, v[1:4], s[16:23] dmask:0x1 dim:SQ_RSRC_IMG_3D
s_waitcnt vmcnt(0)
v_mov_b32_e32 v20, 1
image_store v5, v[20:21], s[16:23] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm
image_load_mip v[5:8], v[1:4], s[24:31] dmask:0xf dim:SQ_RSRC_IMG_3D
s_waitcnt vmcnt(0)
v_mov_b32_e32 v20, 2
image_store v5, v[20:21], s[16:23] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm
v_mov_b32_e32 v20, 3
image_store v6, v[20:21], s[16:23] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm
v_mov_b32_e32 v20, 4
image_store v7, v[20:21], s[16:23] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm
v_mov_b32_e32 v20, 5
image_store v8, v[20:21], s[16:23] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm
v_mov_b32_e32 v1, 1
v_mov_b32_e32 v2, 0
v_mov_b32_e32 v3, 1
v_mov_b32_e32 v4, 1
image_load_mip v5, v[1:4], s[8:15] dmask:0x1 dim:SQ_RSRC_IMG_3D
s_waitcnt vmcnt(0)
v_mov_b32_e32 v20, 6
image_store v5, v[20:21], s[16:23] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm
image_load_mip v5, v[1:4], s[16:23] dmask:0x1 dim:SQ_RSRC_IMG_3D
s_waitcnt vmcnt(0)
v_mov_b32_e32 v20, 7
image_store v5, v[20:21], s[16:23] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm
v_mov_b32_e32 v1, 0
v_mov_b32_e32 v2, 0
v_mov_b32_e32 v3, 0
v_mov_b32_e32 v4, 2
image_load_mip v[5:7], v[1:4], s[24:31] dmask:0x7 dim:SQ_RSRC_IMG_3D
s_waitcnt vmcnt(0)
v_mov_b32_e32 v20, 0
v_mov_b32_e32 v21, 1
image_store v5, v[20:21], s[16:23] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm
v_mov_b32_e32 v20, 1
image_store v6, v[20:21], s[16:23] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm
v_mov_b32_e32 v20, 2
image_store v7, v[20:21], s[16:23] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm
v_mov_b32_e32 v1, 1
v_mov_b32_e32 v2, 0
v_mov_b32_e32 v3, 1
v_mov_b32_e32 v4, 1
image_load_mip v5, v[1:4], s[8:15] dmask:0x3 dim:SQ_RSRC_IMG_3D d16
s_waitcnt vmcnt(0)
v_mov_b32_e32 v20, 3
image_store v5, v[20:21], s[16:23] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm
v_mov_b32_e32 v1, 1
v_mov_b32_e32 v2, 2
v_mov_b32_e32 v3, 3
v_mov_b32_e32 v4, 0
image_load_mip v5, v[1:4], s[8:15] dmask:0x3 dim:SQ_RSRC_IMG_3D d16
s_waitcnt vmcnt(0)
v_mov_b32_e32 v20, 4
image_store v5, v[20:21], s[16:23] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm
v_mov_b32_e32 v1, 1
v_mov_b32_e32 v2, 0
v_mov_b32_e32 v3, 1
v_mov_b32_e32 v4, 1
image_load_mip v5, v[1:4], s[16:23] dmask:0x3 dim:SQ_RSRC_IMG_3D d16
s_waitcnt vmcnt(0)
v_mov_b32_e32 v20, 5
image_store v5, v[20:21], s[16:23] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm
v_mov_b32_e32 v1, 1
v_mov_b32_e32 v2, 2
v_mov_b32_e32 v3, 3
v_mov_b32_e32 v4, 0
image_load_mip v5, v[1:4], s[24:31] dmask:0x3 dim:SQ_RSRC_IMG_3D d16
s_waitcnt vmcnt(0)
v_mov_b32_e32 v20, 6
image_store v5, v[20:21], s[16:23] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm
s_endpgm
