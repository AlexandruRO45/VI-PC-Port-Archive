; Image atomics whose texel is outside the image (EGEN-IMGATOMIC-OOB). Measured on RDNA2 (gfx1036, #5059): an
; image_atomic_* at a coordinate outside the image leaves memory untouched and, with glc, returns 0 in the data
; register; in-range atomics are unchanged. The translation guards OpImageTexelPointer with an unsigned compare per
; axis against OpImageQuerySize (a negative coordinate is a large unsigned value), since an out-of-range texel pointer
; is undefined in Vulkan.
; Run: vk_store_test cs_imgatomic_oob.spv words
; The tool fills the 8x8 uint image (T# 0x2000...) with 100 + i at texel i = 8y + x, the sint image (T# 0x3000...)
; with -2i and the 4x4x4 volume (T# 0x5000...) with 200 + i at texel i = 16z + 4y + x. Only thread 0 runs. Case c
; stores its data register afterwards into texel c of the sint image ((c, 0) for c < 8, (c - 8, 1) after). With an
; 8-wide image, (8, 0) is where a linear wrap would land on texel 8 = (0, 1), so any write that escapes the guard
; changes a word that must keep its fill; every word of the three images is checked.
;    c  op        data             coordinate     image   what happens                         sint word c
;    0  add       1000             (8, 0)         2D      x = width: skipped                   0
;    1  add       1000             (-1, 0)        2D      x negative: skipped                  0
;    2  add       1000             (0, 8)         2D      y = height: skipped                  0
;    3  add       1000             (0, -1)        2D      y negative: skipped                  0
;    4  umax      0xffffffff       (8, 8)         2D      both out: skipped                    0
;    5  cmpswap   0xdead, 108      (8, 0)         2D      skipped (108 is texel 8's fill)      0
;    6  swap      0x5a5a           (1000, 0)      2D      skipped                              0
;    7  add       7, no glc        (8, 0)         2D      skipped, data register untouched     7
;    8  add       1                (2, 0)         2D      in range: texel 2 = 103              102 (the old texel)
;    9  add       0x10             (4, 0, 0)      3D      x = width: skipped                   0
;   10  add       0x10             (0, 0, 4)      3D      z = depth: skipped                   0
;   11  add       1                (1, 0, 0)      3D      in range: volume word 1 = 202        201 (the old texel)
; Every other word keeps its fill: uint word i = 100 + i except word 2, volume word i = 200 + i except word 1, sint
; word i = -2i for i = 12..63.
v_cmpx_eq_u32_e32 0, v0
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
s_mov_b32 s32, 0x5000
s_mov_b32 s33, 0x1400000
s_mov_b32 s34, 0x55
s_mov_b32 s35, 0x55
s_mov_b32 s36, 0x55
s_mov_b32 s37, 0x55
s_mov_b32 s38, 0x55
s_mov_b32 s39, 0x55
v_mov_b32_e32 v6, 0
; case 0: x = width
v_mov_b32_e32 v1, 8
v_mov_b32_e32 v2, 0
v_mov_b32_e32 v3, 1000
image_atomic_add v3, v[1:2], s[16:23] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm glc
s_waitcnt vmcnt(0)
v_mov_b32_e32 v5, 0
image_store v3, v[5:6], s[24:31] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm
; case 1: x = -1
v_mov_b32_e32 v1, -1
v_mov_b32_e32 v2, 0
v_mov_b32_e32 v3, 1000
image_atomic_add v3, v[1:2], s[16:23] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm glc
s_waitcnt vmcnt(0)
v_mov_b32_e32 v5, 1
image_store v3, v[5:6], s[24:31] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm
; case 2: y = height
v_mov_b32_e32 v1, 0
v_mov_b32_e32 v2, 8
v_mov_b32_e32 v3, 1000
image_atomic_add v3, v[1:2], s[16:23] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm glc
s_waitcnt vmcnt(0)
v_mov_b32_e32 v5, 2
image_store v3, v[5:6], s[24:31] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm
; case 3: y = -1
v_mov_b32_e32 v1, 0
v_mov_b32_e32 v2, -1
v_mov_b32_e32 v3, 1000
image_atomic_add v3, v[1:2], s[16:23] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm glc
s_waitcnt vmcnt(0)
v_mov_b32_e32 v5, 3
image_store v3, v[5:6], s[24:31] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm
; case 4: both coordinates out
v_mov_b32_e32 v1, 8
v_mov_b32_e32 v2, 8
v_mov_b32_e32 v3, 0xffffffff
image_atomic_umax v3, v[1:2], s[16:23] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm glc
s_waitcnt vmcnt(0)
v_mov_b32_e32 v5, 4
image_store v3, v[5:6], s[24:31] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm
; case 5: cmpswap whose comparand matches the texel a linear wrap would reach
v_mov_b32_e32 v1, 8
v_mov_b32_e32 v2, 0
v_mov_b32_e32 v3, 0xdead
v_mov_b32_e32 v4, 108
image_atomic_cmpswap v[3:4], v[1:2], s[16:23] dmask:0x3 dim:SQ_RSRC_IMG_2D unorm glc
s_waitcnt vmcnt(0)
v_mov_b32_e32 v5, 5
image_store v3, v[5:6], s[24:31] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm
; case 6: swap, far past the edge
v_mov_b32_e32 v1, 1000
v_mov_b32_e32 v2, 0
v_mov_b32_e32 v3, 0x5a5a
image_atomic_swap v3, v[1:2], s[16:23] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm glc
s_waitcnt vmcnt(0)
v_mov_b32_e32 v5, 6
image_store v3, v[5:6], s[24:31] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm
; case 7: no glc, the data register keeps its value
v_mov_b32_e32 v1, 8
v_mov_b32_e32 v2, 0
v_mov_b32_e32 v3, 7
image_atomic_add v3, v[1:2], s[16:23] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm
s_waitcnt vmcnt(0)
v_mov_b32_e32 v5, 7
image_store v3, v[5:6], s[24:31] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm
; case 8: in-range control on the 2D image
v_mov_b32_e32 v6, 1
v_mov_b32_e32 v1, 2
v_mov_b32_e32 v2, 0
v_mov_b32_e32 v3, 1
image_atomic_add v3, v[1:2], s[16:23] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm glc
s_waitcnt vmcnt(0)
v_mov_b32_e32 v5, 0
image_store v3, v[5:6], s[24:31] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm
; case 9: 3D, x = width
v_mov_b32_e32 v1, 4
v_mov_b32_e32 v2, 0
v_mov_b32_e32 v3, 0
v_mov_b32_e32 v4, 0x10
image_atomic_add v4, v[1:3], s[32:39] dmask:0x1 dim:SQ_RSRC_IMG_3D unorm glc
s_waitcnt vmcnt(0)
v_mov_b32_e32 v5, 1
image_store v4, v[5:6], s[24:31] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm
; case 10: 3D, z = depth
v_mov_b32_e32 v1, 0
v_mov_b32_e32 v2, 0
v_mov_b32_e32 v3, 4
v_mov_b32_e32 v4, 0x10
image_atomic_add v4, v[1:3], s[32:39] dmask:0x1 dim:SQ_RSRC_IMG_3D unorm glc
s_waitcnt vmcnt(0)
v_mov_b32_e32 v5, 2
image_store v4, v[5:6], s[24:31] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm
; case 11: in-range control on the volume
v_mov_b32_e32 v1, 1
v_mov_b32_e32 v2, 0
v_mov_b32_e32 v3, 0
v_mov_b32_e32 v4, 1
image_atomic_add v4, v[1:3], s[32:39] dmask:0x1 dim:SQ_RSRC_IMG_3D unorm glc
s_waitcnt vmcnt(0)
v_mov_b32_e32 v5, 3
image_store v4, v[5:6], s[24:31] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm
s_endpgm
