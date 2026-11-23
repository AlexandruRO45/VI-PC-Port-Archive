; regen_fixtures: this source has a translator fixture in tests/semantic/fixtures/disasm
; The six 32-bit 2D image atomics that cs_imgatomic does not run: image_atomic_swap, inc, dec, fmin, fmax and
; fcmpswap (docs/design.md, "image_atomic_{swap,inc,dec,fmin,fmax,fcmpswap}").
; RDNA2 ISA (MIMG): swap stores DATA and returns the old texel with glc; inc gives (M >= DATA) ? 0 : M + 1 and dec
; gives (M == 0 || M > DATA) ? DATA : M - 1, both unsigned; fmin/fmax read the texel as binary32 and take the
; non-NaN operand (NMin/NMax); fcmpswap writes DATA where the texel equals CMP and leaves it alone otherwise.
; Only lane 0 runs, so every case happens once. The float cases seed the texel with an earlier swap, so their input
; bits are chosen here and not the harness fill (100 + i is a subnormal as binary32).
; Run: vk_store_test cs_imgatomic_ext.spv words
;    c  op          data                M before         uint word c after   returned (sint word c)
;    0  swap        0xdeadbeef          100              0xdeadbeef          0x00000064
;    1  inc         200                 101              102 = 0x00000066    0x00000065
;    2  dec         100                 102              100 = 0x00000064    0x00000066
;    3  swap        0x40000000 (2.0)    103              0x40000000          0x00000067
;       fmin        0x3f800000 (1.0)    2.0              1.0 = 0x3f800000    0x40000000
;    4  swap        0x3f800000 (1.0)    104              0x3f800000          0x00000068
;       fmax        0x40400000 (3.0)    1.0              3.0 = 0x40400000    0x3f800000
;    5  swap        0x3f800000 (1.0)    105              0x3f800000          0x00000069
;       fcmpswap    0x40000000 (2.0), equal to 1.0        2.0 = 0x40000000    0x3f800000
;    6  fcmpswap    0x40800000 (4.0), not 3.0    0x0000006a        kept 0x0000006a     0x0000006a
;    7  swap        0x7fc00000 (NaN)    107              0x7fc00000          0x0000006b
;       fmin        0x40000000 (2.0)    NaN              2.0 = 0x40000000    NaN = 0x7fc00000
; Every other texel keeps its fill: uint word i = 100 + i and sint word i = -2i.
; group: 32,1,1
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
v_cmpx_eq_u32_e32 0, v0
v_mov_b32_e32 v2, 0
v_mov_b32_e32 v1, 0
v_mov_b32_e32 v3, 0xdeadbeef
image_atomic_swap v3, v[1:2], s[16:23] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm glc
s_waitcnt vmcnt(0)
image_store v3, v[1:2], s[24:31] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm
v_mov_b32_e32 v1, 1
v_mov_b32_e32 v3, 200
image_atomic_inc v3, v[1:2], s[16:23] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm glc
s_waitcnt vmcnt(0)
image_store v3, v[1:2], s[24:31] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm
v_mov_b32_e32 v1, 2
v_mov_b32_e32 v3, 100
image_atomic_dec v3, v[1:2], s[16:23] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm glc
s_waitcnt vmcnt(0)
image_store v3, v[1:2], s[24:31] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm
v_mov_b32_e32 v1, 3
v_mov_b32_e32 v3, 0x40000000
image_atomic_swap v3, v[1:2], s[16:23] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm
s_waitcnt vmcnt(0)
v_mov_b32_e32 v3, 0x3f800000
image_atomic_fmin v3, v[1:2], s[16:23] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm glc
s_waitcnt vmcnt(0)
image_store v3, v[1:2], s[24:31] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm
v_mov_b32_e32 v1, 4
v_mov_b32_e32 v3, 0x3f800000
image_atomic_swap v3, v[1:2], s[16:23] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm
s_waitcnt vmcnt(0)
v_mov_b32_e32 v3, 0x40400000
image_atomic_fmax v3, v[1:2], s[16:23] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm glc
s_waitcnt vmcnt(0)
image_store v3, v[1:2], s[24:31] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm
v_mov_b32_e32 v1, 5
v_mov_b32_e32 v3, 0x3f800000
image_atomic_swap v3, v[1:2], s[16:23] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm
s_waitcnt vmcnt(0)
v_mov_b32_e32 v3, 0x40000000
v_mov_b32_e32 v4, 0x3f800000
image_atomic_fcmpswap v[3:4], v[1:2], s[16:23] dmask:0x3 dim:SQ_RSRC_IMG_2D unorm glc
s_waitcnt vmcnt(0)
image_store v3, v[1:2], s[24:31] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm
v_mov_b32_e32 v1, 6
v_mov_b32_e32 v3, 0x40800000
v_mov_b32_e32 v4, 0x40400000
image_atomic_fcmpswap v[3:4], v[1:2], s[16:23] dmask:0x3 dim:SQ_RSRC_IMG_2D unorm glc
s_waitcnt vmcnt(0)
image_store v3, v[1:2], s[24:31] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm
v_mov_b32_e32 v1, 7
v_mov_b32_e32 v3, 0x7fc00000
image_atomic_swap v3, v[1:2], s[16:23] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm
s_waitcnt vmcnt(0)
v_mov_b32_e32 v3, 0x40000000
image_atomic_fmin v3, v[1:2], s[16:23] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm glc
s_waitcnt vmcnt(0)
image_store v3, v[1:2], s[24:31] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm
s_endpgm
