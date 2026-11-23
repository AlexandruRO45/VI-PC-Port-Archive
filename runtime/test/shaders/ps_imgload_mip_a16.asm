; Packed 2D mip-load addresses (docs/design.md): image_load_mip with a16 takes the address of a 2D sample in two
; VGPRs, the unsigned X in the low half of the first and Y in its high half, and the mip level in the low half of the
; second, whose high half is unused. The ordinary fetch and its d16 form read the same two words.
; vk_texture_test --two-mip binds a 128x128 texture whose four 64x64 quadrants are red (x < 64, y < 64), green
; (x >= 64, y < 64), blue (x < 64, y >= 64) and transparent yellow (both halves high), and whose mip 1 is one solid
; cyan 64x64 level, and leaves the T# in the heap at the VA in s[0:1]. Each case is a constant address, so every one
; of the 4096 pixels carries the same four words, each word the fetched texel as a 4-bit mask (component != 0), which
; is the same number in both numeric classes: the UNORM view returns 0.0 or 1.0 and the SINT view 0 or -128.
;   .x  (x, y) = (16, 80), mip 0: blue -> 0 0 1 1 -> 3       reading Y as X instead reads green -> 5
;   .y  (x, y) = (16, 40), mip 1: the solid mip 1 -> 7        reading mip 0 instead reads red -> 9
;   .z  (x, y) = (16, 40), mip 0 with 1 in the high half of the second word, address [v24,v20] (two registers that
;       are not adjacent): red -> 9, so the high half selects nothing (using it as the mip reads mip 1 -> 7)
;   .w  the d16 form of .y: the same address, its four halves packed two per VGPR low half first -> 7
; The SINT view (--two-mip-sint) holds mip 1 and the quadrants masked with 0x80, so each case above gives the same
; mask: the masking keeps exactly the components that were already zero.
; Reading the mip from the high half of the second word, ignoring it, swapping X and Y, or packing the d16 halves
; high half first changes at least one of the four words.
v_mov_b32_e32 v20, 0x00500010
v_mov_b32_e32 v21, 0
s_load_dwordx8 s[8:15], s[0:1], 0x0
s_waitcnt lgkmcnt(0)
image_load_mip v[8:11], v[20:21], s[8:15] dmask:0xf dim:SQ_RSRC_IMG_2D a16
s_waitcnt vmcnt(0)
v_mov_b32_e32 v4, 0
v_cmp_ne_u32_e32 vcc_lo, 0, v8
v_cndmask_b32_e64 v14, 0, 8, vcc_lo
v_or_b32_e32 v4, v4, v14
v_cmp_ne_u32_e32 vcc_lo, 0, v9
v_cndmask_b32_e64 v14, 0, 4, vcc_lo
v_or_b32_e32 v4, v4, v14
v_cmp_ne_u32_e32 vcc_lo, 0, v10
v_cndmask_b32_e64 v14, 0, 2, vcc_lo
v_or_b32_e32 v4, v4, v14
v_cmp_ne_u32_e32 vcc_lo, 0, v11
v_cndmask_b32_e64 v14, 0, 1, vcc_lo
v_or_b32_e32 v4, v4, v14
v_mov_b32_e32 v20, 0x00280010
v_mov_b32_e32 v21, 1
image_load_mip v[8:11], v[20:21], s[8:15] dmask:0xf dim:SQ_RSRC_IMG_2D a16
s_waitcnt vmcnt(0)
v_mov_b32_e32 v5, 0
v_cmp_ne_u32_e32 vcc_lo, 0, v8
v_cndmask_b32_e64 v14, 0, 8, vcc_lo
v_or_b32_e32 v5, v5, v14
v_cmp_ne_u32_e32 vcc_lo, 0, v9
v_cndmask_b32_e64 v14, 0, 4, vcc_lo
v_or_b32_e32 v5, v5, v14
v_cmp_ne_u32_e32 vcc_lo, 0, v10
v_cndmask_b32_e64 v14, 0, 2, vcc_lo
v_or_b32_e32 v5, v5, v14
v_cmp_ne_u32_e32 vcc_lo, 0, v11
v_cndmask_b32_e64 v14, 0, 1, vcc_lo
v_or_b32_e32 v5, v5, v14
v_mov_b32_e32 v24, 0x00280010
v_mov_b32_e32 v20, 0x00010000
image_load_mip v[8:11], [v24,v20], s[8:15] dmask:0xf dim:SQ_RSRC_IMG_2D a16
s_waitcnt vmcnt(0)
v_mov_b32_e32 v6, 0
v_cmp_ne_u32_e32 vcc_lo, 0, v8
v_cndmask_b32_e64 v14, 0, 8, vcc_lo
v_or_b32_e32 v6, v6, v14
v_cmp_ne_u32_e32 vcc_lo, 0, v9
v_cndmask_b32_e64 v14, 0, 4, vcc_lo
v_or_b32_e32 v6, v6, v14
v_cmp_ne_u32_e32 vcc_lo, 0, v10
v_cndmask_b32_e64 v14, 0, 2, vcc_lo
v_or_b32_e32 v6, v6, v14
v_cmp_ne_u32_e32 vcc_lo, 0, v11
v_cndmask_b32_e64 v14, 0, 1, vcc_lo
v_or_b32_e32 v6, v6, v14
v_mov_b32_e32 v20, 0x00280010
v_mov_b32_e32 v21, 1
image_load_mip v[8:9], v[20:21], s[8:15] dmask:0xf dim:SQ_RSRC_IMG_2D a16 d16
s_waitcnt vmcnt(0)
v_and_b32_e32 v10, 0xffff, v8
v_lshrrev_b32_e32 v11, 16, v8
v_and_b32_e32 v12, 0xffff, v9
v_lshrrev_b32_e32 v13, 16, v9
v_mov_b32_e32 v7, 0
v_cmp_ne_u32_e32 vcc_lo, 0, v10
v_cndmask_b32_e64 v14, 0, 8, vcc_lo
v_or_b32_e32 v7, v7, v14
v_cmp_ne_u32_e32 vcc_lo, 0, v11
v_cndmask_b32_e64 v14, 0, 4, vcc_lo
v_or_b32_e32 v7, v7, v14
v_cmp_ne_u32_e32 vcc_lo, 0, v12
v_cndmask_b32_e64 v14, 0, 2, vcc_lo
v_or_b32_e32 v7, v7, v14
v_cmp_ne_u32_e32 vcc_lo, 0, v13
v_cndmask_b32_e64 v14, 0, 1, vcc_lo
v_or_b32_e32 v7, v7, v14
v_cvt_f32_u32_e32 v4, v4
v_cvt_f32_u32_e32 v5, v5
v_cvt_f32_u32_e32 v6, v6
v_cvt_f32_u32_e32 v7, v7
exp mrt0 v4, v5, v6, v7 done vm
s_endpgm
