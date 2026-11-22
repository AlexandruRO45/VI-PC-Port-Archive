; image_sample_c_lz_o: depth compare at LOD 0 with a texel offset (RDNA2 ISA, MIMG address order: offset, z-compare,
; x, y). The first dword packs signed 6-bit texel offsets, x = [5:0], y = [13:8].
; Run: vk_texture_test dref_o [out.raw]  (VKT_NEED_MAINTENANCE8=1: exit 77 without the extension, as ps_texoff)
; "dref_o" binds an 8x8 D32 texture whose texel (x, y) holds depth d(k) = (k + 0.5) / 64 with k = 8y + x, and a point
; sampler, clamp to edge, with DEPTH_COMPARE_FUNC LESS_OR_EQUAL (S# word 0 0x3092): a sample returns 1.0 when its
; reference <= the texel's depth, else 0.0. It draws 64x64 pixels with u, v = (px + 0.5) / 64, (py + 0.5) / 64.
; The shader samples at the fixed (u, v) = (0.5625, 0.5625) = the centre of texel (4, 4), whatever the pixel, and
; moves the texel with the offset, as ps_texoff does: with bx = px >> 3, by = py >> 3 (0..7) the offset is
; (ox, oy) = (bx - 4, by - 4), so the texel read is (bx, by), k = R = 8 by + bx, never clamped.
;   .x .y .z  refs d(R - 1), d(R), d(R + 1) (exact: (R - 1/2) / 64 etc.) against the texel of index R:
;             d(R - 1) <= d(R) -> 1.0, d(R) <= d(R) -> 1.0 (equal), d(R + 1) <= d(R) -> 0.0
;             so (1.0, 1.0, 0.0) at every pixel. Refs below 0 and above 1 (R = 0 and R = 63) fail the same way
;             whether or not the sampler clamps them to 0..1. A dropped offset reads k = 36 instead and breaks every
;             pixel except those with R = 36 (the 8x8 block bx = by = 4); an x and y swap reads k = 8 bx + by and
;             breaks every pixel but the diagonal blocks.
;   .w  two more samples with the offset dword loaded from the immediate 0x023d: x = 0x3d = -3, y = 0x02 = +2 -> texel
;             (1, 6), k = 49. d(49) <= d(49) -> 1.0, d(50) <= d(49) -> 0.0; the shader exports first + 2 * second
;             = 1.0. A dropped offset reads k = 36 (0.0 for the first: 0.0 exported); an offset of +2 in
;             x as well reads k = 51 (the first 0.0 again).
; Expected mrt0 at every pixel: (1.0, 1.0, 0.0, 1.0).
v_interp_p1_f32_e32 v2, v0, attr0.x
v_interp_p2_f32_e32 v2, v1, attr0.x
v_interp_p1_f32_e32 v3, v0, attr0.y
v_interp_p2_f32_e32 v3, v1, attr0.y
v_mul_f32_e32 v4, 0x41000000, v2
v_mul_f32_e32 v5, 0x41000000, v3
v_cvt_u32_f32_e32 v4, v4
v_cvt_u32_f32_e32 v5, v5
v_lshl_or_b32 v6, v5, 3, v4
v_add_nc_u32_e32 v7, -4, v4
v_add_nc_u32_e32 v8, -4, v5
v_and_b32_e32 v7, 63, v7
v_and_b32_e32 v8, 63, v8
v_lshl_or_b32 v10, v8, 8, v7
v_cvt_f32_u32_e32 v6, v6
v_add_f32_e32 v14, -0.5, v6
v_add_f32_e32 v15, 0.5, v6
v_add_f32_e32 v16, 1.0, v15
v_mul_f32_e32 v14, 0x3c800000, v14
v_mul_f32_e32 v15, 0x3c800000, v15
v_mul_f32_e32 v16, 0x3c800000, v16
v_mov_b32_e32 v12, 0x3f100000
v_mov_b32_e32 v13, 0x3f100000
v_mov_b32_e32 v24, 0x23d
v_mov_b32_e32 v25, 0x3f460000
v_mov_b32_e32 v26, 0x3f100000
v_mov_b32_e32 v27, 0x3f100000
v_mov_b32_e32 v30, 0x23d
v_mov_b32_e32 v31, 0x3f4a0000
v_mov_b32_e32 v32, 0x3f100000
v_mov_b32_e32 v33, 0x3f100000
s_load_dwordx8 s[8:15], s[0:1], 0x0
s_waitcnt lgkmcnt(0)
s_mov_b32 s16, 0x3092
s_mov_b32 s17, 0xc00000
s_mov_b32 s18, 0x14500000
s_mov_b32 s19, 0
v_mov_b32_e32 v11, v14
image_sample_c_lz_o v20, v[10:13], s[8:15], s[16:19] dmask:0x1 dim:SQ_RSRC_IMG_2D
v_mov_b32_e32 v11, v15
image_sample_c_lz_o v21, v[10:13], s[8:15], s[16:19] dmask:0x1 dim:SQ_RSRC_IMG_2D
v_mov_b32_e32 v11, v16
image_sample_c_lz_o v22, v[10:13], s[8:15], s[16:19] dmask:0x1 dim:SQ_RSRC_IMG_2D
image_sample_c_lz_o v28, v[24:27], s[8:15], s[16:19] dmask:0x1 dim:SQ_RSRC_IMG_2D
image_sample_c_lz_o v29, v[30:33], s[8:15], s[16:19] dmask:0x1 dim:SQ_RSRC_IMG_2D
s_waitcnt vmcnt(0)
v_mul_f32_e32 v29, 2.0, v29
v_add_f32_e32 v28, v28, v29
exp mrt0 v20, v21, v22, v28 done vm
s_endpgm
