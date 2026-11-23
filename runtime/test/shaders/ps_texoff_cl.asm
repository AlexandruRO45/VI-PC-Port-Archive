; image_sample_cl_o: a texel offset on a min-LOD-clamped sample, on vk_texture_test's --two-mip-index texture.
; Run: vk_texture_test <vs> ps_texoff_cl.spv out.raw 0 92 c00000 14500000 0 --two-mip-index
; The texture is ps_texture_cl's (128x128 mip 0 in four colour quadrants, top-left red), but mip 1 (64x64) has
; texel (x, y) = (4x, 4y, 0x80, 0xFF) in R8G8B8A8_UNORM, so every mip 1 texel is different. Point filters and a
; point mip filter, clamped to the edge (S# 0x92, MIN_LOD 0, MAX_LOD 12). The draw is 64x64 with
; attr0.xy = ((px + 0.5) / 64, (py + 0.5) / 64).
;
; Address v[4:7] = (offset dword, u / 4, v / 4, clamp). The offset dword is 0x3e03: x = +3, y = 0x3e = -2.
; The coordinates are u / 4, v / 4, so adjacent pixels are half a mip 0 texel apart and the unclamped LOD is -1.
; The clamp is 0.0 left of x = 32 and 1.0 from there on.
;
; Expected mrt0 for pixel (px, py), derived from the rule (Vulkan: the offset is in texels of the level the
; sample reads, and MinLod raises the LOD), not from a run:
;   px < 32  (clamp 0): LOD max(-1, 0) = 0, mip 0 texel (floor((px + 0.5) / 2) + 3, floor((py + 0.5) / 2) - 2),
;            clamped to the edge. That is x <= 18 and y <= 29, all in the red quadrant: (1, 0, 0, 1).
;   px >= 32 (clamp 1): LOD 1, mip 1 texel X = floor((px + 0.5) / 4) + 3, Y = max(floor((py + 0.5) / 4) - 2, 0),
;            so (4X / 255, 4Y / 255, 128 / 255, 1). X is 11..18 and Y is 0..13.
; What a wrong translation gives instead:
;   - offset dropped:                   X - 3 and Y + 2 (Y unclamped).
;   - clamp ignored (no MinLod):        the right half is red, like the left.
;   - --texel-offset-fallback folding the offset at mip 0's size (the unclamped LOD's level; before SHADER-842):
;     the shift is half a mip 1 texel per offset texel, so X = floor((px + 0.5) / 4 + 1.5) and
;     Y = max(floor((py + 0.5) / 4 - 1), 0): at px = 32, X is 9 instead of 11.
; Every coordinate above is a multiple of 1/512 plus 1/128 or 1/256, exact in binary32 and at least 1/8 of a texel
; from a texel edge, so nearest sampling has no ties.
; A non-constant Offset outside gather needs VK_KHR_maintenance8 (VKT_NEED_MAINTENANCE8=1: exit 77 without it);
; with LEONIDA_TEXOFF_FALLBACK=1 the translator folds the offset into the coordinate and needs no extension.
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
v_mov_b32_e32 v4, 0x3e03
v_mul_f32_e32 v5, 0.25, v2
v_mul_f32_e32 v6, 0.25, v3
v_cndmask_b32_e64 v7, 0, 1.0, vcc_lo
image_sample_cl_o v[8:11], v[4:7], s[8:15], s[16:19] dmask:0xf dim:SQ_RSRC_IMG_2D
s_waitcnt vmcnt(0)
exp mrt0 v8, v9, v10, v11 done vm
s_endpgm
