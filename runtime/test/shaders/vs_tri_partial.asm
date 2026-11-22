; Vertex shader for the partial-quad pixel tests (TEST-363, w64_graphics.py): one triangle over part of the 64x64
; target, with corners at pixels (0, 0), (64, 0) and (0, 48). A pixel (x, y) is covered when its centre is inside,
; 3x + 4y < 188.5. No pixel centre lies on the long edge (3x + 4y = 188.5 has no integer solution) and the corners are
; whole pixels, so the coverage does not depend on the rasterizer's tie rule or subpixel snapping. The long edge cuts
; 2x2 quads, which then run with helper invocations. param0 = (u, v, 0, 1) with u = x / 64 and v = y / 64 at the
; corners, as vs_ids, so a pixel shader finds its pixel the same way; param1 = 0x101 + vertex index, as vs_ids.
v_cmp_eq_u32_e32 vcc_lo, 1, v0
v_cndmask_b32_e64 v1, 0, 1.0, vcc_lo
v_cmp_eq_u32_e32 vcc_lo, 2, v0
v_cndmask_b32_e64 v2, 0, 0x3f400000, vcc_lo
v_fma_f32 v3, v1, 2.0, -1.0
v_fma_f32 v4, v2, 2.0, -1.0
v_mov_b32_e32 v5, 0
v_mov_b32_e32 v6, 1.0
exp pos0 v3, v4, v5, v6 done
exp param0 v1, v2, v5, v6
v_add_nc_u32_e32 v7, 0x101, v0
exp param1 v7, v7, v7, v7
s_endpgm
