; image_bvh64_intersect_ray on a BOX16 node: the scene of cs_bvh.asm with its root as BOX16 (the XDXR builder's
; MINIMIZE_MEMORY nodes, runtime/src/xdxr_build.comp). vk_compute_test writes it into the heap: BOX16 root at 0x100
; (node 0x24 = 0x100 >> 3 | type 4) -> triangle z=10 node 0x40, triangle z=5 node 0x48 whose box has max.x = +inf
; (0x7c00, the bound the builder writes past the half range), a missed box, box 0x68 with a half NaN min.x. Each
; child's six halves are at 0x110 + 12 i (min.x|min.y<<16, min.z|max.x<<16, max.y|max.z<<16). Ray (0.25,0.25,0) +z.
; Run: vk_compute_test cs_bvh16.spv 1 bvh16
; -> the words of cs_bvh.asm: box result 0x68 0x48 0x40 0xffffffff (sorted; RDNA2 skips the NaN plane and enters box
; 0x68 at t = 0, SHADER-650; the +inf bound keeps the z=5 child at distance 5), triangle result -10 -1 -0.25 -0.25
; (t = 10, bary 0.25/0.25).
v_cmpx_eq_u32_e32 0, v0
v_mov_b32_e32 v1, 0x24
v_mov_b32_e32 v2, 0
v_mov_b32_e32 v3, 0x42c80000
v_mov_b32_e32 v4, 0.25
v_mov_b32_e32 v5, 0.25
v_mov_b32_e32 v6, 0
v_mov_b32_e32 v7, 0
v_mov_b32_e32 v8, 0
v_mov_b32_e32 v9, 1.0
v_mov_b32_e32 v10, 0x7f800000
v_mov_b32_e32 v11, 0x7f800000
v_mov_b32_e32 v12, 1.0
s_mov_b32 s4, 0
s_mov_b32 s5, 0x80000000
s_mov_b32 s6, -1
s_mov_b32 s7, 0x81000000
image_bvh64_intersect_ray v[20:23], [v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12], s[4:7]
v_mov_b32_e32 v30, 0
buffer_store_dwordx4 v[20:23], v30, s[0:3], 0 offen
v_mov_b32_e32 v1, 0x40
image_bvh64_intersect_ray v[24:27], [v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12], s[4:7]
v_mov_b32_e32 v30, 16
buffer_store_dwordx4 v[24:27], v30, s[0:3], 0 offen
s_endpgm
