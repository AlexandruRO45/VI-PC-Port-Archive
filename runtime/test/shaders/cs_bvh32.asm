; 32-bit image_bvh_intersect_ray emulation (TEST-184): companion to cs_bvh.asm for the 32-bit node-id form
; (11 address VGPRs: node32, tmax, org.xyz, dir.xyz, inv.xyz, with no node_hi register).
; vk_compute_test writes the same AMD-format BVH into the heap (box32 root at 0x100 -> triangle z=10 node 0x40 at
; 0x200, triangle z=5 node 0x48 at 0x240, missed box, NaN box). Ray (0.25,0.25,0) +z, tmax=100.0 (non-zero in v2, so
; misreading it as a 64-bit node_hi would fail). Call 1 uses descriptor base 0 and node 0x25 (0x100); call 2 also
; exercises a non-zero descriptor base (s4 = 1 -> 0x100) with node 0x20 (+0x100 = 0x200).
; Run: vk_compute_test cs_bvh32.spv 1 bvh
; -> box result 0x48 0x40 0xffffffff 0xffffffff (sorted), triangle result -10 -1 -0.25 -0.25 (t = 10, bary 0.25/0.25).
v_cmpx_eq_u32_e32 0, v0
v_mov_b32_e32 v1, 0x25
v_mov_b32_e32 v2, 0x42c80000
v_mov_b32_e32 v3, 0.25
v_mov_b32_e32 v4, 0.25
v_mov_b32_e32 v5, 0
v_mov_b32_e32 v6, 0
v_mov_b32_e32 v7, 0
v_mov_b32_e32 v8, 1.0
v_mov_b32_e32 v9, 0x7f800000
v_mov_b32_e32 v10, 0x7f800000
v_mov_b32_e32 v11, 1.0
s_mov_b32 s4, 0
s_mov_b32 s5, 0x80000000
s_mov_b32 s6, -1
s_mov_b32 s7, 0x81000000
image_bvh_intersect_ray v[20:23], [v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11], s[4:7]
v_mov_b32_e32 v30, 0
buffer_store_dwordx4 v[20:23], v30, s[0:3], 0 offen
s_mov_b32 s4, 1
v_mov_b32_e32 v1, 0x20
image_bvh_intersect_ray v[24:27], [v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11], s[4:7]
v_mov_b32_e32 v30, 16
buffer_store_dwordx4 v[24:27], v30, s[0:3], 0 offen
s_endpgm
