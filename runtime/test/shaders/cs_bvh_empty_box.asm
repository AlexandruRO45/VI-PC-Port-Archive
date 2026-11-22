; SHADER-635 / #3366: image_bvh64_intersect_ray on a box32 node whose children include empty boxes (min > max).
; A builder marks an inactive child with an empty box; the instruction must report it as a miss (0xFFFFFFFF). The shader
; (lane 0 only) writes the node into the heap itself, at 0x40 (node id 0x08 | 5 = 0x0D, address = (id & ~7) << 3), then
; runs two rays through it with box sorting off (descriptor bit 63 clear), so child ids come back in slot order.
;   child 0: min (1,1,1) max (0,0,0)   empty on every axis   -> miss
;   child 1: min (0,0,0) max (1,1,1)   the unit box           -> hit, id 0x12 (it is hit by both rays)
;   child 2: min (1,0,0) max (0,1,1)   empty on x only        -> miss
;   child 3: min (0,0,1) max (1,1,0)   empty on z only        -> miss
; A min/max swap per axis (the plain slab test) turns each empty axis into a span of everything, so before the fix the
; empty children were returned as hits. Ray A: origin (0.5,0.5,-1), direction +z. Ray B: origin (0.5,0.5,2), direction -z.
; Both give the four child ids: 0xFFFFFFFF 0x12 0xFFFFFFFF 0xFFFFFFFF.
; Hardware reference (gfx1036, image_bvh64_intersect_ray treats an empty child box as a miss): #3366 comment 7085132018.
; group: 32,1,1
; Run: VKC_SUBGROUP_SIZE=32 vk_compute_test cs_bvh_empty_box.spv 1 dump
;   -> out[0..7] = 4294967295 18 4294967295 4294967295 4294967295 18 4294967295 4294967295
; mattr: +wavefrontsize32,-wavefrontsize64
v_cmpx_eq_u32_e32 0, v0
v_mov_b32_e32 v20, 0x11
v_mov_b32_e32 v21, 0x40
buffer_store_dword v20, v21, s[0:3], 0 offen
v_mov_b32_e32 v20, 0x12
v_mov_b32_e32 v21, 0x44
buffer_store_dword v20, v21, s[0:3], 0 offen
v_mov_b32_e32 v20, 0x13
v_mov_b32_e32 v21, 0x48
buffer_store_dword v20, v21, s[0:3], 0 offen
v_mov_b32_e32 v20, 0x14
v_mov_b32_e32 v21, 0x4c
buffer_store_dword v20, v21, s[0:3], 0 offen
v_mov_b32_e32 v20, 0x3f800000
v_mov_b32_e32 v21, 0x50
buffer_store_dword v20, v21, s[0:3], 0 offen
v_mov_b32_e32 v20, 0x3f800000
v_mov_b32_e32 v21, 0x54
buffer_store_dword v20, v21, s[0:3], 0 offen
v_mov_b32_e32 v20, 0x3f800000
v_mov_b32_e32 v21, 0x58
buffer_store_dword v20, v21, s[0:3], 0 offen
v_mov_b32_e32 v20, 0x0
v_mov_b32_e32 v21, 0x5c
buffer_store_dword v20, v21, s[0:3], 0 offen
v_mov_b32_e32 v20, 0x0
v_mov_b32_e32 v21, 0x60
buffer_store_dword v20, v21, s[0:3], 0 offen
v_mov_b32_e32 v20, 0x0
v_mov_b32_e32 v21, 0x64
buffer_store_dword v20, v21, s[0:3], 0 offen
v_mov_b32_e32 v20, 0x0
v_mov_b32_e32 v21, 0x68
buffer_store_dword v20, v21, s[0:3], 0 offen
v_mov_b32_e32 v20, 0x0
v_mov_b32_e32 v21, 0x6c
buffer_store_dword v20, v21, s[0:3], 0 offen
v_mov_b32_e32 v20, 0x0
v_mov_b32_e32 v21, 0x70
buffer_store_dword v20, v21, s[0:3], 0 offen
v_mov_b32_e32 v20, 0x3f800000
v_mov_b32_e32 v21, 0x74
buffer_store_dword v20, v21, s[0:3], 0 offen
v_mov_b32_e32 v20, 0x3f800000
v_mov_b32_e32 v21, 0x78
buffer_store_dword v20, v21, s[0:3], 0 offen
v_mov_b32_e32 v20, 0x3f800000
v_mov_b32_e32 v21, 0x7c
buffer_store_dword v20, v21, s[0:3], 0 offen
v_mov_b32_e32 v20, 0x3f800000
v_mov_b32_e32 v21, 0x80
buffer_store_dword v20, v21, s[0:3], 0 offen
v_mov_b32_e32 v20, 0x0
v_mov_b32_e32 v21, 0x84
buffer_store_dword v20, v21, s[0:3], 0 offen
v_mov_b32_e32 v20, 0x0
v_mov_b32_e32 v21, 0x88
buffer_store_dword v20, v21, s[0:3], 0 offen
v_mov_b32_e32 v20, 0x0
v_mov_b32_e32 v21, 0x8c
buffer_store_dword v20, v21, s[0:3], 0 offen
v_mov_b32_e32 v20, 0x3f800000
v_mov_b32_e32 v21, 0x90
buffer_store_dword v20, v21, s[0:3], 0 offen
v_mov_b32_e32 v20, 0x3f800000
v_mov_b32_e32 v21, 0x94
buffer_store_dword v20, v21, s[0:3], 0 offen
v_mov_b32_e32 v20, 0x0
v_mov_b32_e32 v21, 0x98
buffer_store_dword v20, v21, s[0:3], 0 offen
v_mov_b32_e32 v20, 0x0
v_mov_b32_e32 v21, 0x9c
buffer_store_dword v20, v21, s[0:3], 0 offen
v_mov_b32_e32 v20, 0x3f800000
v_mov_b32_e32 v21, 0xa0
buffer_store_dword v20, v21, s[0:3], 0 offen
v_mov_b32_e32 v20, 0x3f800000
v_mov_b32_e32 v21, 0xa4
buffer_store_dword v20, v21, s[0:3], 0 offen
v_mov_b32_e32 v20, 0x3f800000
v_mov_b32_e32 v21, 0xa8
buffer_store_dword v20, v21, s[0:3], 0 offen
v_mov_b32_e32 v20, 0x0
v_mov_b32_e32 v21, 0xac
buffer_store_dword v20, v21, s[0:3], 0 offen
s_waitcnt_vscnt null, 0x0
v_mov_b32_e32 v1, 0x0d
v_mov_b32_e32 v2, 0
v_mov_b32_e32 v3, 0x42c80000
v_mov_b32_e32 v4, 0.5
v_mov_b32_e32 v5, 0.5
v_mov_b32_e32 v6, -1.0
v_mov_b32_e32 v7, 0
v_mov_b32_e32 v8, 0
v_mov_b32_e32 v9, 1.0
v_mov_b32_e32 v10, 0x7f800000
v_mov_b32_e32 v11, 0x7f800000
v_mov_b32_e32 v12, 1.0
s_mov_b32 s4, 0
s_mov_b32 s5, 0
s_mov_b32 s6, -1
s_mov_b32 s7, 0x81000000
image_bvh64_intersect_ray v[24:27], [v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12], s[4:7]
v_mov_b32_e32 v6, 2.0
v_mov_b32_e32 v9, -1.0
v_mov_b32_e32 v12, -1.0
image_bvh64_intersect_ray v[28:31], [v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12], s[4:7]
v_mov_b32_e32 v40, 0
buffer_store_dwordx4 v[24:27], v40, s[0:3], 0 offen
v_mov_b32_e32 v41, 16
buffer_store_dwordx4 v[28:31], v41, s[0:3], 0 offen
s_endpgm
