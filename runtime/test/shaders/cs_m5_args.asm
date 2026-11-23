; group: 64,1,1
; rsrc2: 0x808c
; fssm-at: 0x150
; fssm-data: 0x53475241 0x3d354d5f
; M5 compute shader: writes the DISPATCH_MESH records and the count an ExecuteIndirect reads right after it, as a
; GPU-culling pass would. Run: d3d12_m5_mesh_test, case (d). User data, from the root signature: [0] four 32-bit
; constants, [1] a descriptor table of two UAVs. RSRC2 0x808c = six user SGPRs and TGID_X (one group is dispatched):
;   s0 = records written (N, at most 64)   s1 = the count word   s2, s3 = unused
;   s[4:5] = console VA of the table's first slot (32 bytes a slot, a V# in its first 16 bytes):
;   slot 0 (+0x00): V#, raw buffer, the records (written)
;   slot 1 (+0x20): V#, raw buffer, the count (written)
; lane k < N: record k = { 1 + (k & 3), 1, 1 } at byte offset 12 * k; lane 0 also writes the count at offset 0.
s_load_dwordx4 s[8:11], s[4:5], 0x0
s_load_dwordx4 s[12:15], s[4:5], 0x20
s_waitcnt lgkmcnt(0)
v_and_b32_e32 v1, 3, v0
v_add_nc_u32_e32 v1, 1, v1
v_mov_b32_e32 v2, 1
v_mov_b32_e32 v3, 1
v_mul_u32_u24_e32 v4, 12, v0
v_cmpx_gt_u32_e32 s0, v0
buffer_store_dwordx3 v[1:3], v4, s[8:11], 0 offen
v_cmpx_eq_u32_e32 0, v0
v_mov_b32_e32 v5, 0
v_mov_b32_e32 v6, s1
buffer_store_dword v6, v5, s[12:15], 0 offen
s_endpgm
