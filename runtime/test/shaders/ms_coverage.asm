; Mesh shader (NGG, wave64, 128 threads): fetches six vertices through a hand-built 4-dword V# in the heap
; (table pointer in s[8:9], V# base 0x2000, stride 16, six records, structured) and emits two triangles.
; It exports pos0 from the fetched (x,y,z,w) and a constant UV in attr0 for ps_texture_sample.
; Needs a DXBC with PSV0 (128 threads, 64/64 outputs, triangles).
; Run: VKH_MESH=1 VKH_COVERAGE_MASK=<mask> VKH_SGPRS=0x100,0,0,0,0,0,0,0,0x1000,0
;      VKH_HEAP=0x1000=0x2000,0x1004=0x100000,0x1008=6,0x100c=0x10000000,<vertex words>
;      VKH_TEX_RGBA=<texture> VKH_TSHARP_VA=0x100 VKH_SSHARP=0x92,0xc00000,0x14500000,0
;      vk_harness ms_coverage.spv ps_texture_sample.spv out.raw 64 64
; The six NDC vertices span (-0.8,-0.8)..(0.8,0.8): 52x52 covered pixels (x and y 6..57) of a 64x64 viewport,
; so the CPU projection and the stencil mask must agree exactly (IoU 1.0, zero missing, zero extra).
; v5 = first launch vertex ID of the lane = groupIndex * NumThreads + local index (one group: 0..127).
; V# is at heap VA 0x1000, pointed to by s[8:9].
s_load_dwordx4 s[12:15], s[8:9], 0x0
s_waitcnt lgkmcnt(0)
; Fetch vertex v5 (six records, 16 bytes each) from base VA 0x2000.
v_mov_b32_e32 v0, v5
buffer_load_dwordx4 v[0:3], v0, s[12:15], 0 idxen
s_waitcnt vmcnt(0)
v_mov_b32_e32 v4, 0.25
v_mov_b32_e32 v6, 0.25
v_mov_b32_e32 v7, 0
v_mov_b32_e32 v8, 1.0
s_mov_b32 m0, 0x2006
s_sendmsg sendmsg(MSG_GS_ALLOC_REQ)
; Vertices 0..5 export pos0 and attr0. v5 stays the launch index for the lane checks.
v_cmpx_gt_u32_e32 6, v5
exp pos0 v0, v1, v2, v3 done
exp param0 v4, v6, v7, v8
; Primitive 0 = (0,1,2), primitive 1 = (3,4,5). Threads 0 and 1 export them.
s_mov_b32 exec_lo, -1
s_mov_b32 exec_hi, -1
v_cmpx_eq_u32_e32 0, v5
v_mov_b32_e32 v12, 0x200400
exp prim v12, off, off, off
s_mov_b32 exec_lo, -1
s_mov_b32 exec_hi, -1
v_cmpx_eq_u32_e32 1, v5
v_mov_b32_e32 v12, 0x501003
exp prim v12, off, off, off done
s_endpgm
