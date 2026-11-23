; Lanterns compute shader (stage 3): moves the rising light points one step and writes the instance records the
; draw reads. Packed by ../fssm_pack.py: the FSSM header carries the thread-group size and COMPUTE_PGM_RSRC2 at
; 0x138 (docs/design.md, "Compute shaders"). RSRC2 0x8c before D3D-448 (now 0x92, see the moves below) = six user
; SGPRs and TGID_X, so the group index is s6.
; stage: cs
; wave: 32
; group: 64,1,1
; rsrc2: 0x92
;
; User SGPRs, from the root signature [0] four 32-bit constants, [1] a table of two UAVs:
;   s0 = top (896)   s1 = range (832)   s2, s3 unused
;   s[4:5] = console VA of the table; slots are 32 bytes, a V# in the first 16
;   slot 0: V#, the state, stride 16 (x, y, z fixed point in 1/256 units; packed: rise, size and colour bits)
;   slot 1: V#, the instance records, stride 32 (offset xyz, scale, colour rgba), read by the draw's instance V#
; i = 64 * group + lane
;   y += packed & 0xff;  if (y >= top) y -= range
;   offset = (x, y, z) / 256       scale = 0.125 + 0.0625 * bit 8
;   colour = (1, (5 + bits 9-10) / 8, (1 + bits 11-12) / 8, 1)
; lanterns_particles.h does the same on the CPU.
; D3D-448: a D3D12 compute PSO gets its root arguments in the console layout (rt::rootsig::pack_stage,
; docs/d3d12_vulkan.md section 3.5): from s4 in root-signature order, a descriptor table as ONE dword (the 32-bit
; VA of its first descriptor; the heaps live below 4 GiB), a constant one dword per value; s0..s3 are the driver's.
; RSRC2 0x92 therefore counts 9 user SGPRs (4 + 5), and the thread-group ids follow at s9. These moves rebuild the
; layout the code below was written for: constants s4..s7 -> s0..s3; table s8 -> s[4:5] (high half 0); thread-group id
; s9 -> s6.
s_mov_b32 s0, s4
s_mov_b32 s1, s5
s_mov_b32 s2, s6
s_mov_b32 s3, s7
s_mov_b32 s4, s8
s_mov_b32 s5, 0
s_mov_b32 s6, s9
s_load_dwordx4 s[8:11], s[4:5], 0x0
s_load_dwordx4 s[12:15], s[4:5], 0x20
s_lshl_b32 s7, s6, 6
s_waitcnt lgkmcnt(0)
v_add_nc_u32_e32 v1, s7, v0
buffer_load_dwordx4 v[4:7], v1, s[8:11], 0 idxen
s_waitcnt vmcnt(0)
v_and_b32_e32 v8, 0xff, v7
v_add_nc_u32_e32 v5, v5, v8
v_cmp_le_i32_e32 vcc_lo, s0, v5
v_subrev_nc_u32_e32 v9, s1, v5
v_cndmask_b32_e32 v5, v5, v9, vcc_lo
buffer_store_dwordx4 v[4:7], v1, s[8:11], 0 idxen
v_cvt_f32_i32_e32 v10, v4
v_cvt_f32_i32_e32 v11, v5
v_cvt_f32_i32_e32 v12, v6
v_mul_f32_e32 v10, 0x3b800000, v10
v_mul_f32_e32 v11, 0x3b800000, v11
v_mul_f32_e32 v12, 0x3b800000, v12
v_bfe_u32 v14, v7, 8, 1
v_cvt_f32_u32_e32 v14, v14
v_mov_b32_e32 v13, 0x3e000000
v_fmac_f32_e32 v13, 0x3d800000, v14
v_mov_b32_e32 v15, 1.0
v_bfe_u32 v16, v7, 9, 2
v_add_nc_u32_e32 v16, 5, v16
v_cvt_f32_u32_e32 v16, v16
v_mul_f32_e32 v16, 0x3e000000, v16
v_bfe_u32 v17, v7, 11, 2
v_add_nc_u32_e32 v17, 1, v17
v_cvt_f32_u32_e32 v17, v17
v_mul_f32_e32 v17, 0x3e000000, v17
v_mov_b32_e32 v18, 1.0
buffer_store_dwordx4 v[10:13], v1, s[12:15], 0 idxen
buffer_store_dwordx4 v[15:18], v1, s[12:15], 0 idxen offset:16
s_endpgm
