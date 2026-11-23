; Lanterns ray-traced shadows (stage 11). One 8x8 tile per workgroup (two wave32 waves). Each lane reads its pixel's
; G-buffer (stage 9's targets), rebuilds the world position p, and traces one shadow ray from p + 0.02 n towards a
; directional light through the TLAS and BLASes that BuildRaytracingAccelerationStructure wrote (XDXR BVH4,
; docs/bvh.md 3.1, 7a). It stops at the first triangle hit (any hit is enough for a shadow).
; The traversal is a per-lane stack in LDS (32 entries per lane), one image_bvh_intersect_ray per box or triangle node:
;   TLAS box nodes: with the TLAS descriptor and the world ray; children that are not 0xffffffff are pushed.
;   Instance nodes (type 6): the 128-byte record is read through a raw V# over the TLAS (+48 ID | mask << 24, the root
;   pointer with two flag bits, the BLAS address >> 8; +64 world-to-object 3x4); an instance whose mask misses the
;   ray mask is skipped, otherwise the object-space ray is made (the direction is transformed, not normalised, so t is
;   the world t), a return marker (0xfffffffe) and the BLAS root are pushed, the lane keeps the record's BLAS address
;   (v72) and switches to the BLAS.
;   BLAS nodes: lanes may be in different BLASes, so the descriptor is built per BLAS in a waterfall loop
;   (v_readfirstlane_b32 of one lane's address, every lane with the same address, repeat); a triangle hit with
;   t = t_num / t_denom <= tmax occludes.
; Every push checks the stack: a lane whose push would pass entry 32 stops as overflowed (v71) and writes nothing past
; its own stack. A lane stops when its stack is empty, it is occluded or it overflowed; the loop stops after s17 steps
; whatever happens. A lane still tracing then, or overflowed, writes magenta (0xffff00ff), which no reference allows.
; Output per pixel: min(255, (albedo8 * (occluded ? 96 : 256)) >> 8), or the background where the view depth is 0, and
; the lane's step count, with bit 31 set if it overflowed.
; stage: cs
; wave: 32
; group: 64,1,1
; rsrc2: 0x801aa
;
; RSRC2 0x801a4 before D3D-448 (now 0x801aa, see the moves below): 18 user SGPRs, TGID_X (s18), TGID_Y (s19), LDS_SIZE
; 16 x 512 bytes (64 lanes x 32 dwords).
; s[0:1] table: 0 albedo T#, 1 normal T#, 2 view depth T#, 3 the TLAS as a raw V#, 4 output (raw), 5 step counts (raw)
; s2..s13: eye, d0, R, U as stage 9; s14 TLAS address >> 8; s15 unused (0); s16 ray mask; s17 step cap (2048)
; Light direction (to the light) 0.544949, 0.778499, 0.311400, its reciprocal as literals; tmax 64.
; D3D-448: a D3D12 compute PSO gets its root arguments in the console layout (rt::rootsig::pack_stage,
; docs/d3d12_vulkan.md section 3.5): from s4 in root-signature order, a descriptor table as ONE dword (the 32-bit
; VA of its first descriptor; the heaps live below 4 GiB), a constant one dword per value; s0..s3 are the driver's.
; RSRC2 0x801aa therefore counts 21 user SGPRs (4 + 17), and the thread-group ids follow at s21. These moves rebuild
; the
; layout the code below was written for: table s4 -> s[0:1] (high half 0); constants s5..s20 -> s2..s17; thread-group
; ids s21..s22 -> s18..s19.
s_mov_b32 s0, s4
s_mov_b32 s1, 0
s_mov_b32 s2, s5
s_mov_b32 s3, s6
s_mov_b32 s4, s7
s_mov_b32 s5, s8
s_mov_b32 s6, s9
s_mov_b32 s7, s10
s_mov_b32 s8, s11
s_mov_b32 s9, s12
s_mov_b32 s10, s13
s_mov_b32 s11, s14
s_mov_b32 s12, s15
s_mov_b32 s13, s16
s_mov_b32 s14, s17
s_mov_b32 s15, s18
s_mov_b32 s16, s19
s_mov_b32 s17, s20
s_mov_b32 s18, s21
s_mov_b32 s19, s22
s_load_dwordx8 s[20:27], s[0:1], 0x0
s_load_dwordx8 s[28:35], s[0:1], 0x20
s_load_dwordx8 s[36:43], s[0:1], 0x40
s_load_dwordx4 s[44:47], s[0:1], 0x60
s_load_dwordx4 s[48:51], s[0:1], 0x80
s_load_dwordx4 s[52:55], s[0:1], 0xa0
s_mov_b32 s74, exec_lo
v_and_b32_e32 v1, 7, v0
v_lshrrev_b32_e32 v2, 3, v0
s_lshl_b32 s56, s18, 3
s_lshl_b32 s57, s19, 3
v_add_nc_u32_e32 v1, s56, v1
v_add_nc_u32_e32 v2, s57, v2
s_waitcnt lgkmcnt(0)
s_buffer_load_dword s68, s[44:47], 0x30
image_load v[8:11], v[1:2], s[20:27] dmask:0xf dim:SQ_RSRC_IMG_2D
image_load v[12:14], v[1:2], s[28:35] dmask:0x7 dim:SQ_RSRC_IMG_2D
image_load v15, v[1:2], s[36:43] dmask:0x1 dim:SQ_RSRC_IMG_2D
v_cvt_f32_u32_e32 v16, v1
v_cvt_f32_u32_e32 v17, v2
v_add_f32_e32 v16, 0xc33f8000, v16
v_add_f32_e32 v17, 0xc2ff0000, v17
v_fma_f32 v18, v16, s8, s5
v_fmac_f32_e32 v18, s11, v17
v_fma_f32 v19, v16, s9, s6
v_fmac_f32_e32 v19, s12, v17
v_fma_f32 v20, v16, s10, s7
v_fmac_f32_e32 v20, s13, v17
s_waitcnt vmcnt(0)
v_fma_f32 v16, v15, v18, s2
v_fma_f32 v17, v15, v19, s3
v_fma_f32 v18, v15, v20, s4
v_fma_f32 v12, v12, 2.0, -1.0
v_fma_f32 v13, v13, 2.0, -1.0
v_fma_f32 v14, v14, 2.0, -1.0
v_fmac_f32_e32 v16, 0x3ca3d70a, v12
v_fmac_f32_e32 v17, 0x3ca3d70a, v13
v_fmac_f32_e32 v18, 0x3ca3d70a, v14
s_mov_b32 s60, s14
s_mov_b32 s61, 0x80000000
s_mov_b32 s62, 0
s_mov_b32 s63, 0
s_mov_b32 s65, 0x80000000
s_mov_b32 s66, 0
s_mov_b32 s67, 0
v_lshlrev_b32_e32 v19, 7, v0
v_mov_b32_e32 v21, 0
v_mov_b32_e32 v22, 0
v_mov_b32_e32 v23, 0
s_waitcnt lgkmcnt(0)
v_mov_b32_e32 v24, s68
ds_write_b32 v19, v24
v_cmp_lt_f32_e32 vcc_lo, 0, v15
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_mov_b32_e32 v25, 0x3f0b81cb
v_mov_b32_e32 v26, 0x3f474bb5
v_mov_b32_e32 v27, 0x3e9f6fc4
v_mov_b32_e32 v28, 0x3feae25f
v_mov_b32_e32 v29, 0x3fa46b42
v_mov_b32_e32 v30, 0x404d8613
v_mov_b32_e32 v31, 0x42800000
s_mov_b32 s70, 0
v_mov_b32_e32 v71, 0
v_mov_b32_e32 v72, 0
s_waitcnt lgkmcnt(0)
; BB_loop: (byte 400)
v_cmp_ne_u32_e64 s72, 0, v20
v_cmp_eq_u32_e64 s73, 0, v22
s_and_b32 s72, s72, s73
v_cmp_eq_u32_e64 s73, 0, v71
s_and_b32 s72, s72, s73
s_cmp_eq_u32 s72, 0
s_cbranch_scc1 266   ; to BB_done
s_cmp_ge_u32 s70, s17
s_cbranch_scc1 264   ; to BB_done
s_add_u32 s70, s70, 1
s_mov_b32 exec_lo, s72
v_add_nc_u32_e32 v23, 1, v23
v_subrev_nc_u32_e32 v20, 1, v20
v_lshl_add_u32 v41, v20, 2, v19
ds_read_b32 v42, v41
s_waitcnt lgkmcnt(0)
v_and_b32_e32 v43, 7, v42
v_cmp_eq_u32_e32 vcc_lo, -2, v42
v_cndmask_b32_e64 v21, v21, 0, vcc_lo
v_cmp_ne_u32_e64 s77, -2, v42
v_cmp_eq_u32_e64 s76, 6, v43
s_and_b32 s76, s76, s77
s_and_saveexec_b32 s78, s76
v_and_b32_e32 v44, -8, v42
v_lshlrev_b32_e32 v44, 3, v44
buffer_load_dwordx4 v[45:48], v44, s[44:47], 0 offen offset:48
buffer_load_dwordx4 v[49:52], v44, s[44:47], 0 offen offset:64
buffer_load_dwordx4 v[53:56], v44, s[44:47], 0 offen offset:80
buffer_load_dwordx4 v[57:60], v44, s[44:47], 0 offen offset:96
s_waitcnt vmcnt(0)
v_lshrrev_b32_e32 v61, 24, v45
v_and_b32_e32 v61, s16, v61
v_cmp_ne_u32_e64 s79, 0, v61
s_and_saveexec_b32 s80, s79
v_fma_f32 v32, v49, v16, v52
v_fmac_f32_e32 v32, v50, v17
v_fmac_f32_e32 v32, v51, v18
v_fma_f32 v33, v53, v16, v56
v_fmac_f32_e32 v33, v54, v17
v_fmac_f32_e32 v33, v55, v18
v_fma_f32 v34, v57, v16, v60
v_fmac_f32_e32 v34, v58, v17
v_fmac_f32_e32 v34, v59, v18
v_mul_f32_e32 v35, v49, v25
v_fmac_f32_e32 v35, v50, v26
v_fmac_f32_e32 v35, v51, v27
v_mul_f32_e32 v36, v53, v25
v_fmac_f32_e32 v36, v54, v26
v_fmac_f32_e32 v36, v55, v27
v_mul_f32_e32 v37, v57, v25
v_fmac_f32_e32 v37, v58, v26
v_fmac_f32_e32 v37, v59, v27
v_rcp_f32_e32 v38, v35
v_rcp_f32_e32 v39, v36
v_rcp_f32_e32 v40, v37
v_cmp_gt_u32_e64 s86, 31, v20
s_andn2_b32 s88, exec_lo, s86
v_cndmask_b32_e64 v71, v71, 1, s88
s_and_b32 exec_lo, exec_lo, s86
v_lshl_add_u32 v41, v20, 2, v19
v_mov_b32_e32 v62, -2
ds_write_b32 v41, v62
v_and_b32_e32 v63, 0x3fffffff, v47
ds_write_b32 v41, v63 offset:4
v_add_nc_u32_e32 v20, 2, v20
v_mov_b32_e32 v21, 1
v_mov_b32_e32 v72, v48
s_mov_b32 exec_lo, s78
v_cmp_ne_u32_e64 s76, 6, v43
s_and_b32 s76, s76, s77
s_and_b32 s76, s76, exec_lo
v_cmp_eq_u32_e64 s81, 0, v21
s_and_b32 s82, s76, s81
s_and_saveexec_b32 s83, s82
v_mov_b32_e32 v84, v42
v_mov_b32_e32 v85, v31
v_mov_b32_e32 v86, v16
v_mov_b32_e32 v87, v17
v_mov_b32_e32 v88, v18
v_mov_b32_e32 v89, v25
v_mov_b32_e32 v90, v26
v_mov_b32_e32 v91, v27
v_mov_b32_e32 v92, v28
v_mov_b32_e32 v93, v29
v_mov_b32_e32 v94, v30
image_bvh_intersect_ray v[64:67], v[84:94], s[60:63]
s_waitcnt vmcnt(0)
v_cmp_ne_u32_e64 s86, -1, v64
v_cmp_gt_u32_e64 s87, 32, v20
s_and_b32 s88, s86, s87
s_andn2_b32 s86, s86, s87
v_cndmask_b32_e64 v71, v71, 1, s86
s_mov_b32 s89, exec_lo
s_and_b32 exec_lo, exec_lo, s88
v_lshl_add_u32 v41, v20, 2, v19
ds_write_b32 v41, v64
v_add_nc_u32_e32 v20, 1, v20
s_mov_b32 exec_lo, s89
v_cmp_ne_u32_e64 s86, -1, v65
v_cmp_gt_u32_e64 s87, 32, v20
s_and_b32 s88, s86, s87
s_andn2_b32 s86, s86, s87
v_cndmask_b32_e64 v71, v71, 1, s86
s_mov_b32 s89, exec_lo
s_and_b32 exec_lo, exec_lo, s88
v_lshl_add_u32 v41, v20, 2, v19
ds_write_b32 v41, v65
v_add_nc_u32_e32 v20, 1, v20
s_mov_b32 exec_lo, s89
v_cmp_ne_u32_e64 s86, -1, v66
v_cmp_gt_u32_e64 s87, 32, v20
s_and_b32 s88, s86, s87
s_andn2_b32 s86, s86, s87
v_cndmask_b32_e64 v71, v71, 1, s86
s_mov_b32 s89, exec_lo
s_and_b32 exec_lo, exec_lo, s88
v_lshl_add_u32 v41, v20, 2, v19
ds_write_b32 v41, v66
v_add_nc_u32_e32 v20, 1, v20
s_mov_b32 exec_lo, s89
v_cmp_ne_u32_e64 s86, -1, v67
v_cmp_gt_u32_e64 s87, 32, v20
s_and_b32 s88, s86, s87
s_andn2_b32 s86, s86, s87
v_cndmask_b32_e64 v71, v71, 1, s86
s_mov_b32 s89, exec_lo
s_and_b32 exec_lo, exec_lo, s88
v_lshl_add_u32 v41, v20, 2, v19
ds_write_b32 v41, v67
v_add_nc_u32_e32 v20, 1, v20
s_mov_b32 exec_lo, s89
s_mov_b32 exec_lo, s83
s_andn2_b32 s90, s76, s81
; BB_wf: (byte 1112)
s_cmp_eq_u32 s90, 0
s_cbranch_scc1 94   ; to BB_next
s_mov_b32 exec_lo, s90
v_readfirstlane_b32 s64, v72
v_cmp_eq_u32_e64 s91, s64, v72
s_and_b32 exec_lo, s90, s91
s_andn2_b32 s90, s90, s91
v_mov_b32_e32 v84, v42
v_mov_b32_e32 v85, v31
v_mov_b32_e32 v86, v32
v_mov_b32_e32 v87, v33
v_mov_b32_e32 v88, v34
v_mov_b32_e32 v89, v35
v_mov_b32_e32 v90, v36
v_mov_b32_e32 v91, v37
v_mov_b32_e32 v92, v38
v_mov_b32_e32 v93, v39
v_mov_b32_e32 v94, v40
image_bvh_intersect_ray v[64:67], v[84:94], s[64:67]
s_waitcnt vmcnt(0)
v_cmp_gt_u32_e64 s84, 4, v43
s_and_saveexec_b32 s85, s84
v_rcp_f32_e32 v68, v65
v_mul_f32_e32 v68, v64, v68
v_cmp_ge_f32_e32 vcc_lo, v31, v68
v_cndmask_b32_e64 v22, v22, 1, vcc_lo
s_andn2_b32 exec_lo, s85, s84
v_cmp_ne_u32_e64 s86, -1, v64
v_cmp_gt_u32_e64 s87, 32, v20
s_and_b32 s88, s86, s87
s_andn2_b32 s86, s86, s87
v_cndmask_b32_e64 v71, v71, 1, s86
s_mov_b32 s89, exec_lo
s_and_b32 exec_lo, exec_lo, s88
v_lshl_add_u32 v41, v20, 2, v19
ds_write_b32 v41, v64
v_add_nc_u32_e32 v20, 1, v20
s_mov_b32 exec_lo, s89
v_cmp_ne_u32_e64 s86, -1, v65
v_cmp_gt_u32_e64 s87, 32, v20
s_and_b32 s88, s86, s87
s_andn2_b32 s86, s86, s87
v_cndmask_b32_e64 v71, v71, 1, s86
s_mov_b32 s89, exec_lo
s_and_b32 exec_lo, exec_lo, s88
v_lshl_add_u32 v41, v20, 2, v19
ds_write_b32 v41, v65
v_add_nc_u32_e32 v20, 1, v20
s_mov_b32 exec_lo, s89
v_cmp_ne_u32_e64 s86, -1, v66
v_cmp_gt_u32_e64 s87, 32, v20
s_and_b32 s88, s86, s87
s_andn2_b32 s86, s86, s87
v_cndmask_b32_e64 v71, v71, 1, s86
s_mov_b32 s89, exec_lo
s_and_b32 exec_lo, exec_lo, s88
v_lshl_add_u32 v41, v20, 2, v19
ds_write_b32 v41, v66
v_add_nc_u32_e32 v20, 1, v20
s_mov_b32 exec_lo, s89
v_cmp_ne_u32_e64 s86, -1, v67
v_cmp_gt_u32_e64 s87, 32, v20
s_and_b32 s88, s86, s87
s_andn2_b32 s86, s86, s87
v_cndmask_b32_e64 v71, v71, 1, s86
s_mov_b32 s89, exec_lo
s_and_b32 exec_lo, exec_lo, s88
v_lshl_add_u32 v41, v20, 2, v19
ds_write_b32 v41, v67
v_add_nc_u32_e32 v20, 1, v20
s_mov_b32 exec_lo, s89
s_branch -96   ; to BB_wf
; BB_next: (byte 1496)
s_mov_b32 exec_lo, s74
s_branch -276   ; to BB_loop
; BB_done: (byte 1504)
s_mov_b32 exec_lo, s74
v_fma_f32 v8, v8, 0x437f0000, 0.5
v_fma_f32 v9, v9, 0x437f0000, 0.5
v_fma_f32 v10, v10, 0x437f0000, 0.5
v_cvt_u32_f32_e32 v8, v8
v_cvt_u32_f32_e32 v9, v9
v_cvt_u32_f32_e32 v10, v10
v_cmp_ne_u32_e32 vcc_lo, 0, v22
v_mov_b32_e32 v70, 0x60
v_cndmask_b32_e32 v70, 0x100, v70, vcc_lo
v_mul_u32_u24_e32 v8, v8, v70
v_mul_u32_u24_e32 v9, v9, v70
v_mul_u32_u24_e32 v10, v10, v70
v_lshrrev_b32_e32 v8, 8, v8
v_lshrrev_b32_e32 v9, 8, v9
v_lshrrev_b32_e32 v10, 8, v10
v_min_u32_e32 v8, 0xff, v8
v_min_u32_e32 v9, 0xff, v9
v_min_u32_e32 v10, 0xff, v10
v_lshlrev_b32_e32 v9, 8, v9
v_lshlrev_b32_e32 v10, 16, v10
v_or3_b32 v8, v8, v9, v10
v_or_b32_e32 v8, 0xff000000, v8
v_cmp_lt_f32_e32 vcc_lo, 0, v15
v_cndmask_b32_e32 v8, 0xff201010, v8, vcc_lo
v_cmp_ne_u32_e64 s72, 0, v20
v_cmp_eq_u32_e64 s73, 0, v22
s_and_b32 s72, s72, s73
v_cmp_ne_u32_e64 s73, 0, v71
s_or_b32 s72, s72, s73
v_cndmask_b32_e64 v8, v8, 0xffff00ff, s72
v_lshlrev_b32_e32 v73, 31, v71
v_or_b32_e32 v23, v23, v73
v_mul_u32_u24_e32 v43, 0x180, v2
v_add_nc_u32_e32 v43, v43, v1
v_lshlrev_b32_e32 v43, 2, v43
buffer_store_dword v8, v43, s[48:51], 0 offen
buffer_store_dword v23, v43, s[52:55], 0 offen
s_endpgm
