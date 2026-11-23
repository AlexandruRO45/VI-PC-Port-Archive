; Lanterns tiled deferred lighting (stage 9). One workgroup of 256 lanes (eight wave32 waves) per 16x16 screen tile.
; 1. Lane 0 sets the tile's box in LDS (min keys FLT_MAX, max keys -FLT_MAX: an empty tile culls every light).
; 2. Each lane reads its pixel's G-buffer (albedo, normal, view depth t), rebuilds p = eye + t (d0 + cx R + cy U), and,
;    if t > 0, folds p into the box with ds_min_i32/ds_max_i32 on order-preserving keys (negative floats: bits ^ 0x7fffffff).
; 3. Lane i tests light i's sphere against the box; a kept light takes a slot with ds_add_rtn_u32 and writes its index
;    to the list (mode 1: every light is kept). Lane 0 stores the tile's count; lane i < count stores list entry i.
; 4. Each lane loops over the list (a scalar loop: the count is the group's) and adds q * colour in integers, then
;    writes min(255, (albedo8 * (384 + sum)) >> 12) per channel, or the background where t is 0. lanterns_deferred.h.
; stage: cs
; wave: 32
; group: 256,1,1
; rsrc2: 0x181aa
;
; RSRC2 0x181a4 before D3D-448 (now 0x181aa, see the moves below): 18 user SGPRs, TGID_X (s18), TGID_Y (s19), LDS_SIZE
; 3 x 512 bytes.
; s[0:1] table: slot 0 albedo T#, 1 normal T#, 2 view depth T#, 3 lights V# (stride 32), 4 output (raw), 5 tile counts (raw),
;   6 tile lists (raw: 256 dwords a tile, the kept light indices in slot order, read back to check the exact set)
; s2..s17 constants: eye xyz, d0 xyz, R xyz, U xyz, 1/r^2, mode, box test r^2, 0
; LDS: bytes 0-11 min keys, 12-23 max keys, 24 the count, 32.. the light list
; D3D-448: a D3D12 compute PSO gets its root arguments in the console layout (rt::rootsig::pack_stage,
; docs/d3d12_vulkan.md section 3.5): from s4 in root-signature order, a descriptor table as ONE dword (the 32-bit
; VA of its first descriptor; the heaps live below 4 GiB), a constant one dword per value; s0..s3 are the driver's.
; RSRC2 0x181aa therefore counts 21 user SGPRs (4 + 17), and the thread-group ids follow at s21. These moves rebuild
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
s_load_dwordx4 s[64:67], s[0:1], 0xc0
v_and_b32_e32 v1, 15, v0
v_lshrrev_b32_e32 v2, 4, v0
s_lshl_b32 s56, s18, 4
s_lshl_b32 s57, s19, 4
v_add_nc_u32_e32 v1, s56, v1
v_add_nc_u32_e32 v2, s57, v2
v_mov_b32_e32 v6, 0
v_cmp_eq_u32_e32 vcc_lo, 0, v0
s_and_saveexec_b32 s58, vcc_lo
v_mov_b32_e32 v3, 0x7f7fffff
v_mov_b32_e32 v4, 0x80800000
ds_write_b32 v6, v3
ds_write_b32 v6, v3 offset:4
ds_write_b32 v6, v3 offset:8
ds_write_b32 v6, v4 offset:12
ds_write_b32 v6, v4 offset:16
ds_write_b32 v6, v4 offset:20
ds_write_b32 v6, v6 offset:24
s_mov_b32 exec_lo, s58
s_waitcnt lgkmcnt(0)
s_barrier
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
v_cmp_lt_f32_e32 vcc_lo, 0, v15
s_and_saveexec_b32 s58, vcc_lo
v_ashrrev_i32_e32 v24, 31, v16
v_and_b32_e32 v24, 0x7fffffff, v24
v_xor_b32_e32 v21, v16, v24
v_ashrrev_i32_e32 v24, 31, v17
v_and_b32_e32 v24, 0x7fffffff, v24
v_xor_b32_e32 v22, v17, v24
v_ashrrev_i32_e32 v24, 31, v18
v_and_b32_e32 v24, 0x7fffffff, v24
v_xor_b32_e32 v23, v18, v24
ds_min_i32 v6, v21
ds_min_i32 v6, v22 offset:4
ds_min_i32 v6, v23 offset:8
ds_max_i32 v6, v21 offset:12
ds_max_i32 v6, v22 offset:16
ds_max_i32 v6, v23 offset:20
s_mov_b32 exec_lo, s58
s_waitcnt lgkmcnt(0)
s_barrier
ds_read_b32 v24, v6
ds_read_b32 v25, v6 offset:4
ds_read_b32 v26, v6 offset:8
ds_read_b32 v27, v6 offset:12
ds_read_b32 v28, v6 offset:16
ds_read_b32 v29, v6 offset:20
buffer_load_dwordx4 v[32:35], v0, s[44:47], 0 idxen
s_waitcnt lgkmcnt(0)
v_ashrrev_i32_e32 v30, 31, v24
v_and_b32_e32 v30, 0x7fffffff, v30
v_xor_b32_e32 v24, v24, v30
v_ashrrev_i32_e32 v30, 31, v25
v_and_b32_e32 v30, 0x7fffffff, v30
v_xor_b32_e32 v25, v25, v30
v_ashrrev_i32_e32 v30, 31, v26
v_and_b32_e32 v30, 0x7fffffff, v30
v_xor_b32_e32 v26, v26, v30
v_ashrrev_i32_e32 v30, 31, v27
v_and_b32_e32 v30, 0x7fffffff, v30
v_xor_b32_e32 v27, v27, v30
v_ashrrev_i32_e32 v30, 31, v28
v_and_b32_e32 v30, 0x7fffffff, v30
v_xor_b32_e32 v28, v28, v30
v_ashrrev_i32_e32 v30, 31, v29
v_and_b32_e32 v30, 0x7fffffff, v30
v_xor_b32_e32 v29, v29, v30
s_waitcnt vmcnt(0)
v_sub_f32_e32 v30, v24, v32
v_sub_f32_e32 v31, v32, v27
v_max_f32_e32 v30, v30, v31
v_max_f32_e32 v30, 0, v30
v_mul_f32_e32 v36, v30, v30
v_sub_f32_e32 v30, v25, v33
v_sub_f32_e32 v31, v33, v28
v_max_f32_e32 v30, v30, v31
v_max_f32_e32 v30, 0, v30
v_fmac_f32_e32 v36, v30, v30
v_sub_f32_e32 v30, v26, v34
v_sub_f32_e32 v31, v34, v29
v_max_f32_e32 v30, v30, v31
v_max_f32_e32 v30, 0, v30
v_fmac_f32_e32 v36, v30, v30
v_cmp_ge_f32_e32 vcc_lo, s16, v36
s_mov_b32 s59, vcc_lo
s_cmp_eq_u32 s15, 1
s_cselect_b32 s59, exec_lo, s59
s_and_saveexec_b32 s58, s59
v_mov_b32_e32 v37, 1
ds_add_rtn_u32 v38, v6, v37 offset:24
s_waitcnt lgkmcnt(0)
v_lshlrev_b32_e32 v38, 2, v38
ds_write_b32 v38, v0 offset:32
s_mov_b32 exec_lo, s58
s_waitcnt lgkmcnt(0)
s_barrier
ds_read_b32 v39, v6 offset:24
s_waitcnt lgkmcnt(0)
v_readfirstlane_b32 s60, v39
v_cmp_eq_u32_e32 vcc_lo, 0, v0
s_and_saveexec_b32 s58, vcc_lo
s_mul_i32 s61, s19, 24
s_add_u32 s61, s61, s18
s_lshl_b32 s61, s61, 2
v_mov_b32_e32 v40, s61
buffer_store_dword v39, v40, s[52:55], 0 offen
s_mov_b32 exec_lo, s58
v_cmp_gt_u32_e32 vcc_lo, s60, v0
s_and_saveexec_b32 s58, vcc_lo
v_lshlrev_b32_e32 v43, 2, v0
ds_read_b32 v44, v43 offset:32
s_mul_i32 s62, s19, 24
s_add_u32 s62, s62, s18
s_lshl_b32 s62, s62, 10
v_add_nc_u32_e32 v43, s62, v43
s_waitcnt lgkmcnt(0)
buffer_store_dword v44, v43, s[64:67], 0 offen
s_mov_b32 exec_lo, s58
v_fma_f32 v12, v12, 2.0, -1.0
v_fma_f32 v13, v13, 2.0, -1.0
v_fma_f32 v14, v14, 2.0, -1.0
v_mov_b32_e32 v40, 0
v_mov_b32_e32 v41, 0
v_mov_b32_e32 v42, 0
s_mov_b32 s61, 0
s_cmp_eq_u32 s60, 0
s_cbranch_scc1 36   ; to BB_shaded
; BB_light: (byte 836)
v_mov_b32_e32 v43, s61
v_lshlrev_b32_e32 v43, 2, v43
ds_read_b32 v43, v43 offset:32
s_waitcnt lgkmcnt(0)
buffer_load_dwordx4 v[32:35], v43, s[44:47], 0 idxen
buffer_load_dwordx4 v[28:31], v43, s[44:47], 0 idxen offset:16
s_waitcnt vmcnt(0)
v_sub_f32_e32 v36, v32, v16
v_sub_f32_e32 v37, v33, v17
v_sub_f32_e32 v38, v34, v18
v_mul_f32_e32 v39, v36, v36
v_fmac_f32_e32 v39, v37, v37
v_fmac_f32_e32 v39, v38, v38
v_mul_f32_e32 v44, v12, v36
v_fmac_f32_e32 v44, v13, v37
v_fmac_f32_e32 v44, v14, v38
v_fma_f32 v45, -v39, s14, 1.0
v_mul_f32_e32 v45, 0x42800000, v45
v_max_f32_e32 v45, 0, v45
v_cvt_u32_f32_e32 v45, v45
v_cmp_lt_f32_e32 vcc_lo, 0, v44
v_cndmask_b32_e32 v45, 0, v45, vcc_lo
v_mad_u32_u24 v40, v45, v28, v40
v_mad_u32_u24 v41, v45, v29, v41
v_mad_u32_u24 v42, v45, v30, v42
s_add_u32 s61, s61, 1
s_cmp_lt_u32 s61, s60
s_cbranch_scc1 -36   ; to BB_light
; BB_shaded: (byte 980)
v_fma_f32 v8, v8, 0x437f0000, 0.5
v_fma_f32 v9, v9, 0x437f0000, 0.5
v_fma_f32 v10, v10, 0x437f0000, 0.5
v_cvt_u32_f32_e32 v8, v8
v_cvt_u32_f32_e32 v9, v9
v_cvt_u32_f32_e32 v10, v10
v_add_nc_u32_e32 v40, 0x180, v40
v_add_nc_u32_e32 v41, 0x180, v41
v_add_nc_u32_e32 v42, 0x180, v42
v_mul_lo_u32 v40, v8, v40
v_mul_lo_u32 v41, v9, v41
v_mul_lo_u32 v42, v10, v42
v_lshrrev_b32_e32 v40, 12, v40
v_lshrrev_b32_e32 v41, 12, v41
v_lshrrev_b32_e32 v42, 12, v42
v_min_u32_e32 v40, 0xff, v40
v_min_u32_e32 v41, 0xff, v41
v_min_u32_e32 v42, 0xff, v42
v_lshlrev_b32_e32 v41, 8, v41
v_lshlrev_b32_e32 v42, 16, v42
v_or3_b32 v40, v40, v41, v42
v_or_b32_e32 v40, 0xff000000, v40
v_cmp_lt_f32_e32 vcc_lo, 0, v15
v_cndmask_b32_e32 v40, 0xff201010, v40, vcc_lo
v_mul_u32_u24_e32 v43, 0x180, v2
v_add_nc_u32_e32 v43, v43, v1
v_lshlrev_b32_e32 v43, 2, v43
buffer_store_dword v40, v43, s[48:51], 0 offen
s_endpgm
