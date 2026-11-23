; Lanterns bloom, vertical pass (stage 5). One workgroup of 128 lanes, two wave64 waves, takes 128 samples plus 4 on each
; side into LDS (16 bytes each), meets at s_barrier, and sums the 9 taps 1 8 28 56 70 56 28 8 1 (exact: the
; values are small integers). lanterns_post.h computes the same on the CPU.
; stage: cs
; wave: 64
; group: 128,1,1
; rsrc2: 0x2818a
;
; RSRC2 0x28184 before D3D-448 (now 0x2818a, see the moves below): two user SGPRs (s[0:1], the table), TGID_X and
; TGID_Y (s2, s3), LDS_SIZE 5 x 512 bytes.
; v0 = lane in the group; v1 = the sample this lane loads; v3 = its LDS entry's byte address (entry lane, + 4 below).
; Slot 0 (s[0:1] + 0): the V# of the row sums. Slot 1 (+ 0x20): the V# of the bloom (the column sums), stride 16.
; s2 = column, s3 = 128-row segment.
; D3D-448: a D3D12 compute PSO gets its root arguments in the console layout (rt::rootsig::pack_stage,
; docs/d3d12_vulkan.md section 3.5): from s4 in root-signature order, a descriptor table as ONE dword (the 32-bit
; VA of its first descriptor; the heaps live below 4 GiB), a constant one dword per value; s0..s3 are the driver's.
; RSRC2 0x2818a therefore counts 5 user SGPRs (4 + 1), and the thread-group ids follow at s5. These moves rebuild the
; layout the code below was written for: table s4 -> s[0:1] (high half 0); thread-group ids s5..s6 -> s2..s3.
s_mov_b32 s0, s4
s_mov_b32 s1, 0
s_mov_b32 s2, s5
s_mov_b32 s3, s6
s_load_dwordx4 s[4:7], s[0:1], 0x0
s_load_dwordx4 s[8:11], s[0:1], 0x20
s_lshl_b32 s16, s3, 7
v_add_nc_u32_e32 v1, s16, v0
v_lshlrev_b32_e32 v3, 4, v0
v_mul_u32_u24_e32 v20, 0x180, v1
v_add_nc_u32_e32 v20, s2, v20
s_waitcnt lgkmcnt(0)
buffer_load_dwordx4 v[4:7], v20, s[4:7], 0 idxen
s_waitcnt vmcnt(0)
ds_write_b128 v3, v[4:7] offset:64
; left halo: lanes 0-3 fill entries 0-3 with the samples 4 before them (clamped to the edge)
v_cmp_gt_u32_e32 vcc, 4, v0
s_and_saveexec_b64 s[18:19], vcc
v_subrev_nc_u32_e32 v8, 4, v1
v_max_i32_e32 v8, 0, v8
v_mul_u32_u24_e32 v8, 0x180, v8
v_add_nc_u32_e32 v8, s2, v8
buffer_load_dwordx4 v[4:7], v8, s[4:7], 0 idxen
s_waitcnt vmcnt(0)
ds_write_b128 v3, v[4:7]
s_mov_b64 exec, s[18:19]
; right halo: lanes 124-127 fill entries 132-135 with the samples 4 after them (clamped to the edge)
v_cmp_lt_u32_e32 vcc, 0x7b, v0
s_and_saveexec_b64 s[18:19], vcc
v_add_nc_u32_e32 v8, 4, v1
v_min_i32_e32 v8, 0xff, v8
v_mul_u32_u24_e32 v8, 0x180, v8
v_add_nc_u32_e32 v8, s2, v8
buffer_load_dwordx4 v[4:7], v8, s[4:7], 0 idxen
s_waitcnt vmcnt(0)
ds_write_b128 v3, v[4:7] offset:128
s_mov_b64 exec, s[18:19]
s_waitcnt lgkmcnt(0)
s_barrier
ds_read_b128 v[24:27], v3
ds_read_b128 v[28:31], v3 offset:16
ds_read_b128 v[32:35], v3 offset:32
ds_read_b128 v[36:39], v3 offset:48
ds_read_b128 v[40:43], v3 offset:64
ds_read_b128 v[44:47], v3 offset:80
ds_read_b128 v[48:51], v3 offset:96
ds_read_b128 v[52:55], v3 offset:112
ds_read_b128 v[56:59], v3 offset:128
s_waitcnt lgkmcnt(0)
v_mov_b32_e32 v10, v24
v_mov_b32_e32 v11, v25
v_mov_b32_e32 v12, v26
v_fmac_f32_e32 v10, 0x41000000, v28
v_fmac_f32_e32 v11, 0x41000000, v29
v_fmac_f32_e32 v12, 0x41000000, v30
v_fmac_f32_e32 v10, 0x41e00000, v32
v_fmac_f32_e32 v11, 0x41e00000, v33
v_fmac_f32_e32 v12, 0x41e00000, v34
v_fmac_f32_e32 v10, 0x42600000, v36
v_fmac_f32_e32 v11, 0x42600000, v37
v_fmac_f32_e32 v12, 0x42600000, v38
v_fmac_f32_e32 v10, 0x428c0000, v40
v_fmac_f32_e32 v11, 0x428c0000, v41
v_fmac_f32_e32 v12, 0x428c0000, v42
v_fmac_f32_e32 v10, 0x42600000, v44
v_fmac_f32_e32 v11, 0x42600000, v45
v_fmac_f32_e32 v12, 0x42600000, v46
v_fmac_f32_e32 v10, 0x41e00000, v48
v_fmac_f32_e32 v11, 0x41e00000, v49
v_fmac_f32_e32 v12, 0x41e00000, v50
v_fmac_f32_e32 v10, 0x41000000, v52
v_fmac_f32_e32 v11, 0x41000000, v53
v_fmac_f32_e32 v12, 0x41000000, v54
v_add_f32_e32 v10, v10, v56
v_add_f32_e32 v11, v11, v57
v_add_f32_e32 v12, v12, v58
v_mov_b32_e32 v13, 0
buffer_store_dwordx4 v[10:13], v20, s[8:11], 0 idxen
s_endpgm
