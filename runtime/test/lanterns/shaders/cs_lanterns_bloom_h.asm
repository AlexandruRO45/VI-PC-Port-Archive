; Lanterns bloom, horizontal pass (stage 5). One workgroup of 128 lanes, four wave32 waves, takes 128 samples plus 4 on each
; side into LDS (16 bytes each), meets at s_barrier, and sums the 9 taps 1 8 28 56 70 56 28 8 1 (exact: the
; values are small integers). lanterns_post.h computes the same on the CPU.
; Every sample a lane sums was loaded by a lane of ANOTHER wave: lane i loads sample (i + 64) mod 128 of the segment,
; and the left halo comes from lanes 64-67, the right one from lanes 60-63. So the sums read only what other waves
; wrote, and only the s_barrier orders those writes before the reads. The pass is wave32 on purpose: translated with
; --wave64-emulate, a wave64 program runs in rounds that already order every LDS write before a later LDS read, so
; there a missing s_barrier cannot show; wave32 maps s_barrier straight to the workgroup barrier.
; stage: cs
; wave: 32
; group: 128,1,1
; rsrc2: 0x2818a
;
; RSRC2 0x28184 before D3D-448 (now 0x2818a, see the moves below): two user SGPRs (s[0:1], the table), TGID_X and
; TGID_Y (s2, s3), LDS_SIZE 5 x 512 bytes.
; v0 = lane in the group, the sample it sums; v21 = (v0 + 64) mod 128, the sample it loads; v1 = that sample's column;
; v22 = the byte address of v21's LDS entry, v3 = v0's (entry n holds sample n - 4 of the segment).
; Slot 0 (s[0:1] + 0): the scene's T# (R16G16B16A16_FLOAT). Slot 1 (+ 0x20): the V# of the row sums, stride 16.
; s2 = 128-sample segment of the row, s3 = row. Each sample: q = floor(max(c - 0.625, 0) * 64) per colour channel, 0 in w.
; D3D-448: a D3D12 compute PSO gets its root arguments in the console layout (rt::rootsig::pack_stage,
; docs/d3d12_vulkan.md section 3.5): from s4 in root-signature order, a descriptor table as ONE dword (the 32-bit
; VA of its first descriptor; the heaps live below 4 GiB), a constant one dword per value; s0..s3 are the driver's.
; RSRC2 0x2818a therefore counts 5 user SGPRs (4 + 1), and the thread-group ids follow at s5. These moves rebuild the
; layout the code below was written for: table s4 -> s[0:1] (high half 0); thread-group ids s5..s6 -> s2..s3.
s_mov_b32 s0, s4
s_mov_b32 s1, 0
s_mov_b32 s2, s5
s_mov_b32 s3, s6
s_load_dwordx8 s[4:11], s[0:1], 0x0
s_load_dwordx4 s[12:15], s[0:1], 0x20
s_lshl_b32 s16, s2, 7
v_add_nc_u32_e32 v21, 64, v0
v_and_b32_e32 v21, 0x7f, v21
v_add_nc_u32_e32 v1, s16, v21
v_mov_b32_e32 v2, s3
v_lshlrev_b32_e32 v3, 4, v0
v_lshlrev_b32_e32 v22, 4, v21
v_mul_u32_u24_e32 v20, 0x180, v2
v_add_nc_u32_e32 v20, s16, v20
v_add_nc_u32_e32 v20, v20, v0
s_waitcnt lgkmcnt(0)
; left halo: the lanes loading samples 0-3 (lanes 64-67) fill entries 0-3 with the samples 4 before them (clamped)
v_cmp_gt_u32_e32 vcc_lo, 4, v21
s_and_saveexec_b32 s18, vcc_lo
v_subrev_nc_u32_e32 v8, 4, v1
v_max_i32_e32 v8, 0, v8
v_mov_b32_e32 v9, s3
image_load v[4:7], v[8:9], s[4:11] dmask:0xf dim:SQ_RSRC_IMG_2D
s_waitcnt vmcnt(0)
v_add_f32_e32 v4, 0xbf200000, v4
v_max_f32_e32 v4, 0, v4
v_mul_f32_e32 v4, 0x42800000, v4
v_floor_f32_e32 v4, v4
v_add_f32_e32 v5, 0xbf200000, v5
v_max_f32_e32 v5, 0, v5
v_mul_f32_e32 v5, 0x42800000, v5
v_floor_f32_e32 v5, v5
v_add_f32_e32 v6, 0xbf200000, v6
v_max_f32_e32 v6, 0, v6
v_mul_f32_e32 v6, 0x42800000, v6
v_floor_f32_e32 v6, v6
v_mov_b32_e32 v7, 0
ds_write_b128 v22, v[4:7]
s_mov_b32 exec_lo, s18
; right halo: the lanes loading samples 124-127 (lanes 60-63) fill entries 132-135 with the samples 4 after them (clamped)
v_cmp_lt_u32_e32 vcc_lo, 0x7b, v21
s_and_saveexec_b32 s18, vcc_lo
v_add_nc_u32_e32 v8, 4, v1
v_min_i32_e32 v8, 0x17f, v8
v_mov_b32_e32 v9, s3
image_load v[4:7], v[8:9], s[4:11] dmask:0xf dim:SQ_RSRC_IMG_2D
s_waitcnt vmcnt(0)
v_add_f32_e32 v4, 0xbf200000, v4
v_max_f32_e32 v4, 0, v4
v_mul_f32_e32 v4, 0x42800000, v4
v_floor_f32_e32 v4, v4
v_add_f32_e32 v5, 0xbf200000, v5
v_max_f32_e32 v5, 0, v5
v_mul_f32_e32 v5, 0x42800000, v5
v_floor_f32_e32 v5, v5
v_add_f32_e32 v6, 0xbf200000, v6
v_max_f32_e32 v6, 0, v6
v_mul_f32_e32 v6, 0x42800000, v6
v_floor_f32_e32 v6, v6
v_mov_b32_e32 v7, 0
ds_write_b128 v22, v[4:7] offset:128
s_mov_b32 exec_lo, s18
; the segment's own sample last, so nothing but the s_barrier stands between this write and the reads below
; the second wave (lanes 64-127) first walks four dependent fetches 64 rows apart, which miss the cache, while the first
; wave walks its own row; each fetch adds 0 (0 times a finite texel), so the samples are unchanged, but the second
; wave's write lands late. Only the s_barrier makes the first wave wait for it.
v_lshrrev_b32_e32 v23, 6, v0
v_lshlrev_b32_e32 v23, 6, v23
v_mov_b32_e32 v24, v1
v_add_nc_u32_e32 v25, s3, v23
v_and_b32_e32 v25, 0xff, v25
image_load v26, v[24:25], s[4:11] dmask:0x1 dim:SQ_RSRC_IMG_2D
s_waitcnt vmcnt(0)
v_mul_f32_e32 v26, 0, v26
v_cvt_u32_f32_e32 v26, v26
v_add_nc_u32_e32 v25, v25, v26
v_add_nc_u32_e32 v25, v23, v25
v_and_b32_e32 v25, 0xff, v25
image_load v26, v[24:25], s[4:11] dmask:0x1 dim:SQ_RSRC_IMG_2D
s_waitcnt vmcnt(0)
v_mul_f32_e32 v26, 0, v26
v_cvt_u32_f32_e32 v26, v26
v_add_nc_u32_e32 v25, v25, v26
v_add_nc_u32_e32 v25, v23, v25
v_and_b32_e32 v25, 0xff, v25
image_load v26, v[24:25], s[4:11] dmask:0x1 dim:SQ_RSRC_IMG_2D
s_waitcnt vmcnt(0)
v_mul_f32_e32 v26, 0, v26
v_cvt_u32_f32_e32 v26, v26
v_add_nc_u32_e32 v25, v25, v26
v_add_nc_u32_e32 v25, v23, v25
v_and_b32_e32 v25, 0xff, v25
image_load v26, v[24:25], s[4:11] dmask:0x1 dim:SQ_RSRC_IMG_2D
s_waitcnt vmcnt(0)
v_mul_f32_e32 v26, 0, v26
v_cvt_u32_f32_e32 v26, v26
v_add_nc_u32_e32 v25, v25, v26
v_add_nc_u32_e32 v25, v23, v25
v_and_b32_e32 v25, 0xff, v25
v_add_nc_u32_e32 v2, v2, v26
image_load v[4:7], v[1:2], s[4:11] dmask:0xf dim:SQ_RSRC_IMG_2D
s_waitcnt vmcnt(0)
v_add_f32_e32 v4, 0xbf200000, v4
v_max_f32_e32 v4, 0, v4
v_mul_f32_e32 v4, 0x42800000, v4
v_floor_f32_e32 v4, v4
v_add_f32_e32 v5, 0xbf200000, v5
v_max_f32_e32 v5, 0, v5
v_mul_f32_e32 v5, 0x42800000, v5
v_floor_f32_e32 v5, v5
v_add_f32_e32 v6, 0xbf200000, v6
v_max_f32_e32 v6, 0, v6
v_mul_f32_e32 v6, 0x42800000, v6
v_floor_f32_e32 v6, v6
v_mov_b32_e32 v7, 0
ds_write_b128 v22, v[4:7] offset:64
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
buffer_store_dwordx4 v[10:13], v20, s[12:15], 0 idxen
s_endpgm
