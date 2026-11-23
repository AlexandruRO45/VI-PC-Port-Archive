; group: 128,1,1
; rsrc2: 0x200088
; SHADER-579: with a large LDS the barrier-round count is borrowed from LDS word 0 instead of %x_buf (the
; "Barrier-round count in LDS" row in docs/design.md). RSRC2 LDS_SIZE 64 gives a 32 KiB array, so the array plus
; the 12 bytes of %x_buf would be over 16,384 B. Wave32, 128 threads = 4 waves of 32; wave w runs w + 1
; iterations. Iteration k adds 1 to LDS[k + 1] from each of the wave's 32 lanes, waits at s_barrier, then reads:
; wave 3 reads LDS[k + 1], wave 2 reads LDS[0], the word the count borrows. LDS[0] holds the sentinel lane 0
; wrote before the first barrier; every round end saves that word and puts it back, so wave 2 reads the sentinel
; in each of its three iterations although the count used LDS[0] between them.
; Run: vk_compute_test cs_barrier_lds.spv 1 dump
; Expect: out[0..7] = 128 96 64 32 1515849081 1515849081 1515849081 4008636142
; (LDS[1..4] after 4, 3, 2 and 1 waves of 32 lanes added 1 each; 0x5a5a0579 the sentinel; 0xeeeeeeee the heap
; fill, wave 2 has no fourth iteration. A count left in LDS[0], or a save without the restore, reads as 0 to 128
; there.)
v_lshlrev_b32_e32 v2, 2, v0
v_mov_b32_e32 v3, 0
v_cmpx_gt_u32_e32 5, v0
ds_write_b32 v2, v3
s_mov_b32 exec_lo, -1
s_waitcnt lgkmcnt(0)
v_mov_b32_e32 v7, 0x5a5a0579
v_cmpx_eq_u32_e32 0, v0
ds_write_b32 v2, v7
s_mov_b32 exec_lo, -1
s_waitcnt lgkmcnt(0)
s_barrier
v_readfirstlane_b32 s5, v0
s_lshr_b32 s5, s5, 5
s_add_u32 s6, s5, 1
s_sub_u32 s9, 3, s5
s_lshl_b32 s9, s9, 4
s_mov_b32 s7, 0
v_mov_b32_e32 v5, 1
s_add_u32 s13, s7, 4
v_mov_b32_e32 v1, s13
ds_add_u32 v1, v5
s_waitcnt lgkmcnt(0)
s_barrier
s_cmp_eq_u32 s5, 2
s_cselect_b32 s12, 0, s13
v_mov_b32_e32 v1, s12
ds_read_b32 v4, v1
s_waitcnt lgkmcnt(0)
s_add_u32 s10, s9, s7
v_mov_b32_e32 v6, s10
buffer_store_dword v4, v6, s[0:3], 0 offen
s_add_u32 s7, s7, 4
s_sub_u32 s6, s6, 1
s_cmp_lg_u32 s6, 0
s_cbranch_scc1 -20
s_endpgm
