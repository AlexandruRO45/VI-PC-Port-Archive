; Wave32 vertex shader on a 64-wide vertex subgroup (w32_vs_halves.py): a ballot read as a whole. VCC = the lanes
; whose vertex has v % 4 == 1 (v0 = vertex index). Record A at heap 0x1000 + 16 * v, record B at 0x5000 + 16 * v:
;   A = (lane, rank, key, v)    lane = v_mbcnt_lo of -1, rank = v_mbcnt_lo of VCC (selected lanes below this one,
;                               the slot a stream compaction would give), key = v_readfirstlane of v0
;   B = (count, first, sel, 0)  count = s_bcnt1 of vcc_lo, first = s_ff1 of vcc_lo (the lowest selected lane, -1 if
;                               none), sel = v % 4 == 1
; Checked per wave (the vertices with one key): count and first agree with the members' sel and lanes, and each
; member's rank is the number of selected members on lower lanes.
; Run: as vs_w32_lane.asm.
; mattr: +wavefrontsize32,-wavefrontsize64
v_mbcnt_lo_u32_b32 v6, -1, 0
v_and_b32_e32 v4, 3, v0
v_cmp_eq_u32_e32 vcc_lo, 1, v4
v_mbcnt_lo_u32_b32 v7, vcc_lo, 0
v_readfirstlane_b32 s5, v0
v_mov_b32_e32 v8, s5
v_mov_b32_e32 v9, v0
v_lshlrev_b32_e32 v12, 4, v0
v_add_nc_u32_e32 v13, 0x1000, v12
buffer_store_dwordx4 v[6:9], v13, s[0:3], 0 offen
s_bcnt1_i32_b32 s6, vcc_lo
s_ff1_i32_b32 s7, vcc_lo
v_mov_b32_e32 v6, s6
v_mov_b32_e32 v7, s7
v_cndmask_b32_e64 v8, 0, 1, vcc_lo
v_mov_b32_e32 v9, 0
v_add_nc_u32_e32 v13, 0x5000, v12
buffer_store_dwordx4 v[6:9], v13, s[0:3], 0 offen
v_mov_b32_e32 v5, 0
exp pos0 v5, v5, v5, v5 done
exp param0 v5, v5, v5, v5
s_endpgm
