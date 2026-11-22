; Wave32 vertex shader on a 64-wide vertex subgroup (w32_vs_halves.py): a wave-wide branch and a write to EXEC.
;  (1) s_cbranch_vccz over a scalar write: only the wave that holds vertex 37 sets s5 = 7, every other wave keeps 0
;      (a wave-level culling decision has this shape).
;  (2) exec_lo = 0xffff: lanes 0..15 of each wave write 1, lanes 16..31 keep 0, then EXEC is restored.
; Vertex v (v0 = vertex index) stores (lane, s5, key, v) to heap 0x1000 + 16 * v and (low16, 0, 0, 0) to 0x5000 + 16 * v,
; key = v_readfirstlane of v0. Checked per wave: s5 == 7 exactly in the wave that has vertex 37, low16 == lane < 16.
; Run: as vs_w32_lane.asm.
; mattr: +wavefrontsize32,-wavefrontsize64
v_mbcnt_lo_u32_b32 v6, -1, 0
s_mov_b32 s5, 0
v_cmp_eq_u32_e32 vcc_lo, 37, v0
s_cbranch_vccz 1
s_mov_b32 s5, 7
v_mov_b32_e32 v7, s5
v_readfirstlane_b32 s6, v0
v_mov_b32_e32 v8, s6
v_mov_b32_e32 v9, v0
v_lshlrev_b32_e32 v12, 4, v0
v_add_nc_u32_e32 v13, 0x1000, v12
buffer_store_dwordx4 v[6:9], v13, s[0:3], 0 offen
s_mov_b32 s10, exec_lo
v_mov_b32_e32 v6, 0
s_mov_b32 exec_lo, 0xffff
v_mov_b32_e32 v6, 1
s_mov_b32 exec_lo, s10
v_mov_b32_e32 v7, 0
v_mov_b32_e32 v8, 0
v_mov_b32_e32 v9, 0
v_add_nc_u32_e32 v13, 0x5000, v12
buffer_store_dwordx4 v[6:9], v13, s[0:3], 0 offen
v_mov_b32_e32 v5, 0
exp pos0 v5, v5, v5, v5 done
exp param0 v5, v5, v5, v5
s_endpgm
