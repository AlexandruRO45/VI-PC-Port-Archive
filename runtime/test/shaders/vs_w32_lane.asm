; Wave32 vertex shader on a 64-wide vertex subgroup (w32_vs_halves.py): the wave's shape. Vertex v (v0 = vertex
; index) stores (lane, count, key, v) to its record at heap 0x1000 + 16 * v through the V# in s[0:3]:
;   lane  = v_mbcnt_lo_u32_b32 of -1: this vertex's lane in its wave, 0..31 in a wave32
;   count = s_bcnt1_i32_b32 of exec_lo: the wave's active lanes, at most 32
;   key   = v_readfirstlane_b32 of v0: the vertex in the wave's first active lane, which names the wave
; Which vertices share a wave is the driver's choice, so the checks are what every wave32 must satisfy: the vertices
; with one key are one wave, there are `count` of them, at most 32, each on its own lane, and the first of them (the
; lowest lane) is the key's own vertex.
; Run: VKH_VERTICES=960 VKH_SGPRS=0,0,65536,0x30000000 VKH_HEAP=0=0 VKH_HEAP_DUMP=heap.raw vk_harness
; vs_w32_lane.spv <any fragment shader> out.raw (the triangles have zero area, so nothing is drawn).
; mattr: +wavefrontsize32,-wavefrontsize64
v_mbcnt_lo_u32_b32 v6, -1, 0
s_bcnt1_i32_b32 s4, exec_lo
v_mov_b32_e32 v7, s4
v_readfirstlane_b32 s5, v0
v_mov_b32_e32 v8, s5
v_mov_b32_e32 v9, v0
v_lshlrev_b32_e32 v12, 4, v0
v_add_nc_u32_e32 v12, 0x1000, v12
buffer_store_dwordx4 v[6:9], v12, s[0:3], 0 offen
v_mov_b32_e32 v5, 0
exp pos0 v5, v5, v5, v5 done
exp param0 v5, v5, v5, v5
s_endpgm
