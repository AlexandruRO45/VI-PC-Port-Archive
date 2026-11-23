; SHADER-10 tier A, vertex stage: ds_bpermute_b32 in a wave64 vertex shader (the vertex counterpart of ps_w64_a_bpermute).
; Vertex v (v0 = vertex index) sends values to the lane that reads them from lane ^ 1. Which vertex that is depends on
; how the driver packed the draw into waves: the vertex of lane L is the wave's first vertex + L, and the waves differ
; by GPU and driver. Measured with vs_w64_c_lane_id over the 960 vertices of the draw:
;  RADV gfx1036 (Mesa 26.3.1): waves of 64 and 62 vertices at wave64 (first vertices 0, 64, 126, 190, ...), of 32, 32,
;    32 and 30 at wave32 (0, 32, 64, 96, 126, ...). Every wave starts at an even vertex and holds an even number.
;  RADV gfx1010 (RX 5700 XT, Mesa 26.3.1, reported on the pull request): waves of 64 and 59 (0, 64, 123, 187, ...), of 32, 32, 32, 27 at wave32. Half
;    the waves start at an odd vertex, and the last lane of a 59-lane wave has no partner.
; In every layout seen so far each wave32 wave is one half of a wave64 wave.
; So the words are not a function of v alone. w64_graphics.py derives them from the lanes the probe measured at the same
; size (layout_words): the partner q of v is the vertex that holds lane ^ 1 in the same wave, and a vertex without one
; (the last lane of a wave of odd size) reads 0, because a source lane that is not running gives 0. A first draft read
; lane ^ 2 and lane ^ 3 and expected v ^ 2 and v ^ 3, which only holds where every wave starts at a multiple of four.
; The address comes from the lane id (v_mbcnt_lo then v_mbcnt_hi of -1) and uses address bits 2..6 only, so it is the
; same in the wave (0..63) and in a half (0..31). With q the partner of v (all four words 0 without one):
;  r0 = ds_bpermute of v from lane ^ 1: q.
;  r1 = ds_bpermute of 3 * v + 7 from lane ^ 1: 3 * q + 7.
;  r2 = ds_bpermute from lane ^ 1 of r0, back to the own vertex: v.
;  r3 = ds_bpermute of v >> 1 from lane ^ 1: q >> 1.
; It stores (r0, r1, r2, r3) to its record at heap 0x1000 + 16 * v through the V# in s[0:3], like vs_w64_a_skip.
; Run: VKH_VERTICES=960 VKH_SGPRS=0,0,65536,0x30000000 VKH_HEAP=0=0 VKH_HEAP_DUMP=heap.raw vk_harness
; vs_w64_a_bpermute.spv <any fragment shader> out.raw, with and without RADV_PERFTEST=gewave32: equal records where the
; wave32 draw is the wave64 draw cut into halves, and each draw equal to the words its own layout gives.
; mattr: +wavefrontsize64,-wavefrontsize32
s_mov_b64 s[10:11], exec
v_mbcnt_lo_u32_b32 v10, -1, 0
v_mbcnt_hi_u32_b32 v10, -1, v10
v_xor_b32_e32 v11, 1, v10
v_lshlrev_b32_e32 v11, 2, v11
v_mad_u32_u24 v13, v0, 3, 7
v_lshrrev_b32_e32 v14, 1, v0
ds_bpermute_b32 v6, v11, v0
ds_bpermute_b32 v7, v11, v13
ds_bpermute_b32 v9, v11, v14
s_waitcnt lgkmcnt(0)
ds_bpermute_b32 v8, v11, v6
s_waitcnt lgkmcnt(0)
v_lshlrev_b32_e32 v12, 4, v0
v_add_nc_u32_e32 v12, 0x1000, v12
buffer_store_dwordx4 v[6:9], v12, s[0:3], 0 offen
v_mov_b32_e32 v5, 0
exp pos0 v5, v5, v5, v5 done
exp param0 v5, v5, v5, v5
s_endpgm
