; SHADER-10 tier A, vertex stage: ds_bpermute_b32 in a wave64 vertex shader (the vertex counterpart of ps_w64_a_bpermute).
; Vertex v (v0 = vertex index) sends values to the lane that reads them. RADV does not fill a wave from lane 0 with a
; fixed count. Measured with vs_w64_c_lane_id on one GPU only (RADV Mesa 26.3.1, gfx1036; other GPUs may pack
; differently), the 960 vertices of the draw go to waves of
; 64 and 62 vertices at wave64 (first vertices 0, 64, 126, 190, 252, ...) and of 32, 32, 32 and 30 at wave32 (0, 32, 64,
; 96, 126, ...), a period of 126 vertices, and the vertex of lane L is the wave's first vertex + L. Every wave starts at
; an even vertex, so lane ^ 1 holds vertex v ^ 1 at either size, but a group of four lanes is not a group of four
; vertices (the wave that starts at vertex 126 or 190 starts 2 mod 4), and lane ^ 2 or lane ^ 3 is not vertex v ^ 2 or
; v ^ 3: the first draft of this fixture read those and failed at both sizes with the same records. The lane ^ 16 reads
; of the pixel fixtures are left out for the same reason, and the last wave of a half leaves lanes without a vertex.
; The address comes from the lane id (v_mbcnt_lo then v_mbcnt_hi of -1) and uses address bits 2..6 only, so it is the
; same in the wave (0..63) and in a half (0..31). The words are a function of v alone:
;  r0 = ds_bpermute of v from lane ^ 1: v ^ 1.
;  r1 = ds_bpermute of 3 * v + 7 from lane ^ 1: 3 * (v ^ 1) + 7.
;  r2 = ds_bpermute from lane ^ 1 of r0, back to the own vertex: v.
;  r3 = ds_bpermute of v >> 1 from lane ^ 1: v >> 1 (a pair shares it).
; It stores (r0, r1, r2, r3) to its record at heap 0x1000 + 16 * v through the V# in s[0:3], like vs_w64_a_skip.
; Run: VKH_VERTICES=960 VKH_SGPRS=0,0,65536,0x30000000 VKH_HEAP=0=0 VKH_HEAP_DUMP=heap.raw vk_harness
; vs_w64_a_bpermute.spv <any fragment shader> out.raw, with and without RADV_PERFTEST=gewave32: equal records, and equal
; to the expected words.
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
