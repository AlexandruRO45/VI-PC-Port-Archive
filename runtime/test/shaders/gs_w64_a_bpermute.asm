; SHADER-10 tier A, geometry stage: ds_bpermute_b32 in a wave64 legacy (LDS-ring) geometry shader (the geometry
; counterpart of ps_w64_a_bpermute, and of the vertex fixture vs_w64_a_bpermute in the separate pull request #7311,
; which this file does not depend on). Drawn with vs_ids (param1 = 0x101 + vertex index): triangle
; p = (TEXCOORD0.x of input vertex 0 - 0x101) / 3, as in gs_w64_a_ring. Triangle p sends values to the lane that reads
; them from lane ^ 1. Which triangle that is depends on how the driver packed the draw into waves: the triangle of
; lane L is the wave's first triangle + L, and the waves differ by GPU and driver. Measured with gs_w64_c_lane_id over
; the 320 triangles of the draw (reported on the pull request where not RADV gfx1036):
;  RADV gfx1036 (Mesa 26.3.1): waves of 42 invocations at wave64 (first triangles 0, 42, 84, ...), of 32 and 10 at wave32
;    (0, 32, 42, 74, ...). Every wave starts at an even triangle.
;  RADV gfx1010 (RX 5700 XT, Mesa 26.3.1): waves of 41 (0, 41, 82, 123, ...), of 32 and 9 at wave32. Half the waves
;    start at an odd triangle, and the last lane of a 41-lane wave has no partner.
;  AMD Windows driver 26.11.2 (RX 9060 XT): waves of 64, 21, 64, 21, 64, 21, 1, 64 (0, 64, 85, 149, 170, 234, 255, 256).
;  NVIDIA Windows driver 621.24 (RTX 4060 Ti, default size): 32 spans of 10 triangles, every one from an even triangle.
; In every layout seen so far each wave32 wave is one half of a wave64 wave.
; So the words are not a function of p alone. w64_graphics.py derives them from the lanes the probe measured at the same
; size (layout_words): the partner q of p is the triangle that holds lane ^ 1 in the same wave, and a triangle without
; one (the last lane of a wave of odd size) reads 0, because a source lane that is not running gives 0.
; The address comes from the lane id (v_mbcnt_lo then v_mbcnt_hi of -1) and uses address bits 2..6 only. With q the
; partner of p (all four words 0 without one):
;  r0 = ds_bpermute of p from lane ^ 1: q.
;  r1 = ds_bpermute of 3 * p + 7 from lane ^ 1: 3 * q + 7.
;  r2 = ds_bpermute from lane ^ 1 of r0, back to the own triangle: p.
;  r3 = ds_bpermute of p >> 1 from lane ^ 1: q >> 1.
; Each invocation stores them to its record at heap 0x1000 + 16 * p through the V# in s[4:7] (the translator sets s3
; for the prologue), then emits one vertex through the lane-based ring address, which makes no primitive.
; Needs a DXBC with ISG1/OSG1 = {SV_Position reg 0, TEXCOORD0 reg 1} (build.py makes it for gs_*).
; Run: VKH_GS=gs_w64_a_bpermute.spv VKH_VERTICES=960 VKH_SGPRS=0,0,0,0,0,0,65536,0x30000000 VKH_HEAP=0=0
; VKH_HEAP_DUMP=heap.raw vk_harness vs_ids.spv <any fragment shader> out.raw, with and without RADV_PERFTEST=gewave32:
; equal records where the wave32 draw is the wave64 draw cut into halves, and each draw equal to the words its own
; layout gives.
; mattr: +wavefrontsize64,-wavefrontsize32
s_bfe_i32 s0, s3, 0x80008
s_bfm_b64 exec, s0, 0
s_cmp_eq_u32 s0, 64
s_cselect_b64 exec, -1, exec
v_lshlrev_b32_e32 v3, 2, v0
v_and_b32_e32 v10, 0x3fffc, v3
ds_read_b32 v20, v10 offset:16
s_waitcnt lgkmcnt(0)
v_subrev_nc_u32_e32 v20, 0x101, v20
v_mul_hi_u32 v21, v20, 0xaaaaaaab
v_lshrrev_b32_e32 v21, 1, v21
v_mbcnt_lo_u32_b32 v15, -1, 0
v_mbcnt_hi_u32_b32 v15, -1, v15
v_xor_b32_e32 v16, 1, v15
v_lshlrev_b32_e32 v16, 2, v16
v_mad_u32_u24 v17, v21, 3, 7
v_lshrrev_b32_e32 v18, 1, v21
ds_bpermute_b32 v6, v16, v21
ds_bpermute_b32 v7, v16, v17
ds_bpermute_b32 v9, v16, v18
s_waitcnt lgkmcnt(0)
ds_bpermute_b32 v8, v16, v6
s_waitcnt lgkmcnt(0)
v_lshlrev_b32_e32 v12, 4, v21
v_add_nc_u32_e32 v12, 0x1000, v12
buffer_store_dwordx4 v[6:9], v12, s[4:7], 0 offen
v_mbcnt_lo_u32_b32 v13, -1, 0
v_mbcnt_hi_u32_b32 v13, -1, v13
v_mul_u32_u24_e32 v13, 0x60, v13
v_mov_b32_e32 v14, 0
ds_write2_b32 v13, v14, v14 offset0:0 offset1:1
s_sendmsg sendmsg(MSG_GS, GS_OP_EMIT, 0)
s_sendmsg sendmsg(MSG_GS, GS_OP_CUT, 0)
s_sendmsg sendmsg(MSG_GS_DONE, GS_OP_NOP)
s_endpgm
