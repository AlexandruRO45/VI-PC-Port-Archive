; SHADER-529 / #6214: v_mbcnt_lo_u32_b32 reading an SGPR / VCC lane mask after EXEC is narrowed (wave32).
; Under SPIRV-Cross (MoltenVK 1.4.2), a single-use OpGroupNonUniformBitwiseOr from wave_mask was inlined into
; dst's `active ? (...) : old`, so lanes off in EXEC skipped simd_or(...) and dropped their set bits from the mask.
; Full EXEC = 0xffffffff builds:
;   s4     = lanes 0..11   = 0x00000fff (12 bits)
;   vcc_lo = even lanes    = 0x55555555 (16 bits)
;   s6     = saved EXEC    = 0xffffffff (32 bits, via s_and_saveexec_b32 with s5 = 0xffffaa00)
; Destinations v10..v12 start at 1000 + lane. Under EXEC = 0xffffaa00 (lanes 0..7 off, 8..15 odd lanes 9/11/13/15 on,
; 16..31 on), lanes 0..8 of s4, all 8 even lanes 0..14 of vcc_lo, and 12 of the 20 lanes < 20 in s6 are off in EXEC:
;   out[0] v10 lane 0  (off in EXEC)                            -> 1000
;   out[1] v10 lane 9  (mbcnt_lo s4, 100: bits 0..8 = 9)        -> 109  (100 if off lanes drop)
;   out[2] v10 lane 20 (mbcnt_lo s4, 100: bits 0..11 = 12)      -> 112  (102 if off lanes drop)
;   out[3] v11 lane 8  (off in EXEC)                            -> 1008
;   out[4] v11 lane 15 (mbcnt_lo vcc_lo, 200: 8 even in 0..14)  -> 208  (200 if off lanes drop)
;   out[5] v11 lane 25 (mbcnt_lo vcc_lo, 200: 13 even in 0..24) -> 213  (205 if off lanes drop)
;   out[6] v12 lane 2  (off in EXEC)                            -> 1002
;   out[7] v12 lane 20 (mbcnt_lo s6, 300: 20 saved bits 0..19)  -> 320  (308 if off lanes drop)
; group: 32,1,1
; Run: VKC_SUBGROUP_SIZE=32 vk_compute_test cs_w32_mbcnt_exec.spv 1 dump -> out[0..7] = 1000 109 112 1008 208 213 1002 320
; mattr: +wavefrontsize32,-wavefrontsize64
v_add_nc_u32_e32 v10, 1000, v0
v_add_nc_u32_e32 v11, 1000, v0
v_add_nc_u32_e32 v12, 1000, v0
v_cmp_gt_u32_e64 s4, 12, v0
v_and_b32_e32 v1, 1, v0
v_cmp_eq_u32_e32 vcc_lo, 0, v1
s_mov_b32 s5, 0xffffaa00
s_and_saveexec_b32 s6, s5
v_mbcnt_lo_u32_b32 v10, s4, 100
v_mbcnt_lo_u32_b32 v11, vcc_lo, 200
v_mbcnt_lo_u32_b32 v12, s6, 300
s_mov_b32 exec_lo, -1
v_readlane_b32 s8, v10, 0
v_readlane_b32 s9, v10, 9
v_readlane_b32 s10, v10, 20
v_readlane_b32 s11, v11, 8
v_readlane_b32 s12, v11, 15
v_readlane_b32 s13, v11, 25
v_readlane_b32 s14, v12, 2
v_readlane_b32 s15, v12, 20
v_cmpx_eq_u32_e32 0, v0
v_mov_b32_e32 v20, s8
v_mov_b32_e32 v21, 0
buffer_store_dword v20, v21, s[0:3], 0 offen
v_mov_b32_e32 v20, s9
v_mov_b32_e32 v21, 4
buffer_store_dword v20, v21, s[0:3], 0 offen
v_mov_b32_e32 v20, s10
v_mov_b32_e32 v21, 8
buffer_store_dword v20, v21, s[0:3], 0 offen
v_mov_b32_e32 v20, s11
v_mov_b32_e32 v21, 12
buffer_store_dword v20, v21, s[0:3], 0 offen
v_mov_b32_e32 v20, s12
v_mov_b32_e32 v21, 16
buffer_store_dword v20, v21, s[0:3], 0 offen
v_mov_b32_e32 v20, s13
v_mov_b32_e32 v21, 20
buffer_store_dword v20, v21, s[0:3], 0 offen
v_mov_b32_e32 v20, s14
v_mov_b32_e32 v21, 24
buffer_store_dword v20, v21, s[0:3], 0 offen
v_mov_b32_e32 v20, s15
v_mov_b32_e32 v21, 28
buffer_store_dword v20, v21, s[0:3], 0 offen
s_endpgm
