; A float inline constant in a 64-bit integer operand is its f64 encoding (measured natively on RDNA2, gfx1036;
; docs/design.md "64-bit operand literals"). Run once on a GPU.
; The RDNA2 ISA reference lists the float inline constants (0.5, 1.0, 2.0, 4.0, their negatives, 1/(2pi)) without
; saying what a 64-bit integer operand reads; the hardware reads the same 64 bits an f64 operand does:
;   -2.0 = 0xC0000000_00000000, 1.0 = 0x3FF00000_00000000, -4.0 = 0xC0100000_00000000, -0.5 = 0xBFE00000_00000000,
;   4.0 = 0x40100000_00000000, 0.5 = 0x3FE00000_00000000, -1.0 = 0xBFF00000_00000000,
;   1/(2pi) = 0x3FC45F30_6DC9C882 (the f64 constant, not the f32 one widened).
; Lane 0 of a wave32 group stores eight dwords:
;   out[0] = high word of s_mov_b64(-2.0)                                     -> 0xC0000000
;   out[1] = low word of s_mov_b64(1/(2pi))                                   -> 0x6DC9C882
;   out[2] = high word of s_lshl_b64(1.0, 4): 0x3FF00000_00000000 << 4
;            = 0xFF000000_00000000 (the top nibble shifted out)               -> 0xFF000000
;   out[3] = s_bcnt1_i32_b64(-4.0): bits of 0xC0100000 (2 + 1), low word 0    -> 3
;   out[4] = high word of s_ashr_i64(-0.5, 4): 0xBFE00000_00000000 >> 4,
;            sign bit copied in                                               -> 0xFBFE0000
;   out[5] = SCC of s_cmp_eq_u64(s[18:19] = 0x40100000_00000000, 4.0)         -> 1
;   out[6] = high word of v_lshlrev_b64(0, 0.5)                               -> 0x3FE00000
;   out[7] = high word of v_mad_u64_u32(0, 0, -1.0): 0 * 0 + S2               -> 0xBFF00000
; group: 32,1,1
; Run: vk_compute_test cs_w32_f64inl.spv 1 dump -> out[0..7] = 3221225472 1841940610 4278190080 3 4227727360 1 1071644672 3220176896
; mattr: +wavefrontsize32,-wavefrontsize64
s_mov_b64 s[8:9], -2.0
s_mov_b64 s[10:11], 0.15915494309189532
s_lshl_b64 s[12:13], 1.0, 4
s_bcnt1_i32_b64 s14, -4.0
s_ashr_i64 s[16:17], -0.5, 4
s_mov_b32 s18, 0
s_mov_b32 s19, 0x40100000
s_cmp_eq_u64 s[18:19], 4.0
s_cselect_b32 s15, 1, 0
v_lshlrev_b64 v[2:3], 0, 0.5
v_mov_b32_e32 v4, 0
v_mad_u64_u32 v[6:7], s22, v4, v4, -1.0
v_cmpx_eq_u32_e32 0, v0
v_mov_b32_e32 v20, s9
v_mov_b32_e32 v21, 0
buffer_store_dword v20, v21, s[0:3], 0 offen
v_mov_b32_e32 v20, s10
v_mov_b32_e32 v21, 4
buffer_store_dword v20, v21, s[0:3], 0 offen
v_mov_b32_e32 v20, s13
v_mov_b32_e32 v21, 8
buffer_store_dword v20, v21, s[0:3], 0 offen
v_mov_b32_e32 v20, s14
v_mov_b32_e32 v21, 12
buffer_store_dword v20, v21, s[0:3], 0 offen
v_mov_b32_e32 v20, s17
v_mov_b32_e32 v21, 16
buffer_store_dword v20, v21, s[0:3], 0 offen
v_mov_b32_e32 v20, s15
v_mov_b32_e32 v21, 20
buffer_store_dword v20, v21, s[0:3], 0 offen
v_mov_b32_e32 v21, 24
buffer_store_dword v3, v21, s[0:3], 0 offen
v_mov_b32_e32 v21, 28
buffer_store_dword v7, v21, s[0:3], 0 offen
s_endpgm
