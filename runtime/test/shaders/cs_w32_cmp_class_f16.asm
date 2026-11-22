; v_cmp_class_f16 / v_cmpx_class_f16 (e32 and e64), run once on a GPU. The f32 op's ten class bits on the binary16
; word of the source's low half: 0 sNaN, 1 qNaN, 2 -inf, 3 -normal, 4 -denormal, 5 -0, 6 +0, 7 +denormal, 8 +normal,
; 9 +inf (RDNA2 ISA reference, V_CMP_CLASS_F16). The high half of the source is ignored, an inline float constant is
; the f16 value, neg/abs act on bit 15, and f16 denormals are classed as denormals (the f16 denormals the translator
; keeps). Measured natively on RDNA2 (gfx1036, wave32 and wave64; docs/design.md "Floating-point class compare").
; Each of out[0..6] tests one value twice: mask = its class bit C (expected true -> C) and mask = the other nine
; bits (expected false; true would add 0x400), so the word is C exactly when the value is in class C and no other:
;   out[0] = class of sNaN 0x7C01 (high half 0xABCD ignored) bit 0  -> 0x001
;   out[1] = class of -qNaN 0xFE00                           bit 1  -> 0x002
;   out[2] = class of -inf 0xFC00                            bit 2  -> 0x004
;   out[3] = class of -denormal 0x83FF                       bit 4  -> 0x010
;   out[4] = class of -0 0x8000                              bit 5  -> 0x020
;   out[5] = class of +denormal 0x0001                       bit 7  -> 0x080
;   out[6] = class of -|0x3C00| = -1.0, a -normal             bit 3  -> 0x008
;   out[7] = e32 with VCC: the constant 1.0 (f16 0x3C00, +normal) against mask 0x100 -> 0x100, against 0x2FF -> 0;
;            then v_cmpx_class_f16 of +inf against 0x100 (false: v17 keeps 0x100, EXEC restored from s30) and,
;            after the lane-0 select, against 0x200 (true: lane 0 stays on and stores)                 -> 0x100
; Until this test the translator rejected both opcodes ("Unsupported: v_cmp_class_f16").
; group: 32,1,1
; Run: vk_compute_test cs_w32_cmp_class_f16.spv 1 dump -> out[0..7] = 1 2 4 16 32 128 8 256
; mattr: +wavefrontsize32,-wavefrontsize64
v_mov_b32_e32 v1, 0xabcd7c01
v_cmp_class_f16_e64 s20, v1, 1
v_cndmask_b32_e64 v10, 0, 1, s20
v_cmp_class_f16_e64 s20, v1, 0x3fe
v_cndmask_b32_e64 v20, 0, 0x400, s20
v_or_b32_e32 v10, v10, v20
v_mov_b32_e32 v1, 0xfe00
v_cmp_class_f16_e64 s20, v1, 2
v_cndmask_b32_e64 v11, 0, 2, s20
v_cmp_class_f16_e64 s20, v1, 0x3fd
v_cndmask_b32_e64 v20, 0, 0x400, s20
v_or_b32_e32 v11, v11, v20
v_mov_b32_e32 v1, 0xfc00
v_cmp_class_f16_e64 s20, v1, 4
v_cndmask_b32_e64 v12, 0, 4, s20
v_cmp_class_f16_e64 s20, v1, 0x3fb
v_cndmask_b32_e64 v20, 0, 0x400, s20
v_or_b32_e32 v12, v12, v20
v_mov_b32_e32 v1, 0x83ff
v_cmp_class_f16_e64 s20, v1, 16
v_cndmask_b32_e64 v13, 0, 16, s20
v_cmp_class_f16_e64 s20, v1, 0x3ef
v_cndmask_b32_e64 v20, 0, 0x400, s20
v_or_b32_e32 v13, v13, v20
v_mov_b32_e32 v1, 0x8000
v_cmp_class_f16_e64 s20, v1, 32
v_cndmask_b32_e64 v14, 0, 32, s20
v_cmp_class_f16_e64 s20, v1, 0x3df
v_cndmask_b32_e64 v20, 0, 0x400, s20
v_or_b32_e32 v14, v14, v20
v_mov_b32_e32 v1, 0x1
v_cmp_class_f16_e64 s20, v1, 0x80
v_cndmask_b32_e64 v15, 0, 0x80, s20
v_cmp_class_f16_e64 s20, v1, 0x37f
v_cndmask_b32_e64 v20, 0, 0x400, s20
v_or_b32_e32 v15, v15, v20
v_mov_b32_e32 v1, 0x3c00
v_cmp_class_f16_e64 s20, -|v1|, 8
v_cndmask_b32_e64 v16, 0, 8, s20
v_cmp_class_f16_e64 s20, -|v1|, 0x3f7
v_cndmask_b32_e64 v20, 0, 0x400, s20
v_or_b32_e32 v16, v16, v20
v_mov_b32_e32 v2, 0x100
v_cmp_class_f16_e32 vcc_lo, 1.0, v2
v_cndmask_b32_e64 v17, 0, 0x100, vcc_lo
v_mov_b32_e32 v2, 0x2ff
v_cmp_class_f16_e32 vcc_lo, 1.0, v2
v_cndmask_b32_e64 v20, 0, 0x400, vcc_lo
v_or_b32_e32 v17, v17, v20
v_mov_b32_e32 v1, 0x7c00
s_mov_b32 s30, exec_lo
v_cmpx_class_f16_e64 v1, 0x100
v_mov_b32_e32 v17, 0xdead
s_mov_b32 exec_lo, s30
v_cmpx_eq_u32_e32 0, v0
v_cmpx_class_f16_e64 v1, 0x200
v_mov_b32_e32 v21, 0
buffer_store_dword v10, v21, s[0:3], 0 offen
v_mov_b32_e32 v21, 4
buffer_store_dword v11, v21, s[0:3], 0 offen
v_mov_b32_e32 v21, 8
buffer_store_dword v12, v21, s[0:3], 0 offen
v_mov_b32_e32 v21, 12
buffer_store_dword v13, v21, s[0:3], 0 offen
v_mov_b32_e32 v21, 16
buffer_store_dword v14, v21, s[0:3], 0 offen
v_mov_b32_e32 v21, 20
buffer_store_dword v15, v21, s[0:3], 0 offen
v_mov_b32_e32 v21, 24
buffer_store_dword v16, v21, s[0:3], 0 offen
v_mov_b32_e32 v21, 28
buffer_store_dword v17, v21, s[0:3], 0 offen
s_endpgm
