; group: 32,1,1
; v_mul_legacy_f32 signed zeros in the game's MODE (IEEE 0, f32 denormals flushed). The DX9 rule (RDNA2 ISA,
; V_MUL_LEGACY_F32: if either operand is 0 the product is +0, whatever the other operand or the signs) gives +0, never
; -0, and VOP3 omod makes a -0 that enters it +0 (docs/design.md, "Output modifiers on hardware"). A product that omod
; scales into the subnormal range is flushed after the scaling and keeps its sign.
; v1 = -0.0, v2 = 3.0, v4 = 0xbfb2c7b3 (-1.3963...), v5 = FLT_MIN (0x00800000), v6 = -FLT_MIN, v7 = 0.5. Every lane stores the four results at lane * 16:
;   x: v_mul_legacy_f32 v8, v1, v2             -0 * 3.0: either operand 0 -> +0                         -> 0x00000000
;   y: v_mul_legacy_f32_e64 v9, v2, v1         3.0 * -0: the zero may be the second operand too         -> 0x00000000
;   z: v_mul_legacy_f32_e64 v10, v4, v5 div:2
;        v4 * FLT_MIN = -1.396 * 2^-126 is a normal negative product; div:2 halves it to -0.698 * 2^-126, below
;        FLT_MIN, which the flush turns into -0 (the sign stays)                                       -> 0x80000000
;   w: v_mul_legacy_f32_e64 v11, v6, v7 mul:2
;        -FLT_MIN * 0.5 = -2^-127 is subnormal and flushed to -0 before omod; mul:2 then gives +0     -> 0x00000000
; Measured natively on RDNA2 (gfx1036, Ryzen 9 9950X3D iGPU, driver 26.10.1) through tools/hw_oracle (raw HIP kernel,
; wave32, game MODE): the same four instructions give 0x00000000 0x00000000 0x80000000 0x00000000.
; The explicit s_mov_b32 exec_lo, -1 matters: without it the AMD Windows driver (26.10.1) did not show the -0 for x and y.
; Run: vk_compute_test cs_legacy_mul_zero.spv 1 dump
; out[0..7] = 00000000 00000000 80000000 00000000 00000000 00000000 80000000 00000000 (hex)
s_mov_b32 exec_lo, -1
v_mov_b32_e32 v20, v0
v_mov_b32_e32 v1, 0x80000000
v_mov_b32_e32 v2, 0x40400000
v_mov_b32_e32 v4, 0xbfb2c7b3
v_mov_b32_e32 v5, 0x00800000
v_mov_b32_e32 v6, 0x80800000
v_mov_b32_e32 v7, 0x3f000000
v_mul_legacy_f32_e32 v8, v1, v2
v_mul_legacy_f32_e64 v9, v2, v1
v_mul_legacy_f32_e64 v10, v4, v5 div:2
v_mul_legacy_f32_e64 v11, v6, v7 mul:2
v_lshlrev_b32_e32 v13, 4, v20
buffer_store_dword v8, v13, s[0:3], 0 offen
buffer_store_dword v9, v13, s[0:3], 0 offen offset:4
buffer_store_dword v10, v13, s[0:3], 0 offen offset:8
buffer_store_dword v11, v13, s[0:3], 0 offen offset:12
s_endpgm
