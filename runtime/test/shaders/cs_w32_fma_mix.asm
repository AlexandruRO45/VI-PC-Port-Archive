; v_fma_mix_f32, v_fma_mixlo_f16 and v_fma_mixhi_f16 (RDNA2 ISA, VOP3P opcodes 32, 33, 34), CPU-only checks until now.
; D = S0 * S1 + S2 with one rounding. Source i is an f32 VGPR when op_sel_hi[i] = 0, else the f16 in its low
; (op_sel[i] = 0) or high (op_sel[i] = 1) half. mix_f32 writes an f32; mixlo/mixhi round the exact result once
; straight to f16 (round to nearest even, the default MODE) and write bits 15:0 / 31:16, keeping the other half.
; neg (-vN) negates a source; clamp clamps the result to [0, 1].
; Inputs: v1 = 0x40003C00 (hi 2.0, lo 1.0), v2 = 0x44004200 (hi 4.0, lo 3.0), v4 = 0x38003400 (hi 0.5, lo 0.25),
; v3 = 0.5f; every mixlo/mixhi destination starts as 0xABCD1234.
;   out[0] mix_f32  op_sel:[1,0,0] op_sel_hi:[1,1,0]: v1.hi 2.0 * v2.lo 3.0 + f32 0.5 = 6.5 (f32)         = 0x40D00000
;   out[1] mix_f32  all f32: (1 + 2^-12) * (1 + 2^-12) - 1 = 2^-11 + 2^-24, exact in f32 with one rounding
;          (rounding the product first gives 1 + 2^-11, a tie that goes to even, and 2^-11 = 0x3A000000)   = 0x3A000400
;   out[2] mixlo    op_sel:[0,1,0] op_sel_hi:[1,1,1]: 1.0 * v2.hi 4.0 + v4.lo 0.25 = 4.25 = f16 0x4440      = 0xABCD4440
;   out[3] mixhi    op_sel:[1,0,1] op_sel_hi:[1,1,1]: 2.0 * 3.0 + v4.hi 0.5 = 6.5 = f16 0x4680             = 0x46801234
;   out[4] mixlo    all f32: (1 + 2^-11) * 1.0 + 2^-40. 1 + 2^-11 is the f16 midpoint between 1.0 and 1 + 2^-10;
;          the exact sum is above it, so one rounding gives 1 + 2^-10 = f16 0x3C01 (rounding to f32 first
;          would drop the 2^-40 and the tie would go to even, 0x3C00)                                     = 0xABCD3C01
;   out[5] mixhi    all f32, neg src0: -(1 + 2^-11) * 1.0 - 2^-40 -> -(1 + 2^-10) = f16 0xBC01             = 0xBC011234
;   out[6] mixlo    op_sel_hi:[1,1,1] clamp, neg src0: -1.0 * 3.0 + 0.25 = -2.75 -> clamp 0 = f16 0x0000    = 0xABCD0000
;   out[7] mixlo    all f32: 300.0 * 300.0 + 0 = 90000 > 65504 (and >= 65520) -> f16 +inf 0x7C00          = 0xABCD7C00
; group: 32,1,1
; Run: vk_compute_test cs_w32_fma_mix.spv 1 dump
;   -> out[0..7] = 0x40D00000 0x3A000400 0xABCD4440 0x46801234 0xABCD3C01 0xBC011234 0xABCD0000 0xABCD7C00
; mattr: +wavefrontsize32,-wavefrontsize64
v_mov_b32_e32 v1, 0x40003c00
v_mov_b32_e32 v2, 0x44004200
v_mov_b32_e32 v3, 0.5
v_mov_b32_e32 v4, 0x38003400
v_fma_mix_f32 v10, v1, v2, v3 op_sel:[1,0,0] op_sel_hi:[1,1,0]
v_mov_b32_e32 v5, 0x3f800800
v_mov_b32_e32 v6, -1.0
v_fma_mix_f32 v11, v5, v5, v6
v_mov_b32_e32 v12, 0xabcd1234
v_fma_mixlo_f16 v12, v1, v2, v4 op_sel:[0,1,0] op_sel_hi:[1,1,1]
v_mov_b32_e32 v13, 0xabcd1234
v_fma_mixhi_f16 v13, v1, v2, v4 op_sel:[1,0,1] op_sel_hi:[1,1,1]
v_mov_b32_e32 v7, 0x3f801000
v_mov_b32_e32 v8, 1.0
v_mov_b32_e32 v9, 0x2b800000
v_mov_b32_e32 v19, 0xab800000
v_mov_b32_e32 v14, 0xabcd1234
v_fma_mixlo_f16 v14, v7, v8, v9
v_mov_b32_e32 v15, 0xabcd1234
v_fma_mixhi_f16 v15, -v7, v8, v19
v_mov_b32_e32 v16, 0xabcd1234
v_fma_mixlo_f16 v16, -v1, v2, v4 op_sel_hi:[1,1,1] clamp
v_mov_b32_e32 v20, 0x43960000
v_mov_b32_e32 v21, 0
v_mov_b32_e32 v17, 0xabcd1234
v_fma_mixlo_f16 v17, v20, v20, v21
v_cmpx_eq_u32_e32 0, v0
v_mov_b32_e32 v30, 0
buffer_store_dword v10, v30, s[0:3], 0 offen
v_mov_b32_e32 v30, 4
buffer_store_dword v11, v30, s[0:3], 0 offen
v_mov_b32_e32 v30, 8
buffer_store_dword v12, v30, s[0:3], 0 offen
v_mov_b32_e32 v30, 12
buffer_store_dword v13, v30, s[0:3], 0 offen
v_mov_b32_e32 v30, 16
buffer_store_dword v14, v30, s[0:3], 0 offen
v_mov_b32_e32 v30, 20
buffer_store_dword v15, v30, s[0:3], 0 offen
v_mov_b32_e32 v30, 24
buffer_store_dword v16, v30, s[0:3], 0 offen
v_mov_b32_e32 v30, 28
buffer_store_dword v17, v30, s[0:3], 0 offen
s_endpgm
