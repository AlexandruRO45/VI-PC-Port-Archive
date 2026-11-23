; End-to-end GPU test for all 6 explicit-gradient image_sample_d_{g16,cl_g16,o,cl_o,o_g16,cl_o_g16}
; variants against vk_texture_test's --two-mip texture (128x128 quadrants: TL red, TR green, BL blue,
; BR transparent yellow; mip 1: 64x64 solid cyan).
; Six horizontal bands across the 64x64 framebuffer (y in 0..63), each split at x = 32 (u = 0.5):
;   Band 0 (y < 10):       image_sample_d_g16       -- g16 grad 1/128 (L: red) vs 1/64 (R: cyan)
;   Band 1 (10 <= y < 20): image_sample_d_cl_g16    -- a16 + g16 + clamp 0.0 (L: blue) vs 1.0 (R: cyan)
;   Band 2 (20 <= y < 30): image_sample_d_o         -- offset 0 (L: red) vs +3 across midline (R: green)
;   Band 3 (30 <= y < 40): image_sample_d_cl_o      -- offset +3 + clamp 0.0 (L: green) vs 1.0 (R: cyan)
;   Band 4 (40 <= y < 50): image_sample_d_o_g16     -- g16 + offset -3 (L: blue) vs +3 (R: yellow)
;   Band 5 (50 <= y < 64): image_sample_d_cl_o_g16  -- a16 + g16 + offset +3 + clamp 0 (L: yellow) vs 1 (R: cyan)
v_interp_p1_f32_e32 v2, v0, attr0.x
v_interp_p2_f32_e32 v2, v1, attr0.x
v_interp_p1_f32_e32 v3, v0, attr0.y
v_interp_p2_f32_e32 v3, v1, attr0.y
s_load_dwordx8 s[8:15], s[0:1], 0x0
s_waitcnt lgkmcnt(0)
s_mov_b32 s16, 0x92
s_mov_b32 s17, 0xc00000
s_mov_b32 s18, 0x14500000
s_mov_b32 s19, 0

; vcc_lo = (u >= 0.5)  (right half x >= 32)
v_cmp_le_f32_e32 vcc_lo, 0.5, v2

; --- Band 0: image_sample_d_g16 (v[20:23]) ---
; g16 dx: (1/128, 0) = 0x00002000 on left, (1/64, 0) = 0x00002400 on right
; g16 dy: (0, 1/128) = 0x20000000 on left, (0, 1/64) = 0x24000000 on right
; f32 coords: (0.25, 0.25) -> mip 0 TL red, mip 1 cyan
v_mov_b32_e32 v14, 0x00002000
v_mov_b32_e32 v15, 0x00002400
v_cndmask_b32_e32 v8, v14, v15, vcc_lo
v_mov_b32_e32 v14, 0x20000000
v_mov_b32_e32 v15, 0x24000000
v_cndmask_b32_e32 v9, v14, v15, vcc_lo
v_mov_b32_e32 v10, 0x3e800000 ; 0.25
v_mov_b32_e32 v11, 0x3e800000 ; 0.25
image_sample_d_g16 v[20:23], v[8:11], s[8:15], s[16:19] dmask:0xf dim:SQ_RSRC_IMG_2D

; --- Band 1: image_sample_d_cl_g16 a16 (v[24:27]) ---
; g16 dx/dy = 1/128 (LOD 0); a16 coords (0.25, 0.75) = 0x3a003400 (BL blue)
; a16 clamp: 0.0 (pad 1.0) on left, 1.0 (pad 1.0) on right
v_mov_b32_e32 v8, 0x00002000
v_mov_b32_e32 v9, 0x20000000
v_mov_b32_e32 v10, 0x3a003400
v_mov_b32_e32 v14, 0x3c000000 ; clamp 0.0, pad 1.0
v_mov_b32_e32 v15, 0x3c003c00 ; clamp 1.0, pad 1.0
v_cndmask_b32_e32 v11, v14, v15, vcc_lo
image_sample_d_cl_g16 v[24:27], v[8:11], s[8:15], s[16:19] dmask:0xf dim:SQ_RSRC_IMG_2D a16

; --- Band 2: image_sample_d_o (v[28:31]) ---
; offset: ox=0 on left (TL red), ox=+3 on right (TR green); f32 grads=1/128; coords=(62.5/128, 0.25)
v_mov_b32_e32 v14, 0
v_mov_b32_e32 v15, 3
v_cndmask_b32_e32 v8, v14, v15, vcc_lo
v_mov_b32_e32 v9, 0x3c000000  ; du/dx = 1/128
v_mov_b32_e32 v10, 0          ; dv/dx
v_mov_b32_e32 v11, 0          ; du/dy
v_mov_b32_e32 v12, 0x3c000000 ; dv/dy = 1/128
v_mov_b32_e32 v13, 0x3efa0000 ; u = 62.5 / 128
v_mov_b32_e32 v14, 0x3e800000 ; v = 0.25
image_sample_d_o v[28:31], v[8:14], s[8:15], s[16:19] dmask:0xf dim:SQ_RSRC_IMG_2D

; --- Band 3: image_sample_d_cl_o (v[32:35]) ---
; offset: ox=+3 everywhere (TR green at mip 0); f32 grads=1/128; coords=(62.5/128, 0.25)
; f32 clamp: 0.0 on left (TR green), 1.0 on right (mip 1 cyan)
v_mov_b32_e32 v8, 3
v_mov_b32_e32 v9, 0x3c000000
v_mov_b32_e32 v10, 0
v_mov_b32_e32 v11, 0
v_mov_b32_e32 v12, 0x3c000000
v_mov_b32_e32 v13, 0x3efa0000
v_mov_b32_e32 v14, 0x3e800000
v_mov_b32_e32 v16, 0
v_mov_b32_e32 v17, 0x3f800000 ; 1.0
v_cndmask_b32_e32 v15, v16, v17, vcc_lo
image_sample_d_cl_o v[32:35], v[8:15], s[8:15], s[16:19] dmask:0xf dim:SQ_RSRC_IMG_2D

; --- Band 4: image_sample_d_o_g16 (v[36:39]) ---
; offset: ox=-3 (0x3d) on left (BL blue), ox=+3 on right (BR yellow); g16 grads=1/128; f32 coords=(62.5/128, 0.75)
v_mov_b32_e32 v16, 0x3d       ; signed 6-bit -3
v_mov_b32_e32 v17, 3          ; signed 6-bit +3
v_cndmask_b32_e32 v8, v16, v17, vcc_lo
v_mov_b32_e32 v9, 0x00002000
v_mov_b32_e32 v10, 0x20000000
v_mov_b32_e32 v11, 0x3efa0000 ; u = 62.5 / 128
v_mov_b32_e32 v12, 0x3f400000 ; v = 0.75
image_sample_d_o_g16 v[36:39], v[8:12], s[8:15], s[16:19] dmask:0xf dim:SQ_RSRC_IMG_2D

; --- Band 5: image_sample_d_cl_o_g16 a16 (v[40:43]) ---
; offset: ox=+3 (BR yellow at mip 0); g16 grads=1/128; a16 coords=(62.5/128, 0.75)=0x3a0037d0
; a16 clamp: 0.0 (pad 1.0) on left (BR yellow), 1.0 (pad 1.0) on right (mip 1 cyan)
v_mov_b32_e32 v8, 3
v_mov_b32_e32 v9, 0x00002000
v_mov_b32_e32 v10, 0x20000000
v_mov_b32_e32 v11, 0x3a0037d0
v_mov_b32_e32 v16, 0x3c000000 ; clamp 0.0, pad 1.0
v_mov_b32_e32 v17, 0x3c003c00 ; clamp 1.0, pad 1.0
v_cndmask_b32_e32 v12, v16, v17, vcc_lo
image_sample_d_cl_o_g16 v[40:43], v[8:12], s[8:15], s[16:19] dmask:0xf dim:SQ_RSRC_IMG_2D a16

s_waitcnt vmcnt(0)

; Select band into v[4:7] based on v = (y + 0.5) / 64
v_mov_b32_e32 v4, v20
v_mov_b32_e32 v5, v21
v_mov_b32_e32 v6, v22
v_mov_b32_e32 v7, v23

v_cmp_le_f32_e32 vcc_lo, 0x3e200000, v3 ; v >= 10/64 -> Band 1
v_cndmask_b32_e32 v4, v4, v24, vcc_lo
v_cndmask_b32_e32 v5, v5, v25, vcc_lo
v_cndmask_b32_e32 v6, v6, v26, vcc_lo
v_cndmask_b32_e32 v7, v7, v27, vcc_lo

v_cmp_le_f32_e32 vcc_lo, 0x3ea00000, v3 ; v >= 20/64 -> Band 2
v_cndmask_b32_e32 v4, v4, v28, vcc_lo
v_cndmask_b32_e32 v5, v5, v29, vcc_lo
v_cndmask_b32_e32 v6, v6, v30, vcc_lo
v_cndmask_b32_e32 v7, v7, v31, vcc_lo

v_cmp_le_f32_e32 vcc_lo, 0x3ef00000, v3 ; v >= 30/64 -> Band 3
v_cndmask_b32_e32 v4, v4, v32, vcc_lo
v_cndmask_b32_e32 v5, v5, v33, vcc_lo
v_cndmask_b32_e32 v6, v6, v34, vcc_lo
v_cndmask_b32_e32 v7, v7, v35, vcc_lo

v_cmp_le_f32_e32 vcc_lo, 0x3f200000, v3 ; v >= 40/64 -> Band 4
v_cndmask_b32_e32 v4, v4, v36, vcc_lo
v_cndmask_b32_e32 v5, v5, v37, vcc_lo
v_cndmask_b32_e32 v6, v6, v38, vcc_lo
v_cndmask_b32_e32 v7, v7, v39, vcc_lo

v_cmp_le_f32_e32 vcc_lo, 0x3f480000, v3 ; v >= 50/64 -> Band 5
v_cndmask_b32_e32 v4, v4, v40, vcc_lo
v_cndmask_b32_e32 v5, v5, v41, vcc_lo
v_cndmask_b32_e32 v6, v6, v42, vcc_lo
v_cndmask_b32_e32 v7, v7, v43, vcc_lo

exp mrt0 v4, v5, v6, v7 done vm
s_endpgm
