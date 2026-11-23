; cs_fma_fused as it was before TEST-1085 (#8556): the same eight multiply-adds, but the f16 triple of out[7] is three
; literals instead of s8..s10. The expected words and their derivations are in cs_fma_fused.asm.
; It is a known Metal miscompile: on an Apple M1 Max and an M5 Pro (MoltenVK 1.4.2), out[7] reads 0. The f16
; multiply-add alone, or after one f32 multiply-add, gives 0x7351 there, so the f32 part stays. The macOS GPU job lists
; this test in EXPECTED_FAILURES, and its notice shows when a Metal or MoltenVK update fixes it.
; Run: VKC_SGPRS=0x3f800800,0x3f800001,0x3f7ffffe,0xbf800002 vk_compute_test cs_fma_fused_literal.spv 1 dump
; out[0..7] = 33800000 a8800000 3a000400 33800000 33800000 33800000 28800000 00007351 (hex)
; group: 1,1,1
; rsrc2: 0x8090
v_mov_b32_e32 v0, s4
v_mul_f32_e32 v3, v0, v0
v_mov_b32_e32 v20, 0
v_fma_f32 v4, v0, v0, -v3
buffer_store_dword v4, v20, s[0:3], 0 offen
v_mov_b32_e32 v5, s5
v_mov_b32_e32 v6, s6
v_fma_f32 v7, v5, v6, -1.0
buffer_store_dword v7, v20, s[0:3], 0 offen offset:4
v_fma_f32 v8, v0, v0, -1.0
buffer_store_dword v8, v20, s[0:3], 0 offen offset:8
v_fmaak_f32 v9, v0, v0, 0xbf801000
buffer_store_dword v9, v20, s[0:3], 0 offen offset:12
v_sub_f32_e32 v10, 0, v3
v_fmamk_f32 v11, v0, 0x3f800800, v10
buffer_store_dword v11, v20, s[0:3], 0 offen offset:16
v_mov_b32_e32 v12, v10
v_fmac_f32_e32 v12, v0, v0
buffer_store_dword v12, v20, s[0:3], 0 offen offset:20
v_mov_b32_e32 v13, s7
v_fma_f32 v14, v5, v5, v13
buffer_store_dword v14, v20, s[0:3], 0 offen offset:24
v_mov_b32_e32 v15, 0x6ce1
v_mov_b32_e32 v16, 0x4200
v_mov_b32_e32 v17, 0x8004
v_fma_f16 v18, v15, v16, v17
v_and_b32_e32 v18, 0xffff, v18
buffer_store_dword v18, v20, s[0:3], 0 offen offset:28
s_endpgm
