; Lanterns G-buffer vertex shader (stage 9): vs_lanterns.asm without the shade in the colour. The vertex's shade word
; carries the face's packed normal instead (lanterns_deferred.h), passed on as param2.x for ps_lanterns_gbuf.
; stage: vs
; wave: 32
;
; User SGPRs (root signature order, docs/d3d12_vulkan.md 3.5):
;   s[0:1]   table 0: console VA of the mesh's V#      (stride 16: position xyz, face shade)
;   s[2:3]   table 1: console VA of the draw's instance V# (stride 32: offset xyz, scale, colour rgba)
;   s4..s15  root constants: the draw's 3x4 model matrix, row-major
;   s16..s19 root constants: the draw's tint (rgba)
;   s[20:21] root CBV: console VA of the camera's view-projection matrix (16 floats, row-major)
; v0 = vertex index (from the index buffer), v3 = instance index; the draws use BaseVertex 0 and StartInstance 0.
;   local  = M * (pos, 1)                world = local * scale + offset        pos0 = VP * (world, 1)
;   param0 = tint * colour (the albedo), param1 = (world, 1), param2 = (packed normal, 0, 0, 0): flat
s_load_dwordx4 s[24:27], s[0:1], 0x0
s_load_dwordx4 s[28:31], s[2:3], 0x0
s_load_dwordx16 s[32:47], s[20:21], 0x0
s_waitcnt lgkmcnt(0)
buffer_load_dwordx4 v[4:7], v0, s[24:27], 0 idxen
buffer_load_dwordx4 v[8:11], v3, s[28:31], 0 idxen
buffer_load_dwordx4 v[12:15], v3, s[28:31], 0 idxen offset:16
s_waitcnt vmcnt(0)
v_fma_f32 v16, s4, v4, s7
v_fmac_f32_e32 v16, s5, v5
v_fmac_f32_e32 v16, s6, v6
v_fma_f32 v17, s8, v4, s11
v_fmac_f32_e32 v17, s9, v5
v_fmac_f32_e32 v17, s10, v6
v_fma_f32 v18, s12, v4, s15
v_fmac_f32_e32 v18, s13, v5
v_fmac_f32_e32 v18, s14, v6
v_fma_f32 v19, v16, v11, v8
v_fma_f32 v20, v17, v11, v9
v_fma_f32 v21, v18, v11, v10
v_fma_f32 v22, s32, v19, s35
v_fmac_f32_e32 v22, s33, v20
v_fmac_f32_e32 v22, s34, v21
v_fma_f32 v23, s36, v19, s39
v_fmac_f32_e32 v23, s37, v20
v_fmac_f32_e32 v23, s38, v21
v_fma_f32 v24, s40, v19, s43
v_fmac_f32_e32 v24, s41, v20
v_fmac_f32_e32 v24, s42, v21
v_fma_f32 v25, s44, v19, s47
v_fmac_f32_e32 v25, s45, v20
v_fmac_f32_e32 v25, s46, v21
v_mul_f32_e32 v26, s16, v12
v_mul_f32_e32 v27, s17, v13
v_mul_f32_e32 v28, s18, v14
v_mul_f32_e32 v29, s19, v15
v_mov_b32_e32 v30, 1.0
v_mov_b32_e32 v31, 0
exp pos0 v22, v23, v24, v25 done
exp param0 v26, v27, v28, v29
exp param1 v19, v20, v21, v30
exp param2 v7, v31, v31, v31
s_endpgm
