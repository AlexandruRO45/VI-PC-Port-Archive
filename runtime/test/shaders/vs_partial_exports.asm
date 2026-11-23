; regen_fixtures: this source has a translator fixture in tests/semantic/fixtures/disasm
; Vertex shader for ps_partial_exports_vs: partial vertex exports keep the channels an earlier export of the same
; target wrote (docs/design.md "Partial pixel color and vertex exports"). The vs_ids full-screen triangle, with
; pos0 written as x,y and then z,w, and param0 written in three partial exports with the same constant at all
; three vertices:
;   exp param0 x,y = (1.5, -0.25)       -> (1.5, -0.25, 0, 0)
;   exp param0 z,w = (0.375, 0.625)     -> (1.5, -0.25, 0.375, 0.625)
;   exp param0 x   = 2.0                -> (2.0, -0.25, 0.375, 0.625)
; A translator that writes 0.0 into the off channels gives pos0 = (0, 0, 0, 1), so nothing is drawn, and param0 =
; (2.0, 0, 0, 0).
v_lshlrev_b32_e32 v1, 1, v0
v_and_b32_e32 v1, 2, v1
v_and_b32_e32 v2, 2, v0
v_cvt_f32_u32_e32 v1, v1
v_cvt_f32_u32_e32 v2, v2
v_fma_f32 v3, v1, 2.0, -1.0
v_fma_f32 v4, v2, 2.0, -1.0
v_mov_b32_e32 v5, 0
v_mov_b32_e32 v6, 1.0
exp pos0 v3, v4, off, off
exp pos0 off, off, v5, v6 done
v_mov_b32_e32 v7, 0x3fc00000
v_mov_b32_e32 v8, 0xbe800000
v_mov_b32_e32 v9, 0x3ec00000
v_mov_b32_e32 v10, 0x3f200000
v_mov_b32_e32 v11, 2.0
exp param0 v7, v8, off, off
exp param0 off, off, v9, v10
exp param0 v11, off, off, off
s_endpgm
