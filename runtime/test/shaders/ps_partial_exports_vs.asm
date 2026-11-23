; regen_fixtures: this source has a translator fixture in tests/semantic/fixtures/disasm
; Partial vertex exports (docs/design.md "Partial pixel color and vertex exports"). Drawn with vs_partial_exports,
; which writes pos0 and param0 in partial exports. attr0 is read only with v_interp_mov (so its location is Flat),
; and the words are exact: attr0 = (2.0, -0.25, 0.375, 0.625) at every vertex.
; Run: vk_harness vs_partial_exports.spv ps_partial_exports_vs.spv out.raw
; Expected mrt0 bits: 0x40000000, 0xbe800000, 0x3ec00000, 0x3f200000
v_interp_mov_f32_e32 v2, p0, attr0.x
v_interp_mov_f32_e32 v3, p0, attr0.y
v_interp_mov_f32_e32 v4, p0, attr0.z
v_interp_mov_f32_e32 v5, p0, attr0.w
exp mrt0 v2, v3, v4, v5 done vm
s_endpgm
