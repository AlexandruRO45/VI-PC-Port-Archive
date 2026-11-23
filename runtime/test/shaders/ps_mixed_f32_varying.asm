; Mixed flat and smooth reads of one f32 input location (docs/design.md, "Mixed flat and smooth f32 reads"):
; attr1.x is read with v_interp_mov_f32 p0 (the provoking vertex's value) and with v_interp_p1/p2_f32
; (interpolated), so the location is a PerVertexKHR input rather than a Flat one, and both reads of it have to be
; right at once. Drawn with vs_mixed_varying: attr1.x = 1.0, 2.0 and 3.0 at the three vertices, and attr0 = (2i, 2j,
; 0, 1), so a pixel centre reads i = (x + 0.5) / 128 and j = (y + 0.5) / 128.
; Run: vk_harness vs_mixed_varying.spv ps_mixed_f32_varying.spv out.raw
;   -> mrt0.x = 1.0 and mrt0.w = 1.0 (vertex 0, attr1.x and attr1.y) on every pixel, the interpolated mrt0.y =
;      1 + i + 2j, and mrt0.z = 2i interpolated from attr0, which no MOV reads
v_interp_mov_f32_e32 v2, p0, attr1.x
v_interp_p1_f32_e32 v3, v0, attr1.x
v_interp_p2_f32_e32 v3, v1, attr1.x
v_interp_p1_f32_e32 v4, v0, attr0.x
v_interp_p2_f32_e32 v4, v1, attr0.x
v_interp_mov_f32_e32 v5, p0, attr1.y
exp mrt0 v2, v3, v4, v5 done vm
s_endpgm
