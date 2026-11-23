; SHADER-821 tier A (docs/design.md, "Wave64 graphics: partition invariance"): a compiler's `break` loop in a wave64
; pixel shader. The lanes that left the loop are collected in an SGPR pair by `s_or_b64 D, vcc, D` after a v_cmp, the
; pair narrows EXEC inside the loop and widens it again behind it. Each bit of the pair belongs to one lane, so the
; pair is the same in the wave and in a 32-lane half although its in-loop definition sits behind the loop's EXEC vote.
; x = floor(u * 64), y = floor(v * 64) are the pixel's coordinates on the 64x64 target (u, v = attr0.xy of vs_ids), so
; every expected word is a function of (x, y) and not of the lane the pixel landed on.
;  (1) a break loop on x: s[16:17] collects the lanes with v9 >= x % 4 + 1; v11 counts behind the break, so only the
;      lanes that stay see it. Lanes of one quad leave at different trips (x differs inside a quad).
;      r0 = v9 = x % 4 + 1 (the trips), r3 = v11 = x % 4 (the trips that went on behind the break).
;  (2) `s_or_b64 exec, exec, s[16:17]` turns the lanes back on: r1 = 1000 + x is defined behind it, and a pixel whose
;      lane stayed off would keep the cleared target (the export below is the only one).
;  (3) a second break loop on y with its own pair s[18:19], 1..8 trips: a half closes it after the largest y % 8 + 1
;      of its own rows, which can be fewer trips than the wave needs. r2 = 2000 + 5 * (y % 8 + 1).
; The shader keeps no copy of EXEC: the export runs with the EXEC the two pairs rebuilt, so a lane whose bit was lost
; from a pair keeps the cleared target and a lane whose bit was set early leaves its loop with too few trips.
; Run: vk_harness vs_ids.spv ps_w64_a_break.spv out.raw, once with VKH_SUBGROUP_SIZE=64 and once with 32: the two
; images must be equal (and equal to the expected words). Output is raw bits (r0, r1, r2, r3) per pixel.
; mattr: +wavefrontsize64,-wavefrontsize32
v_interp_p1_f32_e32 v2, v0, attr0.x
v_interp_p2_f32_e32 v2, v1, attr0.x
v_interp_p1_f32_e32 v3, v0, attr0.y
v_interp_p2_f32_e32 v3, v1, attr0.y
v_mul_f32_e32 v2, 0x42800000, v2
v_mul_f32_e32 v3, 0x42800000, v3
v_cvt_u32_f32_e32 v2, v2
v_cvt_u32_f32_e32 v3, v3
v_and_b32_e32 v4, 3, v2
v_add_nc_u32_e32 v10, 1, v4
v_mov_b32_e32 v9, 0
v_mov_b32_e32 v11, 0
s_mov_b64 s[16:17], 0
v_add_nc_u32_e32 v9, 1, v9
v_cmp_ge_u32_e32 vcc, v9, v10
s_or_b64 s[16:17], vcc, s[16:17]
s_andn2_b64 exec, exec, s[16:17]
v_add_nc_u32_e32 v11, 1, v11
s_cbranch_execnz -6
s_or_b64 exec, exec, s[16:17]
v_add_nc_u32_e32 v7, 0x3e8, v2
v_and_b32_e32 v5, 7, v3
v_add_nc_u32_e32 v12, 1, v5
v_mov_b32_e32 v13, 0
v_mov_b32_e32 v8, 0x7d0
s_mov_b64 s[18:19], 0
v_add_nc_u32_e32 v13, 1, v13
v_add_nc_u32_e32 v8, 5, v8
v_cmp_ge_u32_e32 vcc, v13, v12
s_or_b64 s[18:19], vcc, s[18:19]
s_andn2_b64 exec, exec, s[18:19]
s_cbranch_execnz -6
s_or_b64 exec, exec, s[18:19]
exp mrt0 v9, v7, v8, v11 done vm
s_endpgm
