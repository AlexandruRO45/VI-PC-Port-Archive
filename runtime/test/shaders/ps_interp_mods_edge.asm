; VOP3 output modifiers on v_interp_mov_f32_e64 and v_interp_p2_f32_e64 at the ends of the f32 range (docs/design.md
; "v_fmac_f16, v_interp_p2/mov_f32, and f16 output modifiers": f32 destinations keep omod, then clamp). Drawn with
; vs_interp_consts: attr3 = (2^126, 2^-125, 0, 0) read only with v_interp_mov (Flat), attr4 = (2^127, 2^-100, 0, 0)
; interpolated.
;   x: mov 2^126 mul:4 = inf, clamp: 1.0 (clamping first would give 4.0)
;   y: interp 2^127 mul:2 = inf, clamp: 1.0 (clamping first would give 2.0)
;   z: mov 2^-125 div:2 = 2^-126, the smallest normal (no omod: 2^-125; div:2 as a doubling: 2^-124)
;   w: interp 2^-100 mul:4 = 2^-98 (no omod: 2^-100; mul:4 as a doubling: 2^-99)
; Run: vk_harness vs_interp_consts.spv ps_interp_mods_edge.spv out.raw
; Expected mrt0 bits: 0x3f800000, 0x3f800000, 0x00800000, 0x0e800000
v_interp_mov_f32_e64 v2, p0, attr3.x clamp mul:4
v_interp_p1_f32_e32 v3, v0, attr4.x
v_interp_p2_f32_e64 v3, v1, attr4.x clamp mul:2
v_interp_mov_f32_e64 v4, p0, attr3.y div:2
v_interp_p1_f32_e32 v5, v0, attr4.y
v_interp_p2_f32_e64 v5, v1, attr4.y mul:4
exp mrt0 v2, v3, v4, v5 done vm
s_endpgm
