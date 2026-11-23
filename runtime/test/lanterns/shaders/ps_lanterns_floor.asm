; Lanterns pixel shader for the floor (stage 1): a checker from the interpolated world position, in two colours that
; sit in the shader's own FSSM program and are read through the code base, times the draw's tint (read flat).
; The first instruction is s_getpc_b64, which returns the address of the next one: the first instruction + 4. The
; colours are "data-at" 0x100 bytes after the first instruction, so the offset added is 0x100 - 4 = 0xfc.
;   parity = (floor(world.x) + floor(world.z)) & 1
;   mrt0   = (parity == 0 ? A : B) * tint        A = (0.4375, 0.4375, 0.5, 1), B = (0.125, 0.1875, 0.25, 1)
; The same values are kFloorA and kFloorB in lanterns_scene.h.
; stage: ps
; wave: 32
; input: flat
; input: interp
; data-at: 0x100
; data: 0x3ee00000 0x3ee00000 0x3f000000 0x3f800000
; data: 0x3e000000 0x3e400000 0x3e800000 0x3f800000
s_getpc_b64 s[6:7]
s_add_u32 s6, s6, 0xfc
s_addc_u32 s7, s7, 0
s_load_dwordx8 s[8:15], s[6:7], 0x0
v_interp_p1_f32_e32 v2, v0, attr1.x
v_interp_p2_f32_e32 v2, v1, attr1.x
v_interp_p1_f32_e32 v3, v0, attr1.z
v_interp_p2_f32_e32 v3, v1, attr1.z
v_interp_mov_f32 v10, p0, attr0.x
v_interp_mov_f32 v11, p0, attr0.y
v_interp_mov_f32 v12, p0, attr0.z
v_interp_mov_f32 v13, p0, attr0.w
v_floor_f32_e32 v2, v2
v_floor_f32_e32 v3, v3
v_cvt_i32_f32_e32 v2, v2
v_cvt_i32_f32_e32 v3, v3
v_add_nc_u32_e32 v2, v2, v3
v_and_b32_e32 v2, 1, v2
v_cmp_eq_u32_e32 vcc_lo, 0, v2
s_waitcnt lgkmcnt(0)
v_mov_b32_e32 v4, s8
v_mov_b32_e32 v5, s9
v_mov_b32_e32 v6, s10
v_mov_b32_e32 v7, s11
v_cndmask_b32_e32 v4, s12, v4, vcc_lo
v_cndmask_b32_e32 v5, s13, v5, vcc_lo
v_cndmask_b32_e32 v6, s14, v6, vcc_lo
v_cndmask_b32_e32 v7, s15, v7, vcc_lo
v_mul_f32_e32 v4, v4, v10
v_mul_f32_e32 v5, v5, v11
v_mul_f32_e32 v6, v6, v12
v_mul_f32_e32 v7, v7, v13
exp mrt0 v4, v5, v6, v7 done vm
s_endpgm
