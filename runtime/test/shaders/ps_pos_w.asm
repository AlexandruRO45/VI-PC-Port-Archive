; SV_Position.xyzw in v2..v5 (the input signature declares SV_Position, register 0, mask 0xf), drawn with vs_pos_w:
; clip w = 2, z = 0.5 (0.25 after the divide). What the hardware puts in those registers, measured on an AMD Radeon
; iGPU (gfx1036, tools/ps_sysvals_probe.py): v2 and v3 are the pixel centre (x + 0.5, y + 0.5), v4 is z = 0.25, and v5
; is the clip-space w, 2.0, not 1/w (Vulkan's FragCoord.w is 0.5 here). The shader recovers SV_Position.w = 1 / v5.
;   out.x = v_rcp_f32(v5)             = 0.5  = 0x3f000000
;   out.y = v4                        = 0.25 = 0x3e800000
;   out.z = v5                        = 2.0  = 0x40000000
;   out.w = fract(v2) + fract(v3)     = 0.5 + 0.5 = 1.0 = 0x3f800000
; The translator filled v5 with FragCoord.w itself (0.5) until the same change that added this test, which gave
; out.x = 2.0 and out.z = 0.5.
v_rcp_f32_e32 v6, v5
v_fract_f32_e32 v7, v2
v_fract_f32_e32 v8, v3
v_add_f32_e32 v7, v7, v8
exp mrt0 v6, v4, v5, v7 done vm
s_endpgm
