; P2 writes the interpolated half (1.75 = 0x3f00) over the incoming I and J VGPRs.
; Their untouched high halves are 0x3e80 (I = 0.25) and 0x3f00 (J = 0.5).
v_interp_p1ll_f16 v4, v0, attr0.x
v_interp_p2_f16 v0, v1, attr0.x, v4
v_interp_p2_f16 v1, v1, attr0.x, v4
exp mrt0 v0, v1, v0, v1 done vm
s_endpgm
