; Lanterns fullscreen vertex shader (stage 5): one triangle over the whole target from the vertex index alone
; (DrawInstanced(3, 1, 0, 0), no vertex buffer): index 0 -> (-1, 1), 1 -> (3, 1), 2 -> (-1, -3). No parameters.
; stage: vs
; wave: 32
v_mov_b32_e32 v1, -1.0
v_mov_b32_e32 v2, 1.0
v_mov_b32_e32 v3, 0x40400000
v_mov_b32_e32 v4, 0xc0400000
v_cmp_eq_u32_e32 vcc_lo, 1, v0
v_cndmask_b32_e32 v1, v1, v3, vcc_lo
v_cmp_eq_u32_e32 vcc_lo, 2, v0
v_cndmask_b32_e32 v2, v2, v4, vcc_lo
v_mov_b32_e32 v5, 0
v_mov_b32_e32 v6, 1.0
exp pos0 v1, v2, v5, v6 done
s_endpgm
