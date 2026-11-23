; group: 32,1,1
; v_cubeid_f32, v_cubesc_f32, v_cubetc_f32 and v_cubema_f32 with NaN, infinite, zero and denormal operands in the game's
; MODE (IEEE 0, f32 denormals flushed). Measured natively on RDNA2 (gfx1036, Ryzen 9 9950X3D iGPU, driver 26.10.1)
; through tools/hw_oracle (raw HIP kernel, wave32), over 4 x 13^3 operand triples: the major-axis compares are float
; compares that a NaN fails and that read a denormal as zero, "less than zero" is false for a NaN, -0 and a denormal;
; id gives the face from those compares, sc and tc pass the raw source words through (a NaN or a denormal survives, neg
; moves the sign bit) and ma is 2 * the major coordinate, +0 for a (flushed) zero. Operands (S0, S1, S2):
;   A = (qNaN, 1.0, -2.0)    B = (+inf, qNaN, +0)    C = (-denormal 0x80000005, +0, -0)
;   word 0 v_cubeid A: the compares against the NaN x fail, so x is the major axis and +X is face 0     -> 0x00000000
;   word 1 v_cubeid C: every operand reads as zero, so z is the major axis, z = -0 is not negative: +Z  -> 0x40800000 (4.0)
;   word 2 v_cubesc A: x major, +X face: sc = -z = 2.0                                                  -> 0x40000000
;   word 3 v_cubesc C: z major, +Z face: sc = x, the raw denormal word passes through                 -> 0x80000005
;   word 4 v_cubetc B: x major (y is a NaN, z = 0 is below +inf), tc = -y: the NaN with its sign flipped -> 0xFFC00000
;   word 5 v_cubetc C: z major, tc = -y = -(+0)                                                         -> 0x80000000
;   word 6 v_cubema B: x is the major axis: 2 * +inf                                                    -> 0x7F800000
;   word 7 v_cubema C: z is the major axis and flushes to zero: +0                                      -> 0x00000000
; The expected words are the measured ones. Every lane stores its eight at lane * 32.
; Run: vk_compute_test cs_cube_nonfinite.spv 1 dump
; out[0..7] = 00000000 40800000 40000000 80000005 FFC00000 80000000 7F800000 00000000 (hex)
s_mov_b32 exec_lo, -1
v_lshlrev_b32_e32 v13, 5, v0
v_mov_b32_e32 v1, 0x7fc00000
v_mov_b32_e32 v2, 0x3f800000
v_mov_b32_e32 v3, 0xc0000000
v_cubeid_f32 v4, v1, v2, v3
buffer_store_dword v4, v13, s[0:3], 0 offen offset:0
v_mov_b32_e32 v1, 0x80000005
v_mov_b32_e32 v2, 0x0
v_mov_b32_e32 v3, 0x80000000
v_cubeid_f32 v4, v1, v2, v3
buffer_store_dword v4, v13, s[0:3], 0 offen offset:4
v_mov_b32_e32 v1, 0x7fc00000
v_mov_b32_e32 v2, 0x3f800000
v_mov_b32_e32 v3, 0xc0000000
v_cubesc_f32 v4, v1, v2, v3
buffer_store_dword v4, v13, s[0:3], 0 offen offset:8
v_mov_b32_e32 v1, 0x80000005
v_mov_b32_e32 v2, 0x0
v_mov_b32_e32 v3, 0x80000000
v_cubesc_f32 v4, v1, v2, v3
buffer_store_dword v4, v13, s[0:3], 0 offen offset:12
v_mov_b32_e32 v1, 0x7f800000
v_mov_b32_e32 v2, 0x7fc00000
v_mov_b32_e32 v3, 0x0
v_cubetc_f32 v4, v1, v2, v3
buffer_store_dword v4, v13, s[0:3], 0 offen offset:16
v_mov_b32_e32 v1, 0x80000005
v_mov_b32_e32 v2, 0x0
v_mov_b32_e32 v3, 0x80000000
v_cubetc_f32 v4, v1, v2, v3
buffer_store_dword v4, v13, s[0:3], 0 offen offset:20
v_mov_b32_e32 v1, 0x7f800000
v_mov_b32_e32 v2, 0x7fc00000
v_mov_b32_e32 v3, 0x0
v_cubema_f32 v4, v1, v2, v3
buffer_store_dword v4, v13, s[0:3], 0 offen offset:24
v_mov_b32_e32 v1, 0x80000005
v_mov_b32_e32 v2, 0x0
v_mov_b32_e32 v3, 0x80000000
v_cubema_f32 v4, v1, v2, v3
buffer_store_dword v4, v13, s[0:3], 0 offen offset:28
s_endpgm
