; --vs-compute GPU fixture (issue #94; d3d12_vs_compute_test): each three vertices make a triangle over the whole
; target (corner v0 % 3 at (-1, -1), (3, -1), (-1, 3)), coloured by the number of lanes of the vertex's 64-lane wave
; that are active (s_bcnt1 of EXEC) / 64. Tier C (wave64_halves): a lane count reaches an export. In a draw of 48
; vertices the last triangle (vertices 45..47) is in a wave of 48 lanes: 0.75 on the hardware and with the --vs-compute
; prepass; at the device's 32-wide subgroup size it is in the second half, 16 lanes: 0.25.
; mattr: +wavefrontsize64,-wavefrontsize32
; fssm-at: 0x10
; fssm-data: 0x77637376 0x00000001
v_mul_hi_u32 v1, v0, 0xaaaaaaab
v_lshrrev_b32_e32 v1, 1, v1
v_mul_lo_u32 v2, v1, 3
v_sub_nc_u32_e32 v2, v0, v2
v_lshlrev_b32_e32 v3, 1, v2
v_and_b32_e32 v3, 2, v3
v_and_b32_e32 v4, 2, v2
v_cvt_f32_u32_e32 v3, v3
v_cvt_f32_u32_e32 v4, v4
v_fma_f32 v5, v3, 2.0, -1.0
v_fma_f32 v6, v4, 2.0, -1.0
s_bcnt1_i32_b64 s10, exec
v_cvt_f32_u32_e32 v7, s10
v_mul_f32_e32 v7, 0x3c800000, v7
v_mov_b32_e32 v8, 0
v_mov_b32_e32 v9, 1.0
exp pos0 v5, v6, v8, v9 done
exp param0 v7, v7, v7, v9
s_endpgm
