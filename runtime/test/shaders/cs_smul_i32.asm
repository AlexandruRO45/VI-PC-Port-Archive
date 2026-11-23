; s_mul_i32 with operands derived from the run-time TGID_X SGPR, so neither the translator nor the driver sees
; literals at the multiply sites. The low 32 bits of each product are written; out[0], out[1], out[5] and out[6]
; overflow.
; The middle three cases make the destination alias src0, src1, then both sources, respectively.
; One group is dispatched, so s4 = TGID_X = 0 and A=0x7fffffff, B=2, C=0x80000001, D=3.
;   out[0] = A * B                 = 0xfffffffe
;   out[1] = C * D                 = 0x80000003
;   out[2] = A * B, dst == src0    = 0xfffffffe
;   out[3] = C * D, dst == src1    = 0x80000003
;   out[4] = B * B, dst == both    = 0x00000004
;   out[5] = A * A                 = 0x00000001
;   out[6] = C * A                 = 0xffffffff
;   out[7] = D * D                 = 0x00000009
; Run: vk_compute_test cs_smul_i32.spv 1 dump
; out[0..7] = 4294967294 2147483651 4294967294 2147483651 4 1 4294967295 9
; group: 1,1,1
s_add_u32 s6, s4, 0x7fffffff
s_add_u32 s7, s4, 2
s_add_u32 s8, s4, 0x80000001
s_add_u32 s9, s4, 3
s_mul_i32 s10, s6, s7
s_mul_i32 s11, s8, s9
s_mov_b32 s12, s6
s_mul_i32 s12, s12, s7
s_mov_b32 s13, s9
s_mul_i32 s13, s8, s13
s_mov_b32 s14, s7
s_mul_i32 s14, s14, s14
s_mul_i32 s15, s6, s6
s_mul_i32 s16, s8, s6
s_mul_i32 s17, s9, s9
v_cmpx_eq_u32_e32 0, v0
v_mov_b32_e32 v20, 0
v_mov_b32_e32 v21, s10
buffer_store_dword v21, v20, s[0:3], 0 offen
v_mov_b32_e32 v21, s11
buffer_store_dword v21, v20, s[0:3], 0 offen offset:4
v_mov_b32_e32 v21, s12
buffer_store_dword v21, v20, s[0:3], 0 offen offset:8
v_mov_b32_e32 v21, s13
buffer_store_dword v21, v20, s[0:3], 0 offen offset:12
v_mov_b32_e32 v21, s14
buffer_store_dword v21, v20, s[0:3], 0 offen offset:16
v_mov_b32_e32 v21, s15
buffer_store_dword v21, v20, s[0:3], 0 offen offset:20
v_mov_b32_e32 v21, s16
buffer_store_dword v21, v20, s[0:3], 0 offen offset:24
v_mov_b32_e32 v21, s17
buffer_store_dword v21, v20, s[0:3], 0 offen offset:28
s_endpgm
