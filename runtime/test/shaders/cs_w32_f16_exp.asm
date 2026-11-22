; v_exp_f16 on exact inputs (2^x with x an integer): the f16 transcendental reads the LOW half of a VGPR, writes the
; low half and keeps the high half of the destination (VOP1 and VOP3 forms), and takes neg/abs on the low half.
; The source registers carry a different, valid f16 in their high half, so a read of the wrong half gives another word.
; D.f16 = pow(2.0, S0.f16) (RDNA2 ISA VOP1 V_EXP_F16). Every result below is exact in f16, so a driver's exp2 only
; has to be accurate to a half ulp. Hardware, measured (docs/design.md, f16 transcendentals): +inf -> +inf,
; -inf -> +0, a NaN stays a NaN; the 2^16 overflow is +inf. A NaN's payload is the driver's, so a NaN is stored as a
; flag (v_cmp_u_f16).
; f16 bits used: 1.0 3c00, 2.0 4000, 4.0 4400, 0.5 3800, 16.0 4c00, 0.0625 2c00, +inf 7c00, -inf fc00, qNaN 7e00.
;   out[0] = v_exp_f16_e32      lo  2.0           = 4.0   keep hi 0xbeef                  = 0xbeef4400
;   out[1] = v_exp_f16_e64      lo -1.0           = 0.5   keep hi 0x1234                  = 0x12343800
;   out[2] = v_exp_f16_e64 -v   lo -(-1.0) = 1.0  = 2.0   keep hi 0xbeef                  = 0xbeef4000
;   out[3] = v_exp_f16_e64 |v|  lo |-4.0| = 4.0   = 16.0  keep hi 0xbeef                  = 0xbeef4c00
;   out[4] = v_exp_f16_e64 -|v| lo -|-1.0| = -1.0 = 0.5   keep hi 0xbeef                  = 0xbeef3800
;   out[5] = v_pack_b32_f16 (exp(+inf) = 7c00, exp(-inf) = 0000)                          = 0x00007c00
;   out[6] = v_pack_b32_f16 (exp(0.0) = 1.0 3c00, exp(16.0) = 2^16 > 65504 = +inf 7c00)   = 0x7c003c00
;   out[7] = 1 if v_exp_f16 of a quiet NaN (lo 7e00, hi 8000) is a NaN, else 0            = 1
; group: 32,1,1
; Run: vk_compute_test cs_w32_f16_exp.spv 1 dump -> out[0..7] = 3203351552 305412096 3203350528 3203353600 3203348480 31744 2080390144 1
; mattr: +wavefrontsize32,-wavefrontsize64
v_cmpx_eq_u32_e32 0, v0
v_mov_b32_e32 v1, 0xc0004000
v_mov_b32_e32 v2, 0x4000bc00
v_mov_b32_e32 v3, 0x3c00c400
v_mov_b32_e32 v4, 0x3c007c00
v_mov_b32_e32 v5, 0x3c00fc00
v_mov_b32_e32 v6, 0x40000000
v_mov_b32_e32 v7, 0x40004c00
v_mov_b32_e32 v8, 0x80007e00
v_mov_b32_e32 v10, 0xbeef5555
v_exp_f16_e32 v10, v1
v_mov_b32_e32 v11, 0x12345555
v_exp_f16_e64 v11, v2
v_mov_b32_e32 v12, 0xbeef5555
v_exp_f16_e64 v12, -v2
v_mov_b32_e32 v13, 0xbeef5555
v_exp_f16_e64 v13, |v3|
v_mov_b32_e32 v14, 0xbeef5555
v_exp_f16_e64 v14, -|v2|
v_exp_f16_e32 v15, v4
v_exp_f16_e32 v16, v5
v_pack_b32_f16 v15, v15, v16
v_exp_f16_e32 v16, v6
v_exp_f16_e32 v17, v7
v_pack_b32_f16 v16, v16, v17
v_exp_f16_e32 v17, v8
v_cmp_u_f16_e32 vcc_lo, v17, v17
v_cndmask_b32_e64 v17, 0, 1, vcc_lo
v_mov_b32_e32 v25, 0
buffer_store_dword v10, v25, s[0:3], 0 offen
v_mov_b32_e32 v25, 4
buffer_store_dword v11, v25, s[0:3], 0 offen
v_mov_b32_e32 v25, 8
buffer_store_dword v12, v25, s[0:3], 0 offen
v_mov_b32_e32 v25, 12
buffer_store_dword v13, v25, s[0:3], 0 offen
v_mov_b32_e32 v25, 16
buffer_store_dword v14, v25, s[0:3], 0 offen
v_mov_b32_e32 v25, 20
buffer_store_dword v15, v25, s[0:3], 0 offen
v_mov_b32_e32 v25, 24
buffer_store_dword v16, v25, s[0:3], 0 offen
v_mov_b32_e32 v25, 28
buffer_store_dword v17, v25, s[0:3], 0 offen
s_endpgm
