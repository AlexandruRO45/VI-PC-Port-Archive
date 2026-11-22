; v_log_f16 on exact inputs (log2 of a power of two): reads the LOW half of a VGPR, writes the low half and keeps the
; high half of the destination (VOP1 and VOP3 forms), takes neg/abs on the low half. The source registers carry a
; different, valid f16 in their high half, so a read of the wrong half gives another word.
; D.f16 = log2(S0.f16) (RDNA2 ISA VOP1 V_LOG_F16). Every result below is exact in f16. Hardware, measured
; (docs/design.md, f16 transcendentals): log(+inf) = +inf and log of a NaN is a NaN. A NaN's payload is the driver's,
; so a NaN is stored as a flag (v_cmp_u_f16). log of zero or of a negative number is in cs_w32_f16_log_undef.
; f16 bits used: 1.0 3c00, 2.0 4000, 4.0 4400, 3.0 4200, -1.0 bc00, -4.0 c400, 8.0 4800, 16.0 4c00, 0.5 3800,
; 0.0625 2c00, +inf 7c00, qNaN 7e00.
;   out[0] = v_log_f16_e32      lo 8.0           = 3.0   keep hi 0xbeef                   = 0xbeef4200
;   out[1] = v_log_f16_e64      lo 0.5           = -1.0  keep hi 0x1234                   = 0x1234bc00
;   out[2] = v_log_f16_e64 |v|  lo |-2.0| = 2.0  = 1.0   keep hi 0xbeef                   = 0xbeef3c00
;   out[3] = v_log_f16_e64 -v   lo -(-4.0) = 4.0 = 2.0   keep hi 0xbeef                   = 0xbeef4000
;   out[4] = v_pack_b32_f16 (log(+inf) = +inf 7c00, log(4.0) = 2.0 4000)                  = 0x40007c00
;   out[5] = v_pack_b32_f16 (log(1.0) = +0.0 0000, log(0.0625) = -4.0 c400)               = 0xc4000000
;   out[6] = v_pack_b32_f16 (log(2.0) = 1.0 3c00, log(16.0) = 4.0 4400)                   = 0x44003c00
;   out[7] = 1 if v_log_f16 of a quiet NaN (lo 7e00) is a NaN, else 0                     = 1
; group: 32,1,1
; Run: vk_compute_test cs_w32_f16_log.spv 1 dump -> out[0..7] = 3203351040 305445888 3203349504 3203350528 1073773568 3288334336 1140866048 1
; mattr: +wavefrontsize32,-wavefrontsize64
v_cmpx_eq_u32_e32 0, v0
v_mov_b32_e32 v1, 0x38004800
v_mov_b32_e32 v2, 0x40003800
v_mov_b32_e32 v3, 0x3c00c000
v_mov_b32_e32 v4, 0x3c00c400
v_mov_b32_e32 v5, 0x3c007c00
v_mov_b32_e32 v6, 0x40004400
v_mov_b32_e32 v7, 0x40003c00
v_mov_b32_e32 v8, 0x40002c00
v_mov_b32_e32 v9, 0x3c004000
v_mov_b32_e32 v18, 0x40004c00
v_mov_b32_e32 v10, 0xbeef5555
v_log_f16_e32 v10, v1
v_mov_b32_e32 v11, 0x12345555
v_log_f16_e64 v11, v2
v_mov_b32_e32 v12, 0xbeef5555
v_log_f16_e64 v12, |v3|
v_mov_b32_e32 v13, 0xbeef5555
v_log_f16_e64 v13, -v4
v_log_f16_e32 v14, v5
v_log_f16_e32 v15, v6
v_pack_b32_f16 v14, v14, v15
v_log_f16_e32 v15, v7
v_log_f16_e32 v16, v8
v_pack_b32_f16 v15, v15, v16
v_log_f16_e32 v16, v9
v_log_f16_e32 v17, v18
v_pack_b32_f16 v16, v16, v17
v_mov_b32_e32 v19, 0x00007e00
v_log_f16_e32 v19, v19
v_cmp_u_f16_e32 vcc_lo, v19, v19
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
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
buffer_store_dword v20, v25, s[0:3], 0 offen
s_endpgm
