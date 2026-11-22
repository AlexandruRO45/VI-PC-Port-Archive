; Packed 16-bit integer ops (VOP3P) with a constant operand, run once on a GPU. The constant is its 32-bit value,
; split into a low half (bits 15:0) and a high half (bits 31:16) like a register; op_sel picks the half for the
; result's low half and op_sel_hi (default 1) for its high half. An integer inline constant is sign-extended to 32
; bits, a float inline constant is its f32 bit pattern, a literal is as written. Measured natively on RDNA2 (gfx1036,
; wave32 and wave64; docs/design.md "Packed 16-bit integer constants"). The RDNA2 ISA reference gives the VOP3P
; half selection but not the 32-bit value of a constant. v1 = 0x00050003 (lo 3, hi 5), v2 = 0.
; Lane 0 of a wave32 group stores eight dwords:
;   out[0] = v_pk_add_u16(v1, 1)              1 = 0x00000001: lo 3 + 1 = 4, hi 5 + 0 = 5          = 0x00050004
;   out[1] = v_pk_add_u16(v1, -1)            -1 = 0xFFFFFFFF: lo 3 + 0xFFFF = 2, hi 5 + 0xFFFF = 4 = 0x00040002
;   out[2] = v_pk_add_u16(v2, 1.0)          1.0 = 0x3F800000: lo 0 + 0x0000, hi 0 + 0x3F80       = 0x3F800000
;   out[3] = v_pk_add_u16(v2, 0x12345678) op_sel:[0,1] op_sel_hi:[1,0]
;            lo takes the literal's high half 0x1234, hi its low half 0x5678                 = 0x56781234
;   out[4] = v_pk_mad_u16(v1, 0x00200010, 1.0)  lo 3 * 0x10 + 0x0000 = 0x30,
;            hi 5 * 0x20 + 0x3F80 = 0x4020                                                   = 0x40200030
;   out[5] = v_pk_sub_i16(0x00050007, v1)     lo 7 - 3 = 4, hi 5 - 5 = 0                     = 0x00000004
;   out[6] = v_pk_min_i16(v2, -16)          -16 = 0xFFFFFFF0: lo min(0, -16) = 0xFFF0,
;            hi min(0, -1) = 0xFFFF                                                          = 0xFFFFFFF0
;   out[7] = v_pk_mul_lo_u16(v1, -2.0)     -2.0 = 0xC0000000: lo 3 * 0 = 0,
;            hi 5 * 0xC000 = 0x3C000, low 16 bits 0xC000                                     = 0xC0000000
; Until this test the translator rejected all eight ("high half of a constant").
; group: 32,1,1
; Run: vk_compute_test cs_w32_pk_int_const.spv 1 dump -> out[0..7] = 327684 262146 1065353216 1450709556 1075839024 4 4294967280 3221225472
; mattr: +wavefrontsize32,-wavefrontsize64
v_mov_b32_e32 v1, 0x50003
v_mov_b32_e32 v2, 0
v_pk_add_u16 v10, v1, 1
v_pk_add_u16 v11, v1, -1
v_pk_add_u16 v12, v2, 1.0
v_pk_add_u16 v13, v2, 0x12345678 op_sel:[0,1] op_sel_hi:[1,0]
v_pk_mad_u16 v14, v1, 0x200010, 1.0
v_pk_sub_i16 v15, 0x50007, v1
v_pk_min_i16 v16, v2, -16
v_pk_mul_lo_u16 v17, v1, -2.0
v_cmpx_eq_u32_e32 0, v0
v_mov_b32_e32 v21, 0
buffer_store_dword v10, v21, s[0:3], 0 offen
v_mov_b32_e32 v21, 4
buffer_store_dword v11, v21, s[0:3], 0 offen
v_mov_b32_e32 v21, 8
buffer_store_dword v12, v21, s[0:3], 0 offen
v_mov_b32_e32 v21, 12
buffer_store_dword v13, v21, s[0:3], 0 offen
v_mov_b32_e32 v21, 16
buffer_store_dword v14, v21, s[0:3], 0 offen
v_mov_b32_e32 v21, 20
buffer_store_dword v15, v21, s[0:3], 0 offen
v_mov_b32_e32 v21, 24
buffer_store_dword v16, v21, s[0:3], 0 offen
v_mov_b32_e32 v21, 28
buffer_store_dword v17, v21, s[0:3], 0 offen
s_endpgm
