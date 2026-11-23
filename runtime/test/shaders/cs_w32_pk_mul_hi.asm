; Packed f16 multiply (RDNA2 ISA VOP3P v_pk_mul_f16) reading a high half that a 16-bit op just wrote through op_sel
; (op_sel[3], the destination high half) from a lane-derived source, the register written again afterwards.
; D.lo = src0.half(op_sel[0]) * src1.half(op_sel[1]), D.hi = src0.half(op_sel_hi[0]) * src1.half(op_sel_hi[1]), each
; rounded to f16. The AMD Windows driver 26.10.1 (RX 9060 XT, RDNA4) gave wrong halves in wave32 for every multiply here
; while the packed-f16 sources were widened with UnpackHalf2x16 (0xBCB64000 0xF8B64000 0xC0064000 0xBCB64000 0x3180BC08
; 0xBCB64000 for out[0..5]; SHADER-777 follow-up); the translator now widens them with integer ops (Gen.f16_widen). out[4]
; is the v_pk_add_f16 control, which was right before as well.
;   A = 0x00003C00 | (lane & 7) << 7 (lane 0: 0x3C00)   K = 0xBE553CB0 (low half 0x3CB0)   M = 0x3C00BC00 (lo -1.0, hi 1.0)
;   each case starts its register at R = 0xC150A000 (low half 0xA000 = -2^-7), writes R.hi from A.lo and K.lo, then:
;   v_pk_mul_f16 d, M, R op_sel:[0,0] op_sel_hi:[0,1]: d.lo = -1.0 * -2^-7 = 2^-7 = 0x2000, d.hi = -1.0 * R.hi
;   out[0] v_max_i16    R.hi = max(0x3C00, 0x3CB0) = 0x3CB0 -> d.hi = -1.171875 = 0xBCB0, R written later  = 0xBCB02000
;   out[1] v_add_nc_u16 R.hi = 0x3C00 + 0x3CB0 = 0x78B0 (38400.0) -> d.hi = 0xF8B0, R written later       = 0xF8B02000
;   out[2] v_mul_lo_u16 R.hi = 0x3C00 * 0x3CB0 mod 2^16 = 0x4000 (2.0) -> d.hi = 0xC000, R written later  = 0xC0002000
;   out[3] v_max_u16    R.hi = 0x3CB0 -> d.hi = 0xBCB0, R written later                                    = 0xBCB02000
;   out[4] v_max_i16, then v_pk_add_f16 d, M, R op_sel:[0,0] op_sel_hi:[0,1]: d.lo = -1.0 + -2^-7 = 0xBC08,
;          d.hi = -1.0 + 1.171875 = 0.171875 = 0x3180, R written later                                    = 0x3180BC08
;   out[5] v_max_i16 -> v_pk_mul_f16, its own register not written again (in this module it failed as well) = 0xBCB02000
;   out[6] A as lane 0 holds it                                                                            = 0x00003C00
;   out[7] out[0]'s register after the later write                                                          = 0x00000000
; group: 32,1,1
; Run: vk_compute_test cs_w32_pk_mul_hi.spv 1 dump
;   -> out[0..7] = 0xBCB02000 0xF8B02000 0xC0002000 0xBCB02000 0x3180BC08 0xBCB02000 0x00003C00 0x00000000
; mattr: +wavefrontsize32,-wavefrontsize64
v_and_b32_e32 v1, 7, v0
v_lshlrev_b32_e32 v1, 7, v1
v_or_b32_e32 v2, 0x00003c00, v1
v_mov_b32_e32 v3, 0xbe553cb0
v_mov_b32_e32 v4, 0x3c00bc00
v_mov_b32_e32 v20, 0xc150a000
v_max_i16_e64 v20, v2, v3 op_sel:[0,0,1]
v_pk_mul_f16 v10, v4, v20 op_sel:[0,0] op_sel_hi:[0,1]
v_mov_b32_e32 v20, 0
v_mov_b32_e32 v21, 0xc150a000
v_add_nc_u16_e64 v21, v2, v3 op_sel:[0,0,1]
v_pk_mul_f16 v11, v4, v21 op_sel:[0,0] op_sel_hi:[0,1]
v_mov_b32_e32 v21, 0
v_mov_b32_e32 v22, 0xc150a000
v_mul_lo_u16_e64 v22, v2, v3 op_sel:[0,0,1]
v_pk_mul_f16 v12, v4, v22 op_sel:[0,0] op_sel_hi:[0,1]
v_mov_b32_e32 v22, 0
v_mov_b32_e32 v23, 0xc150a000
v_max_u16_e64 v23, v2, v3 op_sel:[0,0,1]
v_pk_mul_f16 v13, v4, v23 op_sel:[0,0] op_sel_hi:[0,1]
v_mov_b32_e32 v23, 0
v_mov_b32_e32 v24, 0xc150a000
v_max_i16_e64 v24, v2, v3 op_sel:[0,0,1]
v_pk_add_f16 v14, v4, v24 op_sel:[0,0] op_sel_hi:[0,1]
v_mov_b32_e32 v24, 0
v_mov_b32_e32 v25, 0xc150a000
v_max_i16_e64 v25, v2, v3 op_sel:[0,0,1]
v_pk_mul_f16 v15, v4, v25 op_sel:[0,0] op_sel_hi:[0,1]
v_and_b32_e32 v6, 31, v0
v_lshlrev_b32_e32 v6, 5, v6
buffer_store_dword v10, v6, s[0:3], 0 offen offset:0
buffer_store_dword v11, v6, s[0:3], 0 offen offset:4
buffer_store_dword v12, v6, s[0:3], 0 offen offset:8
buffer_store_dword v13, v6, s[0:3], 0 offen offset:12
buffer_store_dword v14, v6, s[0:3], 0 offen offset:16
buffer_store_dword v15, v6, s[0:3], 0 offen offset:20
buffer_store_dword v2, v6, s[0:3], 0 offen offset:24
buffer_store_dword v20, v6, s[0:3], 0 offen offset:28
s_endpgm
