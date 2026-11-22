; v_log_f16 where SPIR-V's Log2 is undefined (an operand <= 0): GPU-only, the CPU interpreter refuses these inputs.
; The translator leaves them to the driver; the hardware's words (docs/design.md, f16 transcendentals row) are
; log(+0) = log(-0) = -inf and log of a negative number (-inf included) = NaN. A NaN's payload is the driver's,
; so a NaN is stored as a flag (v_cmp_u_f16). Only the low half is read; the high half carries a valid 1.0.
; f16 bits used: -inf fc00, +0 0000, -0 8000, -1.0 bc00, -0.5 b800.
;   out[0] = v_pack_b32_f16 (log(+0.0) = -inf fc00, log(-0.0) = -inf fc00)                = 0xfc00fc00
;   out[1] = flags: bit 0 log(-1.0) NaN, bit 1 log(-inf) NaN, bit 2 log(-0.5) NaN         = 7
; group: 32,1,1
; Run: vk_compute_test cs_w32_f16_log_undef.spv 1 dump -> out[0..7] = 4227922944 7 4008636142 4008636142 4008636142 4008636142 4008636142 4008636142
; mattr: +wavefrontsize32,-wavefrontsize64
v_cmpx_eq_u32_e32 0, v0
v_mov_b32_e32 v1, 0x3c000000
v_mov_b32_e32 v2, 0x3c008000
v_log_f16_e32 v3, v1
v_log_f16_e32 v4, v2
v_pack_b32_f16 v3, v3, v4
v_mov_b32_e32 v5, 0x3c00bc00
v_log_f16_e32 v5, v5
v_cmp_u_f16_e32 vcc_lo, v5, v5
v_cndmask_b32_e64 v10, 0, 1, vcc_lo
v_mov_b32_e32 v5, 0x3c00fc00
v_log_f16_e32 v5, v5
v_cmp_u_f16_e32 vcc_lo, v5, v5
v_cndmask_b32_e64 v11, 0, 2, vcc_lo
v_or_b32_e32 v10, v10, v11
v_mov_b32_e32 v5, 0x3c00b800
v_log_f16_e32 v5, v5
v_cmp_u_f16_e32 vcc_lo, v5, v5
v_cndmask_b32_e64 v11, 0, 4, vcc_lo
v_or_b32_e32 v10, v10, v11
v_mov_b32_e32 v25, 0
buffer_store_dword v3, v25, s[0:3], 0 offen
v_mov_b32_e32 v25, 4
buffer_store_dword v10, v25, s[0:3], 0 offen
s_endpgm
