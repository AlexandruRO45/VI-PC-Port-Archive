; v_mul_f64 and v_fma_f64: D = S0 * S1 (+ S2) rounded once to binary64, to nearest even. Measured natively on RDNA2
; gfx1036 (tools/hw_oracle raw kernels): the four words below, a subnormal product kept (2^-1022 * 0.5 = 2^-1023,
; f64 denormals on), and VOP3 omod ignored while f64 denormals are kept (with IEEE on or off; it applies only with
; IEEE off and f64 denormals flushed), clamp applied. Until this change both raised Unsupported.
; Lane 0 of a wave32 group stores four doubles (low dword first):
;   out[0:1] v_mul_f64 1.5 * -2.0                                   = -3.0           = 0x00000000 0xc0080000
;   out[2:3] v_fma_f64 x * x + z, x = 1 + 2^-30, z = -(1 + 2^-29): the exact x * x = 1 + 2^-29 + 2^-60, so the
;            fused result is 2^-60 (biased exponent 1023 - 60 = 963 = 0x3c3); rounding the product first gives 0
;                                                                   = 2^-60          = 0x00000000 0x3c300000
;   out[4:5] v_mul_f64 (1 + 2^-52)^2 = 1 + 2^-51 + 2^-104, rounded  = 1 + 2^-51      = 0x00000002 0x3ff00000
;   out[6:7] v_mul_f64 1.5 * 1.5 mul:2: omod ignored                = 2.25           = 0x00000000 0x40020000
; group: 32,1,1
; Run: vk_compute_test cs_w32_f64_mul_fma.spv 1 dump -> out[0..7] = 0 3221749760 0 1009778688 2 1072693248 0 1073872896
; mattr: +wavefrontsize32,-wavefrontsize64
v_mov_b32_e32 v2, 0
v_mov_b32_e32 v3, 0x3ff80000
v_mov_b32_e32 v4, 0
v_mov_b32_e32 v5, 0xc0000000
v_mul_f64 v[10:11], v[2:3], v[4:5]
v_mov_b32_e32 v2, 0x400000
v_mov_b32_e32 v3, 0x3ff00000
v_mov_b32_e32 v6, 0x800000
v_mov_b32_e32 v7, 0xbff00000
v_fma_f64 v[12:13], v[2:3], v[2:3], v[6:7]
v_mov_b32_e32 v2, 1
v_mul_f64 v[14:15], v[2:3], v[2:3]
v_mov_b32_e32 v2, 0
v_mov_b32_e32 v3, 0x3ff80000
v_mul_f64 v[16:17], v[2:3], v[2:3] mul:2
v_cmpx_eq_u32_e32 0, v0
v_mov_b32_e32 v21, 0
buffer_store_dwordx4 v[10:13], v21, s[0:3], 0 offen
v_mov_b32_e32 v21, 16
buffer_store_dwordx4 v[14:17], v21, s[0:3], 0 offen
s_endpgm
