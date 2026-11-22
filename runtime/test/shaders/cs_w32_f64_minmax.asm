; v_min_f64 / v_max_f64: one source's 64 bits, as measured natively on RDNA2 gfx1036 (tools/hw_oracle raw kernels,
; MODE.IEEE off, the MODE inferred for the game): a NaN operand (quiet or signalling) gives the other operand, two NaNs
; give src0 with its bits, -0 < +0, subnormals order by value (f64 denormals kept); the omod is ignored. Until this
; change both raised Unsupported. The translator selects the words with integer ops, so no driver NaN or denormal
; handling enters.
; Lane 0 of a wave32 group stores four doubles (low dword first):
;   out[0:1] v_min_f64 qNaN 0x7ff8000000000001, 3.0       -> 3.0                      = 0x00000000 0x40080000
;   out[2:3] v_min_f64 +0, -0                             -> -0                       = 0x00000000 0x80000000
;   out[4:5] v_max_f64 2^-1074 (subnormal), -0, mul:2 (ignored) -> 2^-1074           = 0x00000001 0x00000000
;   out[6:7] v_min_f64 sNaN 0x7ff0000000000001, qNaN      -> src0, the sNaN unchanged = 0x00000001 0x7ff00000
; group: 32,1,1
; Run: vk_compute_test cs_w32_f64_minmax.spv 1 dump -> out[0..7] = 0 1074266112 0 2147483648 1 0 1 2146435072
; mattr: +wavefrontsize32,-wavefrontsize64
v_mov_b32_e32 v2, 1
v_mov_b32_e32 v3, 0x7ff80000
v_mov_b32_e32 v4, 0
v_mov_b32_e32 v5, 0x40080000
v_min_f64 v[10:11], v[2:3], v[4:5]
v_mov_b32_e32 v2, 0
v_mov_b32_e32 v3, 0
v_mov_b32_e32 v4, 0
v_mov_b32_e32 v5, 0x80000000
v_min_f64 v[12:13], v[2:3], v[4:5]
v_mov_b32_e32 v2, 1
v_max_f64 v[14:15], v[2:3], v[4:5] mul:2
v_mov_b32_e32 v3, 0x7ff00000
v_mov_b32_e32 v4, 1
v_mov_b32_e32 v5, 0x7ff80000
v_min_f64 v[16:17], v[2:3], v[4:5]
v_cmpx_eq_u32_e32 0, v0
v_mov_b32_e32 v21, 0
buffer_store_dwordx4 v[10:13], v21, s[0:3], 0 offen
v_mov_b32_e32 v21, 16
buffer_store_dwordx4 v[14:17], v21, s[0:3], 0 offen
s_endpgm
