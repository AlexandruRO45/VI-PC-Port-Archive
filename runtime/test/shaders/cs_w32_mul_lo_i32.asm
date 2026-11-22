; v_mul_lo_i32 (VOP3 opcode 0x16b): D = the low 32 bits of S0 * S1 as signed 32-bit integers. The opcode exists on GFX10.1
; only (gfx1010-gfx1013, the console's gfx1013 included); LLVM's gfx103x tables drop it and keep v_mul_lo_u32, so it can't be
; measured on the RDNA2 gfx1036 oracle. The low 32 bits of a product are the same for signed and unsigned operands
; (two's complement), so it is v_mul_lo_u32. Until this change it raised Unsupported.
; Lane 0 of a wave32 group stores:
;   out[0] 3 * -5                    = -15                                       = 0xfffffff1 = 4294967281
;   out[1] 0x80000000 * -1           = 2^31 (wraps to INT_MIN)                   = 0x80000000 = 2147483648
;   out[2] 0x7fffffff * 0x7fffffff   = 2^62 - 2^32 + 1, low dword                = 1
;   out[3] -7 * -9                   = 63                                        = 63
;   out[4] 0x12345678 * 0x9abcdef0   = 0x0b00ea4e_242d2080, low dword            = 0x242d2080 = 606937216
;   out[5] s4 (0xfffe0001 = -131071) * literal 0x10001 (65537), low dword:
;          = (2^32 - 2^17 + 1)(2^16 + 1) mod 2^32 = 2^16 + 1 - 2^17 = -65535      = 0xffff0001 = 4294901761
;   out[6] inline -1 * 0x5a5a5a5a    = -0x5a5a5a5a                               = 0xa5a5a5a6 = 2779096486
;   out[7] 0x40000000 * inline 4     = 2^32, low dword                           = 0
; group: 32,1,1
; Run: vk_compute_test cs_w32_mul_lo_i32.spv 1 dump -> out[0..7] = 4294967281 2147483648 1 63 606937216 4294901761 2779096486 0
; mattr: +wavefrontsize32,-wavefrontsize64
v_mov_b32_e32 v1, 3
v_mov_b32_e32 v2, -5
v_mul_lo_i32 v10, v1, v2
v_mov_b32_e32 v1, 0x80000000
v_mov_b32_e32 v2, -1
v_mul_lo_i32 v11, v1, v2
v_mov_b32_e32 v1, 0x7fffffff
v_mul_lo_i32 v12, v1, v1
v_mov_b32_e32 v1, -7
v_mov_b32_e32 v2, -9
v_mul_lo_i32 v13, v1, v2
v_mov_b32_e32 v1, 0x12345678
v_mov_b32_e32 v2, 0x9abcdef0
v_mul_lo_i32 v14, v1, v2
s_mov_b32 s4, 0xfffe0001
v_mul_lo_i32 v15, s4, 0x10001
v_mov_b32_e32 v1, 0x5a5a5a5a
v_mul_lo_i32 v16, -1, v1
v_mov_b32_e32 v1, 0x40000000
v_mul_lo_i32 v17, v1, 4
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
