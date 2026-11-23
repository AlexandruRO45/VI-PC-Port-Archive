; v_mul_legacy_f32 in the inferred game MODE (IEEE=0, denorm_mode_32=0). Thread 0 multiplies one pair per word:
;   0  2^-149 * 2^24          a denormal operand reads as +0, so the product is +0                    0x00000000
;   1 -2^-149 * 2^24          the DX9 zero-operand override drops the operand's sign, still +0        0x00000000
;   2 -2^-149 * -NaN          the override wins over a NaN operand                                    0x00000000
;   3  +0 * +inf              the override wins over infinity, so +0 and not NaN                      0x00000000
;   4  FLT_MIN * 0.5 (2^-127) a nonzero subnormal product flushes to a zero that keeps its sign, +0    0x00000000
;   5 -FLT_MIN * 0.5          the same product with the other sign                                    0x80000000
;   6  FLT_MIN * 0.5 mul:2    the flush is before the omod, so mul:2 cannot scale it back to FLT_MIN  0x00000000
;   7 -FLT_MIN * 0.5 div:2    the omod turns the flushed -0 into +0 with IEEE off                     0x00000000
; Word 6 is the sharp one: the product is 2^-127, which mul:2 would make normal (2^-126, 0x00800000) if the
; flush came after the modifier instead of before it.
; One thread, no 64-wide subgroup requirement.
; Run: vk_compute_test cs_mul_legacy_denorm.spv 1 dump
; out[0..7] = 00000000 00000000 00000000 00000000 00000000 80000000 00000000 00000000 (hex)
v_cmpx_eq_u32_e32 0, v0
v_mov_b32 v1, 0x00000001
v_mov_b32 v2, 0x4b800000
v_mul_legacy_f32 v3, v1, v2
buffer_store_dword v3, off, s[0:3], 0
v_mov_b32 v1, 0x80000001
v_mul_legacy_f32 v3, v1, v2
buffer_store_dword v3, off, s[0:3], 0 offset:4
v_mov_b32 v2, 0xffc00000
v_mul_legacy_f32 v3, v1, v2
buffer_store_dword v3, off, s[0:3], 0 offset:8
v_mov_b32 v1, 0x00000000
v_mov_b32 v2, 0x7f800000
v_mul_legacy_f32 v3, v1, v2
buffer_store_dword v3, off, s[0:3], 0 offset:12
v_mov_b32 v1, 0x00800000
v_mov_b32 v2, 0x3f000000
v_mul_legacy_f32 v3, v1, v2
buffer_store_dword v3, off, s[0:3], 0 offset:16
v_mov_b32 v1, 0x80800000
v_mul_legacy_f32 v3, v1, v2
buffer_store_dword v3, off, s[0:3], 0 offset:20
v_mov_b32 v1, 0x00800000
v_mul_legacy_f32_e64 v3, v1, v2 mul:2
buffer_store_dword v3, off, s[0:3], 0 offset:24
v_mov_b32 v1, 0x80800000
v_mul_legacy_f32_e64 v3, v1, v2 div:2
buffer_store_dword v3, off, s[0:3], 0 offset:28
s_endpgm
