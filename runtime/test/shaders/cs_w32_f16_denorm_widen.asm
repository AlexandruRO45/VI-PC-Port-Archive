; regen_fixtures: this source has a translator fixture in tests/semantic/fixtures/disasm
; f16 denormal operands on the ops that widen with UnpackHalf2x16 directly, outside Gen.f16_in: v_cvt_f32_f16,
; v_pk_fma_f16 and v_pk_fmac_f16. RDNA2's f16 MODE keeps denormals, so a denormal operand counts at its value
; (m * 2^-24). Gen.f16_in widens exactly since 9d71037c, because NVIDIA read an SDWA-selected denormal as 0; these
; three ops did not go through it, and no other fixture feeds them a denormal.
; Lane 0 of a wave32 group stores eight dwords (all lanes compute the same words):
;   out[0] = v_cvt_f32_f16(0x0001, VGPR)                       = 0x33800000  (2^-24, the smallest denormal)
;   out[1] = v_cvt_f32_f16(0x83ff, VGPR)                       = 0xb87fc000  (-1023 * 2^-24, the largest negative one)
;   out[2] = v_cvt_f32_f16(literal 0x0200)                     = 0x38000000  (512 * 2^-24 = 2^-15)
;   out[3] = v_cvt_f32_f16_sdwa(0x0040 in WORD_1)              = 0x36800000  (64 * 2^-24 = 2^-18: the SDWA shape of #3220)
;   out[4] = v_pk_fma_f16((0x8001, 0x0040), (1.0, 1.0), 0)     = 0x80010040  (denormal multiplicands kept, -0x0001 hi)
;   out[5] = v_pk_fma_f16(0, (1.0, 1.0), (0x0200, 0x0001))     = 0x02000001  (denormal addends kept)
;   out[6] = v_pk_fmac_f16((1, 1) * (1.0, 1.0) + (3, 2) denorm) = 0x00040003  (2^-24 + 2 * 2^-24 = 3 * 2^-24, hi 4 * 2^-24)
;   out[7] = v_cvt_f32_f16(0x3c00, VGPR)                       = 0x3f800000  (control: a normal 1.0 widens on every path)
; A device that flushes f16 denormals on input gives 0x00000000 / 0x80000000 in out[0..3] and 0 halves in out[4..6].
; group: 32,1,1
; Run: vk_compute_test cs_w32_f16_denorm_widen.spv 1 dump -> out[0..7] = 864026624 3095379968 939524096 914358272 2147549248 33554433 262147 1065353216
; mattr: +wavefrontsize32,-wavefrontsize64
v_mov_b32_e32 v1, 0x0001
v_cvt_f32_f16_e32 v10, v1
v_mov_b32_e32 v1, 0x83ff
v_cvt_f32_f16_e32 v11, v1
v_cvt_f32_f16_e32 v12, 0x200
v_mov_b32_e32 v2, 0x00400000
v_cvt_f32_f16_sdwa v13, v2 dst_sel:DWORD dst_unused:UNUSED_PRESERVE src0_sel:WORD_1
v_mov_b32_e32 v3, 0x80010040
v_mov_b32_e32 v4, 0x3c003c00
v_pk_fma_f16 v14, v3, v4, 0
v_mov_b32_e32 v5, 0x02000001
v_pk_fma_f16 v15, 0, v4, v5
v_mov_b32_e32 v6, 0x00030002
v_mov_b32_e32 v7, 0x00010001
v_pk_fmac_f16 v6, v7, v4
v_mov_b32_e32 v1, 0x3c00
v_cvt_f32_f16_e32 v17, v1
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
buffer_store_dword v6, v25, s[0:3], 0 offen
v_mov_b32_e32 v25, 28
buffer_store_dword v17, v25, s[0:3], 0 offen
s_endpgm
