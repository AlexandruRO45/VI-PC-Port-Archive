; A compressed export whose registers come from v_cvt_pknorm_i16_f32, drawn into an RGBA16_SNORM target
; (EGEN-PS-MRT-TYPES). RDNA2 picks the SNORM16_ABGR export format for that target and the halves reach it bit for bit
; (tools/ps_oracle.py --format rgba16snorm on gfx1036). The translator decodes them as snorm16 (max(s / 32767, -1)),
; which the target converts back to the same s; until this test it decoded them as f16.
;   v2 = pknorm_i16(-1.0, 0.5): lo = -32767 = 0x8001, hi = 0.5 * 32767 = 16383.5 -> 16384 (nearest even) = 0x4000
;   v3 = pknorm_i16(0.25, -0.5): lo = 8191.75 -> 8192 = 0x2000, hi = -16383.5 -> -16384 = 0xc000
; The target's texel (R, G, B, A = 0x8001, 0x4000, 0x2000, 0xc000) as two dwords: 0x40008001, 0xc0002000
; Expected texel words: 0x40008001, 0xc0002000
v_mov_b32 v4, -1.0
v_mov_b32 v5, 0.5
v_cvt_pknorm_i16_f32 v2, v4, v5
v_cvt_pknorm_i16_f32 v3, 0.25, -0.5
exp mrt0 v2, v2, v3, v3 done compr vm
s_endpgm
