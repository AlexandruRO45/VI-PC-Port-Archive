; A compressed export whose registers come from v_cvt_pknorm_u16_f32, drawn into an RGBA16_UNORM target
; (EGEN-PS-MRT-TYPES). RDNA2 picks the UNORM16_ABGR export format for that target and the halves reach it bit for bit
; (tools/ps_oracle.py --format rgba16unorm on gfx1036: 0x80017fff, 0xc0003c01 came back as those words). The
; translator decodes them as unorm16 (u / 65535), which the target converts back to the same u; until this test it
; decoded them as f16 (0xffff is an f16 NaN, 0x8000 is -0.0).
;   v2 = pknorm_u16(1.0, 0.5): lo = 1.0 * 65535 = 0xffff, hi = 0.5 * 65535 = 32767.5 -> 32768 (nearest even) = 0x8000
;   v3 = pknorm_u16(0, 0.25):  lo = 0, hi = 0.25 * 65535 = 16383.75 -> 16384 = 0x4000
; The target's texel (R, G, B, A = 0xffff, 0x8000, 0, 0x4000) as two dwords: 0x8000ffff, 0x40000000
; Expected texel words: 0x8000ffff, 0x40000000
v_mov_b32 v4, 1.0
v_mov_b32 v5, 0.5
v_cvt_pknorm_u16_f32 v2, v4, v5
v_cvt_pknorm_u16_f32 v3, 0, 0.25
exp mrt0 v2, v2, v3, v3 done compr vm
s_endpgm
