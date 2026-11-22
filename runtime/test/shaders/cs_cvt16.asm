; 16-bit and ubyte conversions on the GPU (EGEN-GPU-CVT16). Thread 0 writes 24 words, out[0..23].
; Each 16-bit result goes to a destination preset to 0xaaaa5555: only D[15:0] is written, so D[31:16] stays 0xaaaa
; (docs/design.md, 16-bit results; the _e64 forms keep the high half). Each source has junk in its high half
; (0x1234, 0x5555 ...): the conversions read S0[15:0] only.
; RDNA 2 ISA: v_cvt_f16_i16 / v_cvt_f16_u16 convert the sign-extended / zero-extended low half to binary16, round to
; nearest even (65520 and above give +inf). v_cvt_i16_f16 / v_cvt_u16_f16 truncate toward zero, saturate to
; [-32768, 32767] / [0, 65535] and give 0 for NaN. v_cvt_f32_ubyteN: float(S0[8N+7:8N]), unsigned.
; binary16 = sign | (exp+15)<<10 | mantissa; above 2^11 the spacing is 2^(e-10).
;   out[0]  i16  2049 = 0x801: 2049 is the tie between 2048 (mantissa 0x000) and 2050 (0x001); even -> 2048 = 0x6800
;   out[1]  i16  2051 = 0x803: tie between 2050 (0x001) and 2052 (0x002); even -> 2052 = 0x6802
;   out[2]  i16  0xf7ff = -2049 -> -2048 = 0xe800
;   out[3]  i16  0x8001 = -32767: spacing 16 above 16384, nearest multiple is 32768 -> -32768 = 0xf800
;           (a zero-extending translator sees +32769, which also rounds to 32768, but gives +: 0x7800)
;   out[4]  i16  0xfffb = -5 = -1.25 * 2^2 -> (2+15)<<10 | 0x100 | sign = 0xc500
;   out[5]  u16  2049 -> 0x6800
;   out[6]  u16  0xf801 = 63489: spacing 32 above 32768, nearest multiple 63488 = 1984*32 = 2^15 * (1 + 960/1024)
;           -> (15+15)<<10 | 0x3c0 = 0x7bc0
;   out[7]  u16  0xffff = 65535 >= 65520 -> +inf = 0x7c00
;   out[8]  u16  0x8001 = 32769 -> 32768 = 0x7800 (an i16 reading would give 0xf800)
;   out[9]  i16_f16  0xc180 = -1.375 * 2^1 = -2.75 -> -2 = 0xfffe (toward zero, not -3)
;   out[10] i16_f16  0x78e2 = 2^15 * (1 + 226/1024) = 40000 -> saturates to 0x7fff
;   out[11] i16_f16  0xf8e2 = -40000 -> 0x8000
;   out[12] i16_f16  +inf 0x7c00 -> 0x7fff
;   out[13] i16_f16  qNaN 0x7e00 -> 0
;   out[14] i16_f16  0xb800 = -0.5 -> 0 (toward zero)
;   out[15] u16_f16  0x4180 = 2.75 -> 2
;   out[16] u16_f16  0xbe00 = -1.5 -> 0 (saturates at 0)
;   out[17] u16_f16  0x7bff = 65504 -> 0xffe0
;   out[18] u16_f16  +inf -> 0xffff
;   out[19] u16_f16  -qNaN 0xfe00 -> 0
; out[20..23]: v_cvt_f32_ubyte0..3 of 0x89abcdef (bytes 0xef 0xcd 0xab 0x89 = 239 205 171 137, all >= 128 so a
; signed read would be negative). n in [128, 255] is 0x43000000 | (n - 128) << 16: 239 -> 0x436f0000,
; 205 -> 0x434d0000, 171 -> 0x432b0000, 137 -> 0x43090000.
; Run: vk_compute_test cs_cvt16.spv 1 dump -> out[0..23] =
;   aaaa6800 aaaa6802 aaaae800 aaaaf800 aaaac500 aaaa6800 aaaa7bc0 aaaa7c00 aaaa7800 aaaafffe
;   aaaa7fff aaaa8000 aaaa7fff aaaa0000 aaaa0000 aaaa0002 aaaa0000 aaaaffe0 aaaaffff aaaa0000
;   436f0000 434d0000 432b0000 43090000
v_cmpx_eq_u32_e32 0, v0
v_mov_b32 v1, 0x12340801
v_mov_b32 v2, 0xaaaa5555
v_cvt_f16_i16_e64 v2, v1
buffer_store_dword v2, off, s[0:3], 0 offset:0
v_mov_b32 v1, 0x12340803
v_mov_b32 v2, 0xaaaa5555
v_cvt_f16_i16_e64 v2, v1
buffer_store_dword v2, off, s[0:3], 0 offset:4
v_mov_b32 v1, 0x1234f7ff
v_mov_b32 v2, 0xaaaa5555
v_cvt_f16_i16_e64 v2, v1
buffer_store_dword v2, off, s[0:3], 0 offset:8
v_mov_b32 v1, 0x7fff8001
v_mov_b32 v2, 0xaaaa5555
v_cvt_f16_i16_e64 v2, v1
buffer_store_dword v2, off, s[0:3], 0 offset:12
v_mov_b32 v1, 0xabcdfffb
v_mov_b32 v2, 0xaaaa5555
v_cvt_f16_i16_e64 v2, v1
buffer_store_dword v2, off, s[0:3], 0 offset:16
v_mov_b32 v1, 0x12340801
v_mov_b32 v2, 0xaaaa5555
v_cvt_f16_u16_e64 v2, v1
buffer_store_dword v2, off, s[0:3], 0 offset:20
v_mov_b32 v1, 0x1234f801
v_mov_b32 v2, 0xaaaa5555
v_cvt_f16_u16_e64 v2, v1
buffer_store_dword v2, off, s[0:3], 0 offset:24
v_mov_b32 v1, 0x1234ffff
v_mov_b32 v2, 0xaaaa5555
v_cvt_f16_u16_e64 v2, v1
buffer_store_dword v2, off, s[0:3], 0 offset:28
v_mov_b32 v1, 0x12348001
v_mov_b32 v2, 0xaaaa5555
v_cvt_f16_u16_e64 v2, v1
buffer_store_dword v2, off, s[0:3], 0 offset:32
v_mov_b32 v1, 0x5555c180
v_mov_b32 v2, 0xaaaa5555
v_cvt_i16_f16_e64 v2, v1
buffer_store_dword v2, off, s[0:3], 0 offset:36
v_mov_b32 v1, 0x555578e2
v_mov_b32 v2, 0xaaaa5555
v_cvt_i16_f16_e64 v2, v1
buffer_store_dword v2, off, s[0:3], 0 offset:40
v_mov_b32 v1, 0x5555f8e2
v_mov_b32 v2, 0xaaaa5555
v_cvt_i16_f16_e64 v2, v1
buffer_store_dword v2, off, s[0:3], 0 offset:44
v_mov_b32 v1, 0x55557c00
v_mov_b32 v2, 0xaaaa5555
v_cvt_i16_f16_e64 v2, v1
buffer_store_dword v2, off, s[0:3], 0 offset:48
v_mov_b32 v1, 0x55557e00
v_mov_b32 v2, 0xaaaa5555
v_cvt_i16_f16_e64 v2, v1
buffer_store_dword v2, off, s[0:3], 0 offset:52
v_mov_b32 v1, 0x5555b800
v_mov_b32 v2, 0xaaaa5555
v_cvt_i16_f16_e64 v2, v1
buffer_store_dword v2, off, s[0:3], 0 offset:56
v_mov_b32 v1, 0x55554180
v_mov_b32 v2, 0xaaaa5555
v_cvt_u16_f16_e64 v2, v1
buffer_store_dword v2, off, s[0:3], 0 offset:60
v_mov_b32 v1, 0x5555be00
v_mov_b32 v2, 0xaaaa5555
v_cvt_u16_f16_e64 v2, v1
buffer_store_dword v2, off, s[0:3], 0 offset:64
v_mov_b32 v1, 0x55557bff
v_mov_b32 v2, 0xaaaa5555
v_cvt_u16_f16_e64 v2, v1
buffer_store_dword v2, off, s[0:3], 0 offset:68
v_mov_b32 v1, 0x55557c00
v_mov_b32 v2, 0xaaaa5555
v_cvt_u16_f16_e64 v2, v1
buffer_store_dword v2, off, s[0:3], 0 offset:72
v_mov_b32 v1, 0x5555fe00
v_mov_b32 v2, 0xaaaa5555
v_cvt_u16_f16_e64 v2, v1
buffer_store_dword v2, off, s[0:3], 0 offset:76
v_mov_b32 v1, 0x89abcdef
v_cvt_f32_ubyte0_e32 v2, v1
buffer_store_dword v2, off, s[0:3], 0 offset:80
v_cvt_f32_ubyte1_e32 v2, v1
buffer_store_dword v2, off, s[0:3], 0 offset:84
v_cvt_f32_ubyte2_e32 v2, v1
buffer_store_dword v2, off, s[0:3], 0 offset:88
v_cvt_f32_ubyte3_e32 v2, v1
buffer_store_dword v2, off, s[0:3], 0 offset:92
s_waitcnt vmcnt(0)
s_endpgm
