; v_frexp_mant_f16, v_frexp_exp_i16_f16 and v_ldexp_f16 on every binary16 input, f16 denormals kept (docs/design.md,
; the row for these three opcodes: measured natively on RDNA2 gfx1036 in #1931). No fuzzer generates these opcodes.
; s[0:3] is the output V#, s4 the ldexp exponent S1 (only its low 16 bits count), s5 TGID_X (RSRC2 0x808a: 5 user
; SGPRs, TGID_X; group 64x1x1), thread id in v0. Thread g = s5*64 + v0 takes the f16 input x = g, so 1024 groups
; cover all 65536 encodings. Each destination starts as 0x5a5a5a5a; the _e32 forms write the low half and keep the
; high half. out[4g+0..3] = frexp_mant(x), frexp_exp(x), ldexp(x, s4), each with 0x5a5a in the high half, then x
; itself (high half 0).
; Words measured on gfx1036 (the low halves):
;   x = 0x0001: frexp_mant 0x3800 (0.5), frexp_exp 0xffe9 (-23)    x = 0x03ff: 0x3bfe, 0xfff2 (-14)
;   ldexp(0x3c00 1.0, -15) = 0x0200   ldexp(0x3e00 1.5, -24) = 0x0002   ldexp(0x3c00, -25) = 0x0000
;   ldexp(0x7bff 65504, -30) = 0x0400 (rounds up to the smallest normal)
; check.py's f16_frexp_ldexp_test runs it once per exponent of F16_FREXP_LDEXP_EXPONENTS and compares all 262144
; words with the model next to it. A translator that reads f16 denormals as zero gets 2046 frexp words wrong.
; Run: VKC_SGPRS=<exp> VKC_OUT_WORDS=262144 VKC_DUMP=<file> vk_compute_test cs_f16_frexp_ldexp.spv 1024 dump
; group: 64,1,1
; rsrc2: 0x808a
v_lshl_add_u32 v6, s5, 6, v0
v_lshlrev_b32_e32 v2, 4, v6
v_mov_b32_e32 v3, 0x5a5a5a5a
v_mov_b32_e32 v4, 0x5a5a5a5a
v_mov_b32_e32 v5, 0x5a5a5a5a
v_mov_b32_e32 v7, s4
v_frexp_mant_f16_e32 v3, v6
v_frexp_exp_i16_f16_e32 v4, v6
v_ldexp_f16_e32 v5, v6, v7
buffer_store_dwordx4 v[3:6], v2, s[0:3], 0 offen
s_endpgm
