; Relative SGPR writes through M0: s_movreld_b32, s_movreld_b64 and s_movrelsd_2_b32 (the reads, s_movrels_b32/b64, are
; in docs/design.md already). RDNA2 ISA SOP1:
;   S_MOVRELD_B32:   SGPR[D + M0] = S0         (S0 any scalar source: an SGPR, an inline constant or a literal)
;   S_MOVRELD_B64:   SGPR[D + M0] = S0.lo, SGPR[D + M0 + 1] = S0.hi   (both halves read before either is written)
;   S_MOVRELSD_2_B32: SGPR[D + M0[25:16]] = SGPR[S0 + M0[9:0]]        (GFX10; the other M0 bits are ignored)
; None of them writes SCC or M0. The same facts were measured on an AMD Radeon gfx1036 with tools/int_hw_probe.py
; (--group salu_movreld, wave32 and wave64), which gives the same register values as the ISA text.
; Every case starts from the same fill: s10..s19 = 0x40000000 | (n << 20) | n * 0x111 (F10..F19, n = the register number),
; s4 = 0x4afe0004, s5 = 0x4afe0005, and runs one op. A check is "s_cmp_eq_u32 register, expected", the result goes into
; one bit of the word's accumulator (SCC checks read SCC straight after the op). The word is 0x40000000 | the bits of
; its checks, so with every check true it is 0x40000000 | (2^N - 1) for N checks; a wrong register shows as a missing bit.
; Word 0, s_movreld_b32 (22 checks), each case after a fresh fill:
;   m0 = 2, s_movreld_b32 s10, s4, run with SCC = 0 and again with SCC = 1: SCC kept (1 check each), s12 = s4 (1), and
;     s10, s11, s13, s14 keep F10, F11, F13, F14 (4)                                                -> 2 x 6 = 12
;   m0 = 0, s10, s4: s10 = s4 (1), s11 and s12 unchanged (2)                                        -> 3
;   m0 = 3, s_movreld_b32 s10, 0x4a123456 (literal): s13 = the literal (1), s10, s12, s14 unchanged (3)   -> 4
;   m0 = 1, s_movreld_b32 s11, s12: SGPR[12] = s12, so s12 stays F12 (1), s11 and s13 unchanged (2)        -> 3
;   total 12 + 3 + 4 + 3 = 22 checks: word 0 = 0x40000000 | 0x3fffff = 0x403fffff
; Word 1, s_movreld_b64 (20 checks):
;   m0 = 2, s_movreld_b64 s[10:11], s[4:5], SCC = 0 and SCC = 1: SCC kept (1), s12 = s4 (1), s13 = s5 (1), s10, s11,
;     s14 unchanged (3)                                                                              -> 2 x 6 = 12
;   m0 = 1, s[10:11], s[4:5]: s11 = s4 (1), s12 = s5 (1), s10 and s13 unchanged (2)                  -> 4
;   m0 = 1, s_movreld_b64 s[10:11], s[10:11]: both halves read first, so s11 = F10 (1), s12 = F11 (1), s10 and s13
;     unchanged (2)                                                                                  -> 4
;   total 12 + 4 + 4 = 20 checks: word 1 = 0x40000000 | 0xfffff = 0x400fffff
; Word 2, s_movrelsd_2_b32 (28 checks):
;   m0 = 0x30002, s_movrelsd_2_b32 s10, s14, SCC = 0 and SCC = 1: SGPR[10 + 3] = SGPR[14 + 2]: s13 = F16 (1), SCC kept
;     (1), s10, s11, s12, s14, s16 unchanged (5)                                                     -> 2 x 7 = 14
;   m0 = 0xfc030802 (bits 31:26 and 15:10 set, ignored): same, s13 = F16 (1), five unchanged (5)      -> 6
;   m0 = 0x10003: SGPR[10 + 1] = SGPR[14 + 3]: s11 = F17 (1), s10, s12, s13, s14 unchanged (4)         -> 5
;   m0 = 0x10000, s_movrelsd_2_b32 s10, s10: SGPR[11] = SGPR[10]: s11 = F10 (1), s10, s12 unchanged (2)  -> 3
;   total 14 + 6 + 5 + 3 = 28 checks: word 2 = 0x40000000 | 0xfffffff = 0x4fffffff
; Word 3 (3 checks): after m0 = 2, s_movreld_b32 s10, s4 a read of s12 sees s4 (the write went to the register file, not
;   to a copy: 1) and M0 is still 2 (1); after m0 = 0x20001, s_movrelsd_2_b32 s10, s14, M0 is still 0x20001 (1):
;   word 3 = 0x40000000 | 0x7 = 0x40000007.
; A translation that ignored M0 (wrote s10), took the b64 halves from one source, used all of M0 for the movrelsd_2 index
; or kept a stale value of a register a later s_mov reads, loses bits in the words.
; Expected mrt0 bits: 0x403fffff, 0x400fffff, 0x4fffffff, 0x40000007
s_mov_b32 s20, 0x40000000
s_mov_b32 s21, 0x40000000
s_mov_b32 s22, 0x40000000
s_mov_b32 s23, 0x40000000
s_mov_b32 s10, 0x40a00aaa
s_mov_b32 s11, 0x40b00bbb
s_mov_b32 s12, 0x40c00ccc
s_mov_b32 s13, 0x40d00ddd
s_mov_b32 s14, 0x40e00eee
s_mov_b32 s15, 0x40f00fff
s_mov_b32 s16, 0x41001110
s_mov_b32 s17, 0x41101221
s_mov_b32 s18, 0x41201332
s_mov_b32 s19, 0x41301443
s_mov_b32 s4, 0x4afe0004
s_mov_b32 s5, 0x4afe0005
s_mov_b32 m0, 2
s_cmp_eq_u32 s10, 0
s_movreld_b32 s10, s4
s_cselect_b32 s30, 0, 0x1
s_or_b32 s20, s20, s30
s_cmp_eq_u32 s12, 0x4afe0004
s_cselect_b32 s30, 0x2, 0
s_or_b32 s20, s20, s30
s_cmp_eq_u32 s10, 0x40a00aaa
s_cselect_b32 s30, 0x4, 0
s_or_b32 s20, s20, s30
s_cmp_eq_u32 s11, 0x40b00bbb
s_cselect_b32 s30, 0x8, 0
s_or_b32 s20, s20, s30
s_cmp_eq_u32 s13, 0x40d00ddd
s_cselect_b32 s30, 0x10, 0
s_or_b32 s20, s20, s30
s_cmp_eq_u32 s14, 0x40e00eee
s_cselect_b32 s30, 0x20, 0
s_or_b32 s20, s20, s30
s_mov_b32 s10, 0x40a00aaa
s_mov_b32 s11, 0x40b00bbb
s_mov_b32 s12, 0x40c00ccc
s_mov_b32 s13, 0x40d00ddd
s_mov_b32 s14, 0x40e00eee
s_mov_b32 s15, 0x40f00fff
s_mov_b32 s16, 0x41001110
s_mov_b32 s17, 0x41101221
s_mov_b32 s18, 0x41201332
s_mov_b32 s19, 0x41301443
s_mov_b32 s4, 0x4afe0004
s_mov_b32 s5, 0x4afe0005
s_mov_b32 m0, 2
s_cmp_eq_u32 s10, s10
s_movreld_b32 s10, s4
s_cselect_b32 s30, 0x40, 0
s_or_b32 s20, s20, s30
s_cmp_eq_u32 s12, 0x4afe0004
s_cselect_b32 s30, 0x80, 0
s_or_b32 s20, s20, s30
s_cmp_eq_u32 s10, 0x40a00aaa
s_cselect_b32 s30, 0x100, 0
s_or_b32 s20, s20, s30
s_cmp_eq_u32 s11, 0x40b00bbb
s_cselect_b32 s30, 0x200, 0
s_or_b32 s20, s20, s30
s_cmp_eq_u32 s13, 0x40d00ddd
s_cselect_b32 s30, 0x400, 0
s_or_b32 s20, s20, s30
s_cmp_eq_u32 s14, 0x40e00eee
s_cselect_b32 s30, 0x800, 0
s_or_b32 s20, s20, s30
s_mov_b32 s10, 0x40a00aaa
s_mov_b32 s11, 0x40b00bbb
s_mov_b32 s12, 0x40c00ccc
s_mov_b32 s13, 0x40d00ddd
s_mov_b32 s14, 0x40e00eee
s_mov_b32 s15, 0x40f00fff
s_mov_b32 s16, 0x41001110
s_mov_b32 s17, 0x41101221
s_mov_b32 s18, 0x41201332
s_mov_b32 s19, 0x41301443
s_mov_b32 s4, 0x4afe0004
s_mov_b32 s5, 0x4afe0005
s_mov_b32 m0, 0
s_movreld_b32 s10, s4
s_cmp_eq_u32 s10, 0x4afe0004
s_cselect_b32 s30, 0x1000, 0
s_or_b32 s20, s20, s30
s_cmp_eq_u32 s11, 0x40b00bbb
s_cselect_b32 s30, 0x2000, 0
s_or_b32 s20, s20, s30
s_cmp_eq_u32 s12, 0x40c00ccc
s_cselect_b32 s30, 0x4000, 0
s_or_b32 s20, s20, s30
s_mov_b32 s10, 0x40a00aaa
s_mov_b32 s11, 0x40b00bbb
s_mov_b32 s12, 0x40c00ccc
s_mov_b32 s13, 0x40d00ddd
s_mov_b32 s14, 0x40e00eee
s_mov_b32 s15, 0x40f00fff
s_mov_b32 s16, 0x41001110
s_mov_b32 s17, 0x41101221
s_mov_b32 s18, 0x41201332
s_mov_b32 s19, 0x41301443
s_mov_b32 s4, 0x4afe0004
s_mov_b32 s5, 0x4afe0005
s_mov_b32 m0, 3
s_movreld_b32 s10, 0x4a123456
s_cmp_eq_u32 s13, 0x4a123456
s_cselect_b32 s30, 0x8000, 0
s_or_b32 s20, s20, s30
s_cmp_eq_u32 s10, 0x40a00aaa
s_cselect_b32 s30, 0x10000, 0
s_or_b32 s20, s20, s30
s_cmp_eq_u32 s12, 0x40c00ccc
s_cselect_b32 s30, 0x20000, 0
s_or_b32 s20, s20, s30
s_cmp_eq_u32 s14, 0x40e00eee
s_cselect_b32 s30, 0x40000, 0
s_or_b32 s20, s20, s30
s_mov_b32 s10, 0x40a00aaa
s_mov_b32 s11, 0x40b00bbb
s_mov_b32 s12, 0x40c00ccc
s_mov_b32 s13, 0x40d00ddd
s_mov_b32 s14, 0x40e00eee
s_mov_b32 s15, 0x40f00fff
s_mov_b32 s16, 0x41001110
s_mov_b32 s17, 0x41101221
s_mov_b32 s18, 0x41201332
s_mov_b32 s19, 0x41301443
s_mov_b32 s4, 0x4afe0004
s_mov_b32 s5, 0x4afe0005
s_mov_b32 m0, 1
s_movreld_b32 s11, s12
s_cmp_eq_u32 s12, 0x40c00ccc
s_cselect_b32 s30, 0x80000, 0
s_or_b32 s20, s20, s30
s_cmp_eq_u32 s11, 0x40b00bbb
s_cselect_b32 s30, 0x100000, 0
s_or_b32 s20, s20, s30
s_cmp_eq_u32 s13, 0x40d00ddd
s_cselect_b32 s30, 0x200000, 0
s_or_b32 s20, s20, s30
s_mov_b32 s10, 0x40a00aaa
s_mov_b32 s11, 0x40b00bbb
s_mov_b32 s12, 0x40c00ccc
s_mov_b32 s13, 0x40d00ddd
s_mov_b32 s14, 0x40e00eee
s_mov_b32 s15, 0x40f00fff
s_mov_b32 s16, 0x41001110
s_mov_b32 s17, 0x41101221
s_mov_b32 s18, 0x41201332
s_mov_b32 s19, 0x41301443
s_mov_b32 s4, 0x4afe0004
s_mov_b32 s5, 0x4afe0005
s_mov_b32 m0, 2
s_cmp_eq_u32 s10, 0
s_movreld_b64 s[10:11], s[4:5]
s_cselect_b32 s30, 0, 0x1
s_or_b32 s21, s21, s30
s_cmp_eq_u32 s12, 0x4afe0004
s_cselect_b32 s30, 0x2, 0
s_or_b32 s21, s21, s30
s_cmp_eq_u32 s13, 0x4afe0005
s_cselect_b32 s30, 0x4, 0
s_or_b32 s21, s21, s30
s_cmp_eq_u32 s10, 0x40a00aaa
s_cselect_b32 s30, 0x8, 0
s_or_b32 s21, s21, s30
s_cmp_eq_u32 s11, 0x40b00bbb
s_cselect_b32 s30, 0x10, 0
s_or_b32 s21, s21, s30
s_cmp_eq_u32 s14, 0x40e00eee
s_cselect_b32 s30, 0x20, 0
s_or_b32 s21, s21, s30
s_mov_b32 s10, 0x40a00aaa
s_mov_b32 s11, 0x40b00bbb
s_mov_b32 s12, 0x40c00ccc
s_mov_b32 s13, 0x40d00ddd
s_mov_b32 s14, 0x40e00eee
s_mov_b32 s15, 0x40f00fff
s_mov_b32 s16, 0x41001110
s_mov_b32 s17, 0x41101221
s_mov_b32 s18, 0x41201332
s_mov_b32 s19, 0x41301443
s_mov_b32 s4, 0x4afe0004
s_mov_b32 s5, 0x4afe0005
s_mov_b32 m0, 2
s_cmp_eq_u32 s10, s10
s_movreld_b64 s[10:11], s[4:5]
s_cselect_b32 s30, 0x40, 0
s_or_b32 s21, s21, s30
s_cmp_eq_u32 s12, 0x4afe0004
s_cselect_b32 s30, 0x80, 0
s_or_b32 s21, s21, s30
s_cmp_eq_u32 s13, 0x4afe0005
s_cselect_b32 s30, 0x100, 0
s_or_b32 s21, s21, s30
s_cmp_eq_u32 s10, 0x40a00aaa
s_cselect_b32 s30, 0x200, 0
s_or_b32 s21, s21, s30
s_cmp_eq_u32 s11, 0x40b00bbb
s_cselect_b32 s30, 0x400, 0
s_or_b32 s21, s21, s30
s_cmp_eq_u32 s14, 0x40e00eee
s_cselect_b32 s30, 0x800, 0
s_or_b32 s21, s21, s30
s_mov_b32 s10, 0x40a00aaa
s_mov_b32 s11, 0x40b00bbb
s_mov_b32 s12, 0x40c00ccc
s_mov_b32 s13, 0x40d00ddd
s_mov_b32 s14, 0x40e00eee
s_mov_b32 s15, 0x40f00fff
s_mov_b32 s16, 0x41001110
s_mov_b32 s17, 0x41101221
s_mov_b32 s18, 0x41201332
s_mov_b32 s19, 0x41301443
s_mov_b32 s4, 0x4afe0004
s_mov_b32 s5, 0x4afe0005
s_mov_b32 m0, 1
s_movreld_b64 s[10:11], s[4:5]
s_cmp_eq_u32 s11, 0x4afe0004
s_cselect_b32 s30, 0x1000, 0
s_or_b32 s21, s21, s30
s_cmp_eq_u32 s12, 0x4afe0005
s_cselect_b32 s30, 0x2000, 0
s_or_b32 s21, s21, s30
s_cmp_eq_u32 s10, 0x40a00aaa
s_cselect_b32 s30, 0x4000, 0
s_or_b32 s21, s21, s30
s_cmp_eq_u32 s13, 0x40d00ddd
s_cselect_b32 s30, 0x8000, 0
s_or_b32 s21, s21, s30
s_mov_b32 s10, 0x40a00aaa
s_mov_b32 s11, 0x40b00bbb
s_mov_b32 s12, 0x40c00ccc
s_mov_b32 s13, 0x40d00ddd
s_mov_b32 s14, 0x40e00eee
s_mov_b32 s15, 0x40f00fff
s_mov_b32 s16, 0x41001110
s_mov_b32 s17, 0x41101221
s_mov_b32 s18, 0x41201332
s_mov_b32 s19, 0x41301443
s_mov_b32 s4, 0x4afe0004
s_mov_b32 s5, 0x4afe0005
s_mov_b32 m0, 1
s_movreld_b64 s[10:11], s[10:11]
s_cmp_eq_u32 s11, 0x40a00aaa
s_cselect_b32 s30, 0x10000, 0
s_or_b32 s21, s21, s30
s_cmp_eq_u32 s12, 0x40b00bbb
s_cselect_b32 s30, 0x20000, 0
s_or_b32 s21, s21, s30
s_cmp_eq_u32 s10, 0x40a00aaa
s_cselect_b32 s30, 0x40000, 0
s_or_b32 s21, s21, s30
s_cmp_eq_u32 s13, 0x40d00ddd
s_cselect_b32 s30, 0x80000, 0
s_or_b32 s21, s21, s30
s_mov_b32 s10, 0x40a00aaa
s_mov_b32 s11, 0x40b00bbb
s_mov_b32 s12, 0x40c00ccc
s_mov_b32 s13, 0x40d00ddd
s_mov_b32 s14, 0x40e00eee
s_mov_b32 s15, 0x40f00fff
s_mov_b32 s16, 0x41001110
s_mov_b32 s17, 0x41101221
s_mov_b32 s18, 0x41201332
s_mov_b32 s19, 0x41301443
s_mov_b32 s4, 0x4afe0004
s_mov_b32 s5, 0x4afe0005
s_mov_b32 m0, 0x30002
s_cmp_eq_u32 s10, 0
s_movrelsd_2_b32 s10, s14
s_cselect_b32 s30, 0, 0x1
s_or_b32 s22, s22, s30
s_cmp_eq_u32 s13, 0x41001110
s_cselect_b32 s30, 0x2, 0
s_or_b32 s22, s22, s30
s_cmp_eq_u32 s10, 0x40a00aaa
s_cselect_b32 s30, 0x4, 0
s_or_b32 s22, s22, s30
s_cmp_eq_u32 s11, 0x40b00bbb
s_cselect_b32 s30, 0x8, 0
s_or_b32 s22, s22, s30
s_cmp_eq_u32 s12, 0x40c00ccc
s_cselect_b32 s30, 0x10, 0
s_or_b32 s22, s22, s30
s_cmp_eq_u32 s14, 0x40e00eee
s_cselect_b32 s30, 0x20, 0
s_or_b32 s22, s22, s30
s_cmp_eq_u32 s16, 0x41001110
s_cselect_b32 s30, 0x40, 0
s_or_b32 s22, s22, s30
s_mov_b32 s10, 0x40a00aaa
s_mov_b32 s11, 0x40b00bbb
s_mov_b32 s12, 0x40c00ccc
s_mov_b32 s13, 0x40d00ddd
s_mov_b32 s14, 0x40e00eee
s_mov_b32 s15, 0x40f00fff
s_mov_b32 s16, 0x41001110
s_mov_b32 s17, 0x41101221
s_mov_b32 s18, 0x41201332
s_mov_b32 s19, 0x41301443
s_mov_b32 s4, 0x4afe0004
s_mov_b32 s5, 0x4afe0005
s_mov_b32 m0, 0x30002
s_cmp_eq_u32 s10, s10
s_movrelsd_2_b32 s10, s14
s_cselect_b32 s30, 0x80, 0
s_or_b32 s22, s22, s30
s_cmp_eq_u32 s13, 0x41001110
s_cselect_b32 s30, 0x100, 0
s_or_b32 s22, s22, s30
s_cmp_eq_u32 s10, 0x40a00aaa
s_cselect_b32 s30, 0x200, 0
s_or_b32 s22, s22, s30
s_cmp_eq_u32 s11, 0x40b00bbb
s_cselect_b32 s30, 0x400, 0
s_or_b32 s22, s22, s30
s_cmp_eq_u32 s12, 0x40c00ccc
s_cselect_b32 s30, 0x800, 0
s_or_b32 s22, s22, s30
s_cmp_eq_u32 s14, 0x40e00eee
s_cselect_b32 s30, 0x1000, 0
s_or_b32 s22, s22, s30
s_cmp_eq_u32 s16, 0x41001110
s_cselect_b32 s30, 0x2000, 0
s_or_b32 s22, s22, s30
s_mov_b32 s10, 0x40a00aaa
s_mov_b32 s11, 0x40b00bbb
s_mov_b32 s12, 0x40c00ccc
s_mov_b32 s13, 0x40d00ddd
s_mov_b32 s14, 0x40e00eee
s_mov_b32 s15, 0x40f00fff
s_mov_b32 s16, 0x41001110
s_mov_b32 s17, 0x41101221
s_mov_b32 s18, 0x41201332
s_mov_b32 s19, 0x41301443
s_mov_b32 s4, 0x4afe0004
s_mov_b32 s5, 0x4afe0005
s_mov_b32 m0, 0xfc030802
s_movrelsd_2_b32 s10, s14
s_cmp_eq_u32 s13, 0x41001110
s_cselect_b32 s30, 0x4000, 0
s_or_b32 s22, s22, s30
s_cmp_eq_u32 s10, 0x40a00aaa
s_cselect_b32 s30, 0x8000, 0
s_or_b32 s22, s22, s30
s_cmp_eq_u32 s11, 0x40b00bbb
s_cselect_b32 s30, 0x10000, 0
s_or_b32 s22, s22, s30
s_cmp_eq_u32 s12, 0x40c00ccc
s_cselect_b32 s30, 0x20000, 0
s_or_b32 s22, s22, s30
s_cmp_eq_u32 s14, 0x40e00eee
s_cselect_b32 s30, 0x40000, 0
s_or_b32 s22, s22, s30
s_cmp_eq_u32 s16, 0x41001110
s_cselect_b32 s30, 0x80000, 0
s_or_b32 s22, s22, s30
s_mov_b32 s10, 0x40a00aaa
s_mov_b32 s11, 0x40b00bbb
s_mov_b32 s12, 0x40c00ccc
s_mov_b32 s13, 0x40d00ddd
s_mov_b32 s14, 0x40e00eee
s_mov_b32 s15, 0x40f00fff
s_mov_b32 s16, 0x41001110
s_mov_b32 s17, 0x41101221
s_mov_b32 s18, 0x41201332
s_mov_b32 s19, 0x41301443
s_mov_b32 s4, 0x4afe0004
s_mov_b32 s5, 0x4afe0005
s_mov_b32 m0, 0x10003
s_movrelsd_2_b32 s10, s14
s_cmp_eq_u32 s11, 0x41101221
s_cselect_b32 s30, 0x100000, 0
s_or_b32 s22, s22, s30
s_cmp_eq_u32 s10, 0x40a00aaa
s_cselect_b32 s30, 0x200000, 0
s_or_b32 s22, s22, s30
s_cmp_eq_u32 s12, 0x40c00ccc
s_cselect_b32 s30, 0x400000, 0
s_or_b32 s22, s22, s30
s_cmp_eq_u32 s13, 0x40d00ddd
s_cselect_b32 s30, 0x800000, 0
s_or_b32 s22, s22, s30
s_cmp_eq_u32 s14, 0x40e00eee
s_cselect_b32 s30, 0x1000000, 0
s_or_b32 s22, s22, s30
s_mov_b32 s10, 0x40a00aaa
s_mov_b32 s11, 0x40b00bbb
s_mov_b32 s12, 0x40c00ccc
s_mov_b32 s13, 0x40d00ddd
s_mov_b32 s14, 0x40e00eee
s_mov_b32 s15, 0x40f00fff
s_mov_b32 s16, 0x41001110
s_mov_b32 s17, 0x41101221
s_mov_b32 s18, 0x41201332
s_mov_b32 s19, 0x41301443
s_mov_b32 s4, 0x4afe0004
s_mov_b32 s5, 0x4afe0005
s_mov_b32 m0, 0x10000
s_movrelsd_2_b32 s10, s10
s_cmp_eq_u32 s11, 0x40a00aaa
s_cselect_b32 s30, 0x2000000, 0
s_or_b32 s22, s22, s30
s_cmp_eq_u32 s10, 0x40a00aaa
s_cselect_b32 s30, 0x4000000, 0
s_or_b32 s22, s22, s30
s_cmp_eq_u32 s12, 0x40c00ccc
s_cselect_b32 s30, 0x8000000, 0
s_or_b32 s22, s22, s30
s_mov_b32 s10, 0x40a00aaa
s_mov_b32 s11, 0x40b00bbb
s_mov_b32 s12, 0x40c00ccc
s_mov_b32 s13, 0x40d00ddd
s_mov_b32 s14, 0x40e00eee
s_mov_b32 s15, 0x40f00fff
s_mov_b32 s16, 0x41001110
s_mov_b32 s17, 0x41101221
s_mov_b32 s18, 0x41201332
s_mov_b32 s19, 0x41301443
s_mov_b32 s4, 0x4afe0004
s_mov_b32 s5, 0x4afe0005
s_mov_b32 m0, 2
s_movreld_b32 s10, s4
s_mov_b32 s31, s12
s_cmp_eq_u32 s31, 0x4afe0004
s_cselect_b32 s30, 0x1, 0
s_or_b32 s23, s23, s30
s_cmp_eq_u32 m0, 0x00000002
s_cselect_b32 s30, 0x2, 0
s_or_b32 s23, s23, s30
s_mov_b32 s10, 0x40a00aaa
s_mov_b32 s11, 0x40b00bbb
s_mov_b32 s12, 0x40c00ccc
s_mov_b32 s13, 0x40d00ddd
s_mov_b32 s14, 0x40e00eee
s_mov_b32 s15, 0x40f00fff
s_mov_b32 s16, 0x41001110
s_mov_b32 s17, 0x41101221
s_mov_b32 s18, 0x41201332
s_mov_b32 s19, 0x41301443
s_mov_b32 s4, 0x4afe0004
s_mov_b32 s5, 0x4afe0005
s_mov_b32 m0, 0x20001
s_movrelsd_2_b32 s10, s14
s_cmp_eq_u32 m0, 0x00020001
s_cselect_b32 s30, 0x4, 0
s_or_b32 s23, s23, s30
v_mov_b32_e32 v10, s20
v_mov_b32_e32 v11, s21
v_mov_b32_e32 v12, s22
v_mov_b32_e32 v13, s23
exp mrt0 v10, v11, v12, v13 done vm
s_endpgm
