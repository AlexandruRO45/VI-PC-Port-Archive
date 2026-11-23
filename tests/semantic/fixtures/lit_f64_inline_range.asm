; lit(V) with V in the inline-constant range on a binary64 source: the literal dword V is bits 63:32 and bits 31:0
; are zero (the "f64 literal sources" row of docs/design.md), not the inline integer V in bits 31:0. LLVM's
; disassembler prints such a literal as plain 0x1 (lit() only where literal and inline constant give the same
; value), so this is hand-written text: llvm-mc 23.1.3 -mcpu=gfx1030 assembles each line to the bytes in its
; comment, with the literal dword 1 after the instruction. lit(1) and lit(0x1) are the same literal.
; Exported: 1.0, then 2^-1042 * 2^1074 = 2^32 as (hi, lo), then 0 (0x3f800000, 0x41f00000, 0x0, 0x0).
v_mov_b32_e32 v2, 2                     ; encoding: [0x82,0x02,0x04,0x7e]
v_mov_b32_e32 v3, 0                     ; encoding: [0x80,0x02,0x06,0x7e]
; v[2:3] = 2^-1073. The literal is 2^-1042, so the compare holds; the inline 1 (2^-1074) would fail it.
v_cmp_gt_f64_e32 vcc_lo, lit(1), v[2:3] ; encoding: [0xff,0x04,0x48,0x7c,0x01,0x00,0x00,0x00]
v_cndmask_b32_e64 v0, 0, 1.0, vcc_lo    ; encoding: [0x00,0x00,0x01,0xd5,0x80,0xe4,0xa9,0x01]
; ldexp(2^-1042, 1074) = 2^32 (0x41f00000_00000000); the inline 1 would give 1.0 (0x3ff00000_00000000).
v_mov_b32_e32 v6, 0x432                 ; encoding: [0xff,0x02,0x0c,0x7e,0x32,0x04,0x00,0x00]
v_ldexp_f64 v[4:5], lit(0x1), v6        ; encoding: [0x04,0x00,0x68,0xd5,0xff,0x0c,0x02,0x02,0x01,0x00,0x00,0x00]
v_mov_b32_e32 v1, v5                    ; encoding: [0x05,0x03,0x02,0x7e]
v_mov_b32_e32 v2, v4                    ; encoding: [0x04,0x03,0x04,0x7e]
