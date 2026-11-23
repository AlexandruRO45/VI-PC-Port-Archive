"""Semantic regression tests for VOP3 op_sel (and denormal-safe f16 widening) on v_fmac_f16, v_cvt_f32_f16, and v_cvt_f16_f32.

RDNA2 ISA VOP3 op_sel rules (docs/design.md, "16-bit results") + tools/rdna2_decode.py:
  * v_fmac_f16_e64 D = fma(S0, S1, D) has nsrc=2, so VOP3 prints 3-bit op_sel:[s0,s1,dst].
    Both the accumulator read of D and the writeback to D select half `dst`, keeping the other half intact.
  * v_cvt_f32_f16_e64 D = float(S0) has nsrc=1 with an f16 source and an f32 destination:
    op_sel:[s0,0] selects source half s0 (and widens subnormals cleanly, including SDWA BYTE_1);
    op_sel dst bit 1 is rejected (Unsupported).
  * v_cvt_f16_f32_e64 D = half(S0) has nsrc=1 with an f32 source and an f16 destination:
    op_sel:[0,dst] writes half dst while preserving the other half of D;
    op_sel src0 bit 1 is rejected (Unsupported).

Standard library only; no game files, no LLVM, no GPU required.
"""
import itertools
import math
import os
import struct
import sys
import unittest
from fractions import Fraction

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import support  # noqa: E402

if support.TOOLS_DIR not in sys.path:
    sys.path.insert(0, support.TOOLS_DIR)

import rdna2_decode  # noqa: E402
import rdna2spirv  # noqa: E402
import spirv_asm  # noqa: E402


def h2f(h: int) -> float:
    return struct.unpack("<e", struct.pack("<H", h & 0xFFFF))[0]


def f2h(x: float) -> int:
    try:
        return struct.unpack("<H", struct.pack("<e", x))[0]
    except OverflowError:
        return 0x7C00 if x > 0 else 0xFC00


def f2u(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", x))[0]


def u2f(u: int) -> float:
    return struct.unpack("<f", struct.pack("<I", u & 0xFFFFFFFF))[0]


def round_f16_exact(x: Fraction) -> int:
    """Round an exact rational to IEEE-754 binary16 RNE bits (exact overflow at 65520)."""
    if x == 0:
        return 0
    sign = 0x8000 if x < 0 else 0
    ax = abs(x)
    lo, hi = 0, 0x7C00
    while lo < hi:
        mid = (lo + hi) // 2
        mv = Fraction(h2f(mid)) if mid < 0x7C00 else Fraction(10 ** 9)
        if mv >= ax:
            hi = mid
        else:
            lo = mid + 1
    if lo == 0x7C00:
        return sign | (0x7C00 if ax >= 65520 else 0x7BFF)
    if lo == 0 or Fraction(h2f(lo)) == ax:
        return sign | lo
    below = lo - 1
    da = Fraction(h2f(lo)) - ax
    db = ax - Fraction(h2f(below))
    pick = lo if da < db else below if db < da else (lo if lo % 2 == 0 else below)
    return sign | pick


def ref_fmac_f16(s0_h: int, s1_h: int, acc_h: int, *, clamp: bool = False) -> int:
    """Exact single-rounded f16 FMA reference with optional VOP3 DX10 clamp [0.0, 1.0]."""
    r = round_f16_exact(Fraction(h2f(s0_h)) * Fraction(h2f(s1_h)) + Fraction(h2f(acc_h)))
    if clamp:
        val = h2f(r)
        if val <= 0.0:
            return 0x0000
        if val >= 1.0:
            return 0x3C00
    return r


def encode_vop3a(opc: int, vdst: int, src0: int, src1: int = 0, src2: int = 0, opsel: int = 0) -> bytes:
    """Encode an 8-byte RDNA2 VOP3a instruction word pair."""
    w0 = (0x35 << 26) | ((opc & 0x3FF) << 16) | ((opsel & 0xF) << 11) | (vdst & 0xFF)
    w1 = ((src2 & 0x1FF) << 18) | ((src1 & 0x1FF) << 9) | (src0 & 0x1FF)
    return struct.pack("<II", w0, w1)


def run_ps(lines: list[str]) -> tuple[str, list[int]]:
    spv = support.translate_lines("ps_f16_opsel_fmac_cvt", lines)
    return spv, support.mrt0_words(spv)


class F16OpselFmacCvtTests(unittest.TestCase):
    def test_vop3_decoder_roundtrip(self):
        # Verify raw RDNA2 VOP3 bytes -> rdna2_decode.disasm_bytes -> rdna2spirv -> spirv_mini_interp.
        cases = [
            (
                "v_fmac_f16 op_sel:[0,0,1]",
                encode_vop3a(0x136, 3, 257, 258, opsel=0b1000),
                "op_sel:[0,0,1]",
                [
                    "v_mov_b32 v1, 0x40003800",   # hi=2.0, lo=0.5
                    "v_mov_b32 v2, 0x42003c00",   # hi=3.0, lo=1.0
                    "v_mov_b32 v3, 0x44001234",   # hi=4.0 (acc), lo=0x1234 (kept)
                ],
                # dst.hi = 0.5 * 1.0 + 4.0 = 4.5 (0x4480); dst.lo kept as 0x1234
                (f2h(4.5) << 16) | 0x1234,
            ),
            (
                "v_fmac_f16 op_sel:[1,1,1]",
                encode_vop3a(0x136, 3, 257, 258, opsel=0b1011),
                "op_sel:[1,1,1]",
                [
                    "v_mov_b32 v1, 0x40003800",   # hi=2.0, lo=0.5
                    "v_mov_b32 v2, 0x42003c00",   # hi=3.0, lo=1.0
                    "v_mov_b32 v3, 0x3e00abcd",   # hi=1.5 (acc), lo=0xabcd (kept)
                ],
                # dst.hi = 2.0 * 3.0 + 1.5 = 7.5 (0x4780); dst.lo kept as 0xabcd
                (f2h(7.5) << 16) | 0xABCD,
            ),
            (
                "v_cvt_f32_f16 op_sel:[1,0]",
                encode_vop3a(0x18B, 3, 257, opsel=0b0001),
                "op_sel:[1,0]",
                ["v_mov_b32 v1, 0x42003800"],     # hi=3.0, lo=0.5
                f2u(3.0),
            ),
            (
                "v_cvt_f16_f32 op_sel:[0,1]",
                encode_vop3a(0x18A, 3, 257, opsel=0b1000),
                "op_sel:[0,1]",
                [
                    f"v_mov_b32 v1, {f2u(-2.5):#010x}",
                    "v_mov_b32 v3, 0xaaaa1357",
                ],
                (f2h(-2.5) << 16) | 0x1357,
            ),
        ]
        for label, raw, expected_opsel, setup, want in cases:
            with self.subTest(case=label):
                text, _ = rdna2_decode.disasm_bytes(raw)
                decoded_line = text.strip()
                self.assertIn(expected_opsel, decoded_line)
                lines = setup + [decoded_line, "exp mrt0 v3, v3, v3, v3 done vm", "s_endpgm"]
                _, got = run_ps(lines)
                self.assertEqual(got[0], want, f"{decoded_line}: got {got[0]:#010x}, want {want:#010x}")

    def test_fmac_f16_all_opsel_combinations(self):
        # Distinct finite binary16 values in every half so any wrong half selection fails immediately.
        v1_lo, v1_hi = f2h(0.5), f2h(2.0)       # 0x3800, 0x4000
        v2_lo, v2_hi = f2h(1.5), f2h(-3.0)      # 0x3e00, 0xc200
        v3_lo, v3_hi = f2h(4.0), f2h(0.25)      # 0x4400, 0x3400
        v1 = (v1_hi << 16) | v1_lo
        v2 = (v2_hi << 16) | v2_lo
        v3 = (v3_hi << 16) | v3_lo

        for s0, s1, dhi in itertools.product((0, 1), repeat=3):
            with self.subTest(op_sel=(s0, s1, dhi)):
                a = v1_hi if s0 else v1_lo
                b = v2_hi if s1 else v2_lo
                c = v3_hi if dhi else v3_lo
                res_h = ref_fmac_f16(a, b, c)
                want = ((res_h << 16) | v3_lo) if dhi else ((v3_hi << 16) | res_h)
                _, got = run_ps([
                    f"v_mov_b32 v1, {v1:#010x}",
                    f"v_mov_b32 v2, {v2:#010x}",
                    f"v_mov_b32 v3, {v3:#010x}",
                    f"v_fmac_f16_e64 v3, v1, v2 op_sel:[{s0},{s1},{dhi}]",
                    "exp mrt0 v3, v3, v3, v3 done vm",
                    "s_endpgm",
                ])
                self.assertEqual(got[0], want, f"got {got[0]:#010x}, want {want:#010x}")

    def test_fmac_f16_opsel_mods_alias_const_rounding_exec(self):
        # v0: neg + abs on high halves -> dst.hi: (-2.0) * |-3.0| + 0.25 = -5.75 (0xc5c0), keep lo 0xdead
        # v1: clamp > 1.0 in high half -> dst.hi: 2.0 * 3.0 + 0.5 -> 1.0 (0x3c00), keep lo 0xbeef
        # v2: clamp < 0.0 in high half -> dst.hi: (-2.0) * 1.0 + 0.5 = -1.5 -> 0.0 (0x0000), keep lo 0xcafe
        # v3: dst aliases src0 with distinct halves: op_sel:[0,1,1] -> dst.hi = dst.lo (0.5) * v6.hi (3.0) + dst.hi (4.0) = 5.5 (0x4580), keep lo (0x3800)
        lines = [
            f"v_mov_b32 v4, {(f2h(2.0) << 16) | f2h(9.0):#010x}",
            f"v_mov_b32 v5, {(f2h(-3.0) << 16) | f2h(5.0):#010x}",
            f"v_mov_b32 v6, {(f2h(3.0) << 16) | f2h(1.0):#010x}",
            f"v_mov_b32 v0, {(f2h(0.25) << 16) | 0xDEAD:#010x}",
            "v_fmac_f16_e64 v0, -v4, |v5| op_sel:[1,1,1]",
            f"v_mov_b32 v1, {(f2h(0.5) << 16) | 0xBEEF:#010x}",
            "v_fmac_f16_e64 v1, v4, v6 op_sel:[1,1,1] clamp",
            f"v_mov_b32 v2, {(f2h(0.5) << 16) | 0xCAFE:#010x}",
            "v_fmac_f16_e64 v2, -v4, v6 op_sel:[1,0,1] clamp",
            f"v_mov_b32 v3, {(f2h(4.0) << 16) | f2h(0.5):#010x}",
            "v_fmac_f16_e64 v3, v3, v6 op_sel:[0,1,1]",
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ]
        _, got = run_ps(lines)
        self.assertEqual(got, [
            (f2h(-5.75) << 16) | 0xDEAD,
            (0x3C00 << 16) | 0xBEEF,
            (0x0000 << 16) | 0xCAFE,
            (f2h(5.5) << 16) | f2h(0.5),
        ])

        # Near-tie single-rounding FMA in the high half: a = 0x4401 (4.00390625), b = 3.0 (0x4200),
        # c = -2^-24 (0x8001). Exact sum is 12 + 1.5*ulp(12) - 2^-24, just below the tie between 0x4a01 and
        # 0x4a02 -> rounds once down to 0x4a01 (rounding a*b+c to f32 drops -2^-24 and ties to even 0x4a02).
        a_tie, b_tie, c_nudge = 0x4401, 0x4200, 0x8001
        self.assertEqual(ref_fmac_f16(a_tie, b_tie, c_nudge), 0x4A01)
        _, got2 = run_ps([
            f"v_mov_b32 v4, {(a_tie << 16) | 0x3C00:#010x}",
            f"v_mov_b32 v5, {(0x4000 << 16) | b_tie:#010x}",
            # Single-rounded FMA with op_sel:[1,0,1]: v0.lo kept (0x1111)
            f"v_mov_b32 v0, {(c_nudge << 16) | 0x1111:#010x}",
            "v_fmac_f16_e64 v0, v4, v5 op_sel:[1,0,1]",
            # Inline float constant 1.0 with op_sel[0]=1 reads +0.0 -> (+0.0) * 2.0 + 1.5 = 1.5, keep lo 0x2222
            f"v_mov_b32 v1, {(f2h(1.5) << 16) | 0x2222:#010x}",
            "v_fmac_f16_e64 v1, 1.0, v5 op_sel:[1,1,1]",
            # Inactive EXEC keeps both halves of v2
            "v_mov_b32 v2, 0xdeadbeef",
            "s_mov_b32 exec_lo, 0",
            "v_fmac_f16_e64 v2, v4, v5 op_sel:[1,1,1]",
            "s_mov_b32 exec_lo, -1",
            "v_mov_b32 v3, 0",
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ])
        self.assertEqual(got2[0], (0x4A01 << 16) | 0x1111)
        self.assertEqual(got2[1], (f2h(1.5) << 16) | 0x2222)
        self.assertEqual(got2[2], 0xDEADBEEF)

    def test_cvt_f32_f16_opsel_and_denormals(self):
        # Basic op_sel:[0,0] vs op_sel:[1,0] with signed zeros, subnormals, infinities, and high-half NaN.
        sub_min = 0x0001       # +2^-24 = 0x33800000
        sub_neg_max = 0x83FF   # -(1023 * 2^-24) -> u2f / f2u via exact Python half_to_float
        _, got = run_ps([
            f"v_mov_b32 v4, {(0x4200 << 16) | 0x3800:#010x}",           # hi=3.0, lo=0.5
            f"v_mov_b32 v5, {(0x8000 << 16) | 0x0000:#010x}",           # hi=-0.0, lo=+0.0
            f"v_mov_b32 v6, {(sub_neg_max << 16) | sub_min:#010x}",     # hi=-1023*2^-24, lo=+2^-24
            f"v_mov_b32 v7, {(0xFC00 << 16) | 0x7C00:#010x}",           # hi=-inf, lo=+inf
            "v_cvt_f32_f16_e64 v0, v4 op_sel:[1,0]",
            "v_cvt_f32_f16_e64 v1, v5 op_sel:[1,0]",
            "v_cvt_f32_f16_e64 v2, v6 op_sel:[1,0]",
            "v_cvt_f32_f16_e64 v3, v7 op_sel:[1,0]",
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ])
        self.assertEqual(got, [
            f2u(3.0),
            0x80000000,
            f2u(h2f(sub_neg_max)),
            0xFF800000,
        ])

        # Source mods (-, | |), output mods (mul:2, div:2, clamp), inline constants, and high-half NaN.
        _, got_mods = run_ps([
            f"v_mov_b32 v4, {(0xC000 << 16) | 0x3C00:#010x}",           # hi=-2.0, lo=1.0
            "v_cvt_f32_f16_e64 v0, |v4| op_sel:[1,0] div:2",            # |-2.0| / 2 = 1.0 (0x3f800000)
            "v_cvt_f32_f16_e64 v1, -v4 op_sel:[1,0] mul:2 clamp",       # clamp(2.0 * 2) = 1.0 (0x3f800000)
            "v_cvt_f32_f16_e64 v2, 0.5 op_sel:[1,0]",                   # high half of inline constant = +0.0
            f"v_mov_b32 v5, {(0x7E00 << 16) | 0x3C00:#010x}",           # hi=qNaN, lo=1.0
            "v_cvt_f32_f16_e64 v3, v5 op_sel:[1,0]",
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ])
        self.assertEqual(got_mods[:3], [f2u(1.0), f2u(1.0), 0x00000000])
        self.assertTrue(math.isnan(u2f(got_mods[3])), f"expected NaN, got {got_mods[3]:#010x}")

        # SDWA BYTE_1 subnormal widening must route through f16_in's integer exp==0 fallback (not bare UnpackHalf2x16),
        # plus 32-bit literal and inline-constant low-half selection with op_sel.
        spv_sdwa, got_sdwa = run_ps([
            "v_cvt_f32_f16_sdwa v0, 2.0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1",      # 0x40 = 64*2^-24 = 2^-18
            "v_mov_b32 v4, 0x000081ff",
            "v_cvt_f32_f16_sdwa v1, v4 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1",       # 0x81 = 129*2^-24
            "v_cvt_f32_f16_e64 v2, 0x42003800 op_sel:[1,0]",                                       # hi=3.0 (0x4200)
            "v_cvt_f32_f16_e64 v3, 0.5 op_sel:[0,0]",                                              # lo=0.5
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ])
        self.assertEqual(got_sdwa[0], f2u(64 * (2.0 ** -24)))
        self.assertEqual(got_sdwa[1], f2u(129 * (2.0 ** -24)))
        self.assertEqual(got_sdwa[2], f2u(3.0))
        self.assertEqual(got_sdwa[3], f2u(0.5))
        self.assertIn("OpConvertUToF %float", spv_sdwa)
        self.assertIn("OpSelect %float", spv_sdwa)

        # Destination op_sel bit is unsupported on 32-bit f32 destination.
        for bad_sel in ("op_sel:[0,1]", "op_sel:[1,1]"):
            with self.subTest(bad_sel=bad_sel):
                with self.assertRaises(rdna2spirv.Unsupported):
                    support.translate_lines("ps_bad_cvt_f32_f16", [
                        "v_mov_b32 v1, 0x3c003800",
                        f"v_cvt_f32_f16_e64 v0, v1 {bad_sel}",
                        "exp mrt0 v0, v0, v0, v0 done vm",
                        "s_endpgm",
                    ])

    def test_cvt_f16_f32_opsel_mods_and_exec(self):
        # op_sel:[0,0] keeps high half; op_sel:[0,1] writes high half and keeps low half.
        _, got = run_ps([
            "v_mov_b32 v0, 0xaaaa5555",
            f"v_mov_b32 v4, {f2u(1.5):#010x}",
            "v_cvt_f16_f32_e64 v0, v4 op_sel:[0,1]",                   # hi=1.5 (0x3e00), keep lo 0x5555
            "v_mov_b32 v1, 0xdeadbeef",
            "v_cvt_f16_f32_e64 v1, -v4 op_sel:[0,1]",                  # hi=-1.5 (0xbe00), keep lo 0xbeef
            "v_mov_b32 v2, 0x11112222",
            f"v_mov_b32 v5, {f2u(-(2.0 ** -24)):#010x}",
            "v_cvt_f16_f32_e64 v2, v5 op_sel:[0,1]",                   # hi=-2^-24 (0x8001), keep lo 0x2222
            "v_mov_b32 v3, 0x33334444",
            f"v_mov_b32 v6, {f2u(65520.0):#010x}",
            "v_cvt_f16_f32_e64 v3, v6 op_sel:[0,0]",                   # lo=+inf (0x7c00), keep hi 0x3333
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ])
        self.assertEqual(got, [
            (f2h(1.5) << 16) | 0x5555,
            (f2h(-1.5) << 16) | 0xBEEF,
            (0x8001 << 16) | 0x2222,
            (0x3333 << 16) | 0x7C00,
        ])

        # Output modifiers (div:2 exp-1 negative underflow -> -0.0, mul:2 + clamp) + inactive EXEC with op_sel:[0,1].
        _, got_mods = run_ps([
            "v_mov_b32 v0, 0x12345678",
            f"v_mov_b32 v4, {f2u(-(2.0 ** -14)):#010x}",
            "v_cvt_f16_f32_e64 v0, v4 op_sel:[0,1] div:2",             # exp-1 underflows to -0.0 (0x8000) in hi
            "v_mov_b32 v1, 0x12345678",
            "v_cvt_f16_f32_e64 v1, v4 op_sel:[0,1] div:2 clamp",       # clamp turns -0.0 -> +0.0 (0x0000) in hi
            "v_mov_b32 v2, 0x1234abcd",
            f"v_mov_b32 v5, {f2u(2.0):#010x}",
            "v_cvt_f16_f32_e64 v2, v5 op_sel:[0,1] mul:2 clamp",       # clamp -> 1.0 (0x3c00) in hi
            "v_mov_b32 v3, 0xaaaa5555",
            "s_mov_b32 exec_lo, 0",
            "v_cvt_f16_f32_e64 v3, v5 op_sel:[0,1]",
            "s_mov_b32 exec_lo, -1",
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ])
        self.assertEqual(got_mods, [
            (0x8000 << 16) | 0x5678,
            (0x0000 << 16) | 0x5678,
            (0x3C00 << 16) | 0xABCD,
            0xAAAA5555,
        ])

        # Source op_sel bit is unsupported on 32-bit f32 source, and SGPR dst cannot take op_sel dst bit.
        for bad in (
            ["v_mov_b32 v1, 0x3f800000", "v_cvt_f16_f32_e64 v0, v1 op_sel:[1,0]"],
            ["v_mov_b32 v1, 0x3f800000", "v_cvt_f16_f32_e64 v0, v1 op_sel:[1,1]"],
        ):
            with self.subTest(bad=bad[1]):
                with self.assertRaises(rdna2spirv.Unsupported):
                    support.translate_lines("ps_bad_cvt_f16_f32", bad + ["exp mrt0 v0, v0, v0, v0 done vm", "s_endpgm"])

    def test_spirv_assembles_cleanly(self):
        spv, _ = run_ps([
            "v_mov_b32 v1, 0x40003800",
            "v_mov_b32 v2, 0x42003c00",
            "v_mov_b32 v0, 0x44001234",
            "v_fmac_f16_e64 v0, v1, v2 op_sel:[1,0,1]",
            "v_cvt_f32_f16_e64 v3, v0 op_sel:[1,0]",
            "v_cvt_f16_f32_e64 v3, v3 op_sel:[0,1]",
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ])
        binary = spirv_asm.assemble(spv)
        self.assertGreater(len(binary), 20)
        self.assertEqual(struct.unpack_from("<I", binary, 0)[0], 0x07230203)


if __name__ == "__main__":
    unittest.main()
