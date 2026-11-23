from __future__ import annotations

"""v_pk_add_f16 (RDNA2 VOP3P) hardware-referenced semantic regression tests.

Covers:
- Packed binary16 floating-point addition across low and high halves:
    D.f16[0] = S0.f16[op_sel[0]] + S1.f16[op_sel[1]]
    D.f16[1] = S0.f16[op_sel_hi[0]] + S1.f16[op_sel_hi[1]]
- Modifiers: op_sel, op_sel_hi, neg_lo, neg_hi, and clamp [0.0, 1.0]
- IEEE 754 edge cases: signed zeros (+0, -0), infinities (+inf, -inf), NaNs,
  and subnormal / denormal float16 preservation
- Register operand variations: VGPRs, SGPRs, inline constants, and register aliasing (in-place)
- Full translation to SPIR-V via tools/rdna2spirv.py and execution via spirv_mini_interp

Standard library only: python -I, unittest, struct, math. Strictly zero game data.
"""

import math
import os
import struct
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.dont_write_bytecode = True
sys.path.insert(0, HERE)

import support  # noqa: E402
from spirv_mini_interp import run_fragment  # noqa: E402

# Standard IEEE 754 binary16 bit constants
ZERO_P = 0x0000        # +0.0
ZERO_N = 0x8000        # -0.0
ONE = 0x3C00           # 1.0
TWO = 0x4000           # 2.0
THREE = 0x4200         # 3.0
FOUR = 0x4400          # 4.0
FIVE = 0x4500          # 5.0
HALF = 0x3800          # 0.5
QUARTER = 0x3400       # 0.25
NEG_ONE = 0xBC00       # -1.0
NEG_TWO = 0xC000       # -2.0
NEG_HALF = 0xB800      # -0.5
NEG_QUARTER = 0xB400   # -0.25

MAX_NORMAL = 0x7BFF    # 65504.0
MIN_NORMAL = 0x0400    # 2^-14 ~= 6.1035e-5
MIN_SUB = 0x0001       # 2^-24 ~= 5.9605e-8 (smallest subnormal)
MAX_SUB = 0x03FF       # largest subnormal (2^-14 - 2^-24)

INF_P = 0x7C00         # +Infinity
INF_N = 0xFC00         # -Infinity
QNAN = 0x7E00          # Quiet NaN


def P(lo: int, hi: int) -> int:
    """Pack two 16-bit integers into a single 32-bit dword."""
    return ((hi & 0xFFFF) << 16) | (lo & 0xFFFF)


def f16_to_u16(val: float) -> int:
    """Convert a Python float to 16-bit binary16 uint representation."""
    return struct.unpack("<H", struct.pack("<e", val))[0]


def u16_to_f16(raw: int) -> float:
    """Convert a 16-bit binary16 uint representation to Python float."""
    return struct.unpack("<e", struct.pack("<H", raw & 0xFFFF))[0]


def run_pk_add(instruction: str, seeds: dict[str, int], old: int = 0xDEADBEEF) -> int:
    """Translate and run a minimal pixel shader exercising `v_pk_add_f16`, returning v0 from mrt0."""
    prog = [f"v_mov_b32 v0, {old:#x}"]
    for reg, val in sorted(seeds.items()):
        prefix = "s" if reg.startswith("s") else "v"
        prog.append(f"{prefix}_mov_b32 {reg}, {val:#x}")
    prog.append(instruction)
    prog.append("exp mrt0 v0, v0, v0, v0 done vm")
    prog.append("s_endpgm")
    spv = support.translate_lines("ps_pk_add_f16", prog)
    return support.mrt0_words(spv)[0]


class ValuPkAddF16HardwareTests(unittest.TestCase):
    def test_basic_arithmetic_both_lanes(self) -> None:
        """Test independent lane addition with exact float16 representations."""
        cases = [
            # (lo0, hi0), (lo1, hi1) -> (lo_exp, hi_exp)
            ((ONE, TWO), (TWO, THREE), (THREE, FIVE)),
            ((HALF, QUARTER), (QUARTER, HALF), (P(0x3A00, 0) & 0xFFFF, P(0x3A00, 0) & 0xFFFF)),
            ((ONE, NEG_ONE), (NEG_ONE, ONE), (ZERO_P, ZERO_P)),
            ((TWO, FOUR), (TWO, ONE), (FOUR, FIVE)),
            ((NEG_ONE, NEG_TWO), (NEG_TWO, NEG_ONE), (0xC200, 0xC200)),  # -3.0, -3.0
        ]
        for (lo0, hi0), (lo1, hi1), (exp_lo, exp_hi) in cases:
            s0 = P(lo0, hi0)
            s1 = P(lo1, hi1)
            expected = P(exp_lo, exp_hi)
            with self.subTest(s0=hex(s0), s1=hex(s1)):
                got = run_pk_add("v_pk_add_f16 v0, v1, v2", {"v1": s0, "v2": s1})
                self.assertEqual(got, expected)

    def test_signed_zeros(self) -> None:
        """IEEE 754 signed zero rules: (+0) + (+0) = +0, (-0) + (-0) = -0, (+0) + (-0) = +0."""
        # Low: +0 + +0 = +0; High: -0 + -0 = -0
        s0 = P(ZERO_P, ZERO_N)
        s1 = P(ZERO_P, ZERO_N)
        got = run_pk_add("v_pk_add_f16 v0, v1, v2", {"v1": s0, "v2": s1})
        self.assertEqual(got, P(ZERO_P, ZERO_N))

        # Low: +0 + -0 = +0; High: -0 + +0 = +0
        s0 = P(ZERO_P, ZERO_N)
        s1 = P(ZERO_N, ZERO_P)
        got = run_pk_add("v_pk_add_f16 v0, v1, v2", {"v1": s0, "v2": s1})
        self.assertEqual(got, P(ZERO_P, ZERO_P))

    def test_infinities_and_nans(self) -> None:
        """IEEE 754 infinity arithmetic and NaN propagation."""
        # +Inf + 1.0 = +Inf, -Inf + 1.0 = -Inf
        s0 = P(INF_P, INF_N)
        s1 = P(ONE, ONE)
        got = run_pk_add("v_pk_add_f16 v0, v1, v2", {"v1": s0, "v2": s1})
        self.assertEqual(got, P(INF_P, INF_N))

        # +Inf + -Inf = NaN
        s0 = P(INF_P, ONE)
        s1 = P(INF_N, TWO)
        got = run_pk_add("v_pk_add_f16 v0, v1, v2", {"v1": s0, "v2": s1})
        lo_got = got & 0xFFFF
        hi_got = (got >> 16) & 0xFFFF
        self.assertTrue(math.isnan(u16_to_f16(lo_got)))
        self.assertEqual(hi_got, THREE)

        # NaN operand propagation
        s0 = P(QNAN, TWO)
        s1 = P(ONE, QNAN)
        got = run_pk_add("v_pk_add_f16 v0, v1, v2", {"v1": s0, "v2": s1})
        self.assertTrue(math.isnan(u16_to_f16(got & 0xFFFF)))
        self.assertTrue(math.isnan(u16_to_f16((got >> 16) & 0xFFFF)))

    def test_subnormal_float16_preservation(self) -> None:
        """RDNA2 keeps float16 subnormals (MODE IEEE=0, denormals kept)."""
        # Smallest subnormals addition: 2^-24 + 2^-24 = 2^-23
        s0 = P(MIN_SUB, MIN_SUB)
        s1 = P(MIN_SUB, MIN_SUB)
        got = run_pk_add("v_pk_add_f16 v0, v1, v2", {"v1": s0, "v2": s1})
        self.assertEqual(got, P(0x0002, 0x0002))

        # Normal + subnormal rounding: MIN_NORMAL (2^-14) - MIN_SUB (2^-24) = MAX_SUB
        s0 = P(MIN_NORMAL, MIN_NORMAL)
        s1 = P(0x8001, 0x8001)  # -MIN_SUB
        got = run_pk_add("v_pk_add_f16 v0, v1, v2", {"v1": s0, "v2": s1})
        self.assertEqual(got, P(MAX_SUB, MAX_SUB))

    def test_overflow_saturation(self) -> None:
        """Addition exceeding MAX_NORMAL overflows to infinity."""
        s0 = P(MAX_NORMAL, MAX_NORMAL)
        s1 = P(MAX_NORMAL, MAX_NORMAL)
        got = run_pk_add("v_pk_add_f16 v0, v1, v2", {"v1": s0, "v2": s1})
        self.assertEqual(got, P(INF_P, INF_P))

    def test_modifiers_op_sel_and_op_sel_hi(self) -> None:
        """Modifier op_sel and op_sel_hi routing half words into destinations."""
        # S0 = (lo=1.0, hi=3.0), S1 = (lo=0.5, hi=-0.25)
        s0 = P(ONE, THREE)
        s1 = P(HALF, NEG_QUARTER)

        # Default: lo = S0.lo (1.0) + S1.lo (0.5) = 1.5 (0x3E00)
        #          hi = S0.hi (3.0) + S1.hi (-0.25) = 2.75 (0x4180)
        got = run_pk_add("v_pk_add_f16 v0, v1, v2", {"v1": s0, "v2": s1})
        self.assertEqual(got, P(0x3E00, 0x4180))

        # op_sel:[1, 0] -> lo = S0.hi (3.0) + S1.lo (0.5) = 3.5 (0x4300)
        #                  hi = S0.hi (3.0) + S1.hi (-0.25) = 2.75 (0x4180)
        got = run_pk_add("v_pk_add_f16 v0, v1, v2 op_sel:[1,0]", {"v1": s0, "v2": s1})
        self.assertEqual(got, P(0x4300, 0x4180))

        # op_sel:[0, 1] op_sel_hi:[0, 0] ->
        # lo = S0.lo (1.0) + S1.hi (-0.25) = 0.75 (0x3A00)
        # hi = S0.lo (1.0) + S1.lo (0.5) = 1.5 (0x3E00)
        got = run_pk_add(
            "v_pk_add_f16 v0, v1, v2 op_sel:[0,1] op_sel_hi:[0,0]",
            {"v1": s0, "v2": s1},
        )
        self.assertEqual(got, P(0x3A00, 0x3E00))

    def test_modifiers_neg_lo_and_neg_hi(self) -> None:
        """Negation modifiers neg_lo and neg_hi."""
        s0 = P(ONE, THREE)
        s1 = P(HALF, NEG_QUARTER)

        # neg_lo:[1, 0] -> lo = -S0.lo (-1.0) + S1.lo (0.5) = -0.5 (0xB800)
        got = run_pk_add("v_pk_add_f16 v0, v1, v2 neg_lo:[1,0]", {"v1": s0, "v2": s1})
        self.assertEqual(got, P(NEG_HALF, 0x4180))

        # neg_hi:[0, 1] -> hi = S0.hi (3.0) + -S1.hi (-(-0.25) = 0.25) = 3.25 (0x4280)
        got = run_pk_add("v_pk_add_f16 v0, v1, v2 neg_hi:[0,1]", {"v1": s0, "v2": s1})
        self.assertEqual(got, P(0x3E00, 0x4280))

        # neg_lo:[1, 1] neg_hi:[1, 1] -> both operands negated in both halves
        got = run_pk_add(
            "v_pk_add_f16 v0, v1, v2 neg_lo:[1,1] neg_hi:[1,1]",
            {"v1": s0, "v2": s1},
        )
        self.assertEqual(got, P(0xBE00, 0xC180))

    def test_clamp_modifier(self) -> None:
        """Modifier clamp saturates results to range [0.0, 1.0]."""
        # (1.5, -0.75) clamped -> (1.0, 0.0)
        s0 = P(0x3E00, NEG_QUARTER)
        s1 = P(HALF, NEG_HALF)
        got = run_pk_add("v_pk_add_f16 v0, v1, v2 clamp", {"v1": s0, "v2": s1})
        self.assertEqual(got, P(ONE, ZERO_P))

        # Inside [0.0, 1.0]: (0.25 + 0.25, 0.25 + 0.5) -> (0.5, 0.75) unchanged
        s0 = P(QUARTER, QUARTER)
        s1 = P(QUARTER, HALF)
        got = run_pk_add("v_pk_add_f16 v0, v1, v2 clamp", {"v1": s0, "v2": s1})
        self.assertEqual(got, P(HALF, 0x3A00))

    def test_sgpr_and_aliasing(self) -> None:
        """Register variations: scalar SGPR sources and destination-operand aliasing."""
        # Both SGPR sources: s1 + s2 -> v0
        s0 = P(ONE, TWO)
        s1 = P(TWO, ONE)
        got = run_pk_add("v_pk_add_f16 v0, s1, s2", {"s1": s0, "s2": s1})
        self.assertEqual(got, P(THREE, THREE))

        # Mixed SGPR src0 and VGPR src1: s1 + v2 -> v0
        got = run_pk_add("v_pk_add_f16 v0, s1, v2", {"s1": s0, "v2": s1})
        self.assertEqual(got, P(THREE, THREE))

        # Register aliasing: v0 = v0 + v0 (in-place doubling)
        s0 = P(ONE, TWO)
        got = run_pk_add("v_pk_add_f16 v0, v0, v0", {"v0": s0})
        self.assertEqual(got, P(TWO, FOUR))

    def test_spirv_structure(self) -> None:
        """Verify that translation produces proper SPIR-V vector float addition operations."""
        prog = [
            "v_mov_b32 v1, 0x40003c00",
            "v_mov_b32 v2, 0x42004000",
            "v_pk_add_f16 v0, v1, v2",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_pk_add_f16_struct", prog)
        self.assertIn("OpFAdd %v2float", spv)
        self.assertIn("UnpackHalf2x16", spv)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], P(THREE, FIVE))


if __name__ == "__main__":
    unittest.main()
