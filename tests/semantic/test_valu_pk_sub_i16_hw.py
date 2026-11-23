from __future__ import annotations

"""v_pk_sub_i16 (RDNA2 VOP3P) hardware-referenced semantic regression tests.

Covers:
- Packed 16-bit signed integer subtraction across low and high halves:
    D.i16[0] = (S0.i16[op_sel[0]] - S1.i16[op_sel[1]]) & 0xFFFF
    D.i16[1] = (S0.i16[op_sel_hi[0]] - S1.i16[op_sel_hi[1]]) & 0xFFFF
- Modifiers: op_sel, op_sel_hi, and signed clamp [-32768, 32767] ([-0x8000, 0x7FFF])
- Two's-complement arithmetic wrapping modulo 65536 when clamp is omitted
- Register operand variations: VGPRs, SGPRs, and in-place register aliasing
- Full translation to SPIR-V via tools/rdna2spirv.py and execution via spirv_mini_interp

Standard library only: python -I, unittest, struct, math. Strictly zero game data.
"""

import os
import struct
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.dont_write_bytecode = True
sys.path.insert(0, HERE)

import support  # noqa: E402


def P(lo: int, hi: int) -> int:
    """Pack two 16-bit integers into a single 32-bit dword."""
    return ((hi & 0xFFFF) << 16) | (lo & 0xFFFF)


def to_u16(signed_val: int) -> int:
    """Convert a Python signed integer into 16-bit unsigned bit representation."""
    return signed_val & 0xFFFF


def run_pk_sub(instruction: str, seeds: dict[str, int], old: int = 0xDEADBEEF) -> int:
    """Translate and run a minimal pixel shader exercising `v_pk_sub_i16`, returning v0 from mrt0."""
    prog = [f"v_mov_b32 v0, {old:#x}"]
    for reg, val in sorted(seeds.items()):
        prefix = "s" if reg.startswith("s") else "v"
        prog.append(f"{prefix}_mov_b32 {reg}, {val:#x}")
    prog.append(instruction)
    prog.append("exp mrt0 v0, v0, v0, v0 done vm")
    prog.append("s_endpgm")
    spv = support.translate_lines("ps_pk_sub_i16", prog)
    return support.mrt0_words(spv)[0]


class ValuPkSubI16HardwareTests(unittest.TestCase):
    def test_basic_signed_subtraction_both_lanes(self) -> None:
        """Test independent lane signed subtraction without overflow."""
        cases = [
            # ((lo0, hi0), (lo1, hi1), (exp_lo, exp_hi))
            ((10, 20), (3, 5), (7, 15)),
            ((-10, -20), (-3, 5), (-7, -25)),
            ((0, -5), (5, -5), (-5, 0)),
            ((15000, -10000), (5000, 5000), (10000, -15000)),
            ((-100, 250), (200, -150), (-300, 400)),
        ]
        for (lo0, hi0), (lo1, hi1), (exp_lo, exp_hi) in cases:
            s0 = P(to_u16(lo0), to_u16(hi0))
            s1 = P(to_u16(lo1), to_u16(hi1))
            expected = P(to_u16(exp_lo), to_u16(exp_hi))
            with self.subTest(s0=hex(s0), s1=hex(s1)):
                got = run_pk_sub("v_pk_sub_i16 v0, v1, v2", {"v1": s0, "v2": s1})
                self.assertEqual(got, expected)

    def test_signed_overflow_and_wrap(self) -> None:
        """Signed 16-bit overflow wraps in two's complement modulo 65536 when unclamped."""
        # 32767 - (-1) = 32768 -> wraps to -32768 (0x8000)
        # -32768 - 1 = -32769 -> wraps to 32767 (0x7FFF)
        s0 = P(0x7FFF, 0x8000)
        s1 = P(to_u16(-1), 1)
        got = run_pk_sub("v_pk_sub_i16 v0, v1, v2", {"v1": s0, "v2": s1})
        self.assertEqual(got, P(0x8000, 0x7FFF))

        # Large negative overflow: -20000 - 20000 = -40000 -> wraps to 25536 (0x63C0)
        s0 = P(to_u16(-20000), 0)
        s1 = P(20000, 0)
        got = run_pk_sub("v_pk_sub_i16 v0, v1, v2", {"v1": s0, "v2": s1})
        self.assertEqual(got, P(0x63C0, 0))

    def test_clamp_modifier_signed_saturation(self) -> None:
        """Modifier clamp saturates signed overflows to [-32768, 32767]."""
        # 32767 - (-10) overflows to 32777 -> clamped to 32767 (0x7FFF)
        # -32768 - 10 underflows to -32778 -> clamped to -32768 (0x8000)
        s0 = P(0x7FFF, 0x8000)
        s1 = P(to_u16(-10), 10)
        got = run_pk_sub("v_pk_sub_i16 v0, v1, v2 clamp", {"v1": s0, "v2": s1})
        self.assertEqual(got, P(0x7FFF, 0x8000))

        # Values within range [-32768, 32767] remain unaffected by clamp
        s0 = P(to_u16(-100), 500)
        s1 = P(50, to_u16(-200))
        got = run_pk_sub("v_pk_sub_i16 v0, v1, v2 clamp", {"v1": s0, "v2": s1})
        self.assertEqual(got, P(to_u16(-150), to_u16(700)))

    def test_modifiers_op_sel_and_op_sel_hi(self) -> None:
        """Modifier op_sel and op_sel_hi routing half words into destinations."""
        # S0 = (lo=10, hi=-50), S1 = (lo=-3, hi=20)
        s0 = P(10, to_u16(-50))
        s1 = P(to_u16(-3), 20)

        # Default: lo = S0.lo (10) - S1.lo (-3) = 13 (0x000D)
        #          hi = S0.hi (-50) - S1.hi (20) = -70 (0xFFBA)
        got = run_pk_sub("v_pk_sub_i16 v0, v1, v2", {"v1": s0, "v2": s1})
        self.assertEqual(got, P(13, to_u16(-70)))

        # op_sel:[1, 0] -> lo = S0.hi (-50) - S1.lo (-3) = -47
        #                  hi = S0.hi (-50) - S1.hi (20) = -70
        got = run_pk_sub("v_pk_sub_i16 v0, v1, v2 op_sel:[1,0]", {"v1": s0, "v2": s1})
        self.assertEqual(got, P(to_u16(-47), to_u16(-70)))

        # op_sel:[0, 1] op_sel_hi:[0, 0] ->
        # lo = S0.lo (10) - S1.hi (20) = -10
        # hi = S0.lo (10) - S1.lo (-3) = 13
        got = run_pk_sub(
            "v_pk_sub_i16 v0, v1, v2 op_sel:[0,1] op_sel_hi:[0,0]",
            {"v1": s0, "v2": s1},
        )
        self.assertEqual(got, P(to_u16(-10), 13))

        # op_sel:[1, 1] op_sel_hi:[0, 0] ->
        # lo = S0.hi (-50) - S1.hi (20) = -70
        # hi = S0.lo (10) - S1.lo (-3) = 13
        got = run_pk_sub(
            "v_pk_sub_i16 v0, v1, v2 op_sel:[1,1] op_sel_hi:[0,0]",
            {"v1": s0, "v2": s1},
        )
        self.assertEqual(got, P(to_u16(-70), 13))

    def test_sgpr_and_aliasing(self) -> None:
        """Register variations: scalar SGPR sources and self-subtraction aliasing."""
        # Both SGPR sources: s1 - s2 -> v0
        s0 = P(to_u16(-100), 200)
        s1 = P(25, to_u16(-50))
        got = run_pk_sub("v_pk_sub_i16 v0, s1, s2", {"s1": s0, "s2": s1})
        self.assertEqual(got, P(to_u16(-125), to_u16(250)))

        # Mixed SGPR src0 and VGPR src1: s1 - v2 -> v0
        got = run_pk_sub("v_pk_sub_i16 v0, s1, v2", {"s1": s0, "v2": s1})
        self.assertEqual(got, P(to_u16(-125), to_u16(250)))

        # Register aliasing (subtracting self in-place): v0 = v0 - v0
        s0 = P(0x1234, 0x5678)
        got = run_pk_sub("v_pk_sub_i16 v0, v0, v0", {"v0": s0})
        self.assertEqual(got, P(0, 0))

    def test_spirv_structure(self) -> None:
        """Verify that translation produces proper SPIR-V signed integer subtraction and extraction."""
        prog = [
            "v_mov_b32 v1, 0x00320014",  # hi: 50, lo: 20
            "v_mov_b32 v2, 0x000a0005",  # hi: 10, lo: 5
            "v_pk_sub_i16 v0, v1, v2",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_pk_sub_i16_struct", prog)
        self.assertIn("OpISub %uint", spv)
        self.assertIn("OpBitFieldSExtract", spv)
        words = support.mrt0_words(spv)
        # (20 - 5 = 15 = 0x000F, 50 - 10 = 40 = 0x0028)
        self.assertEqual(words[0], P(15, 40))


if __name__ == "__main__":
    unittest.main()
