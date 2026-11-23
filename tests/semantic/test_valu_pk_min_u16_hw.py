from __future__ import annotations

"""v_pk_min_u16 (RDNA2 VOP3P) hardware-referenced semantic regression tests.

Covers:
- Packed 16-bit unsigned integer minimum across low and high halves:
    D.u16[0] = min(S0.u16[op_sel[0]], S1.u16[op_sel[1]])
    D.u16[1] = min(S0.u16[op_sel_hi[0]], S1.u16[op_sel_hi[1]])
- Unsigned integer comparison (e.g. 0x8000 = 32768 > 0x7FFF = 32767)
- Boundary extremes: 0 and 0xFFFF
- Modifiers: op_sel and op_sel_hi half routing
- Register variations: SGPR sources and in-place register aliasing
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


def run_pk_min(instruction: str, seeds: dict[str, int], old: int = 0xDEADBEEF) -> int:
    """Translate and run a minimal pixel shader exercising `v_pk_min_u16`, returning v0 from mrt0."""
    prog = [f"v_mov_b32 v0, {old:#x}"]
    for reg, val in sorted(seeds.items()):
        prefix = "s" if reg.startswith("s") else "v"
        prog.append(f"{prefix}_mov_b32 {reg}, {val:#x}")
    prog.append(instruction)
    prog.append("exp mrt0 v0, v0, v0, v0 done vm")
    prog.append("s_endpgm")
    spv = support.translate_lines("ps_pk_min_u16", prog)
    return support.mrt0_words(spv)[0]


class ValuPkMinU16HardwareTests(unittest.TestCase):
    def test_basic_unsigned_min_both_lanes(self) -> None:
        """Test independent lane unsigned minimum."""
        cases = [
            # ((lo0, hi0), (lo1, hi1), (exp_lo, exp_hi))
            ((10, 300), (20, 150), (10, 150)),
            ((500, 20), (100, 80), (100, 20)),
            ((0x8000, 0x1234), (0x7FFF, 0x5678), (0x7FFF, 0x1234)),  # 32767 < 32768 unsigned
            ((1, 65534), (2, 65535), (1, 65534)),
            ((0x4000, 0x9000), (0x4000, 0x9000), (0x4000, 0x9000)),
        ]
        for (lo0, hi0), (lo1, hi1), (exp_lo, exp_hi) in cases:
            s0 = P(lo0, hi0)
            s1 = P(lo1, hi1)
            expected = P(exp_lo, exp_hi)
            with self.subTest(s0=hex(s0), s1=hex(s1)):
                got = run_pk_min("v_pk_min_u16 v0, v1, v2", {"v1": s0, "v2": s1})
                self.assertEqual(got, expected)

    def test_boundary_extremes(self) -> None:
        """Unsigned extremes with 0 and 0xFFFF."""
        # min(0, 0xFFFF) = 0
        s0 = P(0, 0xFFFF)
        s1 = P(0xFFFF, 0)
        got = run_pk_min("v_pk_min_u16 v0, v1, v2", {"v1": s0, "v2": s1})
        self.assertEqual(got, P(0, 0))

        # min(0xFFFF, 0xFFFF) = 0xFFFF, min(0, 0) = 0
        s0 = P(0xFFFF, 0)
        s1 = P(0xFFFF, 0)
        got = run_pk_min("v_pk_min_u16 v0, v1, v2", {"v1": s0, "v2": s1})
        self.assertEqual(got, P(0xFFFF, 0))

    def test_modifiers_op_sel_and_op_sel_hi(self) -> None:
        """Modifier op_sel and op_sel_hi routing half words into operands."""
        # S0 = (lo=10, hi=500), S1 = (lo=20, hi=50)
        s0 = P(10, 500)
        s1 = P(20, 50)

        # Default: lo = min(10, 20) = 10, hi = min(500, 50) = 50
        got = run_pk_min("v_pk_min_u16 v0, v1, v2", {"v1": s0, "v2": s1})
        self.assertEqual(got, P(10, 50))

        # op_sel:[1, 0] -> lo = min(S0.hi=500, S1.lo=20) = 20, hi = min(500, 50) = 50
        got = run_pk_min("v_pk_min_u16 v0, v1, v2 op_sel:[1,0]", {"v1": s0, "v2": s1})
        self.assertEqual(got, P(20, 50))

        # op_sel:[0, 1] op_sel_hi:[0, 0] ->
        # lo = min(S0.lo=10, S1.hi=50) = 10
        # hi = min(S0.lo=10, S1.lo=20) = 10
        got = run_pk_min(
            "v_pk_min_u16 v0, v1, v2 op_sel:[0,1] op_sel_hi:[0,0]",
            {"v1": s0, "v2": s1},
        )
        self.assertEqual(got, P(10, 10))

        # op_sel:[1, 1] op_sel_hi:[0, 0] ->
        # lo = min(S0.hi=500, S1.hi=50) = 50
        # hi = min(S0.lo=10, S1.lo=20) = 10
        got = run_pk_min(
            "v_pk_min_u16 v0, v1, v2 op_sel:[1,1] op_sel_hi:[0,0]",
            {"v1": s0, "v2": s1},
        )
        self.assertEqual(got, P(50, 10))

    def test_sgpr_and_aliasing(self) -> None:
        """Register variations: scalar SGPR inputs and aliasing."""
        # Both SGPR sources
        s0 = P(100, 200)
        s1 = P(50, 400)
        got = run_pk_min("v_pk_min_u16 v0, s1, s2", {"s1": s0, "s2": s1})
        self.assertEqual(got, P(50, 200))

        # Mixed SGPR src0 and VGPR src1
        got = run_pk_min("v_pk_min_u16 v0, s1, v2", {"s1": s0, "v2": s1})
        self.assertEqual(got, P(50, 200))

        # Register aliasing (min of self is idempotent)
        s0 = P(0x1234, 0x5678)
        got = run_pk_min("v_pk_min_u16 v0, v0, v0", {"v0": s0})
        self.assertEqual(got, s0)

    def test_spirv_structure(self) -> None:
        """Verify that translation produces OpExtInst %glsl UMin."""
        prog = [
            "v_mov_b32 v1, 0x0064000a",  # hi: 100, lo: 10
            "v_mov_b32 v2, 0x00320014",  # hi: 50, lo: 20
            "v_pk_min_u16 v0, v1, v2",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_pk_min_u16_struct", prog)
        self.assertIn("UMin", spv)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], P(10, 50))


if __name__ == "__main__":
    unittest.main()
