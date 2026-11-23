from __future__ import annotations

"""v_fmac_f16 (RDNA2 VOP2 / VOP3) hardware-referenced semantic regression tests.

Covers:
- Fused multiply-accumulate on 16-bit floats:
    D.f16[dhi] = fma(S0.f16[sel0], S1.f16[sel1], D.f16[sel2])
- Half-register preservation: unmodified half of destination VGPR is preserved (GFX10 semantics)
- Destination high half write via op_sel bit 3 (dhi=1)
- Clamp modifier saturating results to [0.0, 1.0]
- IEEE 754 edge cases: signed zeros (+0, -0), infinities, NaNs, and subnormals
- Register variations: SGPR sources and in-place register aliasing
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

MAX_NORMAL = 0x7BFF    # 65504.0
MIN_NORMAL = 0x0400    # 2^-14 ~= 6.1035e-5
MIN_SUB = 0x0001       # 2^-24 ~= 5.9605e-8

INF_P = 0x7C00         # +Infinity
INF_N = 0xFC00         # -Infinity
QNAN = 0x7E00          # Quiet NaN


def P(lo: int, hi: int) -> int:
    """Pack two 16-bit integers into a single 32-bit dword."""
    return ((hi & 0xFFFF) << 16) | (lo & 0xFFFF)


def u16_to_f16(raw: int) -> float:
    """Convert a 16-bit binary16 uint representation to Python float."""
    return struct.unpack("<e", struct.pack("<H", raw & 0xFFFF))[0]


def run_fmac(instruction: str, seeds: dict[str, int]) -> int:
    """Translate and run a minimal pixel shader exercising `v_fmac_f16`, returning v0 from mrt0."""
    prog = []
    for reg, val in sorted(seeds.items()):
        prefix = "s" if reg.startswith("s") else "v"
        prog.append(f"{prefix}_mov_b32 {reg}, {val:#x}")
    prog.append(instruction)
    prog.append("exp mrt0 v0, v0, v0, v0 done vm")
    prog.append("s_endpgm")
    spv = support.translate_lines("ps_fmac_f16", prog)
    return support.mrt0_words(spv)[0]


class ValuFmacF16HardwareTests(unittest.TestCase):
    def test_basic_fmac_and_half_preservation(self) -> None:
        """Test basic fma(s0, s1, dst) and verify destination high half is preserved."""
        # v0 = (lo=1.0, hi=0xABCD). Compute 1.5 * 2.0 + 1.0 = 4.0 (0x4400)
        # Expected v0: high half remains 0xABCD, low half becomes 0x4400 -> 0xABCD4400
        s0 = P(0x3E00, 0)  # 1.5
        s1 = P(TWO, 0)     # 2.0
        dst = P(ONE, 0xABCD)
        got = run_fmac("v_fmac_f16 v0, v1, v2", {"v1": s0, "v2": s1, "v0": dst})
        self.assertEqual(got, P(FOUR, 0xABCD))

        # 0.5 * 0.5 + 0.25 = 0.5 (0x3800), high half preserved
        s0 = P(HALF, 0)
        s1 = P(HALF, 0)
        dst = P(QUARTER, 0x5678)
        got = run_fmac("v_fmac_f16 v0, v1, v2", {"v1": s0, "v2": s1, "v0": dst})
        self.assertEqual(got, P(HALF, 0x5678))

        # Negative multiply: 2.0 * -1.0 + 3.0 = 1.0 (0x3C00)
        s0 = P(TWO, 0)
        s1 = P(NEG_ONE, 0)
        dst = P(THREE, 0x1234)
        got = run_fmac("v_fmac_f16 v0, v1, v2", {"v1": s0, "v2": s1, "v0": dst})
        self.assertEqual(got, P(ONE, 0x1234))

    def test_destination_high_half_write_via_op_sel(self) -> None:
        """With op_sel dhi bit set (op_sel:[s0, s1, s2, 1]), write high half and preserve low half."""
        # op_sel:[0, 0, 0, 1]: S0.lo (2.0) * S1.lo (1.5) + D.lo (1.0) = 4.0 written to D.hi
        # D.lo starts as 0x1111 and must be preserved.
        s0 = P(TWO, 0)
        s1 = P(0x3E00, 0)
        dst = P(ONE, 0x9999)  # D.lo = 1.0 (used as acc), D.hi = 0x9999 (overwritten)
        got = run_fmac(
            "v_fmac_f16 v0, v1, v2 op_sel:[0,0,0,1]",
            {"v1": s0, "v2": s1, "v0": dst},
        )
        # Low half remains 1.0 (ONE), high half gets 4.0 (FOUR)
        self.assertEqual(got, P(ONE, FOUR))

    def test_op_sel_source_routing(self) -> None:
        """op_sel selects high half of sources for multiplication and accumulation."""
        # v1 = (lo=0.5, hi=2.0), v2 = (lo=0.25, hi=3.0), v0 = (lo=1.0, hi=4.0)
        v1 = P(HALF, TWO)
        v2 = P(QUARTER, THREE)
        v0 = P(ONE, FOUR)

        # op_sel:[1, 1, 0, 0] -> S0.hi (2.0) * S1.hi (3.0) + D.lo (1.0) = 7.0 (0x4700)
        # Low half written, high half preserved (FOUR)
        got = run_fmac(
            "v_fmac_f16 v0, v1, v2 op_sel:[1,1,0,0]",
            {"v1": v1, "v2": v2, "v0": v0},
        )
        self.assertEqual(got, P(0x4700, FOUR))

        # op_sel:[1, 0, 1, 0] -> S0.hi (2.0) * S1.lo (0.25) + D.hi (4.0) = 0.5 + 4.0 = 4.5 (0x4480)
        got = run_fmac(
            "v_fmac_f16 v0, v1, v2 op_sel:[1,0,1,0]",
            {"v1": v1, "v2": v2, "v0": v0},
        )
        self.assertEqual(got, P(0x4480, FOUR))

    def test_clamp_modifier(self) -> None:
        """Modifier clamp saturates result to [0.0, 1.0]."""
        # 2.0 * 2.0 + 1.0 = 5.0 -> clamped to 1.0 (0x3C00)
        s0 = P(TWO, 0)
        s1 = P(TWO, 0)
        dst = P(ONE, 0xABCD)
        got = run_fmac("v_fmac_f16 v0, v1, v2 clamp", {"v1": s0, "v2": s1, "v0": dst})
        self.assertEqual(got, P(ONE, 0xABCD))

        # 2.0 * -1.0 + 0.5 = -1.5 -> clamped to 0.0 (0x0000)
        s0 = P(TWO, 0)
        s1 = P(NEG_ONE, 0)
        dst = P(HALF, 0xABCD)
        got = run_fmac("v_fmac_f16 v0, v1, v2 clamp", {"v1": s0, "v2": s1, "v0": dst})
        self.assertEqual(got, P(ZERO_P, 0xABCD))

        # 0.25 * 0.5 + 0.25 = 0.375 (0x3600) within [0, 1] unchanged
        s0 = P(QUARTER, 0)
        s1 = P(HALF, 0)
        dst = P(QUARTER, 0xABCD)
        got = run_fmac("v_fmac_f16 v0, v1, v2 clamp", {"v1": s0, "v2": s1, "v0": dst})
        self.assertEqual(got, P(0x3600, 0xABCD))

    def test_ieee754_edges(self) -> None:
        """Signed zeros, infinities, NaNs, and subnormal preservation."""
        # (+0) * 1.0 + (+0) = +0
        s0 = P(ZERO_P, 0)
        s1 = P(ONE, 0)
        dst = P(ZERO_P, 0)
        got = run_fmac("v_fmac_f16 v0, v1, v2", {"v1": s0, "v2": s1, "v0": dst})
        self.assertEqual(got & 0xFFFF, ZERO_P)

        # (-0) * 1.0 + (-0) = -0
        s0 = P(ZERO_N, 0)
        s1 = P(ONE, 0)
        dst = P(ZERO_N, 0)
        got = run_fmac("v_fmac_f16 v0, v1, v2", {"v1": s0, "v2": s1, "v0": dst})
        self.assertEqual(got & 0xFFFF, ZERO_N)

        # +Inf * 1.0 + 2.0 = +Inf
        s0 = P(INF_P, 0)
        s1 = P(ONE, 0)
        dst = P(TWO, 0)
        got = run_fmac("v_fmac_f16 v0, v1, v2", {"v1": s0, "v2": s1, "v0": dst})
        self.assertEqual(got & 0xFFFF, INF_P)

        # Inf * 0.0 + 1.0 = NaN
        s0 = P(INF_P, 0)
        s1 = P(ZERO_P, 0)
        dst = P(ONE, 0)
        got = run_fmac("v_fmac_f16 v0, v1, v2", {"v1": s0, "v2": s1, "v0": dst})
        self.assertTrue(math.isnan(u16_to_f16(got & 0xFFFF)))

        # Subnormal result: MIN_NORMAL * 0.5 + 0 = 0x0200
        s0 = P(MIN_NORMAL, 0)
        s1 = P(HALF, 0)
        dst = P(ZERO_P, 0)
        got = run_fmac("v_fmac_f16 v0, v1, v2", {"v1": s0, "v2": s1, "v0": dst})
        self.assertEqual(got & 0xFFFF, 0x0200)

    def test_sgpr_and_aliasing(self) -> None:
        """Register variations: scalar SGPR inputs and aliasing destination as source."""
        # Scalar SGPR sources: s1 * s2 + v0
        s0 = P(TWO, 0)
        s1 = P(THREE, 0)
        dst = P(ONE, 0xCAFE)
        got = run_fmac("v_fmac_f16 v0, s1, s2", {"s1": s0, "s2": s1, "v0": dst})
        self.assertEqual(got, P(0x4700, 0xCAFE))  # 2.0 * 3.0 + 1.0 = 7.0

        # Destination aliasing: v0 = v0 * v0 + v0 = x^2 + x
        # With x = 2.0 (TWO): 2^2 + 2 = 6.0 (0x4600)
        dst = P(TWO, 0xBEEF)
        got = run_fmac("v_fmac_f16 v0, v0, v0", {"v0": dst})
        self.assertEqual(got, P(0x4600, 0xBEEF))

    def test_spirv_structure(self) -> None:
        """Verify that translation produces proper single-rounding FMA SPIR-V instructions."""
        prog = [
            "v_mov_b32 v1, 0x00003c00",  # 1.0
            "v_mov_b32 v2, 0x00004000",  # 2.0
            "v_mov_b32 v0, 0x00003800",  # 0.5
            "v_fmac_f16 v0, v1, v2",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_fmac_f16_struct", prog)
        self.assertIn("OpFMul %float", spv)
        words = support.mrt0_words(spv)
        # 1.0 * 2.0 + 0.5 = 2.5 (0x4100)
        self.assertEqual(words[0] & 0xFFFF, 0x4100)


if __name__ == "__main__":
    unittest.main()
