from __future__ import annotations

"""Hardware semantic tests for RDNA2 SALU s_andn2_b64 through tools/rdna2spirv.py.

Instruction:
  s_andn2_b64 D, S0, S1  (SOP2 format, 64-bit scalar logic)

Architectural Semantics:
  D = S0 & (~S1)
  SCC = (D != 0)

Key architectural properties:
  - Non-commutative bitwise AND with inverted second operand (S1 is negated).
  - Operates on 64-bit register pairs (two adjacent 32-bit dwords).
  - Evaluates both source dwords prior to writing destination, ensuring in-place
    operand preservation even when D aliases S0 or S1.
  - Updates condition code SCC to reflect whether the full 64-bit result is non-zero.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import support  # noqa: E402

MASK32 = 0xFFFFFFFF
MASK64 = 0xFFFFFFFFFFFFFFFF


def ref_andn2_b64(s0: int, s1: int) -> int:
    """Exact architectural reference oracle: D = S0 & (~S1)."""
    return (s0 & (~s1 & MASK64)) & MASK64


class SaluAndn2B64HWTests(unittest.TestCase):
    """Semantic tests executing s_andn2_b64 through rdna2spirv and validating via mrt0."""

    def test_basic_andn2_execution(self) -> None:
        """Verify s_andn2_b64 on distinct non-trivial 64-bit operands and SCC=1."""
        s0_lo, s0_hi = 0x12345678, 0x9ABCDEF0
        s1_lo, s1_hi = 0x00FF00FF, 0x0F0F0F0F

        # Expected:
        # lo = 0x12345678 & ~0x00FF00FF = 0x12005600
        # hi = 0x9ABCDEF0 & ~0x0F0F0F0F = 0x90B0D0F0
        expected_lo = (s0_lo & (~s1_lo & MASK32)) & MASK32
        expected_hi = (s0_hi & (~s1_hi & MASK32)) & MASK32

        lines = [
            f"s_mov_b32 s0, {s0_lo:#x}",
            f"s_mov_b32 s1, {s0_hi:#x}",
            f"s_mov_b32 s2, {s1_lo:#x}",
            f"s_mov_b32 s3, {s1_hi:#x}",
            "s_andn2_b64 s[4:5], s[0:1], s[2:3]",
            "s_cselect_b32 s6, 1, 0",
            "v_mov_b32_e32 v0, s4",
            "v_mov_b32_e32 v1, s5",
            "v_mov_b32_e32 v2, s6",
            "v_mov_b32_e32 v3, 0",
            "exp mrt0, v0, v1, v2, v3 done",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_andn2_basic", lines)
        words = support.mrt0_words(spv)

        self.assertEqual(words[0], expected_lo)
        self.assertEqual(words[1], expected_hi)
        self.assertEqual(words[2], 1)  # SCC set because result is non-zero
        self.assertEqual(words[3], 0)

    def test_zero_result_scc_cleared(self) -> None:
        """Verify that when D == 0, SCC is cleared (0)."""
        s0_lo, s0_hi = 0x12345678, 0x9ABCDEF0
        s1_lo, s1_hi = 0xFFFFFFFF, 0xFFFFFFFF  # ~S1 = 0, so D = 0

        lines = [
            f"s_mov_b32 s0, {s0_lo:#x}",
            f"s_mov_b32 s1, {s0_hi:#x}",
            f"s_mov_b32 s2, {s1_lo:#x}",
            f"s_mov_b32 s3, {s1_hi:#x}",
            "s_andn2_b64 s[4:5], s[0:1], s[2:3]",
            "s_cselect_b32 s6, 1, 0",
            "v_mov_b32_e32 v0, s4",
            "v_mov_b32_e32 v1, s5",
            "v_mov_b32_e32 v2, s6",
            "v_mov_b32_e32 v3, 0",
            "exp mrt0, v0, v1, v2, v3 done",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_andn2_zero", lines)
        words = support.mrt0_words(spv)

        self.assertEqual(words[0], 0)
        self.assertEqual(words[1], 0)
        self.assertEqual(words[2], 0)  # SCC cleared because D == 0
        self.assertEqual(words[3], 0)

    def test_destination_aliasing_s0(self) -> None:
        """Verify in-place execution when destination D aliases S0: s_andn2_b64 s[0:1], s[0:1], s[2:3]."""
        s0_lo, s0_hi = 0x55555555, 0xAAAAAAAA
        s1_lo, s1_hi = 0x0000FFFF, 0xFFFF0000

        expected_lo = (s0_lo & (~s1_lo & MASK32)) & MASK32
        expected_hi = (s0_hi & (~s1_hi & MASK32)) & MASK32

        lines = [
            f"s_mov_b32 s0, {s0_lo:#x}",
            f"s_mov_b32 s1, {s0_hi:#x}",
            f"s_mov_b32 s2, {s1_lo:#x}",
            f"s_mov_b32 s3, {s1_hi:#x}",
            "s_andn2_b64 s[0:1], s[0:1], s[2:3]",
            "s_cselect_b32 s6, 1, 0",
            "v_mov_b32_e32 v0, s0",
            "v_mov_b32_e32 v1, s1",
            "v_mov_b32_e32 v2, s6",
            "v_mov_b32_e32 v3, 0",
            "exp mrt0, v0, v1, v2, v3 done",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_andn2_alias_s0", lines)
        words = support.mrt0_words(spv)

        self.assertEqual(words[0], expected_lo)
        self.assertEqual(words[1], expected_hi)
        self.assertEqual(words[2], 1)

    def test_destination_aliasing_s1(self) -> None:
        """Verify in-place execution when destination D aliases S1: s_andn2_b64 s[2:3], s[0:1], s[2:3]."""
        s0_lo, s0_hi = 0x12345678, 0x9ABCDEF0
        s1_lo, s1_hi = 0x00FF00FF, 0x0F0F0F0F

        expected_lo = (s0_lo & (~s1_lo & MASK32)) & MASK32
        expected_hi = (s0_hi & (~s1_hi & MASK32)) & MASK32

        lines = [
            f"s_mov_b32 s0, {s0_lo:#x}",
            f"s_mov_b32 s1, {s0_hi:#x}",
            f"s_mov_b32 s2, {s1_lo:#x}",
            f"s_mov_b32 s3, {s1_hi:#x}",
            "s_andn2_b64 s[2:3], s[0:1], s[2:3]",
            "s_cselect_b32 s6, 1, 0",
            "v_mov_b32_e32 v0, s2",
            "v_mov_b32_e32 v1, s3",
            "v_mov_b32_e32 v2, s6",
            "v_mov_b32_e32 v3, 0",
            "exp mrt0, v0, v1, v2, v3 done",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_andn2_alias_s1", lines)
        words = support.mrt0_words(spv)

        self.assertEqual(words[0], expected_lo)
        self.assertEqual(words[1], expected_hi)
        self.assertEqual(words[2], 1)

    def test_non_commutative_asymmetry(self) -> None:
        """Verify that s_andn2_b64 is non-commutative: (A & ~B) != (B & ~A)."""
        a_lo, a_hi = 0xF0F0F0F0, 0x0F0F0F0F
        b_lo, b_hi = 0x33333333, 0xCCCCCCCC

        # Order 1: s_andn2_b64 s[4:5], s[0:1], s[2:3]  (A & ~B)
        lines1 = [
            f"s_mov_b32 s0, {a_lo:#x}",
            f"s_mov_b32 s1, {a_hi:#x}",
            f"s_mov_b32 s2, {b_lo:#x}",
            f"s_mov_b32 s3, {b_hi:#x}",
            "s_andn2_b64 s[4:5], s[0:1], s[2:3]",
            "v_mov_b32_e32 v0, s4",
            "v_mov_b32_e32 v1, s5",
            "v_mov_b32_e32 v2, 0",
            "v_mov_b32_e32 v3, 0",
            "exp mrt0, v0, v1, v2, v3 done",
            "s_endpgm",
        ]
        spv1 = support.translate_lines("ps_andn2_order1", lines1)
        words1 = support.mrt0_words(spv1)

        # Order 2: s_andn2_b64 s[4:5], s[2:3], s[0:1]  (B & ~A)
        lines2 = [
            f"s_mov_b32 s0, {a_lo:#x}",
            f"s_mov_b32 s1, {a_hi:#x}",
            f"s_mov_b32 s2, {b_lo:#x}",
            f"s_mov_b32 s3, {b_hi:#x}",
            "s_andn2_b64 s[4:5], s[2:3], s[0:1]",
            "v_mov_b32_e32 v0, s4",
            "v_mov_b32_e32 v1, s5",
            "v_mov_b32_e32 v2, 0",
            "v_mov_b32_e32 v3, 0",
            "exp mrt0, v0, v1, v2, v3 done",
            "s_endpgm",
        ]
        spv2 = support.translate_lines("ps_andn2_order2", lines2)
        words2 = support.mrt0_words(spv2)

        self.assertNotEqual(words1[:2], words2[:2])

    def test_spirv_instruction_emission(self) -> None:
        """Verify that support.translate_lines emits OpBitwiseAnd, OpNot, and SCC logic."""
        lines = [
            "s_mov_b32 s0, 0x1",
            "s_mov_b32 s1, 0x0",
            "s_mov_b32 s2, 0x2",
            "s_mov_b32 s3, 0x0",
            "s_andn2_b64 s[4:5], s[0:1], s[2:3]",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_andn2_emit", lines)
        self.assertIn("OpBitwiseAnd", spv)
        self.assertIn("OpNot", spv)


if __name__ == "__main__":
    unittest.main()
