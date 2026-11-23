"""Hardware-semantic translator-backed test suite for RDNA2 v_add3_u32.

Models RDNA2 / GFX10 VALU 3-operand 32-bit unsigned addition:
  v_add3_u32 vd, src0, src1, src2

RDNA2 Architecture & Semantics:
  1. Computation:
     D.u32 = (S0.u32 + S1.u32 + S2.u32) mod 2^32
  2. Modulo wrapping:
     Can overflow once or twice (sum up to 3 * (2^32 - 1) = 3 * 2^32 - 3), wrapping cleanly modulo 2^32.
  3. Operands:
     - vd: Destination VGPR
     - src0, src1, src2: VGPR, SGPR, or literal/inline constant (VOP3 rules apply).
  4. Condition Codes:
     - Neither SCC nor VCC is modified by v_add3_u32.
  5. Translator Emission:
     - Lowers to chained OpIAdd instructions in SPIR-V.

Standard library Python only; zero third-party dependencies; zero game data.
Python 3.9+ compatible (from __future__ import annotations).
"""
from __future__ import annotations

import os
import struct
import sys
import unittest
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")

sys.dont_write_bytecode = True
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

try:
    import rdna2spirv  # noqa: E402
    import support  # noqa: E402
except Exception:  # noqa: E722
    rdna2spirv = None  # type: ignore
    support = None  # type: ignore

MASK32: int = 0xFFFFFFFF
MASK64: int = 0xFFFFFFFFFFFFFFFF


def ref_add3_u32(a: int, b: int, c: int) -> int:
    """Pure mathematical reference oracle for RDNA2 v_add3_u32.

    Computes: (a + b + c) mod 2^32
    """
    return ((a & MASK32) + (b & MASK32) + (c & MASK32)) & MASK32


def ref_add3_u32_with_carry(a: int, b: int, c: int) -> tuple[int, int]:
    """Returns (sum & MASK32, carry_out in {0, 1, 2})."""
    total = (a & MASK32) + (b & MASK32) + (c & MASK32)
    return total & MASK32, total >> 32


class TestValuAdd3U32Oracle(unittest.TestCase):
    """Unit tests for reference mathematical oracle and algebraic properties."""

    def test_basic_addition(self) -> None:
        self.assertEqual(ref_add3_u32(1, 2, 3), 6)
        self.assertEqual(ref_add3_u32(100, 200, 300), 600)
        self.assertEqual(ref_add3_u32(0, 0, 0), 0)

    def test_identity_zero_operand(self) -> None:
        val = 0xDEADBEEF
        self.assertEqual(ref_add3_u32(val, 0, 0), val)
        self.assertEqual(ref_add3_u32(0, val, 0), val)
        self.assertEqual(ref_add3_u32(0, 0, val), val)

    def test_single_overflow_wrapping(self) -> None:
        # a + b >= 2^32
        res, carry = ref_add3_u32_with_carry(0xFFFFFFFF, 1, 5)
        self.assertEqual(res, 5)
        self.assertEqual(carry, 1)

        # a + b + c >= 2^32 on final stage
        res2, carry2 = ref_add3_u32_with_carry(10, 20, 0xFFFFFFF0)
        self.assertEqual(res2, 0x0E)
        self.assertEqual(carry2, 1)

    def test_double_overflow_wrapping(self) -> None:
        # 3 * 0xFFFFFFFF = 0x2FFFFFFFD -> sum wraps to 0xFFFFFFFD, carry is 2
        res, carry = ref_add3_u32_with_carry(0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF)
        self.assertEqual(res, 0xFFFFFFFD)
        self.assertEqual(carry, 2)

    def test_commutativity_all_permutations(self) -> None:
        v0, v1, v2 = 0x12345678, 0x9ABCDEF0, 0x5555AAAA
        expected = ref_add3_u32(v0, v1, v2)
        permutations = [
            (v0, v1, v2),
            (v0, v2, v1),
            (v1, v0, v2),
            (v1, v2, v0),
            (v2, v0, v1),
            (v2, v1, v0),
        ]
        for a, b, c in permutations:
            self.assertEqual(ref_add3_u32(a, b, c), expected)

    def test_boundaries_and_extrema(self) -> None:
        extrema = [0x00000000, 0x00000001, 0x7FFFFFFF, 0x80000000, 0xFFFFFFFE, 0xFFFFFFFF]
        for a in extrema:
            for b in extrema:
                for c in (0, 1, 0xFFFFFFFF):
                    expected = ((a + b + c) & MASK32)
                    self.assertEqual(ref_add3_u32(a, b, c), expected)


class TestValuAdd3U32TranslationAndExecution(unittest.TestCase):
    """End-to-end SPIR-V translation and fragment execution via support and spirv_mini_interp."""

    def test_translator_v_add3_u32_all_vgprs(self) -> None:
        if support is None or rdna2spirv is None:
            self.skipTest("support/rdna2spirv not available under current Python environment")

        asm = [
            "v_mov_b32 v0, 0x10",
            "v_mov_b32 v1, 0x20",
            "v_mov_b32 v2, 0x30",
            "v_add3_u32 v3, v0, v1, v2",
            "exp mrt0 v3, v3, v3, v3 done vm",
            "s_endpgm",
        ]
        spvasm = support.translate_lines("ps_add3_vgprs", asm)
        self.assertIn("OpIAdd", spvasm)

        words = support.mrt0_words(spvasm)
        self.assertEqual(words, [0x60, 0x60, 0x60, 0x60])

    def test_translator_v_add3_u32_double_modulo_wrap(self) -> None:
        if support is None or rdna2spirv is None:
            self.skipTest("support/rdna2spirv not available under current Python environment")

        asm = [
            "v_mov_b32 v0, 0xffffffff",
            "v_mov_b32 v1, 0xffffffff",
            "v_mov_b32 v2, 0xffffffff",
            "v_add3_u32 v3, v0, v1, v2",
            "exp mrt0 v3, v3, v3, v3 done vm",
            "s_endpgm",
        ]
        spvasm = support.translate_lines("ps_add3_double_wrap", asm)
        words = support.mrt0_words(spvasm)
        self.assertEqual(words, [0xFFFFFFFD, 0xFFFFFFFD, 0xFFFFFFFD, 0xFFFFFFFD])

    def test_translator_v_add3_u32_mixed_sgpr_and_imm(self) -> None:
        if support is None or rdna2spirv is None:
            self.skipTest("support/rdna2spirv not available under current Python environment")

        asm = [
            "s_mov_b32 s0, 100",
            "v_mov_b32 v0, 50",
            "v_add3_u32 v1, s0, v0, 25",
            "exp mrt0 v1, v1, v1, v1 done vm",
            "s_endpgm",
        ]
        spvasm = support.translate_lines("ps_add3_sgpr_imm", asm)
        words = support.mrt0_words(spvasm)
        self.assertEqual(words, [175, 175, 175, 175])

    def test_translator_v_add3_u32_aliased_destination(self) -> None:
        if support is None or rdna2spirv is None:
            self.skipTest("support/rdna2spirv not available under current Python environment")

        asm = [
            "v_mov_b32 v0, 10",
            "v_mov_b32 v1, 20",
            "v_add3_u32 v0, v0, v1, 5",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spvasm = support.translate_lines("ps_add3_alias", asm)
        words = support.mrt0_words(spvasm)
        self.assertEqual(words, [35, 35, 35, 35])

    def test_chained_opiadd_structure(self) -> None:
        if support is None or rdna2spirv is None:
            self.skipTest("support/rdna2spirv not available under current Python environment")

        asm = [
            "v_mov_b32 v0, 1",
            "v_mov_b32 v1, 2",
            "v_mov_b32 v2, 3",
            "v_add3_u32 v3, v0, v1, v2",
            "exp mrt0 v3, v3, v3, v3 done vm",
            "s_endpgm",
        ]
        spvasm = support.translate_lines("ps_add3_structure", asm)
        self.assertGreaterEqual(spvasm.count("OpIAdd"), 2)


if __name__ == "__main__":
    unittest.main()
