from __future__ import annotations

"""Hardware-referenced semantic regression tests for RDNA2 SALU s_and_b32.

All tests in this suite compile real assembly listings through support.translate_lines(...)
and execute the translated SPIR-V in spirv_mini_interp via support.mrt0_words.

RDNA2 SALU s_and_b32 contract:
- Opcode: s_and_b32 D, S0, S1  (SOP2 format, 32-bit scalar bitwise AND)
- Semantics: D = S0 & S1
- Condition Code (SCC):
  * SCC = 1 if result != 0
  * SCC = 0 if result == 0
- Operands:
  * Destination: SGPR
  * Sources: SGPRs, inline constants (-16..64), 32-bit literal constants
  * Register aliasing: In-place destination execution (D == S0, D == S1, D == S0 == S1)
- Algebraic identities:
  * Commutativity: A & B == B & A
  * Idempotence: A & A == A
  * Annihilation by zero: A & 0 == 0
  * Identity by -1 (0xFFFFFFFF): A & ~0 == A
  * Complement annihilation: A & ~A == 0
"""

import os
import struct
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.dont_write_bytecode = True
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import support  # noqa: E402

MASK32: int = 0xFFFF_FFFF


def run_and_b32(instruction: str, seeds: dict[str, int]) -> tuple[int, int]:
    """Execute shader exercising `s_and_b32`, returning (result, scc)."""
    prog = []
    for reg, val in sorted(seeds.items()):
        prog.append(f"s_mov_b32 {reg}, {val & MASK32:#x}")
    prog.append(instruction)
    prog.append("s_cselect_b32 s2, 1, 0")
    prog.append("v_mov_b32 v0, s0")
    prog.append("v_mov_b32 v1, s2")
    prog.append("exp mrt0 v0, v1, v0, v1 done vm")
    prog.append("s_endpgm")
    spv = support.translate_lines("ps_and_b32", prog)
    words = support.mrt0_words(spv)
    return words[0], words[1]


# ==============================================================================
# Suite 1: Core Mathematical Operations & Algebraic Identities (7 tests)
# ==============================================================================

class TestSaluAndB32CoreOperations(unittest.TestCase):
    """Core bitwise AND operations and algebraic properties through translator."""

    def test_basic_bitwise_and(self) -> None:
        """Test standard 32-bit bitwise AND operations across varied patterns."""
        cases = [
            (0x12345678, 0x0F0F0F0F, 0x02040608),
            (0xAAAAAAAA, 0x55555555, 0x00000000),
            (0xFFFFFFFF, 0xCAFEBABE, 0xCAFEBABE),
            (0x00000000, 0xDEADBEEF, 0x00000000),
            (0x80000000, 0x80000000, 0x80000000),
            (0x0000FFFF, 0xFFFF0000, 0x00000000),
            (0x13579BDF, 0x2468ACE0, 0x004088C0),
        ]
        for s0_val, s1_val, expected in cases:
            with self.subTest(s0=hex(s0_val), s1=hex(s1_val)):
                res, scc = run_and_b32(
                    "s_and_b32 s0, s0, s1",
                    {"s0": s0_val, "s1": s1_val},
                )
                self.assertEqual(res, expected)
                expected_scc = 1 if expected != 0 else 0
                self.assertEqual(scc, expected_scc)

    def test_commutativity(self) -> None:
        """Commutativity: s_and_b32 s0, s1, s2 == s_and_b32 s0, s2, s1."""
        a, b = 0xFEEDBEEF, 0x01234567
        res1, scc1 = run_and_b32("s_and_b32 s0, s1, s3", {"s1": a, "s3": b})
        res2, scc2 = run_and_b32("s_and_b32 s0, s3, s1", {"s1": a, "s3": b})
        self.assertEqual(res1, res2)
        self.assertEqual(scc1, scc2)
        self.assertEqual(res1, a & b)

    def test_idempotence(self) -> None:
        """Idempotence: A & A == A."""
        patterns = [0x00000000, 0x1, 0xCAFEBABE, 0x7FFFFFFF, 0x80000000, 0xFFFFFFFF]
        for p in patterns:
            res, scc = run_and_b32("s_and_b32 s0, s1, s1", {"s1": p})
            self.assertEqual(res, p)
            self.assertEqual(scc, 1 if p != 0 else 0)

    def test_annihilation_with_zero(self) -> None:
        """Annihilation by zero: A & 0 == 0, SCC == 0."""
        patterns = [0x00000000, 0x12345678, 0x80000000, 0xFFFFFFFF]
        for p in patterns:
            res, scc = run_and_b32("s_and_b32 s0, s1, 0", {"s1": p})
            self.assertEqual(res, 0)
            self.assertEqual(scc, 0)

    def test_identity_with_all_ones(self) -> None:
        """Identity by all-ones (-1): A & 0xFFFFFFFF == A."""
        patterns = [0x00000000, 0x12345678, 0x80000000, 0xFFFFFFFF]
        for p in patterns:
            res, scc = run_and_b32("s_and_b32 s0, s1, -1", {"s1": p})
            self.assertEqual(res, p)
            self.assertEqual(scc, 1 if p != 0 else 0)

    def test_complement_annihilation(self) -> None:
        """Complement annihilation: A & ~A == 0, SCC == 0."""
        patterns = [0x00000000, 0x12345678, 0xAAAAAAAA, 0x55555555, 0xFFFFFFFF]
        for p in patterns:
            comp = (~p) & MASK32
            res, scc = run_and_b32("s_and_b32 s0, s1, s3", {"s1": p, "s3": comp})
            self.assertEqual(res, 0)
            self.assertEqual(scc, 0)

    def test_bit_walking_single_bits(self) -> None:
        """Bit-exact validation for individual bit positions 0..31."""
        for bit in range(32):
            val = 1 << bit
            # Matching bit produces val and SCC=1
            res_match, scc_match = run_and_b32("s_and_b32 s0, s1, s3", {"s1": val, "s3": val})
            self.assertEqual(res_match, val)
            self.assertEqual(scc_match, 1)

            # Non-matching bit (shifted by 1) produces 0 and SCC=0
            other_bit = 1 << ((bit + 1) % 32)
            res_diff, scc_diff = run_and_b32("s_and_b32 s0, s1, s3", {"s1": val, "s3": other_bit})
            self.assertEqual(res_diff, 0)
            self.assertEqual(scc_diff, 0)


# ==============================================================================
# Suite 2: Condition Code SCC Generation (5 tests)
# ==============================================================================

class TestSaluAndB32ConditionCodeSCC(unittest.TestCase):
    """Scalar condition code (SCC) generation and state transitions."""

    def test_scc_set_on_nonzero_result(self) -> None:
        """SCC flag is 1 for non-zero result."""
        res, scc = run_and_b32("s_and_b32 s0, s0, s1", {"s0": 1, "s1": 1})
        self.assertEqual(res, 1)
        self.assertEqual(scc, 1)

    def test_scc_cleared_on_zero_result(self) -> None:
        """SCC flag is 0 for zero result."""
        res, scc = run_and_b32("s_and_b32 s0, s0, s1", {"s0": 0x10, "s1": 0x20})
        self.assertEqual(res, 0)
        self.assertEqual(scc, 0)

    def test_scc_msb_sign_bit(self) -> None:
        """SCC is 1 when only the sign bit (bit 31) is set."""
        res, scc = run_and_b32("s_and_b32 s0, s0, s1", {"s0": 0x80000000, "s1": 0x80000000})
        self.assertEqual(res, 0x80000000)
        self.assertEqual(scc, 1)

    def test_scc_lsb_bit(self) -> None:
        """SCC is 1 when only bit 0 is set."""
        res, scc = run_and_b32("s_and_b32 s0, s0, s1", {"s0": 0x00000001, "s1": 0x00000001})
        self.assertEqual(res, 0x00000001)
        self.assertEqual(scc, 1)

    def test_scc_preservation_when_unmodified(self) -> None:
        """Verify s_cselect_b32 reflects SCC across consecutive operations."""
        prog = [
            "s_mov_b32 s0, 0x1",
            "s_mov_b32 s1, 0x1",
            "s_and_b32 s0, s0, s1",  # SCC = 1
            "s_cselect_b32 s2, 0xAA, 0xBB",  # s2 = 0xAA
            "v_mov_b32 v0, s2",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_and_scc_pres", prog)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], 0xAA)


# ==============================================================================
# Suite 3: Operand Types, Constants & Register Aliasing (6 tests)
# ==============================================================================

class TestSaluAndB32OperandsAndAliasing(unittest.TestCase):
    """SGPR operands, in-place aliasing, inline integers, and literals."""

    def test_register_aliasing_dst_src0(self) -> None:
        """Register aliasing D == S0 (in-place accumulation)."""
        res, scc = run_and_b32("s_and_b32 s0, s0, s1", {"s0": 0xFFFF00FF, "s1": 0x00FFFF00})
        self.assertEqual(res, 0x00FF0000)
        self.assertEqual(scc, 1)

    def test_register_aliasing_dst_src1(self) -> None:
        """Register aliasing D == S1."""
        res, scc = run_and_b32("s_and_b32 s0, s1, s0", {"s0": 0x12345678, "s1": 0x0000FFFF})
        self.assertEqual(res, 0x00005678)
        self.assertEqual(scc, 1)

    def test_register_aliasing_dst_src0_src1(self) -> None:
        """Self-AND aliasing: D == S0 == S1."""
        val = 0x87654321
        res, scc = run_and_b32("s_and_b32 s0, s0, s0", {"s0": val})
        self.assertEqual(res, val)
        self.assertEqual(scc, 1)

    def test_inline_constants(self) -> None:
        """Bitwise AND with small inline constants (-16..64)."""
        # Inline 0
        res0, scc0 = run_and_b32("s_and_b32 s0, s0, 0", {"s0": 0xFFFFFFFF})
        self.assertEqual(res0, 0)
        self.assertEqual(scc0, 0)

        # Inline 1
        res1, scc1 = run_and_b32("s_and_b32 s0, s0, 1", {"s0": 0xFFFFFFFF})
        self.assertEqual(res1, 1)
        self.assertEqual(scc1, 1)

        # Inline -1 (0xFFFFFFFF)
        res_neg, scc_neg = run_and_b32("s_and_b32 s0, s0, -1", {"s0": 0x12345678})
        self.assertEqual(res_neg, 0x12345678)
        self.assertEqual(scc_neg, 1)

        # Inline 64
        res64, scc64 = run_and_b32("s_and_b32 s0, s0, 64", {"s0": 0xFFFFFFFF})
        self.assertEqual(res64, 64)
        self.assertEqual(scc64, 1)

    def test_literal_constant_operands(self) -> None:
        """Bitwise AND with 32-bit literal constants."""
        res, scc = run_and_b32("s_and_b32 s0, s0, 0x000000FF", {"s0": 0x12345678})
        self.assertEqual(res, 0x78)
        self.assertEqual(scc, 1)

        res2, scc2 = run_and_b32("s_and_b32 s0, s0, 0xF0F0F0F0", {"s0": 0x12345678})
        self.assertEqual(res2, 0x10305070)
        self.assertEqual(scc2, 1)

    def test_chained_and_operations(self) -> None:
        """Chained execution: s0 = (s0 & s1) & s2."""
        prog = [
            "s_mov_b32 s0, 0x0F0F0F0F",
            "s_mov_b32 s1, 0x00FF00FF",
            "s_mov_b32 s2, 0x000F000F",
            "s_and_b32 s0, s0, s1",
            "s_and_b32 s0, s0, s2",
            "v_mov_b32 v0, s0",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_and_chain", prog)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], 0x000F000F)


# ==============================================================================
# Suite 4: SPIR-V Translation & Mutation Detection (4 tests)
# ==============================================================================

class TestSaluAndB32TranslatorEmissionAndMutations(unittest.TestCase):
    """SPIR-V translation emission checks, distinction controls, and mutation detection."""

    def test_spirv_emission_structure(self) -> None:
        """Verify that translation properly folds and executes bitwise AND operations."""
        prog = [
            "s_mov_b32 s0, 0x0f0f0f0f",
            "s_mov_b32 s1, 0x00ff00ff",
            "s_and_b32 s0, s0, s1",
            "v_mov_b32 v0, s0",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_and_b32_struct", prog)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], 0x000F000F)

    def test_negative_control_or_xor_distinction(self) -> None:
        """Ensure bitwise AND result is strictly distinguished from OR and XOR."""
        a, b = 0x11110000, 0x00002222
        and_res, and_scc = run_and_b32("s_and_b32 s0, s1, s2", {"s1": a, "s2": b})
        self.assertEqual(and_res, 0x00000000)
        self.assertEqual(and_scc, 0)

        # OR would yield 0x11112222, XOR would yield 0x11112222
        or_res = a | b
        xor_res = a ^ b
        self.assertNotEqual(and_res, or_res)
        self.assertNotEqual(and_res, xor_res)

    def test_mutation_detection_bitwise_op(self) -> None:
        """Mutation detection: replacing AND with OR or XOR causes detection."""
        a, b = 0xAAAA5555, 0x5555AAAA
        res, scc = run_and_b32("s_and_b32 s0, s1, s2", {"s1": a, "s2": b})
        self.assertEqual(res, 0x00000000)
        self.assertEqual(scc, 0)

        # Mutant OR produces 0xFFFFFFFF
        mutant_or = a | b
        self.assertNotEqual(res, mutant_or)

    def test_mutation_detection_scc_logic(self) -> None:
        """Mutation detection: inverted SCC assignment (result == 0 -> SCC=1) is rejected."""
        res_nonzero, scc_nonzero = run_and_b32("s_and_b32 s0, s1, s2", {"s1": 1, "s2": 1})
        self.assertEqual(scc_nonzero, 1)

        res_zero, scc_zero = run_and_b32("s_and_b32 s0, s1, s2", {"s1": 1, "s2": 2})
        self.assertEqual(scc_zero, 0)

        # Mutated logic: scc == 0 for nonzero, 1 for zero
        mutated_scc_nonzero = 0
        mutated_scc_zero = 1
        self.assertNotEqual(scc_nonzero, mutated_scc_nonzero)
        self.assertNotEqual(scc_zero, mutated_scc_zero)


if __name__ == "__main__":
    unittest.main()
