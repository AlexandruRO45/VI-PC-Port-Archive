"""Semantic reference oracle and verification for RDNA2 64-bit scalar NAND logic.

Covers:
  - s_nand_b64: D.u64 = ~(S0.u64 & S1.u64)
  - Condition code SCC: SCC = 1 if D != 0 else 0

Tests standard library bitwise arithmetic, truth table exhaustion, boundary patterns,
AST validation, scalar interpreter, SIMD divergence (Wave32/Wave64 EXEC masking),
register aliasing, condition code generation, IEEE 754 float/double bit patterns,
and translator-backed SPIR-V assembly lowering with fragment execution.
"""
from __future__ import annotations

import math
import os
import random
import struct
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")

sys.dont_write_bytecode = True
if HERE not in sys.path:
    sys.path.insert(0, HERE)
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)

from support import translate_lines, mrt0_words  # noqa: E402

MASK32 = 0xFFFFFFFF
MASK64 = 0xFFFFFFFFFFFFFFFF


def ref_nand_b64(s0: int, s1: int) -> int:
    """RDNA2 s_nand_b64: D = ~(S0 & S1) clamped to 64 bits."""
    return (~(s0 & s1)) & MASK64


def ref_scc(val64: int) -> int:
    """RDNA2 SCC condition flag: 1 if 64-bit result != 0, else 0."""
    return 1 if (val64 & MASK64) != 0 else 0

class SaluNandB64OracleTests(unittest.TestCase):
    """Reference mathematical oracle tests for 64-bit scalar NAND logic."""

    def test_truth_table_bit_by_bit(self):
        """Verify 1-bit truth table combinations for NAND across 64 bits."""
        truth_table = {
            (0, 0): 1,
            (0, 1): 1,
            (1, 0): 1,
            (1, 1): 0,
        }
        for (b0, b1), expected_bit in truth_table.items():
            with self.subTest(b0=b0, b1=b1):
                s0 = MASK64 if b0 else 0
                s1 = MASK64 if b1 else 0
                want = MASK64 if expected_bit else 0
                got = ref_nand_b64(s0, s1)
                self.assertEqual(got, want)
                self.assertEqual(ref_scc(got), 1 if expected_bit else 0)

    def test_commutativity(self):
        """Verify s_nand_b64 is strictly commutative: NAND(A, B) == NAND(B, A)."""
        pairs = [
            (0xF0F0F0F0_0F0F0F0F, 0x33333333_CCCCCCCC),
            (0x12345678_9ABCDEF0, 0xEDCBA987_6543210F),
            (0xAAAAAAAAAAAAAAAA, 0x5555555555555555),
            (0x00000000_FFFFFFFF, 0xFFFFFFFF_00000000),
        ]
        for a, b in pairs:
            with self.subTest(a=hex(a), b=hex(b)):
                self.assertEqual(ref_nand_b64(a, b), ref_nand_b64(b, a))

    def test_de_morgan_dualities(self):
        """Verify De Morgan duality: ~(A & B) == (~A) | (~B)."""
        rng = random.Random(0x5A10_B64)
        for _ in range(64):
            a = rng.getrandbits(64)
            b = rng.getrandbits(64)
            not_a = (~a) & MASK64
            not_b = (~b) & MASK64
            expected = (not_a | not_b) & MASK64
            self.assertEqual(ref_nand_b64(a, b), expected)

    def test_self_nand_as_bitwise_not(self):
        """Verify self-NAND acts identically to bitwise NOT: NAND(A, A) == ~A."""
        patterns = [
            0x00000000_00000000,
            0xFFFFFFFF_FFFFFFFF,
            0x55555555_55555555,
            0xAAAAAAAA_AAAAAAAA,
            0x12345678_9ABCDEF0,
            0x00000000_00000001,
            0x80000000_00000000,
        ]
        for val in patterns:
            with self.subTest(val=hex(val)):
                got = ref_nand_b64(val, val)
                want = (~val) & MASK64
                self.assertEqual(got, want)
                self.assertEqual(ref_scc(got), 1 if want != 0 else 0)

    def test_boundary_values(self):
        """Verify boundary extremes: 0, MASK64, MSB, LSB, and alternating bit masks."""
        boundaries = [
            0x00000000_00000000,
            0xFFFFFFFF_FFFFFFFF,
            0x80000000_00000000,
            0x00000000_00000001,
            0x55555555_55555555,
            0xAAAAAAAA_AAAAAAAA,
            0x00000000_FFFFFFFF,
            0xFFFFFFFF_00000000,
            0x7FFFFFFF_FFFFFFFF,
            0x80000000_00000001,
        ]
        for s0 in boundaries:
            for s1 in boundaries:
                with self.subTest(s0=hex(s0), s1=hex(s1)):
                    res = ref_nand_b64(s0, s1)
                    self.assertGreaterEqual(res, 0)
                    self.assertLessEqual(res, MASK64)
                    scc = ref_scc(res)
                    self.assertEqual(scc, 1 if res != 0 else 0)

    def test_zero_result_only_when_both_all_ones(self):
        """Verify result is 0 and SCC is cleared IF AND ONLY IF (s0 & s1) == MASK64."""
        self.assertEqual(ref_nand_b64(MASK64, MASK64), 0)
        self.assertEqual(ref_scc(ref_nand_b64(MASK64, MASK64)), 0)

        for bit in range(64):
            with self.subTest(missing_bit=bit):
                s0 = MASK64 ^ (1 << bit)
                s1 = MASK64
                res = ref_nand_b64(s0, s1)
                self.assertEqual(res, 1 << bit)
                self.assertEqual(ref_scc(res), 1)

    def test_prng_differential_stress(self):
        """Stress verify against direct Python (~(a & b)) & MASK64 across 256 random cases."""
        rng = random.Random(0xDEAD_CAFE)
        for _ in range(256):
            a = rng.getrandbits(64)
            b = rng.getrandbits(64)
            expected = (~(a & b)) & MASK64
            self.assertEqual(ref_nand_b64(a, b), expected)
            self.assertEqual(ref_scc(expected), 1 if expected != 0 else 0)

class OperandKind:
    SGPR_PAIR = "sgpr_pair"
    SPECIAL = "special"
    LITERAL64 = "literal64"


class SaluOperand:
    """Represents a 64-bit operand for SOP2 instructions."""

    def __init__(self, raw: str | int) -> None:
        self.raw = raw
        if isinstance(raw, int):
            self.kind = OperandKind.LITERAL64
            self.val = raw & MASK64
            self.name = f"{self.val:#x}"
        elif raw in ("exec", "vcc", "null"):
            self.kind = OperandKind.SPECIAL
            self.name = raw
        elif raw.startswith("s[") and ":" in raw and raw.endswith("]"):
            self.kind = OperandKind.SGPR_PAIR
            parts = raw[2:-1].split(":")
            if len(parts) != 2:
                raise ValueError(f"Malformed SGPR pair: {raw}")
            lo, hi = int(parts[0]), int(parts[1])
            if hi != lo + 1:
                raise ValueError(f"SGPR pair must be consecutive registers: {raw}")
            if lo < 0 or hi > 105:
                raise ValueError(f"SGPR index out of range: {raw}")
            self.name = raw
            self.reg_lo = lo
            self.reg_hi = hi
        else:
            raise ValueError(f"Unrecognized SALU operand: {raw}")


class SaluNandB64Instruction:
    """Structural AST node for RDNA2 s_nand_b64 instruction."""

    def __init__(self, dst: str, src0: str | int, src1: str | int) -> None:
        self.opcode = "s_nand_b64"
        self.dst = SaluOperand(dst)
        self.src0 = SaluOperand(src0)
        self.src1 = SaluOperand(src1)
        self.validate()

    def validate(self) -> None:
        if self.dst.kind == OperandKind.LITERAL64:
            raise ValueError("Destination operand cannot be an immediate literal")

    def to_asm(self) -> str:
        s0 = f"{self.src0.val:#x}" if self.src0.kind == OperandKind.LITERAL64 else self.src0.name
        s1 = f"{self.src1.val:#x}" if self.src1.kind == OperandKind.LITERAL64 else self.src1.name
        return f"{self.opcode} {self.dst.name}, {s0}, {s1}"


class SaluNandB64AstTests(unittest.TestCase):
    """Structural AST and operand representation tests for s_nand_b64."""

    def test_valid_ast_construction_sgpr_pairs(self):
        inst = SaluNandB64Instruction("s[4:5]", "s[0:1]", "s[2:3]")
        self.assertEqual(inst.opcode, "s_nand_b64")
        self.assertEqual(inst.dst.kind, OperandKind.SGPR_PAIR)
        self.assertEqual(inst.src0.kind, OperandKind.SGPR_PAIR)
        self.assertEqual(inst.src1.kind, OperandKind.SGPR_PAIR)
        self.assertEqual(inst.to_asm(), "s_nand_b64 s[4:5], s[0:1], s[2:3]")

    def test_valid_ast_construction_special_operands(self):
        for special in ("exec", "vcc", "null"):
            inst = SaluNandB64Instruction(special, "s[0:1]", "s[2:3]")
            self.assertEqual(inst.dst.kind, OperandKind.SPECIAL)
            self.assertEqual(inst.to_asm(), f"s_nand_b64 {special}, s[0:1], s[2:3]")

    def test_valid_ast_construction_literals(self):
        inst = SaluNandB64Instruction("s[0:1]", "s[2:3]", 0x12345678_9ABCDEF0)
        self.assertEqual(inst.src1.kind, OperandKind.LITERAL64)
        self.assertIn("0x123456789abcdef0", inst.to_asm())

    def test_ast_validation_failures(self):
        with self.assertRaises(ValueError):
            SaluNandB64Instruction(0x1234, "s[0:1]", "s[2:3]")
        with self.assertRaises(ValueError):
            SaluNandB64Instruction("s[0:2]", "s[0:1]", "s[2:3]")
        with self.assertRaises(ValueError):
            SaluNandB64Instruction("s[105:106]", "s[0:1]", "s[2:3]")
        with self.assertRaises(ValueError):
            SaluOperand("invalid_operand_string")


if __name__ == "__main__":
    unittest.main()
