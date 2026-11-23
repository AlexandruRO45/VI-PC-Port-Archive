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

class ScalarState:
    """Models 64-bit RDNA2 scalar execution state: SGPRs, EXEC, VCC, SCC, and NULL."""

    def __init__(self, num_sgprs: int = 106) -> None:
        self.sgprs = [0] * num_sgprs
        self.exec: int = MASK64
        self.vcc: int = 0
        self.scc: int = 0

    def read_sgpr64(self, idx: int) -> int:
        lo = self.sgprs[idx] & MASK32
        hi = self.sgprs[idx + 1] & MASK32
        return (hi << 32) | lo

    def write_sgpr64(self, idx: int, val: int) -> None:
        self.sgprs[idx] = val & MASK32
        self.sgprs[idx + 1] = (val >> 32) & MASK32

    def resolve_src64(self, src: int | str) -> int:
        if isinstance(src, int):
            return src & MASK64
        if src == "exec":
            return self.exec & MASK64
        if src == "vcc":
            return self.vcc & MASK64
        if src == "null":
            return 0
        if src.startswith("s[") and ":" in src:
            idx = int(src[2:].split(":")[0])
            return self.read_sgpr64(idx)
        raise ValueError(f"Unknown 64-bit source operand: {src}")

    def write_dst64(self, dst: str, val: int) -> None:
        val = val & MASK64
        if dst == "exec":
            self.exec = val
        elif dst == "vcc":
            self.vcc = val
        elif dst == "null":
            pass
        elif dst.startswith("s[") and ":" in dst:
            idx = int(dst[2:].split(":")[0])
            self.write_sgpr64(idx, val)
        else:
            raise ValueError(f"Unknown 64-bit destination operand: {dst}")

    def execute_nand_b64(self, dst: str, src0: int | str, src1: int | str) -> int:
        """Executes one s_nand_b64 instruction and updates SCC."""
        s0_val = self.resolve_src64(src0)
        s1_val = self.resolve_src64(src1)
        res = ref_nand_b64(s0_val, s1_val)
        self.write_dst64(dst, res)
        self.scc = ref_scc(res)
        return res


class SaluNandB64InterpreterTests(unittest.TestCase):
    """Scalar interpreter execution tests for s_nand_b64."""

    def test_interpreter_basic_execution(self):
        state = ScalarState()
        v0 = 0x01234567_89ABCDEF
        v1 = 0xFEEDBEEF_CAFED00D
        state.write_sgpr64(0, v0)
        state.write_sgpr64(2, v1)

        res = state.execute_nand_b64("s[4:5]", "s[0:1]", "s[2:3]")
        expected = ref_nand_b64(v0, v1)
        self.assertEqual(res, expected)
        self.assertEqual(state.read_sgpr64(4), expected)
        self.assertEqual(state.scc, ref_scc(expected))

    def test_interpreter_inline_constants(self):
        state = ScalarState()
        v0 = 0x55555555_55555555
        state.write_sgpr64(0, v0)

        # NAND with 0 -> all ones (MASK64)
        res = state.execute_nand_b64("s[2:3]", "s[0:1]", 0)
        self.assertEqual(res, MASK64)
        self.assertEqual(state.read_sgpr64(2), MASK64)
        self.assertEqual(state.scc, 1)

        # NAND with MASK64 -> bitwise NOT of v0
        res = state.execute_nand_b64("s[2:3]", "s[0:1]", MASK64)
        self.assertEqual(res, (~v0) & MASK64)
        self.assertEqual(state.read_sgpr64(2), (~v0) & MASK64)
        self.assertEqual(state.scc, 1)

    def test_interpreter_null_destination(self):
        state = ScalarState()
        state.write_sgpr64(0, 0x11111111_22222222)
        state.write_sgpr64(2, 0x33333333_44444444)

        # Execute to null sink
        res = state.execute_nand_b64("null", "s[0:1]", "s[2:3]")
        self.assertEqual(res, ref_nand_b64(0x11111111_22222222, 0x33333333_44444444))
        self.assertEqual(state.scc, 1)
        # Verify source registers were untouched
        self.assertEqual(state.read_sgpr64(0), 0x11111111_22222222)
        self.assertEqual(state.read_sgpr64(2), 0x33333333_44444444)

    def test_interpreter_randomized_trace(self):
        state = ScalarState()
        rng = random.Random(0x1337_5A10)
        current = rng.getrandbits(64)
        state.write_sgpr64(0, current)

        for step in range(20):
            val = rng.getrandbits(64)
            state.write_sgpr64(2, val)
            res = state.execute_nand_b64("s[0:1]", "s[0:1]", "s[2:3]")
            expected = ref_nand_b64(current, val)
            self.assertEqual(res, expected)
            self.assertEqual(state.read_sgpr64(0), expected)
            self.assertEqual(state.scc, ref_scc(expected))
            current = expected

class SaluNandB64SimdDivergenceTests(unittest.TestCase):
    """SIMD multi-lane divergence and Wave32/Wave64 EXEC masking tests for s_nand_b64."""

    def test_wave64_divergent_branch_nand(self):
        state = ScalarState()
        state.exec = MASK64
        state.vcc = 0x55555555_55555555

        # s_nand_b64 exec, exec, vcc
        state.execute_nand_b64("exec", "exec", "vcc")
        # NAND(MASK64, 0x5555555555555555) = ~0x5555555555555555 = 0xAAAAAAAAAAAAAAAA
        self.assertEqual(state.exec, 0xAAAAAAAA_AAAAAAAA)
        self.assertEqual(state.scc, 1)

    def test_wave32_lower_half_divergence(self):
        state = ScalarState()
        state.exec = 0x00000000_FFFFFFFF
        state.vcc = 0x00000000_0000FFFF

        # s_nand_b64 exec, exec, vcc
        # ~(0x000000000000FFFF) & MASK64 = 0xFFFFFFFFFFFF0000
        state.execute_nand_b64("exec", "exec", "vcc")
        self.assertEqual(state.exec, 0xFFFFFFFF_FFFF0000)
        self.assertEqual(state.scc, 1)

    def test_checkerboard_non_overlapping_masks(self):
        state = ScalarState()
        even_lanes = 0x55555555_55555555
        odd_lanes = 0xAAAAAAAA_AAAAAAAA
        state.exec = even_lanes
        state.vcc = odd_lanes

        # NAND of disjoint lane sets produces all 1s (MASK64)
        state.execute_nand_b64("exec", "exec", "vcc")
        self.assertEqual(state.exec, MASK64)
        self.assertEqual(state.scc, 1)

    def test_single_lane_mask_nand(self):
        state = ScalarState()
        for lane in range(64):
            with self.subTest(lane=lane):
                mask = 1 << lane
                state.exec = mask
                state.vcc = mask
                state.execute_nand_b64("exec", "exec", "vcc")
                expected = (~mask) & MASK64
                self.assertEqual(state.exec, expected)
                self.assertEqual(state.scc, 1)

    def test_all_lanes_active_all_disabled(self):
        state = ScalarState()
        state.exec = MASK64
        state.vcc = MASK64
        # All lanes active & true -> NAND disables all lanes!
        state.execute_nand_b64("exec", "exec", "vcc")
        self.assertEqual(state.exec, 0)
        self.assertEqual(state.scc, 0)

        # All lanes disabled -> NAND enables all lanes!
        state.execute_nand_b64("exec", "exec", "vcc")
        self.assertEqual(state.exec, MASK64)
        self.assertEqual(state.scc, 1)

class SaluNandB64RegisterAliasingTests(unittest.TestCase):
    """Register aliasing and in-place operand preservation tests for s_nand_b64."""

    def test_alias_dst_equals_src0(self):
        state = ScalarState()
        a = 0x11223344_55667788
        b = 0x88776655_44332211
        state.write_sgpr64(0, a)
        state.write_sgpr64(2, b)

        state.execute_nand_b64("s[0:1]", "s[0:1]", "s[2:3]")
        self.assertEqual(state.read_sgpr64(0), ref_nand_b64(a, b))
        self.assertEqual(state.read_sgpr64(2), b)

    def test_alias_dst_equals_src1(self):
        state = ScalarState()
        a = 0x11223344_55667788
        b = 0x88776655_44332211
        state.write_sgpr64(0, a)
        state.write_sgpr64(2, b)

        state.execute_nand_b64("s[2:3]", "s[0:1]", "s[2:3]")
        self.assertEqual(state.read_sgpr64(2), ref_nand_b64(a, b))
        self.assertEqual(state.read_sgpr64(0), a)

    def test_alias_dst_equals_both_srcs(self):
        state = ScalarState()
        a = 0x0F0F0F0F_F0F0F0F0
        state.write_sgpr64(0, a)

        # In-place NOT
        state.execute_nand_b64("s[0:1]", "s[0:1]", "s[0:1]")
        self.assertEqual(state.read_sgpr64(0), (~a) & MASK64)
        self.assertEqual(state.scc, 1)

    def test_alias_overlapping_window_partial(self):
        state = ScalarState()
        state.sgprs[0] = 0x12345678
        state.sgprs[1] = 0x9ABCDEF0
        state.sgprs[2] = 0x0FEDCBA9
        state.sgprs[3] = 0x87654321

        s0_val = state.read_sgpr64(0)
        s1_val = state.read_sgpr64(2)
        expected = ref_nand_b64(s0_val, s1_val)

        # s_nand_b64 s[1:2], s[0:1], s[2:3]
        state.execute_nand_b64("s[1:2]", "s[0:1]", "s[2:3]")

        # s0 and s3 must be preserved
        self.assertEqual(state.sgprs[0], 0x12345678)
        self.assertEqual(state.read_sgpr64(1), expected)
        self.assertEqual(state.sgprs[3], 0x87654321)

    def test_alias_overlapping_shift_down(self):
        state = ScalarState()
        state.sgprs[0] = 0x00000000
        state.sgprs[1] = 0xAAAAAAAA
        state.sgprs[2] = 0x55555555
        state.sgprs[3] = 0x33333333
        state.sgprs[4] = 0xCCCCCCCC

        s0_val = state.read_sgpr64(1)
        s1_val = state.read_sgpr64(3)
        expected = ref_nand_b64(s0_val, s1_val)

        # s_nand_b64 s[0:1], s[1:2], s[3:4]
        state.execute_nand_b64("s[0:1]", "s[1:2]", "s[3:4]")
        self.assertEqual(state.read_sgpr64(0), expected)
        self.assertEqual(state.sgprs[2], 0x55555555)
        self.assertEqual(state.read_sgpr64(3), s1_val)

class SaluNandB64ConditionCodesTests(unittest.TestCase):
    """Condition codes (SCC status flag) generation and propagation tests for s_nand_b64."""

    def test_scc_exhaustive_lo_hi_zero_matrix(self):
        state = ScalarState()
        cases = [
            (MASK64, MASK64, 0),                       # result = 0x00000000_00000000 -> SCC = 0
            (MASK64, 0xFFFFFFFF_00000000, 1),           # result = 0x00000000_FFFFFFFF -> SCC = 1
            (MASK64, 0x00000000_FFFFFFFF, 1),           # result = 0xFFFFFFFF_00000000 -> SCC = 1
            (MASK64, 0, 1),                             # result = 0xFFFFFFFF_FFFFFFFF -> SCC = 1
            (0x12345678_00000000, 0xFFFFFFFF_00000000, 1),
            (0x00000000_12345678, 0x00000000_FFFFFFFF, 1),
        ]
        for s0, s1, expected_scc in cases:
            with self.subTest(s0=hex(s0), s1=hex(s1)):
                state.write_sgpr64(0, s0)
                state.write_sgpr64(2, s1)
                res = state.execute_nand_b64("s[4:5]", "s[0:1]", "s[2:3]")
                self.assertEqual(state.scc, expected_scc)
                self.assertEqual(state.scc, 1 if res != 0 else 0)

    def test_scc_cleared_exclusively_on_all_ones(self):
        state = ScalarState()
        # SCC is 0 ONLY if every single bit of (s0 & s1) is 1
        state.write_sgpr64(0, MASK64)
        state.write_sgpr64(2, MASK64)
        state.execute_nand_b64("s[4:5]", "s[0:1]", "s[2:3]")
        self.assertEqual(state.scc, 0)

        # Clear one bit at a time across all 64 bits -> SCC MUST be 1
        for bit in range(64):
            state.write_sgpr64(0, MASK64 ^ (1 << bit))
            state.write_sgpr64(2, MASK64)
            state.execute_nand_b64("s[4:5]", "s[0:1]", "s[2:3]")
            self.assertEqual(state.scc, 1)

    def test_scc_bitwise_independence_from_arithmetic_overflow(self):
        state = ScalarState()
        # 0x7FFFFFFF_FFFFFFFF & 0x80000000_00000000 == 0 -> NAND is MASK64
        state.write_sgpr64(0, 0x7FFFFFFF_FFFFFFFF)
        state.write_sgpr64(2, 0x80000000_00000000)
        res = state.execute_nand_b64("s[4:5]", "s[0:1]", "s[2:3]")
        self.assertEqual(res, MASK64)
        self.assertEqual(state.scc, 1)

class SaluNandB64Ieee754EdgeCasesTests(unittest.TestCase):
    """IEEE 754 floating-point edge cases and boundary bit patterns for s_nand_b64."""

    @staticmethod
    def float_to_u64(f: float) -> int:
        return struct.unpack("<Q", struct.pack("<d", f))[0]

    @staticmethod
    def u64_to_float(u: int) -> float:
        return struct.unpack("<d", struct.pack("<Q", u & MASK64))[0]

    def test_ieee754_double_special_values(self):
        specials = {
            "+0.0": 0x00000000_00000000,
            "-0.0": 0x80000000_00000000,
            "+1.0": 0x3FF00000_00000000,
            "-1.0": 0xBFF00000_00000000,
            "+inf": 0x7FF00000_00000000,
            "-inf": 0xFFF00000_00000000,
            "qNaN": 0x7FF80000_00000000,
            "sNaN": 0x7FF00000_00000001,
            "min_subnormal": 0x00000000_00000001,
            "max_subnormal": 0x000FFFFF_FFFFFFFF,
            "max_finite":    0x7FEFFFFF_FFFFFFFF,
        }
        for name1, u1 in specials.items():
            for name2, u2 in specials.items():
                with self.subTest(s0=name1, s1=name2):
                    res = ref_nand_b64(u1, u2)
                    expected = (~(u1 & u2)) & MASK64
                    self.assertEqual(res, expected)
                    self.assertEqual(ref_scc(res), 1 if expected != 0 else 0)

    def test_ieee754_sign_bit_inversion(self):
        neg_zero = 0x80000000_00000000
        # NAND with itself inverts the sign bit and sets all other bits
        res = ref_nand_b64(neg_zero, neg_zero)
        self.assertEqual(res, 0x7FFFFFFF_FFFFFFFF)
        self.assertEqual(ref_scc(res), 1)

    def test_ieee754_exponent_mask_preservation(self):
        exp_mask = 0x7FF00000_00000000
        one_d = 0x3FF00000_00000000
        res = ref_nand_b64(exp_mask, one_d)
        self.assertEqual(res, (~0x3FF00000_00000000) & MASK64)
        self.assertEqual(ref_scc(res), 1)

    def test_ieee754_no_flush_to_zero(self):
        subnormal = 0x00000000_0000000F
        all_ones = MASK64
        res = ref_nand_b64(subnormal, all_ones)
        self.assertEqual(res, (~subnormal) & MASK64)
        self.assertEqual(res & 0xF, 0x0)
        self.assertEqual((res >> 4) & 0xF, 0xF)

    def test_ieee754_packed_f32_pairs(self):
        f0 = struct.unpack("<I", struct.pack("<f", 0.0))[0]
        f1 = struct.unpack("<I", struct.pack("<f", -0.0))[0]
        packed = (f1 << 32) | f0
        res = ref_nand_b64(packed, packed)
        self.assertEqual(res, (~packed) & MASK64)
        self.assertEqual(ref_scc(res), 1)

class SaluNandB64TranslatorTests(unittest.TestCase):
    """End-to-end SPIR-V translation and mini-interpreter execution for s_nand_b64."""

    def test_translator_emitted_spirv_opcodes(self):
        lines = [
            "s_mov_b32 s0, 0x12345678",
            "s_mov_b32 s1, 0x9abcdef0",
            "s_mov_b32 s2, 0x0fedcba9",
            "s_mov_b32 s3, 0x87654321",
            "s_nand_b64 s[4:5], s[0:1], s[2:3]",
            "v_mov_b32 v0, s4",
            "v_mov_b32 v1, s5",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ]
        spv = translate_lines("ps_nand_b64_opcodes", lines)
        self.assertIn("OpBitwiseAnd", spv)
        self.assertIn("OpNot", spv)
        self.assertIn("OpBitwiseOr", spv)

    def test_translator_mrt0_execution_against_oracle(self):
        test_pairs = [
            (0x01234567_89ABCDEF, 0x00FF00FF_0F0F0F0F),
            (0x00000000_00000000, 0xFFFFFFFF_FFFFFFFF),
            (0x55555555_55555555, 0xAAAAAAAA_AAAAAAAA),
            (0xFFFFFFFF_FFFFFFFF, 0xFFFFFFFF_FFFFFFFF),
            (0x00000001_00000000, 0x00000001_00000000),
            (0x00000000_00000001, 0x00000000_00000001),
        ]
        for idx, (v0, v1) in enumerate(test_pairs):
            with self.subTest(idx=idx, v0=hex(v0), v1=hex(v1)):
                s0_lo = v0 & MASK32
                s0_hi = (v0 >> 32) & MASK32
                s1_lo = v1 & MASK32
                s1_hi = (v1 >> 32) & MASK32
                lines = [
                    f"s_mov_b32 s0, {s0_lo:#x}",
                    f"s_mov_b32 s1, {s0_hi:#x}",
                    f"s_mov_b32 s2, {s1_lo:#x}",
                    f"s_mov_b32 s3, {s1_hi:#x}",
                    "s_nand_b64 s[4:5], s[0:1], s[2:3]",
                    "v_mov_b32 v0, s4",
                    "v_mov_b32 v1, s5",
                    "s_cselect_b32 s6, 1, 0",
                    "v_mov_b32 v2, s6",
                    "v_mov_b32 v3, 0",
                    "exp mrt0 v0, v1, v2, v3 done vm",
                    "s_endpgm",
                ]
                spv = translate_lines(f"ps_nand_b64_exec_{idx}", lines)
                out = mrt0_words(spv)
                res64 = (out[1] << 32) | out[0]
                expected = ref_nand_b64(v0, v1)
                expected_scc = ref_scc(expected)
                self.assertEqual(res64, expected)
                self.assertEqual(out[2], expected_scc)

    def test_translator_destination_aliasing(self):
        v0 = 0x12345678_9ABCDEF0
        v1 = 0x0F0F0F0F_F0F0F0F0
        lines = [
            f"s_mov_b32 s0, {v0 & MASK32:#x}",
            f"s_mov_b32 s1, {(v0 >> 32) & MASK32:#x}",
            f"s_mov_b32 s2, {v1 & MASK32:#x}",
            f"s_mov_b32 s3, {(v1 >> 32) & MASK32:#x}",
            "s_nand_b64 s[0:1], s[0:1], s[2:3]",
            "v_mov_b32 v0, s0",
            "v_mov_b32 v1, s1",
            "s_cselect_b32 s6, 1, 0",
            "v_mov_b32 v2, s6",
            "v_mov_b32 v3, 0",
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ]
        spv = translate_lines("ps_nand_alias", lines)
        out = mrt0_words(spv)
        res64 = (out[1] << 32) | out[0]
        expected = ref_nand_b64(v0, v1)
        self.assertEqual(res64, expected)
        self.assertEqual(out[2], ref_scc(expected))

    def test_translator_null_destination(self):
        v0 = 0xFFFFFFFF_FFFFFFFF
        v1 = 0xFFFFFFFF_FFFFFFFF
        lines = [
            f"s_mov_b32 s0, {v0 & MASK32:#x}",
            f"s_mov_b32 s1, {(v0 >> 32) & MASK32:#x}",
            f"s_mov_b32 s2, {v1 & MASK32:#x}",
            f"s_mov_b32 s3, {(v1 >> 32) & MASK32:#x}",
            "s_nand_b64 null, s[0:1], s[2:3]",
            "s_cselect_b32 s6, 1, 0",
            "v_mov_b32 v0, s6",
            "v_mov_b32 v1, 0",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ]
        spv = translate_lines("ps_nand_null_dst", lines)
        out = mrt0_words(spv)
        self.assertEqual(out[0], 0)

    def test_translator_mask_lowering(self):
        lines = [
            "s_nand_b64 exec, exec, vcc",
            "v_mov_b32 v0, 0",
            "v_mov_b32 v1, 0",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ]
        spv = translate_lines("ps_nand_mask", lines)
        self.assertIn("OpBitwiseAnd", spv)
        self.assertIn("OpNot", spv)


if __name__ == "__main__":
    unittest.main()
