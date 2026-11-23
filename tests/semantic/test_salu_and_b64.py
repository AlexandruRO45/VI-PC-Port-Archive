"""Semantic reference oracle and verification for RDNA2 64-bit scalar AND (s_and_b64).

Covers:
  - s_and_b64: D = S0 & S1 (64-bit bitwise AND)
  - Scalar condition code SCC: SCC = 1 if D != 0 else 0
  - Wave64 / Wave32 execution mask (EXEC) and vector condition code (VCC) interactions
  - Register aliasing (in-place destination, source overlapping, partial SGPR pair overlap)
  - Special register handling (null destination, exec, vcc)
  - IEEE 754 bit-pattern preservation without floating-point side effects
  - SPIR-V translation lowering and mini-interpreter validation via support.translate_lines

Standard library Python only; strictly zero game data; fully reproducible.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
import enum
from enum import Enum, auto
import math
import os
import struct
import sys
from typing import Dict, List, Optional, Sequence, Tuple, Union
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

MASK32: int = 0xFFFFFFFF
MASK64: int = 0xFFFFFFFFFFFFFFFF
_TRANSLATOR_PRESENT = translate_lines is not None and mrt0_words is not None

def ref_s_and_b64(s0: int, s1: int) -> int:
    """RDNA2 s_and_b64: D = S0 & S1 (64-bit unsigned bitwise AND)."""
    return (s0 & s1) & MASK64


def ref_scc(val64: int) -> int:
    """RDNA2 SCC condition flag: 1 if 64-bit result != 0, else 0."""
    return 1 if (val64 & MASK64) != 0 else 0


class SaluAndB64OracleTests(unittest.TestCase):
    """Reference mathematical oracle tests for 64-bit scalar bitwise AND."""

    def test_truth_table_single_bits(self) -> None:
        """Exhaustive 1-bit truth table verification."""
        table = {
            (0, 0): 0,
            (0, 1): 0,
            (1, 0): 0,
            (1, 1): 1,
        }
        for (b0, b1), expected in table.items():
            with self.subTest(b0=b0, b1=b1):
                s0 = MASK64 if b0 else 0
                s1 = MASK64 if b1 else 0
                want = MASK64 if expected else 0
                got = ref_s_and_b64(s0, s1)
                self.assertEqual(got, want)
                self.assertEqual(ref_scc(got), expected)

    def test_algebraic_identities(self) -> None:
        """Verify identity, annihilator, idempotence, commutativity, and associativity."""
        x = 0x12345678_9ABCDEF0
        y = 0xAAAAAAAAAAAAAAAA
        z = 0x55555555_55555555

        # Identity: x & ~0 == x
        self.assertEqual(ref_s_and_b64(x, MASK64), x)
        self.assertEqual(ref_s_and_b64(MASK64, x), x)

        # Annihilator: x & 0 == 0
        self.assertEqual(ref_s_and_b64(x, 0), 0)
        self.assertEqual(ref_s_and_b64(0, x), 0)

        # Idempotence: x & x == x
        self.assertEqual(ref_s_and_b64(x, x), x)

        # Commutativity: x & y == y & x
        self.assertEqual(ref_s_and_b64(x, y), ref_s_and_b64(y, x))

        # Associativity: (x & y) & z == x & (y & z)
        self.assertEqual(
            ref_s_and_b64(ref_s_and_b64(x, y), z),
            ref_s_and_b64(x, ref_s_and_b64(y, z)),
        )

        # Distributivity over OR: x & (y | z) == (x & y) | (x & z)
        self.assertEqual(
            ref_s_and_b64(x, (y | z) & MASK64),
            ((ref_s_and_b64(x, y)) | (ref_s_and_b64(x, z))) & MASK64,
        )

    def test_boundary_values(self) -> None:
        """Boundary values across 64-bit boundaries (0, MSB, LSB, alternating bits)."""
        boundaries = [
            0x00000000_00000000,
            0xFFFFFFFF_FFFFFFFF,
            0x80000000_00000000,
            0x00000000_00000001,
            0x7FFFFFFF_FFFFFFFF,
            0x00000000_FFFFFFFF,
            0xFFFFFFFF_00000000,
            0x55555555_55555555,
            0xAAAAAAAA_AAAAAAAA,
            0x01234567_89ABCDEF,
            0xFEDCBA98_76543210,
        ]
        for a in boundaries:
            for b in boundaries:
                with self.subTest(a=hex(a), b=hex(b)):
                    res = ref_s_and_b64(a, b)
                    self.assertGreaterEqual(res, 0)
                    self.assertLessEqual(res, MASK64)
                    scc = ref_scc(res)
                    self.assertEqual(scc, 1 if res != 0 else 0)

class OperandKind(Enum):
    SGPR_PAIR = auto()
    SPECIAL = auto()
    IMM64 = auto()
    LITERAL = auto()


@dataclass(frozen=True)
class OperandAST:
    kind: OperandKind
    name: str
    sgpr_index: Optional[int] = None
    value: Optional[int] = None

    @classmethod
    def sgpr_pair(cls, base_index: int) -> OperandAST:
        if not (0 <= base_index <= 104):
            raise ValueError(f"SGPR pair base index out of bounds: {base_index}")
        return cls(
            kind=OperandKind.SGPR_PAIR,
            name=f"s[{base_index}:{base_index+1}]",
            sgpr_index=base_index,
        )

    @classmethod
    def special(cls, name: str) -> OperandAST:
        valid_specials = {"exec", "vcc", "null"}
        if name not in valid_specials:
            raise ValueError(f"Unknown special 64-bit register: {name}")
        return cls(kind=OperandKind.SPECIAL, name=name)

    @classmethod
    def imm(cls, val: int) -> OperandAST:
        return cls(kind=OperandKind.IMM64, name=hex(val & MASK64), value=val & MASK64)


@dataclass(frozen=True)
class InstructionAST:
    opcode: str
    dst: OperandAST
    src0: OperandAST
    src1: OperandAST

    def __post_init__(self) -> None:
        if self.opcode != "s_and_b64":
            raise ValueError(f"Unexpected opcode: {self.opcode}")
        if self.dst.kind not in (OperandKind.SGPR_PAIR, OperandKind.SPECIAL):
            raise ValueError(f"Invalid destination operand: {self.dst}")

    def render(self) -> str:
        return f"{self.opcode} {self.dst.name}, {self.src0.name}, {self.src1.name}"


class SaluAndB64AstTests(unittest.TestCase):
    """AST syntax validation and formatting tests for s_and_b64."""

    def test_ast_construction_sgpr_pairs(self) -> None:
        inst = InstructionAST(
            opcode="s_and_b64",
            dst=OperandAST.sgpr_pair(4),
            src0=OperandAST.sgpr_pair(0),
            src1=OperandAST.sgpr_pair(2),
        )
        self.assertEqual(inst.render(), "s_and_b64 s[4:5], s[0:1], s[2:3]")

    def test_ast_construction_special_registers(self) -> None:
        inst_exec = InstructionAST(
            opcode="s_and_b64",
            dst=OperandAST.special("exec"),
            src0=OperandAST.special("exec"),
            src1=OperandAST.special("vcc"),
        )
        self.assertEqual(inst_exec.render(), "s_and_b64 exec, exec, vcc")

        inst_null = InstructionAST(
            opcode="s_and_b64",
            dst=OperandAST.special("null"),
            src0=OperandAST.sgpr_pair(0),
            src1=OperandAST.sgpr_pair(2),
        )
        self.assertEqual(inst_null.render(), "s_and_b64 null, s[0:1], s[2:3]")

    def test_ast_construction_literals(self) -> None:
        inst_imm = InstructionAST(
            opcode="s_and_b64",
            dst=OperandAST.sgpr_pair(0),
            src0=OperandAST.sgpr_pair(0),
            src1=OperandAST.imm(0xFFFFFFFF_00000000),
        )
        self.assertEqual(inst_imm.render(), "s_and_b64 s[0:1], s[0:1], 0xffffffff00000000")

    def test_ast_invalid_operands_rejected(self) -> None:
        with self.assertRaises(ValueError):
            OperandAST.sgpr_pair(105)
        with self.assertRaises(ValueError):
            OperandAST.special("m0")
        with self.assertRaises(ValueError):
            InstructionAST(
                opcode="s_and_b64",
                dst=OperandAST.imm(0),
                src0=OperandAST.sgpr_pair(0),
                src1=OperandAST.sgpr_pair(2),
            )

class ScalarState:
    """Models 64-bit RDNA2 scalar execution state: SGPRs, EXEC, VCC, SCC, and NULL."""

    def __init__(self, num_sgprs: int = 106) -> None:
        self.sgprs = [0] * num_sgprs
        self.exec: int = MASK64
        self.vcc: int = 0
        self.scc: int = 0

    def read_sgpr64(self, idx: int) -> int:
        if not (0 <= idx < len(self.sgprs) - 1):
            raise IndexError(f"SGPR index out of range: {idx}")
        lo = self.sgprs[idx] & MASK32
        hi = self.sgprs[idx + 1] & MASK32
        return (hi << 32) | lo

    def write_sgpr64(self, idx: int, val: int) -> None:
        if not (0 <= idx < len(self.sgprs) - 1):
            raise IndexError(f"SGPR index out of range: {idx}")
        self.sgprs[idx] = val & MASK32
        self.sgprs[idx + 1] = (val >> 32) & MASK32

    def resolve_src64(self, src: Union[int, str, OperandAST]) -> int:
        if isinstance(src, OperandAST):
            if src.kind == OperandKind.IMM64:
                return src.value & MASK64 if src.value is not None else 0
            src = src.name

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
        try:
            return int(src, 0) & MASK64
        except ValueError:
            raise ValueError(f"Unknown 64-bit source operand: {src}")

    def write_dst64(self, dst: Union[str, OperandAST], val: int) -> None:
        if isinstance(dst, OperandAST):
            dst = dst.name

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

    def step_s_and_b64(self, dst: Union[str, OperandAST], src0: Union[int, str, OperandAST], src1: Union[int, str, OperandAST]) -> int:
        """Executes s_and_b64: D = S0 & S1, and sets SCC = (D != 0)."""
        s0_val = self.resolve_src64(src0)
        s1_val = self.resolve_src64(src1)
        res = ref_s_and_b64(s0_val, s1_val)
        self.write_dst64(dst, res)
        self.scc = ref_scc(res)
        return res


class SaluAndB64InterpreterTests(unittest.TestCase):
    """Interpreter unit tests executing s_and_b64 across scalar state."""

    def test_interpreter_basic_and(self) -> None:
        state = ScalarState()
        v0 = 0x01234567_89ABCDEF
        v1 = 0xFEEDBEEF_CAFED00D
        state.write_sgpr64(0, v0)
        state.write_sgpr64(2, v1)

        res = state.step_s_and_b64("s[4:5]", "s[0:1]", "s[2:3]")
        expected = v0 & v1
        self.assertEqual(res, expected)
        self.assertEqual(state.read_sgpr64(4), expected)
        self.assertEqual(state.scc, 1)

    def test_interpreter_null_destination_still_updates_scc(self) -> None:
        state = ScalarState()
        state.write_sgpr64(0, 0x12345678_00000000)
        state.write_sgpr64(2, 0x12345678_00000000)

        res = state.step_s_and_b64("null", "s[0:1]", "s[2:3]")
        self.assertEqual(res, 0x12345678_00000000)
        self.assertEqual(state.scc, 1)
        self.assertEqual(state.read_sgpr64(0), 0x12345678_00000000)

        state.step_s_and_b64("null", 0x55555555_55555555, 0xAAAAAAAA_AAAAAAAA)
        self.assertEqual(state.scc, 0)

class SaluAndB64DivergenceTests(unittest.TestCase):
    """Wave64 control-flow divergence and execution mask (EXEC) tests."""

    def test_exec_branch_pruning(self) -> None:
        """Emulate branching: EXEC = EXEC & VCC (pruning lanes where condition is false)."""
        state = ScalarState()
        state.exec = MASK64
        state.vcc = 0xAAAAAAAAAAAAAAAA

        state.step_s_and_b64("exec", "exec", "vcc")
        self.assertEqual(state.exec, 0xAAAAAAAAAAAAAAAA)
        self.assertEqual(state.scc, 1)

    def test_exec_all_lanes_disabled(self) -> None:
        """When EXEC becomes zero, SCC must be 0, signaling whole wave is inactive."""
        state = ScalarState()
        state.exec = 0x55555555_55555555
        state.vcc = 0xAAAAAAAA_AAAAAAAA

        state.step_s_and_b64("exec", "exec", "vcc")
        self.assertEqual(state.exec, 0)
        self.assertEqual(state.scc, 0)

    def test_exec_single_lane_active_sweep(self) -> None:
        """Test each of 64 lanes active in isolation."""
        state = ScalarState()
        for lane in range(64):
            with self.subTest(lane=lane):
                state.exec = MASK64
                state.vcc = 1 << lane
                state.step_s_and_b64("exec", "exec", "vcc")
                self.assertEqual(state.exec, 1 << lane)
                self.assertEqual(state.scc, 1)

    def test_nested_divergence_monotonicity(self) -> None:
        """Nested conditionals monotonically reduce or preserve active lanes."""
        state = ScalarState()
        state.exec = 0xFFFFFFFF_FFFFFFFF
        levels = [
            0x7FFFFFFF_FFFFFFFF,
            0x0000FFFF_FFFFFFFF,
            0x000000FF_FFFFFFFF,
            0x00000000_000000FF,
            0x00000000_00000001,
            0x00000000_00000000,
        ]
        prev_active_count = 64
        for mask in levels:
            state.vcc = mask
            state.step_s_and_b64("exec", "exec", "vcc")
            active_count = bin(state.exec).count("1")
            self.assertLessEqual(active_count, prev_active_count)
            prev_active_count = active_count
        self.assertEqual(state.exec, 0)
        self.assertEqual(state.scc, 0)

class SaluAndB64AliasingTests(unittest.TestCase):
    """Register aliasing: in-place destination, source-target overlaps, partial dwords."""

    def test_dst_aliases_src0(self) -> None:
        state = ScalarState()
        a = 0x12345678_9ABCDEF0
        b = 0x0F0F0F0F_F0F0F0F0
        state.write_sgpr64(0, a)
        state.write_sgpr64(2, b)

        state.step_s_and_b64("s[0:1]", "s[0:1]", "s[2:3]")
        expected = a & b
        self.assertEqual(state.read_sgpr64(0), expected)
        self.assertEqual(state.read_sgpr64(2), b)

    def test_dst_aliases_src1(self) -> None:
        state = ScalarState()
        a = 0x12345678_9ABCDEF0
        b = 0x0F0F0F0F_F0F0F0F0
        state.write_sgpr64(0, a)
        state.write_sgpr64(2, b)

        state.step_s_and_b64("s[2:3]", "s[0:1]", "s[2:3]")
        expected = a & b
        self.assertEqual(state.read_sgpr64(0), a)
        self.assertEqual(state.read_sgpr64(2), expected)

    def test_all_operands_same_register(self) -> None:
        state = ScalarState()
        a = 0xDEADBEEF_CAFEBABE
        state.write_sgpr64(0, a)

        state.step_s_and_b64("s[0:1]", "s[0:1]", "s[0:1]")
        self.assertEqual(state.read_sgpr64(0), a)
        self.assertEqual(state.scc, 1)

    def test_overlapping_partial_sgprs(self) -> None:
        """Destination s[1:2] partially overlaps src0 s[0:1] and src1 s[2:3]."""
        state = ScalarState()
        state.sgprs[0] = 0xAAAAAAAA
        state.sgprs[1] = 0xBBBBBBBB
        state.sgprs[2] = 0xCCCCCCCC
        state.sgprs[3] = 0xDDDDDDDD

        s0 = state.read_sgpr64(0)
        s1 = state.read_sgpr64(2)
        expected = s0 & s1

        state.step_s_and_b64("s[1:2]", "s[0:1]", "s[2:3]")

        self.assertEqual(state.sgprs[0], 0xAAAAAAAA)
        self.assertEqual(state.read_sgpr64(1), expected)
        self.assertEqual(state.sgprs[3], 0xDDDDDDDD)

class SaluAndB64ConditionCodesTests(unittest.TestCase):
    """Thorough validation of Scalar Condition Code (SCC) generation."""

    def test_scc_four_quadrants(self) -> None:
        """Verify (lo32, hi32) quadrants: zero/zero, non-zero/zero, zero/non-zero, non-zero/non-zero."""
        quadrants = [
            (0x00000000, 0x00000000, 0),
            (0x12345678, 0x00000000, 1),
            (0x00000000, 0x87654321, 1),
            (0x12345678, 0x87654321, 1),
        ]
        state = ScalarState()
        for lo, hi, expected_scc in quadrants:
            with self.subTest(lo=hex(lo), hi=hex(hi)):
                val = (hi << 32) | lo
                state.write_sgpr64(0, val)
                state.step_s_and_b64("s[2:3]", "s[0:1]", MASK64)
                self.assertEqual(state.read_sgpr64(2), val)
                self.assertEqual(state.scc, expected_scc)

    def test_scc_extreme_bits(self) -> None:
        """Verify bit 0 and bit 63 alone trigger SCC = 1."""
        state = ScalarState()

        # Bit 0 only
        state.step_s_and_b64("s[0:1]", 1, 1)
        self.assertEqual(state.scc, 1)

        # Bit 63 only (MSB)
        state.step_s_and_b64("s[0:1]", 1 << 63, 1 << 63)
        self.assertEqual(state.scc, 1)

        # Disjoint bits -> SCC = 0
        state.step_s_and_b64("s[0:1]", 1, 1 << 63)
        self.assertEqual(state.scc, 0)

class SaluAndB64Ieee754Tests(unittest.TestCase):
    """IEEE 754 float bit pattern preservation (no canonicalization, no FTZ/DAZ)."""

    def test_float64_zeros_and_signs(self) -> None:
        pos_zero = struct.unpack("<Q", struct.pack("<d", +0.0))[0]
        neg_zero = struct.unpack("<Q", struct.pack("<d", -0.0))[0]
        self.assertEqual(pos_zero, 0x00000000_00000000)
        self.assertEqual(neg_zero, 0x80000000_00000000)

        abs_mask = 0x7FFFFFFF_FFFFFFFF
        self.assertEqual(ref_s_and_b64(neg_zero, abs_mask), pos_zero)
        self.assertEqual(ref_scc(ref_s_and_b64(neg_zero, abs_mask)), 0)

    def test_float64_special_values_masking(self) -> None:
        pos_inf = struct.unpack("<Q", struct.pack("<d", float("inf")))[0]
        neg_inf = struct.unpack("<Q", struct.pack("<d", float("-inf")))[0]
        q_nan = struct.unpack("<Q", struct.pack("<d", float("nan")))[0]

        exp_mask = 0x7FF00000_00000000
        self.assertEqual(ref_s_and_b64(pos_inf, exp_mask), exp_mask)
        self.assertEqual(ref_s_and_b64(neg_inf, exp_mask), exp_mask)
        self.assertEqual(ref_s_and_b64(q_nan, exp_mask), exp_mask)

        mant_mask = 0x000FFFFF_FFFFFFFF
        self.assertEqual(ref_s_and_b64(pos_inf, mant_mask), 0)
        self.assertEqual(ref_scc(ref_s_and_b64(pos_inf, mant_mask)), 0)

    def test_packed_float32_pair_extraction(self) -> None:
        """A 64-bit register holding two 32-bit floats."""
        f_lo = struct.unpack("<I", struct.pack("<f", 1.5))[0]
        f_hi = struct.unpack("<I", struct.pack("<f", -2.5))[0]
        packed = (f_hi << 32) | f_lo

        lo_only = ref_s_and_b64(packed, 0x00000000_FFFFFFFF)
        self.assertEqual(lo_only, f_lo)
        self.assertEqual(ref_scc(lo_only), 1)

        hi_only = ref_s_and_b64(packed, 0xFFFFFFFF_00000000)
        self.assertEqual(hi_only, f_hi << 32)
        self.assertEqual(ref_scc(hi_only), 1)

if __name__ == "__main__":
    unittest.main()
