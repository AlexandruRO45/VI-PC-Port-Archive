from __future__ import annotations

"""Semantic reference oracle and architectural model for AMD RDNA2 v_dot2_f32_f16.

AMD RDNA VALU Instruction Set Architecture:
  v_dot2_f32_f16 dst, src0, src1, src2 [clamp]

Format:
  VOP3P / VOP3 (3-operand vector ALU instruction with single-precision output).

Operands:
  dst:  32-bit vector register (v[dst]) storing an IEEE 754 single-precision float (binary32).
  src0: 32-bit register (v[s0] or s[s0]) storing a packed pair of IEEE 754 binary16 half-precision floats:
        - src0.f16[0]: low 16 bits [15:0]
        - src0.f16[1]: high 16 bits [31:16]
  src1: 32-bit register (v[s1] or s[s1]) storing a packed pair of IEEE 754 binary16 half-precision floats:
        - src1.f16[0]: low 16 bits [15:0]
        - src1.f16[1]: high 16 bits [31:16]
  src2: 32-bit register (v[s2], s[s2], or literal) storing an IEEE 754 single-precision float accumulator (binary32).

Operation:
  Converts half-precision float pairs into the single-precision domain and performs
  a 2-element dot product accumulated onto a 32-bit float value:
    prod0   = float32(src0.f16[0]) * float32(src1.f16[0])
    prod1   = float32(src0.f16[1]) * float32(src1.f16[1])
    res.f32 = prod0 + prod1 + src2.f32

Clamp Modifier:
  When clamp is set, the final single-precision result is clamped to [0.0, 1.0]:
    if isnan(res) or res < 0.0: res = +0.0
    elif res > 1.0:             res = 1.0
"""

from dataclasses import dataclass, field
from enum import Enum, auto
import math
import struct
from typing import List, Optional, Tuple, Union
import unittest

MASK16: int = 0xFFFF
MASK32: int = 0xFFFF_FFFF


def f16_to_f32(bits: int) -> float:
    """Convert an IEEE 754 binary16 uint16 bitfield to Python float (float32 compatible)."""
    return struct.unpack("<e", struct.pack("<H", bits & MASK16))[0]


def f16_to_bits(val: float) -> int:
    """Convert a Python float to an IEEE 754 binary16 uint16 bitfield."""
    return struct.unpack("<H", struct.pack("<e", val))[0]


def bits_to_f32(bits: int) -> float:
    """Convert an IEEE 754 binary32 uint32 bitfield to Python float."""
    return struct.unpack("<f", struct.pack("<I", bits & MASK32))[0]


def f32_to_bits(val: float) -> int:
    """Convert a Python float to an IEEE 754 binary32 uint32 bitfield."""
    return struct.unpack("<I", struct.pack("<f", val))[0]


def pack_f16x2(lo_bits: int, hi_bits: int) -> int:
    """Pack two 16-bit binary16 integers into a 32-bit packed dword."""
    return (lo_bits & MASK16) | ((hi_bits & MASK16) << 16)


def unpack_f16x2(packed32: int) -> Tuple[int, int]:
    """Unpack a 32-bit dword into two 16-bit binary16 integers (lo, hi)."""
    return packed32 & MASK16, (packed32 >> 16) & MASK16


def ref_dot2_f32_f16(
    s0_packed: int,
    s1_packed: int,
    s2_f32_bits: int,
    *,
    clamp: bool = False,
) -> float:
    """Reference mathematical oracle for v_dot2_f32_f16 returning Python float."""
    s0_lo, s0_hi = unpack_f16x2(s0_packed)
    s1_lo, s1_hi = unpack_f16x2(s1_packed)

    a0 = f16_to_f32(s0_lo)
    a1 = f16_to_f32(s0_hi)
    b0 = f16_to_f32(s1_lo)
    b1 = f16_to_f32(s1_hi)
    c = bits_to_f32(s2_f32_bits)

    res = (a0 * b0) + (a1 * b1) + c

    if clamp:
        if math.isnan(res) or res < 0.0:
            return 0.0
        if res > 1.0:
            return 1.0
    return res


def ref_dot2_f32_f16_bits(
    s0_packed: int,
    s1_packed: int,
    s2_f32_bits: int,
    *,
    clamp: bool = False,
) -> int:
    """Reference mathematical oracle for v_dot2_f32_f16 returning binary32 integer bits."""
    return f32_to_bits(ref_dot2_f32_f16(s0_packed, s1_packed, s2_f32_bits, clamp=clamp))


class TestValuDot2F32F16ReferenceOracle(unittest.TestCase):
    """Architectural reference tests validating mathematical model for v_dot2_f32_f16."""

    def test_basic_dot2_with_accumulator(self) -> None:
        """(1.0, 2.0) . (3.0, 4.0) + 5.0 = 1*3 + 2*4 + 5 = 16.0"""
        s0 = pack_f16x2(f16_to_bits(1.0), f16_to_bits(2.0))
        s1 = pack_f16x2(f16_to_bits(3.0), f16_to_bits(4.0))
        s2 = f32_to_bits(5.0)
        res = ref_dot2_f32_f16(s0, s1, s2)
        self.assertAlmostEqual(res, 16.0, places=5)
        self.assertEqual(ref_dot2_f32_f16_bits(s0, s1, s2), f32_to_bits(16.0))

    def test_zero_sources_accumulator_pass_through(self) -> None:
        """(0.0, 0.0) . (1.0, 1.0) + 42.5 = 42.5"""
        s0 = pack_f16x2(f16_to_bits(0.0), f16_to_bits(0.0))
        s1 = pack_f16x2(f16_to_bits(1.0), f16_to_bits(1.0))
        s2 = f32_to_bits(42.5)
        res = ref_dot2_f32_f16(s0, s1, s2)
        self.assertAlmostEqual(res, 42.5, places=5)

    def test_negative_elements_and_cancellation(self) -> None:
        """(-1.5, 2.0) . (2.0, -3.0) + 10.0 = -3.0 + (-6.0) + 10.0 = 1.0"""
        s0 = pack_f16x2(f16_to_bits(-1.5), f16_to_bits(2.0))
        s1 = pack_f16x2(f16_to_bits(2.0), f16_to_bits(-3.0))
        s2 = f32_to_bits(10.0)
        res = ref_dot2_f32_f16(s0, s1, s2)
        self.assertAlmostEqual(res, 1.0, places=5)

    def test_orthogonal_vectors(self) -> None:
        """(1.0, 0.0) . (0.0, 1.0) + 7.0 = 0 + 7.0 = 7.0"""
        s0 = pack_f16x2(f16_to_bits(1.0), f16_to_bits(0.0))
        s1 = pack_f16x2(f16_to_bits(0.0), f16_to_bits(1.0))
        s2 = f32_to_bits(7.0)
        res = ref_dot2_f32_f16(s0, s1, s2)
        self.assertAlmostEqual(res, 7.0, places=5)

    def test_clamp_upper_saturation(self) -> None:
        """Values greater than 1.0 saturate to 1.0 with clamp=True."""
        s0 = pack_f16x2(f16_to_bits(1.0), f16_to_bits(1.0))
        s1 = pack_f16x2(f16_to_bits(1.0), f16_to_bits(1.0))
        s2 = f32_to_bits(0.0)
        # 1*1 + 1*1 = 2.0 -> clamped to 1.0
        res = ref_dot2_f32_f16(s0, s1, s2, clamp=True)
        self.assertEqual(res, 1.0)
        self.assertEqual(ref_dot2_f32_f16_bits(s0, s1, s2, clamp=True), f32_to_bits(1.0))

    def test_clamp_lower_saturation(self) -> None:
        """Negative values saturate to 0.0 with clamp=True."""
        s0 = pack_f16x2(f16_to_bits(-2.0), f16_to_bits(0.0))
        s1 = pack_f16x2(f16_to_bits(1.0), f16_to_bits(0.0))
        s2 = f32_to_bits(0.0)
        # -2*1 = -2.0 -> clamped to 0.0
        res = ref_dot2_f32_f16(s0, s1, s2, clamp=True)
        self.assertEqual(res, 0.0)
        self.assertEqual(ref_dot2_f32_f16_bits(s0, s1, s2, clamp=True), f32_to_bits(0.0))

    def test_clamp_in_range_preservation(self) -> None:
        """Values strictly inside [0.0, 1.0] are preserved with clamp=True."""
        s0 = pack_f16x2(f16_to_bits(0.5), f16_to_bits(0.0))
        s1 = pack_f16x2(f16_to_bits(1.0), f16_to_bits(0.0))
        s2 = f32_to_bits(0.25)
        # 0.5 * 1.0 + 0.25 = 0.75
        res = ref_dot2_f32_f16(s0, s1, s2, clamp=True)
        self.assertAlmostEqual(res, 0.75, places=5)


# ==============================================================================
# Symbolic AST Opcode Definitions and Typed Expressions
# ==============================================================================

class RegType(Enum):
    VGPR = auto()
    SGPR = auto()
    LITERAL = auto()


@dataclass(frozen=True)
class Operand:
    """Represents a register operand or inline literal constant."""
    reg_type: RegType
    index: int = 0
    count: int = 1
    literal_value: int | None = None

    def __post_init__(self) -> None:
        if self.count <= 0:
            raise ValueError(f"Operand count must be >= 1, got {self.count}")
        if self.reg_type == RegType.VGPR:
            if not (0 <= self.index <= 255):
                raise ValueError(f"VGPR index {self.index} out of range [0, 255]")
            if self.index + self.count > 256:
                raise ValueError(f"VGPR range exceeds 255")
        elif self.reg_type == RegType.SGPR:
            if not (0 <= self.index <= 105):
                raise ValueError(f"SGPR index {self.index} out of range [0, 105]")
            if self.index + self.count > 106:
                raise ValueError(f"SGPR range exceeds 105")
        elif self.reg_type == RegType.LITERAL:
            if self.literal_value is None:
                raise ValueError("Literal operand requires literal_value")
            if not (0 <= self.literal_value <= MASK32):
                raise ValueError(f"Literal value {self.literal_value:#x} out of range [0, 0xFFFFFFFF]")

    @classmethod
    def vgpr(cls, index: int, count: int = 1) -> Operand:
        return cls(RegType.VGPR, index=index, count=count)

    @classmethod
    def sgpr(cls, index: int, count: int = 1) -> Operand:
        return cls(RegType.SGPR, index=index, count=count)

    @classmethod
    def literal(cls, value: int, count: int = 1) -> Operand:
        return cls(RegType.LITERAL, count=count, literal_value=value & MASK32)

    def disassemble(self) -> str:
        if self.reg_type == RegType.VGPR:
            if self.count == 1:
                return f"v{self.index}"
            return f"v[{self.index}:{self.index + self.count - 1}]"
        if self.reg_type == RegType.SGPR:
            if self.count == 1:
                return f"s{self.index}"
            return f"s[{self.index}:{self.index + self.count - 1}]"
        assert self.literal_value is not None
        return f"0x{self.literal_value:08X}"


@dataclass(frozen=True)
class VDot2F32F16Instruction:
    """Symbolic AST node for AMD RDNA2 v_dot2_f32_f16 instruction."""
    dst: Operand
    src0: Operand
    src1: Operand
    src2: Operand
    clamp: bool = False

    def __post_init__(self) -> None:
        if self.dst.reg_type != RegType.VGPR:
            raise TypeError(f"Destination must be a VGPR, got {self.dst.reg_type}")
        if self.dst.count != 1:
            raise ValueError(f"Destination must be a 32-bit VGPR (count=1), got {self.dst.count}")
        if self.src0.count != 1:
            raise ValueError(f"src0 must be a 32-bit operand (count=1), got {self.src0.count}")
        if self.src1.count != 1:
            raise ValueError(f"src1 must be a 32-bit operand (count=1), got {self.src1.count}")
        if self.src2.count != 1:
            raise ValueError(f"src2 must be a 32-bit operand (count=1), got {self.src2.count}")

    def disassemble(self) -> str:
        base = (
            f"v_dot2_f32_f16 {self.dst.disassemble()}, "
            f"{self.src0.disassemble()}, {self.src1.disassemble()}, "
            f"{self.src2.disassemble()}"
        )
        return f"{base} clamp" if self.clamp else base


class TestValuDot2F32F16SymbolicAST(unittest.TestCase):
    """Unit tests for symbolic AST opcode definitions and typed expressions."""

    def test_ast_instruction_creation(self) -> None:
        instr = VDot2F32F16Instruction(
            dst=Operand.vgpr(0),
            src0=Operand.vgpr(1),
            src1=Operand.sgpr(2),
            src2=Operand.vgpr(3),
        )
        self.assertEqual(instr.dst.index, 0)
        self.assertEqual(instr.src0.index, 1)
        self.assertEqual(instr.src1.index, 2)
        self.assertEqual(instr.src2.index, 3)
        self.assertEqual(
            instr.disassemble(),
            "v_dot2_f32_f16 v0, v1, s2, v3",
        )

    def test_ast_disassembly_with_clamp_and_literal(self) -> None:
        instr = VDot2F32F16Instruction(
            dst=Operand.vgpr(5),
            src0=Operand.vgpr(6),
            src1=Operand.literal(0x3C003C00),
            src2=Operand.vgpr(7),
            clamp=True,
        )
        self.assertEqual(
            instr.disassemble(),
            "v_dot2_f32_f16 v5, v6, 0x3C003C00, v7 clamp",
        )

    def test_ast_immutability(self) -> None:
        instr = VDot2F32F16Instruction(
            dst=Operand.vgpr(0),
            src0=Operand.vgpr(1),
            src1=Operand.vgpr(2),
            src2=Operand.vgpr(3),
        )
        with self.assertRaises(Exception):
            instr.dst = Operand.vgpr(4)  # type: ignore[misc]

    def test_ast_operand_bounds(self) -> None:
        with self.assertRaises(ValueError):
            Operand.vgpr(256)
        with self.assertRaises(ValueError):
            Operand.sgpr(106)
        with self.assertRaises(ValueError):
            Operand.vgpr(-1)


# ==============================================================================
# Deterministic Scalar Instruction Interpreter
# ==============================================================================

@dataclass
class ScalarRegisterState:
    """Architectural register state for a single scalar thread."""
    vgprs: List[int] = field(default_factory=lambda: [0] * 256)
    sgprs: List[int] = field(default_factory=lambda: [0] * 106)

    def read_operand_32(self, op: Operand) -> int:
        if op.reg_type == RegType.VGPR:
            return self.vgprs[op.index] & MASK32
        if op.reg_type == RegType.SGPR:
            return self.sgprs[op.index] & MASK32
        assert op.literal_value is not None
        return op.literal_value & MASK32

    def write_vgpr_32(self, index: int, val: int) -> None:
        self.vgprs[index] = val & MASK32


class ScalarInterpreter:
    """Deterministic scalar interpreter executing v_dot2_f32_f16 instructions."""

    def step(self, instr: VDot2F32F16Instruction, state: ScalarRegisterState) -> int:
        s0 = state.read_operand_32(instr.src0)
        s1 = state.read_operand_32(instr.src1)
        s2 = state.read_operand_32(instr.src2)
        res = ref_dot2_f32_f16_bits(s0, s1, s2, clamp=instr.clamp)
        state.write_vgpr_32(instr.dst.index, res)
        return res


class TestValuDot2F32F16ScalarInterpreter(unittest.TestCase):
    """Unit tests for deterministic scalar instruction execution."""

    def setUp(self) -> None:
        self.state = ScalarRegisterState()
        self.interp = ScalarInterpreter()

    def test_scalar_execution_from_vgprs(self) -> None:
        # s0 in v1: (2.0, 3.0)
        s0_val = pack_f16x2(f16_to_bits(2.0), f16_to_bits(3.0))
        self.state.vgprs[1] = s0_val

        # s1 in v2: (4.0, 5.0)
        s1_val = pack_f16x2(f16_to_bits(4.0), f16_to_bits(5.0))
        self.state.vgprs[2] = s1_val

        # s2 in v3: 6.0f
        s2_val = f32_to_bits(6.0)
        self.state.vgprs[3] = s2_val

        instr = VDot2F32F16Instruction(
            dst=Operand.vgpr(0),
            src0=Operand.vgpr(1),
            src1=Operand.vgpr(2),
            src2=Operand.vgpr(3),
        )

        res_bits = self.interp.step(instr, self.state)
        # Expected: 2*4 + 3*5 + 6 = 8 + 15 + 6 = 29.0
        expected_bits = f32_to_bits(29.0)
        self.assertEqual(res_bits, expected_bits)
        self.assertEqual(self.state.vgprs[0], expected_bits)

    def test_scalar_execution_with_sgpr_and_literal(self) -> None:
        # s0 in v0: (1.5, -2.0)
        s0_val = pack_f16x2(f16_to_bits(1.5), f16_to_bits(-2.0))
        self.state.vgprs[0] = s0_val

        # s1 in s4: (2.0, 3.0)
        s1_val = pack_f16x2(f16_to_bits(2.0), f16_to_bits(3.0))
        self.state.sgprs[4] = s1_val

        # s2 literal: 10.0f
        s2_lit = f32_to_bits(10.0)

        instr = VDot2F32F16Instruction(
            dst=Operand.vgpr(1),
            src0=Operand.vgpr(0),
            src1=Operand.sgpr(4),
            src2=Operand.literal(s2_lit),
        )

        res_bits = self.interp.step(instr, self.state)
        # 1.5*2.0 + (-2.0)*3.0 + 10.0 = 3.0 - 6.0 + 10.0 = 7.0
        self.assertEqual(res_bits, f32_to_bits(7.0))

    def test_multi_step_chained_accumulation(self) -> None:
        """Step 1 produces float in v0; Step 2 feeds v0 as accumulator."""
        s0_1 = pack_f16x2(f16_to_bits(1.0), f16_to_bits(1.0))
        s1_1 = pack_f16x2(f16_to_bits(2.0), f16_to_bits(2.0))
        self.state.vgprs[1] = s0_1
        self.state.vgprs[2] = s1_1
        self.state.vgprs[3] = f32_to_bits(0.0)

        # v0 = (1,1) . (2,2) + 0 = 4.0
        instr1 = VDot2F32F16Instruction(
            dst=Operand.vgpr(0),
            src0=Operand.vgpr(1),
            src1=Operand.vgpr(2),
            src2=Operand.vgpr(3),
        )
        self.interp.step(instr1, self.state)
        self.assertEqual(self.state.vgprs[0], f32_to_bits(4.0))

        # v0 = (3,3) . (1,1) + v0 = 6.0 + 4.0 = 10.0
        s0_2 = pack_f16x2(f16_to_bits(3.0), f16_to_bits(3.0))
        s1_2 = pack_f16x2(f16_to_bits(1.0), f16_to_bits(1.0))
        self.state.vgprs[4] = s0_2
        self.state.vgprs[5] = s1_2

        instr2 = VDot2F32F16Instruction(
            dst=Operand.vgpr(0),
            src0=Operand.vgpr(4),
            src1=Operand.vgpr(5),
            src2=Operand.vgpr(0),
        )
        self.interp.step(instr2, self.state)
        self.assertEqual(self.state.vgprs[0], f32_to_bits(10.0))

    def test_scalar_determinism(self) -> None:
        state_a = ScalarRegisterState()
        state_b = ScalarRegisterState()
        val_s0 = pack_f16x2(f16_to_bits(1.234), f16_to_bits(-0.567))
        val_s1 = pack_f16x2(f16_to_bits(4.321), f16_to_bits(2.468))
        state_a.vgprs[1] = val_s0
        state_b.vgprs[1] = val_s0
        state_a.sgprs[0] = val_s1
        state_b.sgprs[0] = val_s1

        instr = VDot2F32F16Instruction(
            dst=Operand.vgpr(0),
            src0=Operand.vgpr(1),
            src1=Operand.sgpr(0),
            src2=Operand.literal(f32_to_bits(1.0)),
        )
        res_a = self.interp.step(instr, state_a)
        res_b = self.interp.step(instr, state_b)
        self.assertEqual(res_a, res_b)
        self.assertEqual(state_a.vgprs, state_b.vgprs)


if __name__ == "__main__":
    unittest.main()
