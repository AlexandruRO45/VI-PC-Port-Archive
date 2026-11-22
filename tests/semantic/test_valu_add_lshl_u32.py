from __future__ import annotations

"""Semantic interpreter tests and reference mathematical oracle for RDNA2 v_add_lshl_u32.

Tests the vector fused add and logical shift left instruction:
    v_add_lshl_u32 D, S0, S1, S2
Lowered semantics:
    sum32 = (S0 + S1) & 0xFFFFFFFF
    shift = S2 & 0x1F
    dst   = (sum32 << shift) & 0xFFFFFFFF
Mathematical and architectural properties:
    - Addition: 32-bit modular addition with wraparound mod 2^32.
    - Shift: Logical shift left by (S2 & 31); high bits shifted past bit 31 are discarded,
      and vacant low bits are zero-filled.
    - Commutativity: S0 and S1 are interchangeable: v_add_lshl_u32(a, b, c) == v_add_lshl_u32(b, a, c).
    - Shift operand masking: Bits [31:5] of S2 are ignored by hardware and translator.
Architectural model:
    32-bit vector registers (VGPRs) and scalar registers (SGPRs). Overwrites all 32 bits of destination VGPR.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
import math
import struct
import unittest
from typing import List, Optional, Sequence, Tuple, Union

# Architectural bitwise masks and constants
MASK32: int = 0xFFFF_FFFF
SHIFT_MASK: int = 0x1F
SIGN32: int = 0x8000_0000


def u32(val: int) -> int:
    """Normalize integer to unsigned 32-bit dword."""
    return val & MASK32


def s32(val: int) -> int:
    """Interpret unsigned 32-bit dword as signed two's complement integer."""
    val &= MASK32
    return val - 0x1_0000_0000 if (val & SIGN32) else val


def ref_add_lshl_u32(s0: int, s1: int, s2: int) -> int:
    """Reference mathematical computation for v_add_lshl_u32: ((s0 + s1) mod 2^32) << (s2 & 31)."""
    sum32 = (s0 + s1) & MASK32
    shift = s2 & SHIFT_MASK
    return (sum32 << shift) & MASK32


def oracle_v_add_lshl_u32(src0: int, src1: int, src2: int) -> int:
    """Reference architectural oracle for RDNA2 v_add_lshl_u32 instruction."""
    return ref_add_lshl_u32(src0, src1, src2)


class TestAddLshlU32ModelAndOracle(unittest.TestCase):
    """Architectural model and mathematical reference oracle tests for v_add_lshl_u32."""

    def test_basic_add_lshl(self) -> None:
        # (10 + 20) << 2 = 30 << 2 = 120
        res = oracle_v_add_lshl_u32(10, 20, 2)
        self.assertEqual(res, 120)

    def test_shift_zero_identity(self) -> None:
        # Shift count 0 produces plain modular addition
        self.assertEqual(oracle_v_add_lshl_u32(100, 250, 0), 350)
        self.assertEqual(oracle_v_add_lshl_u32(0xFFFFFFFF, 1, 0), 0)

    def test_shift_masking_upper_bits(self) -> None:
        # Shift amount is strictly masked by 0x1F (only lower 5 bits count)
        base = oracle_v_add_lshl_u32(5, 5, 3)
        self.assertEqual(base, 10 << 3)
        # S2 with upper bits set: 32 + 3 = 35 -> shift 3
        self.assertEqual(oracle_v_add_lshl_u32(5, 5, 35), base)
        # S2 = 0x100 -> 256 & 31 = 0 -> shift 0
        self.assertEqual(oracle_v_add_lshl_u32(5, 5, 256), 10)
        # S2 = 0xFFFFFFFF -> 0xFFFFFFFF & 31 = 31 -> shift 31
        self.assertEqual(oracle_v_add_lshl_u32(1, 0, 0xFFFFFFFF), (1 << 31) & MASK32)

    def test_addition_modulo_overflow(self) -> None:
        # Addition wraps mod 2^32 before the shift
        s0 = 0x8000_0000
        s1 = 0x8000_0000
        # (0x80000000 + 0x80000000) mod 2^32 = 0
        self.assertEqual(oracle_v_add_lshl_u32(s0, s1, 5), 0)

        # 0xFFFFFFFF + 2 = 1 (mod 2^32) -> 1 << 4 = 16
        self.assertEqual(oracle_v_add_lshl_u32(0xFFFF_FFFF, 2, 4), 16)

    def test_shift_truncation_overflow(self) -> None:
        # Bits shifted past 31 are discarded
        # 0x00010000 << 16 = 0 (1 << 32 is truncated)
        self.assertEqual(oracle_v_add_lshl_u32(0x0001_0000, 0, 16), 0)
        # 0x40000000 << 1 = 0x80000000, << 2 = 0
        self.assertEqual(oracle_v_add_lshl_u32(0x4000_0000, 0, 1), 0x8000_0000)
        self.assertEqual(oracle_v_add_lshl_u32(0x4000_0000, 0, 2), 0)

    def test_commutativity_src0_src1(self) -> None:
        # S0 and S1 addition is strictly commutative
        pairs = [
            (1234, 5678, 4),
            (0xFFFF_0000, 0x0000_FFFF, 7),
            (0xDEAD_BEEF, 0xCAFE_BABE, 12),
        ]
        for a, b, c in pairs:
            self.assertEqual(oracle_v_add_lshl_u32(a, b, c), oracle_v_add_lshl_u32(b, a, c))

    def test_identity_zero_operands(self) -> None:
        # s0 = 0, s1 = 0 -> 0
        self.assertEqual(oracle_v_add_lshl_u32(0, 0, 15), 0)
        # s1 = 0, s2 = 0 -> identity pass-through of s0
        val = 0x1234_5678
        self.assertEqual(oracle_v_add_lshl_u32(val, 0, 0), val)
        self.assertEqual(oracle_v_add_lshl_u32(0, val, 0), val)



# ==============================================================================
# Symbolic AST Opcode Definitions and Typed Expressions
# ==============================================================================

class RegType(Enum):
    VGPR = auto()
    SGPR = auto()


@dataclass(frozen=True)
class Reg:
    """Represents an architectural 32-bit register (VGPR or SGPR)."""
    index: int
    reg_type: RegType

    def __post_init__(self) -> None:
        if not (0 <= self.index <= 255):
            raise ValueError(f"Register index {self.index} out of range [0, 255]")

    @property
    def name(self) -> str:
        prefix = "v" if self.reg_type == RegType.VGPR else "s"
        return f"{prefix}{self.index}"


@dataclass(frozen=True)
class Literal32:
    """Represents a 32-bit integer literal or inline constant."""
    value: int

    def __post_init__(self) -> None:
        if not (-0x8000_0000 <= self.value <= 0xFFFF_FFFF):
            raise ValueError(f"Literal32 value {self.value} out of 32-bit range")

    @property
    def name(self) -> str:
        return f"0x{self.value & MASK32:08X}" if self.value > 9 or self.value < 0 else str(self.value)


@dataclass(frozen=True)
class VOP3Operand:
    """Typed expression operand representing a register or literal source."""
    source: Union[Reg, Literal32]

    def disassemble(self) -> str:
        return self.source.name


class Opcode(Enum):
    V_ADD_LSHL_U32 = "v_add_lshl_u32"


@dataclass(frozen=True)
class VAddLshlU32Instruction:
    """Symbolic AST node representing an RDNA2 v_add_lshl_u32 VALU instruction."""
    dst: Reg
    src0: VOP3Operand
    src1: VOP3Operand
    src2: VOP3Operand
    opcode: Opcode = Opcode.V_ADD_LSHL_U32

    def __post_init__(self) -> None:
        if self.dst.reg_type != RegType.VGPR:
            raise TypeError(f"Destination of VALU instruction must be VGPR, got {self.dst.reg_type}")

    def disassemble(self) -> str:
        return f"{self.opcode.value} {self.dst.name}, {self.src0.disassemble()}, {self.src1.disassemble()}, {self.src2.disassemble()}"


class TestSymbolicAST(unittest.TestCase):
    """Unit tests for symbolic AST opcode definitions and typed expressions."""

    def test_register_creation_and_disassembly(self) -> None:
        v0 = Reg(0, RegType.VGPR)
        s5 = Reg(5, RegType.SGPR)
        self.assertEqual(v0.name, "v0")
        self.assertEqual(s5.name, "s5")

    def test_register_bounds_validation(self) -> None:
        with self.assertRaises(ValueError):
            Reg(-1, RegType.VGPR)
        with self.assertRaises(ValueError):
            Reg(256, RegType.VGPR)

    def test_literal_operand(self) -> None:
        lit = Literal32(0x1234)
        self.assertEqual(lit.name, "0x00001234")
        lit_small = Literal32(4)
        self.assertEqual(lit_small.name, "4")

    def test_instruction_disassembly(self) -> None:
        dst = Reg(0, RegType.VGPR)
        src0 = VOP3Operand(Reg(1, RegType.VGPR))
        src1 = VOP3Operand(Reg(2, RegType.VGPR))
        src2 = VOP3Operand(Literal32(3))

        inst = VAddLshlU32Instruction(dst=dst, src0=src0, src1=src1, src2=src2)
        self.assertEqual(inst.disassemble(), "v_add_lshl_u32 v0, v1, v2, 3")

    def test_instruction_validation(self) -> None:
        dst_sgpr = Reg(0, RegType.SGPR)
        src = VOP3Operand(Reg(1, RegType.VGPR))
        with self.assertRaises(TypeError):
            VAddLshlU32Instruction(dst=dst_sgpr, src0=src, src1=src, src2=src)



# ==============================================================================
# Deterministic Scalar Instruction Interpreter
# ==============================================================================

@dataclass
class ScalarMachineState:
    """Architectural register file for a single wavefront lane."""
    vgprs: List[int] = field(default_factory=lambda: [0] * 256)
    sgprs: List[int] = field(default_factory=lambda: [0] * 256)

    def get_vgpr(self, index: int) -> int:
        if not (0 <= index <= 255):
            raise IndexError(f"VGPR index {index} out of range [0, 255]")
        return self.vgprs[index] & MASK32

    def set_vgpr(self, index: int, val: int) -> None:
        if not (0 <= index <= 255):
            raise IndexError(f"VGPR index {index} out of range [0, 255]")
        self.vgprs[index] = val & MASK32

    def get_sgpr(self, index: int) -> int:
        if not (0 <= index <= 255):
            raise IndexError(f"SGPR index {index} out of range [0, 255]")
        return self.sgprs[index] & MASK32

    def set_sgpr(self, index: int, val: int) -> None:
        if not (0 <= index <= 255):
            raise IndexError(f"SGPR index {index} out of range [0, 255]")
        self.sgprs[index] = val & MASK32


class ScalarInterpreter:
    """Deterministic scalar instruction interpreter for v_add_lshl_u32."""

    def resolve_operand(self, op: VOP3Operand, state: ScalarMachineState) -> int:
        if isinstance(op.source, Reg):
            if op.source.reg_type == RegType.VGPR:
                return state.get_vgpr(op.source.index)
            elif op.source.reg_type == RegType.SGPR:
                return state.get_sgpr(op.source.index)
            else:
                raise ValueError(f"Unknown register type: {op.source.reg_type}")
        elif isinstance(op.source, Literal32):
            return op.source.value & MASK32
        else:
            raise TypeError(f"Unsupported operand source type: {type(op.source)}")

    def step(self, instr: VAddLshlU32Instruction, state: ScalarMachineState) -> int:
        # Atomic read phase
        s0_raw = self.resolve_operand(instr.src0, state)
        s1_raw = self.resolve_operand(instr.src1, state)
        s2_raw = self.resolve_operand(instr.src2, state)

        # Computation phase
        result = oracle_v_add_lshl_u32(s0_raw, s1_raw, s2_raw)

        # Write-back phase
        state.set_vgpr(instr.dst.index, result)
        return result


class TestScalarInterpreter(unittest.TestCase):
    """Unit tests for the deterministic scalar instruction interpreter."""

    def setUp(self) -> None:
        self.interp = ScalarInterpreter()
        self.state = ScalarMachineState()

    def test_basic_step_vgprs(self) -> None:
        self.state.set_vgpr(1, 15)
        self.state.set_vgpr(2, 25)
        self.state.set_vgpr(3, 3)
        instr = VAddLshlU32Instruction(
            dst=Reg(0, RegType.VGPR),
            src0=VOP3Operand(Reg(1, RegType.VGPR)),
            src1=VOP3Operand(Reg(2, RegType.VGPR)),
            src2=VOP3Operand(Reg(3, RegType.VGPR)),
        )
        res = self.interp.step(instr, self.state)
        # (15 + 25) << 3 = 40 << 3 = 320
        self.assertEqual(res, 320)
        self.assertEqual(self.state.get_vgpr(0), 320)

    def test_step_with_sgpr_and_literal(self) -> None:
        self.state.set_sgpr(4, 50)
        self.state.set_vgpr(5, 50)
        instr = VAddLshlU32Instruction(
            dst=Reg(6, RegType.VGPR),
            src0=VOP3Operand(Reg(4, RegType.SGPR)),
            src1=VOP3Operand(Reg(5, RegType.VGPR)),
            src2=VOP3Operand(Literal32(2)),
        )
        res = self.interp.step(instr, self.state)
        # (50 + 50) << 2 = 100 << 2 = 400
        self.assertEqual(res, 400)
        self.assertEqual(self.state.get_vgpr(6), 400)

    def test_step_with_overflow(self) -> None:
        self.state.set_vgpr(1, 0xFFFF_FFFF)
        self.state.set_vgpr(2, 1)
        self.state.set_vgpr(3, 5)
        instr = VAddLshlU32Instruction(
            dst=Reg(0, RegType.VGPR),
            src0=VOP3Operand(Reg(1, RegType.VGPR)),
            src1=VOP3Operand(Reg(2, RegType.VGPR)),
            src2=VOP3Operand(Reg(3, RegType.VGPR)),
        )
        res = self.interp.step(instr, self.state)
        # (0xFFFFFFFF + 1) = 0 -> 0 << 5 = 0
        self.assertEqual(res, 0)
        self.assertEqual(self.state.get_vgpr(0), 0)

    def test_deterministic_repeatability(self) -> None:
        self.state.set_vgpr(10, 0x1234_5678)
        self.state.set_vgpr(11, 0x0000_0008)
        self.state.set_vgpr(12, 4)
        instr = VAddLshlU32Instruction(
            dst=Reg(13, RegType.VGPR),
            src0=VOP3Operand(Reg(10, RegType.VGPR)),
            src1=VOP3Operand(Reg(11, RegType.VGPR)),
            src2=VOP3Operand(Reg(12, RegType.VGPR)),
        )
        res0 = self.interp.step(instr, self.state)
        for _ in range(50):
            res_n = self.interp.step(instr, self.state)
            self.assertEqual(res0, res_n)


if __name__ == "__main__":
    unittest.main()


